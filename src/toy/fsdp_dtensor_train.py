import torch, os, time
import torch.nn as nn
from dataclasses import dataclass
import torch.distributed as dist
from torch.distributed._composable.fsdp import fully_shard
from tqdm import tqdm

from src.profile.vram import measure_vram
from src.profile.perf_monitor import PerformanceMonitor
from src.utils.text_handle import append_to_txt_file
from torch.distributed._composable.checkpoint_activation import checkpoint

DENSE_LAYER_IDX = {14, 18, 22, 23}
FREEZE_LAYERS = {0, 1, 2, 3, 4, 5, 7, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 23}
NUM_STEPS = 100
WARM_UP_STEPS = 10
DATA_LOADER_LATENCY = 0.05
AC_INTERVAL = 0

BATCH_SIZE = 4
SEQ_LEN = 4096

@dataclass
class ModelArgs:
    dim: int = 512
    dense_dim: int = 1024
    ffn_dim: int = 1536
    ffn_dense_dim: int = 3172
    n_layers: int = 24
    n_heads: int = 8
    vocab_size: int = 10000

class HybridTransformer(nn.Module):
    def __init__(self, model_args: ModelArgs):
        super().__init__()
        self.model_args = model_args
        self.tok_embeddings = nn.Embedding(model_args.vocab_size, model_args.dim)
        self.layers = torch.nn.ModuleDict()
        self.intermediate = torch.nn.ModuleDict()
        self.intermediate_pre = torch.nn.ModuleDict()
        
        for layer_id in range(model_args.n_layers):
            if layer_id in DENSE_LAYER_IDX:
                self.intermediate_pre[str(layer_id)] = nn.Linear(in_features=model_args.dim, out_features=model_args.dense_dim)
                self.layers[str(layer_id)] = nn.TransformerDecoderLayer(model_args.dense_dim, nhead=model_args.n_heads, dim_feedforward=model_args.ffn_dense_dim)
                self.intermediate[str(layer_id)] = nn.Linear(in_features=model_args.dense_dim, out_features=model_args.dim)
            else:
                self.layers[str(layer_id)] = nn.TransformerDecoderLayer(model_args.dim, nhead=model_args.n_heads, dim_feedforward=model_args.ffn_dim)

        self.norm = nn.LayerNorm(model_args.dim)
        self.output = nn.Linear(model_args.dim, model_args.vocab_size)

    def forward(self, tokens: torch.Tensor):
        h = self.tok_embeddings(tokens)
        active_layers = sorted([int(k) for k in self.layers.keys()])
        for i in active_layers:
            layer = self.layers[str(i)]
            if i in DENSE_LAYER_IDX:
                h = self.intermediate_pre[str(i)](h)
                h = layer(h, h)
                h = self.intermediate[str(i)](h)
            else:
                h = layer(h, h)
        h = self.norm(h)
        output = self.output(h)
        return output

global rank, local_rank, device
def init_distributed():
   global rank, local_rank, device
   
   rank = int(os.environ["RANK"]) 
   local_rank = int(os.environ["LOCAL_RANK"])
   world_size = int(os.environ["WORLD_SIZE"])

   torch.cuda.set_device(local_rank)
   
   device = torch.device(f"cuda:{local_rank}")
   
   dist.init_process_group(backend="nccl")

if __name__ == "__main__":
    init_distributed()
    model_args = ModelArgs()
    
    model = HybridTransformer(model_args)
    model.to(device)

    for layer_idx in FREEZE_LAYERS:
        key = str(layer_idx)
        if key in model.layers:
            decoder_layer = model.layers[key]
            for param in decoder_layer.linear1.parameters(): param.requires_grad = False
            for param in decoder_layer.linear2.parameters(): param.requires_grad = False

    for layer_id in model.layers.keys():
        layer = model.layers[layer_id]
        if AC_INTERVAL > 0 and int(layer_id) % AC_INTERVAL == 0:
            checkpoint(layer)
        fully_shard(layer)
 
    fully_shard(model)
    
    
    RES_PATH = f"logs/toy_fsdp_dtensor/fsdp_ac{AC_INTERVAL}_sl{SEQ_LEN}_mbs{BATCH_SIZE}_report.txt"

    def tokenwise_loss_fn(outputs, targets):
        loss_fn = nn.CrossEntropyLoss()
        outputs = outputs.reshape(-1, model_args.vocab_size)
        targets = targets.reshape(-1)
        return loss_fn(outputs, targets)

    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = torch.optim.AdamW(trainable_params, lr=1e-3)

    
    
    TOTAL_TOKENS_PER_STEP = BATCH_SIZE * SEQ_LEN # Per GPU throughput

    perf_monitor = PerformanceMonitor()

    for i in tqdm(range(NUM_STEPS), disable=rank!=0):
        step_start_time = time.perf_counter()
        
        time.sleep(DATA_LOADER_LATENCY)
        x_cpu = torch.randint(0, model_args.vocab_size, (BATCH_SIZE, SEQ_LEN), dtype=torch.long)
        y_cpu = torch.randint(0, model_args.vocab_size, (BATCH_SIZE, SEQ_LEN), dtype=torch.long)
        x = x_cpu.to(device, non_blocking=True)
        y = y_cpu.to(device, non_blocking=True)
        optimizer.zero_grad()
        
        with measure_vram("memory", device=rank) as vram_stats:
            output = model(x)
            loss = tokenwise_loss_fn(output, y)
            loss.backward()
            optimizer.step()
        step_time = time.perf_counter() - step_start_time
        peak_vram_mb = vram_stats.metrics.get(f"memory/peak_allocated_mb", 0.0)
        tokens_per_sec = TOTAL_TOKENS_PER_STEP / step_time
        
        if i >= WARM_UP_STEPS:
            perf_monitor.record_step(
                latency=step_time, 
                throughput=tokens_per_sec, 
                vram=peak_vram_mb
            )

    # --- report ---
    report = f"\n=== Report for Rank {rank} ===\n"
    report += perf_monitor.generate_report_string()
    
    append_to_txt_file(RES_PATH, report)
    
    if rank == 0:
        print(f"Training finished. Reports saved to {RES_PATH}")
        
    dist.destroy_process_group()