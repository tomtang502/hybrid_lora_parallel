import torch, os, time
import torch.nn as nn
from dataclasses import dataclass
import torch.distributed as dist
from torch.distributed.pipelining import PipelineStage, ScheduleGPipe
from tqdm import tqdm
from src.profile.vram import measure_vram
from src.profile.perf_monitor import PerformanceMonitor
from src.utils.text_handle import append_to_txt_file

DENSE_LAYER_IDX = {14, 18, 22, 23}
FREEZE_LAYERS = {0, 1, 2, 3, 4, 5, 7, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 23} 
NUM_STEPS = 100
WARM_UP_STEPS = 10
RES_PATH = "logs/pp_training_report.txt"
DATA_LOADER_LATENCY = 0.05

# experimental hyper parameters
BATCH_SIZE = 8
BATCH_ONCE = 4
SEQ_LEN = 4608
SPLIT_LAYER = 16

@dataclass
class ModelArgs:
    dim: int = 512
    dense_dim: int = 1024
    ffn_dim: int = 1536
    ffn_dense_dim: int = 3172
    n_layers: int = 24
    n_heads: int = 8
    vocab_size: int = 10000

class Transformer(nn.Module):
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
        h = self.tok_embeddings(tokens) if self.tok_embeddings else tokens
        active_layers = sorted([int(k) for k in self.layers.keys()])
        for i in active_layers:
            layer = self.layers[str(i)]
            if i in DENSE_LAYER_IDX:
                h = self.intermediate_pre[str(i)](h)
                h = layer(h, h)
                h = self.intermediate[str(i)](h)
            else:
                h = layer(h, h)
        h = self.norm(h) if self.norm else h
        output = self.output(h).clone() if self.output else h
        return output

global rank, device, pp_group, stage_index, num_stages
def init_distributed():
   global rank, device, pp_group, stage_index, num_stages
   rank = int(os.environ["LOCAL_RANK"])
   world_size = int(os.environ["WORLD_SIZE"])
   device = torch.device(f"cuda:{rank}") if torch.cuda.is_available() else torch.device("cpu")
   dist.init_process_group()
   pp_group = dist.new_group()
   stage_index = rank
   num_stages = world_size

def manual_model_split(model) -> PipelineStage:
    if stage_index == 0:
        for i in range(SPLIT_LAYER, 24): del model.layers[str(i)]
        model.norm = None; model.output = None
    elif stage_index == 1:
        for i in range(SPLIT_LAYER): del model.layers[str(i)]
        model.tok_embeddings = None
    return PipelineStage(model, stage_index, num_stages, device)

if __name__ == "__main__":
    init_distributed()
    num_microbatches = BATCH_SIZE//BATCH_ONCE
    model_args = ModelArgs()
    model = Transformer(model_args)
    stage = manual_model_split(model)

    for layer_idx in FREEZE_LAYERS:
        key = str(layer_idx)
        if key in model.layers:
            decoder_layer = model.layers[key]
            for param in decoder_layer.linear1.parameters(): param.requires_grad = False
            for param in decoder_layer.linear2.parameters(): param.requires_grad = False

    model.to(device)
    
    
    
    RES_PATH = f"logs/pp_ablation/pp_split{SPLIT_LAYER}_sl{SEQ_LEN}_mbs{BATCH_ONCE}_report.txt"
    
    def tokenwise_loss_fn(outputs, targets):
        loss_fn = nn.CrossEntropyLoss()
        outputs = outputs.reshape(-1, model_args.vocab_size)
        targets = targets.reshape(-1)
        return loss_fn(outputs, targets)

    trainable_params = filter(lambda p: p.requires_grad, stage.submod.parameters())
    optimizer = torch.optim.AdamW(trainable_params, lr=1e-3)
    schedule = ScheduleGPipe(stage, n_microbatches=num_microbatches, loss_fn=tokenwise_loss_fn)
    
    TOTAL_TOKENS_PER_STEP = BATCH_SIZE * SEQ_LEN

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
            if rank == 0:
                schedule.step(x)
            elif rank == 1:
                losses = []
                output = schedule.step(target=y, losses=losses)
            
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