import os, yaml, json, time, wandb, hashlib
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Union, Tuple, Literal, List
from transformers import (
    TrainingArguments,

)
from src.config.lora_config import CustomLoraConfig
from src.config.optim_config import LoRAOptimizerConfig
from src.config.constants import MODEL_PATH, DATA_PATH, DATA_DIR_PATH
from src.config.parallel_config.fsdp_config import FSDP2Config
from src.utils.dist import get_dist_rank

@dataclass
class HLParTrainConfig:
    """Training Configuration for Hybrid Parallelism"""
    model_name_or_path: str = MODEL_PATH
    tokenizer_name: str = MODEL_PATH
    VERSION_STRING: str = "1203"
    trust_remote_code: bool = True
    
    # Path to save logs
    log_root_dir: str = "logs"
    run_name: str | None = None
    

    seed: int = 42
    tokenizer_pad_side: str = "right"
    

    # Logging
    trackers: Tuple[str, ...] = ("jsonl", "wandb")
    logging_file_name: str = "train.log"
    result_file_name: str = "perf.txt"
    logging_first_step: bool = True
    logging_steps: int = 1
    reset_log_dir: bool = True

    # profiling
    warm_up_steps: int = 10

    # Training Size
    use_lora = True
    per_device_batch_size: int = 2
    global_batch_size: int = 32
    gradient_accumulation_steps: int = 2
    num_devices: int = 8
    num_steps: int = 100

    # VRAM
    parallel_stretagy: Literal["ddp", "fsdp", "fsdp_dtensor"] = "fsdp_dtensor"
    gradient_checkpointing: bool = True
    gradient_checkpointing_kwargs: dict | None = None
    bf16: bool = True
    low_cpu_mem_usage: bool = False
    mha_only: bool = True
    attn_implementation: str = "sdpa"
    
    # dataset
    dataloader_num_workers: int = 4
    remove_unused_columns: bool = False,            # keep exactly what collate returns
    dataset_path: str = DATA_DIR_PATH
    chunk_size: int = 2048

    # DDP
    ddp_find_unused_parameters = False

    # FSDP
    fsdp_cpu_offload = False
    
    def __post_init__(self):
        assert self.warm_up_steps < self.num_steps
        self.single_iter_batch_size = self.per_device_batch_size * self.num_devices
        self.total_size = self.global_batch_size * self.num_steps
        # adjust grad_accumulation for global batch size
        assert self.global_batch_size % self.per_device_batch_size == 0, "Global batch size has to be a multiple of per_device_batch_size"
        assert self.global_batch_size % self.single_iter_batch_size == 0, "Global batch size has to be a multiple of per_device_batch_size * num_devices!"
        self.gradient_accumulation_steps = self.global_batch_size // self.single_iter_batch_size
        
        # additional configurations
        self.lora_cfg = None
        if self.use_lora:
            self.lora_cfg: CustomLoraConfig = CustomLoraConfig(mha_only=self.mha_only)
        self.opt_cfg = LoRAOptimizerConfig(mha_only=self.mha_only)

        # set up the run_name for this run
        module_opened = 'mha'
        if not self.mha_only:
            module_opened = module_opened+'+gdn'
        module_opened = module_opened+'+embed'+'norms'
        
        if self.run_name is None:
            if self.use_lora:
                self.run_name = f"{self.parallel_stretagy}_{module_opened}_lora_clength{self.chunk_size}_{self.num_devices}_b{self.global_batch_size}_s{self.seed}"
            else:
                self.run_name = f"{self.parallel_stretagy}_{module_opened}_clength{self.chunk_size}_{self.num_devices}_b{self.global_batch_size}_s{self.seed}"

        self.log_dir = Path(f"{self.log_root_dir}/{self.run_name}")
        
        self.log_dir.mkdir(parents=True,
                           exist_ok=True)
        self.logging_file_name = f"train_r{get_dist_rank()}.log"
        self.result_file_name = f"perf_r{get_dist_rank()}.txt"
        self.log_path = self.log_dir / self.logging_file_name
        self.res_path = self.log_dir / self.result_file_name
        
        self.gradient_checkpointing_kwargs = {"use_reentrant": False}

        if self.parallel_stretagy == 'fsdp_dtensor':
            self.par_config = FSDP2Config(fsdp_activation_checkpointing=self.gradient_checkpointing, ac_kwargs=self.gradient_checkpointing_kwargs)
        elif self.parallel_stretagy == 'ddp':
            self.par_config = None
        elif self.parallel_stretagy == 'fsdp':
            self.par_config = None
        else:
            raise NotImplementedError
        self.wandb_run = None
        self.VERSION_STRING = str(int(time.time()))
    
    
    def init_wandb(self):
        # Save a stable run id in a small file so we reuse it across restarts
        run_id = hashlib.sha1((self.run_name+self.VERSION_STRING).encode("utf-8")).hexdigest()
        os.environ["WANDB_RUN_ID"] = run_id

        self.wandb_run = wandb.init(
            entity=os.environ["WANDB_ENTITY"],
            project=os.environ["WANDB_PROJECT"],
            name=self.run_name,
            id=run_id,
            resume=os.environ["WANDB_RESUME"],
            config=asdict(self),
            job_type=os.environ["WANDB_RUNTYPE"]
        )
        self.wandb_dir = self.wandb_run.dir
    def get_lora_config(self, model):
        return self.lora_cfg.get_lora_config(model=model, mha_only=self.mha_only)