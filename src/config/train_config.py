"""
This module defines a dataclass that bundles together the various
hyperparameters and settings needed for finetuning a causal language model
using parameterefficient LoRA adapters.
"""
import os, wandb, json, uuid
from dataclasses import dataclass, field, asdict
from typing import Optional
from pathlib import Path
from transformers import TrainingArguments
from .lora_config import BaseLoraConfig
from .lora_param_heuristic import get_lora_params


@dataclass
class TrainingConfig:
    """Container for training hyperparameters and LoRA settings."""

    # Name or path of the pretrained model to fine‑tune.  When training
    # Qwen‑2.5 models, provide the appropriate identifier from the Hub.
    model_name_or_path: str = "Qwen/Qwen2.5-1.5B"

    # Maximum sequence length for tokenization.  Examples longer than this
    # value will be truncated on the right.  Adjust based on GPU memory.
    # max_seq_length: int = 4096

    # Learning rate used by the AdamW optimizer.
    learning_rate: float = 2e-5

    # Total number of training epochs.
    num_train_epochs: int = 1
    lr_scheduler_type: str = "cosine"
    warm_up_ratio: float = 0.05

    per_device_batch_size: int = 2

    gradient_accumulation_steps: int = 4

    init_with_lora: bool = False

    # Directory where checkpoints and the final model will be saved.
    ckpt_dir_root: str = "./ckpts/baseline"

    # Random seed for reproducibility.
    seed: int = 42
    
    # Logging
    log_dir_root: str = "logs"
    logging_steps=2
    report_to="wandb"
    
    # Evaluation
    evaluation_strategy="no"
    
    # checkpointing
    save_steps=200
    save_total_limit=3
    
    # fintuning dataset type
    dataset_type: str = "math"
    
    # LoRA Config
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    use_rslora: bool = True
    lora_effective_scale: int = 2
    
    def __post_init__(self):
        if self.init_with_lora:
            self.lora_alpha, self.lora_dropout = get_lora_params(lora_rank=self.lora_rank, use_rslora=self.use_rslora, 
                                              scale=self.lora_effective_scale)
            self.lora_config = BaseLoraConfig(rank=self.lora_rank, lora_alpha=self.lora_alpha, 
                                              lora_dropout=self.lora_dropout, use_rslora=self.use_rslora)
            if self.use_rslora:
                self.run_name = f"baseline_lftr={self.lora_config.rank}_rs_{self.dataset_type}"
            else:
                self.run_name = f"baseline_lftr={self.lora_config.rank}_{self.dataset_type}"
        else:
            self.lora_config = None
            self.run_name = f"baseline_fft_{self.dataset_type}"
        lr_str = str(self.learning_rate).replace('.', 'd')
        wu_str = str(self.warm_up_ratio).replace('.', 'd')
        self.run_name += f"_{lr_str}_{self.lr_scheduler_type}{wu_str}"
        os.makedirs(self.log_dir_root, exist_ok=True)
        os.makedirs(self.ckpt_dir_root, exist_ok=True)
        
        self.log_dir = f"{self.log_dir_root}/{self.run_name}"
        os.makedirs(self.log_dir, exist_ok=True)
        self.ckpt_dir = f"{self.ckpt_dir_root}/{self.run_name}"
        os.makedirs(self.ckpt_dir, exist_ok=True)
        self.warm_up_ratio = 0.05
        
    
    def get_trainer_args(self):
        print("batch_size:", self.per_device_batch_size)
        training_args = TrainingArguments(
            run_name=self.run_name,
            do_train=True,
            do_eval=False,
            do_predict=False,
            per_device_train_batch_size=self.per_device_batch_size,
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            num_train_epochs=self.num_train_epochs,
            learning_rate=self.learning_rate,
            
            logging_dir=self.log_dir, 
            logging_strategy="steps",
            logging_steps=self.logging_steps,
            report_to=self.report_to,
            
            lr_scheduler_type=self.lr_scheduler_type, 
            warmup_ratio=self.warm_up_ratio,
            

            
            output_dir=self.ckpt_dir,
            save_steps=self.save_steps,
            save_total_limit=self.save_total_limit,
            eval_strategy=self.evaluation_strategy,
            
            dataloader_pin_memory=True,
            dataloader_num_workers=16,
            gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
            bf16=True,                     # mixed precision autocast in bf16
            # bf16_full_eval=True,           # eval/generation also in bf16
            # optim="adamw_torch_fused",     # good fused optimizer on recent PyTorch
            tf32=True,    
        )
        return training_args
         
    def init_wandb(self):
        runid_file = Path(f"{self.log_dir}") / "wandb_runid.json"
        if runid_file.exists():
            run_id = json.loads(runid_file.read_text())["run_id"]
        else:
            run_id = str(uuid.uuid4())
            runid_file.write_text(json.dumps({"run_id": run_id}))

        os.environ["WANDB_RUN_ID"] = run_id

        run = wandb.init(
            entity=os.environ["WANDB_ENTITY"],
            project=os.environ["WANDB_PROJECT"],
            dir=self.log_dir,
            name=self.run_name,
            id=run_id,
            resume=os.environ["WANDB_RESUME"],
            config=asdict(self)
        )
        self.run = run
