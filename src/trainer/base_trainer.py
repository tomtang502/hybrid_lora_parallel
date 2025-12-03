"""
This module defines the core training routine for finetuning causal language
models with LowRank Adapters (LoRA).
"""


import os
import torch

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
)
from torch.utils.data import DataLoader
from peft import LoraConfig, get_peft_model

import os
import re
import math
import time
import shutil
import traceback
from copy import deepcopy
from datetime import datetime
from typing import Any, Optional
from more_itertools import peekable
from collections import defaultdict
from pathlib import Path
from tqdm import tqdm

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
import torch.nn as nn
from torch.distributed._tensor import DTensor
from torch.nn.parallel import DistributedDataParallel
from torch.amp import GradScaler
from torch.utils.data import DataLoader
from transformers import get_constant_schedule_with_warmup
from contextlib import nullcontext

from src.model.optimizer import build_optimizer
from src.config.train_config import HLParTrainConfig
from src.utils.dist import is_master, dist_barrier, get_dist_rank, sync_tensor
from src.profile.vram import measure_vram
from src.profile.perf_monitor import PerformanceMonitor
from src.utils.custom_type import LogFunc
from src.utils.misc import get_amp_dtype_by_bool
from src.config.constants import (
    RED, CYAN, RESET
)
from src.utils.text_handle import append_to_txt_file


__all__ = ["BaseTrainer"]

class BaseTrainer:
    def __init__(
        self,
        train_loader: DataLoader,
        cfg: HLParTrainConfig,
        model: nn.Module,
        main_LOG: LogFunc,
        wandb_run = None,
        # device_mesh: Optional[torch.distributed.ProcessGroup],
        # reset_log_dir=True,
    ) -> None:
        self.train_loader = train_loader
        self.cfg = cfg
        self.model = model
        self.pad_token_id = train_loader.pad_token_id
        self.wandb_run = wandb_run
        
        self.main_log_fn = main_LOG
        self.global_batch_size = cfg.global_batch_size
        self.gradient_accumulation_steps = self.cfg.gradient_accumulation_steps
        self.num_devices = cfg.num_devices
        
        self.res_path = cfg.res_path
        
        self.next_train_data_for_check = None
        
        # TODO Here for latter pipeline parallelism    
        # self.experimental_cfg = config.get("experimental", None)
        # self.device_mesh = device_mesh
        # TODO

        self.perf_monitor = PerformanceMonitor()
        self.train_iter = peekable(self.train_loader)
        assert isinstance(self.model, (nn.Module, DistributedDataParallel))

        self.parallel_stretagy = cfg.parallel_stretagy

        # progress tracker
        self.global_step = 0
        
        # amp, enable_amp, amp_dtype, amp_scaler
        self._setup_amp(bf16=cfg.bf16, fp16=False)
        
        # work_dir, checkpoint_dir, data_dir, log_dir
        self._setup_dirs(cfg.reset_log_dir)
        
        # optimizer, lr_scheduler (no lr_monitor), wandb setup already in cfg
        self._setup_training()

    @property
    def unwrapped_model(self) -> nn.Module:
        if isinstance(self.model, DistributedDataParallel):
            return self.model.module
        else:
            return self.model
    
    # CHECK
    def _setup_dirs(self, reset_log_dir: bool) -> None:
        self.log_dir = self.cfg.log_dir
        if hasattr(self.cfg, "wandb_dir"):
            self.wandb_dir = self.cfg.wandb_dir

        if os.path.exists(self.log_dir) and reset_log_dir:
            if is_master():
                shutil.rmtree(self.log_dir)
                os.makedirs(self.log_dir, exist_ok=True)
        dist_barrier()

    # CHECK
    def _setup_amp(self, bf16: bool, fp16: bool) -> None:
        self.amp_dtype = get_amp_dtype_by_bool(bf16=bf16, fp16=fp16)
        self.enable_amp = self.amp_dtype != torch.float32
        self.amp_scaler = GradScaler("cuda", enabled=self.enable_amp)

    # CHECK
    def _setup_training(self) -> None:
        # optimizer
        
        self.optimizer = build_optimizer(model=self.model, opt_cfg=self.cfg.opt_cfg)
        # lr scheduler & lr monitor
        self.lr_scheduler = get_constant_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=self.cfg.num_steps*self.cfg.opt_cfg.warmup_ratio,
        )
        # wandb & wandb_cache removed
        self.model.training = True
        self.max_steps = self.cfg.num_steps
        self.global_step = 0
    
    def next_train_data(self) -> dict[str, Any]:
        try:
            feed_dict = next(self.train_iter)
        except StopIteration:
            self.train_iter = peekable(self.train_loader)
            feed_dict = self.next_train_data()
        return feed_dict
    
    def peek_next_train_data(self) -> dict[str, Any]:
        try:
            feed_dict = self.train_iter.peek()
        except StopIteration:
            new_train_iter = peekable(self.train_loader)
            self.train_iter = new_train_iter
            feed_dict = new_train_iter.peek()
        return feed_dict

    def get_train_batch(self) -> dict[str, Any]:
        feed_dict = self.next_train_data()
        feed_dict["input_ids"] = feed_dict["input_ids"].cuda(non_blocking=True)
        
        if "labels" in feed_dict:
            feed_dict["labels"] = feed_dict["labels"].cuda(non_blocking=True)
        else:
            feed_dict["labels"] = feed_dict["input_ids"].clone()
            if self.pad_token_id is not None:
                feed_dict["labels"][feed_dict["labels"] == self.pad_token_id] = -100 

        if "attention_mask" in feed_dict:
            feed_dict["attention_mask"] = feed_dict["attention_mask"].cuda(non_blocking=True)
        else:
            feed_dict["attention_mask"] = None

        return feed_dict
    
    def _micro_step(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
        loss_ratio: float,
        position_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        experimental_args: Optional[dict] = None,
    ) -> dict[str, Any]:
        # forward and backward
        with torch.autocast(device_type="cuda", dtype=self.amp_dtype, enabled=self.enable_amp):
            loss = self.model(input_ids, attention_mask=attention_mask, labels=labels)["loss"]  # B, T, D
            loss = loss * loss_ratio
        loss.backward()
        return {
            "full_loss": loss.detach(),
            "ce_loss": loss.detach(),
        }

    def train_step(
        self, grad_clip: Optional[float] = None
    ) -> dict[str, Any]:
        # self.model.train()
        self.optimizer.zero_grad(set_to_none=True)

        # before step
        feed_dict = self.get_train_batch()

        batch_size = feed_dict["input_ids"].size(0)
        assert batch_size%self.gradient_accumulation_steps == 0, f"batch size {batch_size} need to be divisible by gradient accumulation steps {self.gradient_accumulation_steps}"
        micro_batch_size = batch_size // self.gradient_accumulation_steps
        
        step_ntokens = feed_dict["input_ids"].numel()
        step_time_start = time.perf_counter()
        if batch_size > micro_batch_size:
            chunks = math.ceil(batch_size / micro_batch_size)
            all_input_ids = torch.chunk(feed_dict["input_ids"], chunks=chunks)
            all_labels = torch.chunk(feed_dict["labels"], chunks=chunks)
            all_attention_mask = (
                torch.chunk(feed_dict["attention_mask"], chunks=chunks)
                if feed_dict["attention_mask"] is not None
                else [None] * chunks
            )
        else:
            chunks = 1
            all_input_ids = [feed_dict["input_ids"]]
            all_labels = [feed_dict["labels"]]
            all_attention_mask = [feed_dict["attention_mask"]] if feed_dict["attention_mask"] is not None else [None]

        all_position_ids = []
        for input_ids in all_input_ids:
            position_ids = torch.arange(input_ids.shape[1], dtype=torch.long, device=input_ids.device)
            position_ids = position_ids.unsqueeze(0).expand(input_ids.shape[0], -1)
            all_position_ids.append(position_ids)

        experimental_args = {}

        # # run step
        logging_metrics = defaultdict(float)
        
        loss = 0.0
        
        for i, (input_ids, position_ids, labels, attention_mask) in enumerate(
            zip(all_input_ids, all_position_ids, all_labels, all_attention_mask)
        ):              
            loss_ratio = input_ids.size(0) / batch_size

            experimental_args["is_gradient_accumulation_start"] = (i == 0)

            is_last_microstep = (i == chunks - 1)
            sync_context = nullcontext()
            if self.parallel_stretagy == "ddp":
                # In DDP, use no_sync() for all steps EXCEPT the last one
                if not is_last_microstep:
                    sync_context = self.model.no_sync()
            elif self.parallel_stretagy == 'fsdp_dtensor':
                self.model.set_requires_gradient_sync(is_last_microstep)
                

            with sync_context:
                micro_dict = self._micro_step(
                    input_ids, labels, loss_ratio, position_ids, attention_mask,
                    experimental_args=experimental_args)
                
            loss += micro_dict['ce_loss']
        logging_metrics['train/ce_loss'] = loss

        for k in logging_metrics:
            logging_metrics[k] = sync_tensor(logging_metrics[k], reduce="mean", dim=0).item()
        
        optimizer_stat = self._optimizer_step(grad_clip)
        logging_metrics['train/step_time'] = time.perf_counter() - step_time_start
        logging_metrics['train/grad_norm'] = optimizer_stat['grad_norm']

        tokens_per_sec = step_ntokens / logging_metrics['train/step_time']
        logging_metrics['train/tokens_per_sec']  = tokens_per_sec

        # # update lr and progress
        logging_metrics['train/learning_rate'] = self.optimizer.param_groups[0]["lr"]
        self.lr_scheduler.step()

        self.global_step += 1
        
        return logging_metrics

    def _optimizer_step(self, grad_clip: float) -> dict[str, Any]:
            
        if grad_clip is not None:
            trainable_params = []
            for group in self.optimizer.param_groups:
                trainable_params.extend(group["params"])

            if self.parallel_stretagy == 'fsdp':
                # FSDP specific method: handles global norm reduction internally
                grad_norm = self.model.clip_grad_norm_(grad_clip)
            else:
                # DTensor handles the global norm reduction automatically here
                grad_norm = torch.nn.utils.clip_grad_norm_(trainable_params, grad_clip)
            
            if isinstance(grad_norm, DTensor):
                grad_norm = grad_norm.full_tensor()

            if isinstance(grad_norm, torch.Tensor):
                grad_norm = grad_norm.item()
        else:
            grad_norm = 0.0
            
        optim_stat = {}
        
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)

        optim_stat.update({"grad_norm": grad_norm})
        

        return optim_stat
    
    def train(self):
        # main train loop
        with tqdm(total=self.max_steps, initial=self.global_step, desc="Train", disable=not is_master()) as t:
            total_tokens = self.max_steps * self.global_batch_size * self.cfg.chunk_size
            self.main_log_fn(f"Starting training, {self.global_step=}, global_batch_size={self.global_batch_size} accmulated from {self.cfg.gradient_accumulation_steps} steps with chunk length {self.cfg.chunk_size}, total training tokens = {total_tokens / 1e9:.2f}B", 'critical')

            for step in range(self.global_step, self.max_steps):
                with measure_vram("memory", device=get_dist_rank()) as vram_stats:
                    logging_metrics = self.train_step(grad_clip=self.cfg.opt_cfg.max_grad_norm)
                logging_metrics.update(vram_stats.metrics)

                if step > self.cfg.warm_up_steps:
                    self.perf_monitor.record_step(latency=logging_metrics['train/step_time'], throughput=logging_metrics['train/tokens_per_sec'], vram=logging_metrics[f"memory/peak_allocated_mb"])
                if self.global_step % self.cfg.logging_steps == 0:
                    if self.global_step > 1 or self.cfg.logging_first_step:
                        if is_master():
                            self.wandb_run.log(logging_metrics, step=step, commit=True)
                        self.main_log_fn(f"[Metrics]\n{str(logging_metrics)}")
                t.update()
        append_to_txt_file(self.cfg.res_path, self.perf_monitor.generate_report_string())
        return self.perf_monitor.get_report()