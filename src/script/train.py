#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
LoRA SFT for a hybrid LLM (e.g., attention + SSM/DeltaNet/Mamba + gated-delta/FLA)
- Loads with AutoModelForCausalLM so it works for checkpoints named 'jetlm'
- Optionally 4-bit or 8-bit loading (bitsandbytes)
- Auto-discovers common linear projections to target with LoRA
- Packs sequences for efficient training
- Evaluates perplexity; saves adapters and optionally merged weights

Requires:
  transformers>=4.41, datasets, peft>=0.10, accelerate, bitsandbytes (if 4/8-bit)
"""

import datetime, torch
import torch.distributed as dist
import os, wandb, shlex, tempfile
import logging, draccus
import numpy as np
from typing import Literal
from transformers import (
    set_seed,
)

from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM

from src.config.train_config import HLParTrainConfig
from src.config.constants import (
    ORANGE, RED, BLUE, RESET, CYAN
)

from src.utils.misc import get_trainable_modules_string, seed_all
from src.utils.dist import (
    dist_barrier, 
    dist_init, 
    dist_close, 
    is_master, 
    get_world_size_safe,
    get_dist_rank,
    get_dist_local_rank,
)
from src.utils.custom_type import LogFunc
from src.utils.text_handle import create_txt_file, append_to_txt_file
from src.data.data_builder import get_dataloader
from src.profile.vram import per_device_report
from src.trainer import BaseTrainer

from src.model import (
    build_dist_model,
    wrap_model_with_linear_lora,
    merge_lora_and_unwrap,
    get_base_model,
    toggle_requires_grad,
    prepare_model_for_ddp,
    prepare_model_for_fsdp,
    get_block_class_from_model
    
)
from src.utils.log_utils import setup_logging, silence_logger

def debug_env():
    keys = ["RANK","WORLD_SIZE","LOCAL_RANK","MASTER_ADDR","MASTER_PORT","CUDA_VISIBLE_DEVICES"]
    print("ENV:", {k: os.environ.get(k) for k in keys}, flush=True)

# --------------------------
# Main
# --------------------------
@draccus.wrap()
def main(cfg: HLParTrainConfig):
    if is_master():
        LOG = setup_logging(cfg.log_path)
        cfg.init_wandb()
    
    def main_LOG(s, level: Literal["info", "critical", "warning"]='info'):
        if is_master():
            if level == "info":
                LOG.info(s)
            elif level == "critical":
                LOG.critical(s)
            elif level == "warning":
                LOG.warning(s)
    
   
    dist_init()
    
    seed = seed_all(cfg.seed)
    
    main_LOG(f"Building dataloader with global_batch_size={cfg.global_batch_size} and total_size={cfg.total_size}")
    
    tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer_name, trust_remote_code=cfg.trust_remote_code)
    # pad to pad==eos if pad token undefined
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token if tokenizer.eos_token else tokenizer.unk_token
    tokenizer.padding_side = cfg.tokenizer_pad_side

    train_loader = get_dataloader(data_path=cfg.dataset_path, tokenizer=tokenizer, batch_size=cfg.per_device_batch_size*cfg.gradient_accumulation_steps,
                                  chunk_size=cfg.chunk_size, infinite=True, rank=get_dist_rank(), world_size=cfg.num_devices, num_workers=cfg.dataloader_num_workers)

  
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name_or_path,
        trust_remote_code=cfg.trust_remote_code,
        attn_implementation=cfg.attn_implementation,
        torch_dtype=torch.bfloat16 if cfg.bf16 else torch.float32,
        low_cpu_mem_usage=cfg.low_cpu_mem_usage,
        device_map='cpu',
    )
    
    if cfg.use_lora:
        main_LOG(f"{BLUE}Activating base model embeddings{RESET}")
    
        main_LOG(f"{BLUE}Activating base model norm layers{RESET}")
        model = wrap_model_with_linear_lora(model=model, lora_cfg=cfg.lora_cfg)
    if cfg.bf16:
        model.to(torch.bfloat16)
    model.cuda()
    
    fsdp_block_class = get_block_class_from_model(model)
    toggle_requires_grad(model=model, opt_cfg=cfg.opt_cfg)
    cfg.opt_cfg.toggle_grad = False


    create_txt_file(cfg.res_path)
    model_report = get_trainable_modules_string(model=model, main_LOG=main_LOG)
    append_to_txt_file(file_path=cfg.res_path, content=model_report)

    vram_report = per_device_report("Model footprint Before FSDP build")
    append_to_txt_file(file_path=cfg.res_path, content=vram_report)
    dist_barrier()
    
    if cfg.parallel_stretagy == 'fsdp_dtensor':
        model = build_dist_model(model=model, main_LOG=main_LOG, dist_cfg=cfg.par_config)
    elif cfg.parallel_stretagy == 'ddp':
        model = prepare_model_for_ddp(model=model, local_rank=get_dist_local_rank(), 
                                      enable_grad_ckpt=cfg.gradient_checkpointing, gradient_checkpointing_kwargs=cfg.gradient_checkpointing_kwargs, 
                                      find_unused_parameters=cfg.ddp_find_unused_parameters)
    elif cfg.parallel_stretagy == 'fsdp':
        model = prepare_model_for_fsdp(model=model, local_rank=get_dist_local_rank(), block_class=fsdp_block_class, 
                                       enable_grad_ckpt=cfg.gradient_checkpointing, gradient_checkpointing_kwargs=cfg.gradient_checkpointing_kwargs,
                                       mixed_precision=cfg.bf16, use_cpu_offload=cfg.fsdp_cpu_offload)
    
    
    vram_report = per_device_report("Model footprint AFTER FSDP build")
    append_to_txt_file(file_path=cfg.res_path, content=vram_report)
    dist_barrier()
    
    main_LOG(f"{str(cfg.wandb_run)}")
    trainer = BaseTrainer(model=model, cfg=cfg, train_loader=train_loader, main_LOG=main_LOG, 
                           wandb_run=cfg.wandb_run)
    dist_barrier()
    
    
    main_LOG(f"Random seed: {seed}", level='info')
    
    trainer.train()
    dist_barrier()
    
    main_LOG(f"{ORANGE}Training Done (result saved at {cfg.res_path}){RESET}")


    # ------ closing procedure ------ #
    dist_barrier()
    dist_close()
 


if __name__ == "__main__":
    main()
