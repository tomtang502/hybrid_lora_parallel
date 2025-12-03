# Copyright 2025 NVIDIA CORPORATION & AFFILIATES
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any

import numpy as np
import torch, re, os, time, random, gc
import torch.nn as nn

from src.utils.dist import get_dist_rank, sync_tensor

def mem_gc():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
def val2tuple(x: tuple | list | Any, min_len: int = 1, idx_repeat: int = -1) -> tuple:
    if isinstance(x, (list, tuple)):
        x = list(x)
    else:
        x = [x]

    if len(x) > 0:
        x[idx_repeat:idx_repeat] = [x[idx_repeat] for _ in range(min_len - len(x))]

    return tuple(x)


def get_amp_dtype(name: str) -> torch.dtype:
    amp_dict = {
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
        "fp32": torch.float32,
    }

    if name in amp_dict:
        return amp_dict[name]
    else:
        raise ValueError(f"Unsupported amp_dtype: {name}")

def get_amp_dtype_by_bool(bf16: bool, fp16:bool) -> torch.dtype:
    if bf16: return torch.bfloat16
    elif fp16: return torch.float16
    else: return torch.float32

def seed_all(seed: int, reset: bool = False) -> int:
    if reset:
        seed = int(sync_tensor(int(time.time()), reduce="root"))
    seed += get_dist_rank()

    if seed < 0 or seed > 2**32 - 1:
        raise ValueError(f"Seed {seed} is invalid. It must be on [0; 2^32 - 1]")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    return seed

def print_trainable_modules(model, main_LOG=print):
    # Print header
    main_LOG(f"\033[92m{'Name':60}\033[0m | "
            f"\033[96mShape\033[0m | "
            f"\033[91mCount\033[0m")
    main_LOG("-" * 80)
    for name, param in model.named_parameters():
        if param.requires_grad:
            shape_str = str(list(param.shape))
            main_LOG(f"\033[92m{name:60}\033[0m | "   # green
                    f"\033[96m{shape_str}\033[0m | " # cyan
                    f"\033[91m{param.numel():,}\033[0m")  # red
    trainable, total = 0, 0
    for n, p in model.named_parameters():
        ct = p.numel()
        total += ct
        if p.requires_grad:
            trainable += ct
    main_LOG(f"Trainable params: {trainable/1e6:.2f}M / {total/1e6:.2f}M ({100*trainable/total:.2f}%)")

def get_trainable_modules_string(model, main_LOG=None):
    """
    Returns a string containing the trainable parameters table.
    If main_LOG is provided (e.g., print), it also prints the lines.
    """
    buffer = []

    def log_and_store(msg):
        buffer.append(msg)
        if main_LOG:
            main_LOG(msg)

    # Header
    log_and_store(f"\033[92m{'Name':60}\033[0m | "
                  f"\033[96mShape\033[0m | "
                  f"\033[91mCount\033[0m")
    log_and_store("-" * 80)

    # Params
    trainable, total = 0, 0
    for name, param in model.named_parameters():
        ct = param.numel()
        total += ct
        if param.requires_grad:
            trainable += ct
            shape_str = str(list(param.shape))
            log_and_store(f"\033[92m{name:60}\033[0m | "
                          f"\033[96m{shape_str}\033[0m | "
                          f"\033[91m{param.numel():,}\033[0m")
    
    # Summary
    log_and_store(f"Trainable params: {trainable/1e6:.2f}M / {total/1e6:.2f}M ({100*trainable/total:.2f}%)")
    
    return "\n".join(buffer)