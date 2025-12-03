"""
Implementing fsdp2 sharding of Jet-Nemotron    
    
"""

import torch, gc
import torch.distributed as dist
from typing import Dict
from torch.distributed.fsdp import fully_shard, FSDPModule, MixedPrecisionPolicy
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import checkpoint_wrapper


from src.utils.custom_type import (
    CasualLMModelLike, LogFunc, 
    PeftModelForCausalLM, GenerationMixin, 
    OptimizerNameType,
)
from src.utils.misc import mem_gc

from src.config.parallel_config import FSDP2Config

def get_base_model(model: CasualLMModelLike) -> GenerationMixin:
    if isinstance(model, PeftModelForCausalLM):
        base_model = model.base_model.model
    elif isinstance(model, GenerationMixin):
        base_model = model
    else:
        raise ValueError("Shard require at lesat PeftModelForCausalLM or AutoModelForCausalLM type")
    return base_model

def shard_model(model: CasualLMModelLike, main_LOG: LogFunc):
    """
    shard model, assume already toggled for require_grad layers
    """
    base_model = get_base_model(model=model)
    
    for layer_idx, layer in enumerate(base_model.model.layers):
        reshard_after_forward = int(layer_idx) < len(base_model.model.layers) - 1
        fully_shard(layer,
         mp_policy=MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32),
        reshard_after_forward=reshard_after_forward)
    fully_shard(model, 
                mp_policy=MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32),
                reshard_after_forward=True)
    # main_LOG(str(model), 'critical')
    # TODO remove this line
    assert isinstance(model, FSDPModule)
    return model

def _apply_activation_checkpointing(model: GenerationMixin, ac_kwargs: Dict) -> None:
    """Use activation checkpointing to reduce memory usage."""
    model.config.use_cache = False
    for layer_idx, transformer_block in enumerate(model.model.layers):
        wrapped_block = checkpoint_wrapper(transformer_block, preserve_rng_state=False)
        model.model.layers[layer_idx] = wrapped_block
    
    
def build_dist_model(
    model: torch.nn.Module, main_LOG: LogFunc, dist_cfg: FSDP2Config,
) -> torch.nn.Module:
    model = model.cuda()

    if dist_cfg.fsdp_activation_checkpointing:
        main_LOG("Apply activation checkpointing")
        _apply_activation_checkpointing(model,ac_kwargs=dist_cfg.ac_kwargs)

    if dist_cfg.compile_model:
        main_LOG("Compiling model with torch.compile...")
        # model = compile_model(model, model_cfg, dist_cfg)
        raise ValueError(f"Model compile not implemented yet")
        main_logger("->Model compiled successfully.")

   
    model = shard_model(model=model, main_LOG=main_LOG)
    
    mem_gc()

    return model