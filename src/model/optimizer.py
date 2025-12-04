"""
From pretrained repo
"""

import re
import json
from typing import Any


import torch.nn as nn
from torch.optim import AdamW, Optimizer
from torch.distributed import ProcessGroup
from transformers import GenerationMixin
from src.config.optim_config import OptimizerConfig
from src.utils.misc import val2tuple
from src.utils.dist import is_master, DistLogger

import logging
logger = logging.getLogger(__name__)
dist_logger = DistLogger(logger)


__all__ = ["build_optimizer", "toggle_requires_grad"]


def build_optimizer(model: nn.Module, opt_cfg: OptimizerConfig) -> Optimizer:
    param_groups = _get_param_groups(model, opt_cfg, toggle_grad=opt_cfg.toggle_grad)
    optimizer = _get_optimizer(param_groups, opt_cfg)
    return optimizer


def _get_optimizer(param_groups: list[dict[str, Any]], opt_cfg: OptimizerConfig) -> Optimizer:
    if opt_cfg.optimizer_name == "adamw_torch":
        optimizer = AdamW(
            param_groups,
            lr=opt_cfg.learning_rate,
            betas=opt_cfg.adam_betas,
            eps=opt_cfg.adam_eps,
            weight_decay=opt_cfg.weight_decay,
        )
    else:
        raise NotImplementedError
    return optimizer

def toggle_requires_grad(model, opt_cfg: OptimizerConfig):
    _patterns = ()
    if hasattr(model, "get_no_grad_params"):
        _patterns = val2tuple(model.get_no_grad_params())
    no_grad_patterns = _patterns + tuple(opt_cfg.no_grad_regex)
    
    for k, module in model.named_parameters():
        module.requires_grad = True
        if any([re.match(r_exp, k) for r_exp in no_grad_patterns]):
            module.requires_grad = False

def _get_param_groups(model: GenerationMixin, opt_cfg: OptimizerConfig, toggle_grad=True):
    # if getattr(model, "model_type", "unknown") == "hf":
    #     if getattr(model.base_model.config, "tie_word_embeddings", True):
    #         # hf_model_tie_embeds is True
    #         assert (".*embed.*" in opt_cfg.no_wd_regex) ^ ("head" in opt_cfg.no_wd_regex), "For HF model, embed and head need to both in no wd"

    embed_full_name, embed_param = None, None
    wd_group, nwd_group = {}, {}
    wd_group['weight_decay'] = opt_cfg.weight_decay
    nwd_group['weight_decay'] = 0
    wd_group['params'] = []
    wd_group['param_names'] = []
    nwd_group['params'] = []
    nwd_group['param_names'] = []

    if toggle_grad:
        toggle_requires_grad(model=model, opt_cfg=opt_cfg)
        
    
    for name, param in model.named_parameters():
        if "embed" in name:
            embed_full_name = name
            embed_param = param
        # activate grad first, then deactivate
        # param.requires_grad = True
        if param.requires_grad:
            # explicit no_wd_regex
            if any(re.match(p, name) for p in opt_cfg.no_wd_regex):
                nwd_group["params"].append(param)
                nwd_group['param_names'].append(name)
            # lora weight
            elif any(re.match(p, name) for p in opt_cfg.lora_weight_regex):
                nwd_group['params'].append(param)
                nwd_group['param_names'].append(name)
            else:
                wd_group['params'].append(param)
                wd_group['param_names'].append(name)
        else:
            dist_logger.info(f"Excluding {name} from optimizer")
                    

    # tied embedding post checking
    # assert embed_param is not None, "Need to be able to find embedding layer"
    # assert model.lm_head.weight is embed_param, "Need to make sure embed layer and lm_head is tied under same param"

    param_groups = [wd_group, nwd_group]
    return param_groups