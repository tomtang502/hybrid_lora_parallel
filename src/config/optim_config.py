import re
from dataclasses import dataclass
from typing import Union, Tuple, Literal, List
from src.utils.custom_type import Attn_types
from src.config.constants import (
    MHA_IDX, NUM_DECODER_LAYERS,
    MHA_MODULE_NAMES, GDN_MODULE_NAMES, FFN_MODULE_NAMES, 
    # MHA_LORA_MODULE_NAMES, GDN_LORA_MODULE_NAMES, FFN_LORA_MODULE_NAMES,
    MODUEL_REGEX_FORMAT,
)

def collect_arch_specific_regex(attn_type: Attn_types):
    if attn_type == 'mha':
        layer_idxes = MHA_IDX
        module_name_suffixes = list(MHA_MODULE_NAMES)+list(FFN_MODULE_NAMES)
    elif attn_type == 'gdn':
        layer_idxes = []
        for layer_idx in range(NUM_DECODER_LAYERS):
            if layer_idx not in MHA_IDX:
                layer_idxes.append(layer_idx)
        module_name_suffixes = list(GDN_MODULE_NAMES)+list(FFN_MODULE_NAMES)
    else:
        raise ValueError(f"Unsupported attention type: {attn_type}")
    
    regex_list = []
    for layer_idx in layer_idxes:
        for module_name_suffix in module_name_suffixes:
            regex_list.append(MODUEL_REGEX_FORMAT.format(layer_idx=re.escape(str(layer_idx)), module_name_suffix=re.escape(module_name_suffix)))
    
    return regex_list

# def collect_lora_arch_specific_regex(attn_type: Attn_types):
#     if attn_type == 'mha':
#         layer_idxes = MHA_IDX
#         module_name_suffixes = list(MHA_LORA_MODULE_NAMES)+list(FFN_LORA_MODULE_NAMES)
#     elif attn_type == 'gdn':
#         layer_idxes = []
#         for layer_idx in range(NUM_DECODER_LAYERS):
#             if layer_idx not in MHA_IDX:
#                 layer_idxes.append(layer_idx)
#         module_name_suffixes = list(GDN_LORA_MODULE_NAMES)+list(FFN_LORA_MODULE_NAMES)
#     else:
#         raise ValueError(f"Unsupported attention type: {attn_type}")
    
#     regex_list = []
#     for layer_idx in layer_idxes:
#         for module_name_suffix in module_name_suffixes:
#             regex_list.append(MODUEL_REGEX_FORMAT.format(layer_idx=re.escape(str(layer_idx)), module_name_suffix=re.escape(module_name_suffix)))
    
#     return regex_list

def name_regexes_to_param_regexes_lora(regex_list: List[str]):
    return [s.replace(r'\Z', r'\.base_layer\.weight\Z') for s in regex_list]

def name_regexes_to_param_regexes(regex_list: List[str]):
    return [s.replace(r'\Z', r'\.weight\Z') for s in regex_list]

@dataclass
class OptimizerConfig:
    optimizer_name = "adamw_torch"
    adam_betas = (0.9, 0.999)
    adam_eps = 1e-8
    weight_decay = 0.1
    learning_rate = 2e-5
    lr_scheduler_type = "constant_with_warmup"
    warmup_ratio = 0.05
    max_grad_norm: float = 1.0
    toggle_grad: bool = True
    
    no_wd_regex: Tuple[str]= (r".*\.A_log$", r".*\.dt_bias$", r".*\.bias$", r".*norm.*", r".*embed.*")
    no_grad_regex: Tuple[str] = ()
    
    def __post_init__(self):
        self.mha_weight = name_regexes_to_param_regexes(collect_arch_specific_regex(attn_type='mha'))
        self.gdn_weight = name_regexes_to_param_regexes(collect_arch_specific_regex(attn_type='gdn'))
        

@dataclass 
class LoRAOptimizerConfig(OptimizerConfig):

    mha_only: bool = False
    lora_weight_regex: Tuple[str] = (r".*lora_A.*", r".*lora_B.*")
    
    def __post_init__(self):
        self.mha_weight = name_regexes_to_param_regexes_lora(collect_arch_specific_regex(attn_type='mha'))
        if not self.mha_only:
            self.gdn_weight = name_regexes_to_param_regexes_lora(collect_arch_specific_regex(attn_type='gdn'))
        else:
            self.gdn_weight = name_regexes_to_param_regexes(collect_arch_specific_regex(attn_type='gdn'))
        no_grad_regex_cand = self.mha_weight + self.gdn_weight
        #
        # no_grad_regex_cand = []
        no_grad_regex_cand = no_grad_regex_cand + [r".*\.A_log$", r".*\.dt_bias$", r".*\.bias$"]
        # no_grad_regex_cand.append(r".*embed.*")
        # no_grad_regex_cand.append(r".*norm.*")
        self.no_grad_regex = tuple(set(no_grad_regex_cand))
        
            
            
        
        
        
        
    
    
    
    