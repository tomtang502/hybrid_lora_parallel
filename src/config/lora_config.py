import torch, re
from dataclasses import dataclass, field
from peft import LoraConfig
from typing import List, Optional, Iterable
from transformers import AutoModelForCausalLM

from src.utils.custom_type import Attn_types
from src.config.optim_config import collect_arch_specific_regex


def collect_layer_specific_targets(model, attn_type: Attn_types) -> List[str]:
    """
    Collect fully-qualified module names like:
      'model.layers.14.self_attn.q_proj'
    for the given basenames and layer indices.
    Works for HF models that follow '* .layers.<idx> .' naming.
    """

    module_regex_set = set(collect_arch_specific_regex(attn_type=attn_type))
    # pattern = rf'^(?s).*{re.escape(a)}.*{re.escape(b)}\Z'
    selected = []
    for name, _ in model.named_modules():
        # keep only exact basenames at the end, e.g., '.q_proj' not just 'proj'
        if any([re.match(r_exp, name) for r_exp in module_regex_set]):
            selected.append(name)
    return selected


@dataclass
class CustomLoraConfig:

    r: int = 32                         
    lora_alpha: int = 64                            # 2 * r
    lora_dropout: float = 0.1
    bias: str = "none"
    copy_weight: bool = False
    mha_only: bool = True
    
    
    target_modules: List[str] | None = None
    task_type: str = "CAUSAL_LM"
    # Nice-to-have for stability with higher ranks:
    use_rslora: bool = False
    
    collect_layer_specific_targets

    def get_modules_to_save(self) -> Optional[List[str]]:
        if self.finetune_embedding_and_head:
            # You may choose ["embed_tokens"] or both; careful of untie issue
            return ["embed_tokens", "lm_head"]
        return None

    def get_lora_config(
        self,
        model: AutoModelForCausalLM,
        mha_only: bool | None = None,
    ) -> LoraConfig:
        if mha_only is not None:
            self.mha_only = mha_only
        self.target_modules = collect_layer_specific_targets(model=model, attn_type='mha')
        if not self.mha_only :
            self.target_modules = self.target_modules + collect_layer_specific_targets(model=model, attn_type='gdn')
            
        self.target_modules = list(set(self.target_modules))
            
        common_kwargs = dict(
            r=self.r,
            lora_alpha=self.lora_alpha,
            lora_dropout=self.lora_dropout,
            bias=self.bias,
            target_modules=self.target_modules,
            task_type=self.task_type,
            use_rslora=self.use_rslora,
            modules_to_save=[],
        )

        return LoraConfig(**common_kwargs)
