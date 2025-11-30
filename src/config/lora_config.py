import os, torch
from dataclasses import dataclass, field
from peft import LoraConfig
from typing import List, Optional


@dataclass
class BaseLoraConfig:

    rank: int = 32                         
    lora_alpha: int = 64                            # 2 * r
    lora_dropout: float = 0.1
    bias: str = "none"
    target_modules = [
        "q_proj",      # Attention Query
        "k_proj",      # Attention Key
        "v_proj",      # Attention Value
        "o_proj",      # Attention Output 
        "gate_proj",   # MLP Gate
        "up_proj",     # MLP Up-projection
        "down_proj"    # MLP Down-projection
    ]
    task_type: str = "CAUSAL_LM"
    # # Nice-to-have for stability with higher ranks:
    use_rslora: bool = False
    

    # # Default list (no embedding/head):
    # modules_to_save_default: List[str] = field(default_factory=list)
    # # Toggle for embedding + LM head:
    # finetune_embedding_and_head: bool = True

    # def get_modules_to_save(self) -> Optional[List[str]]:
    #     if self.finetune_embedding_and_head:
    #         # You may choose ["embed_tokens"] or both; careful of untie issue
    #         return ["embed_tokens", "lm_head"]
    #     return self.modules_to_save_default

    def get_lora_config(
        self,
    ) -> LoraConfig:
        common_kwargs = dict(
            r=self.rank,
            lora_alpha=self.lora_alpha,
            lora_dropout=self.lora_dropout,
            task_type=self.task_type,
            use_rslora=self.use_rslora,
            target_modules=self.target_modules,
        )
        return LoraConfig(**common_kwargs)