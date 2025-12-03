
from dataclasses import dataclass, field

from typing import Optional, Dict, Any, Tuple, Literal

@dataclass
class FSDP2Config:
    # cpu_offload: bool = True
    # sync_module_states: bool = True
    # backward_prefetch: Literal["BACKWARD_PRE", "BACKWARD_POST", "NONE"] = "BACKWARD_PRE"
    # use_orig_params: bool = True  # recommended on newer PyTorch (2.2+); safe to keep True
    # limit_all_gathers: bool = True   
    compile_model: bool = False
    fsdp_activation_checkpointing: bool = True
    ac_kwargs: dict = field(default_factory=lambda: {"use_reentrant": False})
    
    