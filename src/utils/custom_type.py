from typing import Callable, Literal, Optional, Sequence, Tuple, Union
from transformers import AutoModelForCausalLM, PreTrainedModel, GenerationMixin
from peft import PeftMixedModel, PeftModel, PeftModelForCausalLM

Attn_types = Literal['mha', 'gdn']
OptimizerNameType = Literal["adamw"]
CasualLMModelLike = Union[PeftModelForCausalLM, GenerationMixin]
ModelLike = Union[PreTrainedModel, PeftModel, PeftMixedModel, AutoModelForCausalLM]
LogFunc = Callable[[str, Literal["info", "critical", "warning"]], None]