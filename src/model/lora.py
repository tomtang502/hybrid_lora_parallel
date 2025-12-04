import torch
import torch.nn as nn
import math
from src.config.lora_config import CustomLoraConfig

class LoraLinear(nn.Module):
    """
    copy_weight: if set to true, will copy weight adn initialize new linear layer, 
                 otherwise link to original linear base layer
    """

    def __init__(
        self,
        base_linear: nn.Linear,
        r: int = 16,
        lora_alpha: int = 16,
        lora_dropout: float = 0.0,
        copy_weight: bool = False,
        freeze_weight: bool = False,
    ):
        super().__init__()
        self.in_features = base_linear.in_features
        self.out_features = base_linear.out_features
        self.r = r
        self.lora_alpha = lora_alpha
        self.merged = False
        self.device = base_linear.weight.device
        self.dtype = base_linear.weight.dtype

        # Base linear (copy or link)
        if copy_weight:
            self.base_layer = LoraLinear.copy_linear(lin=base_linear)
        else:
            self.base_layer = base_linear
        
        # LoRA parts
        if r > 0:
            self.lora_A = nn.Linear(self.in_features, r, bias=False, device=self.device, dtype=self.dtype)
            self.lora_B = nn.Linear(r, self.out_features, bias=False, device=self.device, dtype=self.dtype)
            self.scaling = lora_alpha / r
            self.lora_dropout = nn.Dropout(p=lora_dropout)
        else:
            # degenerate case
            self.lora_A = None
            self.lora_B = None
            self.scaling = 0.0
            self.lora_dropout = nn.Identity()

        self.reset_lora_parameters()

        if freeze_weight:
            # Freeze base weights by default (LoRA fine-tuning only)
            self.base_layer.weight.requires_grad = False
            if self.base_layer.bias is not None:
                self.base_layer.bias.requires_grad = False

    def reset_parameters(self):
        # LoRA init: typical pattern is small A, zeros B
        self.reset_lora_parameters()

    def reset_lora_parameters(self):
        if self.r > 0:
            nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.r <= 0 or self.merged:
            # Either no LoRA, or LoRA already merged into base weights
            return self.base_layer(x)

        # Base path
        result = self.base_layer(x)

        # LoRA path
        result = result + self.scaling * self.lora_B(self.lora_A(self.lora_dropout(x)))

        return result

    # Optional: helpers for merging LoRA into the base weight (for inference)
    def merge(self):
        if self.r > 0 and not self.merged:
            # W <- W + (alpha/r) * B @ A
            # (B: out x r, A: r x in) => (out x in)
            delta_w = self.lora_B.weight @ self.lora_A.weight
            self.base_layer.weight.data += self.scaling * delta_w
            self.merged = True

    def unmerge(self):
        if self.r > 0 and self.merged:
            delta_w = self.lora_B.weight @ self.lora_A.weight
            self.base_layer.weight.data -= self.scaling * delta_w
            self.merged = False

    @classmethod
    def copy_linear(cls, lin: nn.Linear) -> nn.Linear:
        duplicate = nn.Linear(in_features=lin.in_features, out_features=lin.out_features,
                              bias=lin.bias is not None, device=lin.weight.device, dtype=lin.weight.dtype)
        with torch.no_grad():
            duplicate.weight.copy_(lin.weight)
            if lin.bias is not None:
                duplicate.bias.copy_(lin.bias)
        return duplicate


def merge_lora_and_unwrap(model):
    # Iterate over children
    for name, module in list(model.named_children()):
        # 1. Recurse first
        merge_lora_and_unwrap(module)

        # 2. Check if this child is your custom LoraLinear wrapper
        if isinstance(module, LoraLinear):
            base = module.base_layer
            
            # Create a clean nn.Linear
            new_linear = nn.Linear(
                in_features=module.in_features,
                out_features=module.out_features,
                bias=(base.bias is not None),
                device=base.weight.device,
                dtype=base.weight.dtype
            )

            with torch.no_grad():
                # Start with base weights
                W_merged = base.weight.data.clone()

                # Merge LoRA weights if rank > 0
                if module.r > 0:
                    A = module.lora_A
                    B = module.lora_B
                    scaling = module.scaling
                    
                    # W_new = W_base + scaling * (B @ A)
                    delta_w = B.weight.data @ A.weight.data
                    W_merged += scaling * delta_w

                # Copy weights
                new_linear.weight.data.copy_(W_merged)
                if base.bias is not None:
                    new_linear.bias.data.copy_(base.bias.data)

            # 3. Replace the LoraLinear with the new standard Linear
            # FIX: Only use index access if it is a list AND the name is a number
            if isinstance(model, (nn.ModuleList, nn.Sequential)) and name.isdigit():
                model[int(name)] = new_linear
            else:
                # Fallback to setattr for named attributes (e.g., 'w1', 'q_proj')
                setattr(model, name, new_linear)
    return model

def replace_linear_with_AB(model, target_names=("q_proj", "v_proj"), rank=8):
    for name, module in model.named_children():
        # Recurse
        replace_linear_with_AB(module, target_names, rank)

        # Replace leaf modules
        if isinstance(module, nn.Linear) and any(t in name for t in target_names):
            setattr(model, name, LoraLinear.from_linear(module, rank))


def get_parent_and_attr(root: nn.Module, full_name: str):
    """
    Given full_name like 'layers.0.self_attn.q_proj',
    return (parent_module, 'q_proj').
    """
    parts = full_name.split(".")
    parent = root
    for p in parts[:-1]:
        # Handle ModuleList indices vs normal attributes
        if p.isdigit():
            parent = parent[int(p)]
        else:
            parent = getattr(parent, p)
    return parent, parts[-1]

def wrap_model_with_linear_lora(model, lora_cfg: CustomLoraConfig):
    lora_cfg.get_lora_config(model=model)
    target_modules = lora_cfg.target_modules
    
    
    list_of_linear_cands = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and any(name.endswith(t) for t in target_modules):
            list_of_linear_cands.append((name, module))

    for (name, module) in list_of_linear_cands:
        parent, attr = get_parent_and_attr(model, name)

        lora_linear_module = LoraLinear(base_linear=module, r=lora_cfg.r, 
                                    lora_alpha=lora_cfg.lora_alpha, lora_dropout=lora_cfg.lora_dropout,
                                    copy_weight=lora_cfg.copy_weight)
        lora_linear_module.reset_lora_parameters()
        # If parent is ModuleList, attr will be an index string
        if attr.isdigit() and isinstance(parent, nn.ModuleList):
            idx = int(attr)
            parent[idx] =lora_linear_module
        else:
            setattr(
                parent,
                attr,
                lora_linear_module
            )
    
    return model