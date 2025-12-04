import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP

def prepare_model_for_ddp(model: nn.Module, local_rank: int, enable_grad_ckpt: bool, gradient_checkpointing_kwargs: dict, find_unused_parameters: bool):
    """
    Moves model to device, enables gradient checkpointing with specific kwargs,
    and wraps in DDP.
    """
    torch.cuda.set_device(local_rank)
    model = model.to(local_rank)

    if enable_grad_ckpt:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs=gradient_checkpointing_kwargs
        )
    
    model = DDP(
        model, 
        device_ids=[local_rank], 
        output_device=local_rank,
        find_unused_parameters=find_unused_parameters
    )
    
    return model