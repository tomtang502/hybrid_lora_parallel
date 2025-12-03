import torch
import functools
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    ShardingStrategy,
    BackwardPrefetch,
    StateDictType,
)
from torch.distributed.fsdp.wrap import (
    transformer_auto_wrap_policy,
    size_based_auto_wrap_policy,
)
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    apply_activation_checkpointing,
    checkpoint_wrapper,
    CheckpointImpl,
)

def prepare_model_for_fsdp(
    model: torch.nn.Module,
    local_rank: int,
    block_class=None,
    enable_grad_ckpt: bool = True,
    gradient_checkpointing_kwargs: dict ={"use_reentrant": False},
    mixed_precision: bool = True,
    use_cpu_offload: bool = False,
):
    """
    Wraps a model in FSDP1 with Mixed Precision, Auto-Wrapping, and Activation Checkpointing.
    
    Args:
        model: The PyTorch model.
        local_rank: Current GPU device ID.
        block_class: The Transformer Layer class (e.g. LlamaDecoderLayer, GPTBlock). 
                     CRITICAL for memory savings.
        enable_grad_ckpt: Whether to enable gradient checkpointing.
    """
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    else:
        # Manual fallback if method doesn't exist
        def make_inputs_require_grad(module, input, output):
            output.requires_grad_(True)
        model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)
        torch.cuda.set_device(local_rank)

    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs=gradient_checkpointing_kwargs
    )

    mp_policy = None
    if mixed_precision:
        bf16_ready = torch.cuda.is_bf16_supported()
        dtype = torch.bfloat16 if bf16_ready else torch.float16
        mp_policy = MixedPrecision(
            param_dtype=dtype,
            reduce_dtype=dtype, 
            buffer_dtype=dtype,
        )

    
    if block_class is not None:
        # Wrap every Transformer Block individually
        wrap_policy = functools.partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls={block_class},
        )
    else:
        # Fallback: Wrap any layer larger than 100M params
        print("Warning: No block_class provided for FSDP. Using size-based wrapping (less efficient).")
        wrap_policy = functools.partial(
            size_based_auto_wrap_policy, min_num_params=100_000_000
        )

    model = FSDP(
        model,
        auto_wrap_policy=wrap_policy,
        mixed_precision=mp_policy,
        device_id=local_rank,
        sharding_strategy=ShardingStrategy.FULL_SHARD, 
        backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
        forward_prefetch=True,
        limit_all_gathers=True,
        use_orig_params=True, 
    )

    # if enable_grad_ckpt and block_class is not None:
    #     non_reentrant_wrapper = functools.partial(
    #         checkpoint_wrapper,
    #         checkpoint_impl=CheckpointImpl.NO_REENTRANT,
    #     )
        
    #     # Apply checkpointing only to the Transformer Blocks we identified
    #     check_fn = lambda submodule: isinstance(submodule, block_class)
        
    #     apply_activation_checkpointing(
    #         model, 
    #         checkpoint_wrapper_fn=non_reentrant_wrapper, 
    #         check_fn=check_fn
    #     )

    return model

def get_block_class_from_model(model):
    # 1. Get the string name (e.g., "LlamaDecoderLayer")
    if not hasattr(model, "_no_split_modules") or not model._no_split_modules:
        raise ValueError("Model does not have _no_split_modules defined.")
        
    block_name = model._no_split_modules[0] # This is a STRING
    
    # 2. Search the model to find the actual CLASS object matching that name
    for module in model.modules():
        if module.__class__.__name__ == block_name:
            return module.__class__ # This is the CLASS (Type)
            
    raise ValueError(f"Could not find module class matching name: {block_name}")