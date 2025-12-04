import os, torch
import torch.distributed as dist
from types import SimpleNamespace
from contextlib import contextmanager
try:
    import pynvml
    pynvml.nvmlInit()
    NVML_AVAILABLE = True
except pynvml.NVMLError:
    print("Warning: NVML not initialized. Pynvml metrics will be unavailable.")
    NVML_AVAILABLE = False
    

@contextmanager
def measure_vram(label: str = "vram", device: int = 0, printout: bool = False):
    """
    Tracks PyTorch and NVML memory usage.
    Yields a stats object. After the block exits, stats.metrics will contain
    a dictionary suitable for wandb logging.
    """
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    
    # 1. Snapshot Start Stats
    pt_start_alloc = torch.cuda.memory_allocated(device)
    pt_start_reserved = torch.cuda.memory_reserved(device)
    
    nvml_start = 0
    if NVML_AVAILABLE:
        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(device)
            info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            nvml_start = info.used
        except Exception:
            pass

    # Create the container we yield to the user
    stats = SimpleNamespace(metrics={})
    
    yield stats
    
    torch.cuda.synchronize(device)
    
    # 2. Snapshot End Stats
    pt_end_alloc = torch.cuda.memory_allocated(device)
    pt_peak_alloc = torch.cuda.max_memory_allocated(device)
    pt_end_reserved = torch.cuda.memory_reserved(device)
    pt_peak_reserved = torch.cuda.max_memory_reserved(device)
    
    nvml_end = 0
    if NVML_AVAILABLE:
        try:
            info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            nvml_end = info.used
        except Exception:
            pass

    # 3. Calculate Deltas and Conversions
    mb = 1024 ** 2
    
    # PyTorch Internal
    alloc_delta = (pt_end_alloc - pt_start_alloc) / mb
    reserved_delta = (pt_end_reserved - pt_start_reserved) / mb
    peak_alloc_mb = pt_peak_alloc / mb
    peak_reserved_mb = pt_peak_reserved / mb
    
    # NVML (Hardware)
    nvml_delta = (nvml_end - nvml_start) / mb
    
    # Fragmentation (Allocated vs Reserved at the end)
    # If reserved is 0, fragmentation is technically 0
    if pt_end_reserved > 0:
        frag_pct = 100 * (1 - pt_end_alloc / pt_end_reserved)
    else:
        frag_pct = 0.0

    # 4. expose attributes for direct access (e.g. stats.peak_mb)
    stats.peak_mb = peak_alloc_mb
    stats.allocated_delta_mb = alloc_delta
    stats.reserved_delta_mb = reserved_delta
    stats.peak_reserved_mb = peak_reserved_mb
    stats.nvml_total_mb = nvml_end / mb

    # 5. Populate the metrics dictionary for WandB
    # We split metrics into two sections:
    #   - "{label}/..." for PyTorch specific logical memory
    #   - "System/..." for Hardware level memory (shared by all processes)
    
    pt_prefix = f"{label}/" if label else "vram/"
    
    metrics = {
        f"{pt_prefix}allocated_delta_mb": alloc_delta,
        f"{pt_prefix}peak_allocated_mb": peak_alloc_mb,
        f"{pt_prefix}reserved_delta_mb": reserved_delta,
        f"{pt_prefix}peak_reserved_mb": peak_reserved_mb,
        f"{pt_prefix}fragmentation_pct": frag_pct,
    }

    if NVML_AVAILABLE:
        # Puts hardware stats in their own "System" section in WandB
        metrics.update({
            "System/GPU_Memory_Used_MB": nvml_end / mb,
            "System/GPU_Memory_Delta_MB": nvml_delta,
        })
        
    stats.metrics = metrics

    # # 6. Print Summary
    if printout:
        print(f"--- Memory Trace: {label} ---")
        print(f"PyTorch Allocated: {pt_start_alloc/mb:.2f}MB -> {pt_end_alloc/mb:.2f}MB (Delta: {alloc_delta:+.2f}MB, Peak: {peak_alloc_mb:.2f}MB)")
        print(f"PyTorch Reserved:  {pt_start_reserved/mb:.2f}MB -> {pt_end_reserved/mb:.2f}MB (Delta: {reserved_delta:+.2f}MB)")
        print(f"NVML (Actual):     {nvml_start/mb:.2f}MB -> {nvml_end/mb:.2f}MB (Delta: {nvml_delta:+.2f}MB)")
        print(f"Fragmentation:     {frag_pct:.2f}% of reserved is unused")

def per_device_report(tag=""):
    dev = torch.cuda.current_device()
    alloc = torch.cuda.memory_allocated(dev) / 2**20
    reserv = torch.cuda.memory_reserved(dev) / 2**20
    maxalloc = torch.cuda.max_memory_allocated(dev) / 2**20
    if dist.is_initialized():
        rank = dist.get_rank()
        world = dist.get_world_size()
        return (f"[{tag}] rank {rank}/{world-1} device cuda:{dev} → "
                f"allocated={alloc:8.1f} MB  reserved={reserv:8.1f} MB  max_alloc={maxalloc:8.1f} MB\n")
    else:
        return (f"[{tag}] device cuda:{dev} → allocated={alloc:8.1f} MB  reserved={reserv:8.1f} MB  max_alloc={maxalloc:8.1f} MB\n")
