from __future__ import annotations

from functools import lru_cache

try:
    import torch
except Exception:
    torch = None


def gpu_available() -> bool:
    try:
        return torch is not None and torch.cuda.is_available()
    except Exception:
        return False


def gpu_name() -> str:
    try:
        if gpu_available():
            return torch.cuda.get_device_name(0)
    except Exception:
        pass
    return "CPU"


def gpu_memory_total_gb() -> float:
    try:
        if gpu_available():
            total = torch.cuda.get_device_properties(0).total_memory
            return round(total / 1024**3, 2)
    except Exception:
        pass
    return 0.0


def gpu_memory_allocated_gb() -> float:
    try:
        if gpu_available():
            used = torch.cuda.memory_allocated(0)
            return round(used / 1024**3, 2)
    except Exception:
        pass
    return 0.0


@lru_cache(maxsize=1)
def cuda_summary() -> str:
    if not gpu_available():
        return "CUDA not available"
    return f"{gpu_name()} | {gpu_memory_total_gb()} GB VRAM"