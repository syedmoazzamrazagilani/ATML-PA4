from __future__ import annotations

import gc
import time
from typing import Any, Callable, Dict, List

import numpy as np
import torch


def clear_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def measure_cuda_fn(
    fn: Callable[[], Any],
    warmup: int = 3,
    reps: int = 5,
    reset_peak_per_rep: bool = False,
) -> Dict[str, Any]:
    """Run warmed-up, CUDA-synchronized timing and memory measurements."""
    is_cuda = torch.cuda.is_available()

    for _ in range(warmup):
        _ = fn()
        if is_cuda:
            torch.cuda.synchronize()

    if is_cuda:
        torch.cuda.reset_peak_memory_stats()

    times_ms: List[float] = []
    peak_allocs: List[int] = []
    peak_reserveds: List[int] = []

    for _ in range(reps):
        if is_cuda:
            if reset_peak_per_rep:
                torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            start_evt = torch.cuda.Event(enable_timing=True)
            end_evt = torch.cuda.Event(enable_timing=True)
            start_evt.record()
            _ = fn()
            end_evt.record()
            torch.cuda.synchronize()
            elapsed_ms = float(start_evt.elapsed_time(end_evt))
            peak_allocs.append(int(torch.cuda.max_memory_allocated()))
            peak_reserveds.append(int(torch.cuda.max_memory_reserved()))
        else:
            t0 = time.perf_counter()
            _ = fn()
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            peak_allocs.append(0)
            peak_reserveds.append(0)
        times_ms.append(elapsed_ms)

    arr = np.asarray(times_ms, dtype=np.float64)
    return {
        "unit": "ms",
        "warmup": warmup,
        "reps": reps,
        "mean_ms": float(arr.mean()),
        "std_ms": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
        "median_ms": float(np.median(arr)),
        "min_ms": float(arr.min()),
        "max_ms": float(arr.max()),
        "samples_ms": [float(x) for x in times_ms],
        "peak_allocated_bytes": int(max(peak_allocs)) if peak_allocs else 0,
        "peak_reserved_bytes": int(max(peak_reserveds)) if peak_reserveds else 0,
    }
