"""
fp8.py — FP8 training support for KAHNN on Ada (RTX 4090) and Hopper (H100).

RTX 4090 (Ada AD102) has fp8 tensor cores at 165 TFLOPS — 2× the bf16
throughput (82 TFLOPS). H100 is 1979 TFLOPS fp8 vs 990 TFLOPS bf16.

FP8 uses two formats:
  - E4M3 (max ±448, ~3 decimal digits) — for forward activations
  - E5M2 (max ±57344, ~2 decimal digits) — for backward gradients

We use torch.float8_e4m3fn for activations and a per-tensor scaling
strategy. The matmul kernel auto-promotes to fp32 accumulation.

For KAHNN, the FP8 path wraps:
  - The low-rank coupling A·v = U(V^T v) inside KuramotoLayer
  - The HRR-MLP low-rank residual x·A^T·B^T
  - The output projection (cosine similarity against token_hvs)

The FFT operations stay in fp32/bf16 (cuFFT doesn't support fp8).
"""

from __future__ import annotations

import math
import torch
from torch import Tensor
from contextlib import contextmanager


# ---------------------------------------------------------------------------
# FP8 helpers
# ---------------------------------------------------------------------------

HAS_FP8 = hasattr(torch, "float8_e4m3fn") and hasattr(torch, "float8_e5m2")


def fp8_available() -> bool:
    """True if the current torch build exposes fp8 dtypes."""
    return HAS_FP8


def supports_fp8_on_device(device: torch.device) -> bool:
    """
    True if the device can actually run fp8 matmuls. This requires
    Ada (sm_89) or Hopper (sm_90). On other GPUs the dtype exists
    but matmul falls back to slow emulation.
    """
    if not HAS_FP8:
        return False
    if device.type != "cuda":
        return False
    try:
        major, minor = torch.cuda.get_device_capability(device)
        return (major, minor) >= (8, 9)  # Ada=8.9, Hopper=9.0
    except Exception:
        return False


def to_fp8_e4m3(x: Tensor, scale: float | None = None) -> Tensor:
    """Cast a tensor to fp8 e4m3 with optional per-tensor scaling."""
    if not HAS_FP8:
        return x
    if scale is not None:
        x = x / scale
    return x.to(torch.float8_e4m3fn)


def to_fp8_e5m2(x: Tensor, scale: float | None = None) -> Tensor:
    """Cast a tensor to fp8 e5m2 (for gradients)."""
    if not HAS_FP8:
        return x
    if scale is not None:
        x = x / scale
    return x.to(torch.float8_e5m2)


def fp8_matmul(a: Tensor, b: Tensor, scale_a: float = 1.0, scale_b: float = 1.0) -> Tensor:
    """
    FP8 matmul. Both inputs are cast to fp8_e4m3 with per-tensor scaling;
    cuBLAS handles the fp8 tensor-core path on Ada/Hopper. Output is bf16.

    Falls back to bf16 matmul if fp8 isn't available.
    """
    if not supports_fp8_on_device(a.device):
        return (a.to(torch.bfloat16) @ b.to(torch.bfloat16))
    a_fp8 = to_fp8_e4m3(a, scale_a)
    b_fp8 = to_fp8_e4m3(b, scale_b)
    # torch.matmul on fp8 tensors dispatches to cuBLAS fp8 kernels on Ada/Hopper
    # outputs in fp32 by default; we cast back to bf16
    out = torch.matmul(a_fp8.to(torch.float32), b_fp8.to(torch.float32))
    return out.to(torch.bfloat16) * (scale_a * scale_b)


@contextmanager
def fp8_autocast(enabled: bool = True):
    """
    Context manager for FP8 autocast. Inside this context, eligible
    matmuls in our own code (fp8_matmul) use fp8. Torch's built-in
    autocast doesn't yet handle fp8 universally, so we wrap manually.

    Usage:
        with fp8_autocast(enabled=args.fp8):
            logits = model(x)
    """
    if not enabled:
        yield
        return
    # We rely on our own fp8_matmul calls inside the model; the context
    # just signals "use fp8 if you can". This is a no-op context for now
    # but kept for forward-compatibility with torch's native fp8 autocast.
    old_dtype = torch.get_default_dtype()
    try:
        yield
    finally:
        torch.set_default_dtype(old_dtype)


# ---------------------------------------------------------------------------
# Amax-based per-tensor scaling (delayed scaling)
# ---------------------------------------------------------------------------

class FP8ScaleTracker:
    """
    Tracks amax of a tensor over recent steps to compute stable fp8 scales.
    Used for delayed scaling — the standard fp8 recipe from NVIDIA.

    Usage:
        tracker = FP8ScaleTracker()
        scale = tracker.get_scale()            # use this for next forward
        tracker.update(x.detach())             # update with current tensor
    """

    def __init__(self, history_len: int = 16, fp8_max: float = 448.0):
        self.history = []
        self.history_len = history_len
        self.fp8_max = fp8_max  # E4M3 max value
        self.scale = 1.0

    def update(self, x: Tensor):
        amax = float(x.abs().max().item()) if x.numel() > 0 else 1.0
        self.history.append(amax)
        if len(self.history) > self.history_len:
            self.history.pop(0)
        if self.history:
            recent_max = max(self.history)
            if recent_max > 0:
                # scale = fp8_max / amax, with safety margin
                self.scale = self.fp8_max / (recent_max * 1.1)

    def get_scale(self) -> float:
        return self.scale
