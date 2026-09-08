"""
device.py — Auto-select CPU / CUDA / MPS for universal training.

Prefer the fastest *available commodity* device without requiring big-GPU
clusters. Callers should treat the returned device as authoritative.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import torch


@dataclass
class DeviceInfo:
    device: torch.device
    name: str
    kind: str  # "cuda" | "mps" | "cpu"
    vram_gb: float | None
    cpu_threads: int
    amp_dtype: torch.dtype | None  # bf16/fp16 if safe, else None
    notes: str


def pick_device(requested: str = "auto") -> DeviceInfo:
    """
    Resolve training device.

    requested:
      - "auto": cuda → mps → cpu
      - "cpu" / "cuda" / "mps" / "cuda:N"
    """
    threads = int(os.environ.get("OMP_NUM_THREADS")
                  or os.environ.get("MKL_NUM_THREADS")
                  or max(1, (os.cpu_count() or 1)))
    torch.set_num_threads(threads)

    req = (requested or "auto").lower().strip()

    if req == "auto":
        if torch.cuda.is_available():
            req = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            req = "mps"
        else:
            req = "cpu"

    if req.startswith("cuda"):
        if not torch.cuda.is_available():
            return pick_device("cpu")
        device = torch.device(req if ":" in req else "cuda")
        idx = device.index or 0
        props = torch.cuda.get_device_properties(idx)
        vram = props.total_memory / (1024 ** 3)
        # bf16 is reliable on Ampere+ (major>=8); else prefer fp16 autocast
        major = props.major
        amp = torch.bfloat16 if major >= 8 else torch.float16
        # Very small GPUs: stay fp32 for stability
        if vram < 4.0:
            amp = None
        return DeviceInfo(
            device=device,
            name=props.name,
            kind="cuda",
            vram_gb=vram,
            cpu_threads=threads,
            amp_dtype=amp,
            notes=f"cuda sm_{major}{props.minor}, {vram:.1f} GiB",
        )

    if req == "mps":
        if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
            return pick_device("cpu")
        return DeviceInfo(
            device=torch.device("mps"),
            name="Apple MPS",
            kind="mps",
            vram_gb=None,
            cpu_threads=threads,
            amp_dtype=None,  # MPS autocast still uneven across ops (FFT)
            notes="MPS selected; FFT-heavy path stays fp32",
        )

    # CPU default
    return DeviceInfo(
        device=torch.device("cpu"),
        name="CPU",
        kind="cpu",
        vram_gb=None,
        cpu_threads=threads,
        amp_dtype=None,
        notes=f"torch.set_num_threads({threads})",
    )


def recommend_batch(cfg_name: str, info: DeviceInfo) -> tuple[int, int, int]:
    """
    Return (micro_batch, seq_len, grad_accum) heuristics for cost-optimal runs.
    Conservative — prefer fitting memory over peak throughput.
    """
    # defaults tuned from local CPU smoke measurements + VRAM rules of thumb
    table = {
        "smoke": (8, 64, 1),
        "nano": (4, 256, 2),
        "commodity": (2, 512, 4),
        "tiny": (2, 512, 4),
        "medium": (1, 512, 8),
        "large": (1, 256, 8),
        "b1": (1, 256, 16),
        "xl": (1, 128, 16),
    }
    micro, seq, accum = table.get(cfg_name, (2, 256, 4))
    if info.kind == "cpu":
        # CPU: smaller seq, more accum to keep steps meaningful without thrashing
        seq = min(seq, 256 if cfg_name in ("medium", "large", "b1", "xl") else seq)
        micro = min(micro, 2 if cfg_name not in ("smoke", "nano") else micro)
    elif info.kind == "cuda" and info.vram_gb is not None:
        if info.vram_gb < 8:
            micro, seq, accum = max(1, micro // 2), min(seq, 256), max(accum, 4)
        elif info.vram_gb < 12:
            seq = min(seq, 512)
        elif info.vram_gb >= 20 and cfg_name in ("commodity", "tiny", "medium"):
            micro = max(micro, 4)
            seq = max(seq, 1024)
    return micro, seq, accum
