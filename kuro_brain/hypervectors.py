"""
hypervectors.py — Bipolar High-Dimensional Vector primitives.

In KAHNN every concept is a D-dimensional bipolar hypervector (each component
in {-1, +1}). Two random hypervectors are near-orthogonal with overwhelming
probability once D >= 4096; this is the property that lets us symbolically
bind, bundle and permute without collision for hundreds of thousands of
symbols.

Operations are implemented on torch tensors so they run on GPU and batch
cleanly. Holographic binding (circular convolution) lives in holographic.py;
this module is the raw population-code substrate.
"""

from __future__ import annotations

import math
import torch
from torch import Tensor


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

def random_hypervector(
    dim: int,
    device: torch.device | str = "cpu",
    generator: torch.Generator | None = None,
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    """Sample a uniform random bipolar hypervector in {-1, +1}^D."""
    g = generator
    bits = torch.randint(0, 2, (dim,), device=device, generator=g)
    return bits.to(dtype).mul_(2).sub_(1).contiguous()


def random_hypervectors(
    n: int,
    dim: int,
    device: torch.device | str = "cpu",
    generator: torch.Generator | None = None,
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    """Sample a [n, dim] matrix of independent bipolar hypervectors."""
    g = generator
    bits = torch.randint(0, 2, (n, dim), device=device, generator=g)
    return bits.to(dtype).mul_(2).sub_(1).contiguous()


def level_hypervectors(
    n_levels: int,
    dim: int,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    """
    Build a chain of *level* hypervectors where adjacent levels are highly
    correlated and antipodal levels are negatively correlated. Useful for
    encoding continuous scalars (e.g. position, frequency) into HD space.
    """
    base = random_hypervector(dim, device=device, dtype=dtype)
    flips_per_step = max(1, dim // (2 * max(1, n_levels - 1)))
    out = torch.empty(n_levels, dim, device=device, dtype=dtype)
    cur = base.clone()
    out[0] = cur
    for i in range(1, n_levels):
        idx = torch.randperm(dim, device=device)[:flips_per_step]
        cur[idx] *= -1
        out[i] = cur.clone()
    return out.contiguous()


# ---------------------------------------------------------------------------
# Operations
# ---------------------------------------------------------------------------

def bundle(vectors: Tensor, dim: int = -1, keepdim: bool = False, threshold: bool = True) -> Tensor:
    """
    Majority-sum bundle. Sign of the elementwise sum; ties broken randomly.

    vectors: [..., D] tensor of bipolar (or real) hypervectors.
    Returns a bipolar hypervector along `dim`.
    """
    s = vectors.sum(dim=dim, keepdim=keepdim)
    if not threshold:
        return s
    out = torch.sign(s)
    # Break ties (s == 0) with random ±1
    zero_mask = (out == 0)
    if zero_mask.any():
        rand = torch.randint(0, 2, out.shape, device=out.device, dtype=out.dtype)
        out[zero_mask] = rand[zero_mask].mul_(2).sub_(1)
    return out


def normalize(v: Tensor, eps: float = 1e-5) -> Tensor:
    """L2-normalize a hypervector (or batch). Keeps values real in [-1, 1]."""
    return v / (v.norm(dim=-1, keepdim=True) + eps)


def similarity(a: Tensor, b: Tensor, dim: int = -1, eps: float = 1e-5) -> Tensor:
    """
    Cosine similarity between hypervectors. For bipolar vectors this is
    equivalent to (a·b)/D and ranges in [-1, 1].
    """
    a = a.to(torch.float32)
    b = b.to(torch.float32)
    return (a * b).sum(dim=dim) / (a.norm(dim=dim) * b.norm(dim=dim).clamp_min(eps) + eps)


def permute(v: Tensor, shifts: int = 1, dim: int = -1) -> Tensor:
    """
    Cyclic permutation of a hypervector — the classical HDC "role" encoder.
    Permute(x, k) is isometric: sim(perm(x,k), perm(y,k)) == sim(x, y).
    """
    return torch.roll(v, shifts=shifts, dims=dim)


def flip(v: Tensor, p: float = 0.5, generator: torch.Generator | None = None) -> Tensor:
    """Randomly flip a fraction p of components — used for noise injection."""
    g = generator
    mask = torch.rand(v.shape, device=v.device, generator=g) < p
    out = v.clone()
    out[mask] *= -1
    return out


# ---------------------------------------------------------------------------
# Convenience class wrapper (for stateful codebooks)
# ---------------------------------------------------------------------------

class BipolarHDV:
    """Tiny stateful container for an item memory of bipolar hypervectors."""

    def __init__(self, n: int, dim: int, device="cpu", seed: int = 0):
        g = torch.Generator(device=device).manual_seed(seed)
        self.dim = dim
        self.device = torch.device(device)
        self.matrix = random_hypervectors(n, dim, device=device, generator=g)

    def __len__(self) -> int:
        return self.matrix.shape[0]

    def __getitem__(self, idx) -> Tensor:
        return self.matrix[idx]

    def add(self, v: Tensor) -> int:
        """Append a new hypervector to the memory; returns its index."""
        assert v.shape[-1] == self.dim
        v = v.to(self.matrix.device).to(self.matrix.dtype).reshape(1, -1)
        self.matrix = torch.cat([self.matrix, v], dim=0)
        return self.matrix.shape[0] - 1

    def nearest(self, v: Tensor, topk: int = 1):
        """Return (indices, similarities) of the top-k nearest stored HVs."""
        sims = similarity(v.unsqueeze(0), self.matrix).squeeze(0)
        return sims.topk(topk)
