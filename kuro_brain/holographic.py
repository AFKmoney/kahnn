"""
holographic.py — Holographic Reduced Representations (Plate, 1995).

HRR gives us *symbolic* compositionality in fixed-width vectors:
    bind(a, b)   ≈ circular convolution a ⊛ b      — superpositional binding
    unbind(b, x) ≈ circular correlation a ⊛⁻¹ x     — approximate unbinding
    bundle(...)  ≈ elementwise sum / majority        — set-like superposition

Crucially, bind() and unbind() are O(D log D) via FFT, and the *same*
operation can be implemented with phase codes (each hypervector is the
inverse-FFT of a phase vector). We use that equivalence in kuramoto.py to
turn oscillators into binding operators.

For bipolar hypervectors we additionally support a discrete binding that
preserves bipolarity via XNOR — useful for the item memory but not for
compositional roles. The default is continuous HRR.
"""

from __future__ import annotations

import math
import torch
from torch import Tensor

from .hypervectors import normalize, similarity


# ---------------------------------------------------------------------------
# Continuous HRR (real-valued, FFT-based)
# ---------------------------------------------------------------------------

def _fft(x: Tensor, n: int = 0) -> Tensor:
    if n and n > 0:
        return torch.fft.fft(x, n=n, dim=-1)
    return torch.fft.fft(x, dim=-1)


def _ifft(x: Tensor, n: int = 0) -> Tensor:
    if n and n > 0:
        return torch.fft.ifft(x, n=n, dim=-1).real
    return torch.fft.ifft(x, dim=-1).real


def hrr_bind(a: Tensor, b: Tensor) -> Tensor:
    """
    Circular convolution a ⊛ b via FFT. O(D log D).

    Result is real-valued; we renormalize so ||bind|| ≈ ||a||.
    """
    A = _fft(a)
    B = _fft(b)
    C = A * B
    out = _ifft(C)
    # Scale to preserve magnitude
    return out / math.sqrt(a.shape[-1])


def hrr_unbind(x: Tensor, key: Tensor) -> Tensor:
    """
    Approximate inverse: circular correlation with the (conjugate) key.
    For unit-norm real keys, conjugate ≈ key itself, but the proper inverse
    uses 1/conj(FFT(key)).
    """
    X = _fft(x)
    K = _fft(key)
    # regularized inverse to avoid div-by-zero
    eps = 1e-3
    Kinv = K.conj() / (K.real ** 2 + K.imag ** 2 + eps)
    out = _ifft(X * Kinv)
    return out / math.sqrt(x.shape[-1])


def hrr_bundle(vectors: Tensor, dim: int = -1, keepdim: bool = False) -> Tensor:
    """
    Superpose a stack of hypervectors via mean. Mean (vs sum) keeps the
    magnitude stable regardless of how many items are bundled, which matters
    for unbinding fidelity in deep stacks.
    """
    return vectors.mean(dim=dim, keepdim=keepdim)


def hrr_similarity(a: Tensor, b: Tensor) -> Tensor:
    return similarity(a, b)


# ---------------------------------------------------------------------------
# Bipolar (XNOR) binding — popcount-free, GPU-friendly
# ---------------------------------------------------------------------------

def xnor_bind(a: Tensor, b: Tensor) -> Tensor:
    """
    Bipolar binding via XNOR (elementwise product). Strictly distance-
    preserving but less expressive than HRR for compositional structure.
    Used inside the Kuramoto attractor memory where we want strict isometry.
    """
    return (a * b).sign()


def xnor_unbind(x: Tensor, key: Tensor) -> Tensor:
    # XNOR is its own inverse
    return (x * key).sign()


# ---------------------------------------------------------------------------
# Phase-code utilities (bridge to Kuramoto oscillators)
# ---------------------------------------------------------------------------

def hv_to_phase(v: Tensor, eps: float = 1e-5) -> Tensor:
    """
    Treat a real hypervector as the inverse-FFT of a unit-magnitude phase
    spectrum. Return the phase angles θ ∈ [-π, π] such that
        v ≈ Re(IFFT(|FFT(v)| · exp(i θ)))
    i.e. the per-frequency phase of v.

    This is the bridge between holographic binding (multiplication in
    Fourier domain) and Kuramoto oscillators (phase dynamics).
    """
    V = torch.fft.fft(v, dim=-1)
    return torch.angle(V)


def phase_to_hv(theta: Tensor) -> Tensor:
    """
    Inverse of hv_to_phase: rebuild a real hypervector from per-frequency
    phases using unit magnitude. Loses magnitude info but preserves the
    binding algebra (since HRR is multiplicative in phase).
    """
    Z = torch.exp(1j * theta)
    return torch.fft.ifft(Z, dim=-1).real


def hrr_bind_phase(theta_a: Tensor, theta_b: Tensor) -> Tensor:
    """
    HRR binding expressed purely in the phase domain:
        bind = IFFT(|FFT(a)|·|FFT(b)| · exp(i(θ_a + θ_b)))
    With unit-magnitude approx: bind ≈ IFFT(exp(i(θ_a + θ_b))).
    This is *addition in phase space* — exactly the algebra Kuramoto
    oscillator coupling induces when phases synchronize.
    """
    return theta_a + theta_b


def hrr_unbind_phase(theta_x: Tensor, theta_key: Tensor) -> Tensor:
    return theta_x - theta_key
