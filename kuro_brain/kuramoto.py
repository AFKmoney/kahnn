"""
kuramoto.py — Kuramoto oscillator layer + attractor memory.

The core "continuous thoughts" engine. Each hypervector component i is an
oscillator with phase θᵢ ∈ [-π, π] and natural frequency ωᵢ. The layer
evolves all D oscillators under the Kuramoto dynamics:

    dθᵢ/dt = ωᵢ + (K/N) Σⱼ Aᵢⱼ sin(θⱼ − θᵢ) + ξᵢ(t)

where Aᵢⱼ is a learned coupling matrix (compact, low-rank to fit D=8k+).

Key neuroscience correspondences:
    ωᵢ                  → intrinsic firing rate of neuron i
    Aᵢⱼ                  → synaptic efficacy (signed)
    K                    → global neuromodulator / arousal
    phase locking        → cortical assembly binding
    attractor basin      → persistent activity memory (e.g. prefrontal)

When the layer is read out we project phases back to a real-valued
hypervector via `phase_to_hv`. Holographic binding of two such vectors is
exactly *phase addition* of their spectra, so the Kuramoto coupling term
Aᵢⱼ sin(θⱼ−θᵢ) is, in Fourier space, a *local* holographic unbinding —
which is why we never need self-attention.

Attractor memory: a separate set of "engram" phase vectors whose coupling
to the working state is reinforced by an online STDP-like rule. The
working state is pulled toward the nearest engram if their coherence
exceeds a threshold — this is the brain's "remembering" operation, done
without any gradient or replay.
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .holographic import hv_to_phase, phase_to_hv


# ---------------------------------------------------------------------------
# Kuramoto layer
# ---------------------------------------------------------------------------

class KuramotoLayer(nn.Module):
    """
    A bank of D Kuramoto oscillators with a learned low-rank coupling.

    Parameters
    ----------
    dim : int
        Hypervector dimensionality D (oscillator count).
    steps : int
        Number of discrete integration steps per forward pass.
    dt : float
        Integration time-step.
    rank : int
        Rank of the learned coupling matrix A. Full D×D is infeasible for
        D ≥ 8k; rank=128 typically captures enough structure for one layer.
    n_ensembles : int
        Number of *attractor ensembles* (memory slots) this layer holds.
    noise : float
        Std of Gaussian phase noise ξ per step. Models ongoing neural
        variability; also helps escape shallow basins.
    """

    def __init__(
        self,
        dim: int,
        steps: int = 4,
        dt: float = 0.05,
        rank: int = 128,
        n_ensembles: int = 32,
        noise: float = 0.01,
        device: torch.device | str = "cpu",
    ):
        super().__init__()
        self.dim = dim
        self.steps = steps
        self.dt = dt
        self.rank = rank
        self.n_ensembles = n_ensembles
        self.noise = noise

        # Natural frequencies (per-oscillator). Initialize small so phases
        # drift slowly; learnable so different oscillators can specialise.
        self.omega = nn.Parameter(torch.randn(dim) * 0.1)

        # Global coupling gain K (per layer). Acts like a neuromodulator.
        self.K = nn.Parameter(torch.tensor(1.0))

        # Low-rank coupling A = U @ V^T, with U,V in R^{D×rank}.
        # Initialise as identity-ish (small random) so the layer is
        # initially close to a passive oscillator bank.
        self.U = nn.Parameter(torch.randn(dim, rank) * (1.0 / math.sqrt(rank)))
        self.V = nn.Parameter(torch.randn(dim, rank) * (1.0 / math.sqrt(rank)))

        # Attractor ensembles — fixed-size phase memory.
        # Each row is the Fourier-space phase signature of one engram.
        self.register_parameter(
            "engrams",
            nn.Parameter(torch.randn(n_ensembles, dim) * 0.1),
        )
        # Engram gating gain (how strongly the memory pulls on the state).
        self.engram_gain = nn.Parameter(torch.tensor(0.5))

    # ------------------------------------------------------------------
    # Coupling
    # ------------------------------------------------------------------

    def coupling_matrix(self) -> Tensor:
        """Return the (D, D) coupling matrix A = U @ V^T."""
        return self.U @ self.V.t()

    def low_rank_apply(self, sin_dtheta: Tensor) -> Tensor:
        """
        Apply A·v efficiently via low-rank factors: A v = U (V^T v).
        `sin_dtheta`: [..., D] tensor.
        Returns [..., D].
        """
        tmp = torch.einsum("...d,dr->...r", sin_dtheta, self.V)
        return torch.einsum("dr,...r->...d", self.U, tmp)

    # ------------------------------------------------------------------
    # Forward dynamics
    # ------------------------------------------------------------------

    def forward(
        self,
        x: Tensor,
        return_states: bool = False,
        mask: Tensor | None = None,
    ) -> Tensor:
        """
        Run `steps` Kuramoto updates starting from the phase of x.

        x: [..., D] real-valued hypervector (input state).
        Returns: [..., D] real-valued hypervector (evolved state).

        If return_states=True, also returns a [steps, ..., D] tensor of
        intermediate states — useful for the online learning rule.
        """
        # Initial phases from Fourier spectrum of x
        theta = hv_to_phase(x)
        states = []
        for _ in range(self.steps):
            theta = self._step(theta, mask=mask)
            if return_states:
                states.append(phase_to_hv(theta))

        out = phase_to_hv(theta)
        if return_states:
            return out, torch.stack(states, dim=0)
        return out

    def _step(self, theta: Tensor, mask: Tensor | None = None) -> Tensor:
        """One Kuramoto integration step."""
        # 1) Pairwise coupling via low-rank A and sin(θ_j - θ_i).
        # We never materialize the D×D matrix; instead exploit
        #   A sin(θ_j - θ_i) summed over j  ==  A (sin θ cos θ - cos θ sin θ)
        # Use sin(θ_j - θ_i) = sin θ_j cos θ_i - cos θ_j sin θ_i, so
        #   Σ_j A_ij sin(θ_j - θ_i) = (A sin θ)_i cos θ_i - (A cos θ)_i sin θ_i
        sin_t = torch.sin(theta)
        cos_t = torch.cos(theta)
        A_sin = self.low_rank_apply(sin_t)   # [..., D]
        A_cos = self.low_rank_apply(cos_t)
        coupling = A_sin * cos_t - A_cos * sin_t

        # 2) Attractor pull from engrams (memory). The engram contribution
        # is computed in Fourier-phase space as a soft mixture of engram
        # phases weighted by coherence.
        if self.n_ensembles > 0:
            engram_pull = self._engram_pull(theta)
            coupling = coupling + self.engram_gain * engram_pull

        # 3) Update
        omega = self.omega
        # Use 1/sqrt(D) scaling (proper for low-rank coupling) instead of 1/D
        # — keeps gradient magnitudes healthy as D grows.
        dtheta = omega + (self.K / math.sqrt(max(1.0, self.dim))) * coupling
        if self.noise > 0:
            dtheta = dtheta + self.noise * torch.randn_like(theta)

        if mask is not None:
            dtheta = dtheta * mask.unsqueeze(-1)

        new_theta = theta + self.dt * dtheta
        # Wrap to [-π, π]
        new_theta = torch.remainder(new_theta + math.pi, 2 * math.pi) - math.pi
        return new_theta

    def _engram_pull(self, theta: Tensor) -> Tensor:
        """
        Compute the attractor pull on the current phase state from all
        engrams. Pull is sin(θ_engram - θ) weighted by coherence c_k =
        cos(mean_i(θ_engram_k - θ_i)) — a global phase-coherence score
        between the state and engram k.
        """
        # theta: [..., D], engrams: [E, D]
        # diff_k = θ_engram_k - θ_i        -> broadcast [..., E, D]
        # sin(diff_k) averaged over D gives mean pull direction
        # cos(diff_k) averaged over D gives coherence (scalar per engram)
        diff = self.engrams.unsqueeze(0) - theta.unsqueeze(-2)  # [..., E, D]
        sin_d = torch.sin(diff).mean(dim=-1)                    # [..., E]
        cos_d = torch.cos(diff).mean(dim=-1)                    # [..., E]
        # Softmax over engrams by coherence (temperature 1/π)
        weights = F.softmax(cos_d / math.pi, dim=-1)            # [..., E]
        # Weighted pull: Σ_k w_k * sin(θ_engram_k - θ_i) along D
        pull = (weights.unsqueeze(-1) * torch.sin(diff)).sum(dim=-2)  # [..., D]
        return pull

    # ------------------------------------------------------------------
    # Engram management — for continuous learning
    # ------------------------------------------------------------------

    @torch.no_grad()
    def write_engram(self, theta: Tensor, slot: int | None = None) -> int:
        """
        Persist a phase state `theta` (shape [D] or [B, D]) into an engram
        slot. If slot is None, pick the least-coherent slot (overwrite
        weakest memory) — a simple active forgetting policy.
        Returns the slot index used.
        """
        if theta.dim() == 2:
            theta = theta.mean(dim=0)
        assert theta.shape[-1] == self.dim
        if slot is None:
            # compute coherence with each engram
            diff = self.engrams.data - theta.unsqueeze(0)
            coh = torch.cos(diff).mean(dim=-1)
            slot = int(coh.argmin().item())
        self.engrams.data[slot] = 0.9 * self.engrams.data[slot] + 0.1 * theta
        return slot


# ---------------------------------------------------------------------------
# Attractor memory (cross-layer)
# ---------------------------------------------------------------------------

class AttractorMemory(nn.Module):
    """
    A cross-layer attractor memory that records phase signatures from
    multiple Kuramoto layers and pulls each layer's state toward the
    best-matching attractor. This is the system-level "hippocampal" memory
    that binds together states across cortical layers.
    """

    def __init__(self, n_layers: int, dim: int, capacity: int = 256):
        super().__init__()
        self.n_layers = n_layers
        self.dim = dim
        self.capacity = capacity
        # [n_layers, capacity, dim] phase signatures
        self.register_buffer(
            "attractors",
            torch.zeros(n_layers, capacity, dim),
        )
        self.register_buffer("usage", torch.zeros(n_layers, capacity))

    @torch.no_grad()
    def write(self, layer_idx: int, theta: Tensor):
        """Write a phase state into the least-used slot of layer `layer_idx`."""
        if theta.dim() == 2:
            theta = theta.mean(dim=0)
        slot = int(self.usage[layer_idx].argmin().item())
        self.attractors[layer_idx, slot] = theta
        self.usage[layer_idx] += 1e-3  # decay all slots slightly
        self.usage[layer_idx, slot] += 1.0

    @torch.no_grad()
    def pull(self, layer_idx: int, theta: Tensor, gain: float = 0.1) -> Tensor:
        """
        Soft pull `theta` toward the best-matching attractor for layer
        `layer_idx`. Returns a phase correction term to be added to dtheta.
        """
        # theta: [..., D]
        diff = self.attractors[layer_idx].unsqueeze(0) - theta.unsqueeze(-2)
        sin_d = torch.sin(diff).mean(dim=-1)        # [..., capacity]
        cos_d = torch.cos(diff).mean(dim=-1)        # [..., capacity]
        w = F.softmax(cos_d / math.pi, dim=-1)      # [..., capacity]
        pull = (w.unsqueeze(-1) * torch.sin(diff)).sum(dim=-2)  # [..., D]
        return gain * pull

    @torch.no_grad()
    def decay(self, rate: float = 0.999):
        """Exponential forgetting of attractor strengths."""
        self.usage.mul_(rate)
