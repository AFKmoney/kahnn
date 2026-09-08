"""
kuramoto.py — Kuramoto oscillator layer + attractor memory.

The core "continuous thoughts" engine. Each hypervector component i is an
oscillator with phase θᵢ ∈ [-π, π] and natural frequency ωᵢ. The layer
evolves all D oscillators under the Kuramoto dynamics:

    dθᵢ/dt = ωᵢ + (K/N) Σⱼ Aᵢⱼ sin(θⱼ − θᵢ) + ξᵢ(t)

where Aᵢⱼ is a learned coupling matrix (compact, low-rank to fit D=8k+).
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .holographic import hv_to_phase, phase_to_hv


class KuramotoLayer(nn.Module):
    """A bank of D Kuramoto oscillators with a learned low-rank coupling."""

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

        self.omega = nn.Parameter(torch.randn(dim) * 0.1)
        self.K = nn.Parameter(torch.tensor(1.0))
        self.U = nn.Parameter(torch.randn(dim, rank) * (1.0 / math.sqrt(rank)))
        self.V = nn.Parameter(torch.randn(dim, rank) * (1.0 / math.sqrt(rank)))
        self.register_parameter(
            "engrams",
            nn.Parameter(torch.randn(n_ensembles, dim) * 0.1),
        )
        self.engram_gain = nn.Parameter(torch.tensor(0.5))

    def coupling_matrix(self) -> Tensor:
        return self.U @ self.V.t()

    def low_rank_apply(self, sin_dtheta: Tensor) -> Tensor:
        tmp = torch.einsum("...d,dr->...r", sin_dtheta, self.V)
        return torch.einsum("dr,...r->...d", self.U, tmp)

    def forward(
        self,
        x: Tensor,
        return_states: bool = False,
        mask: Tensor | None = None,
    ) -> Tensor:
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
        sin_t = torch.sin(theta)
        cos_t = torch.cos(theta)
        A_sin = self.low_rank_apply(sin_t)
        A_cos = self.low_rank_apply(cos_t)
        coupling = A_sin * cos_t - A_cos * sin_t

        if self.n_ensembles > 0:
            engram_pull = self._engram_pull(theta)
            coupling = coupling + self.engram_gain * engram_pull

        omega = self.omega
        dtheta = omega + (self.K / math.sqrt(max(1.0, self.dim))) * coupling
        if self.noise > 0:
            dtheta = dtheta + self.noise * torch.randn_like(theta)

        if mask is not None:
            dtheta = dtheta * mask.unsqueeze(-1)

        new_theta = theta + self.dt * dtheta
        new_theta = torch.remainder(new_theta + math.pi, 2 * math.pi) - math.pi
        return new_theta

    def _engram_pull(self, theta: Tensor) -> Tensor:
        """
        Attractor pull without materializing [..., E, D] (which for B1 at
        B=2,T=2048,E=64,D=24576 is ~50+ GB). Uses trig identities:
            mean_d cos(e-t) = (cos_e·cos_t + sin_e·sin_t)/D
            sum_k w_k sin(e_k-t) = (w·sin_e)·cos_t - (w·cos_e)·sin_t
        """
        D = self.dim
        sin_t = torch.sin(theta)
        cos_t = torch.cos(theta)
        sin_e = torch.sin(self.engrams)
        cos_e = torch.cos(self.engrams)
        cos_d = (cos_t @ cos_e.t() + sin_t @ sin_e.t()) / D
        weights = F.softmax(cos_d / math.pi, dim=-1)
        w_sin = weights @ sin_e
        w_cos = weights @ cos_e
        return w_sin * cos_t - w_cos * sin_t

    @torch.no_grad()
    def write_engram(self, theta: Tensor, slot: int | None = None) -> int:
        if theta.dim() == 2:
            theta = theta.mean(dim=0)
        assert theta.shape[-1] == self.dim
        if slot is None:
            diff = self.engrams.data - theta.unsqueeze(0)
            coh = torch.cos(diff).mean(dim=-1)
            slot = int(coh.argmin().item())
        self.engrams.data[slot] = 0.9 * self.engrams.data[slot] + 0.1 * theta
        return slot

    @torch.no_grad()
    def forget_engram(self, slot: int):
        """Explicitly clear one per-layer engram slot."""
        self.engrams.data[slot].zero_()

    @torch.no_grad()
    def forget_matching(self, theta: Tensor, min_coherence: float = 0.45) -> list[int]:
        """Clear per-layer engram slots whose phase coherence with theta is high."""
        if theta.dim() == 2:
            theta = theta.mean(dim=0)
        diff = self.engrams.data - theta.unsqueeze(0)
        coh = torch.cos(diff).mean(dim=-1)
        cleared = []
        for s in range(self.n_ensembles):
            if float(coh[s].item()) >= min_coherence:
                self.engrams.data[s].zero_()
                cleared.append(s)
        return cleared

    @torch.no_grad()
    def probe(self, theta: Tensor) -> Tensor:
        """Return [E] mean cos-coherence of theta vs each engram slot."""
        if theta.dim() == 2:
            theta = theta.mean(dim=0)
        diff = self.engrams.data - theta.unsqueeze(0)
        return torch.cos(diff).mean(dim=-1)


class AttractorMemory(nn.Module):
    """Cross-layer attractor memory (hippocampal-style).

    Lifelong default: usage never auto-decays, so consolidated slots stay
    preferred (high usage) and are not overwritten by argmin eviction unless
    the user calls forget_* or enables soft decay of weak slots.
    """

    def __init__(self, n_layers: int, dim: int, capacity: int = 256):
        super().__init__()
        self.n_layers = n_layers
        self.dim = dim
        self.capacity = capacity
        self.register_buffer(
            "attractors",
            torch.zeros(n_layers, capacity, dim),
        )
        self.register_buffer("usage", torch.zeros(n_layers, capacity))

    @torch.no_grad()
    def write(self, layer_idx: int, theta: Tensor):
        if theta.dim() == 2:
            theta = theta.mean(dim=0)
        slot = int(self.usage[layer_idx].argmin().item())
        self.attractors[layer_idx, slot] = theta
        self.usage[layer_idx] += 1e-3
        self.usage[layer_idx, slot] += 1.0

    @torch.no_grad()
    def pull(self, layer_idx: int, theta: Tensor, gain: float = 0.1) -> Tensor:
        diff = self.attractors[layer_idx].unsqueeze(0) - theta.unsqueeze(-2)
        sin_d = torch.sin(diff).mean(dim=-1)
        cos_d = torch.cos(diff).mean(dim=-1)
        w = F.softmax(cos_d / math.pi, dim=-1)
        pull = (w.unsqueeze(-1) * torch.sin(diff)).sum(dim=-2)
        return gain * pull

    @torch.no_grad()
    def decay(
        self,
        rate: float = 0.999,
        only_weak: bool = False,
        usage_threshold: float = 0.5,
    ):
        """Decay usage counters (does not wipe attractor values).

        Global decay makes *all* slots easier to overwrite — avoid in
        lifelong continuous mode. Prefer only_weak=True to age low-usage
        slots only, or skip decay entirely (default lifelong policy).
        """
        if not only_weak:
            self.usage.mul_(rate)
            return
        weak = self.usage < usage_threshold
        self.usage = torch.where(weak, self.usage * rate, self.usage)

    @torch.no_grad()
    def forget_slot(self, layer_idx: int, slot: int):
        """Explicitly erase one cross-layer memory slot."""
        self.attractors[layer_idx, slot].zero_()
        self.usage[layer_idx, slot] = 0.0

    @torch.no_grad()
    def forget_matching(
        self,
        theta: Tensor,
        min_coherence: float = 0.45,
        layers: list[int] | None = None,
    ) -> list[tuple[int, int]]:
        """Erase attractor slots matching theta (phase cos-coherence)."""
        if theta.dim() == 2:
            theta = theta.mean(dim=0)
        layer_ids = layers if layers is not None else list(range(self.n_layers))
        cleared: list[tuple[int, int]] = []
        for li in layer_ids:
            diff = self.attractors[li] - theta.unsqueeze(0)
            coh = torch.cos(diff).mean(dim=-1)
            for s in range(self.capacity):
                if float(coh[s].item()) >= min_coherence:
                    self.forget_slot(li, s)
                    cleared.append((li, s))
        return cleared

    @torch.no_grad()
    def forget_all(self):
        """Wipe entire cross-layer memory (explicit user request only)."""
        self.attractors.zero_()
        self.usage.zero_()

    @torch.no_grad()
    def probe(self, theta: Tensor, layer_idx: int | None = None) -> Tensor:
        """Return coherence of theta vs attractors.

        layer_idx set -> [capacity]; else -> [n_layers, capacity].
        """
        if theta.dim() == 2:
            theta = theta.mean(dim=0)
        if layer_idx is not None:
            diff = self.attractors[layer_idx] - theta.unsqueeze(0)
            return torch.cos(diff).mean(dim=-1)
        # [L, C]
        diff = self.attractors - theta.view(1, 1, -1)
        return torch.cos(diff).mean(dim=-1)

    def occupied_count(self, usage_eps: float = 0.5) -> int:
        """Count slots with real writes (usage += 1.0), ignoring the tiny
        global +1e-3 bump applied on every write."""
        return int((self.usage >= usage_eps).sum().item())
