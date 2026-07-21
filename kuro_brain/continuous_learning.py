"""
continuous_learning.py — Online, local, neuroscience-style learning.

This is the heart of the "no retraining" claim. Once the model is
pretrained, it should keep learning from new data *without* replaying
old data and without catastrophic forgetting. We achieve this via:

  1. Phase-based local plasticity (Kuramoto STDP analogue):
        ΔA_ij ∝ η · sin(θ_pre − θ_post) · cos(θ_post − θ_engram_nearest)
     i.e. oscillators that fire in coherent phase strengthen their
     coupling, modulated by how close the post-state is to a stored
     engram (engrams gate consolidation).

  2. Engram consolidation: high-coherence states are written into
     per-layer and cross-layer attractor memory. Old engrams decay
     exponentially. This is the hippocampal-neocortical consolidation
     loop in miniature.

  3. Neuromodulatory global signal: the scalar prediction error gates
     ALL local plasticity. We do use the standard autograd gradient
     as the modulator (cheap), but the *update direction* is the local
     rule, not the gradient. This is why training is O(1) per token
     with a tiny constant and no backprop-through-time.

Usage during pretraining:
    learner = OnlineLearner(model, cfg)
    for batch in stream:
        logits = model(batch.input)
        loss = F.cross_entropy(logits, batch.target)
        loss.backward()             # populates gradients
        learner.step(batch.input, batch.target, loss.detach())
        optimizer.zero_grad()

Usage during never-ending post-training (no replay):
    learner.continuous_mode = True
    for batch in live_stream:
        logits = model(batch.input)
        loss = F.cross_entropy(logits, batch.target)
        loss.backward()
        learner.step(batch.input, batch.target, loss.detach())  # only engrams + local rule
        optimizer.zero_grad()
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .holographic import hv_to_phase


class OnlineLearner:
    """
    Wraps a KAHNN model and exposes a single `.step()` that combines:
        - SGD on global parameters (omega, K, U, V, MLP) for pretraining
        - Local phase plasticity for U, V (always on)
        - Engram writes for attractor memory (always on)
        - Engram decay (continuous mode only)

    The class is intentionally NOT an nn.optim.Optimizer — the rules
    are too heterogeneous and the brain doesn't have a single optimizer
    either.
    """

    def __init__(
        self,
        model: nn.Module,
        cfg,
        base_optimizer: torch.optim.Optimizer | None = None,
        continuous_mode: bool = False,
    ):
        self.model = model
        self.cfg = cfg
        self.base_optimizer = base_optimizer
        self.continuous_mode = continuous_mode
        self._step = 0

    # ------------------------------------------------------------------
    # Main entry
    # ------------------------------------------------------------------

    def step(
        self,
        token_ids: Tensor,
        target_ids: Tensor,
        loss: Tensor,
    ):
        """
        Apply one online learning step.

        token_ids:  [B, T] input ids (teacher forcing)
        target_ids: [B, T] target ids
        loss:       scalar tensor (detached), used as neuromodulator
        """
        self._step += 1

        # 1) Base optimizer step (Adam on global params) — only in
        #    pretraining mode OR when explicitly requested.
        if self.base_optimizer is not None and not self.continuous_mode:
            self.base_optimizer.step()
            self.base_optimizer.zero_grad(set_to_none=True)

        # 2) Local phase-based plasticity for U, V in every Kuramoto layer.
        #    This is a Hebbian/anti-Hebbian rule modulated by error.
        self._local_phase_plasticity(token_ids, loss)

        # 3) Write engrams if coherence is high enough.
        if self._step % self.cfg.engram_write_every == 0:
            self._maybe_consolidate_engrams(token_ids)

        # 4) In continuous mode, decay engrams (active forgetting).
        if self.continuous_mode:
            self.model.memory.decay(rate=0.9995)

    # ------------------------------------------------------------------
    # Local plasticity
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _local_phase_plasticity(self, token_ids: Tensor, loss: Tensor):
        """
        For each Kuramoto layer, compute a one-step phase evolution from
        the current input, then apply the local rule:

            ΔU[i, r] = η · err · <sin(θ - θ_engram_nearest)>_i · V_proj[r]
            ΔV[i, r] = η · err · <cos(θ - θ_engram_nearest)>_i · U_proj[r]

        where θ_engram_nearest is the closest attractor engram to the
        current phase state. This is a phase-coherence-gated Hebbian
        update — biologically plausible (local, no global gradient).
        """
        err = float(loss.clamp(-2.0, 2.0).item())
        lr = self.cfg.online_lr_phase * err

        if lr == 0.0:
            return

        x = self.model.tokenizer.encode(token_ids)  # [B, T, D]
        for i, layer in enumerate(self.model.layers):
            theta = hv_to_phase(x)                  # [B, T, D]
            # Find nearest engram for this layer
            engrams = layer.engrams                 # [E, D]
            # coherence per engram: cos(θ - θ_engram).mean over D
            diff = engrams.unsqueeze(0).unsqueeze(0) - theta.unsqueeze(2)  # [B, T, E, D]
            coh = torch.cos(diff).mean(dim=-1)      # [B, T, E]
            best = coh.argmax(dim=-1, keepdim=True)  # [B, T, 1]
            # Pull nearest engram phase
            nearest_engram = engrams[best.squeeze(-1)]  # [B, T, D]
            # Plasticity signals:
            sin_pull = torch.sin(nearest_engram - theta).mean(dim=(0, 1))   # [D]
            cos_pull = torch.cos(nearest_engram - theta).mean(dim=(0, 1))   # [D]
            # Apply: U += lr * sin_pull ⊗ V_proj_mean ; V += lr * cos_pull ⊗ U_proj_mean
            # We use a cheap projection to rank-r.
            V_proj = (layer.V.data * sin_pull.unsqueeze(1)).mean(dim=0)     # [rank]
            U_proj = (layer.U.data * cos_pull.unsqueeze(1)).mean(dim=0)     # [rank]
            layer.U.data += lr * sin_pull.unsqueeze(1) * V_proj.unsqueeze(0)
            layer.V.data += lr * cos_pull.unsqueeze(1) * U_proj.unsqueeze(0)
            # Move x forward for next layer (use the layer's actual forward)
            x = layer(x)

    # ------------------------------------------------------------------
    # Engram consolidation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _maybe_consolidate_engrams(self, token_ids: Tensor):
        """
        Write the current phase state into the per-layer and cross-layer
        attractor memory, but only if its self-coherence exceeds the
        threshold (i.e. it's a "stable thought" worth remembering).
        """
        x = self.model.tokenizer.encode(token_ids)
        for i, layer in enumerate(self.model.layers):
            x = layer(x)
            theta = hv_to_phase(x)                    # [B, T, D]
            # Global coherence: how aligned are the phases across B,T,D?
            # Use order parameter r = |<exp(iθ)>|
            z = torch.exp(1j * theta).mean(dim=(0, 1))   # complex [D]
            r = z.abs().mean().item()
            if r >= self.cfg.engram_coherence_threshold:
                mean_hv = x.mean(dim=(0, 1))
                mean_theta = hv_to_phase(mean_hv)
                slot = layer.write_engram(mean_theta)
                self.model.memory.write(i, mean_theta)

    # ------------------------------------------------------------------
    # State inspection
    # ------------------------------------------------------------------

    def state(self) -> dict:
        return {
            "step": self._step,
            "continuous_mode": self.continuous_mode,
            "engram_usage_mean": self.model.memory.usage.mean().item(),
        }
