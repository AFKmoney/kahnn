"""
continuous_learning.py — Online local learning + lifelong engram memory.

After a language+code Adam base, continuous_mode keeps learning without
full retrain. Consolidated engrams do NOT auto-decay (soft_decay default
OFF). Explicit forget: learner.forget / forget_slot / forget_all.
Honest limit: continuous learning does not replace a real base corpus.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .holographic import hv_to_phase


class OnlineLearner:
    """Adam (pretrain) + local plasticity + engram writes; optional soft decay."""

    def __init__(
        self,
        model: nn.Module,
        cfg,
        base_optimizer: torch.optim.Optimizer | None = None,
        continuous_mode: bool = False,
        soft_decay: bool = False,
        soft_decay_rate: float = 0.9995,
        soft_decay_only_weak: bool = True,
        soft_decay_usage_threshold: float = 0.5,
    ):
        self.model = model
        self.cfg = cfg
        self.base_optimizer = base_optimizer
        self.continuous_mode = continuous_mode
        self.soft_decay = soft_decay
        self.soft_decay_rate = soft_decay_rate
        self.soft_decay_only_weak = soft_decay_only_weak
        self.soft_decay_usage_threshold = soft_decay_usage_threshold
        self.enable_local_plasticity = continuous_mode
        self._step = 0

    def step(self, token_ids: Tensor, target_ids: Tensor, loss: Tensor):
        self._step += 1
        if self.base_optimizer is not None and not self.continuous_mode:
            self.base_optimizer.step()
            self.base_optimizer.zero_grad(set_to_none=True)
        if self.enable_local_plasticity or self.continuous_mode:
            self._local_phase_plasticity(token_ids, loss)
        if self._step % self.cfg.engram_write_every == 0:
            self._maybe_consolidate_engrams(token_ids)
        # Lifelong default: no global decay. Opt-in soft_decay ages weak slots only.
        if self.continuous_mode and self.soft_decay:
            self.model.memory.decay(
                rate=self.soft_decay_rate,
                only_weak=self.soft_decay_only_weak,
                usage_threshold=self.soft_decay_usage_threshold,
            )

    @torch.no_grad()
    def _local_phase_plasticity(self, token_ids: Tensor, loss: Tensor):
        err = float(loss.detach().clamp(-2.0, 2.0).item())
        lr = self.cfg.online_lr_phase * err
        if lr == 0.0:
            return
        x = self.model.tokenizer.encode(token_ids)
        for i, layer in enumerate(self.model.layers):
            if hasattr(self.model, "n_active_layers") and i >= self.model.n_active_layers:
                break
            theta = hv_to_phase(x)
            engrams = layer.engrams
            D = engrams.shape[-1]
            sin_t, cos_t = torch.sin(theta), torch.cos(theta)
            sin_e, cos_e = torch.sin(engrams), torch.cos(engrams)
            coh = (cos_t @ cos_e.t() + sin_t @ sin_e.t()) / D
            best = coh.argmax(dim=-1)
            nearest_engram = engrams[best]
            sin_pull = torch.sin(nearest_engram - theta).mean(dim=(0, 1))
            cos_pull = torch.cos(nearest_engram - theta).mean(dim=(0, 1))
            V_proj = (layer.V.data * sin_pull.unsqueeze(1)).mean(dim=0)
            U_proj = (layer.U.data * cos_pull.unsqueeze(1)).mean(dim=0)
            layer.U.data += lr * sin_pull.unsqueeze(1) * V_proj.unsqueeze(0)
            layer.V.data += lr * cos_pull.unsqueeze(1) * U_proj.unsqueeze(0)
            x = layer(x)

    @torch.no_grad()
    def _maybe_consolidate_engrams(self, token_ids: Tensor):
        x = self.model.tokenizer.encode(token_ids)
        for i, layer in enumerate(self.model.layers):
            if hasattr(self.model, "n_active_layers") and i >= self.model.n_active_layers:
                break
            x = layer(x)
            theta = hv_to_phase(x)
            z = torch.exp(1j * theta).mean(dim=(0, 1))
            r = z.abs().mean().item()
            if r >= self.cfg.engram_coherence_threshold:
                mean_hv = x.mean(dim=(0, 1))
                mean_theta = hv_to_phase(mean_hv)
                layer.write_engram(mean_theta)
                self.model.memory.write(i, mean_theta)

    @torch.no_grad()
    def force_consolidate(self, token_ids: Tensor) -> dict:
        written = []
        x = self.model.tokenizer.encode(token_ids)
        n = getattr(self.model, "n_active_layers", len(self.model.layers))
        for i, layer in enumerate(self.model.layers):
            if i >= n:
                break
            x = layer(x)
            mean_hv = x.mean(dim=(0, 1))
            mean_theta = hv_to_phase(mean_hv)
            slot = layer.write_engram(mean_theta)
            self.model.memory.write(i, mean_theta)
            written.append({"layer": i, "slot": slot})
        return {"written": written, "occupied": self.model.memory.occupied_count()}

    def teach(
        self,
        token_ids: Tensor,
        target_ids: Tensor | None = None,
        n_passes: int = 3,
        force_consolidate: bool = True,
    ) -> dict:
        self.continuous_mode = True
        self.enable_local_plasticity = True
        if target_ids is None:
            if token_ids.dim() == 1:
                token_ids = token_ids.unsqueeze(0)
            target_ids = token_ids.clone()
            if token_ids.shape[1] > 1:
                target_ids = token_ids[:, 1:]
                token_ids = token_ids[:, :-1]
        losses = []
        for _ in range(max(1, n_passes)):
            self.model.train()
            logits = self.model(token_ids)
            loss = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                target_ids.reshape(-1),
            )
            if loss.requires_grad:
                loss.backward()
            self.step(token_ids, target_ids, loss.detach())
            self.model.zero_grad(set_to_none=True)
            losses.append(float(loss.detach()))
        consol = (
            self.force_consolidate(token_ids)
            if force_consolidate
            else {"written": [], "occupied": self.model.memory.occupied_count()}
        )
        probe = self.probe(token_ids)
        return {
            "losses": losses,
            "loss_mean": sum(losses) / max(len(losses), 1),
            "consolidate": consol,
            "probe": probe,
        }

    @torch.no_grad()
    def probe(self, token_ids: Tensor) -> dict:
        if token_ids.dim() == 1:
            token_ids = token_ids.unsqueeze(0)
        x = self.model.tokenizer.encode(token_ids)
        n = getattr(self.model, "n_active_layers", len(self.model.layers))
        per_layer = []
        for i, layer in enumerate(self.model.layers):
            if i >= n:
                break
            x = layer(x)
            mean_hv = x.mean(dim=(0, 1))
            mean_theta = hv_to_phase(mean_hv)
            layer_coh = layer.probe(mean_theta)
            mem_coh = self.model.memory.probe(mean_theta, layer_idx=i)
            per_layer.append({
                "layer": i,
                "layer_engram_max": float(layer_coh.max().item()),
                "memory_max": float(mem_coh.max().item()),
                "memory_argmax": int(mem_coh.argmax().item()),
            })
        return {
            "per_layer": per_layer,
            "best_memory": max((p["memory_max"] for p in per_layer), default=0.0),
            "occupied": self.model.memory.occupied_count(),
        }

    @torch.no_grad()
    def forget(
        self,
        query_ids: Tensor | None = None,
        query_theta: Tensor | None = None,
        min_coherence: float = 0.45,
        layers: list[int] | None = None,
    ) -> dict:
        if query_theta is None:
            if query_ids is None:
                raise ValueError("forget() needs query_ids or query_theta")
            if query_ids.dim() == 1:
                query_ids = query_ids.unsqueeze(0)
            x = self.model.tokenizer.encode(query_ids)
            n = getattr(self.model, "n_active_layers", len(self.model.layers))
            for i, layer in enumerate(self.model.layers):
                if i >= n:
                    break
                x = layer(x)
            query_theta = hv_to_phase(x.mean(dim=(0, 1)))
        layer_cleared = []
        n = getattr(self.model, "n_active_layers", len(self.model.layers))
        layer_ids = layers if layers is not None else list(range(n))
        for li in layer_ids:
            cleared = self.model.layers[li].forget_matching(
                query_theta, min_coherence=min_coherence
            )
            if cleared:
                layer_cleared.append({"layer": li, "slots": cleared})
        mem_cleared = self.model.memory.forget_matching(
            query_theta, min_coherence=min_coherence, layers=layer_ids
        )
        return {
            "layer_engrams_cleared": layer_cleared,
            "memory_slots_cleared": mem_cleared,
            "occupied": self.model.memory.occupied_count(),
        }

    @torch.no_grad()
    def forget_slot(self, layer: int, slot: int, also_layer_engram: bool = True):
        self.model.memory.forget_slot(layer, slot)
        if also_layer_engram and 0 <= slot < self.model.layers[layer].n_ensembles:
            self.model.layers[layer].forget_engram(slot)
        return {
            "layer": layer,
            "slot": slot,
            "occupied": self.model.memory.occupied_count(),
        }

    @torch.no_grad()
    def forget_all(self):
        self.model.memory.forget_all()
        for layer in self.model.layers:
            layer.engrams.data.zero_()
        return {"occupied": 0}

    def state(self) -> dict:
        return {
            "step": self._step,
            "continuous_mode": self.continuous_mode,
            "soft_decay": self.soft_decay,
            "engram_usage_mean": self.model.memory.usage.mean().item(),
            "engram_occupied": self.model.memory.occupied_count(),
        }
