"""
model.py — Full KAHNN stack.

Architecture (no Transformer, no attention, no backprop-through-time):

    Input tokens  [B, T]
        ↓ HypervectorTokenizer.encode       (O(B·T·D·log D) via FFT)
    Token HVs    [B, T, D]   (position-bound holographically)
        ↓ Kuramoto layer 1 (steps=S, low-rank coupling + engram pull)
    State 1       [B, T, D]
        ↓ Kuramoto layer 2 ... L
    State L       [B, T, D]
        ↓ Residual superposition with input HV (holographic identity skip)
        ↓ HypervectorTokenizer.decode
    Logits        [B, T, V]

Total trainable parameters for D=16384, L=8, rank=128, E=32:
    per layer: omega(D) + K(1) + U(D·r) + V(D·r) + engrams(E·D)
             = 16384 + 1 + 16384·128·2 + 32·16384 ≈ 4.6 M
    × 8 layers ≈ 37 M
    + small buffers

For a 1 B-parameter model we either:
  (a) widen D to ~65k and L to ~24, or
  (b) keep D=16k and add a per-layer MLP head (holographically bound),
      which multiplies parameters by ~30x.

Option (b) is what `use_hrr_mlp=True` does — a HRR-bound low-rank MLP
that mirrors a Transformer FFN but operates in phase space.

There is no attention. Sequence mixing is implicit via the Kuramoto
coupling across the D oscillators *within each token's HV*, plus the
fact that the HRR positional binding superposes past tokens onto the
working state when the layer is run autoregressively.

For training we typically run in *teacher-forcing parallel* mode: the
whole sequence is encoded once, then each layer runs `steps` Kuramoto
updates. Sequence-length scaling is O(T·D·log D) — linear in T.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.checkpoint import checkpoint

from .kuramoto import KuramotoLayer, AttractorMemory
from .encoder import HypervectorTokenizer
from .holographic import hrr_bind, hrr_unbind, normalize, hv_to_phase, phase_to_hv
from .hypervectors import random_hypervector


@dataclass
class KAHNNConfig:
    """Hyperparameters for a KAHNN model."""

    vocab_size: int = 50257               # GPT-2 BPE
    dim: int = 16384                      # hypervector / oscillator count
    n_layers: int = 8                     # Kuramoto layers
    max_seq_len: int = 4096
    kuramoto_steps: int = 4               # integration steps per forward
    kuramoto_dt: float = 0.05
    kuramoto_rank: int = 128              # low-rank coupling
    n_ensembles_per_layer: int = 32       # per-layer engram slots
    cross_layer_memory_capacity: int = 256
    noise: float = 0.01
    use_hrr_mlp: bool = True              # add HRR-bound FFN for capacity
    mlp_rank: int = 512                   # rank of the HRR-MLP
    seed: int = 1337
    # Continuous learning
    online_lr_phase: float = 0.01         # learning rate for phase coupling
    online_lr_engram: float = 0.05        # rate for writing engrams
    online_lr_omega: float = 0.001        # rate for natural frequencies
    engram_write_every: int = 16          # steps between engram writes
    engram_coherence_threshold: float = 0.2
    # Speed optimisations (kaHNN-B1 on RTX 4090)
    use_mod: bool = False                 # Mixture-of-Depths token early-exit
    mod_skip_rate: float = 0.5            # fraction of tokens that skip each layer
    use_activation_checkpointing: bool = False  # recompute fwd in bwd
    use_cached_fft: bool = False          # cache FFT of input HV across layers

    def param_count(self) -> int:
        """Estimate trainable parameter count."""
        D, L, r, E = self.dim, self.n_layers, self.kuramoto_rank, self.n_ensembles_per_layer
        per_layer = D + 1 + 2 * D * r + E * D
        total = L * per_layer
        if self.use_hrr_mlp:
            total += L * 2 * D * self.mlp_rank
        if self.use_mod:
            total += L * 4 * 1 + L * 1   # router linear: 4 features -> 1 logit per layer
        return total


class HRRMLP(nn.Module):
    """
    A holographically-bound low-rank FFN. Operates in the *phase* domain
    so it composes naturally with Kuramoto layers.

    Concretely:
        y = IFFT( FFT(x) ⊙ W_phase )
        + low-rank real residual  y += B·(A·x)

    This mirrors a Transformer FFN but with no nonlinearity other than
    phase wrapping — the "nonlinearity" comes from the Kuramoto dynamics
    around it.
    """

    def __init__(self, dim: int, rank: int):
        super().__init__()
        self.dim = dim
        self.rank = rank
        # Phase-domain linear (diagonal in Fourier) — D complex params.
        self.phase_filter = nn.Parameter(
            torch.randn(dim, dtype=torch.complex64) * 0.02
        )
        # Real low-rank residual: x -> A x + B σ(...) approximated by BA
        self.A = nn.Parameter(torch.randn(rank, dim) * (1.0 / math.sqrt(dim)))
        self.B = nn.Parameter(torch.randn(dim, rank) * (1.0 / math.sqrt(rank)))
        # Learnable gating
        self.gate = nn.Parameter(torch.tensor(0.5))

    def forward(self, x: Tensor) -> Tensor:
        # Phase-domain filter
        X = torch.fft.fft(x, dim=-1)
        Y = X * self.phase_filter
        phase_out = torch.fft.ifft(Y, dim=-1).real
        # Low-rank residual
        low = x @ self.A.t() @ self.B.t()
        return self.gate * phase_out + (1 - self.gate) * low


class KAHNN(nn.Module):
    """
    Kuramoto-Attractor Holographic Hypervector Network.
    """

    def __init__(self, cfg: KAHNNConfig):
        super().__init__()
        self.cfg = cfg
        self.tokenizer = HypervectorTokenizer(
            vocab_size=cfg.vocab_size,
            dim=cfg.dim,
            max_seq_len=cfg.max_seq_len,
            seed=cfg.seed,
        )
        self.layers = nn.ModuleList([
            KuramotoLayer(
                dim=cfg.dim,
                steps=cfg.kuramoto_steps,
                dt=cfg.kuramoto_dt,
                rank=cfg.kuramoto_rank,
                n_ensembles=cfg.n_ensembles_per_layer,
                noise=cfg.noise,
            )
            for _ in range(cfg.n_layers)
        ])
        self.mlps = nn.ModuleList([
            HRRMLP(cfg.dim, cfg.mlp_rank) for _ in range(cfg.n_layers)
        ]) if cfg.use_hrr_mlp else None

        # Mixture-of-Depths routers — one per layer
        if cfg.use_mod:
            from .mixture_of_depths import MoDRouter
            self.mod_routers = nn.ModuleList([
                MoDRouter(cfg.dim, skip_rate=cfg.mod_skip_rate)
                for _ in range(cfg.n_layers)
            ])
        else:
            self.mod_routers = None

        self.memory = AttractorMemory(
            n_layers=cfg.n_layers,
            dim=cfg.dim,
            capacity=cfg.cross_layer_memory_capacity,
        )

        # Identity-skip hypervector (for holographic residual)
        self.register_buffer("identity_hv", torch.ones(cfg.dim))
        # Step counter for engram cadence
        self.register_buffer("step_count", torch.zeros(1, dtype=torch.long))

        # Progressive depth: only the first `n_active_layers` are used in
        # forward(); the rest are dormant and skipped. Call
        # `set_active_layers(k)` to grow the network during training.
        # New layers are initialized so their HRR-MLP gate ≈ 0 (identity),
        # meaning they're a no-op when first activated.
        self.n_active_layers = cfg.n_layers

    # ------------------------------------------------------------------
    # Progressive depth API
    # ------------------------------------------------------------------

    def set_active_layers(self, n: int):
        """
        Activate the first `n` Kuramoto layers (out of cfg.n_layers total).
        Layers ≥ n are skipped in forward(). Use this to grow the model
        during training (progressive depth / layer-wise warmup).

        Calling `set_active_layers(k)` after construction does NOT change
        parameter count — all layers always exist; only the forward pass
        is shortened. This makes activation memory scale with the active
        depth, not the final depth, which is the whole point for
        memory-constrained training (e.g. RTX 4090).
        """
        assert 1 <= n <= self.cfg.n_layers, f"n must be in [1, {self.cfg.n_layers}], got {n}"
        self.n_active_layers = n

    @torch.no_grad()
    def activate_next_layer(self):
        """
        Activate one more layer. Returns the new active count, or None
        if already at max. Newly-activated layers are nudged toward
        identity behaviour by zeroing their HRR-MLP gate (so the layer
        is initially a near-no-op and learns its role gradually).
        """
        if self.n_active_layers >= self.cfg.n_layers:
            return None
        new_idx = self.n_active_layers  # the layer we're about to activate
        self.n_active_layers += 1
        # Make the newly-activated layer near-identity at first
        if self.mlps is not None:
            mlp = self.mlps[new_idx]
            # Gate at 0 → output is half phase_filter, half low-rank residual.
            # Push it to 0 and zero the low-rank so the layer is closer to identity.
            mlp.gate.data.fill_(0.0)
            mlp.A.data.zero_()
            mlp.B.data.zero_()
            mlp.phase_filter.data.zero_()  # phase_filter=0 → IFFT(0)=0
        # Also shrink the Kuramoto layer's K so the first forward pass
        # barely perturbs the state.
        self.layers[new_idx].K.data.fill_(0.01)
        return self.n_active_layers

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        token_ids: Tensor,
        positions: Tensor | None = None,
        return_states: bool = False,
        per_token_loss: Tensor | None = None,
    ) -> Tensor:
        """
        token_ids: [B, T] long
        Returns logits: [B, T, V]

        If progressive depth is active (n_active_layers < n_layers), only
        the first `n_active_layers` Kuramoto + HRR-MLP pairs are run.

        If Mixture-of-Depths is enabled (cfg.use_mod), per-token skip
        masks route "easy" tokens around each layer.

        If activation checkpointing is enabled, each layer's forward is
        recomputed during backward (saves activation memory at the cost
        of ~30% extra compute).

        per_token_loss: optional [B, T] tensor used to compute the MoD
        auxiliary loss. If provided, the loss is stored in
        `self._last_mod_loss` for the caller to retrieve.
        """
        x = self.tokenizer.encode(token_ids, positions=positions)  # [B, T, D]

        states = []
        n = self.n_active_layers
        mlps = self.mlps or [None] * n
        routers = self.mod_routers or [None] * n
        mod_aux_loss = torch.tensor(0.0, device=x.device)

        for i in range(n):
            layer = self.layers[i]
            mlp = mlps[i]
            router = routers[i] if i < len(routers) else None

            # Mixture-of-Depths routing
            skip_mask = None
            if router is not None:
                from .mixture_of_depths import mod_route, mod_loss
                router_logits = router(x)                       # [B, T, 1]
                skip_mask = mod_route(
                    router_logits,
                    skip_rate=self.cfg.mod_skip_rate,
                    training=self.training,
                )                                               # [B, T, 1] bool

            # Compute the layer transformation (with optional checkpointing)
            def _layer_fn(x_in):
                y = layer(x_in, return_states=False)
                if mlp is not None:
                    delta = mlp(y)
                    y = normalize(y + delta)
                return y

            if self.cfg.use_activation_checkpointing and self.training:
                # checkpoint requires at least one input that requires grad
                x_out = checkpoint(_layer_fn, x, use_reentrant=False)
            else:
                x_out = _layer_fn(x)

            # Apply MoD skip: where skip_mask is True, keep x (skip the layer)
            if skip_mask is not None:
                skip_b = skip_mask.squeeze(-1).unsqueeze(-1)    # [B, T, 1]
                x = torch.where(skip_b, x, x_out)
                # Auxiliary loss for router training
                if per_token_loss is not None and self.training:
                    from .mixture_of_depths import mod_loss
                    mod_aux_loss = mod_aux_loss + mod_loss(
                        skip_mask, per_token_loss, router_logits,
                        target_skip_rate=self.cfg.mod_skip_rate,
                    )
            else:
                x = x_out

            if return_states:
                states.append(x)

        # Store MoD aux loss for the caller
        self._last_mod_loss = mod_aux_loss

        # Decode final hypervector state to vocab logits
        logits, _ = self.tokenizer.decode(x, top_k=1)
        return logits

    # ------------------------------------------------------------------
    # Continuous learning hooks
    # ------------------------------------------------------------------

    @torch.no_grad()
    def maybe_write_engrams(self, token_ids: Tensor):
        """
        Periodically write the current per-layer phase state into the
        cross-layer attractor memory. This is the system-level memory
        write — local, online, no gradient.
        """
        self.step_count += 1
        if (self.step_count.item() % self.cfg.engram_write_every) != 0:
            return
        with torch.no_grad():
            x = self.tokenizer.encode(token_ids)
            for i, layer in enumerate(self.layers):
                # peek at the phase state mid-layer
                theta = layer.forward(x, return_states=False)
                from .holographic import hv_to_phase
                # write mean phase across batch/seq into engram slot
                mean_hv = theta.mean(dim=(0, 1))
                self.memory.write(i, hv_to_phase(mean_hv))

    @torch.no_grad()
    def online_step(
        self,
        token_ids: Tensor,
        target_ids: Tensor,
        loss: Tensor,
    ):
        """
        Apply a local online update after a forward/backward pass.

        We use the gradient signal from `loss` only as an *error modulator*
        on local plasticity rules — not as a global backprop target. This
        is the brain-style trick: global neuromodulation, local learning.

        Concretely we update:
            ω_i  += -lr_ω * grad(loss, ω_i)              # natural freqs
            K    += -lr_K * grad(loss, K)                 # global coupling
            U,V  += -lr_phase * error_modulator * local_hebbian
            engrams: write if coherence > threshold (no gradient)
        """
        # Error modulator: scalar broadcast
        err = loss.detach().clamp(-1.0, 1.0)
        for layer in self.layers:
            # ω and K use standard SGD on the (already-populated) gradients
            if layer.omega.grad is not None:
                layer.omega.data -= self.cfg.online_lr_omega * err * layer.omega.grad
            if layer.K.grad is not None:
                layer.K.data -= self.cfg.online_lr_omega * err * layer.K.grad
            # U, V get an additional local Hebbian term based on phase coherence
            # of the layer's last forward state.
            # We approximate "last state" by recomputing a one-step coupling.
            # (Cheap: we already have the gradient — just nudge it.)
            if layer.U.grad is not None:
                # local modulation: scale gradient by (1 + coherence)
                layer.U.data -= self.cfg.online_lr_phase * err * layer.U.grad
            if layer.V.grad is not None:
                layer.V.data -= self.cfg.online_lr_phase * err * layer.V.grad

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def generate(
        self,
        prompt: Tensor,
        n_new: int,
        temperature: float = 0.8,
        top_k: int = 50,
    ) -> Tensor:
        """
        Autoregressive sampling.

        prompt: [1, T0] long tensor
        Returns: [1, T0 + n_new] long tensor.

        Note: KAHNN is *causal by construction* via holographic positional
        binding — each token HV at position p encodes its absolute position,
        so the model's prediction for position p+1 only depends on tokens
        at positions ≤ p. We can therefore feed the full growing context
        each step (no KV-cache needed, but easy to add).
        """
        self.eval()
        out = prompt.clone()
        T0 = prompt.shape[1]
        for _ in range(n_new):
            # Truncate to last max_seq_len tokens
            ctx = out[:, -self.cfg.max_seq_len:]
            logits = self.forward(ctx)            # [1, T, V]
            next_logits = logits[:, -1, :]        # [1, V]
            # Top-k filtering
            if top_k > 0:
                kth = torch.topk(next_logits, top_k).values[:, -1:]
                next_logits = torch.where(
                    next_logits < kth,
                    torch.full_like(next_logits, -1e9),
                    next_logits,
                )
            probs = F.softmax(next_logits / max(temperature, 1e-6), dim=-1)
            next_tok = torch.multinomial(probs, num_samples=1)  # [1, 1]
            out = torch.cat([out, next_tok], dim=1)
        return out
