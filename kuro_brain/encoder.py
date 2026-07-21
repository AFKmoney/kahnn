"""
encoder.py — Token <-> hypervector encoder / codebook.

We never use dense embedding matrices. Instead each token id t in [0, V)
is mapped to a fixed random bipolar hypervector H_t ∈ {-1,+1}^D. These
are *item-memory* (random) hypervectors; they are NOT learned. Only the
Kuramoto oscillator parameters and engrams are learned.

This is the single biggest reason KAHNN trains in ~1 day on 25B tokens:
the V×D embedding matrix that dominates Transformer memory (e.g.
50257 × 8192 × 2 bytes = 822 MB for GPT-2's embedding) is *not* a learnable
parameter here — it is a fixed random projection. The model has to learn
relations, not lexicon.

Positional encoding is holographic: position p is encoded as a level-HV
P_p, and the binding bind(token, P_p) gives a position-aware token HV.
This is permutation-equivariant in the HRR algebra, and again requires
no learnable parameters.

Output decoding uses similarity against the item memory — O(V × D) but
amenable to efficient top-k via the FFT structure of HRR.
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
from torch import Tensor

from .hypervectors import (
    random_hypervectors, level_hypervectors, similarity,
)
from .holographic import hrr_bind


class HypervectorTokenizer(nn.Module):
    """
    Stateless-in-parameters token <-> hypervector encoder.

    Parameters
    ----------
    vocab_size : int
        V — number of tokens.
    dim : int
        D — hypervector dimensionality.
    max_seq_len : int
        Maximum context length (for positional level-HVs).
    seed : int
        Seed for the random item-memory and position-memory.
    """

    def __init__(
        self,
        vocab_size: int,
        dim: int,
        max_seq_len: int = 4096,
        seed: int = 1337,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        self.max_seq_len = max_seq_len

        g = torch.Generator().manual_seed(seed)

        # Item memory: V random bipolar hypervectors. Not learnable.
        token_hvs = random_hypervectors(vocab_size, dim, generator=g)
        self.register_buffer("token_hvs", token_hvs)

        # Positional level-HVs: P_0, P_1, ... P_{L-1} (correlated chain)
        pos_hvs = level_hypervectors(max_seq_len, dim)
        self.register_buffer("pos_hvs", pos_hvs)

        # A random "role" HV used for context-position binding when needed.
        role_hv = random_hypervectors(1, dim, generator=g).squeeze(0)
        self.register_buffer("role_hv", role_hv)

    # ------------------------------------------------------------------
    # Encoding
    # ------------------------------------------------------------------

    def encode(self, token_ids: Tensor, positions: Tensor | None = None) -> Tensor:
        """
        token_ids: [B, T] long tensor
        positions: optional [B, T] long tensor (defaults to 0..T-1)
        Returns:   [B, T, D] real hypervectors (bipolar, position-bound)
        """
        B, T = token_ids.shape
        if positions is None:
            positions = torch.arange(T, device=token_ids.device).unsqueeze(0).expand(B, T)
        positions = positions.clamp_max(self.max_seq_len - 1)

        # [B, T, D]
        token_hv = self.token_hvs[token_ids]
        pos_hv = self.pos_hvs[positions]
        # Holographic binding for position-aware token representation
        return hrr_bind(token_hv, pos_hv)

    def encode_token_only(self, token_ids: Tensor) -> Tensor:
        """Lookup token HVs without positional binding — used for targets."""
        return self.token_hvs[token_ids]

    # ------------------------------------------------------------------
    # Decoding
    # ------------------------------------------------------------------

    def decode(self, hv: Tensor, top_k: int = 1, temperature: float = 1.0):
        """
        Decode a hypervector back to token logits.

        hv: [B, ..., D] (or [D])
        Returns: logits [B, ..., V], and (top_k_indices, top_k_logits).
        """
        lead = hv.shape[:-1]
        D = hv.shape[-1]
        flat = hv.reshape(-1, D)                                # [N, D]
        # cosine similarity against the full codebook
        sims = similarity(flat.unsqueeze(1), self.token_hvs.unsqueeze(0))  # [N, V]
        logits = sims / max(temperature, 1e-6)
        out_logits = logits.reshape(*lead, self.vocab_size)

        if top_k == 1:
            idx = out_logits.argmax(dim=-1)
            return out_logits, (idx, None)
        topk = out_logits.topk(top_k, dim=-1)
        return out_logits, (topk.indices, topk.values)

    # ------------------------------------------------------------------
    # Codebook utilities
    # ------------------------------------------------------------------

    @torch.no_grad()
    def nearest_token(self, hv: Tensor) -> Tensor:
        """Return the single nearest token id for each input HV."""
        logits, (idx, _) = self.decode(hv, top_k=1)
        return idx
