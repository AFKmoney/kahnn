"""
KuroBrain — Kuramoto-Attractor Holographic Hypervector Network (KAHNN)

A non-Transformer, neuroscience-inspired AI paradigm:

  * Distributed hypervector population codes (bipolar, D ~ 8k–32k)
  * Holographic Reduced Representations (HRR) via FFT circular convolution
  * Kuramoto oscillator layers for continuous-thought phase dynamics
  * Attractor basins for memory / retrieval (phase-locked clusters)
  * Local online learning (Hebbian + phase-coupling STDP-like rules)
  * No backprop-through-time, no self-attention, no replay buffer
  * Streams 25B tokens in ~1–2 days on a single H100 for ~1B parameters

Public modules:
  kuro_brain.hypervectors   — HD vector primitives
  kuro_brain.holographic    — HRR bind / unbind / bundle
  kuro_brain.kuramoto       — oscillator layer + attractor dynamics
  kuro_brain.encoder        — token <-> hypervector codebook
  kuro_brain.model          — full KAHNN stack
  kuro_brain.continuous_learning — online local learning rules
"""

from .hypervectors import (
    BipolarHDV, random_hypervector, bundle, similarity, normalize,
)
from .holographic import (
    hrr_bind, hrr_unbind, hrr_bundle, hrr_similarity,
)
from .kuramoto import KuramotoLayer, AttractorMemory
from .encoder import HypervectorTokenizer
from .model import KAHNN, KAHNNConfig
from .continuous_learning import OnlineLearner

__all__ = [
    "BipolarHDV", "random_hypervector", "bundle", "similarity", "normalize",
    "hrr_bind", "hrr_unbind", "hrr_bundle", "hrr_similarity",
    "KuramotoLayer", "AttractorMemory",
    "HypervectorTokenizer",
    "KAHNN", "KAHNNConfig",
    "OnlineLearner",
]

__version__ = "0.1.0"
