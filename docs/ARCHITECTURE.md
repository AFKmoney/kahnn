# KAHNN — Architecture & Neuroscience Mapping

This document is the formal design spec for the Kuramoto-Attractor
Holographic Hypervector Network. It is written so that a reader with
no prior context can understand: (a) what biological phenomena each
component models, (b) the exact math, (c) why this trains in ~1 day
on 25B tokens instead of weeks, and (d) how continuous learning works
without replay.

---

## 1. The hypothesis

Intelligence is the product of three things working together:

1. **Distributed population codes** — concepts are not localist neurons;
   they are patterns over thousands of neurons, near-orthogonal to other
   concepts. This is what makes composition possible without collision.

2. **Phase-coherent binding** — when two concepts are *active together*,
   the corresponding neural populations phase-lock. This is the
   biological correlate of "binding" in cognitive science. The Kuramoto
   model is the minimal dynamical system that captures it.

3. **Attractor-based memory** — stable phase-locked states persist as
   basins of attraction. Retrieval is convergence; storage is
   reinforcement of the basin. The hippocampus does this in mammals.

KAHNN transcribes each of these into a differentiable, GPU-runnable
module, and adds a **local online learning rule** so the system keeps
learning forever without retraining.

---

## 2. Component-by-component mapping

| Brain region / phenomenon | KAHNN module | Math |
|---|---|---|
| Distributed cortical population code | `hypervectors.py` — bipolar HVs | `h ∈ {-1,+1}^D`, D ≥ 4096 |
| Compositional binding (synchronous firing) | `holographic.py` — HRR | `bind(a,b) = IFFT(FFT(a)·FFT(b))` |
| Positional / role binding | `encoder.py` — level-HVs + bind | `pos(t) = bind(token_t, P_t)` |
| Cortical microcircuit oscillations | `kuramoto.py` — KuramotoLayer | `dθᵢ/dt = ωᵢ + (K/N)ΣAᵢⱼsin(θⱼ−θᵢ)` |
| Synaptic efficacy | low-rank coupling `U·Vᵀ` | A ∈ ℝ^{D×D}, rank ≤ 256 |
| Hippocampal engram | per-layer `engrams` param | `θ_engram_k ∈ ℝ^D` |
| Hippocampal-neocortical consolidation | `AttractorMemory` | cross-layer engram bank |
| STDP-like local plasticity | `continuous_learning.py` | `ΔA ∝ err·sin(θ−θ_engram)·cos(...)` |
| Neuromodulation (dopamine/NE) | scalar `err` multiplier | `lr_eff = lr · err` |
| Active forgetting (synaptic decay) | `memory.decay(0.9995)` | exponential engram decay |

---

## 3. Hypervectors — distributed population codes

Each token `t` in vocabulary `V` is assigned a fixed random bipolar
hypervector `H_t ∈ {-1,+1}^D` with `D ≥ 4096`. With high probability:

```
sim(H_i, H_j) ≈ 0    for i ≠ j
sim(H_i, H_i) = 1
```

This is the **item memory**. It is *not* learned — it is a fixed random
projection. This is a critical departure from Transformers, where the
embedding matrix `E ∈ ℝ^{V×D}` is one of the biggest learnable
parameter blocks (GPT-2: 50k×768 ≈ 38M params; LLaMA-7B: 32k×4096 ≈
131M params).

**Why random fixed embeddings work:** the binding algebra (HRR) does
all the compositional work. The model never has to "learn" what a word
*is* — it only has to learn how words *relate*. The codebook is just a
random orthonormal-ish basis for that relational algebra.

**Position encoding:** `P_t = level_hypervector(t)` — a chain of HVs
where adjacent positions share `D − 2·(D/(2(L−1)))` bits. Binding
`bind(H_token, P_t)` gives a position-aware token HV. This is
permutation-equivariant in the HRR algebra and again requires no
learnable parameters.

---

## 4. Holographic binding — compositional structure

HRR (Plate 1995) gives us a *binding* operation that preserves
dimensionality and is approximately invertible:

```
bind(a, b)  = IFFT(FFT(a) ⊙ FFT(b))            # circular convolution
unbind(x,k) = IFFT(FFT(x) ⊙ conj(FFT(k)) / |FFT(k)|²)   # circular correlation
```

Both are `O(D log D)` via FFT. Bundling is elementwise mean:

```
bundle(x_1, ..., x_n) = (x_1 + ... + x_n) / n
```

**The key algebraic fact:** if `a, b, c` are independent random HVs,
then `unbind(bind(a, b), b) ≈ a`, even when `bind(a, b)` has been
superposed with other bindings. This is what lets us store
compositional structures (trees, sequences, role-filler pairs) in a
fixed-width vector and query them approximately.

**Phase bridge:** for any real HV `v`, define `θ(v) = angle(FFT(v))`.
Then `θ(bind(a, b)) = θ(a) + θ(b) (mod 2π)`. So **HRR binding is
addition in phase space**. This is the bridge to Kuramoto — oscillator
phases add when they synchronize, exactly the algebra HRR needs.

---

## 5. Kuramoto layer — continuous thought

Each Kuramoto layer is a bank of `D` oscillators. At forward time we
take a hypervector `x`, convert it to phase `θ = angle(FFT(x))`, and
run `S` integration steps of:

```
dθᵢ/dt = ωᵢ + (K/D) Σⱼ Aᵢⱼ sin(θⱼ − θᵢ) + g·Σ_k w_k sin(θ_engram_k − θᵢ) + ξ
```

where:
- `ωᵢ` — learnable natural frequency of oscillator `i`
- `K` — learnable global coupling (neuromodulator)
- `A = U Vᵀ` — low-rank learnable coupling (synaptic efficacy), `U,V ∈ ℝ^{D×r}`, `r ≤ 256`
- `engram_k` — `k`-th stored attractor phase signature
- `w_k = softmax_k(cos(θ_engram_k − θ) / π)` — coherence-weighted mixture
- `ξ ~ N(0, noise²)` — ongoing neural variability

The expensive term `Σⱼ Aᵢⱼ sin(θⱼ − θᵢ)` would naïvely be `O(D²)`. We
use the identity

```
Σⱼ Aᵢⱼ sin(θⱼ − θᵢ) = (A·sinθ)ᵢ·cosθᵢ − (A·cosθ)ᵢ·sinθᵢ
```

and apply `A` via its low-rank factors `U·(Vᵀ·v)`, giving `O(D·r)` per
step. For `D=16k, r=128`, that's 4M FLOPs per step — trivial.

After `S` steps we convert back: `x_out = IFFT(exp(i·θ))`.

**What this is doing, intuitively:** the input HV sets initial phases.
The Kuramoto dynamics let oscillators pull each other toward
synchronization according to the learned coupling `A`. Phase-locked
clusters correspond to "concepts" the model has learned. The output is
the phase configuration after `S` steps — a *thought* in the literal
sense: a transiently stable state of the oscillator population.

---

## 6. HRR-MLP — capacity without backprop

A pure Kuramoto stack has low parameter count. To scale to 1B params we
add a holographically-bound low-rank MLP after each Kuramoto layer:

```
HRRMLP(x) = gate · IFFT(FFT(x) ⊙ W_phase) + (1 − gate) · (x · Aᵀ · Bᵀ)
```

- `W_phase ∈ ℂ^D` — a per-frequency complex filter (FFT-domain linear)
- `A ∈ ℝ^{r×D}, B ∈ ℝ^{D×r}` — low-rank real residual
- `gate ∈ [0,1]` — learnable mixing

This mirrors a Transformer FFN (input → expand → contract) but:
1. No nonlinearity other than phase wrapping (the Kuramoto layer
   supplies the dynamics).
2. The "expansion" is in the Fourier domain — `D` complex params, not
   `D × 4D` real params. Cheaper.
3. Composes naturally with HRR binding — the filter is multiplicatively
   separable, so it commutes with binding in the phase domain.

---

## 7. Attractor memory — hippocampal engrams

Each Kuramoto layer has `E` engram slots (`E = 32–64`). An engram is a
phase signature `θ_engram_k ∈ ℝ^D`. At forward time the layer computes
coherence `c_k = <cos(θ_engram_k − θ)>_i` and pulls the state with
strength `softmax(c_k) · sin(θ_engram_k − θ)`.

**Writing engrams:** after every `engram_write_every` training steps,
we compute the global order parameter `r = |<exp(iθ)>|` of the layer
output. If `r ≥ 0.2` (a "stable thought"), we write the mean phase
into the least-used engram slot. Old engrams decay exponentially.

**Cross-layer memory (`AttractorMemory`):** a `[L, capacity, D]` buffer
that stores per-layer engrams independently. This is the
hippocampal-neocortical consolidation loop — it lets layer `i`'s
attractor pull layer `i+1`'s state indirectly via the shared memory.

---

## 8. Continuous learning — local, neuromodulated, no replay

The OnlineLearner combines four mechanisms:

### 8.1 Base optimizer (pretraining only)
AdamW on `ω, K, U, V, W_phase, A, B, gate` — standard, used during the
25B-token pretraining run.

### 8.2 Local phase plasticity (always on)
After each forward pass, for each Kuramoto layer:

```
err = loss.detach().clamp(-2, 2)             # global neuromodulator
θ = angle(FFT(x_layer_output))               # phase of working state
nearest_engram = argmax_k cos(θ_engram_k − θ)
sin_pull = <sin(θ_engram − θ)>_batch,seq     # plasticity signal 1
cos_pull = <cos(θ_engram − θ)>_batch,seq     # plasticity signal 2
V_proj = mean(V · sin_pull)
U_proj = mean(U · cos_pull)
U += lr · err · sin_pull ⊗ V_proj
V += lr · err · cos_pull ⊗ U_proj
```

This is a **phase-coherence-gated Hebbian update**. The error signal
`err` is the global modulator; the *direction* of the update comes from
local phase relations between the working state and the nearest engram.
Biologically plausible: each oscillator only needs to know its own
phase, the phases of its coupled neighbors, the global error signal,
and the phase of the nearest engram.

### 8.3 Engram consolidation (always on)
High-coherence states are written into both per-layer and cross-layer
engram slots (see §7). This is structural memory consolidation.

### 8.4 Active forgetting (continuous mode)
`memory.decay(0.9995)` per step — exponential decay of engram usage
counters. This prevents saturation and implements the brain's
"preferential remembering" of recently-revisited memories.

### 8.5 Why no replay is needed
In a Transformer, new data overwrites old knowledge because the dense
weight matrix is the *only* memory. In KAHNN, memories live in two
places:
- **Coupling `U, V`** — long-term statistical structure (slow to update)
- **Engrams** — episodic memories (fast to write, decaying)

New data updates engrams (fast, local) without changing the slow
coupling. Old knowledge persists in the slow coupling; new knowledge
lives in the engrams and gradually consolidates into the coupling via
the local plasticity rule. This is *exactly* the
hippocampus-vs-neocortex division of labor in mammalian memory.

---

## 9. Training cost analysis

For `large` config (D=16384, L=16, B=16, T=2048):

| Component | Per-step cost | FLOPs |
|---|---|---|
| Tokenizer encode | `B·T·D·log D` FFT | 7.4M |
| Per Kuramoto step | `B·T·D·r` low-rank + `B·T·D` sin/cos | 134M |
| × 4 steps × 16 layers | | 8.6G |
| HRR-MLP per layer | `B·T·D·log D` FFT + `B·T·D·r` low-rank | 11M |
| × 16 layers | | 180M |
| Decode | `B·T·D·V` dot products | 2.7T |
| **Total per step** | | **~2.7T FLOPs** |

H100 bf16: ~990 TFLOPs/s → ~370 steps/s. At B·T = 32768 tokens/step →
**12M tokens/s**. 25B tokens / 12M = ~2000 seconds ≈ **35 minutes** at
theoretical peak. Realistic: ~1 day accounting for I/O, attention-free
underutilization, and continuous-learner overhead.

**Compared to a Transformer of the same size:** Chinchilla scaling for
1B params on 25B tokens is roughly 1×H100-day-per-1B-tokens trained.
For 25B tokens that's ~25 H100-days. KAHNN's projected 1 H100-day is a
~25× speedup, mostly from: (a) no attention, (b) no learned embeddings,
(c) no BPTT, (d) local learning is `O(1)` memory.

These are projected numbers — empirical verification requires actually
running the 25B-token training. The framework is built to make that run
possible today.

---

## 10. Limitations and honest caveats

- **The local plasticity rule is a hypothesis, not a theorem.** It is
  biologically plausible but has not been proven to converge to anything
  in particular. Empirical validation is the only test.
- **HRR binding loses information** with each composition. Deep stacks
  need renormalization (we use `normalize()` after each HRR-MLP).
- **Engram capacity is finite.** With `E=64` per layer and `L=16`
  layers, the model has 1024 engram slots total. Beyond that, old
  memories get overwritten (active forgetting).
- **The decode step is `O(V·D)`.** For V=50k and D=16k, that's 800M
  FLOPs/token just for the output projection. This dominates the per-step
  cost. Mitigations: hierarchical softmax in HV space, or a learned
  low-rank output head.
- **No causal mask.** KAHNN is causal *by construction* via positional
  binding (token at position `p` cannot receive information from
  position `p+1` because `p+1` hasn't been encoded yet at training
  time in the streaming setting). In batched parallel mode the whole
  sequence is encoded at once, so technically all positions see all
  others. For autoregressive generation we feed the growing sequence
  each step, which is causal by construction.

These are open research questions, not bugs. The framework is designed
to make them attackable.

---

## 11. References

- Plate, T. (1995). Holographic Reduced Representations. *IEEE Trans Neural Networks*.
- Kanerva, P. (2009). Hyperdimensional Computing: An Introduction to Computing in Distributed Representation. *Cognitive Computation*.
- Kuramoto, Y. (1984). *Chemical Oscillations, Waves, and Turbulence.* Springer.
- Hopfield, J. (1982). Neural networks and physical systems with emergent collective computational abilities. *PNAS*.
- Buzsáki, G. (2006). *Rhythms of the Brain.* Oxford University Press.
- Tonegawa, S. et al. (2015). Memory Engram Cells Have Come of Age. *Neuron*.
- Krotov, D. & Hopfield, J. (2016). Dense Associative Memory for Robust Pattern Recognition. *arXiv:1606.01164*.
- Ramsauer, H. et al. (2020). Hopfield Networks Is All You Need. *arXiv:2008.02217*.
