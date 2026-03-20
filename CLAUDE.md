# Physics & Math-Inspired LLM Architectures — Research Repo

## What This Is

An active research program exploring whether structures from physics (quantum lattices,
thermodynamics, spin systems) and mathematics (tensor decomposition, renormalization group)
can be ported into LLM design to:

1. **Efficiency goal**: Run LLMs on smaller hardware (better compute/hardware ratio)
2. **Capability goal**: Move LLMs closer to AGI-like generalization

This repo contains prototype code and a companion Obsidian vault with all research notes.

---

## Current State (as of 2026-03-20)

### Completed

| Task | Status | Artifact |
|---|---|---|
| Fetch & summarize 9 key papers | Done | Obsidian notes (see below) |
| 5 research direction deep-dives | Done | Obsidian notes (see below) |
| Hopfield ↔ attention equivalence toy example | Done | `hopfield_attention_equivalence.py` |
| Entropy-based quantization prototype | Done | `entropy_quantization.py` |

### Immediately Next

1. **Scale entropy quantization to LLaMA-7B** — GPT-2 experiment validated that entropy-based
   allocation is 2.6x better than uniform. Next: run on LLaMA-7B (Windows GPU box) with
   GPTQ-style quantization for production-quality absolute PPL numbers.

2. **Implement lattice-structured attention on GPT-2** — replace standard attention with
   exponentially-decaying attention weights (1D lattice, nearest-neighbor kernel). Baseline:
   GPT-2 (124M) perplexity on WikiText-103. See `research/physics-llm/03-lattice-attention.md`.

3. **Read the mean-field paper** — arXiv 2512.01868 (Rigollet 2025) is the most critical
   unread paper. It has been accepted to ICM 2026; check for updated preprint.

---

## Research Directions (5 total)

### Direction 1: Thermodynamic Quantization ⭐ Highest Priority
**File**: `entropy_quantization.py`

Replace Hessian-based quantization sensitivity (OmniQuant) with **activation entropy**.
Layers with high Shannon entropy need more bits; low-entropy layers can be aggressively quantized.

- Prototype: done (toy model)
- **GPT-2 124M experiment: done** (2026-03-20) — `gpt2_entropy_quantization.py`
  - FP32 baseline PPL: 29.95
  - Uniform 4-bit PPL: 12196 | Entropy-linear 4-bit PPL: 4730 | Hessian-linear 4-bit PPL: 12196
  - **Entropy allocation is 2.6x better than uniform at same mean bits**
  - Entropy vs. Hessian correlation: Pearson r=0.024 (weak) — but Hessian scores were degenerate (near-zero diagonal Fisher on GPT-2 eval mode). Inconclusive.
  - Note: absmax quantization without scale calibration causes severe PPL degradation for all methods. The signal is in the relative comparison, not absolute PPL.
  - Plots: `results/entropy_landscape.png`, `results/entropy_vs_hessian.png`, `results/perplexity_comparison.png`
- Next: validate on LLaMA-7B (Windows GPU box), use GPTQ-style quantization for better absolute PPL
- Open question: does per-layer entropy correlate with Hessian when using a proper Fisher estimator?
- Baseline to beat: OmniQuant ICLR 2024 (arXiv 2308.13137)

### Direction 2: Renormalization-Guided Pruning
**File**: (not yet implemented)

Track how singular values evolve across transformer layers (= RG flow). Weights that
"grow" under layer-wise coarse-graining are "relevant operators" (keep); decaying ones
are "irrelevant" (prune). This gives principled pruning without iterative magnitude pruning.

- Status: theoretical design only
- Next: implement SVD tracking across LLaMA-7B layers; plot "RG flow" of singular values

### Direction 3: Lattice-Structured Attention ⭐ High Priority
**File**: (not yet implemented)

Replace full O(n²) attention with lattice Green's function decay:
`a(i,j) ∝ exp(-|i-j|/ξ)` for 1D, generalized to 2D for documents.
Related to MonarchAttention (arXiv 2505.18698, NeurIPS 2025 Spotlight) but with
a physics-motivated kernel instead of Monarch's block structure.

- Status: design done; implementation pending
- Next: implement on GPT-2 (124M), compare perplexity at same FLOP budget
- Baseline: MonarchAttention, BigBird

### Direction 4: Mean-Field Initialization
**File**: (not yet implemented)

Tokens form metastable clusters during forward passes (proven by Rigollet 2025).
Design initializations that maximize metastable state diversity during training.
Hypothesis: more diverse metastable states → better compositional reasoning.

- Status: theory only
- Prerequisite: read arXiv 2512.01868 fully
- Next: measure cluster formation empirically on an existing model

### Direction 5: Tensor Network Compression (DMRG)
**File**: (not yet implemented)

DMRG-style variational sweeps to compress LLM weight matrices, with adaptive bond
dimensions chosen by entanglement spectrum (vs. fixed-rank in TT-LoRA).

- Status: design done; implementation pending
- Next: implement DMRG sweep for 2D tensor (= SVD) as sanity check, then 4D

---

## Code Reference

### `hopfield_attention_equivalence.py`

Numerically verifies that Hopfield retrieval ≡ scaled dot-product attention.

```bash
python3 hopfield_attention_equivalence.py
```

**Key result**: max diff = 0.00e+00 (identical).
Demonstrates 3 things: core equivalence, temperature effect, multi-head = parallel Hopfield nets.

### `entropy_quantization.py`

Entropy-based adaptive bit-width allocation prototype.

```bash
python3 entropy_quantization.py
```

**Only requires numpy** (no torch needed for the prototype).

**Key classes**:
- `activation_entropy(activations, n_bins)` — Shannon entropy from histogram
- `entropy_linear_allocation(entropies, min_bits, max_bits, target_mean_bits)` — main allocation
- `uniform_allocation(layer_names, bits)` — baseline
- `absmax_quantize(weights, bits)` — simulated quantization

**Production usage**: see `print_production_usage()` in the file — shows exact HuggingFace
hook pattern for LLaMA-7B activation collection.

---

## Obsidian Vault

All research notes live in the Obsidian vault at the path configured on this machine.
The vault folder is `research/physics-llm/` and contains:

```
research/physics-llm/
├── 00-index.md                          ← master index, start here
├── 01-thermodynamic-quantization.md    ← Direction 1 detail
├── 02-rg-pruning.md                     ← Direction 2 detail
├── 03-lattice-attention.md              ← Direction 3 detail
├── 04-meanfield-initialization.md       ← Direction 4 detail
├── 05-tensor-network-compression.md     ← Direction 5 detail
├── papers/
│   ├── hopfield-2008.02217.md           ← Hopfield Networks is All You Need
│   ├── meanfield-2512.01868.md          ← Mean-Field Dynamics of Transformers
│   ├── rg-deeplearning-1410.3831.md     ← RG & Deep Learning (Mehta & Schwab)
│   ├── bitnet-2402.17764.md             ← BitNet b1.58
│   ├── mamba-2312.00752.md              ← Mamba
│   ├── omniquant-2308.13137.md          ← OmniQuant (ICLR 2024)
│   ├── tt-lora-2408.01008.md            ← TT-LoRA
│   ├── monarch-2204.00595.md            ← Monarch Matrices
│   ├── monarchattn-2505.18698.md        ← MonarchAttention (NeurIPS 2025 Spotlight)
│   └── normal-computing-cn101.md        ← Normal Computing thermodynamic ASIC
└── experiments/
    ├── hopfield-attention-toy.md         ← Results from equivalence demo
    └── entropy-quant-experiment.md       ← Results from quantization prototype
```

---

## Paper Reading Queue (Priority Order)

1. **arXiv 2512.01868** — Mean-Field Dynamics of Transformers (Rigollet) — **READ FIRST**
2. arXiv 2008.02217 — Hopfield Networks is All You Need
3. arXiv 1410.3831 — RG & Deep Learning (Mehta & Schwab)
4. arXiv 2402.17764 — BitNet b1.58
5. arXiv 2312.00752 — Mamba
6. arXiv 2308.13137 — OmniQuant (ICLR 2024)
7. arXiv 2408.01008 — TT-LoRA
8. arXiv 2204.00595 — Monarch Matrices
9. arXiv 2505.18698 — MonarchAttention (NeurIPS 2025 Spotlight)

---

## Research Questions (Open)

| Question | Direction | Status |
|---|---|---|
| Does per-layer entropy predict optimal quantization bit-width? | 1 | Prototype done; real model pending |
| Does entropy correlate with OmniQuant's Hessian sensitivity scores? | 1 | Pending |
| Do transformer layers implement RG coarse-graining (growing/decaying SVs)? | 2 | Pending |
| Does lattice-structured attention preserve perplexity at lower FLOP budget? | 3 | Pending |
| Do metastable token clusters correlate with semantic coherence? | 4 | Pending |
| Does DMRG-style optimization outperform SVD truncation for LLM compression? | 5 | Pending |

---

## Key Physics ↔ ML Mappings (Quick Reference)

| Physics concept | ML equivalent | Paper |
|---|---|---|
| Hopfield energy minimization | Transformer attention | 2008.02217 |
| Inverse temperature β | Attention scale 1/√d | 2008.02217 |
| Boltzmann distribution | Softmax | — |
| Metastable state | Attention head's "mode" | 2512.01868 |
| Phase transition | Context length collapse | 2512.01868 |
| Kuramoto synchronization | Token clustering | 2512.01868 |
| RG coarse-graining | Layer-wise abstraction | 1410.3831 |
| Relevant/irrelevant operators | Prune-safe vs. critical weights | 1410.3831 |
| Ising spin {-1, +1} | BitNet ternary weight {-1, 0, +1} | 2402.17764 |
| Langevin dynamics | Thermodynamic computing | CN101 |
| MPS bond dimension | Effective rank of weight matrix | 2408.01008 |
| Lattice Green's function | Attention decay kernel | 2505.18698 |

---

## Environment

- Python 3.x with numpy (no torch required for prototypes)
- For production experiments: `pip install transformers torch datasets`
- Obsidian vault: configured on this machine via the obsidian MCP server
