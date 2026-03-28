# Physics & Math-Inspired LLM Architectures — Research Repo

## What This Is

An active research program exploring whether structures from physics (quantum lattices,
thermodynamics, spin systems) and mathematics (tensor decomposition, renormalization group)
can be ported into LLM design to:

1. **Efficiency goal**: Run LLMs on smaller hardware (better compute/hardware ratio)
2. **Capability goal**: Move LLMs closer to AGI-like generalization

This repo contains prototype code and a companion Obsidian vault with all research notes.

---

## Current State (as of 2026-03-28)

### Completed Experiments

| # | Experiment | Script | Key Result |
|---|-----------|--------|------------|
| 1 | Hopfield ↔ attention equivalence | `hopfield_attention_equivalence.py` | max diff = 0.00e+00 (identical) |
| 2 | Entropy quantization prototype | `entropy_quantization.py` | Toy model validates approach |
| 3 | GPT-2 entropy quantization | `gpt2_entropy_quantization.py` | **Entropy 2.6x better than uniform (absmax)** |
| 4 | Hessian fix + correlation | (same as #3) | r=-0.006: entropy & Hessian are orthogonal signals |
| 5 | RG flow analysis | `gpt2_rg_flow.py` | **attn.c_proj 65% "relevant", mlp.c_fc fixed point** |
| 6 | Lattice attention | `gpt2_lattice_attention.py` | **Phase transition at ξ~512 tokens** |
| 7 | Combined compression | `gpt2_combined_compression.py` | Quant+pruning superadditive, lattice interferes with degraded weights |
| 8 | GPTQ-calibrated entropy | `gpt2_gptq_entropy.py` | **GPTQ uniform (985) beats entropy (1500)** — Hessian compensation makes entropy redundant |
| 9 | Metastable cluster analysis | `gpt2_cluster_analysis.py` | **Layers 0-11 maintain ~125 clusters, layer 12 collapses to 6** |
| 10 | KV cache entropy quantization | `gpt2_kv_entropy_quantization.py` | **Entropy uniform in KV tensors — fails for KV allocation; per-head asymmetric breaks model (+290% PPL)** |
| 11 | RMT / GUE attention head analysis | `gpt2_rmt_head_analysis.py` | **Attention scores are Poisson not GUE; MP signal valid for pruning not KV; RMT⊥RG r=0.233** |
| A1 | Gamma-distribution quantizer | `gpt2_gamma_quantizer.py` | **Gamma Lloyd-Max: PPL 230 vs uniform 12196 (53x improvement); k=1.405 validates Dyson BM** |
| A1+ | Gamma + GPTQ | `gpt2_gamma_gptq.py` | **GPTQ-gamma (281) worse than absmax-gamma (230) — grid+compensation are coupled, not orthogonal** |

### Immediately Next (Agreed Roadmap 2026-03-28)

Full roadmap in Obsidian: `research/physics-llm/research-roadmap.md`

1. **Wait for exp 11 (RMT)** — Running on Windows RTX 2060, script: `gpt2_rmt_head_analysis.py`
2. **A1: Gamma-distribution quantizer** — Lloyd-Max grid matched to gamma (Dyson BM prediction); ~1 day; `gpt2_gamma_quantizer.py`
3. **B1: RMT-guided structured pruning** — Prune GUE-conforming (noise) heads entirely; builds on exp 11
4. **A2: Joint entropy+Hessian LP allocator** — Orthogonal signals (r=-0.006) → joint LP beats either alone
5. **C3: Spectral gap → adaptive KV cache** — Spectral gap per head determines context range needed
6. **B5: DMRG tensor compression** — Direction 5, last unexplored; use RMT-filtered bond dimensions
7. **LLaMA-7B scale validation** — Everything above on 7B; findings that hold are publishable

---

## Research Directions (5 total)

### Direction 1: Thermodynamic Quantization ⭐
**Files**: `entropy_quantization.py`, `gpt2_entropy_quantization.py`, `gpt2_gptq_entropy.py`

Replace Hessian-based quantization sensitivity (OmniQuant) with **activation entropy**.

**GPT-2 Results (absmax quantization):**
- FP32 baseline PPL: 29.95
- Uniform 4-bit: 12,196 | **Entropy-linear 4-bit: 4,730** (2.6x better)
- Entropy vs. Hessian correlation: r = -0.006 (genuinely orthogonal signals)

**GPT-2 Results (GPTQ quantization):**
- GPTQ uniform 4-bit: **985** | GPTQ entropy [2,8]: 2,677 | GPTQ entropy [3,6]: 1,500
- **GPTQ's Hessian compensation supersedes entropy** — uniform wins with GPTQ
- Entropy is most valuable in the low-sophistication quantization regime

**Key insight:** Entropy and Hessian measure orthogonal things (information content vs loss sensitivity). Entropy allocation is a cheap proxy that works great without GPTQ, but GPTQ's own error correction handles sensitivity already.

**Plots:** `results/entropy_landscape.png`, `results/entropy_vs_hessian.png`, `results/perplexity_comparison.png`, `results/gptq_perplexity_comparison.png`

### Direction 2: Renormalization-Guided Pruning ⭐
**File**: `gpt2_rg_flow.py`

Track SVD evolution across transformer layers = RG flow. Classify singular values as relevant (growing), irrelevant (decaying), or marginal (fixed point).

**GPT-2 Results:**
| Weight Type | Classification | Mean γ | Pruning Safety |
|-------------|---------------|--------|----------------|
| attn.c_proj | **65% relevant** (growing) | +0.108 | **Dangerous to prune** |
| mlp.c_proj | Marginal (almost relevant) | +0.077 | Moderate risk |
| attn.c_attn | Marginal (slight decay) | -0.027 | Safer to prune |
| mlp.c_fc | **Fixed point** | -0.008 | **Safest to prune** |

**Key insight:** Output projections are "relevant operators" that amplify signal through the residual stream. Input projections are near fixed points — safe targets for low-rank approximation. Zero singular values are strongly decaying (GPT-2 doesn't waste parameters).

**Plots:** `results/rg_flow_trajectories.png`, `results/rg_spectrum_per_layer.png`, `results/rg_growth_rates.png`

### Direction 3: Lattice-Structured Attention ⭐
**File**: `gpt2_lattice_attention.py`

Apply physics-motivated decay mask exp(-|i-j|/ξ) to attention scores before softmax. Sweep correlation length ξ.

**GPT-2 Results:**
| ξ (tokens) | PPL | Status |
|-----------|-----|--------|
| 1024 | 30.20 | Essentially vanilla (+0.8%) |
| 512 | 32.47 | Mild degradation (+8.4%) |
| **256** | **68.20** | **Phase transition (+128%)** |
| 128 | 406 | Broken |
| 64 | 2,391 | Catastrophic |
| 16 | 38,678 | Complete collapse |

**Key insight:** Sharp phase transition at ξ~256-512. GPT-2 needs attention to reach ~512 tokens to function. Below that, it's like cutting the correlation length below the critical point in a spin system. This matches Rigollet 2025's predicted phase transition for long-context attention.

**Plots:** `results/lattice_attention_ppl_vs_xi.png`, `results/lattice_attention_pattern.png`

### Direction 4: Mean-Field Initialization
**Files**: `gpt2_cluster_analysis.py`

Tokens form metastable clusters during forward passes (Rigollet 2025). Design initializations that maximize metastable state diversity.

**Paper read:** arXiv 2512.01868 (Rigollet 2025, ICM 2026). Key predictions:
- Tokens cluster progressively across layers (confirmed)
- Pre-LN gives polynomial 1/t² collapse rate (NOT confirmed on GPT-2 — residual connections fight collapse)
- Metastable multi-cluster states before final collapse (confirmed)

**GPT-2 Cluster Analysis Results:**
- Layers 0-11: **~100-125 clusters maintained** (long metastable plateau)
- Layer 12: **catastrophic collapse to ~6 clusters** (single cliff, not multi-step staircase)
- Max cluster diversity at layer 7 (~125 clusters)
- Pre-LN polynomial fit: R² = -611 (fails — GPT-2 actively maintains diversity until final layer)

**Key insight:** GPT-2 preserves representational diversity for 11 layers then collapses everything in one shot at layer 12. The metastable state IS the useful computation. Controlling which ~6 clusters survive the final collapse could improve model quality.

**Plots:** `results/cluster_cosine_similarity.png`, `results/cluster_count_vs_depth.png`, `results/cluster_similarity_heatmap.png`

### Direction 5: Tensor Network Compression (DMRG)
**File**: (not yet implemented)

DMRG-style variational sweeps to compress LLM weight matrices, with adaptive bond dimensions chosen by entanglement spectrum (vs. fixed-rank in TT-LoRA).

- Status: design done; implementation pending
- Next: implement DMRG sweep for 2D tensor (= SVD) as sanity check, then 4D

---

## Combined Compression Experiment
**File**: `gpt2_combined_compression.py`

Applied all three physics principles simultaneously:

| Method | PPL | Compression |
|--------|-----|-------------|
| FP32 baseline | 29.95 | 1.0x |
| Lattice attn (ξ=512) | 32.47 | 1.0x |
| RG pruning (50% rank) | 2,130 | 1.17x |
| Entropy quant (4-bit) | 4,730 | 2.48x |
| Entropy + RG | 5,515 | 2.63x |
| All three combined | 6,933 | 2.63x |

**Key findings:**
- Entropy + RG pruning are slightly **superadditive** (combining is less bad than sum of individual costs)
- Lattice attention **interferes** with degraded weights (subadditive when added to quant+pruning)
- All methods need GPTQ-style calibration for production-quality results

**Plots:** `results/combined_compression_table.png`, `results/combined_pareto.png`

---

## Big Picture Findings

1. **Entropy knows where the information is** — 2.6x better than uniform allocation (absmax regime)
2. **RG flow reveals pruning structure** — output projections are relevant operators, input projections are safe to prune
3. **Attention has a critical correlation length** — phase transition at ξ~512 tokens
4. **Entropy and Hessian are orthogonal** — r=-0.006, they measure different things
5. **GPTQ makes entropy redundant** — its Hessian compensation already handles layer sensitivity
6. **GPT-2 maintains diversity then collapses at layer 12** — metastable plateau confirmed, but single cliff not staircase

---

## Code Reference

| Script | Direction | What it does |
|--------|-----------|-------------|
| `hopfield_attention_equivalence.py` | Foundation | Proves Hopfield ≡ attention |
| `entropy_quantization.py` | 1 | Entropy-based bit allocation prototype (numpy only) |
| `gpt2_entropy_quantization.py` | 1 | Full pipeline: entropy + Hessian + perplexity on GPT-2 |
| `gpt2_gptq_entropy.py` | 1 | GPTQ-calibrated entropy quantization |
| `gpt2_rg_flow.py` | 2 | SVD trajectory analysis across layers |
| `gpt2_lattice_attention.py` | 3 | Lattice decay mask, ξ sweep |
| `gpt2_cluster_analysis.py` | 4 | Metastable cluster measurement, Rigollet verification |
| `gpt2_combined_compression.py` | 1+2+3 | All three compressions combined |
| `gpt2_kv_entropy_quantization.py` | 1 ext | KV cache entropy quantization (exp 10) |
| `gpt2_rmt_head_analysis.py` | 6 | RMT/GUE attention head analysis (exp 11) |

---

## Obsidian Vault

All research notes live in `research/physics-llm/` in the Obsidian vault:

```
research/physics-llm/
├── 00-index.md
├── 01-thermodynamic-quantization.md
├── 02-rg-pruning.md
├── 03-lattice-attention.md
├── 04-meanfield-initialization.md
├── 05-tensor-network-compression.md
├── big-picture-findings.md              ← summary of all experimental results
├── papers/                              ← 10 paper summaries
│   ├── meanfield-2512.01868.md          ← Rigollet 2025 (fully read & summarized)
│   └── ... (9 more)
└── experiments/
    ├── hopfield-attention-toy.md
    ├── entropy-quant-experiment.md       ← GPT-2 entropy quantization results
    └── lattice-attention-experiment.md   ← GPT-2 lattice attention results
```

---

## Paper Reading Queue

1. ~~arXiv 2512.01868 — Mean-Field Dynamics (Rigollet)~~ **READ** — ICM 2026
2. arXiv 2008.02217 — Hopfield Networks is All You Need
3. arXiv 1410.3831 — RG & Deep Learning (Mehta & Schwab)
4. arXiv 2402.17764 — BitNet b1.58
5. arXiv 2312.00752 — Mamba
6. arXiv 2308.13137 — OmniQuant (ICLR 2024)
7. arXiv 2408.01008 — TT-LoRA
8. arXiv 2204.00595 — Monarch Matrices
9. arXiv 2505.18698 — MonarchAttention (NeurIPS 2025 Spotlight)

---

## Research Questions

| Question | Direction | Status |
|---|---|---|
| Does per-layer entropy predict optimal quantization bit-width? | 1 | **YES** — 2.6x better than uniform (absmax) |
| Does entropy correlate with Hessian sensitivity? | 1 | **NO** — r=-0.006, genuinely orthogonal |
| Does entropy help with GPTQ? | 1 | **NO** — GPTQ's Hessian compensation supersedes it |
| Do transformer layers implement RG coarse-graining? | 2 | **PARTIALLY** — output projections grow, inputs are marginal |
| Does lattice attention preserve perplexity? | 3 | **YES above ξ~512** — phase transition below |
| Do metastable token clusters form across layers? | 4 | **YES** — plateau at ~125 clusters, collapse at layer 12 |
| Does Pre-LN give 1/t² collapse? | 4 | **NO** — residual connections fight collapse |
| Does DMRG outperform SVD truncation? | 5 | Pending |

---

## Key Physics ↔ ML Mappings (Quick Reference)

| Physics concept | ML equivalent | Paper | Verified? |
|---|---|---|---|
| Hopfield energy minimization | Transformer attention | 2008.02217 | ✓ (exp 1) |
| Inverse temperature β | Attention scale 1/√d | 2008.02217 | ✓ |
| Boltzmann distribution | Softmax | — | — |
| Metastable state | Multi-cluster token representation | 2512.01868 | ✓ (exp 9) |
| Phase transition | Attention correlation length collapse | 2512.01868 | ✓ (exp 6) |
| Kuramoto synchronization | Token clustering across layers | 2512.01868 | ✓ (exp 9) |
| RG coarse-graining | Layer-wise SVD evolution | 1410.3831 | ✓ (exp 5) |
| Relevant/irrelevant operators | Prune-safe vs. critical weights | 1410.3831 | ✓ (exp 5) |
| Shannon entropy | Quantization sensitivity proxy | — | ✓ (exp 3) |
| Lattice Green's function | Attention decay kernel | 2505.18698 | ✓ (exp 6) |
| MPS bond dimension | Effective rank of weight matrix | 2408.01008 | Pending |

---

## Environment

- Python 3.x, managed by `uv`
- Dependencies: `requirements.txt` (torch, transformers, datasets, numpy, matplotlib, scipy)
- Obsidian vault: configured on this machine via the obsidian MCP server
- cmux workspace shortcut: `cp-llm`
