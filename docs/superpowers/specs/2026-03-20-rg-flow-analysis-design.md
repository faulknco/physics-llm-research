# RG Flow Analysis on GPT-2

**Date:** 2026-03-20
**Status:** Approved
**Direction:** 2 (Renormalization-Guided Pruning)

## Goal

Test whether transformer layers implement renormalization group (RG) coarse-graining by tracking how singular value spectra evolve across layers. Identify which singular values are "relevant operators" (grow across layers) vs "irrelevant operators" (decay) — providing a principled basis for structured pruning.

## Physics Hypothesis

In RG theory, operators are classified by how they scale under coarse-graining:
- **Relevant:** grow under RG flow (dominate at large scales)
- **Irrelevant:** decay under RG flow (negligible at large scales)
- **Marginal:** neither grow nor decay

If transformers implement RG, we should observe: some singular values of weight matrices consistently grow from layer 0 to layer 11, while others decay. The decaying ones are candidates for pruning.

## Model

GPT-2 124M (12 transformer layers, already loaded from previous experiments)

## Pipeline

### Step 1: Extract Weight Matrices

For each of the 12 transformer layers, extract the weight matrices:
- `attn.c_attn` (combined Q/K/V projection, 768x2304)
- `attn.c_proj` (output projection, 768x768)
- `mlp.c_fc` (FF up-projection, 768x3072)
- `mlp.c_proj` (FF down-projection, 3072x768)

### Step 2: Compute SVD Per Layer

For each weight matrix type across all 12 layers, compute full SVD and extract singular values. Normalize by the largest singular value in layer 0 to track relative growth/decay.

### Step 3: Track Singular Value Trajectories

For each weight type, create a matrix of shape (12 layers, k singular values). Track how each singular value index evolves across depth. Compute per-SV growth rate: `gamma_i = log(sigma_i[L-1] / sigma_i[0]) / (L-1)`.

### Step 4: Classify Operators

- **Relevant:** gamma > threshold (growing)
- **Irrelevant:** gamma < -threshold (decaying)
- **Marginal:** |gamma| < threshold

Threshold: 0.1 (one order of magnitude growth/decay across 12 layers)

### Step 5: Pruning Implication Analysis

For each weight type, report:
- Fraction of singular values that are irrelevant (prune candidates)
- Total parameter fraction that could be pruned
- Compare irrelevant SV indices with low-magnitude weight regions (sanity check)

## Output Artifacts

- `gpt2_rg_flow.py` — single analysis script
- `results/rg_flow_trajectories.png` — main plot: SV trajectories across layers, colored by classification (red=relevant, blue=irrelevant, gray=marginal)
- `results/rg_spectrum_per_layer.png` — full spectrum at each layer (heatmap)
- `results/rg_growth_rates.png` — histogram of growth rates across all weight types
- Console: classification summary table

## Dependencies

Same as existing: torch, transformers, numpy, matplotlib. No new dependencies.

## Hardware

MacBook Air M4 16GB. SVD on 768x3072 matrices is fast (~seconds). Total runtime: <1 minute.

## Success Criteria

1. Singular value trajectories show clear differentiation (not all flat)
2. At least 20% of singular values classifiable as irrelevant (decaying)
3. Growth rate distribution is bimodal or at least non-trivial (not Gaussian centered at 0)
