# GPTQ-Calibrated Entropy Quantization

**Date:** 2026-03-20
**Status:** Approved
**Direction:** 1 (Thermodynamic Quantization) — improved quantizer

## Goal

Replace naive absmax quantization with GPTQ-style layer-wise calibrated quantization, then re-run entropy vs uniform comparison. GPTQ compensates for quantization error using the Hessian, dramatically reducing perplexity degradation.

## Why

Our entropy allocation experiment showed 2.6x relative improvement over uniform, but absolute PPL was terrible (4730 vs 30 FP32) because absmax quantization is crude. GPTQ should bring the quantized PPL much closer to FP32, making the entropy vs uniform comparison meaningful at production quality.

## Algorithm: Minimal GPTQ

For each linear layer with weight W (shape: rows x cols) and calibration input X:

1. Compute Hessian: `H = 2 * X.T @ X` (column-wise sensitivity)
2. Cholesky decompose: `H_inv = cholesky_inv(H + diag_dampening)`
3. For each column j (in order):
   - Quantize: `w_q[j] = quantize(W[:, j], bits)`
   - Compute error: `err = (W[:, j] - w_q[j]) / H_inv[j, j]`
   - Compensate remaining columns: `W[:, j+1:] -= err @ H_inv[j, j+1:]`
4. Replace W with quantized W

This is the core GPTQ loop from Frantar et al. (2022).

## Pipeline

1. Load GPT-2 124M
2. Collect calibration activations (128 samples, WikiText-2 train)
3. Compute per-layer entropy (reuse streaming histograms)
4. Build entropy-linear and uniform-4bit allocation plans
5. For each plan: apply GPTQ quantization per layer at allocated bits
6. Measure perplexity on WikiText-2 test
7. Compare: FP32 vs GPTQ-uniform-4bit vs GPTQ-entropy-4bit

## Expected Results

- GPTQ-uniform-4bit: PPL ~35-45 (vs 12196 with absmax)
- GPTQ-entropy-4bit: PPL ~32-40 (should still beat uniform)
- The relative advantage of entropy allocation should persist

## Output

- `gpt2_gptq_entropy.py` — single script
- `results/gptq_perplexity_comparison.png`
- Console: comparison table

## Dependencies

Same as existing. No new packages needed — we implement GPTQ from scratch (~50 lines).
