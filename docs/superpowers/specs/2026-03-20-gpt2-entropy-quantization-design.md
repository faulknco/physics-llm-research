# GPT-2 Entropy Quantization with Hessian Correlation

**Date:** 2026-03-20
**Status:** Approved
**Direction:** 1 (Thermodynamic Quantization)

## Goal

Validate whether per-layer activation entropy predicts optimal quantization bit-width allocation on a real model (GPT-2 124M), and test whether entropy correlates with Hessian-based sensitivity — the approach used by OmniQuant (ICLR 2024).

## Hypothesis

Layers with high activation entropy carry more information and need more bits to preserve quality. If entropy correlates with Hessian sensitivity, it provides a cheaper proxy (one forward pass vs. gradient computation) for the same signal.

## Model & Data

- **Model:** GPT-2 124M (`openai-community/gpt2`) — FP32, ~500MB
- **Calibration data:** 128 samples from WikiText-2 train split, sequence length 1024
- **Evaluation data:** WikiText-2 test split, full perplexity evaluation

## Pipeline

### Step 1: Activation Collection

Register forward hooks on all `nn.Linear` modules in GPT-2. During the calibration pass (128 samples), maintain a **streaming histogram** per layer (256 bins) rather than storing raw activations — raw storage would exceed 50GB. Each sample's activations are binned and accumulated, then discarded. Bin range is determined by a quick first pass over 8 samples to find [1st, 99th] percentile bounds per layer.

### Step 2: Entropy Computation

For each layer, compute Shannon entropy from the accumulated histogram (256 bins, percentile-clipped range from Step 1). Reuse `activation_entropy()` from existing `entropy_quantization.py`. The percentile clipping prevents outliers from stretching the bin range and artificially depressing entropy.

### Step 3: Hessian Sensitivity Estimation

Compute diagonal Fisher approximation per layer: `H_diag(l) = E[(dL/dW_l)^2]` using the standard **causal language modeling cross-entropy loss** (next-token prediction) on the calibration data.

**Memory strategy:** Process one sample at a time. For each sample, run forward + backward, then accumulate `grad.pow(2)` into a running sum per layer. Zero gradients after each sample. Never store more than one sample's gradients at a time. Summarize as `running_sum / N` per layer — this is the "sensitivity score" that OmniQuant-style methods use.

### Step 4: Bit-Width Allocation

Three allocation strategies, all targeting 4.0 bits average:

1. **Uniform-4bit** — baseline, all layers get 4 bits
2. **Entropy-linear** — linear map from entropy to [2, 8] bits, targeting 4.0 mean
3. **Hessian-linear** — same algorithm but using Hessian sensitivity instead of entropy

After rounding bit-widths to integers, apply a **greedy adjustment step**: iteratively bump the layer closest to a rounding boundary up or down by 1 bit until the actual mean matches the target. Report the actual achieved mean bits in all results.

### Step 5: Quantization

Apply symmetric absmax quantization per layer at the allocated bit-width. Replace FP32 weights with quantized-then-dequantized weights in-place.

### Step 6: Perplexity Evaluation

Evaluate each quantized model on WikiText-2 test set. Report perplexity. Also report FP32 baseline perplexity.

### Step 7: Correlation Analysis

- Scatter plot: per-layer entropy (x) vs. Hessian sensitivity (y)
- Compute Pearson and Spearman correlation coefficients
- If r > 0.7, entropy is a viable cheap proxy for Hessian sensitivity

### Step 8: Results Comparison

Final table:

| Method | Avg Bits | Perplexity | Delta vs FP32 |
|--------|----------|------------|----------------|
| FP32 baseline | 32 | ? | 0 |
| Uniform 4-bit | 4.0 | ? | ? |
| Entropy-linear | 4.0 | ? | ? |
| Hessian-linear | 4.0 | ? | ? |
| spin-quant k-means (external ref, not directly comparable) | ~4.0 | ? | ? |

## Output Artifacts

- `gpt2_entropy_quantization.py` — single script, runnable end-to-end
- Console output: entropy landscape, correlation stats, perplexity table
- Matplotlib plots saved to `results/` directory:
  - `entropy_landscape.png` — per-layer entropy bar chart
  - `entropy_vs_hessian.png` — correlation scatter plot
  - `perplexity_comparison.png` — bar chart of methods

## Dependencies

```
torch
transformers
datasets
numpy
matplotlib
```

## Hardware

- MacBook Air M4 16GB — GPT-2 124M fits comfortably in FP32
- Estimated runtime: 10-20 minutes for full pipeline (calibration + Hessian + eval)

## Success Criteria

1. Entropy-linear allocation achieves lower perplexity than uniform-4bit
2. Entropy vs. Hessian correlation r > 0.7 (strong enough to be a viable cheap proxy)
3. Results are reproducible (seeded: `torch.manual_seed(42)`, `np.random.seed(42)`)

## Non-Goals

- Scaling to LLaMA-7B (future work, Windows GPU box)
- Training or fine-tuning
- Advanced quantization methods (GPTQ, AWQ) — we use simple absmax to isolate the bit-allocation signal
