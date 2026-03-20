# GPT-2 Entropy Quantization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Validate entropy-based quantization bit-width allocation on real GPT-2 124M, and test whether activation entropy correlates with Hessian sensitivity.

**Architecture:** Single script `gpt2_entropy_quantization.py` with a streaming calibration pipeline (forward hooks, histogram accumulation, entropy/Hessian computation, bit allocation, quantize, perplexity measurement). Reuses `QuantizationPlan`, `absmax_quantize`, and allocation functions from existing `entropy_quantization.py`.

**Tech Stack:** PyTorch, HuggingFace transformers, datasets (WikiText-2), numpy, matplotlib

**Spec:** `docs/superpowers/specs/2026-03-20-gpt2-entropy-quantization-design.md`

---

## File Structure

| File | Purpose |
|------|---------|
| `gpt2_entropy_quantization.py` (create) | Main experiment script: calibration, entropy, Hessian, quantization, perplexity |
| `entropy_quantization.py` (modify) | Add greedy bit-adjustment to enforce exact target mean after rounding |
| `results/` (create dir) | Output plots directory |

---

### Task 1: Install Dependencies and Verify GPT-2 Loads

**Files:**
- Create: `requirements.txt`

- [ ] **Step 1: Create requirements.txt**

```
torch
transformers
datasets
numpy
matplotlib
scipy
```

- [ ] **Step 2: Install dependencies**

Run: `cd ~/Projects/physics-llm-research && uv venv && uv pip install -r requirements.txt`
Expected: All packages install successfully

- [ ] **Step 3: Verify GPT-2 loads on M4**

Run:
```bash
cd ~/Projects/physics-llm-research && uv run python -c "
from transformers import GPT2LMHeadModel, GPT2Tokenizer
model = GPT2LMHeadModel.from_pretrained('openai-community/gpt2')
tok = GPT2Tokenizer.from_pretrained('openai-community/gpt2')
print(f'Model params: {sum(p.numel() for p in model.parameters()):,}')
print('OK')
"
```
Expected: `Model params: 124,439,808`, `OK`

- [ ] **Step 4: Commit**

```bash
git add requirements.txt
git commit -m "chore: add requirements for GPT-2 entropy quantization experiment"
```

---

### Task 2: Add Greedy Bit-Adjustment to entropy_quantization.py

**Files:**
- Modify: `entropy_quantization.py:100-141` (the `entropy_linear_allocation` function)

- [ ] **Step 1: Add the adjustment function**

Add this function before `entropy_linear_allocation` in `entropy_quantization.py`:

```python
def enforce_target_mean(assigned_bits: np.ndarray, target_mean: float,
                        min_bits: int, max_bits: int) -> np.ndarray:
    """
    Greedy adjustment: bump the layer with the lowest (or highest) bit
    count up (or down) by 1 until the mean matches the target.
    """
    bits = assigned_bits.copy()
    target_sum = round(target_mean * len(bits))
    while int(bits.sum()) != target_sum:
        diff = target_sum - int(bits.sum())
        if diff > 0:
            candidates = np.where(bits < max_bits)[0]
            if len(candidates) == 0:
                break
            idx = candidates[np.argmin(bits[candidates])]
            bits[idx] += 1
        else:
            candidates = np.where(bits > min_bits)[0]
            if len(candidates) == 0:
                break
            idx = candidates[np.argmax(bits[candidates])]
            bits[idx] -= 1
    return bits
```

- [ ] **Step 2: Wire it into entropy_linear_allocation**

Insert the following code **between** line 135 (`assigned_bits = np.clip(...)`) and line 137 (`return QuantizationPlan(...)`) in `entropy_linear_allocation`:

```python
    if target_mean_bits is not None:
        assigned_bits = enforce_target_mean(assigned_bits, target_mean_bits,
                                            min_bits, max_bits)
```

The return statement stays unchanged — this is an insertion, not a replacement.

- [ ] **Step 3: Verify existing prototype still runs**

Run: `cd ~/Projects/physics-llm-research && uv run python entropy_quantization.py`
Expected: Output matches previous behavior, mean bits closer to 4.0

- [ ] **Step 4: Commit**

```bash
git add entropy_quantization.py
git commit -m "feat: add greedy bit-adjustment to enforce exact target mean bits"
```

---

### Task 3: Scaffold gpt2_entropy_quantization.py with Data Loading

**Files:**
- Create: `gpt2_entropy_quantization.py`

- [ ] **Step 1: Create the script with imports and data loading**

```python
"""
GPT-2 Entropy Quantization Experiment
======================================

Validates entropy-based adaptive bit-width allocation on GPT-2 124M.
Compares entropy-linear vs. uniform-4bit vs. Hessian-linear allocation.
Tests correlation between activation entropy and Hessian sensitivity.

Spec: docs/superpowers/specs/2026-03-20-gpt2-entropy-quantization-design.md
"""

import os
import copy
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from datasets import load_dataset
from entropy_quantization import (
    activation_entropy,
    entropy_linear_allocation,
    uniform_allocation,
    absmax_quantize,
    QuantizationPlan,
    enforce_target_mean,
)

SEED = 42
CALIBRATION_SAMPLES = 128
SEQ_LEN = 1024
RESULTS_DIR = "results"


def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)


def load_model_and_tokenizer():
    print("Loading GPT-2 124M...")
    model = GPT2LMHeadModel.from_pretrained("openai-community/gpt2")
    tokenizer = GPT2Tokenizer.from_pretrained("openai-community/gpt2")
    model.eval()
    return model, tokenizer


def load_wikitext2(tokenizer, split="train", n_samples=CALIBRATION_SAMPLES):
    print(f"Loading WikiText-2 ({split}, {n_samples} samples, seq_len={SEQ_LEN})...")
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
    text = "\n\n".join(dataset["text"])
    tokens = tokenizer.encode(text)
    samples = []
    for i in range(n_samples):
        start = i * SEQ_LEN
        end = start + SEQ_LEN
        if end > len(tokens):
            break
        samples.append(torch.tensor(tokens[start:end], dtype=torch.long).unsqueeze(0))
    print(f"  Prepared {len(samples)} samples")
    return samples


if __name__ == "__main__":
    set_seed(SEED)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    model, tokenizer = load_model_and_tokenizer()
    cal_samples = load_wikitext2(tokenizer, split="train", n_samples=CALIBRATION_SAMPLES)
    print(f"Calibration: {len(cal_samples)} samples ready")
```

- [ ] **Step 2: Verify it runs**

Run: `cd ~/Projects/physics-llm-research && uv run python gpt2_entropy_quantization.py`
Expected: Downloads GPT-2 + WikiText-2 (first run), prints sample counts, exits cleanly

- [ ] **Step 3: Commit**

```bash
git add gpt2_entropy_quantization.py
git commit -m "feat: scaffold GPT-2 experiment with model and data loading"
```

---

### Task 4: Streaming Entropy Collection

**Files:**
- Modify: `gpt2_entropy_quantization.py`

- [ ] **Step 1: Add the streaming histogram collector**

Add after `load_wikitext2`:

```python
def collect_entropies(model, samples, n_bins=256, range_estimation_samples=8):
    """
    Collect per-layer activation entropy via streaming histograms.
    Two passes:
    1. Quick pass to find [1st, 99th] percentile per layer
    2. Full pass accumulating histograms within those bounds
    """
    linear_names = []
    linear_modules = []
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            linear_names.append(name)
            linear_modules.append(module)

    print(f"Collecting entropy from {len(linear_names)} linear layers...")

    # Pass 1: estimate activation ranges
    print("  Pass 1: estimating activation ranges...")
    layer_mins = {n: float("inf") for n in linear_names}
    layer_maxs = {n: float("-inf") for n in linear_names}

    def range_hook_fn(module_name):
        def hook(module, input, output):
            x = input[0].detach().float()
            lo = float(x.quantile(0.01))
            hi = float(x.quantile(0.99))
            if lo < layer_mins[module_name]:
                layer_mins[module_name] = lo
            if hi > layer_maxs[module_name]:
                layer_maxs[module_name] = hi
        return hook

    hooks = []
    for name, module in zip(linear_names, linear_modules):
        hooks.append(module.register_forward_hook(range_hook_fn(name)))

    with torch.no_grad():
        for i in range(min(range_estimation_samples, len(samples))):
            model(samples[i])

    for h in hooks:
        h.remove()

    # Pass 2: accumulate histograms
    print(f"  Pass 2: accumulating histograms over {len(samples)} samples...")
    histograms = {n: np.zeros(n_bins, dtype=np.float64) for n in linear_names}

    def hist_hook_fn(module_name):
        def hook(module, input, output):
            x = input[0].detach().float().cpu().numpy().flatten()
            lo, hi = layer_mins[module_name], layer_maxs[module_name]
            counts, _ = np.histogram(x, bins=n_bins, range=(lo, hi))
            histograms[module_name] += counts.astype(np.float64)
        return hook

    hooks = []
    for name, module in zip(linear_names, linear_modules):
        hooks.append(module.register_forward_hook(hist_hook_fn(name)))

    with torch.no_grad():
        for i, sample in enumerate(samples):
            model(sample)
            if (i + 1) % 32 == 0:
                print(f"    {i+1}/{len(samples)} samples processed")

    for h in hooks:
        h.remove()

    # Compute entropy from accumulated histograms
    entropies = {}
    for name in linear_names:
        counts = histograms[name]
        total = counts.sum()
        if total == 0:
            entropies[name] = 0.0
            continue
        probs = counts / total
        probs = probs[probs > 0]
        entropies[name] = float(-np.sum(probs * np.log(probs)))

    print(f"  Entropy range: [{min(entropies.values()):.4f}, {max(entropies.values()):.4f}]")
    return entropies, linear_names
```

- [ ] **Step 2: Wire it into main**

Replace the `if __name__ == "__main__"` block:

```python
if __name__ == "__main__":
    set_seed(SEED)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    model, tokenizer = load_model_and_tokenizer()
    cal_samples = load_wikitext2(tokenizer, split="train", n_samples=CALIBRATION_SAMPLES)

    # Collect entropies
    entropies, linear_names = collect_entropies(model, cal_samples)

    print("\nPer-layer entropies (first 10):")
    for name in linear_names[:10]:
        print(f"  {name:<50s} {entropies[name]:.4f}")
```

- [ ] **Step 3: Run and verify entropy values**

Run: `cd ~/Projects/physics-llm-research && uv run python gpt2_entropy_quantization.py`
Expected: Entropy values between ~2.0 and ~6.0 (nats). No OOM.

- [ ] **Step 4: Commit**

```bash
git add gpt2_entropy_quantization.py
git commit -m "feat: streaming histogram entropy collection for GPT-2"
```

---

### Task 5: Hessian Sensitivity Estimation

**Files:**
- Modify: `gpt2_entropy_quantization.py`

- [ ] **Step 1: Add the Hessian estimation function**

Add after `collect_entropies`:

```python
def compute_hessian_sensitivity(model, samples):
    """
    Diagonal Fisher approximation: H_diag(l) = E[(dL/dW_l)^2].
    Uses causal LM cross-entropy loss. Processes one sample at a time.
    """
    linear_names = []
    linear_params = {}
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            linear_names.append(name)
            linear_params[name] = module.weight

    print(f"Computing Hessian sensitivity for {len(linear_names)} layers...")

    hessian_accum = {name: torch.zeros_like(p) for name, p in linear_params.items()}
    n_processed = 0

    for i, sample in enumerate(samples):
        model.zero_grad()
        outputs = model(sample, labels=sample)
        loss = outputs.loss
        loss.backward()

        for name, param in linear_params.items():
            if param.grad is not None:
                hessian_accum[name] += param.grad.pow(2).detach()

        model.zero_grad()
        n_processed += 1

        if (i + 1) % 32 == 0:
            print(f"    {i+1}/{len(samples)} samples processed")

    sensitivities = {}
    for name in linear_names:
        sensitivities[name] = float((hessian_accum[name] / n_processed).mean().item())

    print(f"  Sensitivity range: [{min(sensitivities.values()):.6f}, {max(sensitivities.values()):.6f}]")
    return sensitivities
```

- [ ] **Step 2: Wire into main**

Add after entropy collection in `__main__`:

```python
    # Hessian sensitivity
    sensitivities = compute_hessian_sensitivity(model, cal_samples)

    print("\nPer-layer Hessian sensitivity (first 10):")
    for name in linear_names[:10]:
        print(f"  {name:<50s} {sensitivities[name]:.8f}")
```

- [ ] **Step 3: Run and verify**

Run: `cd ~/Projects/physics-llm-research && uv run python gpt2_entropy_quantization.py`
Expected: Sensitivity values printed, no OOM.

- [ ] **Step 4: Commit**

```bash
git add gpt2_entropy_quantization.py
git commit -m "feat: diagonal Fisher Hessian sensitivity estimation"
```

---

### Task 6: Bit-Width Allocation and Quantization

**Files:**
- Modify: `gpt2_entropy_quantization.py`

- [ ] **Step 1: Add Hessian-linear allocation function**

Add after imports:

```python
def hessian_linear_allocation(sensitivities, min_bits=2, max_bits=8,
                               target_mean_bits=None):
    """Same as entropy_linear_allocation but using Hessian sensitivity."""
    names = list(sensitivities.keys())
    values = np.array([sensitivities[n] for n in names])

    v_min, v_max = values.min(), values.max()
    if v_max == v_min:
        normalized = np.ones(len(names)) * 0.5
    else:
        normalized = (values - v_min) / (v_max - v_min)

    raw_bits = min_bits + normalized * (max_bits - min_bits)

    if target_mean_bits is not None:
        current_mean = raw_bits.mean()
        if current_mean > 0:
            scale = target_mean_bits / current_mean
            raw_bits = np.clip(raw_bits * scale, min_bits, max_bits)

    assigned_bits = np.round(raw_bits).astype(int)
    assigned_bits = np.clip(assigned_bits, min_bits, max_bits)

    if target_mean_bits is not None:
        assigned_bits = enforce_target_mean(assigned_bits, target_mean_bits,
                                              min_bits, max_bits)

    return QuantizationPlan(
        layer_bits={name: int(b) for name, b in zip(names, assigned_bits)},
        strategy="hessian-linear",
        mean_bits=float(assigned_bits.mean()),
    )
```

- [ ] **Step 2: Add quantization application function**

```python
def apply_quantization_plan(model, plan):
    """Apply absmax quantization to model weights in-place. Returns new model."""
    quantized_model = copy.deepcopy(model)
    for name, module in quantized_model.named_modules():
        if isinstance(module, torch.nn.Linear) and name in plan.layer_bits:
            bits = plan.layer_bits[name]
            w = module.weight.data.cpu().numpy()
            w_q, mse = absmax_quantize(w, bits)
            module.weight.data = torch.tensor(w_q, dtype=module.weight.dtype)
    return quantized_model
```

- [ ] **Step 3: Wire allocations into main**

Add after Hessian computation in `__main__`:

```python
    # Build allocation plans
    TARGET_BITS = 4.0
    plans = {
        "uniform-4bit": uniform_allocation(linear_names, bits=4),
        "entropy-linear": entropy_linear_allocation(
            entropies, min_bits=2, max_bits=8, target_mean_bits=TARGET_BITS
        ),
        "hessian-linear": hessian_linear_allocation(
            sensitivities, min_bits=2, max_bits=8, target_mean_bits=TARGET_BITS
        ),
    }

    print("\nBit-width allocation plans:")
    for plan_name, plan in plans.items():
        bits_vals = list(plan.layer_bits.values())
        print(f"  {plan_name}: mean={plan.mean_bits:.2f}, "
              f"min={min(bits_vals)}, max={max(bits_vals)}")
```

- [ ] **Step 4: Run and verify plans**

Run: `cd ~/Projects/physics-llm-research && uv run python gpt2_entropy_quantization.py`
Expected: Three plans with mean bits ~4.0

- [ ] **Step 5: Commit**

```bash
git add gpt2_entropy_quantization.py
git commit -m "feat: bit-width allocation plans (entropy, Hessian, uniform)"
```

---

### Task 7: Perplexity Measurement

**Files:**
- Modify: `gpt2_entropy_quantization.py`

- [ ] **Step 1: Add perplexity function**

```python
def measure_perplexity(model, tokenizer, split="test"):
    """Measure perplexity on WikiText-2."""
    print(f"  Measuring perplexity on WikiText-2 {split}...")
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
    text = "\n\n".join(dataset["text"])
    encodings = tokenizer(text, return_tensors="pt")
    input_ids = encodings.input_ids

    max_len = model.config.n_positions  # 1024 for GPT-2
    nlls = []
    n_tokens = 0

    with torch.no_grad():
        for i in range(0, input_ids.size(1) - max_len, max_len):
            chunk = input_ids[:, i:i + max_len]
            outputs = model(chunk, labels=chunk)
            nlls.append(outputs.loss.item() * max_len)
            n_tokens += max_len

    mean_nll = sum(nlls) / n_tokens
    ppl = float(np.exp(mean_nll))
    print(f"    PPL = {ppl:.2f} ({n_tokens:,} tokens)")
    return ppl
```

- [ ] **Step 2: Wire into main**

Add after allocation plans in `__main__`:

```python
    # Quantize and measure perplexity
    print("\n" + "=" * 65)
    print("Perplexity Measurement")
    print("=" * 65)

    print("\nFP32 baseline:")
    ppl_fp32 = measure_perplexity(model, tokenizer)

    results = {"FP32 baseline": {"bits": 32.0, "ppl": ppl_fp32, "delta": 0.0}}

    for plan_name, plan in plans.items():
        print(f"\n{plan_name} (mean {plan.mean_bits:.2f} bits):")
        q_model = apply_quantization_plan(model, plan)
        ppl = measure_perplexity(q_model, tokenizer)
        results[plan_name] = {
            "bits": plan.mean_bits,
            "ppl": ppl,
            "delta": ppl - ppl_fp32,
        }
        del q_model

    print("\n" + "=" * 65)
    print("Results Summary")
    print("=" * 65)
    print(f"  {'Method':<35} {'Avg Bits':>10} {'PPL':>10} {'Delta':>10}")
    print(f"  {'-'*35} {'-'*10} {'-'*10} {'-'*10}")
    for name, r in results.items():
        print(f"  {name:<35} {r['bits']:>10.2f} {r['ppl']:>10.2f} {r['delta']:>+10.2f}")
```

- [ ] **Step 3: Run full pipeline**

Run: `cd ~/Projects/physics-llm-research && uv run python gpt2_entropy_quantization.py`
Expected: FP32 PPL ~29-35. Quantized PPLs higher. Entropy-linear ideally lower than uniform-4bit.

- [ ] **Step 4: Commit**

```bash
git add gpt2_entropy_quantization.py
git commit -m "feat: perplexity measurement for all quantization methods"
```

---

### Task 8: Correlation Analysis and Plots

**Files:**
- Modify: `gpt2_entropy_quantization.py`

- [ ] **Step 1: Add correlation analysis and plotting functions**

```python
def analyze_correlation(entropies, sensitivities, linear_names):
    """Compute and report entropy vs. Hessian correlation."""
    ent_vals = np.array([entropies[n] for n in linear_names])
    hess_vals = np.array([sensitivities[n] for n in linear_names])

    pearson_r, pearson_p = stats.pearsonr(ent_vals, hess_vals)
    spearman_r, spearman_p = stats.spearmanr(ent_vals, hess_vals)

    print("\n" + "=" * 65)
    print("Correlation: Entropy vs. Hessian Sensitivity")
    print("=" * 65)
    print(f"  Pearson  r = {pearson_r:.4f}  (p = {pearson_p:.2e})")
    print(f"  Spearman r = {spearman_r:.4f}  (p = {spearman_p:.2e})")

    if abs(pearson_r) > 0.7:
        print("  -> STRONG correlation: entropy is a viable cheap proxy")
    elif abs(pearson_r) > 0.5:
        print("  -> MODERATE correlation: entropy captures some Hessian signal")
    else:
        print("  -> WEAK correlation: entropy and Hessian measure different things")

    return {"pearson_r": pearson_r, "pearson_p": pearson_p,
            "spearman_r": spearman_r, "spearman_p": spearman_p,
            "ent_vals": ent_vals, "hess_vals": hess_vals}


def plot_entropy_landscape(entropies, linear_names, save_path):
    vals = [entropies[n] for n in linear_names]
    short_names = [n.split(".")[-2] + "." + n.split(".")[-1] if "." in n else n
                   for n in linear_names]

    fig, ax = plt.subplots(figsize=(16, 6))
    ax.bar(range(len(vals)), vals, color="steelblue", alpha=0.8)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Shannon Entropy (nats)")
    ax.set_title("GPT-2 124M: Per-Layer Activation Entropy")
    ax.set_xticks(range(0, len(vals), max(1, len(vals) // 20)))
    ax.set_xticklabels([short_names[i] for i in range(0, len(vals), max(1, len(vals) // 20))],
                       rotation=45, ha="right", fontsize=7)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Saved: {save_path}")


def plot_entropy_vs_hessian(corr_data, save_path):
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(corr_data["ent_vals"], corr_data["hess_vals"],
               alpha=0.6, s=30, color="steelblue")
    ax.set_xlabel("Activation Entropy (nats)")
    ax.set_ylabel("Hessian Sensitivity (mean diag)")
    ax.set_title(f"Entropy vs. Hessian (Pearson r={corr_data['pearson_r']:.3f})")
    ax.set_yscale("log")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Saved: {save_path}")


def plot_perplexity_comparison(results, save_path):
    methods = [k for k in results if k != "FP32 baseline"]
    ppls = [results[k]["ppl"] for k in methods]

    fig, ax = plt.subplots(figsize=(8, 6))
    bars = ax.bar(methods, ppls, color=["gray", "steelblue", "coral"])
    ax.axhline(y=results["FP32 baseline"]["ppl"], color="green",
               linestyle="--", label=f"FP32 ({results['FP32 baseline']['ppl']:.1f})")
    ax.set_ylabel("Perplexity")
    ax.set_title("GPT-2 124M Quantization: Perplexity Comparison (4-bit avg)")
    ax.legend()
    for bar, ppl in zip(bars, ppls):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{ppl:.1f}", ha="center", fontsize=10)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Saved: {save_path}")
```

- [ ] **Step 2: Wire into main**

Add at end of `__main__`:

```python
    # Correlation analysis
    corr_data = analyze_correlation(entropies, sensitivities, linear_names)

    # Plots
    print("\nGenerating plots...")
    plot_entropy_landscape(entropies, linear_names,
                          os.path.join(RESULTS_DIR, "entropy_landscape.png"))
    plot_entropy_vs_hessian(corr_data,
                           os.path.join(RESULTS_DIR, "entropy_vs_hessian.png"))
    plot_perplexity_comparison(results,
                              os.path.join(RESULTS_DIR, "perplexity_comparison.png"))

    print("\n" + "=" * 65)
    print("Experiment complete!")
    print("=" * 65)
```

- [ ] **Step 3: Run full experiment end-to-end**

Run: `cd ~/Projects/physics-llm-research && uv run python gpt2_entropy_quantization.py`
Expected: Full output with all results, correlation stats, 3 PNG files in `results/`

- [ ] **Step 4: Verify plots exist**

Run: `ls -la ~/Projects/physics-llm-research/results/`
Expected: Three .png files

- [ ] **Step 5: Commit**

```bash
git add gpt2_entropy_quantization.py results/
git commit -m "feat: correlation analysis and plots for GPT-2 entropy quantization"
```

---

### Task 9: Update CLAUDE.md and Obsidian with Results

**Files:**
- Modify: `CLAUDE.md`

- [ ] **Step 1: Update CLAUDE.md Direction 1 status**

Update Direction 1 section: change status to reflect GPT-2 experiment completed. Add actual perplexity numbers and correlation result. Move LLaMA-7B to next step.

- [ ] **Step 2: Update Obsidian experiment note**

Use Obsidian MCP to update `research/physics-llm/experiments/entropy-quant-experiment` with actual GPT-2 results.

- [ ] **Step 3: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: update Direction 1 status with GPT-2 experiment results"
```
