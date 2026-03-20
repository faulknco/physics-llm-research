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
from transformers.pytorch_utils import Conv1D  # GPT-2 uses Conv1D, not nn.Linear
from datasets import load_dataset
from entropy_quantization import (
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


def collect_entropies(model, samples, n_bins=256, range_estimation_samples=8):
    """
    Collect per-layer activation entropy via streaming histograms.
    Two passes:
    1. Quick pass to find [1st, 99th] percentile per layer
    2. Full pass accumulating histograms within those bounds

    Handles both torch.nn.Linear and transformers.pytorch_utils.Conv1D
    (GPT-2 uses Conv1D for its attention/MLP projections).
    """
    linear_names = []
    linear_modules = []
    for name, module in model.named_modules():
        if isinstance(module, (torch.nn.Linear, Conv1D)):
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
                print(f"    {i + 1}/{len(samples)} samples processed")

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

    print(
        f"  Entropy range: [{min(entropies.values()):.4f}, {max(entropies.values()):.4f}]"
    )
    return entropies, linear_names


def compute_hessian_sensitivity(model, samples):
    """Diagonal Fisher approximation: H_diag(l) = E[(dL/dW_l)^2]."""
    linear_names = []
    linear_params = {}
    for name, module in model.named_modules():
        if isinstance(module, (torch.nn.Linear, Conv1D)):
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
            print(f"    {i + 1}/{len(samples)} samples processed")

    sensitivities = {}
    for name in linear_names:
        sensitivities[name] = float((hessian_accum[name] / n_processed).mean().item())

    print(
        f"  Sensitivity range: [{min(sensitivities.values()):.6f}, {max(sensitivities.values()):.6f}]"
    )
    return sensitivities


def hessian_linear_allocation(
    sensitivities, min_bits=2, max_bits=8, target_mean_bits=None
):
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
        assigned_bits = enforce_target_mean(
            assigned_bits, target_mean_bits, min_bits, max_bits
        )

    return QuantizationPlan(
        layer_bits={name: int(b) for name, b in zip(names, assigned_bits)},
        strategy="hessian-linear",
        mean_bits=float(assigned_bits.mean()),
    )


def apply_quantization_plan(model, plan):
    """Apply absmax quantization to model weights. Returns new model."""
    quantized_model = copy.deepcopy(model)
    for name, module in quantized_model.named_modules():
        if isinstance(module, (torch.nn.Linear, Conv1D)) and name in plan.layer_bits:
            bits = plan.layer_bits[name]
            w = module.weight.data.cpu().numpy()
            w_q, mse = absmax_quantize(w, bits)
            module.weight.data = torch.tensor(w_q, dtype=module.weight.dtype)
    return quantized_model


def measure_perplexity(model, tokenizer, split="test"):
    """Measure perplexity on WikiText-2."""
    print(f"  Measuring perplexity on WikiText-2 {split}...")
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
    text = "\n\n".join(dataset["text"])
    encodings = tokenizer(text, return_tensors="pt")
    input_ids = encodings.input_ids

    max_len = model.config.n_positions
    nlls = []
    n_tokens = 0

    with torch.no_grad():
        for i in range(0, input_ids.size(1) - max_len, max_len):
            chunk = input_ids[:, i : i + max_len]
            outputs = model(chunk, labels=chunk)
            nlls.append(outputs.loss.item() * max_len)
            n_tokens += max_len

    mean_nll = sum(nlls) / n_tokens
    ppl = float(np.exp(mean_nll))
    print(f"    PPL = {ppl:.2f} ({n_tokens:,} tokens)")
    return ppl


def analyze_correlation(entropies, sensitivities, linear_names):
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

    return {
        "pearson_r": pearson_r,
        "pearson_p": pearson_p,
        "spearman_r": spearman_r,
        "spearman_p": spearman_p,
        "ent_vals": ent_vals,
        "hess_vals": hess_vals,
    }


def plot_entropy_landscape(entropies, linear_names, save_path):
    vals = [entropies[n] for n in linear_names]
    short_names = [
        n.split(".")[-2] + "." + n.split(".")[-1] if "." in n else n
        for n in linear_names
    ]
    fig, ax = plt.subplots(figsize=(16, 6))
    ax.bar(range(len(vals)), vals, color="steelblue", alpha=0.8)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Shannon Entropy (nats)")
    ax.set_title("GPT-2 124M: Per-Layer Activation Entropy")
    ax.set_xticks(range(0, len(vals), max(1, len(vals) // 20)))
    ax.set_xticklabels(
        [short_names[i] for i in range(0, len(vals), max(1, len(vals) // 20))],
        rotation=45,
        ha="right",
        fontsize=7,
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Saved: {save_path}")


def plot_entropy_vs_hessian(corr_data, save_path):
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(
        corr_data["ent_vals"],
        corr_data["hess_vals"],
        alpha=0.6,
        s=30,
        color="steelblue",
    )
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
    ax.axhline(
        y=results["FP32 baseline"]["ppl"],
        color="green",
        linestyle="--",
        label=f"FP32 ({results['FP32 baseline']['ppl']:.1f})",
    )
    ax.set_ylabel("Perplexity")
    ax.set_title("GPT-2 124M Quantization: Perplexity Comparison (4-bit avg)")
    ax.legend()
    for bar, ppl in zip(bars, ppls):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            f"{ppl:.1f}",
            ha="center",
            fontsize=10,
        )
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Saved: {save_path}")


if __name__ == "__main__":
    set_seed(SEED)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    model, tokenizer = load_model_and_tokenizer()
    cal_samples = load_wikitext2(
        tokenizer, split="train", n_samples=CALIBRATION_SAMPLES
    )

    # Collect entropies
    entropies, linear_names = collect_entropies(model, cal_samples)

    print("\nPer-layer entropies (first 10):")
    for name in linear_names[:10]:
        print(f"  {name:<50s} {entropies[name]:.4f}")

    # Hessian sensitivity
    sensitivities = compute_hessian_sensitivity(model, cal_samples)

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
        print(
            f"  {plan_name}: mean={plan.mean_bits:.2f}, min={min(bits_vals)}, max={max(bits_vals)}"
        )

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
    print(f"  {'-' * 35} {'-' * 10} {'-' * 10} {'-' * 10}")
    for name, r in results.items():
        print(f"  {name:<35} {r['bits']:>10.2f} {r['ppl']:>10.2f} {r['delta']:>+10.2f}")

    # Correlation analysis
    corr_data = analyze_correlation(entropies, sensitivities, linear_names)

    # Plots
    print("\nGenerating plots...")
    plot_entropy_landscape(
        entropies, linear_names, os.path.join(RESULTS_DIR, "entropy_landscape.png")
    )
    plot_entropy_vs_hessian(
        corr_data, os.path.join(RESULTS_DIR, "entropy_vs_hessian.png")
    )
    plot_perplexity_comparison(
        results, os.path.join(RESULTS_DIR, "perplexity_comparison.png")
    )

    print("\n" + "=" * 65)
    print("Experiment complete!")
    print("=" * 65)
