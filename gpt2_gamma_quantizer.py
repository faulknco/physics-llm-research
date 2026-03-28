"""
GPT-2 Gamma-Distribution Quantizer (Experiment A1)
===================================================
Tests whether Lloyd-Max quantization grids matched to a gamma-type
weight distribution outperform absmax (uniform grid) quantization.

Theory (Dyson Brownian Motion):
  SGD dynamics on weight matrices are equivalent to Dyson Brownian Motion,
  the same process that evolves GUE eigenvalues. The stationary distribution
  of singular values under this dynamics is gamma-like (arXiv:2507.12709).
  A Lloyd-Max quantizer designed for the actual distribution minimises
  MSE reconstruction error for a fixed number of quantization levels.

Comparisons (all at 4-bit average):
  1. Uniform absmax     - baseline, assumes uniform distribution
  2. Entropy-linear     - entropy-guided bit allocation, absmax grids
  3. Gamma-Lloyd-Max    - uniform bit allocation, gamma-matched grids
  4. Entropy + Gamma    - entropy allocation AND gamma grids (combined)

Direction: A1 (extends Direction 1 -- Thermodynamic Quantization)
"""

import os
import copy
import torch
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import gamma as gamma_dist, kstest
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from transformers.pytorch_utils import Conv1D
from datasets import load_dataset
from entropy_quantization import (
    entropy_linear_allocation,
    uniform_allocation,
    absmax_quantize,
    QuantizationPlan,
    enforce_target_mean,
)

SEED = 42
CALIBRATION_SAMPLES = 64
SEQ_LEN = 512
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)


# ─────────────────────────────────────────────────────────
# Gamma distribution fitting
# ─────────────────────────────────────────────────────────


def fit_gamma(weights):
    """Fit gamma(k, theta) to |weights| via method of moments.

    Returns (k, theta).
    """
    w_abs = np.abs(weights.flatten())
    w_abs = w_abs[w_abs > 1e-10]
    if len(w_abs) < 10:
        return 1.0, 1.0
    mu = w_abs.mean()
    var = w_abs.var()
    if var < 1e-12:
        return 1.0, float(mu)
    k = (mu ** 2) / var
    theta = var / mu
    return float(k), float(theta)


# ─────────────────────────────────────────────────────────
# Lloyd-Max quantizer for gamma distribution
# ─────────────────────────────────────────────────────────


def lloyd_max_gamma(k, theta, n_levels, n_iter=60):
    """Iterative Lloyd-Max quantizer for Gamma(k, theta) distribution.

    For a gamma distribution, the centroid update has a closed form:
      E[X | a < X < b] = k*theta * (F(b; k+1, theta) - F(a; k+1, theta))
                         / (F(b; k, theta) - F(a; k, theta))
    where F is the gamma CDF.

    Returns:
        boundaries: (n_levels+1,) array
        levels:     (n_levels,)   array
    """
    dist = gamma_dist(a=k, scale=theta)
    dist_shifted = gamma_dist(a=k + 1, scale=theta)  # for centroid calculation

    # Initialise at quantiles
    quantiles = np.linspace(1.0 / (n_levels + 1), n_levels / (n_levels + 1), n_levels)
    levels = dist.ppf(quantiles)
    levels = np.maximum(levels, 1e-10)

    upper = dist.ppf(0.9999)

    for _ in range(n_iter):
        # Decision boundaries: midpoints
        boundaries = np.concatenate([
            [0.0],
            (levels[:-1] + levels[1:]) / 2.0,
            [upper]
        ])

        # Centroid update
        new_levels = np.zeros(n_levels)
        for i in range(n_levels):
            lo, hi = boundaries[i], boundaries[i + 1]
            p = dist.cdf(hi) - dist.cdf(lo)
            if p < 1e-12:
                new_levels[i] = levels[i]
                continue
            p_num = dist_shifted.cdf(hi) - dist_shifted.cdf(lo)
            new_levels[i] = k * theta * p_num / p

        if np.all(np.abs(new_levels - levels) < 1e-9):
            break
        levels = new_levels

    return boundaries, levels


def gamma_quantize(weights, bits):
    """Quantize weight matrix using gamma-matched Lloyd-Max grid.

    Weights are symmetric around zero so we:
    1. Fit gamma to |w|
    2. Build Lloyd-Max grid on positive side (n_levels = 2^(bits-1))
    3. Quantize |w|, restore sign

    Returns: (quantized_weights, mse)
    """
    n_levels = 2 ** (bits - 1)
    w = weights.flatten()

    k, theta = fit_gamma(weights)
    boundaries, levels = lloyd_max_gamma(k, theta, n_levels)

    w_abs = np.abs(w)
    w_q_abs = np.zeros_like(w_abs)

    for i in range(len(levels)):
        lo = boundaries[i]
        hi = boundaries[i + 1]
        mask = (w_abs >= lo) & (w_abs < hi)
        w_q_abs[mask] = levels[i]

    # Values above upper boundary get clamped to last level
    w_q_abs[w_abs >= boundaries[-1]] = levels[-1]

    w_q = np.sign(w) * w_q_abs
    mse = float(np.mean((w - w_q) ** 2))
    return w_q.reshape(weights.shape), mse


# ─────────────────────────────────────────────────────────
# Model loading and calibration
# ─────────────────────────────────────────────────────────


def load_model_and_tokenizer():
    print("Loading GPT-2 124M...")
    model = GPT2LMHeadModel.from_pretrained("openai-community/gpt2")
    tokenizer = GPT2Tokenizer.from_pretrained("openai-community/gpt2")
    model.train(False)
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


def collect_entropies(model, samples, n_bins=256):
    """Collect per-layer activation entropy (streaming 2-pass)."""
    linear_names, linear_modules = [], []
    for name, module in model.named_modules():
        if isinstance(module, (torch.nn.Linear, Conv1D)):
            linear_names.append(name)
            linear_modules.append(module)

    print(f"Collecting entropy from {len(linear_names)} layers...")

    layer_mins = {n: float("inf") for n in linear_names}
    layer_maxs = {n: float("-inf") for n in linear_names}

    def range_hook(module_name):
        def hook(module, input, output):
            x = input[0].detach().float()
            lo, hi = float(x.quantile(0.01)), float(x.quantile(0.99))
            if lo < layer_mins[module_name]:
                layer_mins[module_name] = lo
            if hi > layer_maxs[module_name]:
                layer_maxs[module_name] = hi
        return hook

    hooks = [m.register_forward_hook(range_hook(n))
             for n, m in zip(linear_names, linear_modules)]
    with torch.no_grad():
        for s in samples[:8]:
            model(s)
    for h in hooks:
        h.remove()

    histograms = {n: np.zeros(n_bins) for n in linear_names}

    def hist_hook(module_name):
        def hook(module, input, output):
            x = input[0].detach().float().cpu().numpy().flatten()
            lo, hi = layer_mins[module_name], layer_maxs[module_name]
            counts, _ = np.histogram(x, bins=n_bins, range=(lo, hi))
            histograms[module_name] += counts.astype(float)
        return hook

    hooks = [m.register_forward_hook(hist_hook(n))
             for n, m in zip(linear_names, linear_modules)]
    with torch.no_grad():
        for i, s in enumerate(samples):
            model(s)
            if (i + 1) % 16 == 0:
                print(f"    {i+1}/{len(samples)} samples")
    for h in hooks:
        h.remove()

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

    return entropies, linear_names


def apply_absmax_plan(model, plan):
    q_model = copy.deepcopy(model)
    for name, module in q_model.named_modules():
        if isinstance(module, (torch.nn.Linear, Conv1D)) and name in plan.layer_bits:
            bits = plan.layer_bits[name]
            w = module.weight.data.cpu().numpy()
            w_q, _ = absmax_quantize(w, bits)
            module.weight.data = torch.tensor(w_q, dtype=module.weight.dtype)
    return q_model


def apply_gamma_plan(model, plan):
    q_model = copy.deepcopy(model)
    total = len([n for n, m in q_model.named_modules()
                 if isinstance(m, (torch.nn.Linear, Conv1D)) and n in plan.layer_bits])
    done = 0
    for name, module in q_model.named_modules():
        if isinstance(module, (torch.nn.Linear, Conv1D)) and name in plan.layer_bits:
            bits = plan.layer_bits[name]
            w = module.weight.data.cpu().numpy()
            w_q, _ = gamma_quantize(w, bits)
            module.weight.data = torch.tensor(w_q, dtype=module.weight.dtype)
            done += 1
            if done % 10 == 0:
                print(f"  gamma quantizing: {done}/{total} layers")
    return q_model


def measure_perplexity(model, tokenizer, split="test"):
    print(f"  Measuring PPL on WikiText-2 {split}...")
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
    text = "\n\n".join(dataset["text"])
    encodings = tokenizer(text, return_tensors="pt")
    input_ids = encodings.input_ids
    max_len = model.config.n_positions
    nlls, n_tokens = [], 0
    with torch.no_grad():
        for i in range(0, input_ids.size(1) - max_len, max_len):
            chunk = input_ids[:, i:i + max_len]
            out = model(chunk, labels=chunk)
            nlls.append(out.loss.item() * max_len)
            n_tokens += max_len
    ppl = float(np.exp(sum(nlls) / n_tokens))
    print(f"    PPL = {ppl:.2f}  ({n_tokens:,} tokens)")
    return ppl


# ─────────────────────────────────────────────────────────
# Weight distribution analysis
# ─────────────────────────────────────────────────────────


def analyse_weight_distributions(model):
    print("\nFitting gamma to weight matrices...")
    results = []
    for name, module in model.named_modules():
        if not isinstance(module, (torch.nn.Linear, Conv1D)):
            continue
        w = module.weight.data.cpu().numpy().flatten()
        w_abs = np.abs(w[np.abs(w) > 1e-10])
        if len(w_abs) < 50:
            continue
        k, theta = fit_gamma(module.weight.data.cpu().numpy())
        ks_stat, ks_p = kstest(w_abs, lambda x: gamma_dist.cdf(x, a=k, scale=theta))
        results.append({"name": name, "k": k, "theta": theta,
                        "ks_stat": ks_stat, "ks_p": ks_p, "n": len(w_abs)})

    ks_stats = [r["ks_stat"] for r in results]
    k_vals = [r["k"] for r in results]
    print(f"  Layers: {len(results)}")
    print(f"  Gamma shape k:  mean={np.mean(k_vals):.3f}  std={np.std(k_vals):.3f}")
    print(f"  KS statistic:   mean={np.mean(ks_stats):.4f}  min={np.min(ks_stats):.4f}")
    print(f"  (Dyson BM predicts k ~ 1-2; KS < 0.05 = good fit)")
    return results


# ─────────────────────────────────────────────────────────
# Plots
# ─────────────────────────────────────────────────────────


def plot_weight_distributions(model, fit_results, save_path):
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()
    n = len(fit_results)
    idxs = [0, n // 5, 2 * n // 5, 3 * n // 5, 4 * n // 5, n - 1]
    selected = [fit_results[i] for i in idxs if i < n]

    layer_map = {nm: mo for nm, mo in model.named_modules()
                 if isinstance(mo, (torch.nn.Linear, Conv1D))}

    for ax, r in zip(axes, selected):
        name = r["name"]
        if name not in layer_map:
            ax.axis("off")
            continue
        w = layer_map[name].weight.data.cpu().numpy().flatten()
        w_abs = np.abs(w[np.abs(w) > 1e-10])
        ax.hist(w_abs, bins=60, density=True, alpha=0.6,
                color="steelblue", label="|w| empirical")
        x_rng = np.linspace(0, np.percentile(w_abs, 99.5), 300)
        pdf = gamma_dist.pdf(x_rng, a=r["k"], scale=r["theta"])
        ax.plot(x_rng, pdf, "r-", lw=2,
                label=f"Gamma(k={r['k']:.2f}, θ={r['theta']:.4f})")
        short = ".".join(name.split(".")[-3:])
        ax.set_title(f"{short}\nKS={r['ks_stat']:.4f}", fontsize=9)
        ax.set_xlabel("|weight|", fontsize=8)
        ax.legend(fontsize=7)

    fig.suptitle("GPT-2: Weight Distributions vs Gamma Fit (|w|)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


def plot_ks_fit(fit_results, save_path):
    ks_vals = [r["ks_stat"] for r in fit_results]
    k_vals = [r["k"] for r in fit_results]
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 7))
    ax1.bar(range(len(ks_vals)), ks_vals, color="coral", alpha=0.8)
    ax1.axhline(0.05, color="red", linestyle="--", linewidth=1.5,
                label="KS=0.05 (approximate significance threshold)")
    ax1.set_ylabel("KS statistic (lower = better gamma fit)")
    ax1.set_title("Gamma Goodness-of-Fit: KS Statistic per Layer")
    ax1.legend()
    ax2.bar(range(len(k_vals)), k_vals, color="steelblue", alpha=0.8)
    ax2.axhline(1.0, color="orange", linestyle="--", linewidth=1.5, label="k=1 (exponential)")
    ax2.axhline(2.0, color="green", linestyle="--", linewidth=1.5, label="k=2")
    ax2.set_ylabel("Gamma shape k")
    ax2.set_title("Gamma Shape k per Layer (Dyson BM predicts k ~ 1-2)")
    ax2.set_xlabel("Layer index")
    ax2.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


def plot_ppl_comparison(results, save_path):
    order = ["FP32 baseline", "uniform-absmax", "entropy-linear",
             "gamma-uniform", "gamma-entropy"]
    methods = [k for k in order if k in results]
    ppls = [results[k] for k in methods]
    fp32_ppl = results["FP32 baseline"]
    colors = ["#95a5a6", "#3498db", "#e74c3c", "#2ecc71", "#9b59b6"]
    fig, ax = plt.subplots(figsize=(11, 6))
    bars = ax.bar(methods, ppls, color=colors[:len(methods)])
    ax.axhline(fp32_ppl, color="black", linestyle="--", linewidth=1.5,
               label=f"FP32 ({fp32_ppl:.1f})")
    for bar, ppl in zip(bars, ppls):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(ppls) * 0.01,
                f"{ppl:.0f}", ha="center", fontsize=11, fontweight="bold")
    ax.set_ylabel("Perplexity (WikiText-2 test)", fontsize=12)
    ax.set_title("GPT-2 Quantization: Gamma Lloyd-Max vs Baselines (4-bit avg)",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


# ─────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────


if __name__ == "__main__":
    set_seed(SEED)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    model, tokenizer = load_model_and_tokenizer()
    cal_samples = load_wikitext2(tokenizer, split="train",
                                 n_samples=CALIBRATION_SAMPLES)

    # Weight distribution analysis
    fit_results = analyse_weight_distributions(model)

    # Entropy calibration
    entropies, linear_names = collect_entropies(model, cal_samples)

    TARGET_BITS = 4.0
    uniform_plan  = uniform_allocation(linear_names, bits=4)
    entropy_plan  = entropy_linear_allocation(entropies, min_bits=2, max_bits=8,
                                              target_mean_bits=TARGET_BITS)

    print("\n=== PPL Measurements ===")
    print("FP32 baseline:")
    ppl_fp32 = measure_perplexity(model, tokenizer)
    ppl_results = {"FP32 baseline": ppl_fp32}

    print("\nuniform-absmax (4-bit, absmax grid):")
    q = apply_absmax_plan(model, uniform_plan)
    ppl_results["uniform-absmax"] = measure_perplexity(q, tokenizer)
    del q

    print("\nentropy-linear (4-bit avg, absmax grid):")
    q = apply_absmax_plan(model, entropy_plan)
    ppl_results["entropy-linear"] = measure_perplexity(q, tokenizer)
    del q

    print("\ngamma-uniform (4-bit, gamma Lloyd-Max grid):")
    q = apply_gamma_plan(model, uniform_plan)
    ppl_results["gamma-uniform"] = measure_perplexity(q, tokenizer)
    del q

    print("\ngamma-entropy (4-bit avg entropy allocation, gamma grid):")
    q = apply_gamma_plan(model, entropy_plan)
    ppl_results["gamma-entropy"] = measure_perplexity(q, tokenizer)
    del q

    print("\n" + "=" * 65)
    print("FINAL RESULTS — Gamma Quantizer Experiment A1")
    print("=" * 65)
    baseline_ppl = ppl_results["uniform-absmax"]
    for name, ppl in ppl_results.items():
        if name == "FP32 baseline":
            print(f"  {name:<30} PPL={ppl:>8.2f}")
        else:
            delta = (ppl - baseline_ppl) / baseline_ppl * 100
            marker = " *BEST*" if ppl == min(v for k, v in ppl_results.items()
                                              if k != "FP32 baseline") else ""
            print(f"  {name:<30} PPL={ppl:>8.2f}  ({delta:+.1f}% vs uniform-absmax){marker}")

    print("\nGenerating plots...")
    plot_weight_distributions(
        model, fit_results,
        os.path.join(RESULTS_DIR, "gamma_weight_distributions.png"))
    plot_ks_fit(
        fit_results,
        os.path.join(RESULTS_DIR, "gamma_ks_fit.png"))
    plot_ppl_comparison(
        ppl_results,
        os.path.join(RESULTS_DIR, "gamma_ppl_comparison.png"))

    print("\nDone. All results in:", RESULTS_DIR)
