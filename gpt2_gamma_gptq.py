"""
GPT-2 Gamma + GPTQ Quantizer (Experiment A1-GPTQ)
===================================================
Tests whether replacing absmax quantization grids inside GPTQ with
gamma Lloyd-Max grids improves perplexity further.

From exp 8:  GPTQ-uniform 4-bit   -> PPL ~985
From exp A1: Gamma-uniform 4-bit  -> PPL  230  (absmax regime)
Question:    GPTQ + gamma grid    -> PPL  ???

The gamma grid is orthogonal to Hessian error compensation:
  - Gamma grid: WHERE to place quantization levels (matched to weight distribution)
  - GPTQ Hessian: HOW to compensate for rounding error (column-by-column propagation)
If both contribute independently, gamma+GPTQ should beat GPTQ-uniform.

Comparisons:
  1. FP32 baseline
  2. GPTQ uniform 4-bit          (exp 8 baseline)
  3. Absmax gamma-uniform 4-bit  (exp A1 baseline)
  4. GPTQ + gamma grid 4-bit     (new)

Direction: A1 extension (Thermodynamic Quantization)
"""

import os
import copy
import torch
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import gamma as gamma_dist
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from transformers.pytorch_utils import Conv1D
from datasets import load_dataset
from entropy_quantization import uniform_allocation

SEED = 42
CALIBRATION_SAMPLES = 64
SEQ_LEN = 512
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

# Previous results for reference
ABSMAX_UNIFORM_PPL  = 12196.38
ABSMAX_GAMMA_PPL    = 229.84
GPTQ_UNIFORM_PPL    = 985.0   # from exp 8


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)


# ─────────────────────────────────────────────────────────
# Gamma distribution fitting and Lloyd-Max quantizer
# (copied from gpt2_gamma_quantizer.py)
# ─────────────────────────────────────────────────────────


def fit_gamma(weights):
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


def lloyd_max_gamma(k, theta, n_levels, n_iter=60):
    dist = gamma_dist(a=k, scale=theta)
    dist_shifted = gamma_dist(a=k + 1, scale=theta)
    quantiles = np.linspace(1.0 / (n_levels + 1), n_levels / (n_levels + 1), n_levels)
    levels = dist.ppf(quantiles)
    levels = np.maximum(levels, 1e-10)
    upper = dist.ppf(0.9999)

    for _ in range(n_iter):
        boundaries = np.concatenate([
            [0.0],
            (levels[:-1] + levels[1:]) / 2.0,
            [upper]
        ])
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


def gamma_quantize_vector(w_vec, bits, k, theta, boundaries, levels):
    """Quantize a 1D weight vector using a precomputed gamma Lloyd-Max grid.

    Uses the precomputed (boundaries, levels) from the weight matrix fit.
    Preserves sign.
    """
    w_abs = np.abs(w_vec)
    w_q_abs = np.zeros_like(w_abs)
    for i in range(len(levels)):
        mask = (w_abs >= boundaries[i]) & (w_abs < boundaries[i + 1])
        w_q_abs[mask] = levels[i]
    w_q_abs[w_abs >= boundaries[-1]] = levels[-1]
    return np.sign(w_vec) * w_q_abs


# ─────────────────────────────────────────────────────────
# Absmax quantizer (for GPTQ-uniform baseline)
# ─────────────────────────────────────────────────────────


def symmetric_quantize_vec(w_vec, bits):
    n_levels = 2 ** (bits - 1) - 1
    scale = np.abs(w_vec).max() / n_levels
    if scale == 0:
        return w_vec.copy()
    w_int = np.round(w_vec / scale).clip(-n_levels, n_levels)
    return w_int * scale


# ─────────────────────────────────────────────────────────
# GPTQ core (column-by-column with pluggable quantizer)
# ─────────────────────────────────────────────────────────


def gptq_quantize_layer(W_np, H_np, bits, use_gamma=False):
    """GPTQ for one layer with pluggable quantizer.

    W_np: (rows, cols) float32 numpy
    H_np: (cols, cols) float32 numpy  [Hessian = X.T @ X]
    bits: int
    use_gamma: if True, use gamma Lloyd-Max grid per-column; else absmax

    Returns: Q (rows, cols) numpy, total_error float
    """
    rows, cols = W_np.shape
    W = W_np.copy()
    Q = np.zeros_like(W)

    # Precompute gamma grid from the full weight matrix (not per-column)
    # This gives a stable fit; per-column fitting on 768 values would be noisy
    if use_gamma:
        k, theta = fit_gamma(W_np)
        n_levels = 2 ** (bits - 1)
        boundaries, levels = lloyd_max_gamma(k, theta, n_levels)

    # Dampen and invert Hessian
    damp = 0.01 * np.diag(H_np).mean()
    H_damped = H_np + damp * np.eye(cols, dtype=np.float32)
    try:
        L = np.linalg.cholesky(H_damped)
        H_inv = np.linalg.inv(H_damped)
    except np.linalg.LinAlgError:
        H_inv = np.linalg.pinv(H_damped)

    total_error = 0.0

    for j in range(cols):
        w = W[:, j].copy()
        d = H_inv[j, j]

        if use_gamma:
            q = gamma_quantize_vector(w, bits, k, theta, boundaries, levels)
        else:
            q = symmetric_quantize_vec(w, bits)

        Q[:, j] = q
        err = (w - q) / (d + 1e-10)
        total_error += float(np.sum(err ** 2) * d)

        if j + 1 < cols:
            W[:, j + 1:] -= np.outer(err, H_inv[j, j + 1:])

    return Q, total_error


# ─────────────────────────────────────────────────────────
# Hessian collection
# ─────────────────────────────────────────────────────────


def collect_hessians(model, samples):
    linear_names, linear_modules = [], []
    for name, module in model.named_modules():
        if isinstance(module, (torch.nn.Linear, Conv1D)):
            linear_names.append(name)
            linear_modules.append(module)

    in_features_map = {}
    for name, module in zip(linear_names, linear_modules):
        if isinstance(module, Conv1D):
            in_features_map[name] = module.weight.shape[0]
        else:
            in_features_map[name] = module.weight.shape[1]

    hessian_accum = {
        name: np.zeros((in_features_map[name], in_features_map[name]), dtype=np.float64)
        for name in linear_names
    }

    def make_hook(module_name):
        def hook(module, input, output):
            x = input[0].detach().float().cpu().numpy()
            if x.ndim == 3:
                x = x.reshape(-1, x.shape[-1])
            elif x.ndim != 2:
                x = x.reshape(-1, x.shape[-1])
            hessian_accum[module_name] += x.T.astype(np.float64) @ x.astype(np.float64)
        return hook

    hooks = [m.register_forward_hook(make_hook(n))
             for n, m in zip(linear_names, linear_modules)]

    print(f"Collecting Hessians ({len(samples)} samples)...")
    with torch.no_grad():
        for i, s in enumerate(samples):
            model(s)
            if (i + 1) % 16 == 0:
                print(f"  {i+1}/{len(samples)}")
    for h in hooks:
        h.remove()

    hessians = {name: (hessian_accum[name] / len(samples)).astype(np.float32)
                for name in linear_names}
    return hessians, linear_names


# ─────────────────────────────────────────────────────────
# Apply quantization plans
# ─────────────────────────────────────────────────────────


def apply_gptq_plan(model, plan, hessians, use_gamma=False):
    q_model = copy.deepcopy(model)
    total = len(plan.layer_bits)
    done = 0
    label = "gamma+GPTQ" if use_gamma else "GPTQ"

    for name, module in q_model.named_modules():
        if not isinstance(module, (torch.nn.Linear, Conv1D)):
            continue
        if name not in plan.layer_bits:
            continue

        bits = plan.layer_bits[name]
        H = hessians[name]

        if isinstance(module, Conv1D):
            W = module.weight.data.float().cpu().numpy().T   # (out, in)
            Q, _ = gptq_quantize_layer(W, H, bits, use_gamma=use_gamma)
            module.weight.data = torch.tensor(Q.T, dtype=module.weight.dtype)
        else:
            W = module.weight.data.float().cpu().numpy()     # (out, in)
            Q, _ = gptq_quantize_layer(W, H, bits, use_gamma=use_gamma)
            module.weight.data = torch.tensor(Q, dtype=module.weight.dtype)

        done += 1
        if done % 10 == 0 or done == total:
            print(f"  {label}: {done}/{total} layers")

    return q_model


def apply_absmax_gamma_plan(model, plan):
    """Pure gamma absmax (no GPTQ) — replicates exp A1 result for comparison."""
    from gpt2_gamma_quantizer import gamma_quantize
    q_model = copy.deepcopy(model)
    total = len(plan.layer_bits)
    done = 0
    for name, module in q_model.named_modules():
        if not isinstance(module, (torch.nn.Linear, Conv1D)):
            continue
        if name not in plan.layer_bits:
            continue
        bits = plan.layer_bits[name]
        w = module.weight.data.cpu().numpy()
        w_q, _ = gamma_quantize(w, bits)
        module.weight.data = torch.tensor(w_q, dtype=module.weight.dtype)
        done += 1
        if done % 10 == 0 or done == total:
            print(f"  absmax-gamma: {done}/{total} layers")
    return q_model


# ─────────────────────────────────────────────────────────
# Perplexity
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
# Plots
# ─────────────────────────────────────────────────────────


def plot_results(results, save_path):
    order = ["FP32 baseline", "GPTQ-uniform", "absmax-gamma", "GPTQ-gamma"]
    methods = [k for k in order if k in results]
    ppls = [results[k] for k in methods]
    colors = ["#95a5a6", "#3498db", "#2ecc71", "#e74c3c"]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(methods, ppls, color=colors[:len(methods)])
    fp32 = results["FP32 baseline"]
    ax.axhline(fp32, color="black", linestyle="--", linewidth=1.5,
               label=f"FP32 ({fp32:.1f})")
    for bar, ppl in zip(bars, ppls):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(ppls) * 0.01,
                f"{ppl:.0f}", ha="center", fontsize=12, fontweight="bold")
    ax.set_ylabel("Perplexity (WikiText-2 test)", fontsize=12)
    ax.set_title("GPT-2: GPTQ + Gamma Lloyd-Max vs Baselines (4-bit)",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


# ─────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────


if __name__ == "__main__":
    set_seed(SEED)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    model, tokenizer = load_model_and_tokenizer()
    cal_samples = load_wikitext2(tokenizer, split="train",
                                 n_samples=CALIBRATION_SAMPLES)

    linear_names = [n for n, m in model.named_modules()
                    if isinstance(m, (torch.nn.Linear, Conv1D))]
    plan = uniform_allocation(linear_names, bits=4)

    # Collect Hessians (needed for both GPTQ runs)
    hessians, _ = collect_hessians(model, cal_samples)

    print("\n=== PPL Measurements ===")
    results = {}

    print("\nFP32 baseline:")
    results["FP32 baseline"] = measure_perplexity(model, tokenizer)

    print("\nGPTQ-uniform (4-bit, absmax grid):")
    q = apply_gptq_plan(model, plan, hessians, use_gamma=False)
    results["GPTQ-uniform"] = measure_perplexity(q, tokenizer)
    del q

    print("\nabsmax-gamma (4-bit, gamma grid, no GPTQ) [A1 replication]:")
    q = apply_absmax_gamma_plan(model, plan)
    results["absmax-gamma"] = measure_perplexity(q, tokenizer)
    del q

    print("\nGPTQ-gamma (4-bit, gamma grid + Hessian compensation):")
    q = apply_gptq_plan(model, plan, hessians, use_gamma=True)
    results["GPTQ-gamma"] = measure_perplexity(q, tokenizer)
    del q

    print("\n" + "=" * 65)
    print("FINAL RESULTS — Gamma+GPTQ Experiment")
    print("=" * 65)
    baseline = results["GPTQ-uniform"]
    for name, ppl in results.items():
        if name == "FP32 baseline":
            print(f"  {name:<30} PPL={ppl:>8.2f}")
        else:
            delta = (ppl - baseline) / baseline * 100
            marker = " <-- BEST" if ppl == min(
                v for k, v in results.items() if k != "FP32 baseline") else ""
            print(f"  {name:<30} PPL={ppl:>8.2f}  ({delta:+.1f}% vs GPTQ-uniform){marker}")

    print("\nKey questions answered:")
    if "GPTQ-gamma" in results and "GPTQ-uniform" in results:
        g = results["GPTQ-gamma"]
        u = results["GPTQ-uniform"]
        if g < u:
            print(f"  Gamma grid IMPROVES GPTQ: {g:.1f} vs {u:.1f} ({(u-g)/u*100:.1f}% better)")
        else:
            print(f"  Gamma grid does NOT improve GPTQ: {g:.1f} vs {u:.1f}")
    if "absmax-gamma" in results and "GPTQ-gamma" in results:
        ag = results["absmax-gamma"]
        gg = results["GPTQ-gamma"]
        print(f"  GPTQ compensation on gamma grid: {ag:.1f} -> {gg:.1f} ({(ag-gg)/ag*100:.1f}% improvement)")

    print("\nGenerating plots...")
    plot_results(results, os.path.join(RESULTS_DIR, "gamma_gptq_comparison.png"))

    print("\nDone. Results in:", RESULTS_DIR)
