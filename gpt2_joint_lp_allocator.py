"""
GPT-2 A2: Joint Entropy + Hessian LP Bit Allocator
====================================================
Tests whether combining two orthogonal signals (r=-0.006 from exp 4)
beats either signal alone for per-layer bit-width allocation.

Signal 1 -- Activation Entropy:  measures information density
  High entropy -> layer carries diverse activation patterns -> needs more bits

Signal 2 -- Hessian Diagonal:  measures loss sensitivity
  High trace(H) -> small weight perturbations strongly affect loss -> needs more bits

Because entropy and Hessian are uncorrelated, a joint allocator sees both
"what is important" and "what is fragile" -- a strictly larger signal.

LP Formulation (mixed-integer relaxation, solved greedily):
  Minimize   sum_i  H_i * MSE_i(b_i)
  Subject to mean(b_i) = 4.0
             b_i in {2, 3, 4, 6, 8}

where MSE_i(b_i) is the expected absmax quantization error for b_i bits
(approximated analytically: step^2 / 12 for uniform quantization noise).

The greedy solver:
  1. Start all layers at 4 bits.
  2. For each pair (i, j): try upgrading i, downgrading j.
  3. Accept if the swap reduces the weighted objective and preserves mean bits.
  4. Repeat until no improving swap exists.

Comparisons (all at 4-bit average, absmax quantization):
  1. Uniform 4-bit        -- baseline
  2. Entropy-only [2,8]   -- exp 3 approach
  3. Hessian-only [2,8]   -- diagonal H as sole sensitivity signal
  4. Joint LP [2,8]       -- entropy + Hessian via greedy LP

Previous experiment anchors (absmax regime):
  uniform-absmax:    PPL ~12,196
  entropy-linear:    PPL ~4,730    (2.6x improvement)
  gamma-uniform:     PPL ~230      (53x improvement)

Direction: A2 (extends Direction 1 -- Thermodynamic Quantization)
"""

import os
import copy
import torch
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from transformers.pytorch_utils import Conv1D
from datasets import load_dataset
from entropy_quantization import (
    uniform_allocation,
    absmax_quantize,
    QuantizationPlan,
)

SEED = 42
CALIBRATION_SAMPLES = 64
SEQ_LEN = 512
TARGET_BITS = 4.0
BIT_CHOICES = [2, 3, 4, 6, 8]
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

# Prior experiment anchors for comparison plot
PRIOR_UNIFORM_PPL = 12196.34
PRIOR_ENTROPY_PPL = 4730.36
PRIOR_GAMMA_PPL = 230.0


def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)


# -----------------------------------------------------------------
# Model / data loading
# -----------------------------------------------------------------


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


# -----------------------------------------------------------------
# Signal collection
# -----------------------------------------------------------------


def _get_linear_layers(model):
    names, modules = [], []
    for name, module in model.named_modules():
        if isinstance(module, (torch.nn.Linear, Conv1D)):
            names.append(name)
            modules.append(module)
    return names, modules


def collect_signals(model, samples, n_bins=256):
    """
    Single forward-pass sweep collecting both signals:

    Signal 1 -- Activation entropy (Shannon, estimated via histogram).
    Signal 2 -- Hessian diagonal proxy: mean squared activation magnitude.
                diag(X^T X)_k = sum_t x_{t,k}^2  (summed over tokens/batch).
                We store the per-layer total sum of squared activations,
                which is proportional to the Hessian diagonal trace used
                by GPTQ.

    Returns:
        entropies:      {layer_name: float}   nats
        hessian_traces: {layer_name: float}   per-token average trace
        linear_names:   list[str]
        r:              float  correlation between the two signals
    """
    linear_names, linear_modules = _get_linear_layers(model)

    # Pass 1: estimate activation ranges for histogram
    layer_mins = {n: float("inf") for n in linear_names}
    layer_maxs = {n: float("-inf") for n in linear_names}

    def range_hook(module_name):
        def hook(module, input, output):
            x = input[0].detach().float()
            lo = float(x.quantile(0.01))
            hi = float(x.quantile(0.99))
            if lo < layer_mins[module_name]:
                layer_mins[module_name] = lo
            if hi > layer_maxs[module_name]:
                layer_maxs[module_name] = hi

        return hook

    hooks = [
        m.register_forward_hook(range_hook(n))
        for n, m in zip(linear_names, linear_modules)
    ]
    with torch.no_grad():
        for s in samples[:8]:
            model(s)
    for h in hooks:
        h.remove()

    # Pass 2: accumulate histograms + Hessian diagonal
    histograms = {n: np.zeros(n_bins, dtype=np.float64) for n in linear_names}
    hessian_diag_accum = {n: 0.0 for n in linear_names}
    n_tokens_seen = {n: 0 for n in linear_names}

    def signal_hook(module_name):
        def hook(module, input, output):
            x = input[0].detach().float()
            if x.dim() == 3:
                x_flat = x.reshape(-1, x.shape[-1])
            elif x.dim() == 2:
                x_flat = x
            else:
                x_flat = x.reshape(-1, x.shape[-1])

            # Entropy via histogram
            x_np = x_flat.cpu().numpy().flatten()
            lo, hi = layer_mins[module_name], layer_maxs[module_name]
            if lo < hi:
                counts, _ = np.histogram(x_np, bins=n_bins, range=(lo, hi))
                histograms[module_name] += counts.astype(np.float64)

            # Hessian diagonal: sum of squared activations over all tokens
            sq_sum = float(x_flat.pow(2).sum().item())
            hessian_diag_accum[module_name] += sq_sum
            n_tokens_seen[module_name] += x_flat.shape[0]

        return hook

    hooks = [
        m.register_forward_hook(signal_hook(n))
        for n, m in zip(linear_names, linear_modules)
    ]
    with torch.no_grad():
        for i, s in enumerate(samples):
            model(s)
            if (i + 1) % 16 == 0:
                print(f"  {i + 1}/{len(samples)} samples")
    for h in hooks:
        h.remove()

    # Compute entropies
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

    # Normalize Hessian trace by tokens seen
    hessian_traces = {
        name: hessian_diag_accum[name] / max(n_tokens_seen[name], 1)
        for name in linear_names
    }

    print(
        f"  Entropy range:        [{min(entropies.values()):.4f}, {max(entropies.values()):.4f}]"
    )
    print(
        f"  Hessian trace range:  [{min(hessian_traces.values()):.2f}, {max(hessian_traces.values()):.2f}]"
    )

    # Check orthogonality
    names = linear_names
    ent_arr = np.array([entropies[n] for n in names])
    hess_arr = np.array([hessian_traces[n] for n in names])
    r = float(np.corrcoef(ent_arr, hess_arr)[0, 1])
    print(f"  Entropy <-> Hessian correlation: r = {r:.4f}  (expected ~-0.006)")

    return entropies, hessian_traces, linear_names, r


# -----------------------------------------------------------------
# Bit allocation strategies
# -----------------------------------------------------------------


def _normalize(values_dict, names):
    arr = np.array([values_dict[n] for n in names], dtype=float)
    lo, hi = arr.min(), arr.max()
    if hi == lo:
        return np.ones(len(names)) * 0.5
    return (arr - lo) / (hi - lo)


def _next_lower(b):
    """Next smaller value in BIT_CHOICES, or b if already at min."""
    choices = [c for c in BIT_CHOICES if c < b]
    return choices[-1] if choices else b


def _next_higher(b):
    """Next larger value in BIT_CHOICES, or b if already at max."""
    choices = [c for c in BIT_CHOICES if c > b]
    return choices[0] if choices else b


def _snap_to_choices(bits_arr):
    """Snap each element to the nearest value in BIT_CHOICES."""
    return np.array([min(BIT_CHOICES, key=lambda c: abs(c - int(b))) for b in bits_arr])


def _enforce_mean_choices(bits_arr, target, signal_priority=None):
    """
    Greedy mean enforcement that only moves through BIT_CHOICES.

    If mean < target: upgrade the layer with lowest bits (and highest signal_priority).
    If mean > target: downgrade the layer with highest bits (and lowest signal_priority).

    Args:
        bits_arr: array of ints (should all be in BIT_CHOICES)
        target: desired mean (e.g. 4.0)
        signal_priority: optional array of float scores; higher = prefer to give more bits.
                         Used to break ties. If None, index order is used.
    """
    bits = bits_arr.copy().astype(float)
    n = len(bits)
    if signal_priority is None:
        signal_priority = np.arange(n, dtype=float)

    for _ in range(n * 10):
        current_mean = bits.mean()
        if abs(current_mean - target) < 0.5 / n:
            break

        if current_mean < target:
            # Upgrade: among layers not already at max, pick lowest bit / highest priority
            upgradeable = [
                (i, bits[i])
                for i in range(n)
                if _next_higher(int(bits[i])) != int(bits[i])
            ]
            if not upgradeable:
                break
            # Sort by (current bits asc, signal priority desc)
            upgradeable.sort(key=lambda x: (x[1], -signal_priority[x[0]]))
            i = upgradeable[0][0]
            bits[i] = _next_higher(int(bits[i]))
        else:
            # Downgrade: among layers not already at min, pick highest bit / lowest priority
            downgradeable = [
                (i, bits[i])
                for i in range(n)
                if _next_lower(int(bits[i])) != int(bits[i])
            ]
            if not downgradeable:
                break
            downgradeable.sort(key=lambda x: (-x[1], signal_priority[x[0]]))
            i = downgradeable[0][0]
            bits[i] = _next_lower(int(bits[i]))

    return bits.astype(int)


def entropy_only_allocation(
    entropies, names, min_bits=2, max_bits=8, target=TARGET_BITS
):
    """Linear mapping: activation entropy -> bits (constrained to BIT_CHOICES)."""
    norm = _normalize(entropies, names)
    raw = min_bits + norm * (max_bits - min_bits)
    assigned = _snap_to_choices(np.round(raw).astype(int))
    assigned = _enforce_mean_choices(assigned, target, signal_priority=norm)
    return QuantizationPlan(
        layer_bits={n: int(b) for n, b in zip(names, assigned)},
        strategy="entropy-only",
        mean_bits=float(assigned.mean()),
    )


def hessian_only_allocation(
    hessian_traces, names, min_bits=2, max_bits=8, target=TARGET_BITS
):
    """Linear mapping: hessian trace -> bits (constrained to BIT_CHOICES)."""
    norm = _normalize(hessian_traces, names)
    raw = min_bits + norm * (max_bits - min_bits)
    assigned = _snap_to_choices(np.round(raw).astype(int))
    assigned = _enforce_mean_choices(assigned, target, signal_priority=norm)
    return QuantizationPlan(
        layer_bits={n: int(b) for n, b in zip(names, assigned)},
        strategy="hessian-only",
        mean_bits=float(assigned.mean()),
    )


def _absmax_mse_analytic(scale: float, bits: int) -> float:
    """
    Analytic MSE for symmetric absmax quantization (uniform quantization noise).
      step = 2 * scale / (2^bits - 1)
      MSE = step^2 / 12
    """
    if scale <= 0 or bits <= 0:
        return 0.0
    n_levels = (2**bits) - 1
    step = (2.0 * scale) / n_levels
    return (step**2) / 12.0


def get_weight_scales(model, linear_names):
    """Return per-layer absmax scale (max |w|) for analytic MSE estimation."""
    scales = {}
    layer_map = dict(model.named_modules())
    for name in linear_names:
        module = layer_map.get(name)
        if module is None:
            scales[name] = 1.0
            continue
        w = module.weight.data.float().cpu().numpy()
        scales[name] = float(np.abs(w).max())
    return scales


def joint_lp_allocation(
    entropies,
    hessian_traces,
    weight_scales,
    names,
    min_bits=2,
    max_bits=8,
    target=TARGET_BITS,
    alpha=0.5,
):
    """
    Joint LP bit allocator using entropy + Hessian as orthogonal signals.

    Objective: minimize  sum_i  sensitivity_i * MSE_i(b_i)
    where sensitivity_i = alpha * norm_entropy_i + (1-alpha) * norm_hessian_i

    Solved via greedy integer optimization over BIT_CHOICES,
    constrained to mean(b_i) == target.

    Args:
        alpha: weight on entropy (1-alpha on Hessian).
               alpha=1.0 -> entropy-only  alpha=0.0 -> Hessian-only
    """
    norm_ent = _normalize(entropies, names)
    norm_hess = _normalize(hessian_traces, names)
    sensitivity = alpha * norm_ent + (1.0 - alpha) * norm_hess

    scales = np.array([weight_scales.get(n, 1.0) for n in names])
    n = len(names)

    # Initialise at nearest bit choice to target, constrained to BIT_CHOICES
    init_bits = int(round(target))
    bits = np.array(
        [min(BIT_CHOICES, key=lambda c: abs(c - init_bits)) for _ in range(n)]
    )
    bits = _enforce_mean_choices(bits, target, signal_priority=sensitivity)

    # Greedy swap: upgrade i, downgrade j => minimize weighted objective
    for _ in range(500):
        best_delta = 0.0
        best_i, best_j = -1, -1
        best_bi, best_bj = 0, 0

        for i in range(n):
            bi_up = _next_higher(bits[i])
            if bi_up == bits[i]:
                continue
            mse_i_cur = _absmax_mse_analytic(scales[i], int(bits[i]))
            mse_i_up = _absmax_mse_analytic(scales[i], bi_up)
            cost_i = sensitivity[i] * (mse_i_up - mse_i_cur)  # positive (worse)

            for j in range(n):
                if i == j:
                    continue
                bj_down = _next_lower(bits[j])
                if bj_down == bits[j]:
                    continue
                mse_j_cur = _absmax_mse_analytic(scales[j], int(bits[j]))
                mse_j_down = _absmax_mse_analytic(scales[j], bj_down)
                cost_j = sensitivity[j] * (mse_j_down - mse_j_cur)  # negative (better)

                # Mean constraint: swap valid only if bit totals balance
                new_mean = (bits.sum() - bits[i] - bits[j] + bi_up + bj_down) / n
                if abs(new_mean - target) > 0.5:
                    continue

                # Net objective change (negative = improvement)
                net = cost_i + cost_j
                if net < best_delta - 1e-12:
                    best_delta = net
                    best_i, best_j = i, j
                    best_bi, best_bj = bi_up, bj_down

        if best_i < 0:
            break
        bits[best_i] = best_bi
        bits[best_j] = best_bj

    # Final enforcement: all values should already be in BIT_CHOICES; recheck mean
    bits = _enforce_mean_choices(bits, target, signal_priority=sensitivity)

    return QuantizationPlan(
        layer_bits={n: int(b) for n, b in zip(names, bits)},
        strategy=f"joint-lp-a{alpha:.1f}",
        mean_bits=float(bits.mean()),
    )


# -----------------------------------------------------------------
# Quantization + perplexity
# -----------------------------------------------------------------


def apply_absmax_plan(model, plan):
    q_model = copy.deepcopy(model)
    for name, module in q_model.named_modules():
        if isinstance(module, (torch.nn.Linear, Conv1D)) and name in plan.layer_bits:
            bits = plan.layer_bits[name]
            w = module.weight.data.cpu().numpy()
            w_q, _ = absmax_quantize(w, bits)
            module.weight.data = torch.tensor(w_q, dtype=module.weight.dtype)
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
            chunk = input_ids[:, i : i + max_len]
            out = model(chunk, labels=chunk)
            nlls.append(out.loss.item() * max_len)
            n_tokens += max_len
    ppl = float(np.exp(sum(nlls) / n_tokens))
    print(f"    PPL = {ppl:.2f}  ({n_tokens:,} tokens)")
    return ppl


# -----------------------------------------------------------------
# Plots
# -----------------------------------------------------------------


def analyse_signals(entropies, hessian_traces, names, weight_scales, save_path):
    """Scatter + bar charts showing signal orthogonality and per-layer distributions."""
    ent_arr = np.array([entropies[n] for n in names])
    hess_arr = np.array([hessian_traces[n] for n in names])
    scale_arr = np.array([weight_scales[n] for n in names])

    fig, axes = plt.subplots(1, 3, figsize=(17, 5))

    ax = axes[0]
    sc = ax.scatter(ent_arr, hess_arr, c=scale_arr, cmap="viridis", alpha=0.7, s=30)
    plt.colorbar(sc, ax=ax, label="Weight scale (absmax)")
    r = float(np.corrcoef(ent_arr, hess_arr)[0, 1])
    ax.set_xlabel("Activation Entropy (nats)", fontsize=11)
    ax.set_ylabel("Hessian Trace (per token)", fontsize=11)
    ax.set_title(f"Signal Orthogonality\nr = {r:.4f} (expected ~-0.006)", fontsize=11)

    ax = axes[1]
    ax.bar(range(len(names)), ent_arr, color="steelblue", alpha=0.8)
    ax.set_xlabel("Layer index", fontsize=11)
    ax.set_ylabel("Activation Entropy (nats)", fontsize=11)
    ax.set_title("Per-Layer Entropy", fontsize=11)

    ax = axes[2]
    ax.bar(range(len(names)), hess_arr, color="coral", alpha=0.8)
    ax.set_xlabel("Layer index", fontsize=11)
    ax.set_ylabel("Hessian Trace (per token)", fontsize=11)
    ax.set_title("Per-Layer Hessian Trace", fontsize=11)

    plt.suptitle(
        "A2: Joint Signal Analysis -- Entropy & Hessian", fontsize=13, fontweight="bold"
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


def plot_bit_allocation(plans, names, save_path):
    """Bar charts of bit assignments per strategy."""
    n_plans = len(plans)
    fig, axes = plt.subplots(n_plans, 1, figsize=(16, 3 * n_plans), sharex=True)
    if n_plans == 1:
        axes = [axes]

    color_map = {2: "#e74c3c", 3: "#e67e22", 4: "#3498db", 6: "#2ecc71", 8: "#9b59b6"}

    for ax, (plan_name, plan) in zip(axes, plans.items()):
        bits_arr = [plan.layer_bits.get(n, 4) for n in names]
        bar_colors = [color_map.get(b, "grey") for b in bits_arr]
        ax.bar(range(len(names)), bits_arr, color=bar_colors, alpha=0.85)
        ax.set_ylabel("Bits", fontsize=10)
        ax.set_title(f"{plan_name}  (mean={plan.mean_bits:.2f} bits)", fontsize=10)
        ax.set_ylim(0, 10)
        ax.axhline(4.0, color="black", linestyle="--", linewidth=0.8, alpha=0.5)

    from matplotlib.patches import Patch

    legend_elements = [
        Patch(facecolor=color_map[b], label=f"{b}-bit") for b in sorted(color_map)
    ]
    axes[-1].legend(handles=legend_elements, loc="upper right", fontsize=8, ncol=5)
    axes[-1].set_xlabel("Layer index", fontsize=11)

    plt.suptitle("A2: Bit Allocation Comparison", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


def plot_ppl_comparison(results, save_path):
    """Bar chart comparing all methods plus prior experiment anchors."""
    method_order = [
        "FP32 baseline",
        "uniform-4bit",
        "entropy-only",
        "hessian-only",
        "joint-lp",
    ]
    prior_anchors = {
        "gamma-uniform\n(A1, prior)": PRIOR_GAMMA_PPL,
        "entropy-linear\n(exp3, prior)": PRIOR_ENTROPY_PPL,
        "uniform-absmax\n(exp3, prior)": PRIOR_UNIFORM_PPL,
    }

    labels = [m for m in method_order if m in results]
    ppls = [results[m] for m in labels]
    colors_main = ["#2ecc71", "#95a5a6", "#e74c3c", "#3498db", "#9b59b6"]

    all_labels = labels + list(prior_anchors.keys())
    all_ppls = ppls + list(prior_anchors.values())
    all_colors = colors_main[: len(labels)] + ["#bdc3c7"] * len(prior_anchors)

    fig, ax = plt.subplots(figsize=(14, 7))
    bars = ax.bar(
        all_labels,
        all_ppls,
        color=all_colors,
        alpha=0.85,
        edgecolor="white",
        linewidth=1.2,
    )

    fp32 = results.get("FP32 baseline", 29.95)
    ax.axhline(
        fp32,
        color="#2ecc71",
        linestyle="--",
        linewidth=1.5,
        alpha=0.5,
        label=f"FP32 ({fp32:.1f})",
    )

    for bar, ppl in zip(bars, all_ppls):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(all_ppls) * 0.005,
            f"{ppl:.0f}",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )

    ax.set_ylabel("Perplexity (WikiText-2 test)", fontsize=12)
    ax.set_title(
        "GPT-2 A2: Joint Entropy+Hessian LP Allocator\n(4-bit avg, absmax quantization)",
        fontsize=13,
        fontweight="bold",
    )
    ax.legend(fontsize=10)
    plt.xticks(rotation=15, ha="right", fontsize=10)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


def plot_alpha_sweep(alpha_results, save_path):
    """PPL as a function of the entropy/Hessian mixing parameter alpha."""
    alphas = sorted(alpha_results.keys())
    ppls = [alpha_results[a] for a in alphas]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(alphas, ppls, "o-", color="#9b59b6", linewidth=2, markersize=8)
    ax.set_xlabel("alpha (weight on entropy; 1-alpha on Hessian)", fontsize=12)
    ax.set_ylabel("Perplexity (WikiText-2 test)", fontsize=12)
    ax.set_title(
        "A2: Joint LP -- PPL vs. Signal Mixing Parameter alpha\n"
        "(alpha=1.0: entropy-only;  alpha=0.0: Hessian-only)",
        fontsize=12,
        fontweight="bold",
    )
    ax.axvline(
        0.5, color="grey", linestyle="--", linewidth=1, label="alpha=0.5 (equal weight)"
    )

    best_alpha = alphas[int(np.argmin(ppls))]
    best_ppl = min(ppls)
    ppl_range = max(ppls) - min(ppls)
    ax.annotate(
        f"best alpha={best_alpha:.1f}\nPPL={best_ppl:.0f}",
        xy=(best_alpha, best_ppl),
        xytext=(best_alpha + 0.05, best_ppl + max(ppl_range * 0.15, 10)),
        arrowprops=dict(arrowstyle="->", color="black"),
        fontsize=10,
    )
    ax.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


# -----------------------------------------------------------------
# Main
# -----------------------------------------------------------------


if __name__ == "__main__":
    set_seed(SEED)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    model, tokenizer = load_model_and_tokenizer()
    cal_samples = load_wikitext2(
        tokenizer, split="train", n_samples=CALIBRATION_SAMPLES
    )

    # -- 1. Collect both signals ----------------------------------------------
    print("\n" + "=" * 65)
    print("Step 1: Collecting entropy + Hessian signals (single pass)")
    print("=" * 65)
    entropies, hessian_traces, linear_names, r_ent_hess = collect_signals(
        model, cal_samples
    )
    weight_scales = get_weight_scales(model, linear_names)

    # -- 2. Build allocation plans --------------------------------------------
    print("\n" + "=" * 65)
    print("Step 2: Building bit-width allocation plans")
    print("=" * 65)

    plan_uniform = uniform_allocation(linear_names, bits=4)
    plan_entropy = entropy_only_allocation(entropies, linear_names)
    plan_hessian = hessian_only_allocation(hessian_traces, linear_names)
    plan_joint = joint_lp_allocation(
        entropies, hessian_traces, weight_scales, linear_names, alpha=0.5
    )

    all_plans = {
        "uniform-4bit": plan_uniform,
        "entropy-only": plan_entropy,
        "hessian-only": plan_hessian,
        "joint-lp": plan_joint,
    }

    for name, plan in all_plans.items():
        bits_vals = list(plan.layer_bits.values())
        print(
            f"  {name:<18}: mean={plan.mean_bits:.2f}, "
            f"min={min(bits_vals)}, max={max(bits_vals)}, "
            f"dist={sorted(set(bits_vals))}"
        )

    # -- 3. Signal analysis plots ---------------------------------------------
    print("\nGenerating signal analysis plots...")
    analyse_signals(
        entropies,
        hessian_traces,
        linear_names,
        weight_scales,
        os.path.join(RESULTS_DIR, "a2_signal_analysis.png"),
    )
    plot_bit_allocation(
        all_plans,
        linear_names,
        os.path.join(RESULTS_DIR, "a2_bit_allocation.png"),
    )

    # -- 4. FP32 baseline -----------------------------------------------------
    print("\n" + "=" * 65)
    print("Step 3: FP32 baseline")
    print("=" * 65)
    ppl_fp32 = measure_perplexity(model, tokenizer)

    # -- 5. Evaluate each plan ------------------------------------------------
    print("\n" + "=" * 65)
    print("Step 4: Evaluating quantization plans")
    print("=" * 65)

    ppl_results = {"FP32 baseline": ppl_fp32}

    for plan_name, plan in all_plans.items():
        print(f"\n  {plan_name} (mean {plan.mean_bits:.2f} bits):")
        q_model = apply_absmax_plan(model, plan)
        ppl = measure_perplexity(q_model, tokenizer)
        ppl_results[plan_name] = ppl
        del q_model

    # -- 6. Alpha sweep -------------------------------------------------------
    print("\n" + "=" * 65)
    print("Step 5: Alpha sweep (entropy <-> Hessian mixing)")
    print("=" * 65)
    ALPHA_VALUES = [0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0]
    alpha_ppl_results = {}

    for alpha in ALPHA_VALUES:
        plan_a = joint_lp_allocation(
            entropies, hessian_traces, weight_scales, linear_names, alpha=alpha
        )
        q_model = apply_absmax_plan(model, plan_a)
        ppl_a = measure_perplexity(q_model, tokenizer)
        alpha_ppl_results[alpha] = ppl_a
        del q_model
        print(f"  alpha={alpha:.1f}: PPL={ppl_a:.2f}")

    # -- 7. Results table -----------------------------------------------------
    print("\n" + "=" * 65)
    print("FINAL RESULTS -- Experiment A2: Joint Entropy+Hessian LP Allocator")
    print("=" * 65)
    baseline = ppl_results["uniform-4bit"]

    def delta_str(ppl, base):
        d = (ppl - base) / base * 100
        direction = "better" if d < 0 else "worse"
        return f"{abs(d):.1f}% {direction}"

    print(f"  {'Method':<30} {'PPL':>10}  {'vs uniform-4bit':>18}")
    print(f"  {'-' * 30}  {'-' * 10}  {'-' * 18}")
    for name, ppl in ppl_results.items():
        if name == "FP32 baseline":
            print(f"  {name:<30} {ppl:>10.2f}  {'(baseline)':>18}")
        else:
            print(f"  {name:<30} {ppl:>10.2f}  {delta_str(ppl, baseline):>18}")

    print("\n  Prior anchors (absmax regime, for context):")
    print(
        f"  {'gamma-uniform (A1)':<30} {PRIOR_GAMMA_PPL:>10.0f}  {delta_str(PRIOR_GAMMA_PPL, baseline):>18}"
    )
    print(
        f"  {'entropy-linear (exp3)':<30} {PRIOR_ENTROPY_PPL:>10.0f}  {delta_str(PRIOR_ENTROPY_PPL, baseline):>18}"
    )

    print(f"\n  Signal correlation: r(entropy, Hessian) = {r_ent_hess:.4f}")
    best_alpha = min(alpha_ppl_results, key=lambda a: alpha_ppl_results[a])
    print(f"  Best alpha: {best_alpha:.1f}  (PPL={alpha_ppl_results[best_alpha]:.2f})")

    # Verdict
    joint_ppl = ppl_results["joint-lp"]
    ent_ppl = ppl_results["entropy-only"]
    hess_ppl = ppl_results["hessian-only"]
    if joint_ppl < ent_ppl and joint_ppl < hess_ppl:
        print(
            f"\n  *** JOINT LP WINS: {joint_ppl:.0f} < entropy {ent_ppl:.0f} and Hessian {hess_ppl:.0f} ***"
        )
        print("  Orthogonal signals ARE complementary for bit allocation.")
    else:
        better_single = "entropy-only" if ent_ppl < hess_ppl else "hessian-only"
        print(f"\n  Single signal ({better_single}) matches or beats joint LP.")
        print("  Check alpha sweep for optimal mixing ratio.")

    # -- 8. Plots -------------------------------------------------------------
    print("\nGenerating result plots...")
    plot_ppl_comparison(ppl_results, os.path.join(RESULTS_DIR, "a2_ppl_comparison.png"))
    plot_alpha_sweep(alpha_ppl_results, os.path.join(RESULTS_DIR, "a2_alpha_sweep.png"))

    print("\n" + "=" * 65)
    print("A2 complete. All results saved to:", RESULTS_DIR)
    print("=" * 65)
