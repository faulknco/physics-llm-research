"""
GPT-2 Combined Physics-Based Compression
=========================================
Combines all three physics-inspired compression techniques:
  1. Entropy-based adaptive quantization (thermodynamic quantization)
  2. RG-guided low-rank pruning (renormalization group flow)
  3. Lattice attention decay (xi=512)

Tests whether compressions are additive (independent benefits) or interfere.

Experiment 4 from the physics-LLM research plan.
"""

import os
import sys
import copy
import torch
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from transformers.pytorch_utils import Conv1D
from transformers.models.gpt2 import modeling_gpt2 as gpt2_modeling
from datasets import load_dataset

sys.path.insert(0, "/Users/faulknco/Projects/physics-llm-research")
from entropy_quantization import (
    entropy_linear_allocation,
    uniform_allocation,
    absmax_quantize,
    QuantizationPlan,
    enforce_target_mean,
)

RESULTS_DIR = "/Users/faulknco/Projects/physics-llm-research/results"
SEED = 42
CALIBRATION_SAMPLES = 128
SEQ_LEN = 1024
LATTICE_XI = 512
RG_RANK_FRACTION = 0.50

# ---- Lattice Attention Globals ----
_current_xi = None
_attn_registry = gpt2_modeling.ALL_ATTENTION_FUNCTIONS


def build_lattice_mask(seq_len, xi, device="cpu"):
    i_idx = torch.arange(seq_len, device=device).unsqueeze(1)
    j_idx = torch.arange(seq_len, device=device).unsqueeze(0)
    dist = torch.abs(i_idx - j_idx).float()
    log_mask = -dist / xi
    causal_mask = j_idx > i_idx
    log_mask = log_mask.masked_fill(causal_mask, float("-inf"))
    return log_mask


def _lattice_eager_attention_forward(module, query, key, value, attention_mask, **kwargs):
    import torch.nn as nn
    attn_weights = torch.matmul(query, key.transpose(-1, -2))
    if module.scale_attn_weights:
        attn_weights = attn_weights / torch.full(
            [], value.size(-1) ** 0.5,
            dtype=attn_weights.dtype, device=attn_weights.device,
        )
    if module.scale_attn_by_inverse_layer_idx:
        attn_weights = attn_weights / float(module.layer_idx + 1)
    if _current_xi is not None:
        seq_len_q = query.size(-2)
        seq_len_k = key.size(-2)
        if seq_len_q == seq_len_k:
            log_mask = build_lattice_mask(seq_len_q, _current_xi, device=attn_weights.device)
            log_mask = log_mask.to(attn_weights.dtype)
            attn_weights = attn_weights + log_mask.unsqueeze(0).unsqueeze(0)
    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask
    attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    attn_weights = attn_weights.type(value.dtype)
    attn_weights = module.attn_dropout(attn_weights)
    attn_output = torch.matmul(attn_weights, value)
    attn_output = attn_output.transpose(1, 2)
    return attn_output, attn_weights


def patch_attention(xi):
    global _current_xi
    _current_xi = xi
    _attn_registry["eager"] = _lattice_eager_attention_forward


def unpatch_attention():
    global _current_xi
    _current_xi = None
    _attn_registry._local_mapping.pop("eager", None)


# ---- Setup ----

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)


def load_model_and_tokenizer():
    print("Loading GPT-2 124M (attn_implementation='eager')...")
    model = GPT2LMHeadModel.from_pretrained(
        "openai-community/gpt2", attn_implementation="eager"
    )
    tokenizer = GPT2Tokenizer.from_pretrained("openai-community/gpt2")
    model.eval()
    print(f"  attn_implementation: {model.config._attn_implementation}")
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


# ---- Perplexity ----

def measure_perplexity(model, tokenizer, split="test", label=""):
    if label:
        print(f"  [{label}] Measuring PPL on WikiText-2 {split}...")
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
    text = "\n\n".join(dataset["text"])
    encodings = tokenizer(text, return_tensors="pt")
    input_ids = encodings.input_ids
    max_len = model.config.n_positions
    nlls = []
    n_tokens = 0
    with torch.no_grad():
        for i in range(0, input_ids.size(1) - max_len, max_len):
            chunk = input_ids[:, i: i + max_len]
            outputs = model(chunk, labels=chunk)
            nlls.append(outputs.loss.item() * max_len)
            n_tokens += max_len
    mean_nll = sum(nlls) / n_tokens
    ppl = float(np.exp(mean_nll))
    print(f"    PPL = {ppl:.4f} ({n_tokens:,} tokens)")
    return ppl


# ---- Entropy Collection ----

def collect_entropies(model, samples, n_bins=256, range_estimation_samples=8):
    linear_names = []
    linear_modules = []
    for name, module in model.named_modules():
        if isinstance(module, (torch.nn.Linear, Conv1D)):
            linear_names.append(name)
            linear_modules.append(module)

    print(f"  Collecting entropy from {len(linear_names)} linear layers...")

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


# ---- Apply Quantization ----

def apply_quantization_plan(model, plan):
    q_model = copy.deepcopy(model)
    for name, module in q_model.named_modules():
        if isinstance(module, (torch.nn.Linear, Conv1D)) and name in plan.layer_bits:
            bits = plan.layer_bits[name]
            w = module.weight.data.cpu().numpy()
            w_q, _ = absmax_quantize(w, bits)
            module.weight.data = torch.tensor(w_q, dtype=module.weight.dtype)
    return q_model


# ---- RG Pruning ----

def apply_rg_pruning(model, rank_fraction=0.5):
    marginal_types = ["attn.c_attn", "mlp.c_fc"]
    pruned_model = copy.deepcopy(model)
    n_pruned = 0

    for name, module in pruned_model.named_modules():
        if not isinstance(module, Conv1D):
            continue
        is_marginal = any(mt in name for mt in marginal_types)
        if not is_marginal:
            continue
        W = module.weight.data
        U, S, Vt = torch.linalg.svd(W, full_matrices=False)
        k = max(1, int(len(S) * rank_fraction))
        W_approx = U[:, :k] @ torch.diag(S[:k]) @ Vt[:k, :]
        module.weight.data = W_approx
        n_pruned += 1

    print(f"  RG pruning: truncated {n_pruned} weight matrices to rank {rank_fraction:.0%}")
    return pruned_model


# ---- Compression Metrics ----

def compute_compression_metrics(model, original_param_count, plan=None, rank_fraction=None):
    marginal_types = ["attn.c_attn", "mlp.c_fc"]
    total_bits = 0.0

    for name, param in model.named_parameters():
        module_name = name.rsplit(".", 1)[0]

        if plan is not None and module_name in plan.layer_bits:
            bits = plan.layer_bits[module_name]
        else:
            bits = 32

        if rank_fraction is not None and any(mt in module_name for mt in marginal_types):
            if len(param.shape) == 2:
                in_f, out_f = param.shape
                k = max(1, int(min(in_f, out_f) * rank_fraction))
                eff_params = k * (in_f + out_f)
            else:
                eff_params = param.numel()
            total_bits += bits * eff_params
        else:
            total_bits += bits * param.numel()

    fp32_bits = 32.0 * original_param_count
    compression_ratio = fp32_bits / total_bits
    eff_bits_per_param = total_bits / original_param_count
    return compression_ratio, eff_bits_per_param


# ---- Plots ----

def plot_combined_table(results, save_path):
    method_labels = list(results.keys())
    ppls = [results[m]["ppl"] for m in method_labels]
    ratios = [results[m]["comp_ratio"] for m in method_labels]

    x = np.arange(len(method_labels))
    bar_width = 0.38

    fig, ax1 = plt.subplots(figsize=(13, 6))
    ax2 = ax1.twinx()

    bars1 = ax1.bar(x - bar_width / 2, ppls, bar_width, label="PPL", color="#3b82f6", alpha=0.85)
    bars2 = ax2.bar(x + bar_width / 2, ratios, bar_width, label="Compression Ratio", color="#f97316", alpha=0.85)

    for bar, ppl in zip(bars1, ppls):
        ax1.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
            f"{ppl:.1f}", ha="center", va="bottom", fontsize=8.5, color="#1d4ed8",
        )
    for bar, ratio in zip(bars2, ratios):
        ax2.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
            f"{ratio:.1f}x", ha="center", va="bottom", fontsize=8.5, color="#c2410c",
        )

    ax1.set_ylabel("Perplexity (WikiText-2 test)", fontsize=11)
    ax2.set_ylabel("Compression Ratio (FP32 size / compressed size)", fontsize=11)
    ax1.set_xticks(x)
    ax1.set_xticklabels([m.replace(" ", "\n") for m in method_labels], fontsize=9)
    ax1.set_title("GPT-2 Combined Physics Compression: PPL vs Compression Ratio", fontsize=13)
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=9)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


def plot_pareto(results, save_path):
    method_labels = list(results.keys())
    ppls = [results[m]["ppl"] for m in method_labels]
    ratios = [results[m]["comp_ratio"] for m in method_labels]

    fig, ax = plt.subplots(figsize=(10, 7))
    colors = ["#1e40af", "#dc2626", "#16a34a", "#9333ea", "#ca8a04", "#0891b2"]
    markers = ["o", "s", "^", "D", "P", "*"]

    for i, (label, ppl, ratio) in enumerate(zip(method_labels, ppls, ratios)):
        ax.scatter(ratio, ppl,
            color=colors[i % len(colors)],
            marker=markers[i % len(markers)],
            s=120, zorder=5, label=label,
        )
        ax.annotate(label, (ratio, ppl),
            textcoords="offset points", xytext=(8, 4),
            fontsize=8.5, color=colors[i % len(colors)],
        )

    # Pareto frontier: maximize ratio while minimizing PPL
    min_ppl_so_far = float("inf")
    pareto_idxs = []
    for idx in sorted(range(len(ratios)), key=lambda i: ratios[i], reverse=True):
        if ppls[idx] <= min_ppl_so_far:
            pareto_idxs.append(idx)
            min_ppl_so_far = ppls[idx]

    if len(pareto_idxs) > 1:
        pareto_sorted = sorted(pareto_idxs, key=lambda i: ratios[i])
        px = [ratios[i] for i in pareto_sorted]
        py = [ppls[i] for i in pareto_sorted]
        ax.plot(px, py, "k--", linewidth=1.5, alpha=0.6, label="Pareto frontier", zorder=4)

    ax.set_xlabel("Compression Ratio (log scale)", fontsize=11)
    ax.set_ylabel("Perplexity (WikiText-2 test)", fontsize=11)
    ax.set_title(
        "GPT-2 Combined Physics Compression: Pareto Frontier\n"
        "(lower PPL + higher compression = better)",
        fontsize=12,
    )
    ax.set_xscale("log")
    ax.legend(fontsize=8.5, loc="upper left")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


# ---- Results Table ----

def print_results_table(results):
    fp32_ppl = results["FP32 baseline"]["ppl"]
    header = f"{'Method':<35} {'PPL':>8} {'Delta':>8} {'Comp.Ratio':>12} {'Eff.Bits/Param':>15}"
    sep = "-" * len(header)
    print()
    print("=" * len(header))
    print("COMBINED PHYSICS COMPRESSION -- RESULTS SUMMARY")
    print("=" * len(header))
    print(header)
    print(sep)
    for name, r in results.items():
        ppl = r["ppl"]
        delta_pct = (ppl - fp32_ppl) / fp32_ppl * 100
        delta_str = f"+{delta_pct:.1f}%" if delta_pct >= 0 else f"{delta_pct:.1f}%"
        print(f"{name:<35} {ppl:>8.2f} {delta_str:>8} {r['comp_ratio']:>11.2f}x {r['eff_bits']:>14.1f}")
    print(sep)
    print()


# ---- Main ----

def main():
    set_seed(SEED)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("\n" + "=" * 70)
    print("STEP 1: Setup")
    print("=" * 70)

    model, tokenizer = load_model_and_tokenizer()
    cal_samples = load_wikitext2(tokenizer, split="train", n_samples=CALIBRATION_SAMPLES)

    original_param_count = sum(p.numel() for p in model.parameters())
    print(f"  Total parameters: {original_param_count:,}")

    print("\nFP32 baseline perplexity:")
    unpatch_attention()
    ppl_fp32 = measure_perplexity(model, tokenizer, label="FP32")

    fp32_comp_ratio, fp32_eff_bits = compute_compression_metrics(
        model, original_param_count, plan=None, rank_fraction=None
    )
    results = {
        "FP32 baseline": {"ppl": ppl_fp32, "comp_ratio": fp32_comp_ratio, "eff_bits": fp32_eff_bits}
    }

    print("\n" + "=" * 70)
    print("STEP 2a: Entropy Quantization Only (4-bit mean)")
    print("=" * 70)

    print("  Collecting activation entropies from calibration data...")
    entropies, linear_names = collect_entropies(model, cal_samples)
    entropy_plan = entropy_linear_allocation(entropies, min_bits=2, max_bits=8, target_mean_bits=4.0)
    print(f"  Plan: mean={entropy_plan.mean_bits:.2f} bits, layers={len(entropy_plan.layer_bits)}")

    q_model = apply_quantization_plan(model, entropy_plan)
    ppl_entropy = measure_perplexity(q_model, tokenizer, label="Entropy-only")
    comp_ratio_ent, eff_bits_ent = compute_compression_metrics(
        model, original_param_count, plan=entropy_plan, rank_fraction=None
    )
    results["Entropy quant (4-bit)"] = {"ppl": ppl_entropy, "comp_ratio": comp_ratio_ent, "eff_bits": eff_bits_ent}
    del q_model

    print("\n" + "=" * 70)
    print("STEP 2b: RG-Guided Pruning Only (50% rank)")
    print("=" * 70)

    rg_model = apply_rg_pruning(model, rank_fraction=RG_RANK_FRACTION)
    ppl_rg = measure_perplexity(rg_model, tokenizer, label="RG-only")
    comp_ratio_rg, eff_bits_rg = compute_compression_metrics(
        model, original_param_count, plan=None, rank_fraction=RG_RANK_FRACTION
    )
    results["RG pruning (50% rank)"] = {"ppl": ppl_rg, "comp_ratio": comp_ratio_rg, "eff_bits": eff_bits_rg}
    del rg_model

    print("\n" + "=" * 70)
    print(f"STEP 2c: Lattice Attention Only (xi={LATTICE_XI})")
    print("=" * 70)

    patch_attention(LATTICE_XI)
    ppl_lattice = measure_perplexity(model, tokenizer, label=f"Lattice xi={LATTICE_XI}")
    unpatch_attention()
    comp_ratio_lat, eff_bits_lat = compute_compression_metrics(
        model, original_param_count, plan=None, rank_fraction=None
    )
    results[f"Lattice attn (xi={LATTICE_XI})"] = {"ppl": ppl_lattice, "comp_ratio": comp_ratio_lat, "eff_bits": eff_bits_lat}

    print("\n" + "=" * 70)
    print("STEP 3d: Entropy Quant + RG Pruning")
    print("=" * 70)

    rg_model = apply_rg_pruning(model, rank_fraction=RG_RANK_FRACTION)
    q_rg_model = apply_quantization_plan(rg_model, entropy_plan)
    ppl_ent_rg = measure_perplexity(q_rg_model, tokenizer, label="Entropy+RG")
    comp_ratio_ent_rg, eff_bits_ent_rg = compute_compression_metrics(
        model, original_param_count, plan=entropy_plan, rank_fraction=RG_RANK_FRACTION
    )
    results["Entropy + RG"] = {"ppl": ppl_ent_rg, "comp_ratio": comp_ratio_ent_rg, "eff_bits": eff_bits_ent_rg}
    del rg_model, q_rg_model

    print("\n" + "=" * 70)
    print(f"STEP 3e: All Three Combined (Entropy + RG + Lattice xi={LATTICE_XI})")
    print("=" * 70)

    rg_model = apply_rg_pruning(model, rank_fraction=RG_RANK_FRACTION)
    q_rg_model = apply_quantization_plan(rg_model, entropy_plan)
    patch_attention(LATTICE_XI)
    ppl_all = measure_perplexity(q_rg_model, tokenizer, label="All three")
    unpatch_attention()
    comp_ratio_all, eff_bits_all = compute_compression_metrics(
        model, original_param_count, plan=entropy_plan, rank_fraction=RG_RANK_FRACTION
    )
    results["All three combined"] = {"ppl": ppl_all, "comp_ratio": comp_ratio_all, "eff_bits": eff_bits_all}
    del rg_model, q_rg_model

    print_results_table(results)

    print("\n" + "=" * 70)
    print("STEP 5: Generating Plots")
    print("=" * 70)

    table_path = os.path.join(RESULTS_DIR, "combined_compression_table.png")
    pareto_path = os.path.join(RESULTS_DIR, "combined_pareto.png")
    plot_combined_table(results, table_path)
    plot_pareto(results, pareto_path)

    print("=" * 70)
    print("ANALYSIS: Are the methods additive?")
    print("=" * 70)

    ppl_fp32 = results["FP32 baseline"]["ppl"]
    ppl_e = results["Entropy quant (4-bit)"]["ppl"]
    ppl_r = results["RG pruning (50% rank)"]["ppl"]
    ppl_l = results[f"Lattice attn (xi={LATTICE_XI})"]["ppl"]
    ppl_er = results["Entropy + RG"]["ppl"]
    ppl_erl = results["All three combined"]["ppl"]

    delta_e = (ppl_e - ppl_fp32) / ppl_fp32 * 100
    delta_r = (ppl_r - ppl_fp32) / ppl_fp32 * 100
    delta_l = (ppl_l - ppl_fp32) / ppl_fp32 * 100
    delta_er = (ppl_er - ppl_fp32) / ppl_fp32 * 100
    delta_erl = (ppl_erl - ppl_fp32) / ppl_fp32 * 100

    expected_er = delta_e + delta_r
    expected_erl = delta_e + delta_r + delta_l

    print(f"  Individual PPL deltas:")
    print(f"    Entropy:  {delta_e:+.1f}%")
    print(f"    RG:       {delta_r:+.1f}%")
    print(f"    Lattice:  {delta_l:+.1f}%")
    print()
    print(f"  Combined (Entropy + RG):")
    print(f"    Expected if additive: {expected_er:+.1f}%")
    print(f"    Actual:               {delta_er:+.1f}%")
    interference_er = delta_er - expected_er
    if abs(interference_er) < 5.0:
        print(f"    -> APPROXIMATELY ADDITIVE (interference = {interference_er:+.1f}%)")
    elif interference_er > 5.0:
        print(f"    -> SUBADDITIVE -- methods INTERFERE (extra cost = {interference_er:+.1f}%)")
    else:
        print(f"    -> SUPERADDITIVE -- methods COMPLEMENT each other (gain = {-interference_er:.1f}%)")
    print()
    print(f"  All three combined:")
    print(f"    Expected if additive: {expected_erl:+.1f}%")
    print(f"    Actual:               {delta_erl:+.1f}%")
    interference_erl = delta_erl - expected_erl
    if abs(interference_erl) < 5.0:
        print(f"    -> APPROXIMATELY ADDITIVE (interference = {interference_erl:+.1f}%)")
    elif interference_erl > 5.0:
        print(f"    -> SUBADDITIVE -- methods INTERFERE (extra cost = {interference_erl:+.1f}%)")
    else:
        print(f"    -> SUPERADDITIVE -- methods COMPLEMENT each other (gain = {-interference_erl:.1f}%)")

    print()
    best_ratio = max(results[m]["comp_ratio"] for m in results)
    best_on_pareto = min(
        (m for m in results if results[m]["comp_ratio"] >= best_ratio * 0.5),
        key=lambda m: results[m]["ppl"],
    )
    print("Pareto frontier analysis:")
    print(f"  Best PPL at high compression: {best_on_pareto}")
    print(f"    PPL={results[best_on_pareto]['ppl']:.2f}, "
          f"Ratio={results[best_on_pareto]['comp_ratio']:.2f}x, "
          f"Bits={results[best_on_pareto]['eff_bits']:.1f}")

    print()
    print("Done. Results saved to:", RESULTS_DIR)
    return results


if __name__ == "__main__":
    main()
