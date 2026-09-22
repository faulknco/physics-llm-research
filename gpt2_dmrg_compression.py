"""
GPT-2 B5: DMRG-Style Tensor Network Compression
================================================
Compares three weight compression strategies using SVD:

1. Fixed-rank SVD truncation (standard low-rank approximation)
   - Sweep rank fractions from 10% to 90% of min(m, n)
   - Baseline: all methods should beat this at equal compression

2. MP-adaptive truncation (RMT-filtered bond dimensions)
   - Marchenko-Pastur bulk edge: sigma_mp = sigma_1 * sqrt(gamma)
     where gamma = min(m,n) / max(m,n) (aspect ratio)
   - Keep only singular values above the MP bulk -- these are
     "signal" SVs; below is random noise
   - Adaptive rank per weight matrix, no global hyperparameter

3. DMRG-style variational sweep
   - Initialize from MP-adaptive truncation
   - 2 refinement sweeps: for each weight matrix, try rank +-1
     and keep the change if it reduces ||W - W_approx||_F^2
     subject to a fixed total parameter budget
   - Approximates DMRG's local optimization with bond dimension sweeps

Key insight: For 2D matrices, MPS bond dimension = matrix rank.
DMRG = iterative rank refinement. This is the 2D "sanity check"
from the B5 design before extending to reshaped 4D tensors.

Direction: B5 (DMRG tensor network compression)
Script: gpt2_dmrg_compression.py
"""

import os
import copy
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from datasets import load_dataset

SEED = 42
N_LAYERS = 12
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

# Rank fractions for fixed-rank sweep (fraction of min(m, n))
RANK_FRACTIONS = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]

# Weight types to compress (name suffix -> human label)
TARGET_WEIGHTS = {
    "attn.c_attn": "attn.c_attn",
    "attn.c_proj": "attn.c_proj",
    "mlp.c_fc":    "mlp.c_fc",
    "mlp.c_proj":  "mlp.c_proj",
}

# DMRG sweep parameters
DMRG_SWEEPS = 2
DMRG_RANK_DELTA = 1   # try +/- this many singular values per sweep step


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)


# -----------------------------------------------------------------
# Model + data
# -----------------------------------------------------------------

def load_model_and_tokenizer():
    print("Loading GPT-2 124M...")
    model = GPT2LMHeadModel.from_pretrained("openai-community/gpt2")
    tokenizer = GPT2Tokenizer.from_pretrained("openai-community/gpt2")
    model.train(False)
    return model, tokenizer


def measure_perplexity(model, tokenizer, split="test", max_tokens=None):
    print(f"  Measuring PPL on WikiText-2 {split}...")
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
    text = "\n".join(ds["text"])
    enc = tokenizer(text, return_tensors="pt")
    ids = enc.input_ids[0]
    if max_tokens is not None:
        ids = ids[:max_tokens]
    stride = 512
    seq_len = 512
    device = next(model.parameters()).device
    nlls = []
    for i in range(0, len(ids) - seq_len, stride):
        chunk = ids[i : i + seq_len].unsqueeze(0).to(device)
        with torch.no_grad():
            loss = model(chunk, labels=chunk).loss
        nlls.append(loss.item())
    ppl = float(np.exp(np.mean(nlls)))
    n_tokens = len(range(0, len(ids) - seq_len, stride)) * seq_len
    print(f"    PPL = {ppl:.2f}  ({n_tokens:,} tokens)")
    return ppl


# -----------------------------------------------------------------
# SVD utilities
# -----------------------------------------------------------------

def get_weight_matrices(model):
    """Return dict of {name: (W_2d, original_shape, module, attr)}."""
    matrices = {}
    for layer_idx in range(N_LAYERS):
        block = model.transformer.h[layer_idx]
        targets = {
            f"h.{layer_idx}.attn.c_attn": (block.attn.c_attn, "weight"),
            f"h.{layer_idx}.attn.c_proj": (block.attn.c_proj, "weight"),
            f"h.{layer_idx}.mlp.c_fc":    (block.mlp.c_fc, "weight"),
            f"h.{layer_idx}.mlp.c_proj":  (block.mlp.c_proj, "weight"),
        }
        for name, (module, attr) in targets.items():
            W = getattr(module, attr).data
            orig_shape = W.shape
            W_2d = W.reshape(W.shape[0], -1)
            matrices[name] = {
                "W_2d": W_2d,
                "orig_shape": orig_shape,
                "module": module,
                "attr": attr,
            }
    return matrices


def svd_compress(W_2d, rank):
    """Low-rank approximation via truncated SVD. Returns approx W_2d."""
    U, S, Vh = torch.linalg.svd(W_2d, full_matrices=False)
    rank = min(rank, len(S))
    W_approx = (U[:, :rank] * S[:rank]) @ Vh[:rank, :]
    return W_approx


def mp_bulk_edge(W_2d):
    """
    Marchenko-Pastur bulk edge for a random matrix with same shape.
    For W of shape (m, n), aspect ratio gamma = min/max.
    MP upper edge: lambda_+ = sigma^2 * (1 + sqrt(gamma))^2
    where sigma^2 is estimated from the median singular value squared.
    We work in singular value space: sv_mp = sqrt(lambda_+).
    """
    m, n = W_2d.shape
    gamma = min(m, n) / max(m, n)
    U, S, Vh = torch.linalg.svd(W_2d, full_matrices=False)
    # Estimate noise level from median SV (robust to outliers)
    sigma2 = float(torch.median(S).item() ** 2) / (1 + gamma ** 0.5) ** 2
    sv_mp = float((sigma2 ** 0.5) * (1 + gamma ** 0.5))
    # Count SVs above bulk edge
    signal_mask = S > sv_mp
    rank = int(signal_mask.sum().item())
    rank = max(rank, 1)  # keep at least 1
    return rank, S, U, Vh, sv_mp


def apply_svd_to_model(model, rank_map):
    """
    Apply SVD compression in-place. rank_map: {weight_name: rank}.
    Returns compression ratio (params_kept / params_total).
    """
    total_params = 0
    kept_params = 0
    matrices = get_weight_matrices(model)
    for name, info in matrices.items():
        W_2d = info["W_2d"]
        rank = rank_map.get(name, None)
        if rank is None:
            total_params += W_2d.numel()
            kept_params += W_2d.numel()
            continue
        m, n = W_2d.shape
        rank = min(rank, min(m, n))
        W_approx = svd_compress(W_2d, rank)
        # Write back
        module = info["module"]
        attr = info["attr"]
        with torch.no_grad():
            getattr(module, attr).data.copy_(
                W_approx.reshape(info["orig_shape"])
            )
        # Compression: storing U (m x r) + S (r,) + Vh (r x n) = r*(m+n+1)
        total_params += m * n
        kept_params += rank * (m + n + 1)
    compression = total_params / kept_params if kept_params > 0 else 1.0
    return compression, kept_params, total_params


# -----------------------------------------------------------------
# Strategy 1: Fixed-rank sweep
# -----------------------------------------------------------------

def fixed_rank_sweep(base_model, tokenizer):
    print("\n" + "=" * 65)
    print("Strategy 1: Fixed-rank SVD sweep")
    print("=" * 65)
    results = []
    matrices = get_weight_matrices(base_model)
    for frac in RANK_FRACTIONS:
        model = copy.deepcopy(base_model)
        rank_map = {}
        for name, info in matrices.items():
            m, n = info["W_2d"].shape
            r = max(1, int(frac * min(m, n)))
            rank_map[name] = r
        compression, kept, total = apply_svd_to_model(model, rank_map)
        print(f"\n  rank_frac={frac:.0%}  compression={compression:.2f}x  "
              f"params kept={kept/1e6:.2f}M / {total/1e6:.2f}M")
        ppl = measure_perplexity(model, tokenizer)
        results.append({
            "rank_frac": frac,
            "compression": compression,
            "ppl": ppl,
            "kept": kept,
            "total": total,
        })
        del model
    return results


# -----------------------------------------------------------------
# Strategy 2: MP-adaptive truncation
# -----------------------------------------------------------------

def mp_adaptive_compression(base_model, tokenizer):
    print("\n" + "=" * 65)
    print("Strategy 2: MP-adaptive truncation (RMT-filtered bond dims)")
    print("=" * 65)
    model = copy.deepcopy(base_model)
    matrices = get_weight_matrices(model)
    rank_map = {}
    rank_info = {}
    for name, info in matrices.items():
        W_2d = info["W_2d"]
        rank, S, U, Vh, sv_mp = mp_bulk_edge(W_2d)
        rank_map[name] = rank
        m, n = W_2d.shape
        rank_info[name] = {
            "rank": rank,
            "max_rank": min(m, n),
            "frac": rank / min(m, n),
            "sv_mp": sv_mp,
            "sv_max": float(S[0].item()),
        }

    print("\n  Per-weight MP-adaptive ranks:")
    print(f"  {'Weight':<35} {'rank':>6} {'max_r':>6} {'frac':>6}  {'sv_mp':>8}  {'sv_max':>8}")
    for name, info in sorted(rank_info.items()):
        print(f"  {name:<35} {info['rank']:>6} {info['max_rank']:>6} "
              f"{info['frac']:>6.1%}  {info['sv_mp']:>8.4f}  {info['sv_max']:>8.4f}")

    compression, kept, total = apply_svd_to_model(model, rank_map)
    print(f"\n  MP-adaptive compression={compression:.2f}x  "
          f"params kept={kept/1e6:.2f}M / {total/1e6:.2f}M")
    ppl = measure_perplexity(model, tokenizer)
    return {"compression": compression, "ppl": ppl, "kept": kept,
            "total": total, "rank_map": rank_map, "rank_info": rank_info}


# -----------------------------------------------------------------
# Strategy 3: DMRG-style sweep
# -----------------------------------------------------------------

def dmrg_sweep_compression(base_model, tokenizer, mp_rank_map, budget_params):
    """
    Initialize from MP-adaptive ranks. Sweep over weight matrices,
    trying rank +/- DMRG_RANK_DELTA and accepting if reconstruction
    error improves without exceeding total parameter budget.
    Analogous to DMRG: locally optimize each bond dimension while
    keeping the total bond budget fixed.
    """
    print("\n" + "=" * 65)
    print("Strategy 3: DMRG-style variational sweep")
    print(f"  Budget: {budget_params/1e6:.2f}M params  "
          f"({DMRG_SWEEPS} sweeps, delta={DMRG_RANK_DELTA})")
    print("=" * 65)

    matrices_ref = get_weight_matrices(base_model)
    # Current rank assignment (mutable)
    rank_map = {k: v for k, v in mp_rank_map.items()}

    def total_kept(rmap):
        tot = 0
        for name, info in matrices_ref.items():
            m, n = info["W_2d"].shape
            r = rmap.get(name, min(m, n))
            tot += r * (m + n + 1)
        return tot

    def reconstruction_error(W_2d, rank):
        U, S, Vh = torch.linalg.svd(W_2d, full_matrices=False)
        # Frobenius error of truncation = sum of discarded S^2
        if rank >= len(S):
            return 0.0
        return float(S[rank:].pow(2).sum().item())

    # Pre-compute reference reconstruction errors at initial ranks
    errors = {}
    for name, info in matrices_ref.items():
        W_2d = info["W_2d"]
        errors[name] = reconstruction_error(W_2d, rank_map[name])

    names = list(matrices_ref.keys())
    for sweep in range(DMRG_SWEEPS):
        print(f"\n  Sweep {sweep + 1}/{DMRG_SWEEPS}:")
        changes = 0
        for name in names:
            info = matrices_ref[name]
            W_2d = info["W_2d"]
            m, n = W_2d.shape
            max_r = min(m, n)
            cur_r = rank_map[name]
            cur_err = errors[name]

            # Try increasing rank (costs params, reduces error)
            new_r_up = min(cur_r + DMRG_RANK_DELTA, max_r)
            if new_r_up > cur_r:
                extra_params = DMRG_RANK_DELTA * (m + n + 1)
                if total_kept(rank_map) + extra_params <= budget_params:
                    new_err = reconstruction_error(W_2d, new_r_up)
                    if new_err < cur_err:
                        rank_map[name] = new_r_up
                        errors[name] = new_err
                        changes += 1
                        continue

            # Try decreasing rank (frees params, increases error — accept
            # only if another weight benefits more from those params)
            new_r_dn = max(cur_r - DMRG_RANK_DELTA, 1)
            if new_r_dn < cur_r:
                freed = DMRG_RANK_DELTA * (m + n + 1)
                # Find the weight with highest marginal error reduction per param
                best_gain = 0.0
                best_name = None
                for other_name, other_info in matrices_ref.items():
                    if other_name == name:
                        continue
                    om, on = other_info["W_2d"].shape
                    other_r = rank_map[other_name]
                    other_max = min(om, on)
                    if other_r >= other_max:
                        continue
                    try_r = min(other_r + DMRG_RANK_DELTA, other_max)
                    cost = DMRG_RANK_DELTA * (om + on + 1)
                    if cost > freed:
                        continue
                    gain = errors[other_name] - reconstruction_error(
                        matrices_ref[other_name]["W_2d"], try_r
                    )
                    gain_per_param = gain / cost if cost > 0 else 0
                    err_increase = reconstruction_error(W_2d, new_r_dn) - cur_err
                    loss_per_param = err_increase / freed if freed > 0 else 0
                    net = gain_per_param - loss_per_param
                    if net > best_gain:
                        best_gain = net
                        best_name = other_name

                if best_name is not None and best_gain > 0:
                    other_info = matrices_ref[best_name]
                    om, on = other_info["W_2d"].shape
                    new_other_r = min(rank_map[best_name] + DMRG_RANK_DELTA, min(om, on))
                    rank_map[name] = new_r_dn
                    errors[name] = reconstruction_error(W_2d, new_r_dn)
                    rank_map[best_name] = new_other_r
                    errors[best_name] = reconstruction_error(
                        other_info["W_2d"], new_other_r
                    )
                    changes += 2

        print(f"    Changes this sweep: {changes}")
        if changes == 0:
            print("    Converged.")
            break

    # Apply final rank map
    model = copy.deepcopy(base_model)
    compression, kept, total = apply_svd_to_model(model, rank_map)
    print(f"\n  DMRG compression={compression:.2f}x  "
          f"params kept={kept/1e6:.2f}M / {total/1e6:.2f}M")
    ppl = measure_perplexity(model, tokenizer)
    return {"compression": compression, "ppl": ppl, "kept": kept,
            "total": total, "rank_map": rank_map}


# -----------------------------------------------------------------
# Plots
# -----------------------------------------------------------------

def plot_results(baseline_ppl, fixed_results, mp_result, dmrg_result, save_dir):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # --- PPL vs compression ratio ---
    ax = axes[0]
    # Fixed rank
    x_fixed = [r["compression"] for r in fixed_results]
    y_fixed = [r["ppl"] for r in fixed_results]
    ax.plot(x_fixed, y_fixed, "o-", label="Fixed-rank SVD", color="#4477aa", linewidth=2)
    # MP adaptive
    ax.scatter([mp_result["compression"]], [mp_result["ppl"]],
               marker="*", s=200, color="#ee6677", zorder=5, label="MP-adaptive (RMT)")
    # DMRG
    ax.scatter([dmrg_result["compression"]], [dmrg_result["ppl"]],
               marker="D", s=120, color="#228833", zorder=5, label="DMRG sweep")
    # Baseline
    ax.axhline(baseline_ppl, color="gray", linestyle="--", linewidth=1.5, label=f"FP32 baseline ({baseline_ppl:.1f})")

    ax.set_xlabel("Compression ratio (total/kept params)", fontsize=11)
    ax.set_ylabel("Perplexity (↓ better)", fontsize=11)
    ax.set_title("B5: PPL vs Compression", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.set_yscale("log")

    # --- PPL vs rank fraction (fixed only, linear scale) ---
    ax2 = axes[1]
    x_frac = [r["rank_frac"] for r in fixed_results]
    y_frac = [r["ppl"] for r in fixed_results]
    ax2.plot(x_frac, y_frac, "o-", color="#4477aa", linewidth=2, label="Fixed-rank SVD")
    # MP adaptive as effective rank fraction
    # compute effective rank fraction for MP
    mp_fracs = []
    for name, info in mp_result["rank_info"].items():
        mp_fracs.append(info["frac"])
    mean_mp_frac = np.mean(mp_fracs)
    ax2.scatter([mean_mp_frac], [mp_result["ppl"]],
                marker="*", s=200, color="#ee6677", zorder=5, label=f"MP-adaptive (mean frac={mean_mp_frac:.1%})")
    ax2.scatter([mean_mp_frac], [dmrg_result["ppl"]],
                marker="D", s=120, color="#228833", zorder=5, label="DMRG sweep")
    ax2.axhline(baseline_ppl, color="gray", linestyle="--", linewidth=1.5)
    ax2.set_xlabel("Rank fraction (r / min(m,n))", fontsize=11)
    ax2.set_ylabel("Perplexity (↓ better)", fontsize=11)
    ax2.set_title("B5: PPL vs Rank Fraction", fontsize=12, fontweight="bold")
    ax2.legend(fontsize=9)
    ax2.set_yscale("log")

    plt.tight_layout()
    path = os.path.join(save_dir, "b5_dmrg_compression.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def plot_rank_heatmap(mp_result, save_dir):
    rank_info = mp_result["rank_info"]
    # Arrange as (layer, weight_type)
    weight_types = ["attn.c_attn", "attn.c_proj", "mlp.c_fc", "mlp.c_proj"]
    grid = np.zeros((N_LAYERS, len(weight_types)))
    for name, info in rank_info.items():
        for wi, wt in enumerate(weight_types):
            if wt in name:
                layer = int(name.split(".")[1])
                grid[layer, wi] = info["frac"]
                break

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(grid, aspect="auto", cmap="viridis", vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, label="Signal fraction (rank / max_rank)")
    ax.set_xticks(range(len(weight_types)))
    ax.set_xticklabels(weight_types, rotation=30, ha="right", fontsize=9)
    ax.set_yticks(range(N_LAYERS))
    ax.set_yticklabels([f"L{i}" for i in range(N_LAYERS)], fontsize=9)
    ax.set_title("B5: MP-Adaptive Signal Fraction per Weight\n(fraction of SVs above MP bulk edge)", fontsize=11)
    for i in range(N_LAYERS):
        for j in range(len(weight_types)):
            ax.text(j, i, f"{grid[i,j]:.2f}", ha="center", va="center",
                    fontsize=7, color="white" if grid[i,j] < 0.6 else "black")
    plt.tight_layout()
    path = os.path.join(save_dir, "b5_mp_signal_heatmap.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


# -----------------------------------------------------------------
# Main
# -----------------------------------------------------------------

if __name__ == "__main__":
    set_seed(SEED)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    model, tokenizer = load_model_and_tokenizer()

    # FP32 baseline
    print("\n" + "=" * 65)
    print("FP32 baseline")
    print("=" * 65)
    ppl_baseline = measure_perplexity(model, tokenizer)

    # Strategy 1: fixed-rank sweep
    fixed_results = fixed_rank_sweep(model, tokenizer)

    # Strategy 2: MP-adaptive (RMT-filtered bond dims)
    mp_result = mp_adaptive_compression(model, tokenizer)

    # Strategy 3: DMRG sweep initialised from MP-adaptive, same budget
    dmrg_result = dmrg_sweep_compression(
        model, tokenizer,
        mp_rank_map=mp_result["rank_map"],
        budget_params=mp_result["kept"],
    )

    # Results table
    print("\n" + "=" * 65)
    print("FINAL RESULTS -- Experiment B5: DMRG Tensor Compression")
    print("=" * 65)
    print(f"\n  {'Method':<40} {'Compression':>12} {'PPL':>10}")
    print(f"  {'-'*40}  {'-'*12} {'-'*10}")
    print(f"  {'FP32 baseline':<40} {'1.00x':>12} {ppl_baseline:>10.2f}")
    print()
    for r in fixed_results:
        label = f"Fixed SVD rank={r['rank_frac']:.0%}"
        print(f"  {label:<40} {r['compression']:>11.2f}x {r['ppl']:>10.2f}")
    print()
    print(f"  {'MP-adaptive (RMT bond dims)':<40} {mp_result['compression']:>11.2f}x {mp_result['ppl']:>10.2f}")
    print(f"  {'DMRG sweep (from MP)':<40} {dmrg_result['compression']:>11.2f}x {dmrg_result['ppl']:>10.2f}")

    # Key comparison: MP vs fixed at same compression
    target_compression = mp_result["compression"]
    closest_fixed = min(fixed_results, key=lambda r: abs(r["compression"] - target_compression))
    print(f"\n  MP-adaptive vs fixed-rank at similar compression ({closest_fixed['compression']:.2f}x):")
    print(f"  MP-adaptive PPL: {mp_result['ppl']:.2f}  vs  Fixed PPL: {closest_fixed['ppl']:.2f}")
    if mp_result["ppl"] < closest_fixed["ppl"]:
        gain = (closest_fixed["ppl"] - mp_result["ppl"]) / closest_fixed["ppl"] * 100
        print(f"  MP-adaptive is {gain:.1f}% better -> RMT bond dims ARE a useful signal")
    else:
        loss = (mp_result["ppl"] - closest_fixed["ppl"]) / closest_fixed["ppl"] * 100
        print(f"  MP-adaptive is {loss:.1f}% worse -> RMT bond dims do NOT improve on fixed rank")

    # DMRG vs MP
    print(f"\n  DMRG vs MP-adaptive (same budget):")
    print(f"  DMRG PPL: {dmrg_result['ppl']:.2f}  vs  MP PPL: {mp_result['ppl']:.2f}")
    if dmrg_result["ppl"] < mp_result["ppl"]:
        gain = (mp_result["ppl"] - dmrg_result["ppl"]) / mp_result["ppl"] * 100
        print(f"  DMRG sweep is {gain:.1f}% better -> variational sweep DOES improve on MP init")
    else:
        loss = (dmrg_result["ppl"] - mp_result["ppl"]) / mp_result["ppl"] * 100
        print(f"  DMRG sweep is {loss:.1f}% worse -> sweep does NOT improve on MP init")

    # Plots
    print("\nGenerating plots...")
    plot_results(ppl_baseline, fixed_results, mp_result, dmrg_result, RESULTS_DIR)
    plot_rank_heatmap(mp_result, RESULTS_DIR)

    print("\n" + "=" * 65)
    print("B5 complete. Results saved to:", RESULTS_DIR)
    print("=" * 65)
