"""
GPT-2 B1: RMT + RG Joint Structured Head Pruning
=================================================
Prunes attention heads using a joint signal from:
  - RMT (Marchenko-Pastur signal fraction): heads with low MP outliers
    carry little learned structure -- safe to prune
  - RG (growth rate from exp 5): c_attn marginal/decaying heads
    are near fixed-points -- also safe to prune

Joint score = RMT_signal * (1 + RG_growth_normalised)
Heads below a threshold have all Q/K/V weights zeroed (full head pruning).

Sweeps pruning thresholds from 0% to 75% of heads removed.
Measures PPL vs compression ratio at each threshold.

Direction: B1 (RMT-guided structured pruning)
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
N_HEADS = 12
HEAD_DIM = 64
D_MODEL = 768
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

# Pruning fractions to sweep (fraction of 144 total heads removed)
PRUNE_FRACTIONS = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.33, 0.50, 0.75]


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)


# ─────────────────────────────────────────────────────────
# Model + data
# ─────────────────────────────────────────────────────────


def load_model_and_tokenizer():
    print("Loading GPT-2 124M...")
    model = GPT2LMHeadModel.from_pretrained("openai-community/gpt2")
    tokenizer = GPT2Tokenizer.from_pretrained("openai-community/gpt2")
    model.train(False)
    return model, tokenizer


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


# ─────────────────────────────────────────────────────────
# RMT: Marchenko-Pastur signal fraction per head
# ─────────────────────────────────────────────────────────


def marchenko_pastur_edge(gamma):
    """Upper edge of MP bulk: (1 + sqrt(gamma))^2."""
    return (1.0 + np.sqrt(gamma)) ** 2


def mp_signal_fraction(sv, gamma):
    """Fraction of variance explained by outlier SVs above MP bulk edge."""
    edge = marchenko_pastur_edge(gamma)
    # Normalise SVs by largest to compare with MP edge (which assumes unit variance)
    sv_norm = sv / (sv[0] + 1e-10)
    # MP edge in normalised coordinates: scale by mean squared SV
    mean_sv2 = np.mean(sv_norm**2)
    scaled_edge = edge * np.sqrt(mean_sv2)
    outlier_mask = sv_norm > scaled_edge * 1.05  # 5% buffer
    if sv_norm.sum() ** 2 < 1e-12:
        return 0.0
    total_var = float(np.sum(sv**2))
    outlier_var = float(np.sum(sv[outlier_mask] ** 2))
    return outlier_var / (total_var + 1e-10)


def compute_rmt_signals(model):
    """Per-(layer, head) MP signal fraction from Q/K/V weight blocks."""
    print("Computing RMT signal per head...")
    gamma = HEAD_DIM / D_MODEL  # 64/768 = 0.083

    rmt_signal = np.zeros((N_LAYERS, N_HEADS))

    for layer_idx in range(N_LAYERS):
        attn = model.transformer.h[layer_idx].attn
        # c_attn weight: (D_MODEL, 3*D_MODEL) in Conv1D = (768, 2304)
        W_qkv = attn.c_attn.weight.detach().float().numpy()  # (768, 2304)

        for head_idx in range(N_HEADS):
            signals = []
            for proj in range(3):  # Q, K, V
                col_start = proj * D_MODEL + head_idx * HEAD_DIM
                col_end = col_start + HEAD_DIM
                block = W_qkv[:, col_start:col_end]  # (768, 64)
                sv = np.linalg.svd(block, compute_uv=False)
                signals.append(mp_signal_fraction(sv, gamma))
            rmt_signal[layer_idx, head_idx] = np.mean(signals)

    print(f"  RMT signal range: [{rmt_signal.min():.4f}, {rmt_signal.max():.4f}]")
    return rmt_signal


# ─────────────────────────────────────────────────────────
# RG: growth rate per layer from c_attn SVD trajectory
# ─────────────────────────────────────────────────────────


def compute_rg_growth_rates(model):
    """Per-layer RG growth rate for c_attn (mean SV log-ratio across layers).

    Returns (N_LAYERS,) array — same value broadcast to all heads in a layer.
    Positive = relevant (growing), negative = irrelevant (decaying).
    """
    print("Computing RG growth rates...")
    sv_by_layer = []
    for layer_idx in range(N_LAYERS):
        W = model.transformer.h[layer_idx].attn.c_attn.weight.detach().float().numpy()
        W = W.T  # (2304, 768)
        sv = np.linalg.svd(W, compute_uv=False)
        sv_by_layer.append(sv)

    # Normalise by layer 0
    scale = sv_by_layer[0][0]
    sv_matrix = np.array([sv / (scale + 1e-10) for sv in sv_by_layer])  # (12, k)

    # Growth rate: log(sv_last / sv_first) / (L-1), mean over top-k SVs
    L = N_LAYERS
    eps = 1e-10
    sv_first = np.clip(sv_matrix[0], eps, None)
    sv_last = np.clip(sv_matrix[-1], eps, None)
    gamma_per_sv = np.log(sv_last / sv_first) / (L - 1)

    # Per-layer: compute growth rate as layer-to-layer mean change
    layer_growth = np.zeros(N_LAYERS)
    for i in range(N_LAYERS):
        if i == 0:
            layer_growth[i] = gamma_per_sv.mean()
        else:
            prev = np.clip(sv_matrix[i - 1], eps, None)
            curr = np.clip(sv_matrix[i], eps, None)
            layer_growth[i] = np.log(curr / prev).mean()

    print(f"  RG growth range: [{layer_growth.min():.4f}, {layer_growth.max():.4f}]")
    return layer_growth


# ─────────────────────────────────────────────────────────
# Joint pruning score and head zeroing
# ─────────────────────────────────────────────────────────


def compute_joint_scores(rmt_signal, rg_growth):
    """Joint pruning score per (layer, head).

    Low score = safe to prune.
    score = rmt_signal * (1 + rg_growth_norm)
    where rg_growth_norm is per-layer RG rate normalised to [0,1].
    """
    # Normalise RG growth to [0, 1] across layers
    rg_min, rg_max = rg_growth.min(), rg_growth.max()
    if rg_max > rg_min:
        rg_norm = (rg_growth - rg_min) / (rg_max - rg_min)
    else:
        rg_norm = np.ones(N_LAYERS) * 0.5

    # Broadcast layer RG to (N_LAYERS, N_HEADS)
    rg_broadcast = rg_norm[:, np.newaxis] * np.ones((N_LAYERS, N_HEADS))

    # Joint score: both signals must be high to be "important"
    scores = rmt_signal * (1.0 + rg_broadcast)
    return scores


def prune_heads(model, scores, prune_fraction):
    """Zero out Q/K/V weights for the lowest-scoring fraction of heads.

    Returns a deep copy with pruned heads and the list of pruned (layer, head) pairs.
    """
    n_total = N_LAYERS * N_HEADS
    n_prune = int(round(prune_fraction * n_total))

    if n_prune == 0:
        return copy.deepcopy(model), []

    # Flatten scores, find threshold
    flat_scores = scores.flatten()
    threshold = np.sort(flat_scores)[n_prune - 1]

    pruned_heads = []
    for layer_idx in range(N_LAYERS):
        for head_idx in range(N_HEADS):
            if scores[layer_idx, head_idx] <= threshold:
                pruned_heads.append((layer_idx, head_idx))

    # Limit to exactly n_prune (handle ties by taking first n_prune)
    pruned_heads = pruned_heads[:n_prune]
    pruned_set = set(pruned_heads)

    q_model = copy.deepcopy(model)
    for layer_idx in range(N_LAYERS):
        attn = q_model.transformer.h[layer_idx].attn

        # c_attn weight: (768, 2304), c_proj weight: (768, 768)
        W_qkv = attn.c_attn.weight.data  # (768, 2304)
        W_proj = attn.c_proj.weight.data  # (768, 768)

        for head_idx in range(N_HEADS):
            if (layer_idx, head_idx) not in pruned_set:
                continue

            # Zero Q, K, V input projections for this head
            for proj in range(3):
                col_start = proj * D_MODEL + head_idx * HEAD_DIM
                col_end = col_start + HEAD_DIM
                W_qkv[:, col_start:col_end] = 0.0

            # Zero output projection rows for this head
            row_start = head_idx * HEAD_DIM
            row_end = row_start + HEAD_DIM
            W_proj[row_start:row_end, :] = 0.0

    return q_model, pruned_heads


def compression_ratio(n_pruned):
    """Approximate compression from zeroing head weights.

    Each head contributes 4 * D_MODEL * HEAD_DIM params (Q,K,V,O projections).
    Total attention params: N_LAYERS * N_HEADS * 4 * D_MODEL * HEAD_DIM
    """
    params_per_head = 4 * D_MODEL * HEAD_DIM
    total_attn_params = N_LAYERS * N_HEADS * params_per_head
    total_model_params = 124_000_000  # GPT-2 124M
    pruned_params = n_pruned * params_per_head
    # Compression ratio of total model
    return 1.0 / (1.0 - pruned_params / total_model_params)


# ─────────────────────────────────────────────────────────
# Baseline: random pruning for comparison
# ─────────────────────────────────────────────────────────


def prune_heads_random(model, prune_fraction, seed=0):
    """Prune a random fraction of heads (control)."""
    rng = np.random.default_rng(seed)
    n_total = N_LAYERS * N_HEADS
    n_prune = int(round(prune_fraction * n_total))
    all_heads = [(l, h) for l in range(N_LAYERS) for h in range(N_HEADS)]
    chosen_indices = rng.choice(len(all_heads), n_prune, replace=False)
    pruned_heads = [all_heads[int(i)] for i in chosen_indices]
    pruned_set = set(pruned_heads)

    q_model = copy.deepcopy(model)
    for layer_idx in range(N_LAYERS):
        attn = q_model.transformer.h[layer_idx].attn
        W_qkv = attn.c_attn.weight.data
        W_proj = attn.c_proj.weight.data
        for head_idx in range(N_HEADS):
            if (layer_idx, head_idx) not in pruned_set:
                continue
            for proj in range(3):
                col_start = proj * D_MODEL + head_idx * HEAD_DIM
                col_end = col_start + HEAD_DIM
                W_qkv[:, col_start:col_end] = 0.0
            row_start = head_idx * HEAD_DIM
            row_end = row_start + HEAD_DIM
            W_proj[row_start:row_end, :] = 0.0

    return q_model, pruned_heads


# ─────────────────────────────────────────────────────────
# Plots
# ─────────────────────────────────────────────────────────


def plot_score_heatmap(rmt_signal, rg_growth, joint_scores, save_path):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    im0 = axes[0].imshow(rmt_signal, aspect="auto", cmap="RdYlGn", vmin=0)
    axes[0].set_title("RMT: MP signal fraction\n(green = important)", fontsize=10)
    axes[0].set_xlabel("Head")
    axes[0].set_ylabel("Layer")
    plt.colorbar(im0, ax=axes[0])

    rg_2d = rg_growth[:, np.newaxis] * np.ones((N_LAYERS, N_HEADS))
    im1 = axes[1].imshow(rg_2d, aspect="auto", cmap="RdYlGn")
    axes[1].set_title(
        "RG: c_attn growth rate\n(green = relevant operator)", fontsize=10
    )
    axes[1].set_xlabel("Head")
    axes[1].set_ylabel("Layer")
    plt.colorbar(im1, ax=axes[1])

    im2 = axes[2].imshow(joint_scores, aspect="auto", cmap="RdYlGn", vmin=0)
    axes[2].set_title("Joint score (RMT × RG)\nRed = safe to prune", fontsize=10)
    axes[2].set_xlabel("Head")
    axes[2].set_ylabel("Layer")
    plt.colorbar(im2, ax=axes[2])

    fig.suptitle("B1: Head Pruning Scores — GPT-2 124M", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


def plot_ppl_vs_pruning(fractions, rmt_ppls, random_ppls, fp32_ppl, save_path):
    pct = [f * 100 for f in fractions]
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(
        pct,
        rmt_ppls,
        "o-",
        color="#e74c3c",
        linewidth=2,
        markersize=7,
        label="RMT+RG joint pruning",
    )
    ax.plot(
        pct,
        random_ppls,
        "s--",
        color="#95a5a6",
        linewidth=1.5,
        markersize=6,
        label="Random pruning (control)",
    )
    ax.axhline(
        fp32_ppl,
        color="black",
        linestyle=":",
        linewidth=1.5,
        label=f"FP32 ({fp32_ppl:.1f})",
    )
    ax.set_xlabel("Fraction of heads pruned (%)", fontsize=12)
    ax.set_ylabel("Perplexity (WikiText-2 test)", fontsize=12)
    ax.set_title(
        "B1: RMT+RG Structured Pruning — PPL vs Heads Removed",
        fontsize=13,
        fontweight="bold",
    )
    ax.legend(fontsize=10)
    ax.set_yscale("log")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


def plot_pruning_map(joint_scores, pruned_at_thresholds, save_path):
    """Show which heads get pruned at 10%, 25%, 50% thresholds."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    thresholds = [0.10, 0.25, 0.50]
    titles = ["10% pruned", "25% pruned", "50% pruned"]

    for ax, frac, title in zip(axes, thresholds, titles):
        n_total = N_LAYERS * N_HEADS
        n_prune = int(round(frac * n_total))
        flat = joint_scores.flatten()
        thresh = np.sort(flat)[n_prune - 1]
        mask = (joint_scores <= thresh).astype(float)
        # Limit to n_prune
        if mask.sum() > n_prune:
            flat_mask = mask.flatten()
            idx = np.where(flat_mask)[0][n_prune:]
            flat_mask[idx] = 0
            mask = flat_mask.reshape(N_LAYERS, N_HEADS)

        ax.imshow(mask, aspect="auto", cmap="RdYlGn_r", vmin=0, vmax=1)
        ax.set_title(f"{title}\n(red = pruned)", fontsize=10)
        ax.set_xlabel("Head")
        ax.set_ylabel("Layer")
        # Annotate pruned count per layer
        for l in range(N_LAYERS):
            n = int(mask[l].sum())
            if n > 0:
                ax.text(
                    N_HEADS + 0.1, l, str(n), va="center", fontsize=7, color="black"
                )

    fig.suptitle(
        "B1: Which Heads Are Pruned at Each Threshold", fontsize=13, fontweight="bold"
    )
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

    # Compute signals
    rmt_signal = compute_rmt_signals(model)
    rg_growth = compute_rg_growth_rates(model)
    joint_scores = compute_joint_scores(rmt_signal, rg_growth)

    print("\nJoint score summary:")
    print(
        f"  min={joint_scores.min():.4f}  max={joint_scores.max():.4f}  "
        f"mean={joint_scores.mean():.4f}"
    )
    print("  Lowest-score heads (top pruning candidates):")
    flat = [(joint_scores[l, h], l, h) for l in range(N_LAYERS) for h in range(N_HEADS)]
    flat.sort()
    for score, l, h in flat[:10]:
        print(
            f"    Layer {l:2d} Head {h:2d}: score={score:.4f}  "
            f"(RMT={rmt_signal[l, h]:.4f}, RG={rg_growth[l]:.4f})"
        )

    # FP32 baseline
    print("\nFP32 baseline:")
    ppl_fp32 = measure_perplexity(model, tokenizer)

    # Sweep pruning fractions
    print("\n=== Pruning Sweep ===")
    rmt_results = {}
    rand_results = {}

    for frac in PRUNE_FRACTIONS:
        n_prune = int(round(frac * N_LAYERS * N_HEADS))
        print(f"\nPruning {frac * 100:.0f}% of heads ({n_prune}/{N_LAYERS * N_HEADS}):")

        # RMT+RG pruning
        q_rmt, pruned = prune_heads(model, joint_scores, frac)
        ppl_rmt = measure_perplexity(q_rmt, tokenizer)
        rmt_results[frac] = {"ppl": ppl_rmt, "n_pruned": n_prune, "pruned": pruned}
        del q_rmt

        # Random baseline
        q_rand, _ = prune_heads_random(model, frac)
        ppl_rand = measure_perplexity(q_rand, tokenizer)
        rand_results[frac] = {"ppl": ppl_rand}
        del q_rand

        comp = compression_ratio(n_prune)
        print(
            f"  RMT+RG: {ppl_rmt:.2f}  |  Random: {ppl_rand:.2f}  "
            f"|  Compression: {comp:.2f}x"
        )

    # Summary table
    print("\n" + "=" * 70)
    print("B1 RESULTS — RMT+RG Joint Structured Head Pruning")
    print("=" * 70)
    print(
        f"  {'Frac':>6}  {'N pruned':>10}  {'RMT+RG PPL':>12}  "
        f"{'Random PPL':>12}  {'Advantage':>10}  {'Compression':>12}"
    )
    print(f"  {'-' * 6}  {'-' * 10}  {'-' * 12}  {'-' * 12}  {'-' * 10}  {'-' * 12}")
    for frac in PRUNE_FRACTIONS:
        n = rmt_results[frac]["n_pruned"]
        r = rmt_results[frac]["ppl"]
        rd = rand_results[frac]["ppl"]
        adv = (rd - r) / rd * 100  # positive = RMT is better
        comp = compression_ratio(n)
        print(
            f"  {frac:>6.0%}  {n:>10d}  {r:>12.2f}  {rd:>12.2f}  "
            f"{adv:>+10.1f}%  {comp:>12.2f}x"
        )

    print(f"\n  FP32 baseline: {ppl_fp32:.2f}")

    # Plots
    print("\nGenerating plots...")
    plot_score_heatmap(
        rmt_signal,
        rg_growth,
        joint_scores,
        os.path.join(RESULTS_DIR, "b1_score_heatmap.png"),
    )
    plot_ppl_vs_pruning(
        PRUNE_FRACTIONS,
        [rmt_results[f]["ppl"] for f in PRUNE_FRACTIONS],
        [rand_results[f]["ppl"] for f in PRUNE_FRACTIONS],
        ppl_fp32,
        os.path.join(RESULTS_DIR, "b1_ppl_vs_pruning.png"),
    )
    plot_pruning_map(
        joint_scores, rmt_results, os.path.join(RESULTS_DIR, "b1_pruning_map.png")
    )

    print("\nDone. Results in:", RESULTS_DIR)
