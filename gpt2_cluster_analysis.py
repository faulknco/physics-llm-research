"""
GPT-2 Metastable Cluster Analysis
===================================
Empirically verifies Rigollet 2025 (arXiv 2512.01868) predictions:
- Progressive token clustering across layers
- Metastable multi-cluster states
- Pre-LN polynomial collapse rate: 1-rho ~ 1/t^2

Direction 4 from research plan.
"""

import os
import torch
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from datasets import load_dataset

RESULTS_DIR = "/Users/faulknco/Projects/physics-llm-research/results"
N_SAMPLES = 16
SEQ_LEN = 128
CLUSTER_THRESHOLD = 0.05  # cosine distance < 0.05 = same cluster (similarity > 0.95)


# ──────────────────────────────────────────────────────────────────────────────
# Model loading
# ──────────────────────────────────────────────────────────────────────────────


def load_model():
    print("Loading GPT-2 (124M)...")
    model = GPT2LMHeadModel.from_pretrained("openai-community/gpt2")
    tokenizer = GPT2Tokenizer.from_pretrained("openai-community/gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    model.run_mode = "inference"
    return model, tokenizer


# ──────────────────────────────────────────────────────────────────────────────
# Data & hidden state collection
# ──────────────────────────────────────────────────────────────────────────────


def collect_hidden_states(model, tokenizer, n_samples=N_SAMPLES, seq_len=SEQ_LEN):
    """Run samples through GPT-2, collect hidden states at all layers.

    Returns:
        hidden_states_list: list of n_samples elements;
            each element is a list of 13 tensors shaped (seq_len, d_model),
            one per layer (0 = embedding, 1-12 = transformer blocks).
    """
    print("Loading WikiText-2 dataset...")
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")

    # Concatenate text into a long string and chunk
    full_text = " ".join(t for t in dataset["text"] if t.strip())
    tokens = tokenizer.encode(full_text, add_special_tokens=False)

    # Take n_samples consecutive chunks
    samples = []
    for i in range(n_samples):
        start = i * seq_len
        chunk = tokens[start : start + seq_len]
        if len(chunk) < seq_len:
            break
        samples.append(chunk)

    print(f"Collected {len(samples)} samples x {seq_len} tokens each")

    hidden_states_list = []
    model.eval()
    with torch.no_grad():
        for idx, chunk in enumerate(samples):
            input_ids = torch.tensor([chunk])  # (1, seq_len)
            outputs = model(input_ids, output_hidden_states=True)
            # outputs.hidden_states: tuple of 13 tensors, each (1, seq_len, d_model)
            # Index 0 = embedding output; indices 1-12 = after each transformer block
            layer_states = [hs[0].cpu().float() for hs in outputs.hidden_states]
            # Each layer_states[l] is (seq_len, d_model)
            hidden_states_list.append(layer_states)
            if (idx + 1) % 4 == 0:
                print(f"  Processed {idx + 1}/{len(samples)} samples")

    return hidden_states_list


# ──────────────────────────────────────────────────────────────────────────────
# Per-layer metric computation
# ──────────────────────────────────────────────────────────────────────────────


def cosine_similarity_matrix(h):
    """Compute full pairwise cosine similarity matrix.

    Args:
        h: (seq_len, d_model) float tensor

    Returns:
        sim: (seq_len, seq_len) numpy array
    """
    import torch.nn.functional as F

    # h: (seq_len, d_model)
    sim = F.cosine_similarity(h.unsqueeze(1), h.unsqueeze(0), dim=-1)
    return sim.numpy()


def estimate_clusters(sim_matrix, threshold=CLUSTER_THRESHOLD):
    """Estimate number of clusters via hierarchical clustering.

    Args:
        sim_matrix: (n, n) cosine similarity matrix
        threshold: cosine distance threshold (distance = 1 - similarity)

    Returns:
        n_clusters: int
    """
    n = sim_matrix.shape[0]
    # Convert to distance matrix, clip to [0, 2] for safety
    dist_matrix = np.clip(1.0 - sim_matrix, 0.0, 2.0)
    # Zero out diagonal (self-distance)
    np.fill_diagonal(dist_matrix, 0.0)
    # squareform expects a condensed distance matrix
    condensed = squareform(dist_matrix, checks=False)
    # Avoid degenerate cases
    if np.all(condensed < 1e-10):
        return 1
    Z = linkage(condensed, method="average")
    labels = fcluster(Z, t=threshold, criterion="distance")
    return int(labels.max())


def compute_layer_metrics(hidden_states_list):
    """For each layer, compute mean/std/max cosine similarity and cluster count.

    Returns:
        metrics: dict with keys:
            mean_rho       : (n_layers,) mean rho across samples
            std_rho        : (n_layers,) std rho across samples
            max_rho        : (n_layers,) max pairwise sim across samples
            mean_clusters  : (n_layers,) mean cluster count across samples
            std_clusters   : (n_layers,) std cluster count across samples
            per_sample_rho : (n_samples, n_layers) for detailed analysis
    """
    n_layers = len(hidden_states_list[0])  # 13
    n_samples = len(hidden_states_list)

    per_sample_rho = np.zeros((n_samples, n_layers))
    per_sample_max = np.zeros((n_samples, n_layers))
    per_sample_clusters = np.zeros((n_samples, n_layers))

    print("Computing per-layer metrics...")
    for l in range(n_layers):
        for s_idx, layer_states in enumerate(hidden_states_list):
            h = layer_states[l]  # (seq_len, d_model)
            sim = cosine_similarity_matrix(h)  # (seq_len, seq_len)
            n = sim.shape[0]

            # Mask out diagonal for mean pairwise similarity
            mask = ~np.eye(n, dtype=bool)
            off_diag = sim[mask]
            per_sample_rho[s_idx, l] = off_diag.mean()
            per_sample_max[s_idx, l] = off_diag.max()
            per_sample_clusters[s_idx, l] = estimate_clusters(sim)

        if (l + 1) % 4 == 0 or l == 0:
            print(f"  Layer {l:2d}/{n_layers - 1} done")

    metrics = {
        "mean_rho": per_sample_rho.mean(axis=0),
        "std_rho": per_sample_rho.std(axis=0),
        "max_rho": per_sample_max.mean(axis=0),
        "mean_clusters": per_sample_clusters.mean(axis=0),
        "std_clusters": per_sample_clusters.std(axis=0),
        "per_sample_rho": per_sample_rho,
        "per_sample_clusters": per_sample_clusters,
    }
    return metrics


# ──────────────────────────────────────────────────────────────────────────────
# Polynomial fit helper
# ──────────────────────────────────────────────────────────────────────────────


def fit_polynomial_collapse(mean_rho):
    """Fit 1-rho(t) ~ C/t^2 for t >= 1 (layer index 1 onward).

    Returns:
        C: fitted constant
        r2: R^2 of the fit
        fitted: (n_layers-1,) fitted values of 1-rho
    """
    n_layers = len(mean_rho)
    layers = np.arange(1, n_layers)  # layers 1 through 12
    one_minus_rho = 1.0 - mean_rho[1:]

    # Avoid log(0) or log(<0) issues
    valid = one_minus_rho > 0
    if valid.sum() < 2:
        return np.nan, np.nan, np.full(len(layers), np.nan)

    log_t = np.log(layers[valid])
    log_y = np.log(one_minus_rho[valid])

    # Linear regression: log_y = log_C - 2*log_t
    # We fix slope = -2 per theory; fit only intercept
    log_C = np.mean(log_y + 2.0 * log_t)
    C = np.exp(log_C)

    # Compute R^2 with slope fixed at -2
    fitted_all = C / (layers**2)
    residuals = one_minus_rho[valid] - fitted_all[valid]
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((one_minus_rho[valid] - one_minus_rho[valid].mean()) ** 2)
    r2 = 1.0 - ss_res / (ss_tot + 1e-12)

    return C, r2, fitted_all


# ──────────────────────────────────────────────────────────────────────────────
# Plot 1: Cosine similarity vs depth
# ──────────────────────────────────────────────────────────────────────────────


def plot_cosine_similarity(metrics, hidden_states_list, save_path):
    """Main verification plot: rho vs depth with theoretical overlay + histogram inset."""
    mean_rho = metrics["mean_rho"]
    std_rho = metrics["std_rho"]
    n_layers = len(mean_rho)
    layers = np.arange(n_layers)

    C, r2, fitted_collapse = fit_polynomial_collapse(mean_rho)
    fit_layers = np.arange(1, n_layers)

    fig, (ax_main, ax_inset) = plt.subplots(1, 2, figsize=(14, 5))

    # ── Main panel: rho vs layer ──────────────────────────────────────────────
    ax_main.plot(
        layers,
        mean_rho,
        "o-",
        color="#2563eb",
        lw=2,
        ms=6,
        label=r"GPT-2 mean $\rho(t)$",
        zorder=3,
    )
    ax_main.fill_between(
        layers,
        mean_rho - std_rho,
        mean_rho + std_rho,
        alpha=0.2,
        color="#2563eb",
        label=r"$\pm$ 1 std",
    )

    if not np.isnan(C):
        theory_rho = 1.0 - fitted_collapse
        ax_main.plot(
            fit_layers,
            theory_rho,
            "--",
            color="#dc2626",
            lw=2,
            label=rf"Theory: $1 - \rho \sim 1/t^2$ ($R^2={r2:.3f}$)",
        )

    ax_main.set_xlabel("Layer depth $t$", fontsize=12)
    ax_main.set_ylabel(r"Mean pairwise cosine similarity $\rho(t)$", fontsize=12)
    ax_main.set_title(
        "Token Clustering Across GPT-2 Layers\n"
        "(Rigollet 2025 — Pre-LN polynomial prediction)",
        fontsize=12,
    )
    ax_main.legend(fontsize=10)
    ax_main.set_xlim(-0.3, n_layers - 0.7)
    ax_main.set_ylim(None, 1.05)
    ax_main.grid(True, alpha=0.3)
    ax_main.axhline(1.0, color="gray", lw=0.8, ls=":")

    # Annotate the constant C
    if not np.isnan(C):
        ax_main.text(
            0.98,
            0.05,
            rf"Fit: $1-\rho = {C:.4f}/t^2$",
            transform=ax_main.transAxes,
            ha="right",
            va="bottom",
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7),
        )

    # ── Inset panel: distribution at 4 layers ──────────────────────────────
    snapshot_layers = [0, 4, 8, 12]
    colors_hist = ["#6b7280", "#16a34a", "#d97706", "#dc2626"]
    sample_h = hidden_states_list[0]  # use first sample

    for layer_idx, color in zip(snapshot_layers, colors_hist):
        h = sample_h[layer_idx]
        sim = cosine_similarity_matrix(h)
        n = sim.shape[0]
        mask = ~np.eye(n, dtype=bool)
        off_diag = sim[mask]
        ax_inset.hist(
            off_diag,
            bins=50,
            density=True,
            alpha=0.55,
            color=color,
            label=f"Layer {layer_idx}",
        )

    ax_inset.set_xlabel("Pairwise cosine similarity", fontsize=11)
    ax_inset.set_ylabel("Density", fontsize=11)
    ax_inset.set_title(
        "Distribution of Pairwise Similarities\n(Reproducing Rigollet Fig. 1)",
        fontsize=11,
    )
    ax_inset.legend(fontsize=9)
    ax_inset.set_xlim(-0.1, 1.05)
    ax_inset.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


# ──────────────────────────────────────────────────────────────────────────────
# Plot 2: Cluster count staircase
# ──────────────────────────────────────────────────────────────────────────────


def plot_cluster_count(metrics, save_path):
    """Cluster count staircase plot."""
    mean_clusters = metrics["mean_clusters"]
    std_clusters = metrics["std_clusters"]
    n_layers = len(mean_clusters)
    layers = np.arange(n_layers)

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(
        layers,
        mean_clusters,
        "s-",
        color="#7c3aed",
        lw=2,
        ms=7,
        label="Mean cluster count",
        zorder=3,
    )
    ax.fill_between(
        layers,
        mean_clusters - std_clusters,
        mean_clusters + std_clusters,
        alpha=0.2,
        color="#7c3aed",
        label=r"$\pm$ 1 std",
    )

    # Annotate large drops (staircase events)
    for i in range(1, n_layers):
        drop = mean_clusters[i - 1] - mean_clusters[i]
        if drop > 5:
            ax.annotate(
                f"-{drop:.0f}",
                xy=(i, mean_clusters[i]),
                xytext=(i + 0.2, mean_clusters[i] + 2),
                fontsize=8,
                color="#dc2626",
                arrowprops=dict(arrowstyle="-", color="#dc2626", lw=0.8),
            )

    ax.set_xlabel("Layer depth $t$", fontsize=12)
    ax.set_ylabel("Estimated number of clusters", fontsize=12)
    ax.set_title(
        "Metastable Cluster Count Across GPT-2 Layers\n"
        "Expected: staircase — plateaus then sudden merges (Rigollet Thm 6)",
        fontsize=12,
    )
    ax.legend(fontsize=10)
    ax.set_xlim(-0.3, n_layers - 0.7)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


# ──────────────────────────────────────────────────────────────────────────────
# Plot 3: Pairwise similarity heatmaps
# ──────────────────────────────────────────────────────────────────────────────


def plot_similarity_heatmaps(hidden_states_list, save_path):
    """Pairwise similarity matrices at layers 0, 4, 8, 11 (1x4 grid)."""
    snapshot_layers = [0, 4, 8, 11]
    n_tokens = 64  # use first 64 tokens for readability

    sample_h = hidden_states_list[0]  # first sample

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    fig.suptitle(
        "Pairwise Cosine Similarity Matrices — GPT-2 (first sample, 64 tokens)\n"
        "Block-diagonal structure indicates cluster formation",
        fontsize=12,
    )

    for ax, layer_idx in zip(axes, snapshot_layers):
        h = sample_h[layer_idx][:n_tokens]  # (64, d_model)
        sim = cosine_similarity_matrix(h)

        # Sort tokens by first principal component to reveal cluster block structure
        try:
            u, s, vt = np.linalg.svd(sim, full_matrices=False)
            order = np.argsort(u[:, 0])
            sim = sim[order][:, order]
        except Exception:
            pass

        im = ax.imshow(sim, vmin=0.0, vmax=1.0, cmap="viridis", aspect="auto")
        ax.set_title(f"Layer {layer_idx}", fontsize=11)
        ax.set_xlabel("Token index", fontsize=9)
        ax.set_ylabel("Token index", fontsize=9)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


# ──────────────────────────────────────────────────────────────────────────────
# Console summary
# ──────────────────────────────────────────────────────────────────────────────


def print_summary(metrics):
    """Print results table and Rigollet verification."""
    mean_rho = metrics["mean_rho"]
    std_rho = metrics["std_rho"]
    max_rho = metrics["max_rho"]
    mean_clusters = metrics["mean_clusters"]
    n_layers = len(mean_rho)

    print()
    print("=" * 62)
    print("  Metastable Cluster Analysis -- GPT-2 124M")
    print("  Rigollet 2025 (arXiv 2512.01868) Verification")
    print("=" * 62)
    print(
        f"{'Layer':>5}  {'Mean rho':>8}  {'Std rho':>7}  {'Max rho':>8}  {'Est. Clusters':>13}"
    )
    print("-" * 62)
    for l in range(n_layers):
        print(
            f"{l:>5}  {mean_rho[l]:>8.4f}  {std_rho[l]:>7.4f}  {max_rho[l]:>8.4f}  {mean_clusters[l]:>13.1f}"
        )
    print()

    # ── Monotonic increase check ──────────────────────────────────────────
    diffs = np.diff(mean_rho)
    n_violations = int((diffs < 0).sum())

    C, r2, _ = fit_polynomial_collapse(mean_rho)

    # ── Staircase detection ────────────────────────────────────────────────
    cluster_drops = np.diff(mean_clusters)
    large_drops = (
        np.where(cluster_drops < -5)[0] + 1
    )  # layer indices where big drop occurred
    plateau_segments = []
    min_plateau = 2
    i = 0
    while i < len(mean_clusters) - 1:
        j = i
        while (
            j < len(mean_clusters) - 1
            and abs(mean_clusters[j + 1] - mean_clusters[j]) <= 2
        ):
            j += 1
        if j - i >= min_plateau:
            plateau_segments.append((i, j))
        i = j + 1
    staircase_observed = len(large_drops) > 0 and len(plateau_segments) > 1

    # ── Layer of max cluster diversity ────────────────────────────────────
    max_diversity_layer = int(np.argmax(mean_clusters))

    print("Rigollet 2025 Predictions:")
    print(
        f"  - Monotonic increase in rho:        "
        f"{'CONFIRMED' if n_violations == 0 else f'NEARLY ({n_violations} violations of {n_layers - 1})'}"
    )
    if not np.isnan(r2):
        status = "CONFIRMED" if r2 > 0.7 else ("PARTIAL" if r2 > 0.3 else "FAILED")
        print(f"  - Pre-LN polynomial fit (1-rho~1/t^2): R^2 = {r2:.4f}  [{status}]")
    else:
        print("  - Pre-LN polynomial fit (1-rho~1/t^2): [INSUFFICIENT DATA]")
    print(
        f"  - Staircase in cluster count:     "
        f"{'OBSERVED' if staircase_observed else 'NOT OBSERVED'}"
    )
    if large_drops.size > 0:
        print(f"    Large drops at layers: {list(large_drops.astype(int))}")
    if plateau_segments:
        print(
            f"    Plateaus detected: {len(plateau_segments)} "
            f"(e.g., layers {plateau_segments[0][0]}-{plateau_segments[0][1]})"
        )
    print(f"  - Layer of max cluster diversity: {max_diversity_layer}")
    print()


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    os.makedirs(RESULTS_DIR, exist_ok=True)

    model, tokenizer = load_model()
    hidden_states_list = collect_hidden_states(model, tokenizer, N_SAMPLES, SEQ_LEN)
    metrics = compute_layer_metrics(hidden_states_list)

    print_summary(metrics)

    print("Generating plots...")
    plot_cosine_similarity(
        metrics,
        hidden_states_list,
        os.path.join(RESULTS_DIR, "cluster_cosine_similarity.png"),
    )
    plot_cluster_count(
        metrics,
        os.path.join(RESULTS_DIR, "cluster_count_vs_depth.png"),
    )
    plot_similarity_heatmaps(
        hidden_states_list,
        os.path.join(RESULTS_DIR, "cluster_similarity_heatmap.png"),
    )

    print("\nDone. All results saved to:", RESULTS_DIR)
