"""
GPT-2 RG Flow Analysis
======================
Tests whether transformer layers implement RG coarse-graining by
tracking singular value evolution across layer depth.

Direction 2 from research plan.
"""

import os
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from transformers import GPT2LMHeadModel

RESULTS_DIR = "/Users/faulknco/Projects/physics-llm-research/results"
GROWTH_THRESHOLD = 0.1
N_LAYERS = 12
WEIGHT_TYPES = ["attn.c_attn", "attn.c_proj", "mlp.c_fc", "mlp.c_proj"]


def load_model():
    print("Loading GPT-2 124M...")
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    model.train(False)
    print("Model loaded.")
    return model


def extract_singular_values(model):
    """Returns dict: {weight_type: np.array shape (12, k)} of singular values per layer."""
    sv_dict = {}

    for wt in WEIGHT_TYPES:
        layer_svs = []
        for i in range(N_LAYERS):
            # Navigate to the weight matrix
            parts = wt.split(".")  # e.g. ["attn", "c_attn"]
            module = model.transformer.h[i]
            for part in parts:
                module = getattr(module, part)

            # Conv1D weight shape is (in_features, out_features) — transpose to get standard W
            W = module.weight.detach().float().numpy()
            # W shape: (in_features, out_features) -> transpose so rows = output, cols = input
            W = W.T  # now (out_features, in_features)

            # SVD
            sv = np.linalg.svd(W, compute_uv=False)  # descending order
            layer_svs.append(sv)

        # Pad to same length (should all be same for fixed weight types, but be safe)
        min_k = min(len(sv) for sv in layer_svs)
        sv_matrix = np.array([sv[:min_k] for sv in layer_svs])  # (12, k)

        # Normalize by layer 0's largest singular value
        scale = sv_matrix[0, 0]
        if scale > 0:
            sv_matrix = sv_matrix / scale

        sv_dict[wt] = sv_matrix
        print(
            f"  {wt}: shape {sv_matrix.shape}, max_sv[0]={sv_matrix[0, 0]:.4f}, max_sv[-1]={sv_matrix[-1, 0]:.4f}"
        )

    return sv_dict


def compute_growth_rates(sv_matrix):
    """Compute per-SV growth rate gamma = log(sv[-1]/sv[0]) / (L-1).

    sv_matrix: (12, k) array of singular values (normalized).
    Returns: (k,) array of growth rates.
    """
    L = sv_matrix.shape[0]
    sv_first = sv_matrix[0, :]  # layer 0
    sv_last = sv_matrix[-1, :]  # layer L-1

    # Avoid log(0); clip to small positive
    eps = 1e-10
    sv_first_c = np.clip(sv_first, eps, None)
    sv_last_c = np.clip(sv_last, eps, None)

    gamma = np.log(sv_last_c / sv_first_c) / (L - 1)
    return gamma


def classify_operators(growth_rates):
    """Classify as relevant/irrelevant/marginal based on threshold.

    Returns dict with keys: 'relevant', 'irrelevant', 'marginal' -- boolean masks.
    """
    relevant = growth_rates > GROWTH_THRESHOLD
    irrelevant = growth_rates < -GROWTH_THRESHOLD
    marginal = ~relevant & ~irrelevant
    return {
        "relevant": relevant,
        "irrelevant": irrelevant,
        "marginal": marginal,
    }


def plot_trajectories(sv_dict, growth_dict, save_path):
    """2x2 grid: top-20 SV trajectories per weight type, colored by classification."""
    from matplotlib.lines import Line2D

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    layers = np.arange(N_LAYERS)

    for ax, wt in zip(axes, WEIGHT_TYPES):
        sv_matrix = sv_dict[wt]  # (12, k)
        gammas = growth_dict[wt]  # (k,)
        k = sv_matrix.shape[1]
        n_plot = min(20, k)

        for idx in range(n_plot):
            traj = sv_matrix[:, idx]
            g = gammas[idx]

            if g > GROWTH_THRESHOLD:
                color = "red"
                alpha = 0.7
            elif g < -GROWTH_THRESHOLD:
                color = "blue"
                alpha = 0.5
            else:
                color = "gray"
                alpha = 0.4

            ax.plot(layers, traj, color=color, alpha=alpha, linewidth=1.2)

        legend_elements = [
            Line2D([0], [0], color="red", lw=2, label="relevant (gamma > 0.1)"),
            Line2D([0], [0], color="blue", lw=2, label="irrelevant (gamma < -0.1)"),
            Line2D([0], [0], color="gray", lw=2, label="marginal"),
        ]
        ax.legend(handles=legend_elements, fontsize=7)
        ax.set_title(f"{wt}", fontsize=11)
        ax.set_xlabel("Layer depth")
        ax.set_ylabel("Singular value (normalized)")
        ax.set_xticks(layers)

    fig.suptitle(
        "RG Flow: Singular Value Trajectories (top 20 per weight type)",
        fontsize=13,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


def plot_spectrum_heatmap(sv_dict, save_path):
    """2x2 grid: heatmap of full spectrum per weight type."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for ax, wt in zip(axes, WEIGHT_TYPES):
        sv_matrix = sv_dict[wt]  # (12, k)

        im = ax.imshow(
            sv_matrix,
            aspect="auto",
            origin="upper",
            cmap="viridis",
            interpolation="nearest",
        )
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Normalized SV")
        ax.set_title(f"{wt}", fontsize=11)
        ax.set_xlabel("Singular value index")
        ax.set_ylabel("Layer depth")
        ax.set_yticks(range(N_LAYERS))

    fig.suptitle(
        "RG Spectrum: Full SV Heatmap per Weight Type (layer x SV index)",
        fontsize=13,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


def plot_growth_histogram(growth_dict, save_path):
    """Combined histogram of all growth rates."""
    fig, ax = plt.subplots(figsize=(10, 6))

    colors = ["#e74c3c", "#3498db", "#2ecc71", "#9b59b6"]
    all_gammas = []
    for (wt, gammas), color in zip(growth_dict.items(), colors):
        ax.hist(gammas, bins=40, alpha=0.55, color=color, label=wt, edgecolor="none")
        all_gammas.extend(gammas.tolist())

    # Threshold lines
    ax.axvline(
        x=GROWTH_THRESHOLD,
        color="black",
        linestyle="--",
        linewidth=1.5,
        label=f"+{GROWTH_THRESHOLD} threshold",
    )
    ax.axvline(
        x=-GROWTH_THRESHOLD,
        color="black",
        linestyle=":",
        linewidth=1.5,
        label=f"-{GROWTH_THRESHOLD} threshold",
    )

    ax.set_xlabel("Growth rate gamma = log(sigma_L / sigma_0) / (L-1)", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title(
        "RG Growth Rate Distribution -- All Weight Types",
        fontsize=13,
        fontweight="bold",
    )
    ax.legend(fontsize=9)

    # Annotation: fraction in each zone
    all_gammas_arr = np.array(all_gammas)
    n_total = len(all_gammas_arr)
    n_rel = (all_gammas_arr > GROWTH_THRESHOLD).sum()
    n_irr = (all_gammas_arr < -GROWTH_THRESHOLD).sum()
    n_mar = n_total - n_rel - n_irr
    ax.text(
        0.98,
        0.95,
        f"Relevant:  {100 * n_rel / n_total:.1f}%\n"
        f"Marginal:  {100 * n_mar / n_total:.1f}%\n"
        f"Irrelevant: {100 * n_irr / n_total:.1f}%",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=9,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


def print_summary(sv_dict, growth_dict, class_dict):
    """Print classification summary table."""
    print("\n" + "=" * 70)
    print("RG FLOW ANALYSIS -- GPT-2 124M SUMMARY")
    print("=" * 70)
    print(
        f"{'Weight Type':<20} {'Total SVs':>9} {'Relevant':>9} {'Irrelevant':>10} {'Marginal':>9} {'Mean gamma':>10}"
    )
    print("-" * 70)

    interpretations = {
        "attn.c_attn": "Q/K/V projection (combined)",
        "attn.c_proj": "Attention output projection",
        "mlp.c_fc": "Feed-forward expansion (up)",
        "mlp.c_proj": "Feed-forward contraction (down)",
    }

    for wt in WEIGHT_TYPES:
        gammas = growth_dict[wt]
        classes = class_dict[wt]
        n_total = len(gammas)
        n_rel = classes["relevant"].sum()
        n_irr = classes["irrelevant"].sum()
        n_mar = classes["marginal"].sum()
        mean_g = gammas.mean()

        pct_rel = 100 * n_rel / n_total
        pct_irr = 100 * n_irr / n_total
        pct_mar = 100 * n_mar / n_total

        print(
            f"{wt:<20} {n_total:>9d} {pct_rel:>8.1f}% {pct_irr:>9.1f}% {pct_mar:>8.1f}% {mean_g:>10.4f}"
        )

    print("-" * 70)
    print("\nInterpretations:")
    for wt in WEIGHT_TYPES:
        gammas = growth_dict[wt]
        classes = class_dict[wt]
        n_total = len(gammas)
        n_rel = classes["relevant"].sum()
        n_irr = classes["irrelevant"].sum()
        n_mar = classes["marginal"].sum()
        dominant = max(
            ("relevant", n_rel),
            ("irrelevant", n_irr),
            ("marginal", n_mar),
            key=lambda x: x[1],
        )[0]
        print(f"  {wt:<20} -> {interpretations[wt]}")
        print(
            f"    Dominant class: {dominant}  |  mean gamma = {gammas.mean():.4f}  |  std = {gammas.std():.4f}"
        )

    print("\nPhysics interpretation:")
    print(
        "  gamma > 0.1  -> 'Relevant operator': SV grows across layers -- do NOT prune"
    )
    print(
        "  gamma < -0.1 -> 'Irrelevant operator': SV decays across layers -- pruning candidate"
    )
    print("  |gamma| <= 0.1 -> 'Marginal operator': approximately scale-invariant")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    os.makedirs(RESULTS_DIR, exist_ok=True)

    model = load_model()

    print("\nExtracting singular values across layers...")
    sv_dict = extract_singular_values(model)

    print("\nComputing growth rates...")
    growth_dict = {wt: compute_growth_rates(svs) for wt, svs in sv_dict.items()}

    print("Classifying operators...")
    class_dict = {wt: classify_operators(gr) for wt, gr in growth_dict.items()}

    print_summary(sv_dict, growth_dict, class_dict)

    print("Generating plots...")
    plot_trajectories(
        sv_dict, growth_dict, os.path.join(RESULTS_DIR, "rg_flow_trajectories.png")
    )
    plot_spectrum_heatmap(
        sv_dict, os.path.join(RESULTS_DIR, "rg_spectrum_per_layer.png")
    )
    plot_growth_histogram(growth_dict, os.path.join(RESULTS_DIR, "rg_growth_rates.png"))

    print("\nDone. All results saved to:", RESULTS_DIR)
