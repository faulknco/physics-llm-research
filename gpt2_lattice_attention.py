"""
GPT-2 Lattice Attention Experiment
===================================
Tests physics-motivated attention decay: multiply attention scores by
exp(-|i-j|/xi) before softmax. Sweeps correlation length xi to find
the critical scale where perplexity degrades.

Direction 3 from research plan.
"""

import os
import torch
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from transformers.models.gpt2 import modeling_gpt2 as gpt2_modeling
from datasets import load_dataset

RESULTS_DIR = "/Users/faulknco/Projects/physics-llm-research/results"
XI_VALUES = [16, 32, 64, 128, 256, 512, 1024]

# Global xi for the current patch (None = no lattice mask)
_current_xi = None

# Reference to the AttentionInterface registry
_attn_registry = gpt2_modeling.ALL_ATTENTION_FUNCTIONS


def build_lattice_mask(seq_len, xi, device="cpu"):
    """
    Build causal lattice decay mask.
    mask[i,j] = -|i-j|/xi  for j <= i  (additive in log space)
               = -inf        for j > i  (causal: future tokens masked)

    Shape: (seq_len, seq_len)
    """
    i_idx = torch.arange(seq_len, device=device).unsqueeze(1)  # (L, 1)
    j_idx = torch.arange(seq_len, device=device).unsqueeze(0)  # (1, L)
    dist = torch.abs(i_idx - j_idx).float()

    log_mask = -dist / xi  # exp(-|i-j|/xi) in log space

    # Apply causal constraint: future tokens get -inf
    causal_mask = j_idx > i_idx  # True where j > i
    log_mask = log_mask.masked_fill(causal_mask, float("-inf"))

    return log_mask


def _lattice_eager_attention_forward(
    module, query, key, value, attention_mask, **kwargs
):
    """
    Patched attention forward that adds the lattice log-decay mask
    to attention scores before softmax.
    """
    import torch.nn as nn

    attn_weights = torch.matmul(query, key.transpose(-1, -2))

    if module.scale_attn_weights:
        attn_weights = attn_weights / torch.full(
            [],
            value.size(-1) ** 0.5,
            dtype=attn_weights.dtype,
            device=attn_weights.device,
        )

    if module.scale_attn_by_inverse_layer_idx:
        attn_weights = attn_weights / float(module.layer_idx + 1)

    # Apply lattice decay mask BEFORE the causal/padding attention_mask
    if _current_xi is not None:
        seq_len_q = query.size(-2)
        seq_len_k = key.size(-2)
        # Only patch when q and k have same length (not KV-cache generation)
        if seq_len_q == seq_len_k:
            log_mask = build_lattice_mask(
                seq_len_q, _current_xi, device=attn_weights.device
            )
            # log_mask is (L, L); broadcast to (batch, heads, L, L)
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
    """
    Inject lattice attention into GPT-2.

    Strategy: register our function under the 'eager' key in the
    AttentionInterface._local_mapping.  When GPT-2Attention.forward calls
    ALL_ATTENTION_FUNCTIONS.get_interface('eager', default), the dict lookup
    in _local_mapping returns our function before falling back to the default.
    """
    global _current_xi
    _current_xi = xi
    _attn_registry["eager"] = _lattice_eager_attention_forward


def unpatch_attention():
    """Restore original attention by removing the 'eager' override."""
    global _current_xi
    _current_xi = None
    # Remove the 'eager' entry — get_interface will fall back to the default
    _attn_registry._local_mapping.pop("eager", None)


def measure_perplexity(model, tokenizer, split="test"):
    """Measure perplexity on WikiText-2 using sliding window of model max context."""
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
    text = "\n\n".join(dataset["text"])
    encodings = tokenizer(text, return_tensors="pt")
    input_ids = encodings.input_ids
    max_len = model.config.n_positions  # 1024
    nlls = []
    n_tokens = 0
    with torch.no_grad():
        for i in range(0, input_ids.size(1) - max_len, max_len):
            chunk = input_ids[:, i : i + max_len]
            outputs = model(chunk, labels=chunk)
            nlls.append(outputs.loss.item() * max_len)
            n_tokens += max_len
    mean_nll = sum(nlls) / n_tokens
    return float(np.exp(mean_nll))


def plot_ppl_vs_xi(results, save_path):
    """
    Line plot: x-axis = xi (log scale), y-axis = perplexity.
    Horizontal dashed line = vanilla baseline.
    Mark critical xi where PPL first exceeds baseline by >10%.
    """
    vanilla_ppl = results["vanilla"]
    threshold = vanilla_ppl * 1.10

    xi_vals = sorted([k for k in results.keys() if k != "vanilla"])
    ppls = [results[xi] for xi in xi_vals]

    # Find critical xi: smallest xi where PPL > threshold
    critical_xi = None
    for xi, ppl in zip(xi_vals, ppls):
        if ppl > threshold:
            critical_xi = xi
            break

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(
        xi_vals,
        ppls,
        "o-",
        color="#2563eb",
        linewidth=2,
        markersize=7,
        label="Lattice attention PPL",
    )
    ax.axhline(
        vanilla_ppl,
        color="#dc2626",
        linestyle="--",
        linewidth=1.5,
        label=f"Vanilla baseline ({vanilla_ppl:.2f})",
    )
    ax.axhline(
        threshold,
        color="#f97316",
        linestyle=":",
        linewidth=1.2,
        label=f"+10% threshold ({threshold:.2f})",
    )

    if critical_xi is not None:
        critical_ppl = results[critical_xi]
        ax.axvline(
            critical_xi,
            color="#7c3aed",
            linestyle="-.",
            linewidth=1.5,
            label=f"Critical xi = {critical_xi} (PPL={critical_ppl:.2f})",
        )
        ax.scatter([critical_xi], [critical_ppl], color="#7c3aed", zorder=5, s=80)

    ax.set_xscale("log", base=2)
    ax.set_xticks(xi_vals)
    ax.set_xticklabels([str(xi) for xi in xi_vals])
    ax.set_xlabel("Correlation length xi (tokens, log2 scale)", fontsize=12)
    ax.set_ylabel("Perplexity (WikiText-2 test)", fontsize=12)
    ax.set_title(
        "GPT-2 Lattice Attention: Perplexity vs. Correlation Length xi", fontsize=13
    )
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {save_path}")
    return critical_xi


def plot_mask_patterns(save_path):
    """
    2x2 grid showing causal lattice decay mask for xi = 16, 64, 256, 1024.
    Sequence length 64 for visualization.
    """
    xi_plot = [16, 64, 256, 1024]
    seq_len = 64

    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    axes = axes.flatten()

    for ax, xi in zip(axes, xi_plot):
        log_mask = build_lattice_mask(seq_len, xi)
        # Convert from log space back to actual weights for display
        mask = torch.exp(log_mask)  # upper triangle is 0 (exp(-inf)=0), diagonal is 1
        mask_np = mask.numpy()

        im = ax.imshow(
            mask_np, aspect="auto", origin="upper", cmap="viridis", vmin=0, vmax=1
        )
        ax.set_title(f"xi = {xi}", fontsize=12)
        ax.set_xlabel("Key position j", fontsize=9)
        ax.set_ylabel("Query position i", fontsize=9)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(
        "Causal Lattice Decay Masks: exp(-|i-j|/xi)\n"
        "Sequence length = 64. Upper triangle = 0 (causal masking).",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {save_path}")


def print_results_table(results):
    """Print formatted results table."""
    vanilla_ppl = results["vanilla"]
    xi_vals = sorted([k for k in results.keys() if k != "vanilla"], reverse=True)

    print()
    print(f"{'xi':<10}{'PPL':<12}{'Delta vs Vanilla':<22}{'% Change':<12}")
    print("-" * 56)
    print(f"{'vanilla':<10}{vanilla_ppl:<12.2f}{'0.00':<22}{'0.0%':<12}")
    for xi in xi_vals:
        ppl = results[xi]
        delta = ppl - vanilla_ppl
        pct = (delta / vanilla_ppl) * 100
        delta_str = f"+{delta:.2f}" if delta >= 0 else f"{delta:.2f}"
        pct_str = f"+{pct:.1f}%" if pct >= 0 else f"{pct:.1f}%"
        print(f"{xi:<10}{ppl:<12.2f}{delta_str:<22}{pct_str:<12}")
    print()


if __name__ == "__main__":
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("Loading GPT-2 124M (eager attention mode)...")
    # Force eager attention so our monkey-patch can intercept the forward pass.
    # The default 'sdpa' mode uses a fused PyTorch kernel that bypasses our hook.
    model = GPT2LMHeadModel.from_pretrained("gpt2", attn_implementation="eager")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model.train(False)
    print(f"  attn_implementation: {model.config._attn_implementation}")
    print("Model loaded.\n")

    results = {}

    # Vanilla baseline
    print("Measuring vanilla (no lattice mask) perplexity...")
    unpatch_attention()  # Ensure clean state
    vanilla_ppl = measure_perplexity(model, tokenizer)
    results["vanilla"] = vanilla_ppl
    print(f"  Vanilla PPL = {vanilla_ppl:.4f}\n")

    # Sweep xi values
    for xi in XI_VALUES:
        print(f"Patching attention with xi = {xi}...")
        patch_attention(xi)
        ppl = measure_perplexity(model, tokenizer)
        results[xi] = ppl
        unpatch_attention()
        delta = ppl - vanilla_ppl
        pct = (delta / vanilla_ppl) * 100
        sign = "+" if delta >= 0 else ""
        print(f"  xi={xi:>5}  PPL={ppl:.4f}  ({sign}{delta:.2f}, {sign}{pct:.1f}%)\n")

    # Print table
    print_results_table(results)

    # Plots
    print("Generating plots...")
    ppl_plot_path = os.path.join(RESULTS_DIR, "lattice_attention_ppl_vs_xi.png")
    pattern_plot_path = os.path.join(RESULTS_DIR, "lattice_attention_pattern.png")

    critical_xi = plot_ppl_vs_xi(results, ppl_plot_path)
    plot_mask_patterns(pattern_plot_path)

    if critical_xi is not None:
        print(
            f"\nCritical xi: {critical_xi} tokens (first xi where PPL exceeds baseline by >10%)"
        )
    else:
        print(
            "\nNo critical xi found -- all xi values remain within 10% of vanilla baseline."
        )

    print("\nDone.")
