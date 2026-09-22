"""
GPT-2 C3: Spectral Gap -> Adaptive KV Cache Truncation
=======================================================
Tests whether the spectral gap of each attention head's transition matrix
predicts how much KV history that head actually needs.

Physics motivation (Markov chain / transfer matrix theory):
  The softmax attention output A[t] is a row-stochastic matrix -- a Markov
  transition kernel. The spectral gap of this kernel is:

      gap = 1 - lambda_2

  where lambda_2 is the second-largest eigenvalue (lambda_1 = 1 always for
  a stochastic matrix). The spectral gap controls the *mixing time* of the
  chain:

      t_mix ~ 1 / gap

  A head with LARGE spectral gap mixes quickly -- it forgets old context fast.
  It only needs short KV history (small context window).

  A head with SMALL spectral gap mixes slowly -- it has long memory.
  It needs long KV history (large context window).

  This gives a principled, per-head context window size:
      context_h = min(T, floor(C / gap_h))
  where C is a calibration constant and T is the sequence length.

  Practically: we truncate the KV cache for each head to its derived
  context window, measuring PPL degradation vs compression ratio.

Comparisons:
  1. Full KV (no truncation) -- baseline
  2. Uniform truncation (all heads same window) -- ablation
  3. Spectral-gap-guided truncation -- our method
  4. Inverted spectral gap (control) -- sanity check

Direction: C3 (new -- extends RMT/spectral analysis into the generation loop)
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
CALIBRATION_SAMPLES = 32
SEQ_LEN = 512
N_LAYERS = 12
N_HEADS = 12
HEAD_DIM = 64
D_MODEL = 768
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

# Context window sizes to sweep for uniform truncation ablation
UNIFORM_WINDOWS = [64, 128, 256, 384, 512]

# Calibration constant for gap -> context conversion
# context_h = min(SEQ_LEN, ceil(C / gap_h))
# C is fit so mean(context_h) matches a target compression level
TARGET_MEAN_CONTEXT = 256  # aim for ~50% KV compression on average


def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)


# -----------------------------------------------------------------
# Model / data
# -----------------------------------------------------------------


def load_model_and_tokenizer(output_attentions=False):
    print("Loading GPT-2 124M...")
    if output_attentions:
        # transformers 5.x: output_attentions must be set in config, not forward() kwargs
        from transformers import GPT2Config

        config = GPT2Config.from_pretrained(
            "openai-community/gpt2", output_attentions=True
        )
        model = GPT2LMHeadModel.from_pretrained("openai-community/gpt2", config=config)
    else:
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
# Spectral gap measurement
# -----------------------------------------------------------------


def compute_spectral_gaps(model, samples):
    """
    Collect attention weight matrices from calibration samples and compute
    the spectral gap of each head's average transition matrix.

    Attention weights A[l,h] are shape (seq_len, seq_len), causal (lower triangular),
    and row-stochastic (each row sums to 1 after masking + softmax).

    For the spectral gap we need the full (non-causal) spectrum. We symmetrize
    the average attention matrix by taking the symmetric part A_sym = (A + A^T) / 2
    and computing eigenvalues. The spectral gap is 1 - lambda_2 / lambda_1.

    Returns:
        gaps: np.ndarray shape (N_LAYERS, N_HEADS) -- spectral gap per head
        attn_means: np.ndarray shape (N_LAYERS, N_HEADS, SEQ_LEN, SEQ_LEN) -- avg attention
    """
    gaps = np.zeros((N_LAYERS, N_HEADS))
    attn_accum = [
        [np.zeros((SEQ_LEN, SEQ_LEN)) for _ in range(N_HEADS)] for _ in range(N_LAYERS)
    ]
    n_collected = [0]

    # Hook into each attention layer
    hooks = []

    def make_hook(layer_idx):
        def hook(module, input, output):
            # GPT-2 attention output: (hidden_states, present, attentions)
            # We need attn_weights which are returned when output_attentions=True
            # We'll capture them via the module's internal state instead
            pass

        return hook

    # Use output_attentions=True at the model level and capture via forward
    print(f"Collecting attention matrices over {len(samples)} samples...")

    for sample_idx, sample in enumerate(samples):
        with torch.no_grad():
            output = model(sample)  # output_attentions already set in config

        # output.attentions: tuple of (N_LAYERS,) each (1, N_HEADS, seq, seq)
        for layer_idx, attn in enumerate(output.attentions):
            attn_np = attn[0].cpu().numpy()  # (N_HEADS, seq, seq)
            for head_idx in range(N_HEADS):
                a = attn_np[head_idx]  # (seq, seq), lower-triangular after causal mask
                # Accumulate
                if a.shape == (SEQ_LEN, SEQ_LEN):
                    attn_accum[layer_idx][head_idx] += a
                else:
                    # Pad or skip if seq shorter than SEQ_LEN (last batch)
                    s = a.shape[0]
                    attn_accum[layer_idx][head_idx][:s, :s] += a

        n_collected[0] += 1
        if (sample_idx + 1) % 8 == 0:
            print(f"  {sample_idx + 1}/{len(samples)} samples")

    print("  Computing spectral gaps...")
    for layer_idx in range(N_LAYERS):
        for head_idx in range(N_HEADS):
            A = attn_accum[layer_idx][head_idx] / n_collected[0]

            # Symmetrize: make it undirected
            A_sym = (A + A.T) / 2.0

            # Remove zero rows/cols (padding)
            row_sums = A_sym.sum(axis=1)
            nonzero = row_sums > 1e-10
            A_sub = A_sym[np.ix_(nonzero, nonzero)]

            if A_sub.shape[0] < 3:
                gaps[layer_idx, head_idx] = 0.5  # fallback
                continue

            # Row-normalize to get stochastic matrix
            row_s = A_sub.sum(axis=1, keepdims=True)
            row_s = np.where(row_s < 1e-12, 1.0, row_s)
            P = A_sub / row_s

            # Eigenvalues of transition matrix
            try:
                # Use symmetric version for numerical stability
                P_sym = (P + P.T) / 2.0
                eigvals = np.linalg.eigvalsh(P_sym)  # sorted ascending
                eigvals = np.sort(eigvals)[::-1]  # descending
                lam1 = eigvals[0]
                lam2 = eigvals[1] if len(eigvals) > 1 else 0.0
                gap = float(lam1 - lam2) / (abs(lam1) + 1e-10)
                gaps[layer_idx, head_idx] = max(0.0, min(1.0, gap))
            except Exception:
                gaps[layer_idx, head_idx] = 0.5

    print(f"  Gap range: [{gaps.min():.4f}, {gaps.max():.4f}]  mean={gaps.mean():.4f}")
    return gaps


def gaps_to_context_windows(gaps, target_mean=TARGET_MEAN_CONTEXT, seq_len=SEQ_LEN):
    """
    Convert per-head spectral gaps to context window sizes.

    context_h = min(seq_len, ceil(C / gap_h))

    We solve for C such that mean(context_h) == target_mean.
    Large gap -> small context (fast mixing head).
    Small gap -> large context (slow mixing head).
    """
    # Avoid divide by zero
    safe_gaps = np.where(gaps < 1e-4, 1e-4, gaps)

    # Binary search for C
    lo, hi = 0.001, float(seq_len)
    for _ in range(60):
        mid = (lo + hi) / 2.0
        windows = np.minimum(seq_len, np.ceil(mid / safe_gaps)).astype(int)
        if windows.mean() < target_mean:
            lo = mid
        else:
            hi = mid

    C = (lo + hi) / 2.0
    windows = np.minimum(seq_len, np.ceil(C / safe_gaps)).astype(int)
    windows = np.clip(windows, 1, seq_len)

    print(
        f"  C={C:.3f}  context window range: [{windows.min()}, {windows.max()}]  mean={windows.mean():.1f}"
    )
    return windows, C


# -----------------------------------------------------------------
# KV cache truncation and PPL measurement
# -----------------------------------------------------------------


def truncate_kv_hook(layer_idx, head_windows):
    """
    Returns a forward hook for a GPT-2 attention block that truncates
    the past KV cache for each head to its assigned window before attention.

    head_windows: array of shape (N_HEADS,) with context window per head.

    Note: GPT-2's attention is implemented as a single fused operation in
    Conv1D. To apply per-head KV truncation we hook into the attention
    module's forward and modify the attention weights after softmax.

    Specifically: for each head h, zero out attention weights to positions
    more than window_h tokens back, then renormalize. This is equivalent
    to not attending to those KV positions.
    """

    def hook(module, input, output):
        # output from GPT-2 attention block is (attn_output, present_kv[, attn_weights])
        # We can't easily modify the fused attention post-hoc for full PPL measurement.
        # Instead we register a pre-forward hook that patches the attention weight matrix.
        pass

    return hook


class SpectralGapKVModel(torch.nn.Module):
    """
    Wraps GPT-2 and applies per-head KV context truncation by masking
    attention scores before softmax.

    For each attention head h in layer l, we mask out attention to positions
    more than window[l,h] tokens back (i.e., keep only the most recent
    window[l,h] positions). This is the KV cache truncation equivalent:
    in generation you'd simply not store/attend to older KV pairs.
    """

    def __init__(self, model, windows):
        super().__init__()
        self.model = copy.deepcopy(model)
        self.windows = windows  # (N_LAYERS, N_HEADS)
        self._hooks = []
        self._install_hooks()

    def _install_hooks(self):
        for layer_idx, block in enumerate(self.model.transformer.h):
            windows_for_layer = self.windows[layer_idx]  # (N_HEADS,)
            hook = block.attn.register_forward_hook(
                self._make_attn_hook(layer_idx, windows_for_layer)
            )
            self._hooks.append(hook)

    def _make_attn_hook(self, layer_idx, head_windows):
        """
        Hook into GPT-2 attention block output to mask long-range attention.

        GPT-2 attention block returns (attn_output, present).
        We cannot easily intercept the raw attention weights post-softmax
        via output hooks without modifying GPT-2 internals.

        Instead: we hook the *input* to the attention block and replace the
        causal attention bias to add a per-head rolling window mask.

        Actually the cleanest approach: patch the attention bias tensor
        in the block before each forward call.
        """

        def hook(module, inputs, output):
            # output is (attn_output, present_kv) or (attn_output, present_kv, attn_weights)
            # attn_output shape: (batch, seq, d_model)
            # We can't re-run attention here without extra cost.
            # Instead: this hook re-runs the masked attention and replaces output.

            # Get the input hidden states
            hidden_states = inputs[0]  # (batch, seq, d_model)
            batch, seq_len, _ = hidden_states.shape

            if seq_len <= 1:
                return output  # single token, no truncation needed

            # Compute Q, K, V from the Conv1D projection
            # GPT-2 Conv1D: weight is (in_features, 3*d_model), bias is (3*d_model,)
            qkv = module.c_attn(hidden_states)  # (batch, seq, 3*d_model)
            q, k, v = qkv.split(D_MODEL, dim=2)

            # Reshape to (batch, heads, seq, head_dim)
            def split_heads(x):
                b, s, d = x.shape
                x = x.view(b, s, N_HEADS, HEAD_DIM)
                return x.permute(0, 2, 1, 3)  # (b, heads, s, hd)

            q = split_heads(q)
            k = split_heads(k)
            v = split_heads(v)

            # Attention scores
            scale = HEAD_DIM**-0.5
            scores = torch.matmul(q, k.transpose(-2, -1)) * scale  # (b, h, s, s)

            # Build per-head window mask: for each head h, allow attention only to
            # positions in [t - window_h + 1, t] for each query position t
            mask = torch.full(
                (seq_len, seq_len),
                float("-inf"),
                device=hidden_states.device,
                dtype=scores.dtype,
            )

            for h in range(N_HEADS):
                w = int(head_windows[h])
                h_mask = torch.full(
                    (seq_len, seq_len),
                    float("-inf"),
                    device=hidden_states.device,
                    dtype=scores.dtype,
                )
                for t in range(seq_len):
                    start = max(0, t - w + 1)
                    h_mask[t, start : t + 1] = 0.0  # allow these positions
                scores[:, h, :, :] = scores[:, h, :, :] + h_mask

            attn_weights = torch.softmax(scores, dim=-1)

            # Handle NaN from rows that are all -inf (shouldn't happen with window>=1)
            attn_weights = torch.nan_to_num(attn_weights, nan=0.0)

            # Apply dropout (use module's attn_dropout if present)
            if hasattr(module, "attn_dropout"):
                attn_weights = module.attn_dropout(attn_weights)

            # Weighted sum over values
            attn_out = torch.matmul(attn_weights, v)  # (b, h, s, hd)

            # Merge heads
            b, h, s, hd = attn_out.shape
            attn_out = attn_out.permute(0, 2, 1, 3).contiguous().view(b, s, h * hd)

            # Output projection
            attn_out = module.c_proj(attn_out)
            if hasattr(module, "resid_dropout"):
                attn_out = module.resid_dropout(attn_out)

            # Return same structure as original output
            if len(output) == 2:
                return (attn_out, output[1])
            else:
                return (attn_out, output[1], output[2])

        return hook

    def remove_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks = []

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)


def measure_perplexity(model, tokenizer, split="test"):
    print(f"  Measuring PPL on WikiText-2 {split}...")
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
    text = "\n\n".join(dataset["text"])
    encodings = tokenizer(text, return_tensors="pt")
    input_ids = encodings.input_ids
    max_len = SEQ_LEN
    nlls, n_tokens = [], 0
    with torch.no_grad():
        for i in range(0, input_ids.size(1) - max_len, max_len):
            chunk = input_ids[:, i : i + max_len]
            if hasattr(model, "model"):
                out = model(chunk, labels=chunk)
            else:
                out = model(chunk, labels=chunk)
            nlls.append(out.loss.item() * max_len)
            n_tokens += max_len
    ppl = float(np.exp(sum(nlls) / n_tokens))
    print(f"    PPL = {ppl:.2f}  ({n_tokens:,} tokens)")
    return ppl


# -----------------------------------------------------------------
# Plots
# -----------------------------------------------------------------


def plot_spectral_gaps(gaps, windows, save_dir):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Heatmap: spectral gaps
    ax = axes[0]
    im = ax.imshow(gaps, aspect="auto", cmap="viridis", origin="upper")
    ax.set_xlabel("Head", fontsize=11)
    ax.set_ylabel("Layer", fontsize=11)
    ax.set_title(
        "Spectral Gap per (Layer, Head)\nLarge gap = fast mixing = short context needed",
        fontsize=10,
    )
    plt.colorbar(im, ax=ax, label="Spectral gap")
    ax.set_xticks(range(N_HEADS))
    ax.set_yticks(range(N_LAYERS))

    # Heatmap: derived context windows
    ax = axes[1]
    im2 = ax.imshow(
        windows, aspect="auto", cmap="RdYlGn", origin="upper", vmin=1, vmax=SEQ_LEN
    )
    ax.set_xlabel("Head", fontsize=11)
    ax.set_ylabel("Layer", fontsize=11)
    ax.set_title(
        f"Derived Context Window per Head\n(target mean={TARGET_MEAN_CONTEXT})",
        fontsize=10,
    )
    plt.colorbar(im2, ax=ax, label="Context window (tokens)")
    ax.set_xticks(range(N_HEADS))
    ax.set_yticks(range(N_LAYERS))

    # Scatter: gap vs context window
    ax = axes[2]
    gap_flat = gaps.flatten()
    win_flat = windows.flatten()
    ax.scatter(
        gap_flat,
        win_flat,
        alpha=0.6,
        s=25,
        c=np.tile(np.arange(N_LAYERS), N_HEADS),
        cmap="tab10",
    )
    ax.set_xlabel("Spectral Gap", fontsize=11)
    ax.set_ylabel("Context Window (tokens)", fontsize=11)
    ax.set_title(
        "Gap -> Context Window Mapping\n(C3: large gap = small window)", fontsize=10
    )

    plt.suptitle(
        "C3: Spectral Gap Analysis -- GPT-2 Attention Heads",
        fontsize=13,
        fontweight="bold",
    )
    plt.tight_layout()
    path = os.path.join(save_dir, "c3_spectral_gap_heatmap.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def plot_ppl_vs_compression(results, save_dir):
    """
    Plot PPL vs mean context window for all methods.
    X axis: mean context window (proxy for KV cache size).
    Y axis: PPL.
    Spectral-gap method should be on the Pareto frontier.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # Uniform truncation sweep
    uni_windows = [r["mean_window"] for r in results["uniform"]]
    uni_ppls = [r["ppl"] for r in results["uniform"]]
    ax.plot(
        uni_windows,
        uni_ppls,
        "o--",
        color="#3498db",
        linewidth=2,
        markersize=8,
        label="Uniform truncation",
    )

    # Spectral gap (our method)
    sg = results["spectral_gap"]
    ax.scatter(
        [sg["mean_window"]],
        [sg["ppl"]],
        color="#2ecc71",
        s=200,
        zorder=5,
        marker="*",
        label=f"Spectral gap (C3)  PPL={sg['ppl']:.0f}",
    )

    # Inverted gap (control)
    inv = results["inverted_gap"]
    ax.scatter(
        [inv["mean_window"]],
        [inv["ppl"]],
        color="#e74c3c",
        s=120,
        zorder=5,
        marker="X",
        label=f"Inverted gap (control)  PPL={inv['ppl']:.0f}",
    )

    # Full KV baseline
    full_ppl = results["full_kv"]["ppl"]
    ax.axhline(
        full_ppl,
        color="black",
        linestyle="--",
        linewidth=1.5,
        label=f"Full KV (no truncation)  PPL={full_ppl:.1f}",
    )

    ax.set_xlabel("Mean Context Window (tokens)", fontsize=12)
    ax.set_ylabel("Perplexity (WikiText-2 test)", fontsize=12)
    ax.set_title(
        "C3: Spectral Gap -> Adaptive KV Cache\nPPL vs Compression Trade-off",
        fontsize=13,
        fontweight="bold",
    )
    ax.legend(fontsize=10)
    plt.tight_layout()
    path = os.path.join(save_dir, "c3_ppl_vs_compression.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def plot_gap_distribution(gaps, save_dir):
    """Per-layer mean gap and per-head gap distribution."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax = axes[0]
    layer_mean_gaps = gaps.mean(axis=1)  # (N_LAYERS,)
    layer_std_gaps = gaps.std(axis=1)
    ax.bar(
        range(N_LAYERS),
        layer_mean_gaps,
        yerr=layer_std_gaps,
        color="steelblue",
        alpha=0.8,
        capsize=4,
    )
    ax.set_xlabel("Layer", fontsize=11)
    ax.set_ylabel("Mean Spectral Gap", fontsize=11)
    ax.set_title(
        "Spectral Gap by Layer\n(higher = faster mixing = shorter context needed)",
        fontsize=10,
    )

    ax = axes[1]
    ax.hist(gaps.flatten(), bins=30, color="coral", alpha=0.8, edgecolor="white")
    ax.axvline(
        gaps.mean(),
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"mean={gaps.mean():.3f}",
    )
    ax.set_xlabel("Spectral Gap", fontsize=11)
    ax.set_ylabel("Count (heads)", fontsize=11)
    ax.set_title("Spectral Gap Distribution\nacross all 144 heads", fontsize=10)
    ax.legend()

    plt.suptitle("C3: Spectral Gap Distribution", fontsize=13, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(save_dir, "c3_gap_distribution.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


# -----------------------------------------------------------------
# Main
# -----------------------------------------------------------------


if __name__ == "__main__":
    set_seed(SEED)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Load model with output_attentions=True for gap calibration only
    attn_model, tokenizer = load_model_and_tokenizer(output_attentions=True)
    cal_samples = load_wikitext2(
        tokenizer, split="train", n_samples=CALIBRATION_SAMPLES
    )

    # -- 1. Compute spectral gaps -----------------------------------------
    print("\n" + "=" * 65)
    print("Step 1: Computing per-head spectral gaps")
    print("=" * 65)
    gaps = compute_spectral_gaps(attn_model, cal_samples)
    del attn_model

    # Load clean model (no attention output overhead) for all PPL evals
    model, _ = load_model_and_tokenizer()
    np.save(os.path.join(RESULTS_DIR, "c3_spectral_gaps.npy"), gaps)

    print("\nSpectral gap summary:")
    print(
        f"  Overall:  min={gaps.min():.4f}  max={gaps.max():.4f}  mean={gaps.mean():.4f}"
    )
    for l in range(N_LAYERS):
        print(
            f"  Layer {l:2d}: min={gaps[l].min():.4f}  max={gaps[l].max():.4f}  mean={gaps[l].mean():.4f}"
        )

    # -- 2. Derive context windows ----------------------------------------
    print("\n" + "=" * 65)
    print("Step 2: Deriving per-head context windows from spectral gaps")
    print("=" * 65)
    windows, C = gaps_to_context_windows(gaps, target_mean=TARGET_MEAN_CONTEXT)
    print(f"  Calibration constant C={C:.3f}")
    print(
        f"  Mean context window: {windows.mean():.1f} / {SEQ_LEN} tokens ({windows.mean() / SEQ_LEN * 100:.1f}%)"
    )

    # Inverted windows (control): invert the gap -> large gap gets large window
    windows_inv, _ = gaps_to_context_windows(
        1.0 - gaps + gaps.min(), target_mean=TARGET_MEAN_CONTEXT
    )

    # -- 3. Plots: gap analysis -------------------------------------------
    print("\nGenerating gap analysis plots...")
    plot_spectral_gaps(gaps, windows, RESULTS_DIR)
    plot_gap_distribution(gaps, RESULTS_DIR)

    # -- 4. Full KV baseline PPL ------------------------------------------
    print("\n" + "=" * 65)
    print("Step 3: Full KV baseline (no truncation)")
    print("=" * 65)
    ppl_full = measure_perplexity(model, tokenizer)

    # -- 5. Spectral-gap adaptive KV truncation ---------------------------
    print("\n" + "=" * 65)
    print("Step 4: Spectral-gap adaptive KV truncation")
    print("=" * 65)
    print(f"  Mean context window: {windows.mean():.1f} tokens")
    sg_model = SpectralGapKVModel(model, windows)
    ppl_sg = measure_perplexity(sg_model, tokenizer)
    sg_model.remove_hooks()
    del sg_model

    # -- 6. Inverted gap (control) ----------------------------------------
    print("\n" + "=" * 65)
    print("Step 5: Inverted spectral gap (control)")
    print("=" * 65)
    print(f"  Mean context window: {windows_inv.mean():.1f} tokens")
    inv_model = SpectralGapKVModel(model, windows_inv)
    ppl_inv = measure_perplexity(inv_model, tokenizer)
    inv_model.remove_hooks()
    del inv_model

    # -- 7. Uniform truncation sweep --------------------------------------
    print("\n" + "=" * 65)
    print("Step 6: Uniform truncation sweep")
    print("=" * 65)
    uniform_results = []
    for w in UNIFORM_WINDOWS:
        print(f"\n  Uniform window = {w} tokens:")
        uni_windows = np.full((N_LAYERS, N_HEADS), w)
        uni_model = SpectralGapKVModel(model, uni_windows)
        ppl_uni = measure_perplexity(uni_model, tokenizer)
        uni_model.remove_hooks()
        del uni_model
        uniform_results.append({"window": w, "mean_window": float(w), "ppl": ppl_uni})

    # -- 8. Results table -------------------------------------------------
    print("\n" + "=" * 65)
    print("FINAL RESULTS -- Experiment C3: Spectral Gap -> Adaptive KV Cache")
    print("=" * 65)

    def delta(ppl, base):
        d = (ppl - base) / base * 100
        direction = "better" if d < 0 else "worse"
        return f"{abs(d):.1f}% {direction}"

    print(f"\n  {'Method':<35} {'Mean window':>12} {'PPL':>10}  {'vs full KV':>15}")
    print(f"  {'-' * 35}  {'-' * 12} {'-' * 10}  {'-' * 15}")
    print(
        f"  {'Full KV (no truncation)':<35} {SEQ_LEN:>12} {ppl_full:>10.2f}  {'(baseline)':>15}"
    )
    print(
        f"  {'Spectral gap (C3)':<35} {windows.mean():>12.1f} {ppl_sg:>10.2f}  {delta(ppl_sg, ppl_full):>15}"
    )
    print(
        f"  {'Inverted gap (control)':<35} {windows_inv.mean():>12.1f} {ppl_inv:>10.2f}  {delta(ppl_inv, ppl_full):>15}"
    )
    print()
    for r in uniform_results:
        print(
            f"  {'Uniform w=' + str(r['window']):<35} {r['mean_window']:>12.1f} {r['ppl']:>10.2f}  {delta(r['ppl'], ppl_full):>15}"
        )

    # Find uniform baseline at same mean window as spectral gap
    sg_mean_w = windows.mean()
    closest_uni = min(uniform_results, key=lambda r: abs(r["mean_window"] - sg_mean_w))
    print(f"\n  Spectral gap vs uniform at similar window ({closest_uni['window']}):")
    print(
        f"  Spectral gap PPL: {ppl_sg:.2f}  vs  Uniform PPL: {closest_uni['ppl']:.2f}"
    )
    if ppl_sg < closest_uni["ppl"]:
        gain = (closest_uni["ppl"] - ppl_sg) / closest_uni["ppl"] * 100
        print(
            f"  Spectral gap is {gain:.1f}% better -> gap IS a useful KV allocation signal"
        )
    else:
        loss = (ppl_sg - closest_uni["ppl"]) / closest_uni["ppl"] * 100
        print(
            f"  Spectral gap is {loss:.1f}% worse -> gap does NOT improve on uniform truncation"
        )

    # Verdict: does inverted control perform worse? (sanity check)
    if ppl_inv > ppl_sg:
        print(
            f"\n  Sanity check PASSED: inverted gap ({ppl_inv:.0f}) > spectral gap ({ppl_sg:.0f})"
        )
    else:
        print(
            f"\n  Sanity check FAILED: inverted gap ({ppl_inv:.0f}) <= spectral gap ({ppl_sg:.0f})"
        )
        print("  The spectral gap signal is NOT directionally correct.")

    # -- 9. Final plots ---------------------------------------------------
    all_results = {
        "full_kv": {"ppl": ppl_full, "mean_window": SEQ_LEN},
        "spectral_gap": {"ppl": ppl_sg, "mean_window": float(windows.mean())},
        "inverted_gap": {"ppl": ppl_inv, "mean_window": float(windows_inv.mean())},
        "uniform": uniform_results,
    }
    print("\nGenerating result plots...")
    plot_ppl_vs_compression(all_results, RESULTS_DIR)

    print("\n" + "=" * 65)
    print("C3 complete. All results saved to:", RESULTS_DIR)
    print("=" * 65)
