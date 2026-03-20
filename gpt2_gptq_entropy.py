"""
GPTQ-Calibrated Entropy Quantization Experiment
================================================

Combines minimal GPTQ (Cholesky-based Hessian error compensation) with
entropy-based per-layer bit allocation. Should dramatically improve absolute PPL
vs naive absmax quantization.

Previous absmax results (for comparison):
  absmax-uniform-4bit PPL:  12196.34
  absmax-entropy-4bit PPL:   4730.36

Spec: docs/superpowers/specs/2026-03-20-gptq-entropy-quantization-design.md
"""

import os
import copy
import torch
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from transformers.pytorch_utils import Conv1D  # GPT-2 uses Conv1D, not nn.Linear
from datasets import load_dataset
from entropy_quantization import (
    entropy_linear_allocation,
    uniform_allocation,
    QuantizationPlan,
)

SEED = 42
CALIBRATION_SAMPLES = 128
SEQ_LEN = 1024
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

# Previous experiment results (absmax, for comparison bar chart)
ABSMAX_UNIFORM_PPL = 12196.34
ABSMAX_ENTROPY_PPL = 4730.36


def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)


def load_model_and_tokenizer():
    print("Loading GPT-2 124M...")
    model = GPT2LMHeadModel.from_pretrained("openai-community/gpt2")
    tokenizer = GPT2Tokenizer.from_pretrained("openai-community/gpt2")
    model.eval()
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


# ---------------------------------------------------------------------------
# GPTQ Core
# ---------------------------------------------------------------------------


def symmetric_quantize(w: torch.Tensor, bits: int) -> torch.Tensor:
    """Symmetric per-tensor quantization to n bits."""
    n_levels = 2 ** (bits - 1) - 1
    scale = w.abs().max() / n_levels
    if scale == 0:
        return w.clone()
    q = torch.round(w / scale).clamp(-n_levels, n_levels) * scale
    return q


def gptq_quantize_layer(W: torch.Tensor, H: torch.Tensor, bits: int) -> tuple:
    """
    GPTQ quantization for a single weight matrix.

    Args:
        W: weight matrix (rows x cols), will be modified in-place (copy made internally)
        H: Hessian matrix (cols x cols) = 2 * X.T @ X where X is calibration input
        bits: quantization bit-width

    Returns:
        Q: quantized weight matrix
        total_error: sum of per-column squared errors
    """
    W = W.clone().float()
    rows, cols = W.shape
    Q = torch.zeros_like(W)

    # Dampening for numerical stability
    damp = 0.01 * torch.mean(torch.diag(H))
    H_damped = H.float() + damp * torch.eye(cols, device=H.device, dtype=torch.float32)

    # Cholesky inverse
    try:
        L = torch.linalg.cholesky(H_damped)
        H_inv = torch.cholesky_inverse(L)
    except Exception:
        # Fallback: pseudo-inverse
        H_inv = torch.linalg.pinv(H_damped)

    Losses = torch.zeros(rows, device=W.device)

    for j in range(cols):
        w = W[:, j].clone()
        d = H_inv[j, j]

        # Quantize column j
        q = symmetric_quantize(w, bits)
        Q[:, j] = q

        # Error and compensation
        err = (w - q) / (d + 1e-10)
        Losses += err.pow(2) * d

        # Update remaining unquantized columns
        if j + 1 < cols:
            W[:, j + 1 :] -= err.unsqueeze(1) * H_inv[j, j + 1 :].unsqueeze(0)

    return Q, Losses.sum().item()


# ---------------------------------------------------------------------------
# Hessian Collection
# ---------------------------------------------------------------------------


def collect_hessians(model, samples):
    """
    Collect per-layer Hessians H = X.T @ X (averaged over calibration samples).

    For Conv1D: input shape is (batch, seq_len, in_features).
                Hessian is (in_features x in_features).
    For nn.Linear: input shape is (batch, seq_len, in_features).
                   Hessian is (in_features x in_features).

    Returns:
        hessians: {layer_name: H tensor (in_features x in_features)}
        linear_names: list of layer names
    """
    linear_names = []
    linear_modules = []
    for name, module in model.named_modules():
        if isinstance(module, (torch.nn.Linear, Conv1D)):
            linear_names.append(name)
            linear_modules.append(module)

    print(
        f"Collecting Hessians for {len(linear_names)} layers over {len(samples)} samples..."
    )

    # Determine in_features for each layer
    in_features_map = {}
    for name, module in zip(linear_names, linear_modules):
        if isinstance(module, Conv1D):
            # Conv1D weight shape: (in_features, out_features)
            in_features_map[name] = module.weight.shape[0]
        else:
            # nn.Linear weight shape: (out_features, in_features)
            in_features_map[name] = module.weight.shape[1]

    # Accumulate H = X.T @ X per layer
    hessian_accum = {
        name: torch.zeros(
            in_features_map[name], in_features_map[name], dtype=torch.float64
        )
        for name in linear_names
    }
    n_samples = len(samples)

    def make_hook(module_name):
        def hook(module, input, output):
            x = input[0].detach().float()
            # x shape: (batch, seq_len, in_features) or (batch, in_features)
            if x.dim() == 3:
                # Conv1D or Linear with 3D input: flatten batch and seq dims
                x_flat = x.reshape(-1, x.shape[-1])  # (batch*seq_len, in_features)
            elif x.dim() == 2:
                x_flat = x  # (batch, in_features)
            else:
                x_flat = x.reshape(-1, x.shape[-1])

            # H += X.T @ X
            H_contrib = x_flat.double().T @ x_flat.double()
            hessian_accum[module_name] += H_contrib.cpu()

        return hook

    hooks = []
    for name, module in zip(linear_names, linear_modules):
        hooks.append(module.register_forward_hook(make_hook(name)))

    with torch.no_grad():
        for i, sample in enumerate(samples):
            model(sample)
            if (i + 1) % 32 == 0:
                print(f"  {i + 1}/{n_samples} samples processed")

    for h in hooks:
        h.remove()

    # Average over samples
    hessians = {}
    for name in linear_names:
        H = (hessian_accum[name] / n_samples).float()
        hessians[name] = H

    print("  Hessian collection complete.")
    return hessians, linear_names


# ---------------------------------------------------------------------------
# Entropy Collection
# ---------------------------------------------------------------------------


def collect_entropies(model, samples, n_bins=256, range_estimation_samples=8):
    """
    Collect per-layer activation entropy via streaming histograms.
    Same approach as gpt2_entropy_quantization.py.
    """
    linear_names = []
    linear_modules = []
    for name, module in model.named_modules():
        if isinstance(module, (torch.nn.Linear, Conv1D)):
            linear_names.append(name)
            linear_modules.append(module)

    print(f"Collecting entropies from {len(linear_names)} layers...")

    # Pass 1: estimate activation ranges
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

    # Pass 2: accumulate histograms
    histograms = {n: np.zeros(n_bins, dtype=np.float64) for n in linear_names}

    def hist_hook_fn(module_name):
        def hook(module, input, output):
            x = input[0].detach().float().cpu().numpy().flatten()
            lo, hi = layer_mins[module_name], layer_maxs[module_name]
            if lo >= hi:
                return
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
                print(f"  {i + 1}/{len(samples)} samples processed")

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

    print(
        f"  Entropy range: [{min(entropies.values()):.4f}, {max(entropies.values()):.4f}]"
    )
    return entropies, linear_names


# ---------------------------------------------------------------------------
# GPTQ Model Application
# ---------------------------------------------------------------------------


def apply_gptq_plan(model, plan: QuantizationPlan, hessians: dict) -> torch.nn.Module:
    """
    Deep copy the model and apply GPTQ quantization to each layer
    according to the bit allocation plan.

    H is (in_features x in_features). GPTQ requires W of shape (rows, cols) where
    cols = in_features so that H matches. Therefore:

    For nn.Linear: weight is (out_features, in_features) — use directly, cols=in_features.
    For Conv1D: weight is (in_features, out_features) — transpose to (out_features, in_features),
                run GPTQ, transpose result back to (in_features, out_features).
    """
    q_model = copy.deepcopy(model)
    total_layers = len(plan.layer_bits)
    done = 0

    for name, module in q_model.named_modules():
        if not isinstance(module, (torch.nn.Linear, Conv1D)):
            continue
        if name not in plan.layer_bits:
            continue

        bits = plan.layer_bits[name]
        H = hessians[name]  # (in_features x in_features)

        if isinstance(module, Conv1D):
            # Conv1D weight: (in_features, out_features)
            # Transpose so W = (out_features, in_features), cols=in_features matches H
            W = module.weight.data.float().T  # (out_features, in_features)
            Q, err = gptq_quantize_layer(W, H, bits)
            module.weight.data = Q.T.to(
                module.weight.dtype
            )  # back to (in_features, out_features)
        else:
            # nn.Linear weight: (out_features, in_features) — use directly
            W = module.weight.data.float()  # (out_features, in_features)
            Q, err = gptq_quantize_layer(W, H, bits)
            module.weight.data = Q.to(module.weight.dtype)

        done += 1
        if done % 10 == 0 or done == total_layers:
            print(f"  GPTQ quantized {done}/{total_layers} layers")

    return q_model


# ---------------------------------------------------------------------------
# Perplexity Measurement
# ---------------------------------------------------------------------------


def measure_perplexity(model, tokenizer, split="test"):
    """Measure perplexity on WikiText-2."""
    print(f"  Measuring perplexity on WikiText-2 {split}...")
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
    text = "\n\n".join(dataset["text"])
    encodings = tokenizer(text, return_tensors="pt")
    input_ids = encodings.input_ids

    max_len = model.config.n_positions
    nlls = []
    n_tokens = 0

    model.eval()
    with torch.no_grad():
        for i in range(0, input_ids.size(1) - max_len, max_len):
            chunk = input_ids[:, i : i + max_len]
            outputs = model(chunk, labels=chunk)
            nlls.append(outputs.loss.item() * max_len)
            n_tokens += max_len

    mean_nll = sum(nlls) / n_tokens
    ppl = float(np.exp(mean_nll))
    print(f"    PPL = {ppl:.2f} ({n_tokens:,} tokens)")
    return ppl


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def plot_gptq_comparison(results: dict, save_path: str):
    """
    Bar chart comparing FP32, GPTQ-uniform, GPTQ-entropy, absmax-uniform, absmax-entropy.
    Log scale on y-axis since absmax values are huge.
    """
    method_labels = [
        "FP32\nbaseline",
        "GPTQ\nuniform-4bit",
        "GPTQ\nentropy-4bit",
        "Absmax\nuniform-4bit\n(prev)",
        "Absmax\nentropy-4bit\n(prev)",
    ]
    ppls = [
        results["FP32 baseline"],
        results["GPTQ uniform 4-bit"],
        results["GPTQ entropy-linear 4-bit"],
        ABSMAX_UNIFORM_PPL,
        ABSMAX_ENTROPY_PPL,
    ]
    colors = ["#2ecc71", "#3498db", "#9b59b6", "#e74c3c", "#e67e22"]

    fig, ax = plt.subplots(figsize=(12, 7))
    bars = ax.bar(
        method_labels, ppls, color=colors, alpha=0.85, edgecolor="white", linewidth=1.2
    )
    ax.set_yscale("log")
    ax.set_ylabel("Perplexity (log scale)", fontsize=12)
    ax.set_title(
        "GPT-2 124M: GPTQ vs Absmax Quantization Comparison\n(4-bit avg, WikiText-2 test)",
        fontsize=13,
    )
    ax.axhline(
        y=results["FP32 baseline"],
        color="#2ecc71",
        linestyle="--",
        alpha=0.5,
        linewidth=1.5,
    )

    for bar, ppl in zip(bars, ppls):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() * 1.1,
            f"{ppl:.1f}",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    ax.set_ylim(bottom=max(10, min(ppls) * 0.5))
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Saved: {save_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    set_seed(SEED)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    model, tokenizer = load_model_and_tokenizer()
    cal_samples = load_wikitext2(
        tokenizer, split="train", n_samples=CALIBRATION_SAMPLES
    )

    # -- 1. Collect Hessians --------------------------------------------------
    print("\n" + "=" * 65)
    print("Step 1: Collecting Hessians (calibration data)")
    print("=" * 65)
    hessians, linear_names = collect_hessians(model, cal_samples)

    # -- 2. Collect Entropies -------------------------------------------------
    print("\n" + "=" * 65)
    print("Step 2: Collecting Activation Entropies")
    print("=" * 65)
    entropies, _ = collect_entropies(model, cal_samples)

    # -- 3. Build Allocation Plans --------------------------------------------
    print("\n" + "=" * 65)
    print("Step 3: Building Bit-Width Allocation Plans")
    print("=" * 65)
    TARGET_BITS = 4.0

    plan_uniform = uniform_allocation(linear_names, bits=4)
    plan_entropy = entropy_linear_allocation(
        entropies, min_bits=2, max_bits=8, target_mean_bits=TARGET_BITS
    )

    for plan_name, plan in [
        ("uniform-4bit", plan_uniform),
        ("entropy-linear", plan_entropy),
    ]:
        bits_vals = list(plan.layer_bits.values())
        print(
            f"  {plan_name}: mean={plan.mean_bits:.2f}, "
            f"min={min(bits_vals)}, max={max(bits_vals)}"
        )

    # -- 4. FP32 Baseline Perplexity ------------------------------------------
    print("\n" + "=" * 65)
    print("Step 4: FP32 Baseline Perplexity")
    print("=" * 65)
    ppl_fp32 = measure_perplexity(model, tokenizer)

    # -- 5. GPTQ Uniform 4-bit ------------------------------------------------
    print("\n" + "=" * 65)
    print("Step 5: GPTQ Uniform 4-bit")
    print("=" * 65)
    print("  Applying GPTQ with uniform 4-bit allocation...")
    q_model_uniform = apply_gptq_plan(model, plan_uniform, hessians)
    ppl_gptq_uniform = measure_perplexity(q_model_uniform, tokenizer)
    del q_model_uniform

    # -- 6. GPTQ Entropy 4-bit ------------------------------------------------
    print("\n" + "=" * 65)
    print("Step 6: GPTQ Entropy-Linear 4-bit")
    print("=" * 65)
    print("  Applying GPTQ with entropy-linear allocation...")
    q_model_entropy = apply_gptq_plan(model, plan_entropy, hessians)
    ppl_gptq_entropy = measure_perplexity(q_model_entropy, tokenizer)
    del q_model_entropy

    # -- 7. Results Table -----------------------------------------------------
    results = {
        "FP32 baseline": ppl_fp32,
        "GPTQ uniform 4-bit": ppl_gptq_uniform,
        "GPTQ entropy-linear 4-bit": ppl_gptq_entropy,
    }

    def delta_str(ppl, base):
        d = (ppl - base) / base * 100
        return f"{d:+.1f}%"

    print("\n" + "=" * 65)
    print("Results Summary")
    print("=" * 65)
    header = f"{'Method':<40} {'PPL':>10}  {'Delta vs FP32':>15}"
    sep = "-" * 67
    print(header)
    print(sep)
    print(f"{'FP32 baseline':<40} {ppl_fp32:>10.2f}  {'0.0%':>15}")
    print(
        f"{'GPTQ uniform 4-bit':<40} {ppl_gptq_uniform:>10.2f}  "
        f"{delta_str(ppl_gptq_uniform, ppl_fp32):>15}"
    )
    print(
        f"{'GPTQ entropy-linear 4-bit':<40} {ppl_gptq_entropy:>10.2f}  "
        f"{delta_str(ppl_gptq_entropy, ppl_fp32):>15}"
    )
    print(
        f"{'Absmax uniform 4-bit (prev)':<40} {ABSMAX_UNIFORM_PPL:>10.2f}  "
        f"{delta_str(ABSMAX_UNIFORM_PPL, ppl_fp32):>15}"
    )
    print(
        f"{'Absmax entropy-linear 4-bit (prev)':<40} {ABSMAX_ENTROPY_PPL:>10.2f}  "
        f"{delta_str(ABSMAX_ENTROPY_PPL, ppl_fp32):>15}"
    )

    # GPTQ improvement over absmax
    gptq_uni_vs_absmax_uni = (
        (ABSMAX_UNIFORM_PPL - ppl_gptq_uniform) / ABSMAX_UNIFORM_PPL * 100
    )
    gptq_ent_vs_absmax_ent = (
        (ABSMAX_ENTROPY_PPL - ppl_gptq_entropy) / ABSMAX_ENTROPY_PPL * 100
    )
    entropy_vs_uniform_gptq = (
        (ppl_gptq_uniform - ppl_gptq_entropy) / ppl_gptq_uniform * 100
    )

    print()
    print("=" * 65)
    print("Analysis")
    print("=" * 65)
    print(
        f"  GPTQ uniform vs absmax uniform:  {gptq_uni_vs_absmax_uni:+.1f}% PPL reduction"
    )
    print(
        f"  GPTQ entropy vs absmax entropy:  {gptq_ent_vs_absmax_ent:+.1f}% PPL reduction"
    )
    if entropy_vs_uniform_gptq > 0:
        print(
            f"  Entropy vs uniform (GPTQ):  entropy is {entropy_vs_uniform_gptq:.1f}% better"
        )
    else:
        print(
            f"  Entropy vs uniform (GPTQ):  uniform is {-entropy_vs_uniform_gptq:.1f}% better"
        )

    # -- 8. Plot --------------------------------------------------------------
    print("\nGenerating comparison plot...")
    plot_gptq_comparison(
        results,
        os.path.join(RESULTS_DIR, "gptq_perplexity_comparison.png"),
    )

    print("\n" + "=" * 65)
    print("Experiment complete!")
    print("=" * 65)
