"""
GPT-2 KV Cache Entropy Quantization Experiment
===============================================

Extends Direction 1 (Thermodynamic Quantization) from weight activations to
the KV cache tensors -- the Key and Value matrices that accumulate during
autoregressive generation.

Research Questions
------------------
1. Do Keys and Values have different entropy profiles per layer/head?
2. Does per-head KV entropy predict attention score degradation under quantization?
3. Does entropy-guided KV bit allocation beat uniform KV quantization on PPL?
4. Do Keys have higher entropy than Values? (KVQuant finds this empirically --
   we test whether entropy gives it a thermodynamic motivation.)
5. Does our RG-informed layer sensitivity (earlier = more critical) align
   with which layers have highest KV entropy?
"""

import os
import copy
import torch
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats as scipy_stats
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from datasets import load_dataset

SEED = 42
CALIBRATION_SAMPLES = 64
SEQ_LEN = 512
N_LAYERS = 12
N_HEADS = 12
HEAD_DIM = 64
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)


# ---------------------------------------------------------------------------
# Model + Data
# ---------------------------------------------------------------------------

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
# KV Entropy Collection
# ---------------------------------------------------------------------------

def _tensor_entropy(t: torch.Tensor, n_bins: int = 256) -> float:
    """Shannon entropy of a tensor's value distribution."""
    x = t.detach().float().cpu().numpy().flatten()
    lo = float(np.percentile(x, 1))
    hi = float(np.percentile(x, 99))
    if lo >= hi:
        return 0.0
    counts, _ = np.histogram(x, bins=n_bins, range=(lo, hi))
    total = counts.sum()
    if total == 0:
        return 0.0
    probs = counts / total
    probs = probs[probs > 0]
    return float(-np.sum(probs * np.log(probs)))


def collect_kv_entropies(model, samples):
    """
    Hook into each attention block to capture Key and Value tensors after
    the QKV split but before softmax.

    Returns:
        key_entropies: np.ndarray shape (n_layers, n_heads)
        val_entropies: np.ndarray shape (n_layers, n_heads)
    """
    key_ent_sum = np.zeros((N_LAYERS, N_HEADS), dtype=np.float64)
    val_ent_sum = np.zeros((N_LAYERS, N_HEADS), dtype=np.float64)
    count = np.zeros((N_LAYERS, N_HEADS), dtype=np.int64)

    original_forwards = {}

    def make_probe_hook(layer_idx):
        # Capture KV entropy without changing output -- run original, then probe
        original_fwd = None

        def patched_forward(self, hidden_states, **kwargs):
            # Split QKV manually to probe K and V
            query_states, key_states, value_states = self.c_attn(hidden_states).split(
                self.split_size, dim=2
            )
            shape_kv = (*key_states.shape[:-1], -1, HEAD_DIM)
            k = key_states.view(shape_kv).transpose(1, 2)   # (B, H, T, D)
            v = value_states.view(shape_kv).transpose(1, 2)

            for h in range(k.shape[1]):
                key_ent_sum[layer_idx, h] += _tensor_entropy(k[:, h, :, :])
                val_ent_sum[layer_idx, h] += _tensor_entropy(v[:, h, :, :])
                count[layer_idx, h] += 1

            # Call the original unpatched forward
            return original_forwards[layer_idx](self, hidden_states, **kwargs)

        return patched_forward

    for layer_idx, block in enumerate(model.transformer.h):
        attn = block.attn
        original_forwards[layer_idx] = type(attn).forward
        type(attn).forward = make_probe_hook(layer_idx)

    print(f"Collecting KV entropies over {len(samples)} samples...")
    with torch.no_grad():
        for i, sample in enumerate(samples):
            model(sample)
            if (i + 1) % 16 == 0:
                print(f"  {i + 1}/{len(samples)} samples processed")

    for layer_idx, block in enumerate(model.transformer.h):
        type(block.attn).forward = original_forwards[layer_idx]

    safe_count = np.maximum(count, 1)
    key_entropies = key_ent_sum / safe_count
    val_entropies = val_ent_sum / safe_count

    print(f"  Key entropy range:   [{key_entropies.min():.3f}, {key_entropies.max():.3f}]")
    print(f"  Value entropy range: [{val_entropies.min():.3f}, {val_entropies.max():.3f}]")
    print(f"  Mean key entropy:    {key_entropies.mean():.3f}")
    print(f"  Mean value entropy:  {val_entropies.mean():.3f}")

    return key_entropies, val_entropies


# ---------------------------------------------------------------------------
# Bit Allocation Plans
# ---------------------------------------------------------------------------

def _entropy_to_bits(entropy_matrix, min_bits=2, max_bits=8, target_mean=4.0):
    """Linear scaling of entropy values to integer bit-widths."""
    e_min, e_max = entropy_matrix.min(), entropy_matrix.max()
    if e_max == e_min:
        normalized = np.ones_like(entropy_matrix) * 0.5
    else:
        normalized = (entropy_matrix - e_min) / (e_max - e_min)
    raw_bits = min_bits + normalized * (max_bits - min_bits)
    current_mean = raw_bits.mean()
    if current_mean > 0:
        raw_bits = np.clip(raw_bits * (target_mean / current_mean), min_bits, max_bits)
    return np.round(raw_bits).astype(int)


def build_allocation_plans(key_entropies, val_entropies, target_bits=4.0):
    """Build three KV quantization plans."""
    # Plan A: uniform
    plan_uniform = {
        "name": "Uniform 4-bit",
        "head_key_bits": np.full((N_LAYERS, N_HEADS), 4, dtype=int),
        "head_val_bits": np.full((N_LAYERS, N_HEADS), 4, dtype=int),
    }

    # Plan B: per-layer entropy (mean across heads)
    layer_joint_ent = (key_entropies.mean(axis=1) + val_entropies.mean(axis=1)) / 2.0
    layer_bits = _entropy_to_bits(layer_joint_ent, target_mean=target_bits)
    plan_layer = {
        "name": "Entropy per-layer",
        "head_key_bits": np.tile(layer_bits[:, None], (1, N_HEADS)),
        "head_val_bits": np.tile(layer_bits[:, None], (1, N_HEADS)),
    }

    # Plan C: per-head entropy, asymmetric K vs V
    plan_head_asym = {
        "name": "Entropy per-head (K!=V)",
        "head_key_bits": _entropy_to_bits(key_entropies, target_mean=target_bits),
        "head_val_bits": _entropy_to_bits(val_entropies, target_mean=target_bits),
    }

    print("Allocation plan summary:")
    for plan in [plan_uniform, plan_layer, plan_head_asym]:
        k_mean = plan["head_key_bits"].mean()
        v_mean = plan["head_val_bits"].mean()
        print(f"  {plan['name']:<40}  K_avg={k_mean:.2f}  V_avg={v_mean:.2f}")

    return plan_uniform, plan_layer, plan_head_asym


# ---------------------------------------------------------------------------
# KV Quantization + PPL Measurement
# ---------------------------------------------------------------------------

def _absmax_quantize(t: torch.Tensor, bits: int) -> torch.Tensor:
    """Symmetric absmax quantization."""
    if bits >= 16:
        return t
    n_levels = 2 ** (bits - 1) - 1
    scale = t.abs().max()
    if scale == 0:
        return t
    return torch.round(t / scale * n_levels).clamp(-n_levels, n_levels) * scale / n_levels


def measure_ppl_with_kv_quant(model, tokenizer, plan):
    """
    Measure WikiText-2 PPL with KV tensors quantized per plan.
    Patches each attention block to quantize K and V before the attention
    score computation, simulating an online KV cache compressor.
    """
    head_key_bits = plan["head_key_bits"]
    head_val_bits = plan["head_val_bits"]
    original_forwards = {}

    def make_quant_forward(layer_idx):
        def patched_forward(self, hidden_states, **kwargs):
            # QKV split
            query_states, key_states, value_states = self.c_attn(hidden_states).split(
                self.split_size, dim=2
            )
            shape_kv = (*key_states.shape[:-1], -1, HEAD_DIM)
            k = key_states.view(shape_kv).transpose(1, 2)   # (B, H, T, D)
            v = value_states.view(shape_kv).transpose(1, 2)
            shape_q = (*query_states.shape[:-1], -1, HEAD_DIM)
            q = query_states.view(shape_q).transpose(1, 2)

            # Quantize each head's K and V
            B, H, T, D = k.shape
            k_q = torch.zeros_like(k)
            v_q = torch.zeros_like(v)
            for h in range(H):
                k_q[:, h] = _absmax_quantize(k[:, h], int(head_key_bits[layer_idx, h]))
                v_q[:, h] = _absmax_quantize(v[:, h], int(head_val_bits[layer_idx, h]))

            # Manual causal self-attention with quantized K/V
            scale = HEAD_DIM ** -0.5
            attn_weights = torch.matmul(q, k_q.transpose(-1, -2)) * scale
            causal_mask = torch.tril(
                torch.ones(T, T, dtype=torch.bool, device=attn_weights.device)
            )
            attn_weights = attn_weights.masked_fill(~causal_mask, float("-inf"))
            attn_weights = torch.softmax(attn_weights, dim=-1)
            attn_weights = torch.nan_to_num(attn_weights, nan=0.0)
            attn_output = torch.matmul(attn_weights, v_q)             # (B, H, T, D)
            attn_output = attn_output.transpose(1, 2).contiguous()
            attn_output = attn_output.view(B, T, H * D)
            attn_output = self.c_proj(attn_output)
            attn_output = self.resid_dropout(attn_output)
            return attn_output, None

        return patched_forward

    for layer_idx, block in enumerate(model.transformer.h):
        original_forwards[layer_idx] = type(block.attn).forward
        type(block.attn).forward = make_quant_forward(layer_idx)

    print(f"  PPL [{plan['name']}]...")
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(dataset["text"])
    input_ids = tokenizer(text, return_tensors="pt").input_ids

    nlls, n_tokens = [], 0
    model.eval()
    with torch.no_grad():
        for i in range(0, min(input_ids.size(1) - SEQ_LEN, SEQ_LEN * 50), SEQ_LEN):
            chunk = input_ids[:, i: i + SEQ_LEN]
            try:
                out = model(chunk, labels=chunk)
                nlls.append(out.loss.item() * SEQ_LEN)
                n_tokens += SEQ_LEN
            except Exception as exc:
                print(f"    chunk {i} skipped: {exc}")

    for layer_idx, block in enumerate(model.transformer.h):
        type(block.attn).forward = original_forwards[layer_idx]

    if n_tokens == 0:
        return float("inf")
    ppl = float(np.exp(sum(nlls) / n_tokens))
    print(f"    PPL = {ppl:.2f}  ({n_tokens:,} tokens)")
    return ppl


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_kv_entropy_landscape(key_entropies, val_entropies, save_path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, data, title, cmap in zip(
        axes,
        [key_entropies, val_entropies],
        ["Key Entropy (nats)", "Value Entropy (nats)"],
        ["Blues", "Oranges"],
    ):
        im = ax.imshow(data, aspect="auto", cmap=cmap)
        ax.set_xlabel("Attention Head")
        ax.set_ylabel("Layer")
        ax.set_title(title)
        ax.set_xticks(range(N_HEADS))
        ax.set_yticks(range(N_LAYERS))
        plt.colorbar(im, ax=ax)
    plt.suptitle("GPT-2 124M: KV Tensor Entropy per (Layer, Head)", fontsize=13)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Saved: {save_path}")


def plot_key_vs_value_entropy(key_entropies, val_entropies, save_path):
    k_flat = key_entropies.flatten()
    v_flat = val_entropies.flatten()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    ax.scatter(k_flat, v_flat, alpha=0.5, s=20, c="steelblue")
    lim_min = min(k_flat.min(), v_flat.min()) * 0.95
    lim_max = max(k_flat.max(), v_flat.max()) * 1.05
    ax.plot([lim_min, lim_max], [lim_min, lim_max], "r--", alpha=0.5, label="K=V")
    ax.set_xlabel("Key Entropy (nats)")
    ax.set_ylabel("Value Entropy (nats)")
    ax.set_title("Key vs Value Entropy per Head")
    ax.legend()
    r, p = scipy_stats.pearsonr(k_flat, v_flat)
    ax.text(0.05, 0.95, f"Pearson r={r:.3f}\n(p={p:.2e})", transform=ax.transAxes,
            verticalalignment="top", fontsize=9,
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    ax2 = axes[1]
    layers = np.arange(N_LAYERS)
    ax2.plot(layers, key_entropies.mean(axis=1), "o-", color="steelblue", label="Mean Key")
    ax2.plot(layers, val_entropies.mean(axis=1), "s-", color="coral", label="Mean Value")
    ax2.set_xlabel("Layer")
    ax2.set_ylabel("Mean Entropy (nats)")
    ax2.set_title("Per-Layer Mean KV Entropy")
    ax2.legend()
    ax2.grid(alpha=0.3)

    plt.suptitle("GPT-2 124M: Key vs Value Entropy", fontsize=13)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Saved: {save_path}")


def plot_bit_allocation(plan_layer, plan_head_asym, save_path):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    titles = ["Per-Layer Plan (K bits)", "Per-Head Key bits", "Per-Head Value bits"]
    data_list = [
        plan_layer["head_key_bits"],
        plan_head_asym["head_key_bits"],
        plan_head_asym["head_val_bits"],
    ]
    for ax, data, title in zip(axes, data_list, titles):
        im = ax.imshow(data, aspect="auto", cmap="RdYlGn", vmin=2, vmax=8)
        ax.set_xlabel("Head")
        ax.set_ylabel("Layer")
        ax.set_title(title)
        plt.colorbar(im, ax=ax, label="bits")
    plt.suptitle("KV Bit-Width Allocation Plans (entropy-guided, 4-bit avg)", fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Saved: {save_path}")


def plot_ppl_comparison(results, save_path):
    methods = list(results.keys())
    ppls = [results[m] for m in methods]
    colors = ["#95a5a6", "#3498db", "#9b59b6", "#2ecc71"][:len(methods)]
    fig, ax = plt.subplots(figsize=(9, 6))
    bars = ax.bar(methods, ppls, color=colors, alpha=0.85, edgecolor="white", linewidth=1.2)
    fp32_ppl = results.get("FP32 baseline")
    if fp32_ppl:
        ax.axhline(y=fp32_ppl, color="green", linestyle="--", alpha=0.5,
                   label=f"FP32 ({fp32_ppl:.1f})")
    for bar, ppl in zip(bars, ppls):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{ppl:.1f}", ha="center", fontsize=10, fontweight="bold")
    ax.set_ylabel("Perplexity (WikiText-2 test)")
    ax.set_title("GPT-2 KV Cache Quantization\nUniform vs Entropy-Guided (4-bit avg)")
    ax.legend()
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels(methods, rotation=15, ha="right")
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
    cal_samples = load_wikitext2(tokenizer, split="train", n_samples=CALIBRATION_SAMPLES)

    print("\n" + "=" * 65)
    print("Step 1: Profiling KV Entropy (per layer, per head)")
    print("=" * 65)
    key_entropies, val_entropies = collect_kv_entropies(model, cal_samples)

    print("\nPer-layer mean entropies:")
    print(f"  {'Layer':<6}  {'Key':>8}  {'Value':>8}  {'K>V?':>6}")
    print("  " + "-" * 34)
    for i in range(N_LAYERS):
        k = key_entropies[i].mean()
        v = val_entropies[i].mean()
        print(f"  {i:<6}  {k:>8.3f}  {v:>8.3f}  {'YES' if k > v else 'no':>6}")

    global_k_mean = key_entropies.mean()
    global_v_mean = val_entropies.mean()
    print(f"\n  Global mean -- Keys: {global_k_mean:.3f}  Values: {global_v_mean:.3f}")
    kv_asym = global_k_mean > global_v_mean
    print(f"  Keys have {'HIGHER' if kv_asym else 'LOWER'} entropy than Values "
          f"(KVQuant asymmetry {'CONFIRMED' if kv_asym else 'NOT confirmed'})")

    print("\n" + "=" * 65)
    print("Step 2: Building KV Bit-Width Allocation Plans")
    print("=" * 65)
    plan_uniform, plan_layer, plan_head_asym = build_allocation_plans(
        key_entropies, val_entropies, target_bits=4.0
    )

    print("\n" + "=" * 65)
    print("Step 3: FP32 Baseline Perplexity")
    print("=" * 65)
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(dataset["text"])
    input_ids = tokenizer(text, return_tensors="pt").input_ids
    nlls, n_tokens = [], 0
    model.eval()
    with torch.no_grad():
        for i in range(0, min(input_ids.size(1) - SEQ_LEN, SEQ_LEN * 50), SEQ_LEN):
            chunk = input_ids[:, i: i + SEQ_LEN]
            out = model(chunk, labels=chunk)
            nlls.append(out.loss.item() * SEQ_LEN)
            n_tokens += SEQ_LEN
    ppl_fp32 = float(np.exp(sum(nlls) / n_tokens))
    print(f"  FP32 PPL = {ppl_fp32:.2f}")

    print("\n" + "=" * 65)
    print("Step 4: KV-Quantized Perplexity")
    print("=" * 65)
    results = {"FP32 baseline": ppl_fp32}
    for plan in [plan_uniform, plan_layer, plan_head_asym]:
        results[plan["name"]] = measure_ppl_with_kv_quant(model, tokenizer, plan)

    print("\n" + "=" * 65)
    print("Results Summary")
    print("=" * 65)
    print(f"  {'Method':<42}  {'PPL':>10}  {'vs FP32':>10}")
    print("  " + "-" * 66)
    for name, ppl in results.items():
        delta = (ppl - ppl_fp32) / ppl_fp32 * 100
        quant_ppls = [v for k, v in results.items() if k != "FP32 baseline"]
        best_marker = " <-- BEST" if (name != "FP32 baseline" and ppl == min(quant_ppls)) else ""
        print(f"  {name:<42}  {ppl:>10.2f}  {delta:>+9.1f}%{best_marker}")

    print("\n" + "=" * 65)
    print("Analysis")
    print("=" * 65)
    u = results["Uniform 4-bit"]
    l = results["Entropy per-layer"]
    h = results["Entropy per-head (K!=V)"]
    print(f"  Entropy per-layer vs uniform:     {'BETTER' if l < u else 'WORSE'} by {abs(u-l)/u*100:.1f}%")
    print(f"  Entropy per-head vs uniform:      {'BETTER' if h < u else 'WORSE'} by {abs(u-h)/u*100:.1f}%")
    print(f"  Per-head vs per-layer resolution: {'per-head wins' if h < l else 'per-layer wins'} by {abs(l-h)/max(l,h)*100:.1f}%")
    print(f"\n  KV Asymmetry (K entropy {'>' if kv_asym else '<='} V entropy): "
          f"{'SUPPORTS' if kv_asym else 'CONTRADICTS'} KVQuant's finding")

    print("\n" + "=" * 65)
    print("Generating Plots")
    print("=" * 65)
    plot_kv_entropy_landscape(key_entropies, val_entropies,
                              os.path.join(RESULTS_DIR, "kv_entropy_landscape.png"))
    plot_key_vs_value_entropy(key_entropies, val_entropies,
                              os.path.join(RESULTS_DIR, "kv_key_vs_value_entropy.png"))
    plot_bit_allocation(plan_layer, plan_head_asym,
                        os.path.join(RESULTS_DIR, "kv_bit_allocation.png"))
    plot_ppl_comparison(results, os.path.join(RESULTS_DIR, "kv_ppl_comparison.png"))

    print("\n" + "=" * 65)
    print("Experiment complete!")
    print("=" * 65)
