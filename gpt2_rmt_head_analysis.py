"""
GPT-2 RMT / GUE Attention Head Analysis
========================================

Direction 6: Use Random Matrix Theory to classify attention heads by how
much their weight spectra deviate from the GUE (Gaussian Unitary Ensemble)
universal distribution.

The Montgomery Connection
--------------------------
Riemann zeta zeros have GUE pair-correlation statistics (Montgomery 1973).
Trained NN weight matrices evolve under Dyson Brownian Motion (NeurIPS 2024),
whose stationary distribution is GUE. Therefore GUE is the null hypothesis
for a weight matrix that has learned nothing. Deviation from GUE = signal.

Two measurements
-----------------
1. STATIC (weight-based): Extract per-head Q/K/V weight blocks from c_attn.
   Fit bulk singular value spectrum to Marchenko-Pastur. Count outliers above
   the MP spectral edge -- these are the "signal" singular values.

2. RUNTIME (attention-based): Hook into attention forward passes, capture the
   QK^T/sqrt(d) score matrix per head. Unfold the eigenvalue spectrum and
   compute KS distance to the Wigner surmise (GUE nearest-neighbour spacing).

Research Questions
-------------------
Q1: Do early layers conform more to GUE (random) than late layers?
Q2: Which heads deviate most -- are these the "important" heads?
Q3: Does GUE deviation correlate with RG flow growth rates (Direction 2)?
Q4: Can GUE deviation replace entropy as a KV bit-allocation signal?
Q5: Does RMT-guided KV allocation beat uniform 4-bit on PPL?
"""

import os
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats as scipy_stats
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from datasets import load_dataset

SEED = 42
N_LAYERS = 12
N_HEADS = 12
HEAD_DIM = 64
D_MODEL = 768
CALIBRATION_SAMPLES = 64
SEQ_LEN = 512
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


def set_seed(seed):
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


def load_calibration_data(tokenizer, n_samples=CALIBRATION_SAMPLES):
    print(f"Loading WikiText-2 ({n_samples} samples, seq_len={SEQ_LEN})...")
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
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
# Part 1: Static (weight-based) -- Marchenko-Pastur analysis
# ---------------------------------------------------------------------------

def marchenko_pastur_edge(gamma, sigma=1.0):
    """
    Upper and lower spectral edges of the Marchenko-Pastur distribution.
    gamma = n_cols / n_rows (aspect ratio of the weight matrix).
    Returns (lambda_minus, lambda_plus).
    """
    lam_plus = sigma**2 * (1 + gamma**0.5)**2
    lam_minus = sigma**2 * (1 - gamma**0.5)**2
    return lam_minus, lam_plus


def marchenko_pastur_pdf(x, gamma, sigma=1.0):
    """MP density at points x."""
    lam_plus = sigma**2 * (1 + gamma**0.5)**2
    lam_minus = sigma**2 * (1 - gamma**0.5)**2
    pdf = np.zeros_like(x, dtype=float)
    mask = (x >= lam_minus) & (x <= lam_plus)
    xm = x[mask]
    pdf[mask] = (np.sqrt((lam_plus - xm) * (xm - lam_minus))
                 / (2 * np.pi * gamma * sigma**2 * xm))
    return pdf


def analyze_weight_spectrum(model):
    """
    For each (layer, head), extract the Q, K, V weight blocks from c_attn,
    compute singular values, and count outliers above the Marchenko-Pastur edge.

    c_attn weight shape: (D_MODEL, 3*D_MODEL) = (768, 2304) for Conv1D
    After transpose: (2304, 768)
    Q block: rows 0..768,         per head: cols h*64..(h+1)*64  -> (768, 64)
    K block: rows 768..1536,      per head: cols h*64..(h+1)*64  -> (768, 64)
    V block: rows 1536..2304,     per head: cols h*64..(h+1)*64  -> (768, 64)

    Each block is (768, 64). Aspect ratio gamma = 64/768.
    Singular values squared are eigenvalues of W^T W (64x64).
    """
    # MP parameters for a (768, 64) block
    n_rows, n_cols = D_MODEL, HEAD_DIM
    gamma = n_cols / n_rows          # 64/768 ~= 0.083
    # Estimate sigma^2 from empirical variance of a random layer
    # We will estimate it per-block as the mean squared singular value / n_cols

    outlier_counts = {
        "Q": np.zeros((N_LAYERS, N_HEADS), dtype=int),
        "K": np.zeros((N_LAYERS, N_HEADS), dtype=int),
        "V": np.zeros((N_LAYERS, N_HEADS), dtype=int),
    }
    mp_scores = {
        "Q": np.zeros((N_LAYERS, N_HEADS)),
        "K": np.zeros((N_LAYERS, N_HEADS)),
        "V": np.zeros((N_LAYERS, N_HEADS)),
    }

    print(f"\nStatic weight analysis (MP edge, gamma={gamma:.4f})...")
    for layer_idx, block in enumerate(model.transformer.h):
        # c_attn is Conv1D: weight shape (in=768, out=2304)
        W_full = block.attn.c_attn.weight.detach().float().numpy()
        # W_full: (768, 2304) -- columns are [Q_all | K_all | V_all]
        # Each QKV chunk: 768 columns, split into 12 heads of 64 each

        for qkv_idx, qkv_name in enumerate(["Q", "K", "V"]):
            offset = qkv_idx * D_MODEL
            for h in range(N_HEADS):
                col_start = offset + h * HEAD_DIM
                col_end = col_start + HEAD_DIM
                block_W = W_full[:, col_start:col_end]    # (768, 64)

                sv = np.linalg.svd(block_W, compute_uv=False)  # (64,) descending
                eigenvalues = sv**2    # eigenvalues of W^T W

                # Estimate sigma^2 as mean eigenvalue / (1 + gamma)
                sigma2 = eigenvalues.mean() / (1 + gamma)
                sigma = np.sqrt(max(sigma2, 1e-10))

                _, lam_plus = marchenko_pastur_edge(gamma, sigma)
                n_outliers = int((eigenvalues > lam_plus * 1.05).sum())  # 5% buffer

                outlier_counts[qkv_name][layer_idx, h] = n_outliers
                # Score: fraction of variance explained by outliers
                total_var = eigenvalues.sum()
                outlier_var = eigenvalues[eigenvalues > lam_plus * 1.05].sum()
                mp_scores[qkv_name][layer_idx, h] = outlier_var / max(total_var, 1e-10)

    # Summary
    for name in ["Q", "K", "V"]:
        mean_out = outlier_counts[name].mean()
        print(f"  {name}: mean outliers per head = {mean_out:.2f}, "
              f"max = {outlier_counts[name].max()}, "
              f"mean signal fraction = {mp_scores[name].mean():.4f}")

    return outlier_counts, mp_scores


# ---------------------------------------------------------------------------
# Part 2: Runtime (attention-based) -- GUE spacing analysis
# ---------------------------------------------------------------------------

def wigner_surmise_goe(s):
    """GOE Wigner surmise: P(s) = (pi/2) s exp(-pi s^2 / 4). Real symmetric."""
    return (np.pi / 2) * s * np.exp(-np.pi * s**2 / 4)


def wigner_surmise_gue(s):
    """GUE Wigner surmise: P(s) = (32/pi^2) s^2 exp(-4 s^2 / pi). Complex Hermitian."""
    return (32 / np.pi**2) * s**2 * np.exp(-4 * s**2 / np.pi)


def poisson_surmise(s):
    """Poisson (no level repulsion): P(s) = exp(-s)."""
    return np.exp(-s)


def unfold_spectrum(eigenvalues):
    """
    Unfold eigenvalue spectrum to unit mean spacing using cumulative smoothing.
    Returns sorted unfolded eigenvalues.
    """
    ev = np.sort(eigenvalues)
    n = len(ev)
    if n < 4:
        return ev
    # Smooth cumulative count using a polynomial fit
    indices = np.arange(1, n + 1, dtype=float)
    # Fit degree-5 polynomial to (ev, index) mapping
    try:
        coeffs = np.polyfit(ev, indices, deg=min(5, n - 1))
        unfolded = np.polyval(coeffs, ev)
    except Exception:
        # Fallback: linear unfolding
        unfolded = (ev - ev.min()) / (ev.max() - ev.min() + 1e-10) * n
    return unfolded


def spacings_ks_distance(eigenvalues, reference_fn, n_bins=30):
    """
    Compute nearest-neighbour spacing distribution and KS distance
    to a reference distribution (GOE or GUE Wigner surmise).

    Returns (ks_distance, spacings) tuple.
    """
    if len(eigenvalues) < 8:
        return 1.0, np.array([])

    unfolded = unfold_spectrum(eigenvalues)
    spacings = np.diff(np.sort(unfolded))
    spacings = spacings[spacings > 0]
    if len(spacings) < 4:
        return 1.0, spacings

    # Normalise to mean spacing = 1
    spacings = spacings / spacings.mean()

    # KS test against theoretical distribution via CDF comparison
    # Build empirical CDF
    s_sorted = np.sort(spacings)
    empirical_cdf = np.arange(1, len(s_sorted) + 1) / len(s_sorted)

    # Theoretical CDF by numerical integration
    s_grid = np.linspace(0, s_sorted[-1] + 0.1, 500)
    pdf_vals = reference_fn(s_grid)
    cdf_vals = np.cumsum(pdf_vals) * (s_grid[1] - s_grid[0])
    cdf_vals = cdf_vals / max(cdf_vals[-1], 1e-10)

    # Interpolate theoretical CDF at empirical points
    theo_at_empirical = np.interp(s_sorted, s_grid, cdf_vals)
    ks_dist = float(np.max(np.abs(empirical_cdf - theo_at_empirical)))
    return ks_dist, spacings


def collect_attention_spectra(model, samples):
    """
    Hook into attention forward passes to capture QK^T / sqrt(d) per head.
    Returns:
        goe_ks: np.ndarray (N_LAYERS, N_HEADS) -- KS distance to GOE
        gue_ks: np.ndarray (N_LAYERS, N_HEADS) -- KS distance to GUE
        poisson_ks: np.ndarray (N_LAYERS, N_HEADS) -- KS distance to Poisson
    """
    goe_ks_sum = np.zeros((N_LAYERS, N_HEADS))
    gue_ks_sum = np.zeros((N_LAYERS, N_HEADS))
    poisson_ks_sum = np.zeros((N_LAYERS, N_HEADS))
    count = np.zeros((N_LAYERS, N_HEADS), dtype=int)

    original_forwards = {}

    def make_probe(layer_idx):
        def patched_forward(self, hidden_states, **kwargs):
            # QKV split
            q, k, v = self.c_attn(hidden_states).split(self.split_size, dim=2)
            shape_kv = (*k.shape[:-1], -1, HEAD_DIM)
            k = k.view(shape_kv).transpose(1, 2)    # (B, H, T, D)
            q = q.view((*q.shape[:-1], -1, HEAD_DIM)).transpose(1, 2)

            # Attention score matrix per head: (B, H, T, T)
            scale = HEAD_DIM ** -0.5
            scores = torch.matmul(q, k.transpose(-1, -2)) * scale  # (B, H, T, T)

            B, H, T, _ = scores.shape
            for h in range(H):
                # Take mean over batch, symmetrise for eigenvalue analysis
                score_mat = scores[0, h].detach().float().cpu().numpy()  # (T, T)
                sym = (score_mat + score_mat.T) / 2.0
                try:
                    eigenvalues = np.linalg.eigvalsh(sym)  # real symmetric
                    goe_d, _ = spacings_ks_distance(eigenvalues, wigner_surmise_goe)
                    gue_d, _ = spacings_ks_distance(eigenvalues, wigner_surmise_gue)
                    poi_d, _ = spacings_ks_distance(eigenvalues, poisson_surmise)
                    goe_ks_sum[layer_idx, h] += goe_d
                    gue_ks_sum[layer_idx, h] += gue_d
                    poisson_ks_sum[layer_idx, h] += poi_d
                    count[layer_idx, h] += 1
                except Exception:
                    pass

            return original_forwards[layer_idx](self, hidden_states, **kwargs)
        return patched_forward

    for layer_idx, block in enumerate(model.transformer.h):
        original_forwards[layer_idx] = type(block.attn).forward
        type(block.attn).forward = make_probe(layer_idx)

    print(f"\nCollecting attention spectra over {len(samples)} samples...")
    with torch.no_grad():
        for i, sample in enumerate(samples):
            model(sample)
            if (i + 1) % 16 == 0:
                print(f"  {i + 1}/{len(samples)} done")

    for layer_idx, block in enumerate(model.transformer.h):
        type(block.attn).forward = original_forwards[layer_idx]

    safe_count = np.maximum(count, 1)
    goe_ks = goe_ks_sum / safe_count
    gue_ks = gue_ks_sum / safe_count
    poisson_ks = poisson_ks_sum / safe_count

    # Which ensemble fits best (lowest KS = closest match)?
    goe_better = (goe_ks < gue_ks).sum()
    gue_better = (gue_ks < goe_ks).sum()
    print(f"  GOE fits better for {goe_better}/144 heads")
    print(f"  GUE fits better for {gue_better}/144 heads")
    print(f"  Mean GOE KS distance: {goe_ks.mean():.4f}")
    print(f"  Mean GUE KS distance: {gue_ks.mean():.4f}")
    print(f"  Mean Poisson KS distance: {poisson_ks.mean():.4f}")

    return goe_ks, gue_ks, poisson_ks


# ---------------------------------------------------------------------------
# Part 3: Combined signal score + PPL with RMT-guided KV allocation
# ---------------------------------------------------------------------------

def build_rmt_signal_score(mp_scores, goe_ks, gue_ks):
    """
    Combine static (MP outlier fraction) and runtime (GUE KS deviation)
    into a single per-(layer, head) signal score.

    High score = head carries learned structure = deserves more bits.

    We use the mean MP signal across Q/K/V plus the inverse of the best-fit
    KS distance (lower KS = closer to GUE = more random = LESS signal).
    """
    mp_mean = (mp_scores["Q"] + mp_scores["K"] + mp_scores["V"]) / 3.0

    # Best-fit KS: minimum of GOE and GUE per head
    best_ks = np.minimum(goe_ks, gue_ks)

    # Signal from KS: high KS distance to the universal ensemble = more signal
    # Normalise both to [0,1]
    def norm01(x):
        r = x.max() - x.min()
        return (x - x.min()) / r if r > 1e-10 else np.ones_like(x) * 0.5

    score = 0.5 * norm01(mp_mean) + 0.5 * norm01(best_ks)
    return score, mp_mean, best_ks


def _absmax_quantize(t, bits):
    if bits >= 16:
        return t
    n_levels = 2 ** (bits - 1) - 1
    scale = t.abs().max()
    if scale == 0:
        return t
    return torch.round(t / scale * n_levels).clamp(-n_levels, n_levels) * scale / n_levels


def measure_ppl_with_kv_plan(model, tokenizer, head_key_bits, head_val_bits, plan_name):
    """Measure PPL with a given per-head KV bit allocation."""
    original_forwards = {}

    def make_quant_forward(layer_idx):
        def patched_forward(self, hidden_states, **kwargs):
            q, k, v = self.c_attn(hidden_states).split(self.split_size, dim=2)
            shape_kv = (*k.shape[:-1], -1, HEAD_DIM)
            k = k.view(shape_kv).transpose(1, 2)
            v = v.view(shape_kv).transpose(1, 2)
            q = q.view((*q.shape[:-1], -1, HEAD_DIM)).transpose(1, 2)

            B, H, T, D = k.shape
            k_q = torch.zeros_like(k)
            v_q = torch.zeros_like(v)
            for h in range(H):
                k_q[:, h] = _absmax_quantize(k[:, h], int(head_key_bits[layer_idx, h]))
                v_q[:, h] = _absmax_quantize(v[:, h], int(head_val_bits[layer_idx, h]))

            scale = HEAD_DIM ** -0.5
            attn_w = torch.matmul(q, k_q.transpose(-1, -2)) * scale
            causal = torch.tril(torch.ones(T, T, dtype=torch.bool, device=attn_w.device))
            attn_w = attn_w.masked_fill(~causal, float("-inf"))
            attn_w = torch.softmax(attn_w, dim=-1)
            attn_w = torch.nan_to_num(attn_w, nan=0.0)
            out = torch.matmul(attn_w, v_q)
            out = out.transpose(1, 2).contiguous().view(B, T, H * D)
            out = self.c_proj(out)
            out = self.resid_dropout(out)
            return out, None

        return patched_forward

    for layer_idx, block in enumerate(model.transformer.h):
        original_forwards[layer_idx] = type(block.attn).forward
        type(block.attn).forward = make_quant_forward(layer_idx)

    print(f"  PPL [{plan_name}]...")
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

def plot_mp_spectrum_examples(model, save_path):
    """
    Show singular value histogram vs MP fit for 4 example heads
    (early/late layer, low/high signal).
    """
    gamma = HEAD_DIM / D_MODEL
    examples = [(0, 0), (0, 6), (11, 0), (11, 6)]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()

    for ax, (li, hi) in zip(axes, examples):
        W_full = model.transformer.h[li].attn.c_attn.weight.detach().float().numpy()
        # Q block for this head
        col_start = hi * HEAD_DIM
        col_end = col_start + HEAD_DIM
        block_W = W_full[:, col_start:col_end]
        sv = np.linalg.svd(block_W, compute_uv=False)
        eigenvalues = sv**2

        sigma2 = eigenvalues.mean() / (1 + gamma)
        sigma = np.sqrt(max(sigma2, 1e-10))
        lam_minus, lam_plus = marchenko_pastur_edge(gamma, sigma)

        # Histogram
        ax.hist(eigenvalues, bins=20, density=True, alpha=0.6,
                color="steelblue", label="Observed")
        # MP curve
        x = np.linspace(lam_minus * 0.5, lam_plus * 1.5, 300)
        mp = marchenko_pastur_pdf(x, gamma, sigma)
        ax.plot(x, mp, "r-", lw=2, label="Marchenko-Pastur")
        ax.axvline(lam_plus, color="red", linestyle="--", alpha=0.5, label="MP edge")

        n_out = int((eigenvalues > lam_plus * 1.05).sum())
        ax.set_title(f"Layer {li}, Head {hi} (Q block) — {n_out} outliers")
        ax.set_xlabel("Eigenvalue (sv²)")
        ax.set_ylabel("Density")
        ax.legend(fontsize=7)

    plt.suptitle("GPT-2: Weight Spectrum vs Marchenko-Pastur (Q blocks)", fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Saved: {save_path}")


def plot_heatmaps(data_dict, titles, suptitle, save_path, cmap="RdYlGn"):
    n = len(data_dict)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5))
    if n == 1:
        axes = [axes]
    for ax, (key, data), title in zip(axes, data_dict.items(), titles):
        im = ax.imshow(data, aspect="auto", cmap=cmap)
        ax.set_xlabel("Head")
        ax.set_ylabel("Layer")
        ax.set_title(title)
        ax.set_xticks(range(N_HEADS))
        ax.set_yticks(range(N_LAYERS))
        plt.colorbar(im, ax=ax)
    plt.suptitle(suptitle, fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Saved: {save_path}")


def plot_rmt_vs_rg(signal_score, save_path):
    """
    Cross-validate: does RMT signal score correlate with RG flow growth rates?
    Re-computes RG growth rates inline for c_attn (mean across heads).
    """
    from transformers import GPT2LMHeadModel as _GPT2
    model_rg = _GPT2.from_pretrained("openai-community/gpt2")
    model_rg.eval()

    # RG growth rates for c_attn (per layer)
    rg_rates = []
    for li in range(N_LAYERS):
        W = model_rg.transformer.h[li].attn.c_attn.weight.detach().float().numpy()
        sv = np.linalg.svd(W, compute_uv=False)
        rg_rates.append(sv.mean())
    rg_rates = np.array(rg_rates)
    # Normalise
    rg_norm = (rg_rates - rg_rates.min()) / (rg_rates.max() - rg_rates.min() + 1e-10)

    # RMT signal score: mean across heads per layer
    rmt_per_layer = signal_score.mean(axis=1)
    rmt_norm = (rmt_per_layer - rmt_per_layer.min()) / (rmt_per_layer.max() - rmt_per_layer.min() + 1e-10)

    r, p = scipy_stats.pearsonr(rg_norm, rmt_norm)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(rg_norm, rmt_norm, c=np.arange(N_LAYERS), cmap="viridis", s=80, zorder=3)
    for i, (x, y) in enumerate(zip(rg_norm, rmt_norm)):
        ax.annotate(str(i), (x, y), textcoords="offset points", xytext=(5, 5), fontsize=8)
    ax.set_xlabel("RG growth rate (normalised, c_attn mean SV)")
    ax.set_ylabel("RMT signal score (normalised, mean over heads)")
    ax.set_title(f"RMT signal vs RG growth rate per layer\nPearson r={r:.3f} (p={p:.3f})")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Saved: {save_path}")
    return r, p


def plot_spacing_distribution(goe_ks, gue_ks, poisson_ks, save_path):
    """Show which ensemble best fits the attention score matrices."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: theoretical curves
    ax = axes[0]
    s = np.linspace(0, 4, 300)
    ax.plot(s, wigner_surmise_goe(s), "b-", lw=2, label="GOE (Wigner)")
    ax.plot(s, wigner_surmise_gue(s), "r-", lw=2, label="GUE (Wigner)")
    ax.plot(s, poisson_surmise(s), "g--", lw=2, label="Poisson")
    ax.set_xlabel("Spacing s")
    ax.set_ylabel("P(s)")
    ax.set_title("Theoretical Spacing Distributions")
    ax.legend()
    ax.grid(alpha=0.3)

    # Right: mean KS distance per layer
    ax2 = axes[1]
    layers = np.arange(N_LAYERS)
    ax2.plot(layers, goe_ks.mean(axis=1), "bo-", label=f"GOE (mean KS={goe_ks.mean():.3f})")
    ax2.plot(layers, gue_ks.mean(axis=1), "rs-", label=f"GUE (mean KS={gue_ks.mean():.3f})")
    ax2.plot(layers, poisson_ks.mean(axis=1), "g^-", label=f"Poisson (mean KS={poisson_ks.mean():.3f})")
    ax2.set_xlabel("Layer")
    ax2.set_ylabel("Mean KS distance (lower = better fit)")
    ax2.set_title("Which Ensemble Best Fits Attention Spectra?\n(lower KS = closer to that distribution)")
    ax2.legend()
    ax2.grid(alpha=0.3)

    plt.suptitle("GPT-2 Attention Score Eigenvalue Spacing Statistics", fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Saved: {save_path}")


def plot_ppl_comparison(results, save_path):
    methods = list(results.keys())
    ppls = [results[m] for m in methods]
    colors = ["#95a5a6", "#3498db", "#e74c3c", "#9b59b6", "#2ecc71"][:len(methods)]
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(methods, ppls, color=colors, alpha=0.85, edgecolor="white", linewidth=1.2)
    fp32 = results.get("FP32 baseline")
    if fp32:
        ax.axhline(y=fp32, color="green", linestyle="--", alpha=0.5, label=f"FP32 ({fp32:.1f})")
    for bar, ppl in zip(bars, ppls):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                f"{ppl:.1f}", ha="center", fontsize=10, fontweight="bold")
    ax.set_ylabel("Perplexity (WikiText-2 test)")
    ax.set_title("GPT-2 KV Quantization: RMT-Guided vs Uniform\n(4-bit avg)")
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
    cal_samples = load_calibration_data(tokenizer)

    # -- 1. Static analysis: Marchenko-Pastur ----------------------------------
    print("\n" + "=" * 65)
    print("Step 1: Static Weight Spectrum -- Marchenko-Pastur Analysis")
    print("=" * 65)
    outlier_counts, mp_scores = analyze_weight_spectrum(model)

    # -- 2. Runtime analysis: Attention score GUE/GOE spacings ----------------
    print("\n" + "=" * 65)
    print("Step 2: Runtime Attention Spectra -- GUE/GOE Spacing Analysis")
    print("=" * 65)
    goe_ks, gue_ks, poisson_ks = collect_attention_spectra(model, cal_samples)

    # -- 3. Combined signal score ----------------------------------------------
    print("\n" + "=" * 65)
    print("Step 3: Combined RMT Signal Score")
    print("=" * 65)
    signal_score, mp_mean, best_ks = build_rmt_signal_score(mp_scores, goe_ks, gue_ks)

    print(f"  Signal score range: [{signal_score.min():.4f}, {signal_score.max():.4f}]")
    print(f"  Top 5 highest-signal (layer, head):")
    flat_idx = np.argsort(signal_score.flatten())[::-1]
    for rank, idx in enumerate(flat_idx[:5]):
        li, hi = np.unravel_index(idx, signal_score.shape)
        print(f"    #{rank+1}: layer={li}, head={hi}, score={signal_score[li, hi]:.4f}")

    print(f"\n  Top 5 lowest-signal (most random, layer, head):")
    for rank, idx in enumerate(flat_idx[-5:]):
        li, hi = np.unravel_index(idx, signal_score.shape)
        print(f"    #{rank+1}: layer={li}, head={hi}, score={signal_score[li, hi]:.4f}")

    # -- 4. Build KV bit allocation from signal score -------------------------
    print("\n" + "=" * 65)
    print("Step 4: Build KV Bit Allocation Plans")
    print("=" * 65)

    def signal_to_bits(score, min_bits=2, max_bits=8, target=4.0):
        s_min, s_max = score.min(), score.max()
        if s_max == s_min:
            return np.full_like(score, 4, dtype=int)
        norm = (score - s_min) / (s_max - s_min)
        raw = min_bits + norm * (max_bits - min_bits)
        current = raw.mean()
        if current > 0:
            raw = np.clip(raw * (target / current), min_bits, max_bits)
        return np.round(raw).astype(int)

    # Uniform baseline
    uniform_bits = np.full((N_LAYERS, N_HEADS), 4, dtype=int)

    # RMT-guided: same bits for K and V per head (lessons from KV experiment)
    rmt_bits = signal_to_bits(signal_score, target=4.0)

    # Inverted: give random heads MORE bits (control experiment -- should fail)
    inv_bits = signal_to_bits(1.0 - signal_score, target=4.0)

    print(f"  Uniform:   mean={uniform_bits.mean():.2f}")
    print(f"  RMT-guided: mean={rmt_bits.mean():.2f}, "
          f"min={rmt_bits.min()}, max={rmt_bits.max()}")
    print(f"  Inverted:   mean={inv_bits.mean():.2f}")

    # -- 5. FP32 baseline PPL -------------------------------------------------
    print("\n" + "=" * 65)
    print("Step 5: FP32 Baseline Perplexity")
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

    # -- 6. PPL for each plan -------------------------------------------------
    print("\n" + "=" * 65)
    print("Step 6: KV-Quantized Perplexity")
    print("=" * 65)
    results = {"FP32 baseline": ppl_fp32}
    results["Uniform 4-bit"] = measure_ppl_with_kv_plan(
        model, tokenizer, uniform_bits, uniform_bits, "Uniform 4-bit")
    results["RMT-guided (K=V)"] = measure_ppl_with_kv_plan(
        model, tokenizer, rmt_bits, rmt_bits, "RMT-guided (K=V)")
    results["Inverted RMT (control)"] = measure_ppl_with_kv_plan(
        model, tokenizer, inv_bits, inv_bits, "Inverted RMT (control)")

    # -- 7. Results summary ---------------------------------------------------
    print("\n" + "=" * 65)
    print("Results Summary")
    print("=" * 65)
    print(f"  {'Method':<35}  {'PPL':>8}  {'vs FP32':>10}")
    print("  " + "-" * 57)
    quant_ppls = [v for k, v in results.items() if k != "FP32 baseline"]
    for name, ppl in results.items():
        delta = (ppl - ppl_fp32) / ppl_fp32 * 100
        best = " <-- BEST" if (name != "FP32 baseline" and ppl == min(quant_ppls)) else ""
        print(f"  {name:<35}  {ppl:>8.2f}  {delta:>+9.1f}%{best}")

    print("\n" + "=" * 65)
    print("Analysis")
    print("=" * 65)
    u = results["Uniform 4-bit"]
    r = results["RMT-guided (K=V)"]
    inv = results["Inverted RMT (control)"]
    print(f"  RMT-guided vs uniform:  {'BETTER' if r < u else 'WORSE'} by {abs(u-r)/u*100:.1f}%")
    print(f"  Inverted vs uniform:    {'BETTER' if inv < u else 'WORSE'} by {abs(u-inv)/u*100:.1f}%")
    if inv > r:
        print("  Control check PASSED: giving random heads more bits is worse than signal-guided.")
    else:
        print("  Control check FAILED: inverted performed similarly -- signal score may be noisy.")

    # -- 8. Plots -------------------------------------------------------------
    print("\n" + "=" * 65)
    print("Generating Plots")
    print("=" * 65)

    plot_mp_spectrum_examples(model,
        os.path.join(RESULTS_DIR, "rmt_mp_spectrum_examples.png"))

    plot_heatmaps(
        {"MP signal fraction": mp_mean},
        ["MP Signal Fraction (outlier variance / total variance)"],
        "GPT-2: Marchenko-Pastur Signal per (Layer, Head)",
        os.path.join(RESULTS_DIR, "rmt_mp_signal_heatmap.png"),
        cmap="RdYlGn"
    )
    plot_heatmaps(
        {"GUE KS distance": gue_ks, "GOE KS distance": goe_ks},
        ["GUE KS distance (high = more deviation)", "GOE KS distance (high = more deviation)"],
        "GPT-2: KS Distance to GUE/GOE per (Layer, Head)\nHigh = head deviates from universal random matrix statistics",
        os.path.join(RESULTS_DIR, "rmt_gue_deviation_heatmap.png"),
        cmap="YlOrRd"
    )
    plot_heatmaps(
        {"RMT signal score": signal_score, "Bit allocation (RMT)": rmt_bits.astype(float)},
        ["Combined signal score", "Resulting KV bit allocation"],
        "GPT-2: RMT Signal Score and Bit Allocation",
        os.path.join(RESULTS_DIR, "rmt_signal_score_heatmap.png"),
        cmap="RdYlGn"
    )

    plot_spacing_distribution(goe_ks, gue_ks, poisson_ks,
        os.path.join(RESULTS_DIR, "rmt_spacing_distributions.png"))

    rg_r, rg_p = plot_rmt_vs_rg(signal_score,
        os.path.join(RESULTS_DIR, "rmt_signal_vs_rg.png"))
    print(f"  RMT vs RG correlation: r={rg_r:.3f} (p={rg_p:.3f})")

    plot_ppl_comparison(results,
        os.path.join(RESULTS_DIR, "rmt_ppl_comparison.png"))

    print("\n" + "=" * 65)
    print("Experiment complete!")
    print("=" * 65)
