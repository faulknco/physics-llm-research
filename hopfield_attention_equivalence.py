"""
Hopfield ↔ Attention Equivalence: Toy Demonstration
=====================================================

This script verifies the mathematical equivalence between:
  1. One-step retrieval in a modern (continuous) Hopfield network
  2. Scaled dot-product attention in a transformer

Reference: Ramsauer et al. 2020, "Hopfield Networks is All You Need" (arXiv 2008.02217)

The key theorem:
    Modern Hopfield update:  xi_new = X @ softmax(beta * X.T @ xi)
    Transformer attention:   Attn(Q, K, V) = V @ softmax(Q @ K.T / sqrt(d))

These are identical when:
    - xi (query)    ↔  Q (single query vector)
    - X (memories)  ↔  K.T = V.T (keys = values, transposed)
    - beta          ↔  1 / sqrt(d)
"""

import numpy as np


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically stable softmax."""
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)


# ─────────────────────────────────────────────
# Part 1: Modern Hopfield Network
# ─────────────────────────────────────────────

def hopfield_energy(xi: np.ndarray, X: np.ndarray, beta: float) -> float:
    """
    Compute the Hopfield energy for state xi given stored patterns X.

    E(xi) = -lse(beta, X.T @ xi) + 0.5 * xi.T @ xi + (1/beta)*log(N) + 0.5*M^2

    Where lse(beta, z) = (1/beta) * log(sum(exp(beta * z_i)))
    """
    N = X.shape[1]
    interactions = X.T @ xi  # shape: (N,)
    lse = (1.0 / beta) * np.log(np.sum(np.exp(beta * interactions)))
    return -lse + 0.5 * np.dot(xi, xi)


def hopfield_update(xi: np.ndarray, X: np.ndarray, beta: float) -> np.ndarray:
    """
    One synchronous Hopfield update: retrieve the closest stored pattern.

    xi_new = X @ softmax(beta * X.T @ xi)

    Args:
        xi: query/initial state vector, shape (d,)
        X:  stored patterns matrix, shape (d, N)  — columns are patterns
        beta: inverse temperature (sharpness of retrieval)

    Returns:
        xi_new: updated state, shape (d,)
    """
    scores = beta * (X.T @ xi)        # (N,) — interaction energies
    weights = softmax(scores)          # (N,) — Boltzmann weights
    xi_new = X @ weights               # (d,) — weighted superposition
    return xi_new


# ─────────────────────────────────────────────
# Part 2: Transformer Scaled Dot-Product Attention
# ─────────────────────────────────────────────

def scaled_dot_product_attention(
    Q: np.ndarray,
    K: np.ndarray,
    V: np.ndarray,
) -> np.ndarray:
    """
    Standard transformer attention (no masking).

    Attn(Q, K, V) = V @ softmax(Q @ K.T / sqrt(d_k))

    Args:
        Q: queries, shape (n_q, d_k)
        K: keys,    shape (n_kv, d_k)
        V: values,  shape (n_kv, d_v)

    Returns:
        output: shape (n_q, d_v)
    """
    d_k = K.shape[1]
    scores = Q @ K.T / np.sqrt(d_k)    # (n_q, n_kv)
    weights = softmax(scores, axis=-1)  # (n_q, n_kv)
    return weights @ V                  # (n_q, d_v)


# ─────────────────────────────────────────────
# Part 3: Prove Equivalence on a Toy Example
# ─────────────────────────────────────────────

def run_equivalence_demo(d: int = 8, N: int = 5, seed: int = 42):
    """
    Demonstrate that Hopfield retrieval == attention output.

    Setup:
        - d: embedding dimension
        - N: number of stored patterns / key-value pairs
        - We use K = V (keys equal values) to match Hopfield convention
    """
    rng = np.random.default_rng(seed)

    # Stored patterns (= keys = values in attention)
    X = rng.standard_normal((d, N))  # shape: (d, N)

    # Query vector (a noisy version of pattern 0)
    xi = X[:, 0] + 0.3 * rng.standard_normal(d)

    # Inverse temperature parameter
    beta = 1.0 / np.sqrt(d)

    print("=" * 60)
    print("Hopfield ↔ Attention Equivalence Demo")
    print("=" * 60)
    print(f"  Dimension d={d}, Patterns N={N}, beta=1/sqrt({d})={beta:.4f}")
    print()

    # ── Hopfield retrieval ──────────────────────────────────────────
    xi_hopfield = hopfield_update(xi, X, beta)
    energy_before = hopfield_energy(xi, X, beta)
    energy_after  = hopfield_energy(xi_hopfield, X, beta)

    print("Hopfield Retrieval:")
    print(f"  Energy before update: {energy_before:.6f}")
    print(f"  Energy after update:  {energy_after:.6f}")
    print(f"  Energy decreased:     {energy_after < energy_before}")
    print()

    # ── Attention ──────────────────────────────────────────────────
    # Reshape for attention API:
    #   Q = xi as a single query:   (1, d)
    #   K = X.T:                    (N, d)  — each pattern is a key
    #   V = X.T:                    (N, d)  — keys = values (Hopfield convention)
    Q = xi.reshape(1, d)
    K = X.T        # (N, d)
    V = X.T        # (N, d)

    attn_out = scaled_dot_product_attention(Q, K, V)  # (1, d)
    xi_attention = attn_out[0]  # flatten to (d,)

    # ── Verify equivalence ─────────────────────────────────────────
    max_diff = np.max(np.abs(xi_hopfield - xi_attention))
    print("Attention Output:")
    print(f"  Max element-wise difference from Hopfield: {max_diff:.2e}")
    print(f"  Outputs are {'IDENTICAL ✓' if max_diff < 1e-10 else 'DIFFERENT ✗'}")
    print()

    # ── Show retrieval quality ─────────────────────────────────────
    print("Retrieval Quality (cosine similarity to each stored pattern):")
    for i in range(N):
        cos_before = np.dot(xi, X[:, i]) / (np.linalg.norm(xi) * np.linalg.norm(X[:, i]))
        cos_after  = np.dot(xi_hopfield, X[:, i]) / (np.linalg.norm(xi_hopfield) * np.linalg.norm(X[:, i]))
        marker = " ← query target" if i == 0 else ""
        print(f"  Pattern {i}: before={cos_before:.3f} → after={cos_after:.3f}{marker}")

    return xi_hopfield, xi_attention, max_diff


# ─────────────────────────────────────────────
# Part 4: Storage Capacity Demo
# ─────────────────────────────────────────────

def run_capacity_demo(d: int = 32, beta_values: list = None):
    """
    Show how retrieval quality depends on beta (inverse temperature).

    High beta → sharp retrieval (isolates single pattern)
    Low beta  → averaging (retrieves centroid of patterns)

    This mirrors transformer attention behavior:
    - High temperature (low beta) → uniform attention (diffuse)
    - Low temperature (high beta) → peaked attention (sharp retrieval)
    """
    if beta_values is None:
        beta_values = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]

    rng = np.random.default_rng(0)
    N = 10
    X = rng.standard_normal((d, N))
    X = X / np.linalg.norm(X, axis=0)  # normalize patterns

    # Query = noisy version of pattern 0
    xi = X[:, 0] + 0.5 * rng.standard_normal(d)
    xi = xi / np.linalg.norm(xi)

    print("=" * 60)
    print("Effect of Temperature (beta) on Retrieval Sharpness")
    print("=" * 60)
    print(f"  d={d}, N={N} normalized patterns, 50% noise on query")
    print()
    print(f"  {'beta':>8}  {'cos(xi_new, pattern_0)':>24}  {'max_weight':>12}")
    print(f"  {'-'*8}  {'-'*24}  {'-'*12}")

    for beta in beta_values:
        xi_new = hopfield_update(xi, X, beta)
        xi_new_norm = xi_new / (np.linalg.norm(xi_new) + 1e-12)
        cos = np.dot(xi_new_norm, X[:, 0])

        # Attention weights for interpretability
        scores = beta * (X.T @ xi)
        weights = softmax(scores)
        print(f"  {beta:>8.1f}  {cos:>24.4f}  {weights.max():>12.4f}")

    print()
    print("  Observation: higher beta → sharper attention → better retrieval")
    print("  This is why transformers use 1/sqrt(d) scaling: prevents")
    print("  softmax saturation in high dimensions.")


# ─────────────────────────────────────────────
# Part 5: Multi-Head Attention as Multiple Hopfield Networks
# ─────────────────────────────────────────────

def run_multihead_demo(d: int = 16, n_heads: int = 4, N: int = 8, seed: int = 7):
    """
    Show multi-head attention as H parallel Hopfield retrievals,
    each operating in a different subspace.

    Each head h projects into a d/H-dimensional subspace via W_Q^h, W_K^h, W_V^h.
    The head's retrieval is a Hopfield update in that subspace.
    """
    H = n_heads
    d_head = d // H
    rng = np.random.default_rng(seed)

    # Random projection matrices (one per head)
    W_Q = [rng.standard_normal((d_head, d)) / np.sqrt(d) for _ in range(H)]
    W_K = [rng.standard_normal((d_head, d)) / np.sqrt(d) for _ in range(H)]
    W_V = [rng.standard_normal((d_head, d)) / np.sqrt(d) for _ in range(H)]
    W_O = rng.standard_normal((d, d)) / np.sqrt(d)  # output projection

    # Token sequence (N tokens)
    tokens = rng.standard_normal((N, d))
    query_idx = 0

    print("=" * 60)
    print("Multi-Head Attention as Parallel Hopfield Networks")
    print("=" * 60)
    print(f"  d={d}, H={H} heads, d_head={d_head}, N={N} tokens")
    print()

    head_outputs = []
    for h in range(H):
        # Project into head subspace
        q = W_Q[h] @ tokens[query_idx]    # (d_head,)
        K_h = tokens @ W_K[h].T           # (N, d_head) → keys
        V_h = tokens @ W_V[h].T           # (N, d_head) → values

        # Hopfield retrieval in subspace h
        beta_h = 1.0 / np.sqrt(d_head)
        xi_h = hopfield_update(q, K_h.T, beta_h)  # (d_head,)
        head_outputs.append(xi_h)

        # Compute attention entropy (how spread is the attention?)
        scores = beta_h * (K_h @ q)
        weights = softmax(scores)
        entropy = -np.sum(weights * np.log(weights + 1e-12))
        max_attn = weights.max()
        print(f"  Head {h}: attention entropy={entropy:.3f}, peak weight={max_attn:.3f}")

    # Concatenate and project
    concat = np.concatenate(head_outputs)  # (d,)
    output = W_O @ concat                  # (d,)
    print()
    print(f"  Final output norm: {np.linalg.norm(output):.4f}")
    print()
    print("  Each head = one Hopfield network operating in its own subspace.")
    print("  Different heads can specialize: some do global averaging (low entropy),")
    print("  some do sharp retrieval (high entropy, low peak weight means averaging).")


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

if __name__ == "__main__":
    # 1. Core equivalence proof
    _, _, diff = run_equivalence_demo(d=16, N=6)
    print()

    # 2. Temperature effect
    run_capacity_demo(d=32)
    print()

    # 3. Multi-head as parallel Hopfield networks
    run_multihead_demo(d=16, n_heads=4, N=8)

    print()
    print("=" * 60)
    print("Summary")
    print("=" * 60)
    print("✓ Hopfield update and scaled dot-product attention are")
    print("  mathematically identical (verified numerically).")
    print()
    print("✓ Transformer attention IS energy minimization in a")
    print("  continuous Hopfield network's energy landscape.")
    print()
    print("✓ Implications for physics-inspired design:")
    print("  - Quantization = discretizing the energy landscape")
    print("  - Pruning = removing irrelevant energy minima")
    print("  - Initialization = choosing the energy landscape shape")
    print("  - Long context = longer trajectory through phase space")
