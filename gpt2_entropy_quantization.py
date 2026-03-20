"""
GPT-2 Entropy Quantization Experiment
======================================

Validates entropy-based adaptive bit-width allocation on GPT-2 124M.
Compares entropy-linear vs. uniform-4bit vs. Hessian-linear allocation.
Tests correlation between activation entropy and Hessian sensitivity.

Spec: docs/superpowers/specs/2026-03-20-gpt2-entropy-quantization-design.md
"""

import os
import torch
import numpy as np
import matplotlib

matplotlib.use("Agg")
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from transformers.pytorch_utils import Conv1D  # GPT-2 uses Conv1D, not nn.Linear
from datasets import load_dataset

SEED = 42
CALIBRATION_SAMPLES = 128
SEQ_LEN = 1024
RESULTS_DIR = "results"


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


def collect_entropies(model, samples, n_bins=256, range_estimation_samples=8):
    """
    Collect per-layer activation entropy via streaming histograms.
    Two passes:
    1. Quick pass to find [1st, 99th] percentile per layer
    2. Full pass accumulating histograms within those bounds

    Handles both torch.nn.Linear and transformers.pytorch_utils.Conv1D
    (GPT-2 uses Conv1D for its attention/MLP projections).
    """
    linear_names = []
    linear_modules = []
    for name, module in model.named_modules():
        if isinstance(module, (torch.nn.Linear, Conv1D)):
            linear_names.append(name)
            linear_modules.append(module)

    print(f"Collecting entropy from {len(linear_names)} linear layers...")

    # Pass 1: estimate activation ranges
    print("  Pass 1: estimating activation ranges...")
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
    print(f"  Pass 2: accumulating histograms over {len(samples)} samples...")
    histograms = {n: np.zeros(n_bins, dtype=np.float64) for n in linear_names}

    def hist_hook_fn(module_name):
        def hook(module, input, output):
            x = input[0].detach().float().cpu().numpy().flatten()
            lo, hi = layer_mins[module_name], layer_maxs[module_name]
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
                print(f"    {i + 1}/{len(samples)} samples processed")

    for h in hooks:
        h.remove()

    # Compute entropy from accumulated histograms
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


if __name__ == "__main__":
    set_seed(SEED)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    model, tokenizer = load_model_and_tokenizer()
    cal_samples = load_wikitext2(
        tokenizer, split="train", n_samples=CALIBRATION_SAMPLES
    )

    # Collect entropies
    entropies, linear_names = collect_entropies(model, cal_samples)

    print("\nPer-layer entropies (first 10):")
    for name in linear_names[:10]:
        print(f"  {name:<50s} {entropies[name]:.4f}")
