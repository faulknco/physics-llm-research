"""
GPT-2 Entropy Quantization Experiment
======================================

Validates entropy-based adaptive bit-width allocation on GPT-2 124M.
Compares entropy-linear vs. uniform-4bit vs. Hessian-linear allocation.
Tests correlation between activation entropy and Hessian sensitivity.

Spec: docs/superpowers/specs/2026-03-20-gpt2-entropy-quantization-design.md
"""

import os
import copy
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from datasets import load_dataset
from entropy_quantization import (
    activation_entropy,
    entropy_linear_allocation,
    uniform_allocation,
    absmax_quantize,
    QuantizationPlan,
    enforce_target_mean,
)

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


if __name__ == "__main__":
    set_seed(SEED)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    model, tokenizer = load_model_and_tokenizer()
    cal_samples = load_wikitext2(tokenizer, split="train", n_samples=CALIBRATION_SAMPLES)
    print(f"Calibration: {len(cal_samples)} samples ready")
