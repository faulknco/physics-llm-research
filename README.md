# Physics & Math-Inspired LLM Architectures

Research program exploring whether structures from physics and mathematics can improve LLM efficiency and generalization capabilities.

## Research Directions

| # | Direction | Key Idea | Status |
|---|---|---|---|
| 1 | **Thermodynamic Quantization** | Per-layer entropy → adaptive bit-width allocation | Prototype complete |
| 2 | **RG-Guided Pruning** | Renormalization group flow identifies irrelevant weights | Design only |
| 3 | **Lattice-Structured Attention** | Physics lattice Green's function as attention kernel | Design only |
| 4 | **Mean-Field Initialization** | Maximize metastable token cluster diversity | Design only |
| 5 | **Tensor Network Compression** | DMRG-style adaptive bond dimension compression | Design only |

## Code

```bash
# Verify Hopfield ↔ Attention equivalence (no dependencies beyond numpy)
python3 hopfield_attention_equivalence.py

# Run entropy-based quantization prototype
python3 entropy_quantization.py
```

## Key Result

Transformer attention is mathematically identical to one-step retrieval in a modern Hopfield network (verified: max diff = `0.00e+00`). This means transformers are already energy-based physical systems — all physics-inspired modifications build from this foundation.

## See CLAUDE.md for agent continuity context.
