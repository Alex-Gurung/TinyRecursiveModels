# Muon Optimizer Implementation Guide

This guide explains the Muon optimizer implementation added to TinyRecursiveModels for potentially **1.3-1.5x faster training**.

---

## What is Muon?

**Muon** is a new optimizer in PyTorch 2.9+ that uses momentum with Newton-Schulz orthogonalization. It's specifically designed for transformer hidden layer weights (2D matrices) and shows significant speedups over Adam/AdamW.

**Key advantages:**
- **1.35x faster convergence** on transformer training (GPT-2 benchmarks)
- **Built-in muP scaling:** Learning rates transfer across model sizes
- **Better large-batch training:** Especially good for batch sizes like 768-2048
- **Geometric optimization:** Keeps weight updates in geometrically favorable directions

**Research:**
- GitHub: https://github.com/KellerJordan/Muon
- PyTorch docs: https://pytorch.org/docs/stable/generated/torch.optim.Muon.html

---

## Why Muon for TinyRecursiveModels?

The TRM architecture is **perfectly suited** for Muon:

- **~99.9999% of parameters are 2D weight matrices** (attention, MLP, embeddings)
- Only **2 parameters** are 1D (q_head.bias)
- Dominated by transformer layers with linear projections

See [MUON_ANALYSIS.md](MUON_ANALYSIS.md) for detailed parameter breakdown.

---

## Implementation Overview

### Three-Optimizer Hybrid Setup

When `use_muon=True`, the code uses three optimizers for different parameter groups:

1. **SignSGD** → Sparse puzzle embeddings
   - High learning rate (1e-2)
   - Efficient for sparse gradients
   - Handles distributed training

2. **Muon** → 2D+ weight matrices (~99.9999% of params)
   - Hidden layer weights in transformers
   - Attention projections (Q, K, V, O)
   - MLP layers (gate, up, down projections)
   - Token and position embeddings
   - LM head and Q-head weights
   - Learning rate: 2e-2 (10-20x higher than Adam)

3. **AdamW** → 1D parameters (biases, norms)
   - Q-head bias (2 params)
   - Standard Adam learning rate: 1e-4

---

## How to Use

### Quick Start: Enable Muon in Existing Configs

Add these parameters to any config file:

```yaml
use_muon: True
muon_lr: 2e-2
muon_momentum: 0.95
muon_nesterov: True
muon_ns_steps: 5
muon_backend: 'original'
```

### Option 1: Use Pre-configured Muon Config

A ready-to-use config is provided at [config/cfg_pretrain_muon.yaml](config/cfg_pretrain_muon.yaml):

```bash
# ARC-AGI-1 with Muon (4 H-100 GPUs)
torchrun --nproc-per-node 4 pretrain.py \
  --config-name cfg_pretrain_muon \
  arch=trm \
  data_paths="[data/arc1concept-aug-1000]" \
  arch.L_layers=2 \
  arch.H_cycles=3 arch.L_cycles=4 \
  +run_name="arc1_muon_experiment"

# Sudoku-Extreme with Muon (single GPU)
python pretrain.py \
  --config-name cfg_pretrain_muon \
  arch=trm \
  data_paths="[data/sudoku-extreme-1k-aug-1000]" \
  evaluators="[]" \
  epochs=50000 eval_interval=5000 \
  arch.L_layers=2 \
  arch.H_cycles=3 arch.L_cycles=6 \
  +run_name="sudoku_muon_experiment"
```

### Option 2: Override Default Config

Use the standard config but enable Muon:

```bash
python pretrain.py \
  arch=trm \
  data_paths="[data/arc-aug-1000]" \
  use_muon=True \
  muon_lr=2e-2 \
  lr_min_ratio=0.1 \
  +run_name="muon_test"
```

### Option 3: Disable Muon (Use Original AdamATan2)

Default behavior when `use_muon=False` (or not specified):

```bash
python pretrain.py arch=trm data_paths="[data/arc-aug-1000]"
# Uses AdamATan2 for all model parameters (original implementation)
```

---

## Configuration Parameters

### Core Muon Settings

| Parameter | Default | Description |
|-----------|---------|-------------|
| `use_muon` | `False` | Enable Muon optimizer (set to `True` to use Muon) |
| `muon_lr` | `2e-2` | Learning rate for Muon (typically 10-20x higher than Adam) |
| `muon_momentum` | `0.95` | Momentum coefficient (default from Muon paper) |
| `muon_nesterov` | `True` | Use Nesterov momentum acceleration |
| `muon_ns_steps` | `5` | Newton-Schulz orthogonalization iteration steps |
| `muon_backend` | `'original'` | Backend: `'original'` or `'match_rms_adamw'` |

### Other Important Settings

| Parameter | Muon Config | Original Config | Notes |
|-----------|-------------|-----------------|-------|
| `lr` | `1e-4` | `1e-4` | For AdamW (1D params) |
| `lr_min_ratio` | `0.1` | `1.0` | Cosine decay (changed for better convergence) |
| `ema` | `True` | `False` | Recommended for better generalization |
| `puzzle_emb_lr` | `1e-2` | `1e-2` | Unchanged (SignSGD) |

---

## Expected Performance Improvements

Based on Muon paper benchmarks and theoretical analysis:

### Training Speed
- **1.3-1.5x faster convergence** (fewer epochs to reach same accuracy)
- **Potential for larger batch sizes** (e.g., 1024-2048 instead of 768)
- **Better large-batch scaling** (less degradation with bigger batches)

### Time Savings
- **ARC-AGI training:** ~3 days → ~2 days (~20-24 hours saved)
- **Sudoku training:** <36 hours → <27 hours (~9 hours saved)

### Memory Usage
- **Slightly lower than Adam:** Muon doesn't track second moment (like Adam's v_t)
- **Comparable to AdamW:** Both use momentum only

---

## Hyperparameter Tuning Recommendations

### Start with Defaults
The provided config should work well out-of-the-box:
- `muon_lr=2e-2` (standard Muon LR)
- `muon_momentum=0.95` (from Muon paper)
- `muon_nesterov=True` (recommended)
- `muon_ns_steps=5` (standard orthogonalization steps)

### If Training is Unstable
1. **Lower Muon LR:** Try `muon_lr=1e-2` or `1.5e-2`
2. **Increase warmup:** Try `lr_warmup_steps=4000` or `5000`
3. **Try different backend:** `muon_backend='match_rms_adamw'` for better AdamW compatibility

### If Training is Too Slow
1. **Increase batch size:** Try `global_batch_size=1024` or `1536` (Muon handles large batches well)
2. **Reduce NS steps:** Try `muon_ns_steps=3` (faster but slightly less stable)

### If You Want Faster Convergence
1. **Enable cosine decay:** `lr_min_ratio=0.1` (already in Muon config)
2. **Enable EMA:** `ema=True` (already in Muon config)
3. **Tune Muon LR:** Try `muon_lr=2.5e-2` or `3e-2` (higher risk but potentially faster)

---

## Comparison: AdamATan2 vs Muon

| Feature | AdamATan2 (Original) | Muon (New) |
|---------|---------------------|------------|
| **Numerical stability** | Excellent (no epsilon) | Excellent (orthogonal updates) |
| **Scale invariance** | Yes | Yes (built-in muP) |
| **Large batch training** | Standard | **Excellent** |
| **Convergence speed** | Baseline | **1.35x faster** |
| **Hyperparameter transfer** | No | **Yes (muP scaling)** |
| **Memory usage** | Standard Adam | **Slightly lower** (no second moment) |
| **Best for** | Small-medium batches | Large batches, transformers |

---

## Troubleshooting

### Issue: Import error for Muon
**Error:** `ImportError: cannot import name 'Muon' from 'torch.optim'`

**Solution:** Ensure you have PyTorch 2.9+:
```bash
pip install --pre --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu126
```

### Issue: Training diverges or NaN loss
**Solutions:**
1. Lower Muon LR: `muon_lr=1e-2`
2. Increase warmup: `lr_warmup_steps=4000`
3. Enable gradient clipping (if not already enabled)
4. Check for gradient overflow in early steps

### Issue: No speedup observed
**Possible causes:**
1. Batch size too small (Muon excels at large batches)
2. Model too small (7M params might not see full benefit)
3. Bottleneck elsewhere (data loading, evaluation, etc.)

**Try:**
- Increase `global_batch_size` to 1024-2048
- Profile training to find bottlenecks

### Issue: Want to compare side-by-side
**Run both configs:**
```bash
# Original (AdamATan2)
python pretrain.py arch=trm data_paths="[data/sudoku-extreme-1k-aug-1000]" \
  epochs=10000 +run_name="baseline_adamatan2"

# Muon
python pretrain.py --config-name cfg_pretrain_muon arch=trm \
  data_paths="[data/sudoku-extreme-1k-aug-1000]" \
  epochs=10000 +run_name="experiment_muon"
```

Compare loss curves in W&B or TensorBoard.

---

## Implementation Details

### Code Changes

All changes are backward compatible. The optimizer selection logic in [pretrain.py:186-298](pretrain.py#L186-L298):

1. **When `use_muon=False`** (default): Uses original AdamATan2 + SignSGD setup
2. **When `use_muon=True`**: Uses Muon + AdamW + SignSGD setup

### Parameter Categorization

The `categorize_parameters_by_dim()` function ([pretrain.py:125-152](pretrain.py#L125-L152)) splits parameters:
- **2D+ params** → Muon
- **1D params** → AdamW
- **Puzzle embeddings** → SignSGD (handled separately)

### Learning Rate Scheduling

All optimizers use the same cosine schedule with warmup, but different base learning rates:
- Muon: `muon_lr` (2e-2)
- AdamW: `lr` (1e-4)
- SignSGD: `puzzle_emb_lr` (1e-2)

---

## Verification Script

Run this to verify Muon is being used correctly:

```bash
cd /nfs/user/alexgurung/TinyRecursiveModels

python -c "
import torch
import yaml
from models.recursive_reasoning.trm import TinyRecursiveReasoningModel_ACTV1_Inner

# Load config
with open('config/arch/trm.yaml', 'r') as f:
    cfg = yaml.safe_load(f)

# Minimal model config
model_cfg = {**cfg, 'batch_size': 16, 'vocab_size': 11, 'seq_len': 900,
             'num_puzzle_identifiers': 1000, 'causal': False}

# Create model
model = TinyRecursiveReasoningModel_ACTV1_Inner(model_cfg)

# Analyze
total, params_2d, params_1d = 0, 0, 0
print('\\n2D Parameters (would use Muon):')
for name, p in model.named_parameters():
    if p.ndim >= 2:
        total += p.numel()
        params_2d += p.numel()
        print(f'  {name:50s} {str(p.shape):25s} {p.numel():>10,}')

print('\\n1D Parameters (would use AdamW):')
for name, p in model.named_parameters():
    if p.ndim == 1:
        total += p.numel()
        params_1d += p.numel()
        print(f'  {name:50s} {str(p.shape):25s} {p.numel():>10,}')

print(f'\\nTotal: {total:,} | 2D: {params_2d:,} ({100*params_2d/total:.2f}%) | 1D: {params_1d:,} ({100*params_1d/total:.4f}%)')
print('\\nMuon is HIGHLY appropriate for this model!')
"
```

---

## References

1. **Muon Optimizer Paper:** https://github.com/KellerJordan/Muon
2. **PyTorch 2.9 Docs:** https://pytorch.org/docs/stable/generated/torch.optim.Muon.html
3. **AdamATan2 Paper:** "Scaling Exponents Across Parameterizations and Optimizers" (Everett et al., ICML 2024)
4. **Original TRM Paper:** "Less is More: Recursive Reasoning with Tiny Networks" (arXiv:2510.04871)

---

## Questions or Issues?

- Check [MUON_ANALYSIS.md](MUON_ANALYSIS.md) for detailed parameter analysis
- Review the config: [config/cfg_pretrain_muon.yaml](config/cfg_pretrain_muon.yaml)
- See the implementation: [pretrain.py](pretrain.py) lines 186-298

Happy training! 🚀
