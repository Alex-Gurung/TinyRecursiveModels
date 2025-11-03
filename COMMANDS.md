# TinyRecursiveModels: Training & Analysis Commands

Quick reference for all training, evaluation, and analysis commands.

---

## Dataset Preparation

### Sudoku-Extreme
```bash
python dataset/build_sudoku_dataset.py \
  --output-dir data/sudoku-extreme-1k-aug-1000 \
  --subsample-size 1000 \
  --num-aug 1000
```

### ARC-AGI-1
```bash
python -m dataset.build_arc_dataset \
  --input-file-prefix kaggle/combined/arc-agi \
  --output-dir data/arc1concept-aug-1000 \
  --subsets training evaluation concept \
  --test-set-name evaluation
```

### ARC-AGI-2
```bash
python -m dataset.build_arc_dataset \
  --input-file-prefix kaggle/combined/arc-agi \
  --output-dir data/arc2concept-aug-1000 \
  --subsets training2 evaluation2 concept \
  --test-set-name evaluation2
```

### Maze-Hard
```bash
python dataset/build_maze_dataset.py
# Creates data/maze-30x30-hard-1k (1000 examples, 8 augments)
```

---

## Training Commands

> **Note:** There are two approaches:
> 1. **Original style** (from README) - uses default config, override params on command line
> 2. **Muon style** - uses pre-configured `cfg_pretrain_muon.yaml`

### Sudoku-Extreme (Single GPU)

**Original Style (from README):**
```bash
run_name="pretrain_att_sudoku"
python pretrain.py \
  arch=trm \
  data_paths="[data/sudoku-extreme-1k-aug-1000]" \
  evaluators="[]" \
  epochs=50000 eval_interval=5000 \
  lr=1e-4 puzzle_emb_lr=1e-4 weight_decay=1.0 puzzle_emb_weight_decay=1.0 \
  arch.L_layers=2 \
  arch.H_cycles=3 arch.L_cycles=6 \
  +run_name=${run_name} ema=True
```

**Baseline (AdamW only):**
```bash
python pretrain.py \
  --config-name cfg_pretrain_muon \
  arch=trm \
  data_paths="[data/sudoku-extreme-1k-aug-1000]" \
  evaluators="[]" \
  epochs=5000 \
  eval_interval=2500 \
  arch.L_layers=2 \
  arch.H_cycles=3 \
  arch.L_cycles=6 \
  use_muon=False \
  +run_name="sudoku_baseline_adamw"
```

**With Muon (recommended LR):**
```bash
python pretrain.py \
  --config-name cfg_pretrain_muon \
  arch=trm \
  data_paths="[data/sudoku-extreme-1k-aug-1000]" \
  evaluators="[]" \
  epochs=5000 \
  eval_interval=2500 \
  arch.L_layers=2 \
  arch.H_cycles=3 \
  arch.L_cycles=6 \
  use_muon=True \
  +run_name="sudoku_muon_4e3"
```

**With Muon (conservative LR for stability):**
```bash
python pretrain.py \
  --config-name cfg_pretrain_muon \
  arch=trm \
  data_paths="[data/sudoku-extreme-1k-aug-1000]" \
  evaluators="[]" \
  epochs=5000 \
  eval_interval=2500 \
  arch.L_layers=2 \
  arch.H_cycles=3 \
  arch.L_cycles=6 \
  use_muon=True \
  muon_lr=1e-2 \
  lr_warmup_steps=4000 \
  +run_name="sudoku_muon_conservative"
```

**Full Sudoku training (50k epochs):**
```bash
python pretrain.py \
  --config-name cfg_pretrain_muon \
  arch=trm \
  data_paths="[data/sudoku-extreme-1k-aug-1000]" \
  evaluators="[]" \
  epochs=50000 \
  eval_interval=5000 \
  lr=1e-4 \
  puzzle_emb_lr=1e-4 \
  weight_decay=1.0 \
  puzzle_emb_weight_decay=1.0 \
  arch.L_layers=2 \
  arch.H_cycles=3 \
  arch.L_cycles=6 \
  use_muon=True \
  ema=True \
  +run_name="sudoku_full_training"
```

---

### ARC-AGI-1 (4 GPUs)

**Original Style (from README):**
```bash
run_name="pretrain_att_arc1concept_4"
torchrun --nproc-per-node 4 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 --nnodes=1 pretrain.py \
  arch=trm \
  data_paths="[data/arc1concept-aug-1000]" \
  arch.L_layers=2 \
  arch.H_cycles=3 arch.L_cycles=4 \
  +run_name=${run_name} ema=True
```

**With Muon:**
```bash
torchrun --nproc-per-node 4 pretrain.py \
  --config-name cfg_pretrain_muon \
  arch=trm \
  data_paths="[data/arc1concept-aug-1000]" \
  arch.L_layers=2 \
  arch.H_cycles=3 \
  arch.L_cycles=4 \
  use_muon=True \
  ema=True \
  +run_name="arc1_muon"
```

**Baseline (AdamW):**
```bash
torchrun --nproc-per-node 4 pretrain.py \
  --config-name cfg_pretrain_muon \
  arch=trm \
  data_paths="[data/arc1concept-aug-1000]" \
  arch.L_layers=2 \
  arch.H_cycles=3 \
  arch.L_cycles=4 \
  use_muon=False \
  ema=True \
  +run_name="arc1_baseline"
```

---

### ARC-AGI-2 (4 GPUs)

**Original Style (from README):**
```bash
run_name="pretrain_att_arc2concept_4"
torchrun --nproc-per-node 4 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 --nnodes=1 pretrain.py \
  arch=trm \
  data_paths="[data/arc2concept-aug-1000]" \
  arch.L_layers=2 \
  arch.H_cycles=3 arch.L_cycles=4 \
  +run_name=${run_name} ema=True
```

**With Muon:**
```bash
torchrun --nproc-per-node 4 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 --nnodes=1 pretrain.py \
  --config-name cfg_pretrain_muon \
  arch=trm \
  data_paths="[data/arc2concept-aug-1000]" \
  arch.L_layers=2 \
  arch.H_cycles=3 \
  arch.L_cycles=4 \
  use_muon=True \
  ema=True \
  +run_name="arc2_muon"
```

---

### Maze-Hard (4 GPUs)

**Original Style (from README):**
```bash
run_name="pretrain_att_maze30x30"
torchrun --nproc-per-node 4 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 --nnodes=1 pretrain.py \
  arch=trm \
  data_paths="[data/maze-30x30-hard-1k]" \
  evaluators="[]" \
  epochs=50000 eval_interval=5000 \
  lr=1e-4 puzzle_emb_lr=1e-4 weight_decay=1.0 puzzle_emb_weight_decay=1.0 \
  arch.L_layers=2 \
  arch.H_cycles=3 arch.L_cycles=4 \
  +run_name=${run_name} ema=True
```

**With Muon:**
```bash
torchrun --nproc-per-node 4 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 --nnodes=1 pretrain.py \
  --config-name cfg_pretrain_muon \
  arch=trm \
  data_paths="[data/maze-30x30-hard-1k]" \
  evaluators="[]" \
  epochs=50000 \
  eval_interval=5000 \
  lr=1e-4 \
  puzzle_emb_lr=1e-4 \
  weight_decay=1.0 \
  puzzle_emb_weight_decay=1.0 \
  arch.L_layers=2 \
  arch.H_cycles=3 \
  arch.L_cycles=4 \
  use_muon=True \
  ema=True \
  +run_name="maze_muon"
```

---

## Convergence Analysis

### Basic Analysis (with checkpoint)

```bash
python analyze_convergence.py \
  --checkpoint checkpoints/Sudoku-extreme-1k-aug-1000-ACT-torch/muon_quick_test/step_6510
```

### Analysis with More Cycles

```bash
python analyze_convergence.py \
  --checkpoint checkpoints/your_run_name/step_XXXXX \
  --extra-cycles 10
```

### Analysis Without Checkpoint (Random Initialization)

```bash
python analyze_convergence.py
```

### Analysis with Larger Batch

```bash
python analyze_convergence.py \
  --checkpoint checkpoints/your_run_name/step_XXXXX \
  --batch-size 32 \
  --extra-cycles 10
```

### Analysis with Different Config

```bash
python analyze_convergence.py \
  --config config/arch/hrm.yaml \
  --checkpoint checkpoints/your_run_name/step_XXXXX
```

---

## Side-by-Side Comparison (Muon vs AdamW)

Run both and compare in W&B:

```bash
# Terminal 1: Baseline
python pretrain.py \
  --config-name cfg_pretrain_muon \
  arch=trm \
  data_paths="[data/sudoku-extreme-1k-aug-1000]" \
  evaluators="[]" \
  epochs=10000 \
  eval_interval=2500 \
  arch.L_layers=2 \
  arch.H_cycles=3 \
  arch.L_cycles=6 \
  use_muon=False \
  +run_name="comparison_adamw"

# Terminal 2: Muon
python pretrain.py \
  --config-name cfg_pretrain_muon \
  arch=trm \
  data_paths="[data/sudoku-extreme-1k-aug-1000]" \
  evaluators="[]" \
  epochs=10000 \
  eval_interval=2500 \
  arch.L_layers=2 \
  arch.H_cycles=3 \
  arch.L_cycles=6 \
  use_muon=True \
  muon_lr=4e-3 \
  +run_name="comparison_muon"
```

---

## Hyperparameter Tuning Examples

### Lower Muon LR for Stability

```bash
python pretrain.py \
  --config-name cfg_pretrain_muon \
  [... other args ...] \
  muon_lr=1e-2 \
  lr_warmup_steps=4000
```

### Increase Batch Size (Muon benefits)

```bash
python pretrain.py \
  --config-name cfg_pretrain_muon \
  [... other args ...] \
  global_batch_size=1024 \
  muon_lr=2e-2
```

### Try Different Muon Backend

```bash
python pretrain.py \
  --config-name cfg_pretrain_muon \
  [... other args ...] \
  muon_adjust_lr_fn='match_rms_adamw'
```

### Disable EMA

```bash
python pretrain.py \
  --config-name cfg_pretrain_muon \
  [... other args ...] \
  ema=False
```

### Change Learning Rate Schedule

```bash
python pretrain.py \
  --config-name cfg_pretrain_muon \
  [... other args ...] \
  lr_min_ratio=0.1 \
  lr_warmup_steps=5000
```

---

## Resuming from Checkpoint

```bash
python pretrain.py \
  --config-name cfg_pretrain_muon \
  [... other args ...] \
  load_checkpoint=checkpoints/your_run_name/step_XXXXX \
  +run_name="resumed_training"
```

---

## Useful Monitoring Commands

### Check GPU Usage

```bash
watch -n 1 nvidia-smi
```

### Monitor Training Logs

```bash
tail -f outputs/your_run_name/train.log
```

### Find Latest Checkpoint

```bash
ls -lth checkpoints/your_run_name/ | head
```

### Check W&B Run

```bash
# Your W&B project should be visible at:
# https://wandb.ai/your-username/your-project
```

---

## Configuration Files

- **Default config:** `config/cfg_pretrain.yaml`
- **Muon config:** `config/cfg_pretrain_muon.yaml`
- **Architecture configs:** `config/arch/trm.yaml`, `config/arch/hrm.yaml`

---

## Key Parameter Reference

| Parameter | Default (Muon) | Default (Original) | Notes |
|-----------|----------------|-------------------|-------|
| `use_muon` | `True` | `False` | Enable Muon optimizer |
| `muon_lr` | `4e-3` | N/A | Muon learning rate (was 2e-2, reduced for stability) |
| `lr` | `1e-4` | `1e-4` | AdamW learning rate (1D params) |
| `puzzle_emb_lr` | `1e-2` | `1e-2` | SignSGD learning rate (embeddings) |
| `lr_min_ratio` | `0.1` | `1.0` | Cosine decay min (0.1 = decay to 10%) |
| `lr_warmup_steps` | `2000` | `2000` | Warmup steps |
| `weight_decay` | `0.1` | `0.1` | Weight decay for model params |
| `global_batch_size` | `768` | `768` | Total batch size across GPUs |
| `ema` | `True` | `False` | Use exponential moving average |
| `ema_rate` | `0.999` | `0.999` | EMA decay rate |

---

## Troubleshooting

### Training Diverges

```bash
# Lower learning rates
python pretrain.py [...] muon_lr=1e-2 lr_warmup_steps=4000

# Or disable Muon
python pretrain.py [...] use_muon=False
```

### Out of Memory

```bash
# Reduce batch size
python pretrain.py [...] global_batch_size=384

# Or reduce model size
python pretrain.py [...] arch.hidden_size=256
```

### Slow Training

```bash
# Increase batch size (if GPU memory allows)
python pretrain.py [...] global_batch_size=1024

# Or reduce cycles
python pretrain.py [...] arch.L_cycles=4
```

---

## Quick Tests (Fast Debugging)

**Sanity check (1 epoch):**
```bash
python pretrain.py arch=trm data_paths="[data/sudoku-extreme-1k-aug-1000]" epochs=1
```

**Short run (100 steps):**
```bash
python pretrain.py \
  arch=trm \
  data_paths="[data/sudoku-extreme-1k-aug-1000]" \
  epochs=100 \
  eval_interval=50 \
  +run_name="quick_test"
```

---

## Analysis Output

The `analyze_convergence.py` script creates:
- `latent_convergence_analysis.png` - 4 plots showing convergence behavior
- Console output with convergence assessment

Look for:
- ✓ z_H has CONVERGED
- ✓ Outputs have STABILIZED
- Recommendations on adjusting H_cycles/L_cycles
