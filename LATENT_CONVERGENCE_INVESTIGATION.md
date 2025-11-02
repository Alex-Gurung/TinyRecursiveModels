# Investigating Latent Convergence and Halting Behavior

## Hypothesis
The latent representations `z_H` and `z_L` should "settle" during recursive refinement cycles, and this convergence may correlate with when the Q-head decides to halt.

## Current Halting Mechanism

**From the code ([models/recursive_reasoning/trm.py](models/recursive_reasoning/trm.py)):**

### Training Time:
- Model runs for up to `halt_max_steps` iterations
- Halts early if `q_halt_logits > 0` (learned halting decision)
- Exploration: Random early stopping with probability `halt_exploration_prob`

### Evaluation Time:
- **Always runs for exactly `halt_max_steps`** (for batching consistency)
- Doesn't use Q-head during eval

### Recursive Structure:
```
For each iteration:
  For H_cycles (e.g., 3):
    For L_cycles (e.g., 6):
      z_L = L_level(z_L, z_H + input)  # Refine lower level
    z_H = L_level(z_H, z_L)             # Update higher level
```

Only the **last H-cycle has gradients**, previous cycles run with `torch.no_grad()`.

---

## Metrics to Track

### 1. Latent Change Magnitude
Track how much the latents change between cycles:

```python
# After each L-cycle update
delta_z_L = torch.norm(z_L_new - z_L_old, dim=-1).mean()
delta_z_H = torch.norm(z_H_new - z_H_old, dim=-1).mean()
```

**Expected behavior:**
- Large changes initially
- Smaller changes as it converges
- Near-zero changes when settled

### 2. Q-Halt vs Convergence
Track the correlation between:
- Latent convergence (small delta)
- Q-head halt signal (`q_halt_logits`)

**Hypothesis:** When latents stop changing, `q_halt_logits` should be positive.

### 3. Output Stability
Track how much the predicted output changes:

```python
# After each iteration
output_change = (outputs_new.argmax(-1) != outputs_old.argmax(-1)).float().mean()
```

**Expected:** Predictions should stabilize even if latents haven't fully converged.

### 4. Per-Example Halting Steps
During training, log:
- How many steps each example actually took before halting
- Distribution across the batch

**Question:** Do hard examples take more steps?

---

## Implementation Plan

### Option 1: Quick Logging (Minimal Code Changes)

Add logging to the forward pass in [models/recursive_reasoning/trm.py](models/recursive_reasoning/trm.py):

```python
# In TinyRecursiveReasoningModel_ACTV1_Inner.forward()
# After line 205 (before iterations)

z_L_deltas = []
z_H_deltas = []

# During the iteration loop
for _H_step in range(self.config.H_cycles):
    z_H_old = z_H.clone()

    for _L_step in range(self.config.L_cycles):
        z_L_old = z_L.clone()
        z_L = self.L_level(z_L, z_H + input_embeddings, **seq_info)

        # Track convergence
        if not self.training:  # Only during eval to avoid overhead
            delta = torch.norm(z_L - z_L_old, dim=-1).mean()
            z_L_deltas.append(delta.item())

    z_H = self.L_level(z_H, z_L, **seq_info)

    if not self.training:
        delta = torch.norm(z_H - z_H_old, dim=-1).mean()
        z_H_deltas.append(delta.item())

# Store in outputs for logging
return new_carry, output, (q_logits[..., 0], q_logits[..., 1]), \
       {"z_L_deltas": z_L_deltas, "z_H_deltas": z_H_deltas}
```

### Option 2: Full Analysis (Separate Script)

Create a dedicated analysis script that:
1. Loads a checkpoint
2. Runs eval on a batch
3. Manually steps through cycles
4. Plots convergence metrics

See below for implementation.

---

## Quick Analysis Script

Save this as `analyze_convergence.py`:

```python
import torch
import yaml
from models.recursive_reasoning.trm import TinyRecursiveReasoningModel_ACTV1_Inner
import matplotlib.pyplot as plt
import numpy as np

# Load model
with open('config/arch/trm.yaml', 'r') as f:
    cfg = yaml.safe_load(f)

model_cfg = {
    **cfg,
    'batch_size': 16,
    'vocab_size': 11,
    'seq_len': 900,
    'num_puzzle_identifiers': 1000,
    'causal': False
}

model = TinyRecursiveReasoningModel_ACTV1_Inner(model_cfg)
model.eval()

# Create dummy batch
batch = {
    'inputs': torch.randint(0, 11, (16, 900)),
    'puzzle_identifiers': torch.randint(0, 1000, (16,))
}

# Initialize carry
carry = model.empty_carry(16)
carry = model.reset_carry(torch.ones(16, dtype=torch.bool), carry)

# Track convergence
z_L_history = []
z_H_history = []
output_history = []

with torch.no_grad():
    input_embeddings = model._input_embeddings(batch["inputs"], batch["puzzle_identifiers"])
    seq_info = dict(cos_sin=model.rotary_emb() if hasattr(model, "rotary_emb") else None)

    z_H, z_L = carry.z_H, carry.z_L

    # Run multiple cycles manually
    for h_cycle in range(cfg['H_cycles'] * 3):  # Run 3x more to see if it converges
        z_H_history.append(z_H.clone())

        for l_cycle in range(cfg['L_cycles']):
            z_L = model.L_level(z_L, z_H + input_embeddings, **seq_info)
            z_L_history.append(z_L.clone())

        z_H = model.L_level(z_H, z_L, **seq_info)

        # Get output
        output = model.lm_head(z_H)[:, model.puzzle_emb_len:]
        output_history.append(output.clone())

# Compute deltas
z_L_deltas = [torch.norm(z_L_history[i+1] - z_L_history[i], dim=-1).mean().item()
              for i in range(len(z_L_history)-1)]
z_H_deltas = [torch.norm(z_H_history[i+1] - z_H_history[i], dim=-1).mean().item()
              for i in range(len(z_H_history)-1)]

# Plot
fig, axes = plt.subplots(2, 1, figsize=(10, 8))

axes[0].plot(z_L_deltas, label='z_L delta', marker='o')
axes[0].axvline(x=cfg['H_cycles'], color='red', linestyle='--', label='Original H_cycles')
axes[0].set_ylabel('L2 Norm of Change')
axes[0].set_xlabel('L-cycle Step')
axes[0].set_title('z_L Convergence (should decrease if settling)')
axes[0].legend()
axes[0].grid(True)

axes[1].plot(z_H_deltas, label='z_H delta', marker='s', color='orange')
axes[1].axvline(x=cfg['H_cycles'], color='red', linestyle='--', label='Original H_cycles')
axes[1].set_ylabel('L2 Norm of Change')
axes[1].set_xlabel('H-cycle Step')
axes[1].set_title('z_H Convergence (should decrease if settling)')
axes[1].legend()
axes[1].grid(True)

plt.tight_layout()
plt.savefig('latent_convergence.png', dpi=150)
print("Saved to latent_convergence.png")

# Print summary
print(f"\\nz_L deltas (first 5): {z_L_deltas[:5]}")
print(f"z_L deltas (last 5):  {z_L_deltas[-5:]}")
print(f"\\nz_H deltas (first 5): {z_H_deltas[:5]}")
print(f"z_H deltas (last 5):  {z_H_deltas[-5:]}")

# Check if converged
if z_H_deltas[-1] < 0.01:
    print("\\n✓ Model appears to have converged (delta < 0.01)")
else:
    print(f"\\n✗ Model still changing (final delta = {z_H_deltas[-1]:.4f})")
```

Run with:
```bash
python analyze_convergence.py
```

---

## Questions to Answer

1. **Do the latents converge?**
   - If yes: How many cycles does it actually need?
   - If no: Maybe the task requires continuous refinement

2. **What does Q-head learn?**
   - Does it predict halt when latents converge?
   - Or does it learn task-specific stopping criteria?

3. **Are cycles being wasted?**
   - If converges after 2 H-cycles but using 3, you're wasting compute
   - Could reduce `H_cycles` or `L_cycles`

4. **Do different tasks have different convergence?**
   - Sudoku might converge faster than ARC
   - Could use adaptive halting per task

---

## About Your Diverging Training

Looking at your W&B plots:
- `train/lm_loss` spikes around step 2k-2.5k
- `train/accuracy` drops from ~0.9 to ~0.5

**Possible causes:**

1. **Learning rate too high:** The Muon LR of `2e-2` might be too aggressive
   - Try: `muon_lr=1e-2` or `1.5e-2`

2. **Warmup too short:** 2000 steps might not be enough
   - Try: `lr_warmup_steps=4000` or `5000`

3. **Gradient explosion:** The recursive nature can cause gradient issues
   - Check gradient norms in W&B
   - Try: Add gradient clipping (max_norm=1.0)

4. **Q-head instability:** The halt logits might be messing with training
   - The Q-head is initialized with bias=-5 (very negative)
   - Might be learning unstable halt patterns

5. **Short run artifact:** 5000 epochs might be too short for Sudoku
   - The paper uses 50k epochs for Sudoku
   - Early instability might resolve later

**Quick fix to try:**
```bash
python pretrain.py \
  --config-name cfg_pretrain_muon \
  muon_lr=1e-2 \
  lr_warmup_steps=4000 \
  ... (rest of args)
```

Or for a more conservative test, just use AdamW baseline first to verify the setup works.
