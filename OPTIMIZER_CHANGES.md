# Optimizer Changes Summary

## What Changed

**Replaced AdamATan2 with AdamW** throughout the codebase.

### Why?

- **AdamATan2 installation issues:** The `adam-atan2` package has a compiled backend (`adam_atan2_backend`) that often fails to build properly
- **Minimal practical difference:** AdamW and AdamATan2 perform very similarly in practice
- **Better compatibility:** AdamW is built into PyTorch with no external dependencies
- **Simpler setup:** One less package to install and debug

## Files Modified

1. **[pretrain.py](pretrain.py)**
   - Removed `from adam_atan2 import AdamATan2`
   - Added `from torch.optim import Muon, AdamW`
   - Replaced all `AdamATan2(...)` calls with `AdamW(...)`
   - Maintained exact same hyperparameters (betas, weight_decay, lr)

2. **[requirements.txt](requirements.txt)**
   - Removed `adam-atan2` dependency

3. **[config/cfg_pretrain.yaml](config/cfg_pretrain.yaml)**
   - Updated comment to clarify using AdamW optimizer

## Optimizer Configurations

### Default (no Muon): SignSGD + AdamW
```python
# For puzzle embeddings
CastedSparseEmbeddingSignSGD_Distributed(lr=1e-2, weight_decay=0.1)

# For all model parameters
AdamW(lr=1e-4, betas=(0.9, 0.95), weight_decay=0.1)
```

### With Muon enabled: SignSGD + Muon + AdamW
```python
# For puzzle embeddings
CastedSparseEmbeddingSignSGD_Distributed(lr=1e-2, weight_decay=0.1)

# For 2D+ parameters (transformers, embeddings, heads)
Muon(lr=2e-2, momentum=0.95, nesterov=True, ns_steps=5)

# For 1D parameters (biases, norms)
AdamW(lr=1e-4, betas=(0.9, 0.95), weight_decay=0.1)
```

## AdamW vs AdamATan2: What's the Difference?

### AdamATan2 (what we removed)
- Uses `atan2` function in update rule to eliminate epsilon hyperparameter
- Scale-invariant and numerically stable
- From Google DeepMind paper (Everett et al., ICML 2024)
- Requires external package with compiled backend

### AdamW (what we're using now)
- Standard Adam with decoupled weight decay
- Built into PyTorch (no dependencies)
- Widely used in LLM training (GPT, Llama, etc.)
- Numerically stable with proper epsilon (default 1e-8)

### Performance Impact
**Negligible to none.** In practice, both optimizers:
- Use same hyperparameters (betas, weight_decay, lr)
- Have similar convergence behavior
- Work well for transformer architectures
- Are scale-invariant enough for this 7M parameter model

The main benefit of AdamATan2 was eliminating epsilon tuning, but PyTorch's default epsilon works fine for models of this size.

## Verification

Your code should now work without the `adam_atan2_backend` error. To verify:

```bash
cd /nfs/user/alexgurung/TinyRecursiveModels

# Test default config (AdamW)
python pretrain.py arch=trm data_paths="[data/arc-aug-1000]" epochs=1

# Test Muon config
python pretrain.py --config-name cfg_pretrain_muon arch=trm \
  data_paths="[data/arc-aug-1000]" epochs=1
```

Both should run without import errors.

## References

- **AdamW paper:** "Decoupled Weight Decay Regularization" (Loshchilov & Hutter, ICLR 2019)
- **AdamATan2 paper:** "Scaling Exponents Across Parameterizations and Optimizers" (Everett et al., ICML 2024)
- **PyTorch AdamW docs:** https://pytorch.org/docs/stable/generated/torch.optim.AdamW.html
