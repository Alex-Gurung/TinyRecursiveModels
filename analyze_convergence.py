"""
Analyze latent convergence in TinyRecursiveModels.

This script investigates whether the latent representations z_H and z_L
converge during recursive refinement cycles, and how this relates to the
halting mechanism.

Usage:
    python analyze_convergence.py [--checkpoint path/to/checkpoint.pt]
"""

import torch
import yaml
import argparse
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

from models.recursive_reasoning.trm import TinyRecursiveReasoningModel_ACTV1_Inner


def analyze_convergence(model_cfg, checkpoint_path=None, num_extra_cycles=5, batch_size=16):
    """
    Run the model for extra cycles and track latent convergence.

    Args:
        model_cfg: Model configuration dict
        checkpoint_path: Optional path to load weights from
        num_extra_cycles: How many extra H-cycles to run beyond configured amount
        batch_size: Batch size for dummy data
    """
    # Create model
    model = TinyRecursiveReasoningModel_ACTV1_Inner(model_cfg)
    model.eval()

    # Load checkpoint if provided
    if checkpoint_path:
        print(f"Loading checkpoint from {checkpoint_path}")
        state_dict = torch.load(checkpoint_path, map_location='cpu')
        model.load_state_dict(state_dict, strict=False)

    # Create dummy batch (random inputs for now)
    print(f"Creating dummy batch with size {batch_size}")
    batch = {
        'inputs': torch.randint(0, model_cfg['vocab_size'], (batch_size, model_cfg['seq_len'])),
        'puzzle_identifiers': torch.randint(0, model_cfg['num_puzzle_identifiers'], (batch_size,))
    }

    # Initialize carry
    carry = model.empty_carry(batch_size)
    carry = model.reset_carry(torch.ones(batch_size, dtype=torch.bool), carry)

    # Track convergence
    z_L_history = []
    z_H_history = []
    output_history = []
    q_halt_history = []

    total_cycles = model_cfg['H_cycles'] + num_extra_cycles

    print(f"Running {total_cycles} H-cycles (configured: {model_cfg['H_cycles']}, extra: {num_extra_cycles})")

    with torch.no_grad():
        input_embeddings = model._input_embeddings(batch["inputs"], batch["puzzle_identifiers"])
        seq_info = dict(cos_sin=model.rotary_emb() if hasattr(model, "rotary_emb") else None)

        z_H, z_L = carry.z_H, carry.z_L

        # Run cycles manually
        for h_cycle in range(total_cycles):
            z_H_history.append(z_H.clone())

            # L-cycles within each H-cycle
            for l_cycle in range(model_cfg['L_cycles']):
                z_L = model.L_level(z_L, z_H + input_embeddings, **seq_info)
                z_L_history.append(z_L.clone())

            # H update
            z_H = model.L_level(z_H, z_L, **seq_info)

            # Get outputs
            output = model.lm_head(z_H)[:, model.puzzle_emb_len:]
            q_logits = model.q_head(z_H[:, 0])
            output_history.append(output.clone())
            q_halt_history.append(q_logits[:, 0].clone())  # q_halt_logits

    # Compute deltas
    print("Computing deltas...")
    z_L_deltas = [torch.norm(z_L_history[i+1] - z_L_history[i], dim=-1).mean().item()
                  for i in range(len(z_L_history)-1)]
    z_H_deltas = [torch.norm(z_H_history[i+1] - z_H_history[i], dim=-1).mean().item()
                  for i in range(len(z_H_history)-1)]

    # Compute output changes (classification change rate)
    output_changes = [(output_history[i+1].argmax(-1) != output_history[i].argmax(-1)).float().mean().item()
                      for i in range(len(output_history)-1)]

    # Compute average q_halt across batch
    q_halt_means = [q.mean().item() for q in q_halt_history]

    # Plot results
    print("Creating plots...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: z_L convergence
    axes[0, 0].plot(z_L_deltas, label='z_L L2 delta', marker='o', markersize=3)
    axes[0, 0].axvline(x=model_cfg['H_cycles'] * model_cfg['L_cycles'], color='red',
                       linestyle='--', label=f"Configured cycles ({model_cfg['H_cycles']} H)")
    axes[0, 0].set_ylabel('L2 Norm of Change')
    axes[0, 0].set_xlabel('L-cycle Step')
    axes[0, 0].set_title('z_L Convergence (should decrease if settling)')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_yscale('log')

    # Plot 2: z_H convergence
    axes[0, 1].plot(z_H_deltas, label='z_H L2 delta', marker='s', markersize=4, color='orange')
    axes[0, 1].axvline(x=model_cfg['H_cycles'], color='red',
                       linestyle='--', label=f"Configured cycles ({model_cfg['H_cycles']})")
    axes[0, 1].set_ylabel('L2 Norm of Change')
    axes[0, 1].set_xlabel('H-cycle Step')
    axes[0, 1].set_title('z_H Convergence (should decrease if settling)')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_yscale('log')

    # Plot 3: Output stability
    axes[1, 0].plot(output_changes, label='Prediction change rate', marker='^', markersize=4, color='green')
    axes[1, 0].axvline(x=model_cfg['H_cycles'], color='red',
                       linestyle='--', label=f"Configured cycles ({model_cfg['H_cycles']})")
    axes[1, 0].set_ylabel('Fraction of Predictions Changed')
    axes[1, 0].set_xlabel('H-cycle Step')
    axes[1, 0].set_title('Output Stability (should approach 0 if converged)')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Plot 4: Q-halt signal
    axes[1, 1].plot(q_halt_means, label='Mean q_halt_logits', marker='D', markersize=4, color='purple')
    axes[1, 1].axhline(y=0, color='black', linestyle=':', label='Halt threshold (0)')
    axes[1, 1].axvline(x=model_cfg['H_cycles'], color='red',
                       linestyle='--', label=f"Configured cycles ({model_cfg['H_cycles']})")
    axes[1, 1].set_ylabel('Q-Halt Logits (mean across batch)')
    axes[1, 1].set_xlabel('H-cycle Step')
    axes[1, 1].set_title('Halting Signal (>0 means "halt now")')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = 'latent_convergence_analysis.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved plots to {output_path}")

    # Print summary
    print("\n" + "="*80)
    print("CONVERGENCE ANALYSIS SUMMARY")
    print("="*80)

    print(f"\nz_H deltas:")
    print(f"  First 3 H-cycles: {z_H_deltas[:3]}")
    print(f"  Last 3 H-cycles:  {z_H_deltas[-3:]}")
    print(f"  Final delta:      {z_H_deltas[-1]:.6f}")

    print(f"\nOutput changes:")
    print(f"  First 3 H-cycles: {output_changes[:3]}")
    print(f"  Last 3 H-cycles:  {output_changes[-3:]}")
    print(f"  Final change:     {output_changes[-1]:.6f}")

    print(f"\nQ-halt signal:")
    print(f"  First 3 H-cycles: {q_halt_means[:3]}")
    print(f"  At configured end (H={model_cfg['H_cycles']}): {q_halt_means[model_cfg['H_cycles']-1]:.3f}")
    print(f"  Last 3 H-cycles:  {q_halt_means[-3:]}")

    # Convergence assessment
    print("\n" + "="*80)
    print("CONVERGENCE ASSESSMENT")
    print("="*80)

    converged_delta_threshold = 0.01
    converged_output_threshold = 0.01

    if z_H_deltas[-1] < converged_delta_threshold:
        print(f"✓ z_H has CONVERGED (final delta {z_H_deltas[-1]:.6f} < {converged_delta_threshold})")
    else:
        print(f"✗ z_H still CHANGING (final delta {z_H_deltas[-1]:.6f} >= {converged_delta_threshold})")

    if output_changes[-1] < converged_output_threshold:
        print(f"✓ Outputs have STABILIZED (final change {output_changes[-1]:.6f} < {converged_output_threshold})")
    else:
        print(f"✗ Outputs still CHANGING (final change {output_changes[-1]:.6f} >= {converged_output_threshold})")

    # Find when it converged
    try:
        converged_at = next(i for i, d in enumerate(z_H_deltas) if d < converged_delta_threshold)
        print(f"\nz_H converged after H-cycle {converged_at+1} (configured: {model_cfg['H_cycles']})")
        if converged_at + 1 < model_cfg['H_cycles']:
            print(f"  → Could potentially reduce H_cycles from {model_cfg['H_cycles']} to {converged_at+1}")
        elif converged_at + 1 == model_cfg['H_cycles']:
            print(f"  → Current H_cycles={model_cfg['H_cycles']} is well-tuned")
        else:
            print(f"  → Might benefit from increasing H_cycles to {converged_at+1}")
    except StopIteration:
        print(f"\nz_H did NOT converge within {total_cycles} H-cycles")

    # Q-halt analysis
    if q_halt_means[model_cfg['H_cycles']-1] > 0:
        print(f"\nQ-head wants to halt at configured end (q_halt={q_halt_means[model_cfg['H_cycles']-1]:.3f} > 0)")
    else:
        print(f"\nQ-head does NOT want to halt at configured end (q_halt={q_halt_means[model_cfg['H_cycles']-1]:.3f} <= 0)")

    print("="*80 + "\n")

    return {
        'z_H_deltas': z_H_deltas,
        'z_L_deltas': z_L_deltas,
        'output_changes': output_changes,
        'q_halt_means': q_halt_means
    }


def main():
    parser = argparse.ArgumentParser(description='Analyze latent convergence in TRM')
    parser.add_argument('--config', type=str, default='config/arch/trm.yaml',
                        help='Path to architecture config')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to model checkpoint (optional)')
    parser.add_argument('--extra-cycles', type=int, default=5,
                        help='Number of extra H-cycles to run beyond configured amount')
    parser.add_argument('--batch-size', type=int, default=16,
                        help='Batch size for dummy data')

    args = parser.parse_args()

    # Load config
    print(f"Loading config from {args.config}")
    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)

    # Add required fields
    model_cfg = {
        **cfg,
        'batch_size': args.batch_size,
        'vocab_size': 11,  # Default for Sudoku/ARC
        'seq_len': 900,    # Default
        'num_puzzle_identifiers': 1000,
        'causal': False
    }

    print(f"Model config: H_cycles={cfg['H_cycles']}, L_cycles={cfg['L_cycles']}, "
          f"L_layers={cfg['L_layers']}, hidden_size={cfg['hidden_size']}")

    # Run analysis
    results = analyze_convergence(
        model_cfg=model_cfg,
        checkpoint_path=args.checkpoint,
        num_extra_cycles=args.extra_cycles,
        batch_size=args.batch_size
    )


if __name__ == '__main__':
    main()
