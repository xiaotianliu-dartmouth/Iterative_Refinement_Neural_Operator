"""
Evaluation metrics for IRNO.
"""

import torch
import numpy as np
from the_well.benchmark.metrics import VRMSE


def compute_vrmse(pred, target, metadata):
    """
    Compute Variance-scaled RMSE.

    VRMSE = sqrt(MSE / Var(target))

    Args:
        pred: Predictions (B, C, H, W)
        target: Ground truth (B, C, H, W)
        metadata: Dataset metadata from the_well

    Returns:
        VRMSE values per sample
    """
    return VRMSE.eval(pred, target, meta=metadata)


def compute_vrmse_per_step(dataloader, base_op, refinement_op, K, alpha, device, extra_steps=0):
    """
    Compute VRMSE at each refinement step.

    Args:
        dataloader: DataLoader for evaluation
        base_op: Base operator T_base
        refinement_op: Refinement operator Φ_θ
        K: Training horizon
        alpha: Step size
        device: Compute device
        extra_steps: Additional steps beyond K for extrapolation

    Returns:
        List of average VRMSE per step
    """
    base_op.eval()
    refinement_op.eval()

    total_steps = K + extra_steps + 1  # +1 for base operator
    total_losses = [0.0] * total_steps
    n_samples = [0] * total_steps

    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)

            # Step 0: Base operator
            h = base_op(x)
            predictions = [h.clone()]

            # Steps 1 to K + extra_steps
            for _ in range(K + extra_steps):
                phi_input = torch.cat([x, h], dim=1)
                residual = refinement_op(phi_input)
                h = h + alpha * residual
                predictions.append(h.clone())

            # Compute VRMSE for each step
            metadata = dataloader.dataset.dataset.metadata
            for i, pred in enumerate(predictions):
                vrmse = compute_vrmse(pred, y, metadata)

                # Filter invalid values
                valid_mask = torch.isfinite(vrmse)
                valid_values = vrmse[valid_mask]

                if valid_values.numel() > 0:
                    total_losses[i] += valid_values.sum().item()
                    n_samples[i] += valid_values.numel()

    # Compute averages
    avg_vrmse = [
        total / n if n > 0 else float('nan')
        for total, n in zip(total_losses, n_samples)
    ]

    return avg_vrmse


def print_vrmse_table(vrmse_scores, K):
    """Print formatted VRMSE results table."""
    print("\n" + "=" * 50)
    print("VRMSE at Each Refinement Step")
    print("=" * 50)

    for i, score in enumerate(vrmse_scores):
        if i == 0:
            label = "Base Operator (k=0)"
        elif i <= K:
            label = f"Step {i}"
        else:
            label = f"Step {i} (extrapolation)"

        print(f"  {label:25s}: {score:.6f}")

    print("=" * 50)

    # Compute improvement
    if len(vrmse_scores) > 1:
        base = vrmse_scores[0]
        final = vrmse_scores[K] if K < len(vrmse_scores) else vrmse_scores[-1]
        reduction = (base - final) / base * 100
        print(f"  Reduction: {reduction:.2f}%")
