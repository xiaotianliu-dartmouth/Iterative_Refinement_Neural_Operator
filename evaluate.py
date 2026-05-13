"""
IRNO Evaluation Script.

Evaluates the trained refinement operator by computing VRMSE
at each refinement step k = 0, 1, ..., K, K+1, ...
"""

import os
import argparse
import torch
from torch.utils.data import DataLoader

from utils import load_config
from utils.metrics import compute_vrmse_per_step, print_vrmse_table
from models import RefinementOperator, load_base_operator
from train import ActiveMatterDataset


def load_refinement_operator(checkpoint_path, in_channels, out_channels, cfg, device):
    """Load trained refinement operator from checkpoint."""
    ref_cfg = cfg.model['refinement']
    refinement_op = RefinementOperator(
        in_channels=in_channels,
        out_channels=out_channels,
        base_channels=ref_cfg['base_channels'],
        depth=ref_cfg['depth'],
        padding_type=ref_cfg['padding_type'],
        norm_type=ref_cfg.get('norm_type', 'layer'),
        num_groups=ref_cfg.get('num_groups', 8),
    )

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Handle DataParallel state dict
    state_dict = checkpoint['model_state_dict']
    state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

    refinement_op.load_state_dict(state_dict)
    refinement_op.to(device)
    refinement_op.eval()

    K = checkpoint.get('K', cfg.training['K'])

    return refinement_op, K


def evaluate(config_path, checkpoint_path, extra_steps=6):
    """
    Evaluate IRNO model.

    Args:
        config_path: Path to config file
        checkpoint_path: Path to model checkpoint
        extra_steps: Additional refinement steps beyond K for extrapolation
    """
    cfg = load_config(config_path)

    # Device setup
    device, _ = cfg.get_device()
    print(f"Device: {device}")

    # Create test dataloader
    test_ds = ActiveMatterDataset(
        base_path=cfg.dataset['base_path'],
        split='test',
        n_frames_input=cfg.dataset['n_frames_input'],
        normalization=cfg.dataset['normalization']
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=cfg.training['batch_size'],
        shuffle=False,
        num_workers=cfg.dataset['dataloader']['num_workers'],
        pin_memory=cfg.dataset['dataloader']['pin_memory']
    )

    print(f"Test samples: {len(test_ds)}")

    # Get channel dimensions
    x, y = next(iter(test_loader))
    in_channels = x.shape[1] + y.shape[1]
    out_channels = y.shape[1]

    # Load models
    base_op = load_base_operator(cfg.model['base']['pretrained_name'], device)
    print("Base operator loaded")

    refinement_op, K = load_refinement_operator(
        checkpoint_path, in_channels, out_channels, cfg, device
    )
    print(f"Refinement operator loaded (K={K})")

    alpha = cfg.training['alpha']
    print(f"Alpha: {alpha}")

    # Compute VRMSE at each step
    print("\nComputing VRMSE...")
    vrmse_scores = compute_vrmse_per_step(
        test_loader, base_op, refinement_op, K, alpha, device, extra_steps
    )

    # Print results
    print_vrmse_table(vrmse_scores, K)

    return vrmse_scores


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate IRNO')
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Path to config file')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--extra_steps', type=int, default=6,
                       help='Extra refinement steps for extrapolation')
    args = parser.parse_args()

    cfg = load_config(args.config)
    os.environ["CUDA_VISIBLE_DEVICES"] = cfg.gpu['ids']

    evaluate(args.config, args.checkpoint, args.extra_steps)
