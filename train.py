"""
IRNO Training Script.

Trains the refinement operator Φ_θ with multi-step supervision,
progressive spectral loss, and fixed-point regularization.
"""

import os
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from utils import load_config
from models import RefinementOperator, load_base_operator
from models.losses import spatial_loss, progressive_spectral_loss, fixed_point_loss


# Dataset wrapper
from the_well.data import WellDataset
from the_well.data.normalization import ZScoreNormalization, RMSNormalization
from einops import rearrange


class ActiveMatterDataset:
    """Dataset wrapper for Active Matter from The Well."""

    def __init__(self, base_path, split, n_frames_input=4, normalization="zscore"):
        norm_type = ZScoreNormalization if normalization == "zscore" else RMSNormalization

        self.dataset = WellDataset(
            well_base_path=base_path,
            well_dataset_name="active_matter",
            well_split_name=split,
            n_steps_input=n_frames_input,
            n_steps_output=1,
            use_normalization=True,
            normalization_type=norm_type,
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        x = sample["input_fields"]   # [T, H, W, F]
        y = sample["output_fields"]  # [1, H, W, F]

        # Rearrange to (C, H, W) format with TF ordering (Time-first)
        x = rearrange(x, "T H W F -> (T F) H W")
        y = rearrange(y, "B H W F -> (B F) H W")

        return x.float(), y.float()


def set_seed(seed):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def create_dataloaders(cfg, n_gpu):
    """Create train and validation dataloaders."""
    batch_size = cfg.training['batch_size'] * n_gpu
    num_workers = cfg.dataset['dataloader']['num_workers'] * n_gpu

    train_ds = ActiveMatterDataset(
        base_path=cfg.dataset['base_path'],
        split='train',
        n_frames_input=cfg.dataset['n_frames_input'],
        normalization=cfg.dataset['normalization']
    )

    valid_ds = ActiveMatterDataset(
        base_path=cfg.dataset['base_path'],
        split='valid',
        n_frames_input=cfg.dataset['n_frames_input'],
        normalization=cfg.dataset['normalization']
    )

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=cfg.dataset['dataloader']['pin_memory']
    )

    valid_loader = DataLoader(
        valid_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=cfg.dataset['dataloader']['pin_memory']
    )

    return train_loader, valid_loader


def get_channel_dims(dataloader):
    """Get input/output channel dimensions."""
    x, y = next(iter(dataloader))
    in_channels = x.shape[1] + y.shape[1]  # Concatenate x and h
    out_channels = y.shape[1]
    return in_channels, out_channels


def train_epoch(base_op, refinement_op, train_loader, optimizer, device, cfg, epoch):
    """
    Train one epoch with multi-step supervision.

    From Section 2.3:
    L_total = (1/K) Σ_k [L_spatial + β_s·L_spectral] + β_fp·L_fp
    """
    refinement_op.train()
    base_op.eval()

    K = cfg.training['K']
    alpha = cfg.training['alpha']
    grad_clip = cfg.training.get('grad_clip', 1.0)

    # Loss settings
    spectral_cfg = cfg.training.get('spectral_loss', {})
    use_spectral = spectral_cfg.get('enabled', False)
    spectral_weight = spectral_cfg.get('weight', 1.0)
    spectral_warmup = spectral_cfg.get('warmup_epochs', 0)
    lambda_start = spectral_cfg.get('lambda_start', 1.0)
    lambda_end = spectral_cfg.get('lambda_end', 2.0)

    fp_cfg = cfg.training.get('fixed_point_reg', {})
    use_fp = fp_cfg.get('enabled', False)
    fp_weight = fp_cfg.get('weight', 0.01)
    fp_warmup = fp_cfg.get('warmup_epochs', 0)

    # Apply warmup
    if use_spectral and epoch < spectral_warmup:
        spectral_weight *= epoch / spectral_warmup
    if use_fp and epoch < fp_warmup:
        fp_weight *= epoch / fp_warmup

    epoch_losses = {'total': 0, 'spatial': 0, 'spectral': 0, 'fp': 0}
    n_batches = len(train_loader)

    for batch_idx, (x, y) in enumerate(train_loader):
        x, y = x.to(device), y.to(device)

        # Get base operator prediction (frozen)
        with torch.no_grad():
            h = base_op(x)

        step_losses = []
        total_spatial = 0
        total_spectral = 0

        # Multi-step supervision: iterate K steps
        for k in range(K):
            phi_input = torch.cat([x, h], dim=1)
            residual = refinement_op(phi_input)
            h = h + alpha * residual

            # Compute step loss
            loss_spatial = spatial_loss(h, y)
            total_spatial += loss_spatial.item()

            if use_spectral:
                loss_spectral = progressive_spectral_loss(
                    h, y, k, K, lambda_start, lambda_end
                )
                step_loss = loss_spatial + spectral_weight * loss_spectral
                total_spectral += loss_spectral.item()
            else:
                step_loss = loss_spatial

            step_losses.append(step_loss)

        # Average over trajectory
        loss = sum(step_losses) / K

        # Fixed-point regularization
        if use_fp:
            loss_fp = fixed_point_loss(refinement_op, x, y)
            loss = loss + fp_weight * loss_fp
            epoch_losses['fp'] += loss_fp.item()

        # Skip invalid losses
        if torch.isnan(loss) or torch.isinf(loss):
            continue

        optimizer.zero_grad()
        loss.backward()

        if grad_clip:
            nn.utils.clip_grad_norm_(refinement_op.parameters(), grad_clip)

        optimizer.step()

        epoch_losses['total'] += loss.item()
        epoch_losses['spatial'] += total_spatial / K
        epoch_losses['spectral'] += total_spectral / K

        if cfg.logging['verbose'] and (batch_idx + 1) % cfg.logging['print_every'] == 0:
            print(f"  Batch {batch_idx + 1}/{n_batches} | Loss: {loss.item():.6f}")

    # Average over batches
    for key in epoch_losses:
        epoch_losses[key] /= n_batches

    return epoch_losses


def validate(base_op, refinement_op, valid_loader, device, cfg):
    """Validate model on validation set."""
    refinement_op.eval()
    base_op.eval()

    K = cfg.training['K']
    alpha = cfg.training['alpha']
    threshold = cfg.validation['loss_threshold']

    valid_losses = []
    n_outliers = 0
    n_total = 0

    with torch.no_grad():
        for x, y in valid_loader:
            x, y = x.to(device), y.to(device)
            batch_size = x.shape[0]
            n_total += batch_size

            # Run K refinement steps
            h = base_op(x)
            for _ in range(K):
                phi_input = torch.cat([x, h], dim=1)
                residual = refinement_op(phi_input)
                h = h + alpha * residual

            # Per-sample loss
            per_sample = F.mse_loss(h, y, reduction='none').mean(dim=[1, 2, 3]).cpu()

            # Filter outliers
            valid_mask = per_sample <= threshold
            valid_losses.extend(per_sample[valid_mask].numpy())
            n_outliers += (batch_size - valid_mask.sum().item())

    if not valid_losses:
        return float('inf'), 100.0

    avg_loss = np.mean(valid_losses)
    outlier_pct = n_outliers / n_total * 100

    return avg_loss, outlier_pct


def save_checkpoint(refinement_op, optimizer, epoch, loss, K, filepath):
    """Save model checkpoint."""
    state_dict = refinement_op.module.state_dict() if isinstance(
        refinement_op, nn.DataParallel) else refinement_op.state_dict()

    torch.save({
        'epoch': epoch,
        'model_state_dict': state_dict,
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'K': K,
    }, filepath)


def train(config_path):
    """Main training function."""
    cfg = load_config(config_path)
    set_seed(cfg.seed)

    # Setup device
    device, n_gpu = cfg.get_device()
    print(f"Device: {device}, GPUs: {n_gpu}")

    # Create dataloaders
    train_loader, valid_loader = create_dataloaders(cfg, n_gpu)
    in_channels, out_channels = get_channel_dims(train_loader)
    print(f"Channels: in={in_channels}, out={out_channels}")

    # Load base operator (frozen)
    base_op = load_base_operator(cfg.model['base']['pretrained_name'], device)
    print("Base operator loaded")

    # Create refinement operator
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

    if n_gpu > 1:
        refinement_op = nn.DataParallel(refinement_op)
    refinement_op.to(device)

    n_params = sum(p.numel() for p in refinement_op.parameters() if p.requires_grad)
    print(f"Refinement operator parameters: {n_params:,}")

    # Optimizer and scheduler
    optimizer = torch.optim.Adam(
        refinement_op.parameters(),
        lr=float(cfg.training['learning_rate']),
        weight_decay=float(cfg.training['weight_decay'])
    )

    scheduler = None
    if cfg.training['scheduler']['enabled']:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=cfg.training['num_epochs'],
            eta_min=float(cfg.training['scheduler']['eta_min'])
        )

    # Training loop
    os.makedirs(cfg.checkpoint['dir'], exist_ok=True)
    best_loss = float('inf')
    best_epoch = 0

    print(f"\nTraining for {cfg.training['num_epochs']} epochs")
    print(f"K={cfg.training['K']}, alpha={cfg.training['alpha']}")
    print("=" * 50)

    for epoch in range(1, cfg.training['num_epochs'] + 1):
        print(f"\nEpoch {epoch}/{cfg.training['num_epochs']}")

        # Train
        train_losses = train_epoch(
            base_op, refinement_op, train_loader, optimizer, device, cfg, epoch
        )

        # Validate
        valid_loss, outlier_pct = validate(
            base_op, refinement_op, valid_loader, device, cfg
        )

        if scheduler:
            scheduler.step()
            lr = scheduler.get_last_lr()[0]
        else:
            lr = cfg.training['learning_rate']

        # Log
        print(f"  Train Loss: {train_losses['total']:.6f}")
        print(f"  Valid Loss: {valid_loss:.6f}")
        print(f"  LR: {lr:.2e}")

        # Save best
        if cfg.checkpoint['save_best'] and valid_loss < best_loss:
            if outlier_pct < cfg.validation['max_outlier_pct']:
                best_loss = valid_loss
                best_epoch = epoch
                path = os.path.join(cfg.checkpoint['dir'], 'best_model.pth')
                save_checkpoint(refinement_op, optimizer, epoch, valid_loss,
                              cfg.training['K'], path)
                print(f"  * New best model saved")

    # Save final
    if cfg.checkpoint['save_final']:
        path = os.path.join(cfg.checkpoint['dir'], 'final_model.pth')
        save_checkpoint(refinement_op, optimizer, epoch, valid_loss,
                       cfg.training['K'], path)

    print("\n" + "=" * 50)
    print(f"Training complete. Best: epoch {best_epoch}, loss {best_loss:.6f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train IRNO')
    parser.add_argument('--config', type=str, default='config.yaml')
    args = parser.parse_args()

    cfg = load_config(args.config)
    os.environ["CUDA_VISIBLE_DEVICES"] = cfg.gpu['ids']

    train(args.config)
