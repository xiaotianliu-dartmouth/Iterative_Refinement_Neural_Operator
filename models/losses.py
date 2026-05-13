"""
IRNO Loss Functions.

Implements the training objective from Section 2.3:
L_total = L_spatial + β_spectral · L_spectral + β_fp · L_fp
"""

import torch
import torch.nn.functional as F


def spatial_loss(pred, target):
    """L2 spatial loss: ||h_k - y||^2"""
    return F.mse_loss(pred, target)


def spectral_loss(pred, target, lambda_k=1.5):
    """
    Frequency-domain loss with radial weighting.

    From Section 2.3.2:
    ρ(ω, λ_k) = 1 + (|ω|/|ω|_nyq)^λ_k
    L_spectral = (1/HW) · (1/ρ̄_k) · Σ_ω ρ(ω, λ_k) · ||ĥ(ω)| - |ŷ(ω)||^2

    Args:
        pred: Predicted tensor (B, C, H, W)
        target: Target tensor (B, C, H, W)
        lambda_k: Frequency weighting exponent

    Returns:
        Normalized spectral loss
    """
    H, W = pred.shape[-2], pred.shape[-1]

    # Compute FFT
    pred_fft = torch.fft.rfft2(pred, dim=(-2, -1))
    target_fft = torch.fft.rfft2(target, dim=(-2, -1))

    pred_amp = torch.abs(pred_fft)
    target_amp = torch.abs(target_fft)

    # Compute radial frequency weights
    freq_h = torch.fft.fftfreq(H, d=1.0).to(pred.device)
    freq_w = torch.fft.rfftfreq(W, d=1.0).to(pred.device)

    freq_h_grid = freq_h[:, None].expand(-1, freq_w.shape[0])
    freq_w_grid = freq_w[None, :].expand(freq_h.shape[0], -1)

    radial_freq = torch.sqrt(freq_h_grid**2 + freq_w_grid**2)
    nyquist = 0.5
    relative_freq = radial_freq / nyquist

    # ρ(ω, λ_k) = 1 + (|ω|/|ω|_nyq)^λ_k
    weights = (1.0 + relative_freq**lambda_k).unsqueeze(0).unsqueeze(0)

    # Normalize by mean weight
    weights = weights / weights.mean()

    # Weighted amplitude MSE
    loss = torch.mean(weights * (pred_amp - target_amp)**2)

    # Extra /HW compensates for PyTorch's unnormalized FFT (coefficients scale as √(HW)·σ)
    loss = loss / (H * W)

    return loss


def progressive_spectral_loss(pred, target, step, K, lambda_start=1.0, lambda_end=2.0):
    """
    Progressive spectral loss where λ_k increases linearly over refinement steps.

    Args:
        pred: Predicted tensor
        target: Target tensor
        step: Current refinement step (0 to K-1)
        K: Total refinement steps
        lambda_start: Initial exponent
        lambda_end: Final exponent

    Returns:
        Spectral loss with progressive weighting
    """
    progress = step / max(K - 1, 1)
    lambda_k = lambda_start + (lambda_end - lambda_start) * progress
    return spectral_loss(pred, target, lambda_k=lambda_k)


def fixed_point_loss(refinement_op, x, y):
    """
    Fixed-point regularization: L_fp = ||Φ_θ(x, y)||^2

    Ensures that at the true solution, the refinement output is zero.
    This minimizes the bias term in Theorem 3.1.

    Args:
        refinement_op: The refinement operator Φ_θ
        x: Input tensor
        y: Ground truth solution

    Returns:
        L2 norm of Φ_θ(x, y)
    """
    phi_input = torch.cat([x, y], dim=1)
    residual_at_solution = refinement_op(phi_input)
    return torch.mean(residual_at_solution ** 2)


def combined_loss(pred, target, step, K, cfg):
    """
    Compute combined loss for a single refinement step.

    Args:
        pred: Prediction at step k
        target: Ground truth
        step: Current refinement step
        K: Total refinement steps
        cfg: Config with loss weights

    Returns:
        total_loss, spatial_loss, spectral_loss (for logging)
    """
    loss_spatial = spatial_loss(pred, target)

    spectral_cfg = cfg.training.get('spectral_loss', {})
    use_spectral = spectral_cfg.get('enabled', False)

    if use_spectral:
        weight = spectral_cfg.get('weight', 1.0)
        lambda_start = spectral_cfg.get('lambda_start', 1.0)
        lambda_end = spectral_cfg.get('lambda_end', 2.0)

        loss_spectral = progressive_spectral_loss(
            pred, target, step, K, lambda_start, lambda_end
        )
        total = loss_spatial + weight * loss_spectral
    else:
        loss_spectral = torch.tensor(0.0)
        total = loss_spatial

    return total, loss_spatial, loss_spectral
