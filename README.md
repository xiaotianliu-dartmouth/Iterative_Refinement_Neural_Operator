# IRNO — Iterative Refinement Neural Operator

A learned **iterative refinement mechanism** for neural operators that improves accuracy, spectral fidelity, and stability by applying a contraction-based residual correction map at inference time.

## Overview

IRNO refines predictions from a **base operator** using a learned refinement operator Φ(x, h) that operates on the current solution estimate and input:

```
h_0 = T_base(x)                  # Base operator initialization
h_{k+1} = h_k + α · Φ(x, h_k)    # Iterative refinement
```

The refinement operator is trained with multi-step supervision, progressive spectral loss, and fixed-point regularization to ensure stable convergence.

## Installation

```bash
pip install -r requirements.txt
```

### Dependencies

- PyTorch ≥ 2.0
- the-well
- neuralop
- einops, PyYAML, numpy, matplotlib

## Project Structure

```
IRNO/
├── config.yaml          # Training configuration
├── train.py             # Training script
├── evaluate.py          # Evaluation script
├── requirements.txt     # Dependencies
├── data/
│   └── download_data.py # Dataset download utility
├── models/
│   ├── model.py         # RefinementOperator (Φ_θ)
│   └── losses.py        # Loss functions
└── utils/
    ├── config_loader.py # YAML config loader
    └── metrics.py       # VRMSE computation
```

## Quick Start

### 1. Configure

Edit `config.yaml` to set your dataset path:

```yaml
model:
  base:
    pretrained_name: "polymathic-ai/FNO-active_matter"

dataset:
  base_path: "/path/to/datasets"
```

### 2. Train

```bash
python train.py --config config.yaml
```

### 3. Evaluate

```bash
python evaluate.py --config config.yaml --checkpoint checkpoints/best_model.pth
```

## Configuration

```yaml
model:
  base:
    pretrained_name: "polymathic-ai/FNO-active_matter"
  refinement:
    base_channels: 16
    depth: 4
    padding_type: "circular"

training:
  num_epochs: 150
  batch_size: 16
  learning_rate: 3e-4
  K: 6                    # Training horizon (refinement steps)
  alpha: 0.2              # Step size

  spectral_loss:
    enabled: true
    weight: 1.0
    lambda_start: 1.0     # Progressive exponent
    lambda_end: 2.0

  fixed_point_reg:
    enabled: true
    weight: 0.01

dataset:
  name: "active_matter"
  base_path: "/path/to/datasets"
  n_frames_input: 4
```

## Model Components

### Base Operator (T_base)

Pretrained FNO loaded from HuggingFace. Remains frozen during training.

```python
from the_well.benchmark.models import FNO
model = FNO.from_pretrained("polymathic-ai/FNO-active_matter")
```

### Refinement Operator (Φ_θ)

U-Net architecture with:
- Circular padding for periodic boundary conditions
- GELU activation and BatchNorm

## Training Objective

The total loss combines multi-step supervision with spectral and fixed-point regularization:

```
L_total = L_spatial + β_spectral · L_spectral + β_fp · L_fp
```

Where:
- **L_spatial** = (1/K) Σ_k ||h_k - y||² — Multi-step supervision
- **L_spectral** — Progressive spectral loss with λ_k increasing from λ_start to λ_end
- **L_fp** = ||Φ(x, y)||² — Fixed-point regularization

