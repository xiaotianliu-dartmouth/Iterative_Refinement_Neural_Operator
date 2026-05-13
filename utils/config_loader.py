"""
Configuration loader for YAML config files.
"""

import yaml
import torch
from pathlib import Path


class Config:
    """Load and access configuration from YAML file."""

    def __init__(self, config_path="config.yaml"):
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(config_path, 'r') as f:
            self._config = yaml.safe_load(f)

    def __getattr__(self, name):
        if name.startswith('_'):
            return super().__getattribute__(name)
        return self._config.get(name)

    def get(self, key, default=None):
        return self._config.get(key, default)

    def get_device(self):
        """Returns device and number of GPUs."""
        if not torch.cuda.is_available():
            return torch.device("cpu"), 1

        n_gpu = torch.cuda.device_count()
        device = torch.device("cuda:0")

        if self.gpu.get('use_multi_gpu', False) and n_gpu > 1:
            return device, n_gpu
        return device, 1


def load_config(config_path="config.yaml"):
    """Load configuration from YAML file."""
    return Config(config_path)
