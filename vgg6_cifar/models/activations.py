
from __future__ import annotations
import torch.nn as nn

ACTIVATIONS = {
    "relu":    lambda: nn.ReLU(inplace=True),
    "sigmoid": lambda: nn.Sigmoid(),
    "tanh":    lambda: nn.Tanh(),
    "silu":    lambda: nn.SiLU(inplace=True),
    "gelu":    lambda: nn.GELU(),
}

def get_activation(name: str) -> nn.Module:
    key = name.lower()
    if key not in ACTIVATIONS:
        raise ValueError(f"Unknown activation: {name}. Choices: {list(ACTIVATIONS.keys())}")
    return ACTIVATIONS[key]()
