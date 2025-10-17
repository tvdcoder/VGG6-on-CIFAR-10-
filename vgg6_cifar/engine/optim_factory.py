
from __future__ import annotations
import torch

def make_optimizer(params, name: str, lr: float, weight_decay: float, momentum: float = 0.9):
    n = name.lower()
    if n == "sgd":
        return torch.optim.SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay, nesterov=False)
    if n == "nesterov-sgd":
        return torch.optim.SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay, nesterov=True)
    if n == "adam":
        return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    if n == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    if n == "adagrad":
        return torch.optim.Adagrad(params, lr=lr, weight_decay=weight_decay)
    if n == "rmsprop":
        return torch.optim.RMSprop(params, lr=lr, momentum=momentum, weight_decay=weight_decay, centered=False)
    if n == "nadam":
        return torch.optim.NAdam(params, lr=lr, weight_decay=weight_decay)
    raise ValueError(f"Unsupported optimizer: {name}")
