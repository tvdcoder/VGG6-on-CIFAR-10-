
from __future__ import annotations
import torch

def accuracy_top1(logits: torch.Tensor, targets: torch.Tensor) -> float:
    _, pred = logits.max(1)
    return pred.eq(targets).sum().item() / targets.size(0)
