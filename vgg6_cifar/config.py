
from __future__ import annotations
import argparse
from dataclasses import dataclass

@dataclass
class TrainConfig:
    data_dir: str
    out_dir: str
    epochs: int = 60
    batch_size: int = 128
    lr: float = 0.1
    optimizer: str = "sgd"
    momentum: float = 0.9
    weight_decay: float = 5e-4
    label_smoothing: float = 0.0
    amp: bool = True
    seed: int = 42
    num_workers: int = 4
    aug_hflip: bool = True
    aug_crop: bool = True
    aug_cutout: bool = True
    aug_jitter: bool = True

def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="VGG6 Baseline on CIFAR-10")
    p.add_argument("--data_dir", type=str, default="./data")
    p.add_argument("--out_dir", type=str, default="./runs/baseline")
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--optimizer", type=str, default="sgd", choices=["sgd", "adamw"])
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--weight_decay", type=float, default=5e-4)
    p.add_argument("--label_smoothing", type=float, default=0.0)
    p.add_argument("--amp", action="store_true")
    p.add_argument("--no_amp", dest="amp", action="store_false"); p.set_defaults(amp=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--aug_hflip", action="store_true")
    p.add_argument("--aug_crop", action="store_true")
    p.add_argument("--aug_cutout", action="store_true")
    p.add_argument("--aug_jitter", action="store_true")
    return p
