
from __future__ import annotations
import math, random
from typing import List
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as T

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD  = (0.2470, 0.2435, 0.2616)

class RandomErasingSquare(nn.Module):
    def __init__(self, p: float = 0.5, area_ratio: float = 0.02, value: float = 0.0):
        super().__init__(); self.p=p; self.area_ratio=area_ratio; self.value=value
    def forward(self, img: torch.Tensor) -> torch.Tensor:
        if random.random() > self.p: return img
        c,h,w = img.shape; area=h*w; erase_area=max(1,int(self.area_ratio*area))
        side=max(1,min(int(math.sqrt(erase_area)),h,w))
        y=random.randint(0,h-side); x=random.randint(0,w-side)
        img[:, y:y+side, x:x+side] = self.value
        return img

def build_transforms(aug_hflip: bool, aug_crop: bool, aug_cutout: bool, aug_jitter: bool):
    aug_used: List[str] = []; train_tfms: List[torch.nn.Module] = []
    if aug_crop: train_tfms.append(T.RandomCrop(32, padding=4)); aug_used.append("RandomCrop(32, padding=4)")
    if aug_hflip: train_tfms.append(T.RandomHorizontalFlip()); aug_used.append("RandomHorizontalFlip(p=0.5)")
    if aug_jitter: train_tfms.append(T.ColorJitter(0.2,0.2,0.2,0.02)); aug_used.append("ColorJitter(0.2/0.2/0.2/0.02)")
    train_tfms.extend([T.ToTensor(), T.Normalize(CIFAR10_MEAN, CIFAR10_STD)])
    if aug_cutout: train_tfms.append(RandomErasingSquare(0.5, 0.02)); aug_used.append("RandomErasingSquare(~2%)")
    val_tfms = T.Compose([T.ToTensor(), T.Normalize(CIFAR10_MEAN, CIFAR10_STD)])
    return T.Compose(train_tfms), val_tfms, val_tfms, aug_used

def make_dataloaders(data_dir: str, batch_size: int, num_workers: int, seed: int,
                     train_tfms, val_tfms, test_tfms):
    train_set = torchvision.datasets.CIFAR10(root=data_dir, train=True, download=True, transform=train_tfms)
    train_len = int(0.9 * len(train_set)); val_len = len(train_set) - train_len
    g = torch.Generator().manual_seed(seed)
    train_subset, val_subset = torch.utils.data.random_split(train_set, [train_len, val_len], generator=g)
    test_set = torchvision.datasets.CIFAR10(root=data_dir, train=False, download=True, transform=test_tfms)
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True,  num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(val_subset,   batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader  = DataLoader(test_set,     batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader, test_loader
