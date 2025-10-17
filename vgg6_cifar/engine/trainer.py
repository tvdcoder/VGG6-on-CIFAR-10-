
from __future__ import annotations
import math
import torch
import torch.nn as nn
from ..utils.metrics import accuracy_top1

def train_one_epoch(model, loader, optimizer, device, scaler, criterion, epoch, writer=None):
    model.train(); rl=ra=n=0.0
    for i,(x,y) in enumerate(loader):
        x=x.to(device,non_blocking=True); y=y.to(device,non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        if scaler is not None:
            with torch.cuda.amp.autocast():
                logits=model(x); loss=criterion(logits,y)
            scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
        else:
            logits=model(x); loss=criterion(logits,y); loss.backward(); optimizer.step()
        b=y.size(0); rl+=loss.item()*b; ra+=accuracy_top1(logits,y)*b; n+=b
        if writer and (i%50==0): writer.add_scalar("train/step_loss", loss.item(), epoch*len(loader)+i)
    return rl/n, ra/n

@torch.no_grad()
def evaluate(model, loader, device, criterion):
    model.eval(); rl=ra=n=0.0
    for x,y in loader:
        x=x.to(device,non_blocking=True); y=y.to(device,non_blocking=True)
        logits=model(x); loss=criterion(logits,y)
        b=y.size(0); rl+=loss.item()*b; ra+=accuracy_top1(logits,y)*b; n+=b
    return rl/n, ra/n

def make_optimizer(params, optimizer_name: str, lr: float, momentum: float, weight_decay: float):
    if optimizer_name.lower() == "sgd":
        return torch.optim.SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay, nesterov=True)
    elif optimizer_name.lower() == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")

def make_scheduler(optimizer, epochs: int, steps_per_epoch: int):
    warmup_epochs = min(5, max(1, epochs // 10))
    def lr_lambda(step):
        e = step / max(1, steps_per_epoch)
        if e < warmup_epochs: return (e+1) / warmup_epochs
        prog = (e - warmup_epochs) / max(1e-8, (epochs - warmup_epochs))
        return 0.5 * (1 + math.cos(math.pi * min(1.0, prog)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
