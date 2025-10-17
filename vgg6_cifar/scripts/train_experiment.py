
#!/usr/bin/env python3
from __future__ import annotations
import os, json, argparse, random
import torch, torch.nn as nn

from vgg6_cifar.models.vgg6 import VGG6
from vgg6_cifar.data.cifar10 import build_transforms, make_dataloaders, CIFAR10_MEAN, CIFAR10_STD
from vgg6_cifar.utils.logger import CSVLogger, make_tb_writer
from vgg6_cifar.engine.trainer import train_one_epoch, evaluate, make_scheduler
from vgg6_cifar.engine.optim_factory import make_optimizer

def set_seed(seed: int):
    random.seed(seed); os.environ["PYTHONHASHSEED"]=str(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic=True; torch.backends.cudnn.benchmark=False

def get_device(): return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def build_parser():
    p = argparse.ArgumentParser("VGG6 CIFAR10 Experiment Runner (Q2/Q3/Q4)")
    p.add_argument("--data_dir", type=str, default="./data")
    p.add_argument("--out_dir", type=str, default="./runs/exp")
    p.add_argument("--activation", type=str, default="relu", choices=["relu","sigmoid","tanh","silu","gelu"])
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--optimizer", type=str, default="sgd", choices=["sgd","nesterov-sgd","adam","adamw","adagrad","rmsprop","nadam"])
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--weight_decay", type=float, default=5e-4)
    p.add_argument("--label_smoothing", type=float, default=0.0)
    p.add_argument("--amp", action="store_true"); p.add_argument("--no_amp", dest="amp", action="store_false"); p.set_defaults(amp=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--aug_hflip", action="store_true")
    p.add_argument("--aug_crop", action="store_true")
    p.add_argument("--aug_cutout", action="store_true")
    p.add_argument("--aug_jitter", action="store_true")
    p.add_argument("--wandb", action="store_true")
    p.add_argument("--wandb_project", type=str, default="vgg6-cifar10-assignment")
    p.add_argument("--wandb_group", type=str, default="q2-sweeps")
    p.add_argument("--run_name", type=str, default=None)
    return p

def main():
    args = build_parser().parse_args()
    set_seed(args.seed); device = get_device()
    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "config.json"), "w") as f: json.dump(vars(args), f, indent=2)

    wb = None
    if args.wandb:
        import wandb
        wb = wandb.init(project=args.wandb_project, group=args.wandb_group, name=args.run_name, config=vars(args))

    writer = make_tb_writer(args.out_dir)
    csv_path = os.path.join(args.out_dir, "metrics.csv")
    csv_logger = CSVLogger(csv_path, ["epoch","lr","train_loss","train_acc","val_loss","val_acc"])

    train_tfms, val_tfms, test_tfms, aug_used = build_transforms(args.aug_hflip, args.aug_crop, args.aug_cutout, args.aug_jitter)
    print(f"CIFAR-10 normalization: mean={CIFAR10_MEAN}, std={CIFAR10_STD}")
    print("Augmentations used for TRAIN:", ", ".join(aug_used) if aug_used else "None")
    print("VAL/TEST transforms: ToTensor + Normalize")
    train_loader, val_loader, test_loader = make_dataloaders(args.data_dir, args.batch_size, args.num_workers, args.seed, train_tfms, val_tfms, test_tfms)

    model = VGG6(num_classes=10, dropout=args.dropout, act_name=args.activation).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing).to(device)
    optimizer = make_optimizer(model.parameters(), args.optimizer, args.lr, args.weight_decay, momentum=args.momentum)
    scheduler = make_scheduler(optimizer, args.epochs, steps_per_epoch=len(train_loader))
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    best_val = 0.0; best_path = os.path.join(args.out_dir, "best.pt")
    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, device, scaler, criterion, epoch, writer)
        va_loss, va_acc = evaluate(model, val_loader, device, criterion)
        scheduler.step()

        lr_now = optimizer.param_groups[0]["lr"]
        row = {"epoch": epoch, "lr": lr_now, "train_loss": tr_loss, "train_acc": tr_acc, "val_loss": va_loss, "val_acc": va_acc}
        csv_logger.log(row)
        if wb: wb.log(row)

        print(f"[Epoch {epoch:03d}] lr={lr_now:.5f} train_loss={tr_loss:.4f} acc={tr_acc:.4f} val_loss={va_loss:.4f} acc={va_acc:.4f}")
        if va_acc > best_val:
            best_val = va_acc
            torch.save({"model": model.state_dict(), "epoch": epoch, "val_acc": best_val}, best_path)

    ckpt = torch.load(best_path, map_location=device); model.load_state_dict(ckpt["model"])
    te_loss, te_acc = evaluate(model, test_loader, device, criterion)
    print(f"FINAL TEST: loss={te_loss:.4f}  top1_acc={te_acc:.4f}")
    with open(os.path.join(args.out_dir, "final_test_metrics.json"), "w") as f:
        json.dump({"test_loss": te_loss, "test_top1_acc": te_acc, "best_val_acc": best_val}, f, indent=2)
    if wb:
        wb.summary["final_test_top1_acc"] = te_acc
        wb.summary["best_val_acc"] = best_val
        wb.finish()
    writer.close()

if __name__ == "__main__":
    main()
