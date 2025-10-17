
#!/usr/bin/env python3
from __future__ import annotations
import os, json
import torch, torch.nn as nn
from vgg6_cifar.config import build_argparser
from vgg6_cifar.models.vgg6 import VGG6
from vgg6_cifar.data.cifar10 import build_transforms, make_dataloaders, CIFAR10_MEAN, CIFAR10_STD
from vgg6_cifar.utils.logger import CSVLogger, make_tb_writer
from vgg6_cifar.engine.trainer import train_one_epoch, evaluate, make_scheduler

def set_seed(seed: int):
    import random, torch, os
    random.seed(seed); os.environ["PYTHONHASHSEED"]=str(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic=True; torch.backends.cudnn.benchmark=False

def get_device(): return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    parser = build_argparser(); args = parser.parse_args()
    set_seed(args.seed); device = get_device()

    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "config.json"), "w") as f: json.dump(vars(args), f, indent=2)

    writer = make_tb_writer(args.out_dir)
    csv_path = os.path.join(args.out_dir, "metrics.csv")
    csv_logger = CSVLogger(csv_path, ["epoch","lr","train_loss","train_acc","val_loss","val_acc"])

    # (a) Data
    train_tfms, val_tfms, test_tfms, aug_used = build_transforms(args.aug_hflip, args.aug_crop, args.aug_cutout, args.aug_jitter)
    print(f"CIFAR-10 normalization: mean={CIFAR10_MEAN}, std={CIFAR10_STD}")
    print("Augmentations used for TRAIN:", ", ".join(aug_used) if aug_used else "None")
    print("VAL/TEST transforms: ToTensor + Normalize")
    train_loader, val_loader, test_loader = make_dataloaders(args.data_dir, args.batch_size, args.num_workers, args.seed, train_tfms, val_tfms, test_tfms)

    # Model & training
    model = VGG6(num_classes=10, dropout=0.3, act_name="relu").to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=True)
    scheduler = make_scheduler(optimizer, args.epochs, steps_per_epoch=len(train_loader))
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    best_val = 0.0; best_path = os.path.join(args.out_dir, "best.pt")
    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, device, scaler, criterion, epoch, writer)
        va_loss, va_acc = evaluate(model, val_loader, device, criterion); scheduler.step()

        lr_now = optimizer.param_groups[0]["lr"]
        csv_logger.log({"epoch": epoch, "lr": lr_now, "train_loss": tr_loss, "train_acc": tr_acc, "val_loss": va_loss, "val_acc": va_acc})
        writer.add_scalar("lr", lr_now, epoch)
        writer.add_scalar("train/loss", tr_loss, epoch); writer.add_scalar("train/acc", tr_acc, epoch)
        writer.add_scalar("val/loss", va_loss, epoch); writer.add_scalar("val/acc", va_acc, epoch)

        print(f"[Epoch {epoch:03d}] lr={lr_now:.5f} train_loss={tr_loss:.4f} acc={tr_acc:.4f} val_loss={va_loss:.4f} acc={va_acc:.4f}")
        if va_acc > best_val:
            best_val = va_acc
            torch.save({"model": model.state_dict(), "epoch": epoch, "val_acc": best_val}, best_path)

    ckpt = torch.load(best_path, map_location=device); model.load_state_dict(ckpt["model"])
    te_loss, te_acc = evaluate(model, test_loader, device, criterion)
    print(f"FINAL TEST: loss={te_loss:.4f}  top1_acc={te_acc:.4f}")
    with open(os.path.join(args.out_dir, "final_test_metrics.json"), "w") as f:
        json.dump({"test_loss": te_loss, "test_top1_acc": te_acc}, f, indent=2)

    with open(os.path.join(args.out_dir, "README_BASELINE.txt"), "w") as f:
        f.write("Baseline VGG6 on CIFAR-10\n"
                "Transforms (train): " + (", ".join(aug_used) if aug_used else "None") + "\n"
                f"Normalize: mean={CIFAR10_MEAN}, std={CIFAR10_STD}\n"
                f"Final test top-1 accuracy: {te_acc:.4f}\n")

    writer.close()

if __name__ == "__main__":
    main()
