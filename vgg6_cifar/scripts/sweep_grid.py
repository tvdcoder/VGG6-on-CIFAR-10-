
#!/usr/bin/env python3
from __future__ import annotations
import os, itertools, subprocess, argparse, json, csv

def build_parser():
    p = argparse.ArgumentParser("Run grid sweeps for Q2 configs")
    p.add_argument("--data_dir", type=str, default="./data")
    p.add_argument("--base_out", type=str, default="./runs/sweeps")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch_sizes", type=str, default="64,128")
    p.add_argument("--lrs", type=str, default="0.1,0.05,0.01")
    p.add_argument("--optimizers", type=str, default="sgd,nesterov-sgd,adam,adamw,rmsprop,nadam,adagrad")
    p.add_argument("--activations", type=str, default="relu,silu,gelu,tanh,sigmoid")
    p.add_argument("--weight_decay", type=float, default=5e-4)
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--label_smoothing", type=float, default=0.0)
    p.add_argument("--amp", action="store_true")
    p.add_argument("--wandb", action="store_true")
    p.add_argument("--wandb_project", type=str, default="vgg6-cifar10-assignment")
    p.add_argument("--wandb_group", type=str, default="q2-sweeps")
    p.add_argument("--seed", type=int, default=42)
    return p

def main():
    args = build_parser().parse_args()
    os.makedirs(args.base_out, exist_ok=True)
    batch_sizes = [int(x) for x in args.batch_sizes.split(",") if x]
    lrs = [float(x) for x in args.lrs.split(",") if x]
    optimizers = [x.strip() for x in args.optimizers.split(",") if x]
    activations = [x.strip() for x in args.activations.split(",") if x]
    grid = list(itertools.product(batch_sizes, lrs, optimizers, activations))

    rows = []
    for i, (bs, lr, opt, act) in enumerate(grid, 1):
        out_dir = os.path.join(args.base_out, f"bs{bs}_lr{lr}_{opt}_{act}")
        cmd = [
            "python", "vgg6_cifar/scripts/train_experiment.py",
            "--data_dir", args.data_dir, "--out_dir", out_dir,
            "--epochs", str(args.epochs), "--batch_size", str(bs),
            "--optimizer", opt, "--lr", str(lr), "--activation", act,
            "--weight_decay", str(args.weight_decay), "--momentum", str(args.momentum),
            "--label_smoothing", str(args.label_smoothing), "--seed", str(args.seed),
            "--aug_hflip", "--aug_crop", "--aug_cutout", "--aug_jitter",
        ]
        if args.amp: cmd.append("--amp")
        if args.wandb:
            cmd += ["--wandb", "--wandb_project", args.wandb_project, "--wandb_group", args.wandb_group, "--run_name", f"i{i:03d}"]
        print(">>>", " ".join(cmd))
        subprocess.run(cmd, check=True)

        mpath = os.path.join(out_dir, "final_test_metrics.json")
        if os.path.exists(mpath):
            with open(mpath) as f: m = json.load(f)
            rows.append({"out_dir": out_dir, "batch_size": bs, "lr": lr, "optimizer": opt, "activation": act,
                         "best_val_acc": m.get("best_val_acc"), "test_top1_acc": m.get("test_top1_acc")})

    sum_path = os.path.join(args.base_out, "sweep_summary.csv")
    with open(sum_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["out_dir","batch_size","lr","optimizer","activation","best_val_acc","test_top1_acc"])
        w.writeheader(); [w.writerow(r) for r in rows]
    print("Wrote sweep summary:", sum_path)

if __name__ == "__main__":
    main()
