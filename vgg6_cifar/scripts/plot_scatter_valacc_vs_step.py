
#!/usr/bin/env python3
import argparse, os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def main():
    ap = argparse.ArgumentParser("Scatter: validation accuracy vs step")
    ap.add_argument("--metrics_csv", type=str, required=True)
    ap.add_argument("--out_png", type=str, default=None)
    args = ap.parse_args()
    if args.out_png is None:
        args.out_png = os.path.join(os.path.dirname(args.metrics_csv), "scatter_valacc_vs_step.png")
    df = pd.read_csv(args.metrics_csv)
    steps = df["epoch"]; vals = df["val_acc"]
    plt.figure(); plt.scatter(steps, vals, s=16)
    plt.xlabel("Step (epoch)"); plt.ylabel("Validation Accuracy (top-1)")
    plt.title("Validation Accuracy vs Step"); plt.grid(True, alpha=0.3)
    plt.savefig(args.out_png, bbox_inches="tight"); plt.close()
    print("Saved:", args.out_png)

if __name__ == "__main__":
    main()
