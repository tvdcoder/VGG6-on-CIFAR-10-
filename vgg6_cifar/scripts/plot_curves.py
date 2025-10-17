
#!/usr/bin/env python3
import argparse, os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics_csv", type=str, required=True)
    ap.add_argument("--out_dir", type=str, required=True)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    df = pd.read_csv(args.metrics_csv)

    plt.figure(); plt.plot(df["epoch"], df["train_loss"], label="train_loss"); plt.plot(df["epoch"], df["val_loss"], label="val_loss")
    plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.title("Loss Curves"); plt.legend()
    plt.savefig(os.path.join(args.out_dir, "loss_curves.png"), bbox_inches="tight"); plt.close()

    plt.figure(); plt.plot(df["epoch"], df["train_acc"], label="train_acc"); plt.plot(df["epoch"], df["val_acc"], label="val_acc")
    plt.xlabel("Epoch"); plt.ylabel("Accuracy (top-1)"); plt.title("Accuracy Curves"); plt.legend()
    plt.savefig(os.path.join(args.out_dir, "accuracy_curves.png"), bbox_inches="tight"); plt.close()

    print("Saved plots to:", args.out_dir)

if __name__ == "__main__":
    main()
