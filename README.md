
# VGG6 on CIFAR-10 — Full Assignment (Q1–Q5)

This repository is structured to satisfy the entire assignment:
- **Q1**: Baseline training with proper normalization & augmentations; final test accuracy; loss/accuracy curves.
- **Q2**: Experiments varying **activations**, **optimizers**, **batch size / epochs / LR**.
- **Q3**: Plots — W&B parallel-coordinates, validation-accuracy vs step (scatter), and training/validation curves.
- **Q4**: Final best configuration (reproducible).
- **Q5**: Clean, modular code + README with exact commands; include seed; upload best checkpoint to GitHub.

## Environment
```bash
pip install -r requirements.txt
# Optional: W&B login for Q3a
python - <<'PY'
import os, wandb
# os.environ['WANDB_API_KEY'] = '<your-key>'  # or use wandb.login() interactively
PY
```

## Structure
```
vgg6_cifar_full/
├─ vgg6_cifar/
│  ├─ config.py
│  ├─ models/
│  │  ├─ vgg6.py                # VGG6 with configurable activation
│  │  └─ activations.py         # ReLU/Sigmoid/Tanh/SiLU/GELU registry
│  ├─ data/cifar10.py           # Normalization, augmentations, dataloaders
│  ├─ engine/
│  │  ├─ trainer.py             # train/eval loops, cosine LR with warmup
│  │  └─ optim_factory.py       # SGD, Nesterov-SGD, Adam/AdamW, Adagrad, RMSprop, Nadam
│  └─ utils/{logger.py,metrics.py}
├─ vgg6_cifar/scripts/
│  ├─ train_baseline.py         # Q1 entry: baseline
│  ├─ train_experiment.py       # Q2/Q3/Q4: configurable runner (+W&B)
│  ├─ sweep_grid.py             # Q2 sweeps over hparams
│  ├─ plot_curves.py            # Q1/Q3(c) curves
│  └─ plot_scatter_valacc_vs_step.py  # Q3(b) scatter
└─ train_vgg6_cifar10_baseline.py     # Optional single-file baseline
```

---

## Q1 — Baseline (a–c)

**Run:**
```bash
python vgg6_cifar/scripts/train_baseline.py --data_dir ./data --out_dir ./runs/baseline   --epochs 60 --batch_size 128 --lr 0.1 --optimizer sgd --momentum 0.9   --weight_decay 5e-4 --label_smoothing 0.0   --aug_hflip --aug_crop --aug_cutout --aug_jitter --amp --seed 42
```

**Artifacts in `--out_dir`:**
- `metrics.csv`, `best.pt`, `final_test_metrics.json`
- Curves:
```bash
python vgg6_cifar/scripts/plot_curves.py --metrics_csv ./runs/baseline/metrics.csv --out_dir ./runs/baseline
```

---

## Q2 — Model Performance on Different Configurations

### (a) Vary activation
```bash
python vgg6_cifar/scripts/train_experiment.py --activation gelu --optimizer sgd --lr 0.1 --batch_size 128   --epochs 40 --data_dir ./data --out_dir ./runs/exp_act_gelu   --aug_hflip --aug_crop --aug_cutout --aug_jitter --amp --wandb --seed 42
```

### (b) Vary optimizer
```bash
python vgg6_cifar/scripts/train_experiment.py --optimizer nadam --activation relu --lr 0.01 --batch_size 128   --epochs 40 --data_dir ./data --out_dir ./runs/exp_opt_nadam   --aug_hflip --aug_crop --aug_cutout --aug_jitter --amp --wandb --seed 42
```

### (c) Vary batch size, epochs, learning rate
```bash
python vgg6_cifar/scripts/train_experiment.py --activation relu --optimizer sgd --lr 0.05 --batch_size 64   --epochs 80 --data_dir ./data --out_dir ./runs/exp_bs64_lr005_e80   --aug_hflip --aug_crop --aug_cutout --aug_jitter --amp --wandb --seed 42
```

**Grid sweep (recommended):**
```bash
python vgg6_cifar/scripts/sweep_grid.py --data_dir ./data --base_out ./runs/sweeps   --epochs 30 --batch_sizes 64,128 --lrs 0.1,0.05,0.01   --optimizers sgd,nesterov-sgd,adam,adamw,rmsprop,nadam,adagrad   --activations relu,silu,gelu,tanh,sigmoid --amp --wandb
```
Outputs `./runs/sweeps/sweep_summary.csv` with `best_val_acc` and `test_top1_acc` for each config.

---

## Q3 — Plots

### (a) W&B parallel-coordinates
Enable `--wandb`, then in the W&B UI choose **Parallel Coordinates** and select axes: `activation`, `optimizer`, `batch_size`, `lr`, `best_val_acc`. Screenshot for the PDF.

### (b) Validation accuracy vs. step (scatter)
```bash
python vgg6_cifar/scripts/plot_scatter_valacc_vs_step.py   --metrics_csv ./runs/exp_act_gelu/metrics.csv   --out_png     ./runs/exp_act_gelu/scatter_valacc_vs_step.png
```

### (c) Train/val curves
```bash
python vgg6_cifar/scripts/plot_curves.py   --metrics_csv ./runs/exp_act_gelu/metrics.csv   --out_dir     ./runs/exp_act_gelu
```

---

## Q4 — Final Model Performance
Select the **single best** configuration (highest validation accuracy) from the W&B parallel plot. Re-run it exactly:

```bash
python vgg6_cifar/scripts/train_experiment.py   --data_dir ./data --out_dir ./runs/final_best   --activation <best_act> --optimizer <best_opt> --lr <best_lr>   --batch_size <best_bs> --epochs <best_epochs>   --aug_hflip --aug_crop --aug_cutout --aug_jitter --amp --wandb --seed 42
```
Report the resulting `best_val_acc` and `test_top1_acc` and upload `best.pt` to your GitHub repo.

---

## Q5 — Reproducibility & Repository
- **Modular code** across `models/`, `data/`, `engine/`, `utils/`, `scripts/`.
- **Exact commands** in this README; dependencies in `requirements.txt`; seed via `--seed` (42 default).
- **Upload trained model**: include `best.pt` for your final configuration in the repo or release assets.
- **Colab**: set GPU runtime and run commands with paths under `/content`.

---

## Colab Quick Start
```bash
# After uploading/unzipping this folder into /content/vgg6_cifar_full
%cd /content/vgg6_cifar_full
!pip install -r requirements.txt

# Baseline
!python -m vgg6_cifar.scripts.train_baseline --data_dir /content/data --out_dir /content/runs/baseline   --epochs 60 --batch_size 128 --lr 0.1 --optimizer sgd --momentum 0.9   --weight_decay 5e-4 --label_smoothing 0.0   --aug_hflip --aug_crop --aug_cutout --aug_jitter --amp --seed 42
```

**If you see `ModuleNotFoundError: vgg6_cifar`**, run from the project root with `-m` (as above), or add the root to `sys.path`.
