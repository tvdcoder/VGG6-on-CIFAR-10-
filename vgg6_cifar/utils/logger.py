
from __future__ import annotations
import os, csv
from typing import List, Dict
from torch.utils.tensorboard import SummaryWriter

class CSVLogger:
    def __init__(self, filepath: str, fieldnames: List[str]):
        self.filepath = filepath; self.fieldnames = fieldnames
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(self.filepath, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=self.fieldnames).writeheader()
    def log(self, row: Dict):
        with open(self.filepath, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=self.fieldnames).writerow(row)

def make_tb_writer(log_dir: str) -> SummaryWriter:
    return SummaryWriter(log_dir=log_dir)
