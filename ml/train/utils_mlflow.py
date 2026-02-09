# ml/train/utils_mlflow.py

from __future__ import annotations
from pathlib import Path
import matplotlib.pyplot as plt

def save_plot(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()
