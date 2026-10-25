import json
import os
from typing import List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


class MetricsLogger:
    """
    Keeps a list of {"step": ..., metric: value, ...} records, prints them, and writes
    history.json plus a loss-curve plot into `out_dir`.
    """
    def __init__(self, out_dir: str, history: List[dict] = None):
        self.out_dir = out_dir
        self.history: List[dict] = list(history or [])
        os.makedirs(out_dir, exist_ok=True)

    def log(self, step: int, **metrics):
        self.history.append({"step": step, **metrics})
        parts = [f"{k} {v:.4f}" if isinstance(v, float) else f"{k} {v}" for k, v in metrics.items()]
        print(f"step {step:5d} | " + " | ".join(parts), flush=True)

    def save(self):
        with open(os.path.join(self.out_dir, "history.json"), "w") as f:
            json.dump(self.history, f, indent=1)

    def plot(self, title: str = ""):
        train = [(r["step"], r["train_loss"]) for r in self.history if "train_loss" in r]
        val = [(r["step"], r["val_loss"]) for r in self.history if "val_loss" in r]
        if not train and not val:
            return
        fig, ax = plt.subplots(figsize=(6.5, 3.4))
        if train:
            ax.plot(*zip(*train), color="#2f6db5", lw=1.8, label="train")
        if val:
            ax.plot(*zip(*val), color="#d98324", lw=1.8, marker="o", ms=4, label="validation")
        ax.set_xlabel("step")
        ax.set_ylabel("loss")
        ax.set_title(title, fontsize=10, loc="left")
        ax.grid(axis="y", color="#e3e5e8", lw=0.6)
        ax.legend(frameon=False, fontsize=8)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        plt.tight_layout()
        fig.savefig(os.path.join(self.out_dir, "loss_curve.png"), dpi=130)
        plt.close(fig)
