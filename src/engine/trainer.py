import copy
import math
import os
import random
import time
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import torch

from src.data.collator import batch_to_device
from src.models.vlm import MiniVLM
from src.utils.logger import MetricsLogger

DEFAULTS = dict(
    epochs=3,
    batch_size=16,
    grad_accum=1,
    lr_projector=1e-3,
    lr_lora=2e-4,
    lr_vit=1e-5,
    weight_decay=0.0,
    warmup_steps=None,        # default: warmup_ratio * total steps
    warmup_ratio=0.05,
    min_lr_ratio=0.1,         # cosine decays to this fraction of the peak lr
    grad_clip=1.0,
    log_every=10,
    eval_every=100,
    shuffle=True,
    seed=0,
)


def num_target_tokens(batch: dict) -> int:
    """Tokens that contribute to the loss (labels are shifted by one inside the model)."""
    return int((batch["labels"][:, 1:] != -100).sum())


def split_by_image(dataset, val_images: int, seed: int = 0):
    """Hold out `val_images` whole images (all their records) for a validation loss; returns (train, val)."""
    ids = sorted({r["image_id"] for r in dataset.records})
    random.Random(seed).shuffle(ids)
    held_out = set(ids[:val_images])
    train, val = copy.copy(dataset), copy.copy(dataset)
    train.records = [r for r in dataset.records if r["image_id"] not in held_out]
    val.records = [r for r in dataset.records if r["image_id"] in held_out]
    return train, val


def trainable_state(vlm: MiniVLM) -> Dict[str, torch.Tensor]:
    return {n: p.detach().cpu().clone() for n, p in vlm.named_parameters() if p.requires_grad}


def load_trainable_weights(vlm: MiniVLM, path: str, prefixes: Optional[Sequence[str]] = None) -> List[str]:
    """
    Copy weights from a checkpoint into `vlm` by parameter name. With `prefixes`, only names starting
    with one of them are loaded (e.g. ("projector.",) to start stage 2 from the stage 1 projector).
    Raises if a saved name does not exist in the model. Returns the loaded names.
    """
    saved = torch.load(path, map_location="cpu", weights_only=False)["trainable"]
    params = dict(vlm.named_parameters())
    loaded = []
    for name, tensor in saved.items():
        if prefixes and not name.startswith(tuple(prefixes)):
            continue
        if name not in params:
            raise KeyError(f"Checkpoint parameter '{name}' not found in the model")
        params[name].data.copy_(tensor.to(params[name].device, params[name].dtype))
        loaded.append(name)
    return loaded


class Trainer:
    """
    Stage-aware training loop for MiniVLM. It trains whatever is currently trainable
    (`vlm.set_stage(...)` decides), with one parameter group per component.

    - gradient accumulation weights micro-batches by their number of target tokens, so
      accumulating N batches equals one big batch exactly
    - cosine lr schedule with linear warmup
    - epoch order is a function of (seed, epoch), so a run can be resumed mid-epoch
    - checkpoints hold only the trainable parameters, plus optimizer state and progress
    """
    def __init__(
        self,
        vlm: MiniVLM,
        train_ds,
        val_ds,
        collate_fn: Callable,
        device,
        out_dir: str,
        training: Optional[dict] = None,
        config: Optional[dict] = None,
    ):
        self.vlm, self.train_ds, self.val_ds, self.collate, self.device = vlm, train_ds, val_ds, collate_fn, device
        self.cfg = {**DEFAULTS, **(training or {})}
        self.config = config or {}
        self.out_dir = out_dir
        os.makedirs(out_dir, exist_ok=True)
        self.logger = MetricsLogger(out_dir)

        self.batch_size, self.grad_accum = self.cfg["batch_size"], self.cfg["grad_accum"]
        self.steps_per_epoch = (len(train_ds) // self.batch_size) // self.grad_accum
        assert self.steps_per_epoch > 0, "Dataset is smaller than one optimizer step"
        self.total_steps = self.steps_per_epoch * self.cfg["epochs"]
        self.warmup_steps = (
            self.cfg["warmup_steps"] if self.cfg["warmup_steps"] is not None
            else int(self.cfg["warmup_ratio"] * self.total_steps)
        )

        self.optimizer = self._build_optimizer()
        self.step = 0
        self.best_val = float("inf")

    # ------------------------------------------------------------------ setup

    def _build_optimizer(self) -> torch.optim.Optimizer:
        groups: Dict[str, list] = {"projector": [], "lora": [], "vit": []}
        for name, p in self.vlm.named_parameters():
            if not p.requires_grad:
                continue
            if name.startswith("projector."):
                groups["projector"].append(p)
            elif "lora_" in name:
                groups["lora"].append(p)
            elif name.startswith("vit."):
                groups["vit"].append(p)
            else:
                raise ValueError(f"Unexpected trainable parameter '{name}'")
        lrs = {"projector": self.cfg["lr_projector"], "lora": self.cfg["lr_lora"], "vit": self.cfg["lr_vit"]}
        param_groups = [
            {"params": ps, "lr": lrs[k], "base_lr": lrs[k], "name": k} for k, ps in groups.items() if ps
        ]
        assert param_groups, "Nothing is trainable; call vlm.set_stage(...) first"
        return torch.optim.AdamW(param_groups, weight_decay=self.cfg["weight_decay"])

    def lr_factor(self, step: int) -> float:
        """Multiplier on each group's peak lr at (0-based) optimizer step `step`."""
        if step < self.warmup_steps:
            return (step + 1) / self.warmup_steps
        progress = (step - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
        floor = self.cfg["min_lr_ratio"]
        return floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))

    def epoch_batches(self, epoch: int) -> List[List[int]]:
        """Index lists for one epoch; a function of (seed, epoch) only. Incomplete groups are dropped."""
        n = len(self.train_ds)
        if self.cfg["shuffle"]:
            g = torch.Generator().manual_seed(self.cfg["seed"] + epoch)
            order = torch.randperm(n, generator=g).tolist()
        else:
            order = list(range(n))
        count = self.steps_per_epoch * self.grad_accum
        return [order[i * self.batch_size:(i + 1) * self.batch_size] for i in range(count)]

    # --------------------------------------------------------------- training

    def _load_batch(self, ds, indices: Sequence[int]) -> dict:
        batch = batch_to_device(self.collate([ds[i] for i in indices]), self.device)
        batch.pop("meta")
        return batch

    def _train_step(self, group: List[List[int]]) -> Tuple[float, float, int]:
        t0 = time.time()
        batches = [self._load_batch(self.train_ds, idx) for idx in group]
        counts = [num_target_tokens(b) for b in batches]
        total = sum(counts)
        if total == 0:
            raise RuntimeError("An optimizer step has no target tokens")

        self.optimizer.zero_grad(set_to_none=True)
        loss_value = 0.0
        for batch, c in zip(batches, counts):
            if c == 0:                                   # nothing to learn from (loss would be NaN)
                continue
            loss = self.vlm(**batch).loss * (c / total)  # token-weighted so accumulation == one big batch
            loss.backward()
            loss_value += loss.item()

        params = [p for g in self.optimizer.param_groups for p in g["params"]]
        grad_norm = float(torch.nn.utils.clip_grad_norm_(params, self.cfg["grad_clip"]))
        factor = self.lr_factor(self.step)
        for g in self.optimizer.param_groups:
            g["lr"] = g["base_lr"] * factor
        self.optimizer.step()
        self.step += 1
        return loss_value, grad_norm, time.time() - t0

    @torch.no_grad()
    def evaluate_loss(self) -> float:
        """Token-weighted mean loss over the validation set."""
        if self.val_ds is None or len(self.val_ds) == 0:
            return float("nan")
        self.vlm.eval()
        total_loss, total_tokens = 0.0, 0
        for start in range(0, len(self.val_ds), self.batch_size):
            batch = self._load_batch(self.val_ds, range(start, min(start + self.batch_size, len(self.val_ds))))
            c = num_target_tokens(batch)
            if c:
                total_loss += self.vlm(**batch).loss.item() * c
                total_tokens += c
        self.vlm.train()
        return total_loss / max(1, total_tokens)

    def train(self, until_step: Optional[int] = None) -> List[dict]:
        """Train until `until_step` optimizer steps in total (default: the whole schedule)."""
        until = min(until_step or self.total_steps, self.total_steps)
        self.vlm.train()
        window = {"loss": [], "grad": [], "time": []}
        while self.step < until:
            epoch, offset = divmod(self.step, self.steps_per_epoch)
            batches = self.epoch_batches(epoch)
            for s in range(offset, self.steps_per_epoch):
                if self.step >= until:
                    break
                loss, grad_norm, dt = self._train_step(batches[s * self.grad_accum:(s + 1) * self.grad_accum])
                window["loss"].append(loss); window["grad"].append(grad_norm); window["time"].append(dt)

                if self.step % self.cfg["log_every"] == 0:
                    n = len(window["loss"])
                    self.logger.log(
                        self.step,
                        train_loss=sum(window["loss"]) / n,
                        lr=self.optimizer.param_groups[0]["lr"],
                        grad_norm=sum(window["grad"]) / n,
                        sec_per_step=sum(window["time"]) / n,
                        epoch=round(self.step / self.steps_per_epoch, 2),
                    )
                    window = {"loss": [], "grad": [], "time": []}
                if self.step % self.cfg["eval_every"] == 0 or self.step == until:
                    self._validate_and_checkpoint()

        self.logger.save()
        self.logger.plot(self.config.get("run_name", ""))
        return self.logger.history

    def _validate_and_checkpoint(self):
        val = self.evaluate_loss()
        if not math.isnan(val):
            self.logger.log(self.step, val_loss=val)
            if val < self.best_val:
                self.best_val = val
                self.save_checkpoint(os.path.join(self.out_dir, "best.pt"))
        self.save_checkpoint(os.path.join(self.out_dir, "last.pt"))
        self.logger.save()

    # ------------------------------------------------------------ checkpoints

    def save_checkpoint(self, path: str):
        torch.save({
            "trainable": trainable_state(self.vlm),
            "optimizer": self.optimizer.state_dict(),
            "step": self.step,
            "best_val": self.best_val,
            "history": self.logger.history,
            "config": self.config,
        }, path)

    def load_checkpoint(self, path: str):
        """Restore trainable weights, optimizer state and progress, so train() continues the same run."""
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        params = dict(self.vlm.named_parameters())
        missing = set(ckpt["trainable"]) ^ {n for n, p in params.items() if p.requires_grad}
        if missing:
            raise ValueError(f"Checkpoint and model trainable parameters differ: {sorted(missing)[:3]} ...")
        for name, tensor in ckpt["trainable"].items():
            params[name].data.copy_(tensor.to(params[name].device, params[name].dtype))
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.step, self.best_val = ckpt["step"], ckpt["best_val"]
        self.logger = MetricsLogger(self.out_dir, ckpt["history"])
