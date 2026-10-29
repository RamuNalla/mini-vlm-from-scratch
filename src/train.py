"""
Entry point: train one stage from a YAML config.

    python -m src.train --config configs/stage1_align.yaml
    python -m src.train --config configs/stage1_align.yaml --max-steps 50 --out checkpoints/smoke
    python -m src.train --config configs/stage1_align.yaml --resume checkpoints/stage1_align/last.pt
    python -m src.train --config configs/stage1_align.yaml --set training.lr_projector=3e-4 --set dtype=bfloat16
"""
import argparse
import os

import torch
import yaml
from transformers import AutoTokenizer

from src.data.caption_dataset import CaptionDataset
from src.data.collator import make_collator
from src.data.feature_cache import FeatureStore
from src.data.vqa_dataset import VQADataset
from src.engine.loading import build_vlm, move_vlm
from src.engine.trainer import Trainer, split_by_image
from src.utils.config import get_device, load_config

TASKS = {"caption": CaptionDataset, "vqa": VQADataset}


def apply_overrides(cfg: dict, overrides) -> dict:
    """Apply "a.b.c=value" overrides (value parsed as YAML, so 3e-4, true, null, bfloat16 all work)."""
    for item in overrides:
        key, _, raw = item.partition("=")
        if not key or not _:
            raise ValueError(f"Override must look like key.path=value, got '{item}'")
        node = cfg
        *parents, leaf = key.split(".")
        for part in parents:
            node = node.setdefault(part, {})
        node[leaf] = yaml.safe_load(raw)
    return cfg


def build_trainer(cfg: dict, out_dir: str) -> Trainer:
    torch.manual_seed(cfg.get("seed", 0))
    device = get_device(cfg.get("device", "auto"))
    m, d = cfg["model"], cfg["data"]
    n_tokens = m["num_visual_tokens"]

    vlm = move_vlm(build_vlm(cfg), device)       # features come from the cache, so the ViT stays on the CPU

    tok = AutoTokenizer.from_pretrained(m["lm"])
    store = FeatureStore(d["features"])
    full = TASKS[d["task"]](d["train_manifest"], d["root"], tok, n_tokens, feature_store=store, mode="train",
                            **d.get("dataset_kwargs", {}))
    train_ds, val_ds = split_by_image(full, d.get("val_images", 200), cfg.get("seed", 0))
    summary = {k: f"{v[0] / 1e6:.2f}M" for k, v in vlm.parameter_summary().items()}
    print(f"train {len(train_ds)} | val {len(val_ds)} records | trainable: {summary}")

    return Trainer(vlm, train_ds, val_ds, make_collator(tok, "train"), device, out_dir,
                   training=cfg.get("training"), config=cfg)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--out", default=None, help="default: checkpoints/<run_name>")
    p.add_argument("--max-steps", type=int, default=None, help="stop after this many optimizer steps")
    p.add_argument("--resume", default=None, help="checkpoint to continue from")
    p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="override a config value, e.g. training.lr_projector=3e-4")
    args = p.parse_args()

    cfg = apply_overrides(load_config(args.config), args.set)
    out_dir = args.out or os.path.join("checkpoints", cfg["run_name"])
    trainer = build_trainer(cfg, out_dir)
    with open(os.path.join(out_dir, "config.yaml"), "w") as f:
        yaml.safe_dump(cfg, f)
    if args.resume:
        trainer.load_checkpoint(args.resume)
        print(f"resumed from {args.resume} at step {trainer.step}")
    print(f"{trainer.steps_per_epoch} steps/epoch, {trainer.total_steps} total, warmup {trainer.warmup_steps}")
    trainer.train(until_step=args.max_steps)


if __name__ == "__main__":
    main()
