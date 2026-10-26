"""
Entry point: train one stage from a YAML config.

    python -m src.train --config configs/stage1_align.yaml
    python -m src.train --config configs/stage1_align.yaml --max-steps 50 --out checkpoints/smoke
    python -m src.train --config configs/stage1_align.yaml --resume checkpoints/stage1_align/last.pt
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
from src.engine.trainer import Trainer, load_trainable_weights, split_by_image
from src.models.language_model import LanguageModel
from src.models.vit.load_pretrained import load_clip_vit
from src.models.vlm import MiniVLM
from src.utils.config import get_device, load_config

TASKS = {"caption": CaptionDataset, "vqa": VQADataset}


def build_trainer(cfg: dict, out_dir: str) -> Trainer:
    torch.manual_seed(cfg.get("seed", 0))
    device = get_device(cfg.get("device", "auto"))
    m, d = cfg["model"], cfg["data"]

    lm = LanguageModel.from_pretrained(m["lm"], dtype=cfg.get("dtype", "float32"))
    vit = load_clip_vit(m["vit"])
    n_tokens = m["num_visual_tokens"]
    vlm = MiniVLM.build(vit, lm, m.get("projector", "linear"), n_tokens)

    if cfg["stage"] == "instruct":
        lora = m.get("lora", {})
        lm.apply_lora(r=lora.get("r", 16), alpha=lora.get("alpha", 32), dropout=lora.get("dropout", 0.05))
    vlm.set_stage(cfg["stage"])
    if m.get("unfreeze_vit_blocks") is not None:
        vlm.unfreeze_vit(last_n_blocks=m["unfreeze_vit_blocks"])
    if m.get("init_from"):
        loaded = load_trainable_weights(vlm, m["init_from"], prefixes=("projector.",))
        print(f"initialised {len(loaded)} projector tensors from {m['init_from']}")

    # features come from the cache, so the ViT is never run: keep it off the accelerator
    vlm.projector.to(device)
    vlm.lm.to(device)

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
    args = p.parse_args()

    cfg = load_config(args.config)
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
