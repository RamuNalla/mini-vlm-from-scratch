"""
Evaluate a VLM on the held-out VQA questions and caption images, or compute a baseline.

    # a trained checkpoint (architecture and tokens come from the checkpoint's saved config)
    python -m scripts.evaluate --name stage1 --checkpoint checkpoints/stage1_align/best.pt

    # baselines
    python -m scripts.evaluate --name untrained                  # random-init projector, same architecture
    python -m scripts.evaluate --name text_only --text-only      # the LM with no image at all
    python -m scripts.evaluate --name constant_yes --constant yes   # answers "yes" to every question, no model

Writes results/<name>/{vqa,caption}_metrics.json and {vqa,caption}_predictions.jsonl.
"""
import argparse
import json
import os

import torch
from transformers import AutoTokenizer

from src.data.caption_dataset import CaptionDataset
from src.data.feature_cache import FeatureStore
from src.data.manifest_dataset import read_jsonl
from src.data.vqa_dataset import VQADataset
from src.engine.evaluator import Evaluator, constant_vqa_baseline
from src.engine.loading import build_vlm, load_vlm_from_checkpoint, move_vlm
from src.utils.config import get_device, load_config


def write_jsonl(path, rows):
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def save(out_dir, task, result, info):
    with open(os.path.join(out_dir, f"{task}_metrics.json"), "w") as f:
        json.dump({**info, "metrics": result["metrics"], "latency": result.get("latency")}, f, indent=2)
    if "predictions" in result:
        write_jsonl(os.path.join(out_dir, f"{task}_predictions.jsonl"), result["predictions"])


def print_vqa(m):
    print(f"VQA accuracy {m['accuracy']:.2f}% (n={m['n']})")
    for k, v in m["by_answer_type"].items():
        print(f"   {k:7} {v['accuracy']:6.2f}%  (n={v['n']})")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--name", required=True, help="results go to results/<name>")
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--config", default="configs/stage1_align.yaml", help="architecture for baselines without a checkpoint")
    p.add_argument("--text-only", action="store_true", help="no image: the LM answers from the question alone")
    p.add_argument("--constant", default=None, help="answer every VQA question with this string (no model)")
    p.add_argument("--tasks", nargs="+", default=["vqa", "caption"], choices=["vqa", "caption"])
    p.add_argument("--data", default="data")
    p.add_argument("--features", default=None, help="default: <data>/features/clip_b16_l-2_224")
    p.add_argument("--limit", type=int, default=None, help="evaluate only the first N items of each task")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--max-new-tokens-vqa", type=int, default=8)
    p.add_argument("--max-new-tokens-caption", type=int, default=40)
    p.add_argument("--device", default="auto")
    args = p.parse_args()

    out_dir = os.path.join("results", args.name)
    os.makedirs(out_dir, exist_ok=True)
    manifests = os.path.join(args.data, "manifests")
    info = {"name": args.name, "checkpoint": args.checkpoint, "text_only": args.text_only}

    if args.constant is not None:
        metas = read_jsonl(os.path.join(manifests, "vqa_eval.jsonl"))
        metas = metas[: args.limit] if args.limit else metas
        for m in metas:
            m["answers"] = m["answers"]
        result = {"metrics": constant_vqa_baseline(metas, args.constant)}
        save(out_dir, "vqa", result, {**info, "constant": args.constant})
        print_vqa(result["metrics"])
        return

    device = get_device(args.device)
    if args.checkpoint:
        vlm, cfg = load_vlm_from_checkpoint(args.checkpoint, device)
    else:
        torch.manual_seed(0)
        cfg = load_config(args.config)
        cfg["stage"] = "align"                       # baselines have no LoRA; the projector is random-init
        cfg["model"].pop("init_from", None)
        vlm = move_vlm(build_vlm(cfg), device).eval()
    n_tokens = cfg["model"]["num_visual_tokens"]
    tok = AutoTokenizer.from_pretrained(cfg["model"]["lm"])
    info["config"] = cfg

    kwargs = {"use_image": False} if args.text_only else {
        "feature_store": FeatureStore(args.features or os.path.join(args.data, "features", "clip_b16_l-2_224"))
    }
    evaluator = Evaluator(vlm, tok, device, args.batch_size)

    if "vqa" in args.tasks:
        ds = VQADataset(os.path.join(manifests, "vqa_eval.jsonl"), args.data, tok, n_tokens, mode="eval", **kwargs)
        result = evaluator.run_vqa(ds, args.max_new_tokens_vqa, args.limit)
        save(out_dir, "vqa", result, info)
        print_vqa(result["metrics"])
        print(f"   {result['latency']['ms_per_sample']:.0f} ms/sample, {result['latency']['tokens_per_second']:.0f} tok/s")
    if "caption" in args.tasks and not args.text_only:
        ds = CaptionDataset(os.path.join(manifests, "caption_eval.jsonl"), args.data, tok, n_tokens, mode="eval", **kwargs)
        result = evaluator.run_captions(ds, args.max_new_tokens_caption, args.limit)
        save(out_dir, "caption", result, info)
        m = result["metrics"]
        print(f"captions: BLEU-4 {m['bleu4']:.3f} | CIDEr {m['cider']:.3f} | avg {m['avg_words']:.1f} words (n={m['n']})")
        print(f"   {result['latency']['ms_per_sample']:.0f} ms/sample, {result['latency']['tokens_per_second']:.0f} tok/s")


if __name__ == "__main__":
    main()
