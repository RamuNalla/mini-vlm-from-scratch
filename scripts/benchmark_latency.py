"""
Time VQA generation for the Phase 5 variants, with repeats (single runs are noisy: the same model has
measured 26 ms and 40 ms per question).

    HF_HUB_OFFLINE=1 python -m scripts.benchmark_latency

Cached features, greedy decoding, 8 new tokens max. For each variant and each batch size: one warm-up pass,
then `--repeats` timed passes over the same questions; reports the median and the spread of ms per question.
Writes results/latency.json.
"""
import argparse
import json
import os
import statistics

import torch
from transformers import AutoTokenizer

from src.data.collator import batch_to_device, make_collator
from src.data.feature_cache import FeatureStore
from src.data.vqa_dataset import VQADataset
from src.engine.loading import load_vlm_from_checkpoint
from src.eval.latency import sync
import time

VARIANTS = {
    "tokens_16 (linear)": "checkpoints/exp/tokens_16/stage2/best.pt",
    "ref: 64 tokens (linear)": "checkpoints/exp/ref/stage2/best.pt",
    "mlp, 64 tokens": "checkpoints/exp/proj_mlp/stage2/best.pt",
    "resampler, 64 tokens": "checkpoints/exp/proj_resampler/stage2/best.pt",
}


@torch.no_grad()
def time_pass(vlm, batches, max_new_tokens, device):
    sync(device)
    start = time.perf_counter()
    for batch in batches:
        vlm.generate(**batch, max_new_tokens=max_new_tokens)
    sync(device)
    return time.perf_counter() - start


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--questions", type=int, default=32)
    p.add_argument("--repeats", type=int, default=7)
    p.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 16])
    p.add_argument("--out", default="results/latency.json")
    args = p.parse_args()

    out = {}
    for name, ckpt in VARIANTS.items():
        vlm, cfg = load_vlm_from_checkpoint(ckpt, "auto")
        device = next(vlm.lm.parameters()).device
        tok = AutoTokenizer.from_pretrained(cfg["model"]["lm"])
        ds = VQADataset("data/manifests/vqa_eval.jsonl", "data", tok, cfg["model"]["num_visual_tokens"], mode="eval",
                        feature_store=FeatureStore("data/features/clip_b16_l-2_224"))
        collate = make_collator(tok, "eval")
        items = [ds[i] for i in range(args.questions)]
        out[name] = {"tokens": cfg["model"]["num_visual_tokens"], "projector": cfg["model"].get("projector", "linear")}
        for bs in args.batch_sizes:
            batches = []
            for s in range(0, len(items), bs):
                b = batch_to_device(collate(items[s:s + bs]), device)
                b.pop("meta"); b.pop("labels")
                batches.append(b)
            time_pass(vlm, batches, 8, device)                                       # warm-up
            ms = [1000 * time_pass(vlm, batches, 8, device) / len(items) for _ in range(args.repeats)]
            out[name][f"batch{bs}"] = {"median_ms": statistics.median(ms), "min_ms": min(ms), "max_ms": max(ms), "runs": ms}
            print(f"{name:28s} batch {bs:2d}: median {statistics.median(ms):6.1f} ms/question  (range {min(ms):.1f}-{max(ms):.1f})", flush=True)
        del vlm
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
