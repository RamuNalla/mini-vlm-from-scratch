"""
Does the model use the image? Re-answer the 3,000 held-out questions with the image features replaced:

    real      the question's own image (the normal evaluation)
    other     the features of a different image (item i gets the image of item i + n/2)
    mean      the average image feature of the whole eval set at each patch (an "empty" image: same statistics, no content)

    HF_HUB_OFFLINE=1 python -m scripts.image_ablation
Writes results/failure_analysis/ablation.{json,md}. Reports accuracy per question group and how often the answer changes.
"""
import json
import os

import numpy as np
import torch
from transformers import AutoTokenizer

from src.analysis.question_groups import GROUPS, question_group
from src.data.collator import batch_to_device, make_collator
from src.data.feature_cache import FeatureStore
from src.data.vqa_dataset import VQADataset
from src.engine.loading import load_vlm_from_checkpoint
from src.eval.vqa_accuracy import clean_vqa_prediction, evaluate_vqa

CKPT = "checkpoints/stage2_instruct/best.pt"
OUT = "results/failure_analysis"


@torch.no_grad()
def run(vlm, tok, ds, device, features, bs=32):
    collate, texts, metas = make_collator(tok, "eval"), [], []
    for s in range(0, len(ds), bs):
        items = [dict(ds[i]) for i in range(s, min(s + bs, len(ds)))]
        for k, it in enumerate(items):
            it["vision_features"] = features[s + k]
        batch = batch_to_device(collate(items), device)
        metas += batch.pop("meta")
        batch.pop("labels")
        texts += tok.batch_decode(vlm.generate(**batch, max_new_tokens=8), skip_special_tokens=True)
    return texts, metas


def main():
    vlm, cfg = load_vlm_from_checkpoint(CKPT, "auto")
    device = next(vlm.lm.parameters()).device
    tok = AutoTokenizer.from_pretrained(cfg["model"]["lm"])
    store = FeatureStore("data/features/clip_b16_l-2_224")
    ds = VQADataset("data/manifests/vqa_eval.jsonl", "data", tok, cfg["model"]["num_visual_tokens"], mode="eval", feature_store=store)
    n = len(ds)
    real = [store[r["image_id"]] for r in ds.records]
    mean = torch.stack(real).float().mean(0).to(real[0].dtype)
    conditions = {"real": real, "other": [real[(i + n // 2) % n] for i in range(n)], "mean": [mean] * n}

    preds, stats = {}, {}
    for name, feats in conditions.items():
        texts, metas = run(vlm, tok, ds, device, feats)
        res = evaluate_vqa(texts, metas)
        scores = res["scores"]
        preds[name] = [clean_vqa_prediction(t) for t in texts]
        groups = [question_group(m["question"], m["answer_type"]) for m in metas]
        stats[name] = {g: 100 * float(np.mean([s for s, gg in zip(scores, groups) if gg == g])) for g in GROUPS}
        stats[name]["all"] = res["accuracy"]
        print(name, {k: round(v, 1) for k, v in stats[name].items()}, flush=True)
    for name in ("other", "mean"):
        stats[name]["answer_changed_pct"] = 100 * float(np.mean([a != b for a, b in zip(preds["real"], preds[name])]))

    cols = GROUPS + ["all"]
    lines = ["| Image given | " + " | ".join(cols) + " | answer differs from real |", "|---|" + "---|" * (len(cols) + 1)]
    for name in conditions:
        chg = f"{stats[name]['answer_changed_pct']:.0f}%" if name != "real" else "-"
        lines.append(f"| {name} | " + " | ".join(f"{stats[name][c]:.1f}" for c in cols) + f" | {chg} |")
    os.makedirs(OUT, exist_ok=True)
    json.dump(stats, open(os.path.join(OUT, "ablation.json"), "w"), indent=2)
    open(os.path.join(OUT, "ablation.md"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
