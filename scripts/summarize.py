"""
Collect the Phase 5 results into one table.

    python -m scripts.summarize            # prints the table and writes results/exp_summary.md
"""
import glob
import json
import os


def load(path):
    with open(path) as f:
        return json.load(f)


def collect(results_dir="results/exp", ckpt_dir="checkpoints/exp"):
    rows = []
    for vqa_path in sorted(glob.glob(os.path.join(results_dir, "*", "vqa_metrics.json"))):
        name = os.path.basename(os.path.dirname(vqa_path))
        vqa = load(vqa_path)
        cap_path = os.path.join(os.path.dirname(vqa_path), "caption_metrics.json")
        cap = load(cap_path) if os.path.exists(cap_path) else None
        cfg = vqa.get("config", {})
        model = cfg.get("model", {})
        trainable = {}
        for stage in ("stage2", "stage1"):                       # the last stage that was trained
            done = os.path.join(ckpt_dir, name, stage, "done.json")
            if os.path.exists(done):
                trainable = load(done)["trainable_params"]
                break
        m = vqa["metrics"]
        rows.append({
            "name": name,
            "projector": model.get("projector", "?"),
            "tokens": model.get("num_visual_tokens", "?"),
            "vqa": m["accuracy"],
            "yes_no": m["by_answer_type"].get("yes/no", {}).get("accuracy", float("nan")),
            "number": m["by_answer_type"].get("number", {}).get("accuracy", float("nan")),
            "other": m["by_answer_type"].get("other", {}).get("accuracy", float("nan")),
            "cider": cap["metrics"]["cider"] if cap else float("nan"),
            "ms_per_q": (vqa.get("latency") or {}).get("ms_per_sample", float("nan")),
            "trainable_m": (trainable.get("projector", 0) + trainable.get("lm", 0) + trainable.get("vit", 0)) / 1e6,
        })
    return rows


def to_markdown(rows) -> str:
    head = "| Variant | Projector | Tokens | VQA | yes/no | number | other | CIDEr | ms/question | Trainable (M) |\n|---|---|---|---|---|---|---|---|---|---|\n"
    lines = [
        f"| {r['name']} | {r['projector']} | {r['tokens']} | {r['vqa']:.2f} | {r['yes_no']:.2f} | {r['number']:.2f} | "
        f"{r['other']:.2f} | {r['cider']:.3f} | {r['ms_per_q']:.0f} | {r['trainable_m']:.2f} |"
        for r in rows
    ]
    return head + "\n".join(lines) + "\n"


def main():
    rows = collect()
    if not rows:
        print("no results found under results/exp")
        return
    table = to_markdown(rows)
    print(table)
    with open("results/exp_summary.md", "w") as f:
        f.write(table)


if __name__ == "__main__":
    main()
