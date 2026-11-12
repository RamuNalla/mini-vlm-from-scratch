"""
Pick random failures (soft accuracy 0) of the full-recipe model per question group and draw a contact sheet of each,
so they can be read against the images.

    python -m scripts.failure_examples --groups counting spatial colour --n 8
Writes results/failure_analysis/examples_<group>.{png,json}. Seeded, so the same failures come back every time.
"""
import argparse
import json
import os
import random
import textwrap

from PIL import Image, ImageDraw

from src.analysis.question_groups import question_group
from src.data.manifest_dataset import read_jsonl

OUT = "results/failure_analysis"
CELL_W, IMG_H, TEXT_H, COLS = 330, 250, 110, 4


def sheet(items, root, path):
    rows = (len(items) + COLS - 1) // COLS
    canvas = Image.new("RGB", (COLS * CELL_W, rows * (IMG_H + TEXT_H)), "white")
    draw = ImageDraw.Draw(canvas)
    for i, it in enumerate(items):
        x, y = (i % COLS) * CELL_W, (i // COLS) * (IMG_H + TEXT_H)
        img = Image.open(os.path.join(root, it["image"])).convert("RGB")
        img.thumbnail((CELL_W - 6, IMG_H))
        canvas.paste(img, (x + 3, y + 3))
        label = f"#{i + 1} Q: {it['question']}\nmodel: {it['cleaned']}\ntruth: {it['answer']}"
        draw.multiline_text((x + 6, y + IMG_H + 4), "\n".join(textwrap.fill(l, 46) for l in label.split("\n")), fill="black")
    canvas.save(path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--groups", nargs="+", default=["counting", "spatial", "colour"])
    p.add_argument("--n", type=int, default=8)
    p.add_argument("--model", default="results/stage2")
    args = p.parse_args()
    paths = {r["question_id"]: r["image"] for r in read_jsonl("data/manifests/vqa_eval.jsonl")}
    rows = read_jsonl(os.path.join(args.model, "vqa_predictions.jsonl"))
    os.makedirs(OUT, exist_ok=True)
    for g in args.groups:
        wrong = [r for r in rows if r["score"] == 0 and question_group(r["question"], r["answer_type"]) == g]
        picked = random.Random(0).sample(wrong, min(args.n, len(wrong)))
        items = [{**r, "image": paths[r["question_id"]]} for r in picked]
        sheet(items, "data", os.path.join(OUT, f"examples_{g}.png"))
        with open(os.path.join(OUT, f"examples_{g}.json"), "w") as f:
            json.dump([{k: v for k, v in it.items() if k in ("question_id", "question", "cleaned", "answer", "image")} for it in items], f, indent=1)
        print(f"{g}: {len(wrong)} failures, drew {len(items)}")


if __name__ == "__main__":
    main()
