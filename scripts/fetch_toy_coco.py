"""
Download a handful of COCO val images with captions for quick sanity runs.
Uses the Karpathy-split caption metadata (streamed from the HF hub) and the public COCO image URLs.

    python scripts/fetch_toy_coco.py --n 8        # -> data/toy_coco/{images/, captions.json}
"""
import argparse
import json
import os
import urllib.request

from datasets import load_dataset
from PIL import Image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=8)
    parser.add_argument("--out", default="data/toy_coco")
    args = parser.parse_args()

    os.makedirs(os.path.join(args.out, "images"), exist_ok=True)
    stream = load_dataset("yerevann/coco-karpathy", split="validation", streaming=True)

    records = []
    for ex in stream:
        if len(records) == args.n:
            break
        path = os.path.join(args.out, "images", ex["filename"])
        if not os.path.exists(path):
            urllib.request.urlretrieve(ex["url"], path)
        Image.open(path).convert("RGB")      # fail early on a corrupt download
        records.append({
            "cocoid": ex["cocoid"],
            "image": os.path.relpath(path, args.out),
            "caption": ex["sentences"][0].strip(),
            "all_captions": [s.strip() for s in ex["sentences"]],
        })

    with open(os.path.join(args.out, "captions.json"), "w") as f:
        json.dump(records, f, indent=2)
    for r in records:
        print(f"{r['cocoid']}: {r['caption']}")


if __name__ == "__main__":
    main()
