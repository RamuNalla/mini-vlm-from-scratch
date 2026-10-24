"""
Cache frozen CLIP ViT patch features (layer -2, CLS dropped) for every image in the manifests.

    python -m scripts.extract_features                      # data/features/clip_b16_l-2_224
    python -m scripts.extract_features --dtype float32      # 2x the size, exact fp32 features

The result is a 14x14 = 196 token grid per image; the VLM pools it down to 64 or 16 tokens.
"""
import argparse
import glob
import os
import time

from src.data.feature_cache import extract_features
from src.data.manifest_dataset import read_jsonl
from src.models.vit.load_pretrained import clip_transform, load_clip_vit
from src.utils.config import get_device


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data")
    p.add_argument("--out", default=None, help="default: <data>/features/clip_b16_l<layer>_<res>")
    p.add_argument("--model", default="openai/clip-vit-base-patch16")
    p.add_argument("--layer", type=int, default=-2)
    p.add_argument("--resolution", type=int, default=224)
    p.add_argument("--dtype", default="float16", choices=["float16", "float32"])
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--device", default="auto")
    args = p.parse_args()

    out = args.out or os.path.join(args.data, "features", f"clip_b16_l{args.layer}_{args.resolution}")
    paths = {}
    for manifest in sorted(glob.glob(os.path.join(args.data, "manifests", "*.jsonl"))):
        for r in read_jsonl(manifest):
            paths[r["image_id"]] = os.path.join(args.data, r["image"])
    print(f"{len(paths)} unique images -> {out}")

    device = get_device(args.device)
    vit = load_clip_vit(args.model).to(device)
    t = time.time()
    store = extract_features(
        vit, paths, out, clip_transform(args.resolution), device,
        layer=args.layer, batch_size=args.batch_size, dtype=args.dtype, num_workers=args.workers,
        model_name=args.model, resolution=args.resolution,
    )
    dt = time.time() - t
    size_gb = os.path.getsize(os.path.join(out, "features.npy")) / 1e9
    print(f"done in {dt:.0f}s ({len(store) / dt:.1f} img/s) | complete={store.is_complete()} | features.npy {size_gb:.2f} GB")


if __name__ == "__main__":
    main()
