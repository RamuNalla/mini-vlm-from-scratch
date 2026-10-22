"""
Phase 2 sanity run: can the VLM memorise 8 image-caption pairs by training only the projector?

Real CLIP ViT + real Qwen2.5-0.5B-Instruct, both frozen. If the plumbing (feature extraction,
pooling, splicing, label masking, loss) is correct, the loss goes to ~0 and greedy generation
reproduces the training captions. Before training the output is generic.

    python scripts/fetch_toy_coco.py --n 8
    python scripts/overfit_toy.py --projector linear --tokens 64 --steps 300
"""
import argparse
import json
import os
import time

import matplotlib.pyplot as plt
import torch
from PIL import Image

from src.data.templates import build_conversation_inputs
from src.models.language_model import LanguageModel
from src.models.vit.load_pretrained import clip_transform, load_clip_vit
from src.models.vlm import MiniVLM
from src.utils.config import get_device

PROMPT = "<image>\nDescribe the image briefly."


def sync(device):
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()


def device_memory_gb(device):
    if device.type == "mps":
        return torch.mps.current_allocated_memory() / 1e9
    if device.type == "cuda":
        return torch.cuda.max_memory_allocated() / 1e9
    return float("nan")


def generate_all(vlm, tok, features, n_tokens, device):
    prompt = build_conversation_inputs(
        tok, [{"role": "user", "content": PROMPT}], n_tokens, add_generation_prompt=True
    )
    b = features.shape[0]
    ids = torch.tensor([prompt["input_ids"]] * b, device=device)       # identical prompts: no padding
    mask = torch.tensor([prompt["attention_mask"]] * b, device=device)
    vlm.eval()
    out = vlm.generate(ids, mask, vision_features=features, max_new_tokens=40)
    return [tok.decode(o, skip_special_tokens=True).strip() for o in out]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/toy_coco")
    parser.add_argument("--projector", default="linear", choices=["linear", "mlp"])
    parser.add_argument("--tokens", type=int, default=64)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--out", default="results/overfit_toy.json")
    args = parser.parse_args()

    torch.manual_seed(0)
    device = get_device(args.device)
    print(f"device={device} projector={args.projector} tokens={args.tokens} dtype={args.dtype}")

    with open(os.path.join(args.data, "captions.json")) as f:
        records = json.load(f)
    captions = [r["caption"] for r in records]

    vit = load_clip_vit().to(device)
    lm = LanguageModel.from_pretrained(dtype=args.dtype, device=device)
    tok = lm.tokenizer
    vlm = MiniVLM.build(vit, lm, args.projector, args.tokens).to(device).set_stage("align")
    summary = vlm.parameter_summary()
    print(f"trainable: projector {summary['projector'][0] / 1e6:.2f}M | lm {summary['lm'][0]} | vit {summary['vit'][0]}")

    # the encoder is frozen, so encode the 8 images once and train on the cached features
    transform = clip_transform(224)
    pixels = torch.stack([transform(Image.open(os.path.join(args.data, r["image"])).convert("RGB")) for r in records]).to(device)
    with torch.no_grad():
        features = vlm.extract_vision_features(pixels)

    examples = [
        build_conversation_inputs(
            tok,
            [{"role": "user", "content": PROMPT}, {"role": "assistant", "content": c}],
            args.tokens,
        )
        for c in captions
    ]
    width = max(len(e["input_ids"]) for e in examples)                    # right-pad for training
    pad = tok.pad_token_id
    batch = {
        "input_ids": torch.tensor([e["input_ids"] + [pad] * (width - len(e["input_ids"])) for e in examples]),
        "attention_mask": torch.tensor([e["attention_mask"] + [0] * (width - len(e["input_ids"])) for e in examples]),
        "labels": torch.tensor([e["labels"] + [-100] * (width - len(e["input_ids"])) for e in examples]),
    }
    batch = {k: v.to(device) for k, v in batch.items()}
    print(f"batch: {tuple(batch['input_ids'].shape)}, trained-on tokens: {(batch['labels'] != -100).sum().item()}")

    before = generate_all(vlm, tok, features, args.tokens, device)

    optimizer = torch.optim.AdamW(vlm.trainable_parameters(), lr=args.lr, weight_decay=0.0)
    vlm.train()
    losses, step_times = [], []
    for step in range(1, args.steps + 1):
        t = time.time()
        loss = vlm(vision_features=features, **batch).loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(vlm.trainable_parameters(), 1.0)
        optimizer.step()
        sync(device)
        step_times.append(time.time() - t)
        losses.append(loss.item())
        if step == 1 or step % 25 == 0:
            print(f"step {step:4d}  loss {losses[-1]:.4f}  ({step_times[-1]:.2f}s/step)")
        if losses[-1] < 1e-3:
            print(f"loss < 1e-3 at step {step}, stopping")
            break

    after = generate_all(vlm, tok, features, args.tokens, device)
    exact = [a == c for a, c in zip(after, captions)]

    print("\n" + "=" * 70)
    for c, b, a, ok in zip(captions, before, after, exact):
        print(f"target : {c}\nbefore : {b}\nafter  : {a}  [{'MATCH' if ok else 'differs'}]\n")
    steady = step_times[1:] or step_times
    print(f"exact matches: {sum(exact)}/{len(exact)} | final loss {losses[-1]:.5f} | "
          f"{sum(steady) / len(steady):.2f}s/step | memory {device_memory_gb(device):.2f} GB")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({
            "config": vars(args),
            "losses": losses,
            "mean_step_seconds": sum(steady) / len(steady),
            "memory_gb": device_memory_gb(device),
            "exact_matches": sum(exact),
            "samples": [{"target": c, "before": b, "after": a, "match": ok}
                        for c, b, a, ok in zip(captions, before, after, exact)],
        }, f, indent=2)

    os.makedirs("docs/assets", exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 3.2))
    ax.plot(range(1, len(losses) + 1), losses, color="#2f6db5", lw=2)
    ax.set_yscale("log")
    ax.set_xlabel("step")
    ax.set_ylabel("loss")
    ax.set_title(f"Overfit 8 pairs, projector only ({args.projector}, {args.tokens} tokens)", fontsize=10, loc="left")
    ax.grid(axis="y", color="#e3e5e8", lw=0.6)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    plt.tight_layout()
    plt.savefig("docs/assets/overfit_loss.png", dpi=130)


if __name__ == "__main__":
    main()
