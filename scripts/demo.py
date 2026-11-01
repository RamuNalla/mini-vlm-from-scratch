"""
Ask the trained VLM about any image.

    # visual question answering (stage 2 checkpoint)
    python -m scripts.demo --image photo.jpg --question "How many people are there?"
    python -m scripts.demo --image photo.jpg -q "What color is the car?" -q "Is it raining?"

    # captioning (stage 1 checkpoint: stage 2 only learned short answers)
    python -m scripts.demo --image photo.jpg --caption --checkpoint checkpoints/stage1_align/best.pt

    # free-form question, sent exactly as typed
    python -m scripts.demo --image photo.jpg -q "Describe this picture." --raw

The architecture and visual-token count come from the config stored in the checkpoint.
The image is resized and centre-cropped to 224x224 (CLIP preprocessing).
"""
import argparse
import os

from PIL import Image

from src.engine.inference import ask, answer_vqa, build_user_prompt
from src.engine.loading import load_vlm_from_checkpoint
from src.models.vit.load_pretrained import clip_transform
from src.utils.config import get_device


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--image", required=True, help="path to any image file")
    p.add_argument("--question", "-q", action="append", default=[], help="a question (repeatable)")
    p.add_argument("--caption", action="store_true", help="also print a caption")
    p.add_argument("--checkpoint", default=None,
                   help="default: checkpoints/stage2_instruct/best.pt, or stage1_align for --caption only")
    p.add_argument("--raw", action="store_true", help="send questions as typed (no short-answer instruction)")
    p.add_argument("--max-new-tokens", type=int, default=40)
    p.add_argument("--device", default="auto")
    args = p.parse_args()

    if not args.question and not args.caption:
        p.error("give at least one --question, or --caption")
    if not os.path.exists(args.image):
        p.error(f"image not found: {args.image}")
    default = "stage1_align" if args.caption and not args.question else "stage2_instruct"
    checkpoint = args.checkpoint or f"checkpoints/{default}/best.pt"

    device = get_device(args.device)
    vlm, cfg = load_vlm_from_checkpoint(checkpoint, device, vit_on_device=True)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(cfg["model"]["lm"])
    transform = clip_transform(224)
    image = Image.open(args.image).convert("RGB")
    print(f"{args.image} ({image.size[0]}x{image.size[1]}) | {checkpoint} | {vlm.num_visual_tokens} visual tokens")

    if args.caption:
        print("caption :", ask(vlm, tok, image, build_user_prompt("caption"), transform, device, args.max_new_tokens))
    for q in args.question:
        print(f"Q: {q}\nA: {answer_vqa(vlm, tok, image, q, transform, device, raw=args.raw)}")


if __name__ == "__main__":
    main()
