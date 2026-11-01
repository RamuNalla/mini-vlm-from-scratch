from typing import Callable, Optional

import torch
from PIL import Image

from src.data.caption_dataset import CAPTION_PROMPTS
from src.data.templates import build_conversation_inputs
from src.data.vqa_dataset import vqa_prompt
from src.eval.vqa_accuracy import clean_vqa_prediction
from src.models.vlm import MiniVLM


def build_user_prompt(task: str, question: Optional[str] = None, raw: bool = False) -> str:
    """
    The user turn (without the <image> marker) for one request.
    caption: the first training caption prompt. vqa: the question plus the short-answer instruction
    the model was trained with. raw=True sends the question exactly as typed (free-form; the model was
    only trained on short-answer and caption prompts, so long answers are out of distribution).
    """
    assert task in ("caption", "vqa"), f"task must be 'caption' or 'vqa', got '{task}'"
    if task == "caption":
        return CAPTION_PROMPTS[0]
    if not question:
        raise ValueError("A question is required for the vqa task")
    return question if raw else vqa_prompt(question)


@torch.no_grad()
def ask(
    vlm: MiniVLM,
    tokenizer,
    image: Image.Image,
    user_prompt: str,
    transform: Callable,
    device,
    max_new_tokens: int = 40,
) -> str:
    """Generate the model's reply to one image and one user prompt."""
    device = torch.device(device)
    prompt = build_conversation_inputs(
        tokenizer,
        [{"role": "user", "content": f"<image>\n{user_prompt}"}],
        vlm.num_visual_tokens,
        add_generation_prompt=True,
    )
    ids = torch.tensor([prompt["input_ids"]], device=device)
    mask = torch.tensor([prompt["attention_mask"]], device=device)
    pixels = transform(image.convert("RGB")).unsqueeze(0).to(device)
    vlm.eval()
    out = vlm.generate(ids, mask, pixel_values=pixels, max_new_tokens=max_new_tokens)
    return tokenizer.decode(out[0], skip_special_tokens=True).strip()


def answer_vqa(vlm, tokenizer, image, question, transform, device, raw=False, max_new_tokens=8) -> str:
    """Convenience wrapper: a VQA question, with the reply cleaned like the evaluation does (unless raw)."""
    reply = ask(vlm, tokenizer, image, build_user_prompt("vqa", question, raw), transform, device,
                max_new_tokens=40 if raw else max_new_tokens)
    return reply if raw else clean_vqa_prediction(reply)
