"""Inference helpers on a tiny random VLM (CPU): prompt construction and the single-image generate path."""
import numpy as np
import pytest
import torch
from PIL import Image
from transformers import AutoTokenizer, Qwen2Config, Qwen2ForCausalLM

from src.data.caption_dataset import CAPTION_PROMPTS
from src.data.templates import get_image_token_id
from src.data.vqa_dataset import VQA_SUFFIX
from src.engine.inference import answer_vqa, ask, build_user_prompt
from src.models.language_model import LanguageModel
from src.models.vit.vit import VisionTransformer
from src.models.vlm import MiniVLM

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
N, SIZE = 4, 64


@pytest.fixture(scope="module")
def tok():
    try:
        return AutoTokenizer.from_pretrained(MODEL_NAME)
    except OSError as e:
        pytest.skip(f"Tokenizer unavailable: {e}")


@pytest.fixture(scope="module")
def vlm(tok):
    torch.manual_seed(0)
    vit = VisionTransformer(image_size=SIZE, patch_size=16, emb_dim=24, depth=3, num_heads=4,
                            num_classes=None, dropout=0.0).eval()
    cfg = Qwen2Config(vocab_size=151936, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=512,
                      tie_word_embeddings=True)
    lm = LanguageModel(Qwen2ForCausalLM(cfg).eval(), tok)
    return MiniVLM.build(vit, lm, "linear", N).eval()


def to_tensor(img):
    return torch.from_numpy(np.array(img.convert("RGB").resize((SIZE, SIZE)))).permute(2, 0, 1).float() / 255


def make_image(color, size=(90, 70)):
    return Image.new("RGB", size, color)


def test_vqa_prompt_has_the_short_answer_suffix():
    prompt = build_user_prompt("vqa", "What color is the car?")
    assert prompt.startswith("What color is the car?") and prompt.endswith(VQA_SUFFIX)


def test_raw_prompt_is_the_question_as_typed():
    assert build_user_prompt("vqa", "Describe this.", raw=True) == "Describe this."


def test_caption_prompt_is_a_training_prompt():
    assert build_user_prompt("caption") == CAPTION_PROMPTS[0]


def test_vqa_requires_a_question_and_a_known_task():
    with pytest.raises(ValueError):
        build_user_prompt("vqa")
    with pytest.raises(AssertionError):
        build_user_prompt("segment", "x")


def test_ask_returns_text_and_is_deterministic(vlm, tok):
    img = make_image("red")
    a = ask(vlm, tok, img, "What is this?", to_tensor, "cpu", max_new_tokens=6)
    b = ask(vlm, tok, img, "What is this?", to_tensor, "cpu", max_new_tokens=6)
    assert isinstance(a, str) and a == b


def test_ask_actually_feeds_the_image(vlm, tok):
    """Different images must change the LM input; check at the embedding level, where it is exact."""
    from src.data.templates import build_conversation_inputs
    p = build_conversation_inputs(tok, [{"role": "user", "content": "<image>\nHi"}], N, add_generation_prompt=True)
    ids = torch.tensor([p["input_ids"]])
    with torch.no_grad():
        e_red = vlm.build_inputs_embeds(ids, vlm.encode_images(pixel_values=to_tensor(make_image("red"))[None]))
        e_blue = vlm.build_inputs_embeds(ids, vlm.encode_images(pixel_values=to_tensor(make_image("blue"))[None]))
    changed = (e_red != e_blue).any(-1)[0]
    assert torch.equal(changed, ids[0] == get_image_token_id(tok))


def test_ask_accepts_non_rgb_images(vlm, tok):
    gray = Image.new("L", (50, 80), 128)
    rgba = Image.new("RGBA", (50, 80), (10, 20, 30, 255))
    for img in (gray, rgba):
        assert isinstance(ask(vlm, tok, img, "Hi", to_tensor, "cpu", max_new_tokens=3), str)


def test_answer_vqa_cleans_the_reply_unless_raw(vlm, tok, monkeypatch):
    import src.engine.inference as inference
    monkeypatch.setattr(inference, "ask", lambda *a, **k: "Yes.\nbecause it is")
    assert answer_vqa(vlm, tok, make_image("red"), "Is it red?", to_tensor, "cpu") == "yes"
    assert answer_vqa(vlm, tok, make_image("red"), "Is it red?", to_tensor, "cpu", raw=True) == "Yes.\nbecause it is"
