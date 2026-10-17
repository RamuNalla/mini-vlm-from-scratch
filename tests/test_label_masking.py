"""Chat template and label masking. Uses the real Qwen tokenizer (small download, cached)."""
import pytest
from transformers import AutoTokenizer

from src.data.templates import (
    IGNORE_INDEX,
    IMAGE_PAD,
    VISION_END,
    VISION_START,
    build_conversation_inputs,
    get_image_token_id,
)

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
N = 16

SINGLE = [
    {"role": "user", "content": "<image>\nWhat color is the car?"},
    {"role": "assistant", "content": "The car is red."},
]
MULTI = [
    {"role": "user", "content": "<image>\nWhat color is the car?"},
    {"role": "assistant", "content": "The car is red."},
    {"role": "user", "content": "How many wheels are visible?"},
    {"role": "assistant", "content": "Two."},
]


@pytest.fixture(scope="module")
def tok():
    try:
        return AutoTokenizer.from_pretrained(MODEL_NAME)
    except OSError as e:
        pytest.skip(f"Tokenizer unavailable: {e}")


def target_text(tok, out):
    """Decode only the positions that contribute to the loss."""
    return tok.decode([i for i, l in zip(out["input_ids"], out["labels"]) if l != IGNORE_INDEX])


def test_lengths_and_alignment(tok):
    out = build_conversation_inputs(tok, SINGLE, N)
    assert len(out["input_ids"]) == len(out["labels"]) == len(out["attention_mask"])
    assert all(m == 1 for m in out["attention_mask"])
    # wherever a label is set, it equals the input id (labels are not shifted here)
    assert all(l in (IGNORE_INDEX, i) for i, l in zip(out["input_ids"], out["labels"]))


def test_image_block_structure(tok):
    out = build_conversation_inputs(tok, SINGLE, N)
    ids = out["input_ids"]
    pad = get_image_token_id(tok)
    start, end = tok.convert_tokens_to_ids(VISION_START), tok.convert_tokens_to_ids(VISION_END)
    pad_positions = [i for i, t in enumerate(ids) if t == pad]
    assert len(pad_positions) == N
    # contiguous, and wrapped by <|vision_start|> ... <|vision_end|>
    assert pad_positions == list(range(pad_positions[0], pad_positions[0] + N))
    assert ids[pad_positions[0] - 1] == start and ids[pad_positions[-1] + 1] == end


@pytest.mark.parametrize("n", [1, 16, 64, 196])
def test_image_token_count_matches_request(tok, n):
    out = build_conversation_inputs(tok, SINGLE, n)
    assert out["input_ids"].count(get_image_token_id(tok)) == n


def test_single_turn_labels_cover_only_the_answer(tok):
    out = build_conversation_inputs(tok, SINGLE, N)
    assert target_text(tok, out) == "The car is red.<|im_end|>"


def test_multi_turn_labels_cover_every_assistant_reply(tok):
    out = build_conversation_inputs(tok, MULTI, N)
    assert target_text(tok, out) == "The car is red.<|im_end|>Two.<|im_end|>"


def test_image_system_and_user_tokens_are_masked(tok):
    out = build_conversation_inputs(tok, SINGLE, N)
    pad = get_image_token_id(tok)
    first_label = next(i for i, l in enumerate(out["labels"]) if l != IGNORE_INDEX)
    # everything before the first target token (system prompt, user turn, image block) is masked
    assert all(l == IGNORE_INDEX for l in out["labels"][:first_label])
    assert all(l == IGNORE_INDEX for t, l in zip(out["input_ids"], out["labels"]) if t == pad)


def test_matches_tokenizers_own_chat_template(tok):
    """Our hand-built ChatML equals apply_chat_template on the expanded text."""
    out = build_conversation_inputs(tok, MULTI, N)
    block = VISION_START + IMAGE_PAD * N + VISION_END
    expanded = [dict(m, content=m["content"].replace("<image>", block)) for m in MULTI]
    expected = tok.apply_chat_template(
        [{"role": "system", "content": "You are a helpful assistant."}] + expanded, tokenize=False
    )
    assert out["input_ids"] == tok(expected, add_special_tokens=False)["input_ids"]


def test_custom_system_prompt_is_used_and_masked(tok):
    out = build_conversation_inputs(tok, SINGLE, N, system_prompt="Be brief.")
    assert "Be brief." in tok.decode(out["input_ids"])
    assert "Be brief." not in target_text(tok, out)


def test_text_only_conversation(tok):
    msgs = [{"role": "user", "content": "Name a color."}, {"role": "assistant", "content": "Blue."}]
    out = build_conversation_inputs(tok, msgs)
    assert get_image_token_id(tok) not in out["input_ids"]
    assert target_text(tok, out) == "Blue.<|im_end|>"


def test_generation_prompt_mode(tok):
    msgs = [{"role": "user", "content": "<image>\nDescribe the image."}]
    out = build_conversation_inputs(tok, msgs, N, add_generation_prompt=True)
    assert tok.decode(out["input_ids"]).endswith("<|im_start|>assistant\n")
    assert all(l == IGNORE_INDEX for l in out["labels"])


def test_generation_prompt_requires_user_last(tok):
    with pytest.raises(ValueError):
        build_conversation_inputs(tok, SINGLE, N, add_generation_prompt=True)


def test_training_conversation_must_end_with_assistant(tok):
    with pytest.raises(ValueError):
        build_conversation_inputs(tok, SINGLE[:1], N)


def test_marker_and_token_count_must_agree(tok):
    with pytest.raises(ValueError):
        build_conversation_inputs(tok, SINGLE, None)                       # marker, no count
    text_only = [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello"}]
    with pytest.raises(ValueError):
        build_conversation_inputs(tok, text_only, N)                        # count, no marker
    two = [{"role": "user", "content": "<image> and <image>"}, {"role": "assistant", "content": "ok"}]
    with pytest.raises(ValueError):
        build_conversation_inputs(tok, two, N)                              # two markers


def test_truncation_after_image_is_fine_but_cutting_the_image_raises(tok):
    full = build_conversation_inputs(tok, SINGLE, N)
    pad_pos = [i for i, t in enumerate(full["input_ids"]) if t == get_image_token_id(tok)]
    ok = build_conversation_inputs(tok, SINGLE, N, max_length=pad_pos[-1] + 5)
    assert len(ok["input_ids"]) == pad_pos[-1] + 5
    with pytest.raises(ValueError, match="cut"):
        build_conversation_inputs(tok, SINGLE, N, max_length=pad_pos[0] + N // 2)
