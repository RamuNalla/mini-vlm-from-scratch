import pytest
import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

from src.models.language_model import LanguageModel

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"


def tiny_lm() -> LanguageModel:
    cfg = Qwen2Config(
        vocab_size=100, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=128,
        tie_word_embeddings=True,
    )
    torch.manual_seed(0)
    return LanguageModel(Qwen2ForCausalLM(cfg))


def test_hidden_size_and_embeddings():
    lm = tiny_lm()
    assert lm.hidden_size == 32
    assert lm.get_input_embeddings().weight.shape == (100, 32)


def test_forward_with_inputs_embeds_returns_loss_and_logits():
    lm = tiny_lm()
    embeds = torch.randn(2, 7, 32)
    labels = torch.randint(0, 100, (2, 7))
    out = lm(inputs_embeds=embeds, attention_mask=torch.ones(2, 7, dtype=torch.long), labels=labels)
    assert out.logits.shape == (2, 7, 100)
    assert torch.isfinite(out.loss)


def test_ignore_index_labels_are_excluded_from_loss():
    lm = tiny_lm().eval()
    embeds = torch.randn(1, 6, 32)
    full = torch.randint(0, 100, (1, 6))
    masked = full.clone()
    masked[:, :3] = -100
    loss_full = lm(inputs_embeds=embeds, labels=full).loss
    loss_masked = lm(inputs_embeds=embeds, labels=masked).loss
    assert not torch.isclose(loss_full, loss_masked)


def test_freeze_stops_all_gradients():
    lm = tiny_lm().freeze()
    assert lm.num_parameters(trainable_only=True) == 0
    embeds = torch.randn(1, 5, 32, requires_grad=True)
    lm(inputs_embeds=embeds, labels=torch.randint(0, 100, (1, 5))).loss.backward()
    assert embeds.grad is not None                       # gradients still flow *through* the LM
    assert all(p.grad is None for p in lm.parameters())  # but none are stored in its weights


def test_lora_leaves_only_adapters_trainable():
    lm = tiny_lm()
    total_before = lm.num_parameters()
    lm.apply_lora(r=4, alpha=8)
    trainable = [n for n, p in lm.named_parameters() if p.requires_grad]
    assert trainable and all("lora_" in n for n in trainable)
    assert 0 < lm.num_parameters(trainable_only=True) < 0.2 * total_before
    # base weights are frozen
    assert all(not p.requires_grad for n, p in lm.named_parameters() if "lora_" not in n)


def test_lora_covers_every_block_linear():
    lm = tiny_lm().apply_lora(r=4)
    names = [n for n, p in lm.named_parameters() if p.requires_grad]
    for proj in ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]:
        assert sum(proj in n for n in names) == 2 * 2        # 2 layers x (lora_A, lora_B)


def test_lora_starts_as_identity_and_gets_gradients():
    lm = tiny_lm().eval()
    embeds = torch.randn(1, 5, 32)
    before = lm(inputs_embeds=embeds).logits
    lm.apply_lora(r=4)
    lm.eval()
    assert torch.allclose(lm(inputs_embeds=embeds).logits, before, atol=1e-5)   # lora_B is zero-initialised

    lm.train()
    lm(inputs_embeds=embeds, labels=torch.randint(0, 100, (1, 5))).loss.backward()
    grads = [p.grad for n, p in lm.named_parameters() if "lora_B" in n]
    assert all(g is not None and g.abs().sum() > 0 for g in grads)


def test_lora_cannot_be_applied_twice():
    lm = tiny_lm().apply_lora(r=4)
    with pytest.raises(RuntimeError):
        lm.apply_lora(r=4)


def test_input_embedding_access_after_lora():
    lm = tiny_lm().apply_lora(r=4)
    assert lm.get_input_embeddings()(torch.tensor([[1, 2, 3]])).shape == (1, 3, 32)


def test_unknown_dtype_rejected():
    with pytest.raises(AssertionError):
        LanguageModel.from_pretrained(MODEL_NAME, dtype="int8")


@pytest.fixture(scope="module")
def qwen():
    try:
        return LanguageModel.from_pretrained(MODEL_NAME, dtype="float32")
    except OSError as e:
        pytest.skip(f"Qwen weights unavailable: {e}")


def test_real_qwen_loads_and_generates(qwen):
    assert qwen.hidden_size == 896
    tok = qwen.tokenizer
    prompt = tok.apply_chat_template(
        [{"role": "user", "content": "What is the capital of France? Answer in one word."}],
        tokenize=False, add_generation_prompt=True,
    )
    ids = tok(prompt, return_tensors="pt")
    out = qwen.generate(**ids, max_new_tokens=8, do_sample=False)
    text = tok.decode(out[0, ids["input_ids"].shape[1]:], skip_special_tokens=True)
    assert "Paris" in text


def test_real_qwen_inputs_embeds_match_input_ids(qwen):
    """Feeding embedded tokens must equal feeding ids (the path the VLM relies on)."""
    ids = qwen.tokenizer("A cat sat on the mat", return_tensors="pt")["input_ids"]
    with torch.no_grad():
        by_ids = qwen.model(input_ids=ids).logits
        by_embeds = qwen(inputs_embeds=qwen.get_input_embeddings()(ids)).logits
    assert torch.allclose(by_ids, by_embeds, atol=1e-5)
