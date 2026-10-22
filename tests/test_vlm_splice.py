"""
VLM assembly: pooling, splicing visual tokens into the text embeddings, stage freezing,
loss/gradients, generation. Most tests use a tiny random ViT and Qwen-style LM on CPU
(with the real Qwen tokenizer); one parametrization also uses the real Qwen weights.
"""
import pytest
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, Qwen2Config, Qwen2ForCausalLM

from src.data.templates import build_conversation_inputs, get_image_token_id
from src.models.language_model import LanguageModel
from src.models.vit.vit import VisionTransformer
from src.models.vlm import MiniVLM, adaptive_avg_pool_matrix

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
VIT_DIM = 24
GRID = 8                       # tiny ViT: 128px / 16px patches -> 8x8 = 64 patch tokens
MSGS = [
    {"role": "user", "content": "<image>\nWhat is in the picture?"},
    {"role": "assistant", "content": "A cat."},
]


@pytest.fixture(scope="module")
def tok():
    try:
        return AutoTokenizer.from_pretrained(MODEL_NAME)
    except OSError as e:
        pytest.skip(f"Tokenizer unavailable: {e}")


def make_vit():
    torch.manual_seed(0)
    return VisionTransformer(
        image_size=128, patch_size=16, emb_dim=VIT_DIM, depth=3, num_heads=4,
        num_classes=None, dropout=0.0,
    ).eval()


def make_tiny_lm(tok) -> LanguageModel:
    cfg = Qwen2Config(
        vocab_size=151936, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=512,
        tie_word_embeddings=True,
    )
    torch.manual_seed(0)
    return LanguageModel(Qwen2ForCausalLM(cfg).eval(), tok)


@pytest.fixture
def lm(tok):
    return make_tiny_lm(tok)


def make_vlm(lm, n=16, kind="linear"):
    torch.manual_seed(1)
    return MiniVLM.build(make_vit(), lm, kind, n).set_stage("align")


def make_batch(tok, n, batch=2):
    ex = [build_conversation_inputs(tok, MSGS, n) for _ in range(batch)]
    return {k: torch.tensor([e[k] for e in ex]) for k in ex[0]}


def images(batch=2, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(batch, 3, 128, 128, generator=g)


# ------------------------------------------------------------------ pooling

@pytest.mark.parametrize("side,out", [(8, 4), (8, 3), (8, 5), (14, 8), (21, 16)])
def test_pooling_matches_torch_adaptive_avg_pool(tok, lm, side, out):
    vlm = make_vlm(lm, out * out)
    x = torch.randn(2, side * side, 6)
    expected = F.adaptive_avg_pool2d(x.transpose(1, 2).reshape(2, 6, side, side), out).flatten(2).transpose(1, 2)
    assert torch.allclose(vlm.pool_tokens(x), expected, atol=1e-6)


def test_pooling_is_identity_at_native_grid(tok, lm):
    vlm = make_vlm(lm, GRID * GRID)
    x = torch.randn(2, GRID * GRID, 6)
    assert torch.equal(vlm.pool_tokens(x), x)


def test_pooling_cannot_upsample_or_use_non_square_grids(tok, lm):
    vlm = make_vlm(lm, 16)
    with pytest.raises(AssertionError):
        make_vlm(lm, 64).pool_tokens(torch.randn(1, 16, 6))     # 4x4 grid -> 8x8 tokens
    with pytest.raises(AssertionError):
        vlm.pool_tokens(torch.randn(1, 15, 6))                  # not a square grid


def test_pool_matrix_rows_are_averages():
    m = adaptive_avg_pool_matrix(14, 8)
    assert m.shape == (8, 14)
    assert torch.allclose(m.sum(dim=1), torch.ones(8))


def test_num_visual_tokens_must_be_square(tok, lm):
    with pytest.raises(AssertionError):
        MiniVLM.build(make_vit(), lm, "linear", 20)


# ------------------------------------------------------------------ splicing

@pytest.mark.parametrize("n", [4, 16, 64])
def test_encode_images_shape(tok, lm, n):
    vlm = make_vlm(lm, n)
    assert vlm.encode_images(images()).shape == (2, n, lm.hidden_size)


def test_image_positions_get_exactly_the_visual_tokens(tok, lm):
    n = 16
    vlm = make_vlm(lm, n)
    batch = make_batch(tok, n)
    visual = vlm.encode_images(images())
    embeds = vlm.build_inputs_embeds(batch["input_ids"], visual)
    text_embeds = lm.get_input_embeddings()(batch["input_ids"])
    is_image = batch["input_ids"] == get_image_token_id(tok)

    for b in range(2):
        assert torch.equal(embeds[b][is_image[b]], visual[b])               # sample b got its own image
        assert torch.equal(embeds[b][~is_image[b]], text_embeds[b][~is_image[b]])   # text untouched


def test_samples_in_a_batch_get_different_images(tok, lm):
    vlm = make_vlm(lm, 16)
    visual = vlm.encode_images(images())
    assert not torch.allclose(visual[0], visual[1])


def test_changing_the_image_only_changes_image_positions(tok, lm):
    n = 16
    vlm = make_vlm(lm, n)
    batch = make_batch(tok, n)
    e1 = vlm.build_inputs_embeds(batch["input_ids"], vlm.encode_images(images(seed=0)))
    e2 = vlm.build_inputs_embeds(batch["input_ids"], vlm.encode_images(images(seed=1)))
    changed = (e1 != e2).any(dim=-1)
    assert torch.equal(changed, batch["input_ids"] == get_image_token_id(tok))


def test_wrong_number_of_pad_tokens_raises(tok, lm):
    vlm = make_vlm(lm, 16)
    batch = make_batch(tok, 4)                       # prompt built for 4 tokens, VLM expects 16
    with pytest.raises(ValueError, match="exactly 16"):
        vlm(pixel_values=images(), **batch)


def test_image_count_must_match_prompt_count(tok, lm):
    vlm = make_vlm(lm, 16)
    with pytest.raises(ValueError, match="images"):
        vlm(pixel_values=images(batch=3), **make_batch(tok, 16, batch=2))


def test_pad_tokens_without_an_image_raise(tok, lm):
    vlm = make_vlm(lm, 16)
    with pytest.raises(ValueError, match="no image"):
        vlm(**make_batch(tok, 16))


def test_exactly_one_image_source(tok, lm):
    vlm = make_vlm(lm, 16)
    feats = vlm.extract_vision_features(images())
    with pytest.raises(ValueError):
        vlm.encode_images(pixel_values=images(), vision_features=feats)
    with pytest.raises(ValueError):
        vlm.encode_images()


def test_text_only_forward(tok, lm):
    vlm = make_vlm(lm, 16)
    ex = build_conversation_inputs(
        tok, [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello"}]
    )
    batch = {k: torch.tensor([v]) for k, v in ex.items()}
    assert torch.isfinite(vlm(**batch).loss)


def test_cached_vision_features_equal_pixel_path(tok, lm):
    vlm = make_vlm(lm, 16)
    pix = images()
    feats = vlm.extract_vision_features(pix)
    assert torch.equal(vlm.encode_images(pixel_values=pix), vlm.encode_images(vision_features=feats))


# ------------------------------------------------------- end-to-end through the LM

@pytest.fixture(params=["tiny", "real_qwen"])
def e2e_lm(request, tok):
    if request.param == "tiny":
        return make_tiny_lm(tok)
    try:
        return LanguageModel.from_pretrained(MODEL_NAME, dtype="float32").eval()
    except OSError as e:
        pytest.skip(f"Qwen weights unavailable: {e}")


def test_visual_tokens_act_exactly_like_word_embeddings(tok, e2e_lm):
    """
    If the projector emitted the embedding of the word " cat" at every image position,
    the VLM must give the same logits as a text-only prompt where those positions are
    simply the token " cat". This proves the splice feeds the LM what we think it does.
    """
    n = 4
    vlm = make_vlm(e2e_lm, n)
    batch = make_batch(tok, n, batch=1)
    cat_id = tok(" cat", add_special_tokens=False)["input_ids"][0]
    cat_vec = e2e_lm.get_input_embeddings().weight[cat_id].detach()
    vlm.encode_images = lambda *args, **kw: cat_vec.expand(1, n, -1).clone()

    with torch.no_grad():
        via_vlm = vlm(pixel_values=images(1), **batch).logits
        replaced = batch["input_ids"].clone()
        replaced[replaced == get_image_token_id(tok)] = cat_id
        via_words = e2e_lm(inputs_embeds=e2e_lm.get_input_embeddings()(replaced)).logits
    assert torch.allclose(via_vlm, via_words, atol=1e-5)


def test_logits_before_the_image_ignore_it_but_logits_after_do_not(tok, lm):
    n = 16
    vlm = make_vlm(lm, n)
    batch = make_batch(tok, n, batch=1)
    first_image = (batch["input_ids"][0] == get_image_token_id(tok)).nonzero()[0].item()
    with torch.no_grad():
        l1 = vlm(pixel_values=images(1, seed=0), **batch).logits
        l2 = vlm(pixel_values=images(1, seed=1), **batch).logits
    assert torch.allclose(l1[:, :first_image], l2[:, :first_image], atol=1e-6)    # causal: past can't see it
    assert not torch.allclose(l1[:, -1], l2[:, -1], atol=1e-4)                    # the answer depends on it


# --------------------------------------------------------- loss and gradients

def test_loss_is_finite_and_only_the_projector_gets_gradients(tok, lm):
    vlm = make_vlm(lm, 16)
    out = vlm(pixel_values=images(), **make_batch(tok, 16))
    assert torch.isfinite(out.loss)
    out.loss.backward()
    assert all(p.grad is not None and p.grad.abs().sum() > 0 for p in vlm.projector.parameters())
    assert all(p.grad is None for p in vlm.vit.parameters())
    assert all(p.grad is None for p in vlm.lm.parameters())


def test_frozen_vit_features_carry_no_graph(tok, lm):
    vlm = make_vlm(lm, 16)
    assert not vlm.extract_vision_features(images()).requires_grad
    vlm.unfreeze_vit()
    assert vlm.extract_vision_features(images()).requires_grad


def test_unfrozen_vit_receives_gradients(tok, lm):
    vlm = make_vlm(lm, 16).unfreeze_vit(last_n_blocks=1)     # block 1 = last block used at layer -2
    vlm(pixel_values=images(), **make_batch(tok, 16)).loss.backward()
    assert all(p.grad is not None for p in vlm.vit.encoder_blocks[1].parameters())
    assert all(p.grad is None for p in vlm.vit.encoder_blocks[0].parameters())


def test_loss_is_cross_entropy_over_assistant_tokens_only(tok, lm):
    """Recompute the loss by hand: next-token CE averaged over the unmasked label positions."""
    vlm = make_vlm(lm, 16)
    batch = make_batch(tok, 16)
    with torch.no_grad():
        out = vlm(pixel_values=images(), **batch)
        logits, labels = out.logits[:, :-1], batch["labels"][:, 1:]
        keep = labels != -100
        assert 0 < keep.sum() < labels.numel()                  # some tokens masked, some trained on
        manual = F.cross_entropy(logits[keep], labels[keep])
    assert torch.allclose(out.loss, manual, atol=1e-5)


# ----------------------------------------------------------------- stages

def trainable_names(vlm):
    return [n for n, p in vlm.named_parameters() if p.requires_grad]


def test_align_stage_trains_only_the_projector(tok, lm):
    vlm = make_vlm(lm, 16)
    assert trainable_names(vlm) and all(n.startswith("projector.") for n in trainable_names(vlm))


def test_instruct_stage_requires_lora(tok, lm):
    with pytest.raises(RuntimeError, match="LoRA"):
        make_vlm(lm, 16).set_stage("instruct")


def test_instruct_stage_trains_projector_and_lora_only(tok, lm):
    lm.apply_lora(r=4)
    vlm = make_vlm(lm, 16).set_stage("instruct")
    names = trainable_names(vlm)
    assert any(n.startswith("projector.") for n in names) and any("lora_" in n for n in names)
    assert all(n.startswith("projector.") or "lora_" in n for n in names)


def test_switching_back_to_align_freezes_lora(tok, lm):
    lm.apply_lora(r=4)
    vlm = make_vlm(lm, 16).set_stage("instruct").set_stage("align")
    assert all(n.startswith("projector.") for n in trainable_names(vlm))


def test_unfreeze_vit_counts_only_blocks_that_feed_the_vlm(tok, lm):
    # tiny ViT has 3 blocks; with vit_layer=-2 only blocks 0 and 1 are run
    vlm = make_vlm(lm, 16).unfreeze_vit(last_n_blocks=1)
    vit_trainable = {n for n in trainable_names(vlm) if n.startswith("vit.")}
    assert vit_trainable and all(n.startswith("vit.encoder_blocks.1.") for n in vit_trainable)

    vlm = make_vlm(lm, 16).unfreeze_vit(last_n_blocks=2)
    vit_trainable = {n for n in trainable_names(vlm) if n.startswith("vit.")}
    assert all(n.startswith(("vit.encoder_blocks.0.", "vit.encoder_blocks.1.")) for n in vit_trainable)


def test_full_unfreeze_skips_unused_final_block_and_head(tok, lm):
    vlm = make_vlm(lm, 16).unfreeze_vit()
    names = trainable_names(vlm)
    assert any(n.startswith("vit.embeddings.") for n in names)
    assert not any(n.startswith(("vit.encoder_blocks.2.", "vit.norm.")) for n in names)
    # and every unfrozen ViT parameter really receives a gradient
    vlm(pixel_values=images(), **make_batch(tok, 16)).loss.backward()
    assert all(p.grad is not None for n, p in vlm.named_parameters() if n.startswith("vit.") and p.requires_grad)


def test_unknown_stage_rejected(tok, lm):
    with pytest.raises(AssertionError):
        make_vlm(lm, 16).set_stage("pretrain")


# -------------------------------------------------------------- generation

def test_generate_returns_new_tokens_only(tok, lm):
    n = 16
    vlm = make_vlm(lm, n)
    p = build_conversation_inputs(tok, MSGS[:1], n, add_generation_prompt=True)
    ids, am = torch.tensor([p["input_ids"]]), torch.tensor([p["attention_mask"]])
    out = vlm.generate(ids, am, pixel_values=images(1), max_new_tokens=5)
    assert out.shape[0] == 1 and 1 <= out.shape[1] <= 5


def test_generate_is_deterministic(tok, lm):
    n = 16
    vlm = make_vlm(lm, n)
    p = build_conversation_inputs(tok, MSGS[:1], n, add_generation_prompt=True)
    ids, am = torch.tensor([p["input_ids"]]), torch.tensor([p["attention_mask"]])
    a = vlm.generate(ids, am, pixel_values=images(1, seed=0), max_new_tokens=8)
    b = vlm.generate(ids, am, pixel_values=images(1, seed=0), max_new_tokens=8)
    assert torch.equal(a, b)


def reference_greedy(vlm, ids, visual, steps):
    """Cache-free greedy decoding straight from the LM forward pass (ground truth)."""
    embeds = vlm.build_inputs_embeds(ids, visual)
    out = []
    for _ in range(steps):
        nxt = vlm.lm(inputs_embeds=embeds).logits[:, -1].argmax(-1)
        out.append(nxt)
        embeds = torch.cat([embeds, vlm.lm.get_input_embeddings()(nxt[:, None])], dim=1)
    return torch.stack(out, dim=1)


def test_generate_matches_plain_greedy_despite_hostile_generation_config(tok, lm):
    """
    Qwen's generation_config ships repetition_penalty=1.1 (plus sampling defaults), which
    corrupted decoding from inputs_embeds (garbage first token). generate() must neutralise it.
    """
    n = 16
    vlm = make_vlm(lm, n)
    cfg = lm.model.generation_config
    cfg.repetition_penalty, cfg.do_sample, cfg.temperature, cfg.top_p, cfg.top_k = 2.0, True, 0.7, 0.8, 20
    p = build_conversation_inputs(tok, MSGS[:1], n, add_generation_prompt=True)
    ids, am = torch.tensor([p["input_ids"]]), torch.tensor([p["attention_mask"]])
    pix = images(1)

    steps = 6
    # eos can't be hit early with a random LM, but guard the comparison anyway
    got = vlm.generate(ids, am, pixel_values=pix, max_new_tokens=steps, eos_token_id=None)
    with torch.no_grad():
        expected = reference_greedy(vlm, ids, vlm.encode_images(pixel_values=pix), steps)
    assert torch.equal(got, expected)


def test_generate_rejects_right_padded_batches(tok, lm):
    n = 16
    vlm = make_vlm(lm, n)
    p = build_conversation_inputs(tok, MSGS[:1], n, add_generation_prompt=True)
    ids = torch.tensor([p["input_ids"] + [tok.pad_token_id]] * 2)
    am = torch.tensor([p["attention_mask"] + [0]] * 2)
    with pytest.raises(ValueError, match="left-padded"):
        vlm.generate(ids, am, pixel_values=images(2))


def test_batched_left_padded_generation_matches_single(tok, lm):
    """A left-padded batch element generates the same tokens as when run alone."""
    n = 16
    vlm = make_vlm(lm, n)
    short = build_conversation_inputs(tok, [{"role": "user", "content": "<image>\nHi"}], n, add_generation_prompt=True)
    long = build_conversation_inputs(tok, MSGS[:1], n, add_generation_prompt=True)
    pad = len(long["input_ids"]) - len(short["input_ids"])
    assert pad > 0
    ids = torch.tensor([[tok.pad_token_id] * pad + short["input_ids"], long["input_ids"]])
    am = torch.tensor([[0] * pad + short["attention_mask"], long["attention_mask"]])
    pix = images(2)

    batched = vlm.generate(ids, am, pixel_values=pix, max_new_tokens=6)
    alone = vlm.generate(torch.tensor([short["input_ids"]]), torch.tensor([short["attention_mask"]]),
                         pixel_values=pix[:1], max_new_tokens=6)
    k = min(batched.shape[1], alone.shape[1])
    assert torch.equal(batched[0, :k], alone[0, :k])
