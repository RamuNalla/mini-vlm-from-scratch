"""
Trainer: schedule, accumulation, validation, checkpoints, resume. Uses a tiny random ViT + Qwen-style
LM on CPU with cached-feature items (and the real Qwen tokenizer), so it runs in seconds.
"""
import os

import pytest
import torch
from transformers import AutoTokenizer, Qwen2Config, Qwen2ForCausalLM

from src.data.collator import make_collator
from src.data.templates import build_conversation_inputs
from src.engine.trainer import (
    Trainer, load_trainable_weights, num_target_tokens, split_by_image, trainable_state,
)
from src.models.language_model import LanguageModel
from src.models.vit.vit import VisionTransformer
from src.models.vlm import MiniVLM

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
N = 4
CPU = torch.device("cpu")


@pytest.fixture(scope="module")
def tok():
    try:
        return AutoTokenizer.from_pretrained(MODEL_NAME)
    except OSError as e:
        pytest.skip(f"Tokenizer unavailable: {e}")


class ListDataset:
    """In-memory dataset of cached-feature items with `records`, like the real datasets."""
    def __init__(self, items, records):
        self.items, self.records = items, records

    def __len__(self):
        return len(self.records)

    def __getitem__(self, i):
        return self.items[i]


def make_dataset(tok, n, seed=0):
    g = torch.Generator().manual_seed(seed)
    items, records = [], []
    for i in range(n):
        reply = f"a{' very' * (i % 4)} cat {i}"                      # different lengths -> different token counts
        msgs = [{"role": "user", "content": "<image>\nDescribe the image."}, {"role": "assistant", "content": reply}]
        item = build_conversation_inputs(tok, msgs, N)
        item["vision_features"] = torch.randn(16, 24, generator=g)   # 4x4 patch grid, dim 24
        item["meta"] = {"i": i}
        items.append(item)
        records.append({"image_id": i // 2})
    return ListDataset(items, records)


def make_vlm(tok, lora=False):
    torch.manual_seed(0)
    vit = VisionTransformer(image_size=64, patch_size=16, emb_dim=24, depth=3, num_heads=4,
                            num_classes=None, dropout=0.0).eval()
    cfg = Qwen2Config(vocab_size=151936, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=512,
                      tie_word_embeddings=True)
    lm = LanguageModel(Qwen2ForCausalLM(cfg), tok)
    torch.manual_seed(1)
    vlm = MiniVLM.build(vit, lm, "linear", N)
    if lora:
        lm.apply_lora(r=4, alpha=8, dropout=0.0)
    return vlm.set_stage("instruct" if lora else "align")


def make_trainer(tok, tmp_path, vlm=None, train_n=8, val_ds=None, lora=False, name="run", **training):
    vlm = vlm or make_vlm(tok, lora)
    cfg = dict(epochs=1, batch_size=4, warmup_steps=0, log_every=1, eval_every=1000, shuffle=False)
    cfg.update(training)
    return Trainer(vlm, make_dataset(tok, train_n), val_ds, make_collator(tok, "train"), CPU,
                   str(tmp_path / name), training=cfg, config={"run_name": name})


def weights(vlm):
    return {n: p.detach().clone() for n, p in vlm.named_parameters() if p.requires_grad}


def assert_same(a, b, atol=1e-6):
    assert a.keys() == b.keys()
    for k in a:
        assert torch.allclose(a[k], b[k], atol=atol), f"{k} differs by {(a[k] - b[k]).abs().max():.2e}"


# ---------------------------------------------------------------- schedule and data order

def test_lr_schedule_warmup_then_cosine(tok, tmp_path):
    t = make_trainer(tok, tmp_path, batch_size=2, epochs=5, warmup_steps=4)      # 4 steps/epoch -> 20 total
    assert t.total_steps == 20
    assert t.lr_factor(0) == pytest.approx(0.25) and t.lr_factor(3) == pytest.approx(1.0)
    assert t.lr_factor(4) == pytest.approx(1.0)
    after = [t.lr_factor(s) for s in range(4, 20)]
    assert all(a >= b for a, b in zip(after, after[1:]))                           # monotone decay
    assert t.lr_factor(19) == pytest.approx(0.1, abs=0.02)                         # ends near min_lr_ratio


def test_steps_per_epoch_drops_incomplete_groups(tok, tmp_path):
    t = make_trainer(tok, tmp_path, train_n=10, batch_size=3, grad_accum=2)
    assert t.steps_per_epoch == 1                                                   # (10 // 3) // 2
    batches = t.epoch_batches(0)
    assert len(batches) == 2 and all(len(b) == 3 for b in batches)


def test_epoch_order_is_deterministic_per_epoch_and_varies_between_epochs(tok, tmp_path):
    t = make_trainer(tok, tmp_path, shuffle=True, batch_size=2)
    first, again, second = t.epoch_batches(0), t.epoch_batches(0), t.epoch_batches(1)
    assert first == again and first != second
    flat = [i for b in first for i in b]
    assert sorted(flat) == list(range(8))                                           # every example exactly once


def test_split_by_image_holds_out_whole_images(tok):
    ds = make_dataset(tok, 10)
    train, val = split_by_image(ds, val_images=2, seed=0)
    assert {r["image_id"] for r in train.records}.isdisjoint({r["image_id"] for r in val.records})
    assert len({r["image_id"] for r in val.records}) == 2 and len(train) + len(val) == 10
    again = split_by_image(ds, val_images=2, seed=0)[1]
    assert again.records == val.records


# ----------------------------------------------------------------------- training

def test_loss_decreases(tok, tmp_path):
    t = make_trainer(tok, tmp_path, epochs=15, lr_projector=1e-2)
    history = t.train()
    losses = [r["train_loss"] for r in history if "train_loss" in r]
    assert len(losses) == 30 and losses[-1] < 0.9 * losses[0]


def test_only_the_projector_changes_in_align_stage(tok, tmp_path):
    vlm = make_vlm(tok)
    frozen_before = {n: p.detach().clone() for n, p in vlm.named_parameters() if not p.requires_grad}
    proj_before = weights(vlm)
    make_trainer(tok, tmp_path, vlm=vlm, epochs=2, lr_projector=1e-2).train()
    frozen_after = {n: p.detach() for n, p in vlm.named_parameters() if not p.requires_grad}
    assert_same(frozen_before, frozen_after, atol=0)                                # LM and ViT untouched
    assert any(not torch.allclose(proj_before[k], v) for k, v in weights(vlm).items())


def test_accumulation_equals_one_big_batch(tok, tmp_path):
    big = make_trainer(tok, tmp_path, name="big", batch_size=4, grad_accum=1, lr_projector=1e-2)
    acc = make_trainer(tok, tmp_path, name="acc", batch_size=2, grad_accum=2, lr_projector=1e-2)
    # the two micro-batches have different numbers of target tokens, so weighting matters
    collate = make_collator(tok, "train")
    counts = [num_target_tokens(collate([acc.train_ds[i] for i in idx])) for idx in ([0, 1], [2, 3])]
    assert counts[0] != counts[1]

    big.train(until_step=1)
    acc.train(until_step=1)
    assert_same(weights(big.vlm), weights(acc.vlm))


def test_validation_loss_is_token_weighted(tok, tmp_path):
    val = make_dataset(tok, 7, seed=5)
    t = make_trainer(tok, tmp_path, val_ds=val, batch_size=3)                      # 3 + 3 + 1 examples
    got = t.evaluate_loss()
    batch = make_collator(tok, "train")([val[i] for i in range(7)])
    batch.pop("meta")
    t.vlm.eval()
    with torch.no_grad():
        expected = t.vlm(**batch).loss.item()
    assert got == pytest.approx(expected, rel=1e-5)


def test_training_writes_history_plot_and_checkpoints(tok, tmp_path):
    val = make_dataset(tok, 4, seed=5)
    t = make_trainer(tok, tmp_path, val_ds=val, epochs=2, eval_every=1)
    t.train()
    for f in ("history.json", "loss_curve.png", "last.pt", "best.pt"):
        assert os.path.exists(os.path.join(t.out_dir, f)), f
    assert any("val_loss" in r for r in t.logger.history)


def test_nothing_trainable_is_an_error(tok, tmp_path):
    vlm = make_vlm(tok)
    for p in vlm.parameters():
        p.requires_grad = False
    with pytest.raises(AssertionError, match="Nothing is trainable"):
        make_trainer(tok, tmp_path, vlm=vlm)


# -------------------------------------------------------------------- checkpoints

def test_checkpoint_holds_only_trainable_parameters(tok, tmp_path):
    align = make_vlm(tok)
    assert set(trainable_state(align)) == {"projector.proj.weight", "projector.proj.bias"}

    instruct = make_vlm(tok, lora=True)
    names = set(trainable_state(instruct))
    assert any("lora_" in n for n in names) and any(n.startswith("projector.") for n in names)
    assert all(n.startswith("projector.") or "lora_" in n for n in names)           # no base weights


def test_checkpoint_roundtrip(tok, tmp_path):
    t = make_trainer(tok, tmp_path, epochs=2, lr_projector=1e-2)
    t.train()
    path = str(tmp_path / "ck.pt")
    t.save_checkpoint(path)
    saved = weights(t.vlm)
    for p in t.vlm.projector.parameters():
        p.data.add_(1.0)
    t.load_checkpoint(path)
    assert_same(saved, weights(t.vlm), atol=0)
    assert t.step == 4


def test_resume_matches_an_uninterrupted_run(tok, tmp_path):
    kwargs = dict(batch_size=2, epochs=2, shuffle=True, lr_projector=1e-2, warmup_steps=2)   # 4 steps/epoch, 8 total
    full = make_trainer(tok, tmp_path, name="full", **kwargs)
    full.train()

    first = make_trainer(tok, tmp_path, name="part", **kwargs)
    first.train(until_step=3)                                                       # stops mid-epoch
    path = str(tmp_path / "part" / "mid.pt")
    first.save_checkpoint(path)

    resumed = make_trainer(tok, tmp_path, name="part2", **kwargs)
    resumed.load_checkpoint(path)
    assert resumed.step == 3
    resumed.train()
    assert resumed.step == 8
    assert_same(weights(full.vlm), weights(resumed.vlm))
    assert [r["step"] for r in resumed.logger.history if "train_loss" in r] == list(range(1, 9))


def test_resume_restores_lora_and_projector(tok, tmp_path):
    kwargs = dict(batch_size=2, epochs=2, lora=True, lr_projector=1e-2, lr_lora=1e-2)
    full = make_trainer(tok, tmp_path, name="full", **kwargs)
    full.train()
    first = make_trainer(tok, tmp_path, name="part", **kwargs)
    first.train(until_step=3)
    path = str(tmp_path / "part" / "mid.pt")
    first.save_checkpoint(path)
    resumed = make_trainer(tok, tmp_path, name="part2", **kwargs)
    resumed.load_checkpoint(path)
    resumed.train()
    assert_same(weights(full.vlm), weights(resumed.vlm))


def test_load_checkpoint_rejects_a_different_trainable_set(tok, tmp_path):
    align = make_trainer(tok, tmp_path, name="a")
    path = str(tmp_path / "align.pt")
    align.save_checkpoint(path)
    with pytest.raises(ValueError, match="differ"):
        make_trainer(tok, tmp_path, name="b", lora=True).load_checkpoint(path)


def test_load_trainable_weights_with_prefix_takes_only_the_projector(tok, tmp_path):
    src = make_trainer(tok, tmp_path, name="src", lora=True, epochs=2, lr_projector=1e-2, lr_lora=1e-2)
    src.train()
    path = str(tmp_path / "src" / "last.pt")

    target = make_vlm(tok, lora=True)
    lora_before = {n: p.detach().clone() for n, p in target.named_parameters() if "lora_" in n}
    loaded = load_trainable_weights(target, path, prefixes=("projector.",))
    assert loaded == ["projector.proj.weight", "projector.proj.bias"]
    assert_same({k: v for k, v in weights(src.vlm).items() if k.startswith("projector.")},
                {k: v for k, v in weights(target).items() if k.startswith("projector.")}, atol=0)
    assert_same(lora_before, {n: p.detach() for n, p in target.named_parameters() if "lora_" in n}, atol=0)


def test_load_trainable_weights_rejects_unknown_names(tok, tmp_path):
    lora_trainer = make_trainer(tok, tmp_path, name="l", lora=True)
    path = str(tmp_path / "lora.pt")
    lora_trainer.save_checkpoint(path)
    with pytest.raises(KeyError, match="not found"):
        load_trainable_weights(make_vlm(tok), path)                                 # model without LoRA
