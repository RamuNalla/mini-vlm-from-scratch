"""
Evaluator on a tiny random VLM (CPU, cached features, real Qwen tokenizer): ordering, batching
invariance, metric wiring, latency, text-only mode and the constant baseline.
"""
import json

import pytest
import torch
from transformers import AutoTokenizer, Qwen2Config, Qwen2ForCausalLM

from src.data.caption_dataset import CaptionDataset
from src.data.vqa_dataset import VQADataset
from src.engine.evaluator import Evaluator, constant_vqa_baseline
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


@pytest.fixture(scope="module")
def vlm(tok):
    torch.manual_seed(0)
    vit = VisionTransformer(image_size=64, patch_size=16, emb_dim=24, depth=3, num_heads=4,
                            num_classes=None, dropout=0.0).eval()
    cfg = Qwen2Config(vocab_size=151936, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=512,
                      tie_word_embeddings=True)
    lm = LanguageModel(Qwen2ForCausalLM(cfg).eval(), tok)
    torch.manual_seed(1)
    return MiniVLM.build(vit, lm, "linear", N).eval()


@pytest.fixture(scope="module")
def manifests(tmp_path_factory):
    root = tmp_path_factory.mktemp("eval")
    ids = list(range(10, 17))                                             # 7 images
    caps = [{"image_id": i, "image": f"images/{i}.jpg", "captions": [f"a cat {i}", f"the cat number {i}"]} for i in ids]
    vqa = [{
        "question_id": i * 10, "image_id": i, "image": f"images/{i}.jpg",
        "question": "Is " + "really " * (i % 4) + f"cat {i} here?", "answer": "yes" if i % 2 else "no",
        "answers": ["yes" if i % 2 else "no"] * 8 + ["maybe"] * 2,
        "question_type": "is the", "answer_type": "yes/no",
    } for i in ids]
    for name, rows in [("caption.jsonl", caps), ("vqa.jsonl", vqa)]:
        with open(root / name, "w") as f:
            f.write("\n".join(json.dumps(r) for r in rows) + "\n")
    return str(root)


@pytest.fixture(scope="module")
def store():
    g = torch.Generator().manual_seed(3)
    return {i: torch.randn(16, 24, generator=g) for i in range(10, 17)}


def vqa_ds(manifests, tok, store=None, **kw):
    kw = {"use_image": False} if store is None else {"feature_store": store}
    return VQADataset(f"{manifests}/vqa.jsonl", manifests, tok, N, mode="eval", **kw)


def evaluator(vlm, tok, batch_size=3):
    return Evaluator(vlm, tok, CPU, batch_size)


def test_generate_all_keeps_dataset_order_across_batches(vlm, tok, manifests, store):
    texts, metas, _ = evaluator(vlm, tok, 3).generate_all(vqa_ds(manifests, tok, store), max_new_tokens=4)
    assert len(texts) == len(metas) == 7
    assert [m["question_id"] for m in metas] == [i * 10 for i in range(10, 17)]     # 3 + 3 + 1 examples


def test_predictions_do_not_depend_on_batch_size(vlm, tok, manifests, store):
    ds = vqa_ds(manifests, tok, store)
    one, _, _ = evaluator(vlm, tok, 1).generate_all(ds, max_new_tokens=4)
    many, _, _ = evaluator(vlm, tok, 4).generate_all(ds, max_new_tokens=4)
    assert one == many                                                               # left padding is harmless


def test_limit(vlm, tok, manifests, store):
    texts, metas, latency = evaluator(vlm, tok).generate_all(vqa_ds(manifests, tok, store), 4, limit=2)
    assert len(texts) == 2 and latency["samples"] == 2


def test_run_vqa_structure_and_consistency(vlm, tok, manifests, store):
    r = evaluator(vlm, tok).run_vqa(vqa_ds(manifests, tok, store), max_new_tokens=4)
    m, rows = r["metrics"], r["predictions"]
    assert m["n"] == 7 and set(m["by_answer_type"]) == {"yes/no"} and "scores" not in m
    assert len(rows) == 7 and all(0.0 <= row["score"] <= 1.0 for row in rows)
    assert m["accuracy"] == pytest.approx(100 * sum(row["score"] for row in rows) / 7)
    assert {"question_id", "prediction", "cleaned", "answer", "answer_type", "question_type"} <= set(rows[0])
    assert r["latency"]["samples"] == 7 and r["latency"]["generated_tokens"] > 0
    assert r["latency"]["ms_per_sample"] > 0


def test_run_captions_structure(vlm, tok, manifests, store):
    ds = CaptionDataset(f"{manifests}/caption.jsonl", manifests, tok, N, feature_store=store, mode="eval")
    r = evaluator(vlm, tok).run_captions(ds, max_new_tokens=6)
    assert set(r["metrics"]) == {"bleu1", "bleu2", "bleu3", "bleu4", "cider", "avg_words", "n"}
    assert r["metrics"]["n"] == 7 and len(r["predictions"]) == 7
    assert r["predictions"][0]["references"] == ["a cat 10", "the cat number 10"]


def test_text_only_evaluation_runs_without_any_visual_input(vlm, tok, manifests):
    ds = vqa_ds(manifests, tok, store=None)
    assert not ds.use_image
    r = evaluator(vlm, tok).run_vqa(ds, max_new_tokens=4)
    assert r["metrics"]["n"] == 7 and len(r["predictions"]) == 7


def test_text_only_prompt_is_the_image_prompt_minus_the_image_block(vlm, tok, manifests, store):
    """Same question with and without image: the prompts differ by exactly the image block and its newline."""
    with_image = vqa_ds(manifests, tok, store)[0]["input_ids"]
    without = vqa_ds(manifests, tok, None)[0]["input_ids"]
    assert len(with_image) - len(without) == N + 3                  # N pads + vision_start/end + newline


def test_constant_baseline_matches_hand_computation(manifests):
    from src.data.manifest_dataset import read_jsonl
    rows = read_jsonl(f"{manifests}/vqa.jsonl")
    r = constant_vqa_baseline(rows, "yes")
    # odd image ids have 8 "yes" annotators (score 1.0), even ids have 0 (score 0.0)
    expected = 100 * sum(1.0 for row in rows if row["answer"] == "yes") / len(rows)
    assert r["accuracy"] == pytest.approx(expected) and r["n"] == 7
    assert r["by_answer_type"]["yes/no"]["n"] == 7 and "scores" not in r
