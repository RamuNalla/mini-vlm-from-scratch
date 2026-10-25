"""
Datasets, collator and feature cache. Everything runs on a tiny synthetic dataset (generated
images and manifests in a temp dir, a tiny ViT), plus the real Qwen tokenizer; the last tests
check the real manifests if they have been downloaded.
"""
import json
import os
import pickle
import random

import numpy as np
import pytest
import torch
from PIL import Image
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from src.data.caption_dataset import CAPTION_PROMPTS, CaptionDataset
from src.data.collator import VLMCollator, batch_to_device, make_collator
from src.data.feature_cache import DONE, FeatureStore, extract_features
from src.data.manifest_dataset import read_jsonl
from src.data.templates import IGNORE_INDEX, get_image_token_id
from src.data.vqa_dataset import VQA_SUFFIX, VQADataset
from src.models.vit.vit import VisionTransformer

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
N = 4                      # visual tokens per image in these tests
SIZE = 32                  # synthetic image size


@pytest.fixture(scope="module")
def tok():
    try:
        return AutoTokenizer.from_pretrained(MODEL_NAME)
    except OSError as e:
        pytest.skip(f"Tokenizer unavailable: {e}")


def to_tensor(img: Image.Image) -> torch.Tensor:
    """Deterministic stand-in for clip_transform."""
    arr = np.array(img.convert("RGB").resize((SIZE, SIZE)))
    return torch.from_numpy(arr).permute(2, 0, 1).float() / 255.0


@pytest.fixture(scope="module")
def data_dir(tmp_path_factory):
    root = tmp_path_factory.mktemp("data")
    (root / "images").mkdir()
    rng = np.random.default_rng(0)
    image_ids = [101, 102, 103, 104, 105]
    for i in image_ids:
        Image.fromarray(rng.integers(0, 256, (40 + i % 7, 30 + i % 5, 3), dtype=np.uint8)).save(root / "images" / f"{i}.jpg")

    captions = [{"image_id": i, "image": f"images/{i}.jpg", "captions": [f"caption {i} number {k}." for k in range(5)]}
                for i in image_ids]
    vqa = []
    for n, i in enumerate(image_ids):
        for q in range(2):
            k = 2 * n + q                                  # vary question/answer length so padding is exercised
            answer = "a very small cat" * (k % 3) if k % 3 else "yes"
            vqa.append({
                "question_id": i * 10 + q, "image_id": i, "image": f"images/{i}.jpg",
                "question": "What is " + "really " * (k % 4) + f"in picture {i}?", "answer": answer,
                "answers": [answer] * 9 + ["dog"],
                "question_type": "what is", "answer_type": "other" if q else "yes/no",
            })
    (root / "manifests").mkdir()
    for name, rows in [("caption.jsonl", captions), ("vqa.jsonl", vqa)]:
        with open(root / "manifests" / name, "w") as f:
            f.write("\n".join(json.dumps(r) for r in rows) + "\n")
    return str(root)


def caption_ds(data_dir, tok, **kw):
    kw.setdefault("transform", to_tensor)
    return CaptionDataset(f"{data_dir}/manifests/caption.jsonl", data_dir, tok, N, **kw)


def vqa_ds(data_dir, tok, **kw):
    kw.setdefault("transform", to_tensor)
    return VQADataset(f"{data_dir}/manifests/vqa.jsonl", data_dir, tok, N, **kw)


def trained_on(tok, item):
    return tok.decode([i for i, l in zip(item["input_ids"], item["labels"]) if l != IGNORE_INDEX])


# ----------------------------------------------------------------- caption dataset

def test_caption_train_item(data_dir, tok):
    ds = caption_ds(data_dir, tok)
    item = ds[0]
    assert len(ds) == 5
    assert item["image"].shape == (3, SIZE, SIZE)
    assert item["input_ids"].count(get_image_token_id(tok)) == N
    assert len(item["input_ids"]) == len(item["labels"]) == len(item["attention_mask"])
    reply = trained_on(tok, item)
    assert reply.endswith("<|im_end|>") and reply[: -len("<|im_end|>")] in ds.records[0]["captions"]
    assert any(p in tok.decode(item["input_ids"]) for p in CAPTION_PROMPTS)
    assert item["meta"] == {"image_id": 101, "references": ds.records[0]["captions"]}


def test_caption_first_mode_is_deterministic(data_dir, tok):
    ds = caption_ds(data_dir, tok, sample="first")
    items = [ds[0] for _ in range(5)]
    assert all(it["input_ids"] == items[0]["input_ids"] for it in items)
    assert trained_on(tok, items[0]) == ds.records[0]["captions"][0] + "<|im_end|>"
    assert CAPTION_PROMPTS[0] in tok.decode(items[0]["input_ids"])


def test_caption_random_mode_varies(data_dir, tok):
    random.seed(0)
    ds = caption_ds(data_dir, tok)
    assert len({tuple(ds[0]["input_ids"]) for _ in range(20)}) > 1


def test_caption_eval_is_prompt_only(data_dir, tok):
    ds = caption_ds(data_dir, tok, mode="eval")
    item = ds[2]
    assert all(l == IGNORE_INDEX for l in item["labels"])
    assert tok.decode(item["input_ids"]).endswith("<|im_start|>assistant\n")
    assert CAPTION_PROMPTS[0] in tok.decode(item["input_ids"])
    assert item["meta"]["references"] == ds.records[2]["captions"]


# --------------------------------------------------------------------- vqa dataset

def test_vqa_train_item(data_dir, tok):
    ds = vqa_ds(data_dir, tok)
    item = ds[1]
    text = tok.decode(item["input_ids"])
    assert ds.records[1]["question"] in text and VQA_SUFFIX in text
    assert trained_on(tok, item) == ds.records[1]["answer"] + "<|im_end|>"
    assert item["input_ids"].count(get_image_token_id(tok)) == N
    assert item["meta"]["question_id"] == ds.records[1]["question_id"]


def test_vqa_eval_item_carries_everything_the_metric_needs(data_dir, tok):
    ds = vqa_ds(data_dir, tok, mode="eval")
    item = ds[0]
    assert all(l == IGNORE_INDEX for l in item["labels"])
    assert tok.decode(item["input_ids"]).endswith("<|im_start|>assistant\n")
    meta = item["meta"]
    assert len(meta["answers"]) == 10 and meta["answer_type"] in ("yes/no", "other", "number")
    assert {"question", "question_type", "question_id", "image_id"} <= set(meta)


# --------------------------------------------------------------- dataset contracts

def test_exactly_one_visual_source_is_required(data_dir, tok):
    with pytest.raises(ValueError):
        caption_ds(data_dir, tok, transform=None)                       # neither
    with pytest.raises(ValueError):
        caption_ds(data_dir, tok, feature_store={101: torch.zeros(4, 8)})   # both (transform defaults on)


def test_invalid_mode_rejected(data_dir, tok):
    with pytest.raises(AssertionError):
        caption_ds(data_dir, tok, mode="test")
    with pytest.raises(AssertionError):
        caption_ds(data_dir, tok, sample="best")


def test_max_length_cutting_the_image_raises(data_dir, tok):
    full = caption_ds(data_dir, tok)[0]["input_ids"]
    first_pad = full.index(get_image_token_id(tok))
    with pytest.raises(ValueError, match="cut"):
        caption_ds(data_dir, tok, max_length=first_pad + 2)[0]          # ends inside the image block
    assert len(caption_ds(data_dir, tok, max_length=first_pad + N + 5)[0]["input_ids"]) == first_pad + N + 5


def test_feature_store_items_have_features_not_pixels(data_dir, tok):
    store = {r["image_id"]: torch.randn(4, 8) for r in read_jsonl(f"{data_dir}/manifests/vqa.jsonl")}
    ds = vqa_ds(data_dir, tok, transform=None, feature_store=store)
    item = ds[3]
    assert "image" not in item and torch.equal(item["vision_features"], store[ds.records[3]["image_id"]])


# ---------------------------------------------------------------------- collator

def make_items(data_dir, tok, mode, n=5):
    ds = vqa_ds(data_dir, tok, mode=mode)
    return [ds[i] for i in range(n)]


def test_right_padding(data_dir, tok):
    items = make_items(data_dir, tok, "train")
    lens = [len(it["input_ids"]) for it in items]
    assert len(set(lens)) > 1, "test needs items of different lengths"
    b = make_collator(tok, "train")(items)
    assert b["input_ids"].shape == (5, max(lens)) and b["pixel_values"].shape == (5, 3, SIZE, SIZE)
    for i, L in enumerate(lens):
        assert b["attention_mask"][i, :L].all() and not b["attention_mask"][i, L:].any()
        assert b["input_ids"][i, :L].tolist() == items[i]["input_ids"]
        assert (b["input_ids"][i, L:] == tok.pad_token_id).all()
        assert b["labels"][i, :L].tolist() == items[i]["labels"] and (b["labels"][i, L:] == IGNORE_INDEX).all()


def test_left_padding_keeps_the_last_token_real(data_dir, tok):
    items = make_items(data_dir, tok, "eval")
    lens = [len(it["input_ids"]) for it in items]
    assert len(set(lens)) > 1
    b = make_collator(tok, "eval")(items)
    W = b["input_ids"].shape[1]
    assert b["attention_mask"][:, -1].all()
    for i, L in enumerate(lens):
        assert not b["attention_mask"][i, : W - L].any() and b["attention_mask"][i, W - L:].all()
        assert b["input_ids"][i, W - L:].tolist() == items[i]["input_ids"]
        assert (b["input_ids"][i, : W - L] == tok.pad_token_id).all()


def test_padding_never_adds_trained_tokens(data_dir, tok):
    items = make_items(data_dir, tok, "train")
    b = make_collator(tok, "train")(items)
    assert (b["labels"] != IGNORE_INDEX).sum() == sum((torch.tensor(it["labels"]) != IGNORE_INDEX).sum() for it in items)


def test_collator_stacks_cached_features_and_keeps_meta(data_dir, tok):
    store = {r["image_id"]: torch.randn(4, 8) for r in read_jsonl(f"{data_dir}/manifests/vqa.jsonl")}
    ds = vqa_ds(data_dir, tok, transform=None, feature_store=store, mode="eval")
    b = make_collator(tok, "eval")([ds[i] for i in range(3)])
    assert b["vision_features"].shape == (3, 4, 8) and "pixel_values" not in b
    assert [m["question_id"] for m in b["meta"]] == [ds.records[i]["question_id"] for i in range(3)]


def test_collator_rejects_mixed_visual_sources(data_dir, tok):
    pix = caption_ds(data_dir, tok)[0]
    feat = caption_ds(data_dir, tok, transform=None, feature_store={101: torch.zeros(4, 8)})[0]
    with pytest.raises(AssertionError):
        make_collator(tok)([pix, feat])


def test_collator_option_validation(tok):
    with pytest.raises(AssertionError):
        VLMCollator(tok.pad_token_id, "middle")
    with pytest.raises(AssertionError):
        make_collator(tok, "test")


def test_batch_to_device_leaves_meta_alone(data_dir, tok):
    b = batch_to_device(make_collator(tok)(make_items(data_dir, tok, "train", 2)), "cpu")
    assert all(torch.is_tensor(v) for k, v in b.items() if k != "meta") and isinstance(b["meta"], list)


def test_dataloader_integration(data_dir, tok):
    dl = DataLoader(vqa_ds(data_dir, tok), batch_size=4, collate_fn=make_collator(tok), shuffle=False)
    batches = list(dl)
    assert [b["input_ids"].shape[0] for b in batches] == [4, 4, 2]


# ------------------------------------------------------------------ feature cache

@pytest.fixture(scope="module")
def tiny_vit():
    torch.manual_seed(0)
    return VisionTransformer(
        image_size=SIZE, patch_size=16, emb_dim=24, depth=3, num_heads=4, num_classes=None, dropout=0.0
    ).eval()


@pytest.fixture
def image_paths(data_dir):
    return {r["image_id"]: os.path.join(data_dir, r["image"]) for r in read_jsonl(f"{data_dir}/manifests/caption.jsonl")}


def run_extract(vit, image_paths, out, **kw):
    kw.setdefault("dtype", "float32")
    return extract_features(vit, image_paths, str(out), to_tensor, torch.device("cpu"), layer=-2,
                            batch_size=2, num_workers=0, model_name="tiny", resolution=SIZE, **kw)


def direct(vit, path):
    with torch.no_grad():
        return vit.forward_features(to_tensor(Image.open(path).convert("RGB"))[None], layer=-2, drop_cls=True)[0]


def test_cache_matches_direct_encoding(tiny_vit, image_paths, tmp_path):
    store = run_extract(tiny_vit, image_paths, tmp_path / "c")
    assert len(store) == 5 and store.is_complete() and set(store) == set(image_paths)
    for i, path in image_paths.items():
        assert store[i].shape == (4, 24)                       # 2x2 patch grid, dim 24
        assert torch.allclose(store[i], direct(tiny_vit, path), atol=1e-6)


def test_fp16_cache_is_close_and_half_the_size(tiny_vit, image_paths, tmp_path):
    full = run_extract(tiny_vit, image_paths, tmp_path / "f32")
    half = run_extract(tiny_vit, image_paths, tmp_path / "f16", dtype="float16")
    assert half[101].dtype == torch.float16
    assert os.path.getsize(tmp_path / "f16" / "features.npy") < 0.6 * os.path.getsize(tmp_path / "f32" / "features.npy")
    for i in image_paths:
        assert torch.allclose(half[i].float(), full[i], rtol=2e-3, atol=2e-3)


def test_extraction_resumes_and_only_fills_missing_rows(tiny_vit, image_paths, tmp_path):
    out = tmp_path / "c"
    store = run_extract(tiny_vit, image_paths, out)
    expected = {i: store[i].clone() for i in image_paths}

    # simulate an interrupted run: two rows lost
    feats = np.load(out / "features.npy", mmap_mode="r+")
    done = np.load(out / DONE, mmap_mode="r+")
    lost = [store.index[102], store.index[104]]
    feats[lost], done[lost] = 0.0, False
    feats.flush(); done.flush()
    with pytest.raises(KeyError):
        FeatureStore(str(out))[102]                               # not extracted yet
    assert not FeatureStore(str(out)).is_complete()

    resumed = run_extract(tiny_vit, image_paths, out)
    assert resumed.is_complete()
    for i in image_paths:
        assert torch.allclose(resumed[i], expected[i], atol=1e-6)


def test_resuming_with_different_settings_is_refused(tiny_vit, image_paths, tmp_path):
    run_extract(tiny_vit, image_paths, tmp_path / "c")
    with pytest.raises(ValueError, match="built with"):
        run_extract(tiny_vit, image_paths, tmp_path / "c", dtype="float16")


def test_store_survives_pickling_for_dataloader_workers(tiny_vit, image_paths, tmp_path):
    store = run_extract(tiny_vit, image_paths, tmp_path / "c")
    _ = store[101]                                              # memmap already opened
    clone = pickle.loads(pickle.dumps(store))
    assert torch.equal(clone[101], store[101])


def test_cached_item_equals_pixel_item_through_the_encoder(data_dir, tok, tiny_vit, image_paths, tmp_path):
    store = run_extract(tiny_vit, image_paths, tmp_path / "c")
    pix_ds = caption_ds(data_dir, tok, sample="first")
    feat_ds = caption_ds(data_dir, tok, transform=None, feature_store=store, sample="first")
    for i in range(len(pix_ds)):
        pixel_item, feat_item = pix_ds[i], feat_ds[i]
        assert pixel_item["input_ids"] == feat_item["input_ids"]
        with torch.no_grad():
            from_pixels = tiny_vit.forward_features(pixel_item["image"][None], layer=-2, drop_cls=True)[0]
        assert torch.allclose(from_pixels, feat_item["vision_features"], atol=1e-6)


# --------------------------------------------------------------- the real manifests

REAL = "data/manifests"


@pytest.fixture(scope="module")
def real():
    if not os.path.exists(f"{REAL}/meta.json"):
        pytest.skip("real data not downloaded (run scripts/download_data.py)")
    return {name: read_jsonl(f"{REAL}/{name}.jsonl") for name in ("caption_train", "caption_eval", "vqa_train", "vqa_eval")}


def test_real_train_and_eval_images_are_disjoint(real):
    train = {r["image_id"] for r in real["caption_train"]}
    held_out = {r["image_id"] for r in real["caption_eval"]}
    assert not train & held_out
    assert {r["image_id"] for r in real["vqa_train"]} <= train
    assert {r["image_id"] for r in real["vqa_eval"]} <= held_out


def test_real_manifests_are_consistent(real):
    with open(f"{REAL}/meta.json") as f:
        counts = json.load(f)["counts"]
    assert counts["caption_train_images"] == len(real["caption_train"])
    assert counts["vqa_train_questions"] == len(real["vqa_train"])
    assert counts["vqa_eval_questions"] == len(real["vqa_eval"])
    assert all(len(r["captions"]) >= 5 for r in real["caption_train"] + real["caption_eval"])
    assert all(len(r["answers"]) == 10 for r in real["vqa_train"] + real["vqa_eval"])
    assert len({r["question_id"] for r in real["vqa_train"]}) == len(real["vqa_train"])
    assert all(os.path.exists(os.path.join("data", r["image"])) for r in real["caption_eval"])
