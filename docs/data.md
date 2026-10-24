# Data

A COCO-captions + VQAv2 subset sharing one pool of images. Built by `scripts/download_data.py` (seeded, resumable); the files live under `data/` (git-ignored).

## Subsets

| Manifest | Contents | Size |
|---|---|---|
| `caption_train.jsonl` | image + 5 captions | 10,000 images |
| `vqa_train.jsonl` | image + question + answers | 20,000 questions (2 per image) |
| `caption_eval.jsonl` | image + 5 captions | 1,000 images |
| `vqa_eval.jsonl` | image + question + answers | 3,000 questions (3 per image) |

Images: 11,000 JPEGs in `data/images/<cocoid>.jpg`, 384 MB. They are resized so the short side is 224 (aspect ratio kept, never upsampled); CLIP preprocessing centre-crops to 224x224 at load time.

## Selection rules

- **Seed 0**, applied through `random.Random(seed)` after sorting by id, so the selection is reproducible.
- **Train images:** random sample from the Karpathy *train* split restricted to COCO train2014 (82k candidates). VQAv2 *train* questions cover exactly those images.
- **Eval images:** random sample from the Karpathy *val* split (val2014). VQAv2 *val* questions cover the same images, so one held-out image set serves both caption and VQA evaluation.
- **Questions:** up to 2 per train image and 3 per eval image, sampled at random from VQAv2.
- **Leakage check:** train and eval image ids are disjoint (asserted by the script, recorded as `train_eval_image_overlap: 0` in `manifests/meta.json`). All VQA records belong to images of their own pool.

## Record formats

`caption_*.jsonl`
```json
{"image_id": 574108, "image": "images/574108.jpg", "captions": ["A person riding skis ...", "..."]}
```

`vqa_*.jsonl`
```json
{"question_id": 354734006, "image_id": 354734, "image": "images/354734.jpg",
 "question": "Are the buildings in the background?", "answer": "yes",
 "answers": ["yes", "yes", "yes", "yes", "yes", "no", "yes", "yes", "yes", "yes"],
 "question_type": "are the", "answer_type": "yes/no"}
```

`answer` is VQAv2's most common answer (the training target). `answers` holds all 10 human answers, needed for the official soft-accuracy metric in Phase 4.

## Answer-type mix

| Split | yes/no | number | other |
|---|---|---|---|
| train (20,000) | 7,458 | 2,559 | 9,983 |
| eval (3,000) | 1,090 | 390 | 1,520 |

## Sources

- Caption text and COCO image URLs: `yerevann/coco-karpathy` (HF hub), images from `images.cocodataset.org`.
- VQAv2 questions and annotations: the official zips from visualqa.org (cached in `data/raw/`, 41 MB).
- Downloads use `certifi`'s CA bundle because the python.org build of Python here has no system certificates.

## Reproduce

```bash
python scripts/download_data.py                       # defaults above
python scripts/download_data.py --train-images 20 --eval-images 10 --out /tmp/data_smoke   # quick smoke test
```

## Feature cache

With a frozen encoder the ViT output for an image never changes, so `scripts/extract_features.py` encodes every image once and stores the result in `data/features/clip_b16_l-2_224/` (`src/data/feature_cache.py`).

| | |
|---|---|
| Features | CLIP ViT-B/16, layer -2, CLS dropped: 196 tokens (14x14 grid) x 768 per image |
| Storage | `features.npy` memmap, float16, 11,000 images, 3.31 GB (`--dtype float32` gives the exact fp32 features at twice the size) |
| Extraction | 197 s on the M4 Pro (56 img/s); resumable, and refuses to resume with different settings |
| Token counts | 196, 64 and 16 are all served from this cache through pooling in the VLM. 256 tokens needs a 21x21 grid from 336px images and is not covered |

Usage: `FeatureStore(dir)` behaves like a mapping `image_id -> tensor (196, 768)` and plugs into the datasets via `feature_store=`.

Checked against freshly computed fp32 features on 64 random images:

| | |
|---|---|
| Max abs difference | 0.031 (features reach 95; this is the fp16 rounding step at that magnitude) |
| Mean abs difference | 5.4e-5 (relative 1.8e-4) |
| Min per-token cosine similarity | 0.999999 |
| Time for a batch of 64 | on-the-fly (decode + transform + ViT) 999 ms, cached 7 ms: **142x faster** (15.6 vs 0.11 ms per image) |

Training on cached features therefore removes the image pipeline and the ViT forward from every step. The features are not bit-identical to fp32 (fp16 storage); the error is far below what matters for a projector's input.
