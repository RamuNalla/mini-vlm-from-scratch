"""
Build the Phase 3 data: a COCO-captions + VQAv2 subset sharing one pool of images.

Train images come from the Karpathy *train* split restricted to COCO train2014 (VQAv2 train questions
cover exactly those images). Eval images come from the Karpathy *val* split (val2014, covered by VQAv2
val). The two pools are disjoint, so the same held-out images serve caption and VQA evaluation.

    python scripts/download_data.py                                    # defaults: 10k train / 1k eval images
    python scripts/download_data.py --train-images 20 --eval-images 10 --out /tmp/data_smoke   # smoke test

Outputs under --out:
    images/<cocoid>.jpg            short side resized to --size (CLIP preprocessing centre-crops later)
    manifests/caption_train.jsonl  {image_id, image, captions[5]}
    manifests/caption_eval.jsonl
    manifests/vqa_train.jsonl      {question_id, image_id, image, question, answer, answers[10], question_type, answer_type}
    manifests/vqa_eval.jsonl
    manifests/meta.json            config, counts, leakage check
Everything is seeded and the script is resumable (existing images and raw files are reused).
"""
import argparse
import json
import os
import random
import shutil
import ssl
import urllib.request
import zipfile
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

import certifi
from datasets import load_dataset
from PIL import Image
from tqdm import tqdm

VQA_BASE = "https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/"
VQA_FILES = {
    ("train", "questions"): "v2_Questions_Train_mscoco.zip",
    ("train", "annotations"): "v2_Annotations_Train_mscoco.zip",
    ("val", "questions"): "v2_Questions_Val_mscoco.zip",
    ("val", "annotations"): "v2_Annotations_Val_mscoco.zip",
}
IMAGE_URL = "http://images.cocodataset.org/{folder}/COCO_{folder}_{id:012d}.jpg"

# python.org builds of Python often lack system CA certs; use certifi's bundle (verification stays on)
SSL_CONTEXT = ssl.create_default_context(cafile=certifi.where())


def download(url: str, dest: str):
    with urllib.request.urlopen(url, context=SSL_CONTEXT, timeout=60) as r, open(dest, "wb") as f:
        shutil.copyfileobj(r, f)


def fetch(url: str, path: str):
    if os.path.exists(path):
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".part"
    download(url, tmp)
    os.replace(tmp, path)


def load_vqa(raw_dir: str, split: str):
    """Return {question_id: record} for one VQAv2 split, merging questions with annotations."""
    contents = {}
    for kind in ("questions", "annotations"):
        name = VQA_FILES[(split, kind)]
        path = os.path.join(raw_dir, name)
        fetch(VQA_BASE + name, path)
        with zipfile.ZipFile(path) as z:
            contents[kind] = json.loads(z.read(z.namelist()[0]))

    questions = {q["question_id"]: q for q in contents["questions"]["questions"]}
    merged = {}
    for a in contents["annotations"]["annotations"]:
        q = questions[a["question_id"]]
        merged[a["question_id"]] = {
            "question_id": a["question_id"],
            "image_id": a["image_id"],
            "question": q["question"],
            "answer": a["multiple_choice_answer"],
            "answers": [x["answer"] for x in a["answers"]],
            "question_type": a["question_type"],
            "answer_type": a["answer_type"],
        }
    return merged


def sample_vqa(vqa: dict, image_ids, per_image: int, rng: random.Random):
    by_image = defaultdict(list)
    for qid in sorted(vqa):                       # sorted -> deterministic regardless of dict order
        by_image[vqa[qid]["image_id"]].append(qid)
    rows = []
    for image_id in image_ids:
        qids = by_image.get(image_id, [])
        for qid in rng.sample(qids, min(per_image, len(qids))):
            rows.append(vqa[qid])
    return rows


def download_resize(job):
    url, path, size = job
    if os.path.exists(path):
        return True
    for _ in range(3):
        try:
            tmp = path + ".part"
            download(url, tmp)
            img = Image.open(tmp).convert("RGB")
            w, h = img.size
            scale = size / min(w, h)
            if scale < 1:                          # never upsample
                img = img.resize((round(w * scale), round(h * scale)), Image.BICUBIC)
            img.save(path, "JPEG", quality=95)
            os.remove(tmp)
            return True
        except Exception:
            continue
    return False


def write_jsonl(path, rows):
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="data")
    p.add_argument("--train-images", type=int, default=10000)
    p.add_argument("--eval-images", type=int, default=1000)
    p.add_argument("--vqa-per-image", type=int, default=2, help="questions sampled per train image")
    p.add_argument("--eval-vqa-per-image", type=int, default=3, help="questions sampled per eval image")
    p.add_argument("--size", type=int, default=224, help="short side of the stored images")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--workers", type=int, default=16)
    args = p.parse_args()

    rng = random.Random(args.seed)
    img_dir = os.path.join(args.out, "images")
    man_dir = os.path.join(args.out, "manifests")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(man_dir, exist_ok=True)

    # --- caption metadata (5 captions per image) and the two image pools
    print("loading COCO caption metadata ...")
    karpathy = load_dataset("yerevann/coco-karpathy")
    train_pool = [r for r in karpathy["train"] if r["filepath"] == "train2014"]
    eval_pool = [r for r in karpathy["validation"] if r["filepath"] == "val2014"]
    train_pool.sort(key=lambda r: r["cocoid"])
    eval_pool.sort(key=lambda r: r["cocoid"])
    train_imgs = rng.sample(train_pool, min(args.train_images, len(train_pool)))
    eval_imgs = rng.sample(eval_pool, min(args.eval_images, len(eval_pool)))

    # --- VQAv2 questions for exactly those images
    print("loading VQAv2 ...")
    raw_dir = os.path.join(args.out, "raw")
    vqa_train = sample_vqa(load_vqa(raw_dir, "train"), [r["cocoid"] for r in train_imgs], args.vqa_per_image, rng)
    vqa_eval = sample_vqa(load_vqa(raw_dir, "val"), [r["cocoid"] for r in eval_imgs], args.eval_vqa_per_image, rng)

    # --- download + resize the images (resumable)
    jobs = [
        (IMAGE_URL.format(folder=r["filepath"], id=r["cocoid"]), os.path.join(img_dir, f"{r['cocoid']}.jpg"), args.size)
        for r in train_imgs + eval_imgs
    ]
    with ThreadPoolExecutor(args.workers) as pool:
        ok = list(tqdm(pool.map(download_resize, jobs), total=len(jobs), desc="images"))
    failed = {int(os.path.basename(j[1])[:-4]) for j, good in zip(jobs, ok) if not good}
    if failed:
        print(f"WARNING: {len(failed)} images failed to download and are dropped: {sorted(failed)[:10]} ...")

    # --- manifests (drop anything whose image failed)
    rel = lambda i: f"images/{i}.jpg"
    caption = lambda r: {"image_id": r["cocoid"], "image": rel(r["cocoid"]), "captions": [s.strip() for s in r["sentences"]]}
    with_image = lambda row: dict(row, image=rel(row["image_id"]))
    cap_train = [caption(r) for r in train_imgs if r["cocoid"] not in failed]
    cap_eval = [caption(r) for r in eval_imgs if r["cocoid"] not in failed]
    vqa_tr = [with_image(r) for r in vqa_train if r["image_id"] not in failed]
    vqa_ev = [with_image(r) for r in vqa_eval if r["image_id"] not in failed]

    train_ids = {r["image_id"] for r in cap_train}
    eval_ids = {r["image_id"] for r in cap_eval}
    overlap = train_ids & eval_ids
    assert not overlap, f"train/eval image overlap: {sorted(overlap)[:5]}"
    assert {r["image_id"] for r in vqa_tr} <= train_ids and {r["image_id"] for r in vqa_ev} <= eval_ids

    write_jsonl(os.path.join(man_dir, "caption_train.jsonl"), cap_train)
    write_jsonl(os.path.join(man_dir, "caption_eval.jsonl"), cap_eval)
    write_jsonl(os.path.join(man_dir, "vqa_train.jsonl"), vqa_tr)
    write_jsonl(os.path.join(man_dir, "vqa_eval.jsonl"), vqa_ev)

    meta = {
        "config": vars(args),
        "counts": {
            "caption_train_images": len(cap_train), "caption_eval_images": len(cap_eval),
            "vqa_train_questions": len(vqa_tr), "vqa_eval_questions": len(vqa_ev),
            "failed_images": len(failed),
        },
        "train_eval_image_overlap": 0,
    }
    with open(os.path.join(man_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(json.dumps(meta["counts"], indent=2))


if __name__ == "__main__":
    main()
