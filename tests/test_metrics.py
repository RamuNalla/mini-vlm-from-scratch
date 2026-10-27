"""
VQA accuracy, BLEU, CIDEr-D and latency helpers. BLEU/CIDEr are also compared against pycocoevalcap's
scorers (pure Python; skipped when the package is not installed) on synthetic and real references.
"""
import math
import os

import numpy as np
import pytest
import torch

from src.eval.caption_metrics import bleu, caption_report, cider_d, tokenize
from src.eval.latency import LatencyTracker, count_generated_tokens
from src.eval.vqa_accuracy import (
    clean_vqa_prediction, evaluate_vqa, normalize_answer, process_digit_article, process_punctuation,
    vqa_soft_accuracy,
)


# ------------------------------------------------------------------ VQA normalisation

@pytest.mark.parametrize("raw,expected", [
    ("Two", "2"), ("the cat", "cat"), ("A dog", "dog"), ("dont", "don't"), ("none", "0"),
    ("ten", "10"), ("yes.", "yes"), ("red, white", "red white"), ("10,000", "10000"),
    ("hot-dog", "hot dog"), ("what?", "what"), ("3.5", "3.5"),
])
def test_normalize_answer(raw, expected):
    assert normalize_answer(raw) == expected


def test_punctuation_between_digits_is_kept_for_decimals():
    assert process_punctuation("3.5") == "3.5"
    assert process_punctuation("end.") == "end"


def test_digit_article_lowercases_and_drops_articles():
    assert process_digit_article("The Two Dogs") == "2 dogs"


@pytest.mark.parametrize("raw,expected", [
    ("Yes.", "yes"), ("  No  ", "no"), ("Two\nbecause there are two", "two"),
    ("a red car.", "a red car"), ("", ""), ("3", "3"),
])
def test_clean_vqa_prediction(raw, expected):
    assert clean_vqa_prediction(raw) == expected


# -------------------------------------------------------------------- VQA accuracy

def answers(n_match, match="no", other="yes", total=10):
    return [match] * n_match + [other] * (total - n_match)


@pytest.mark.parametrize("n_match,expected", [(0, 0.0), (1, 0.3), (2, 0.6), (3, 0.9), (4, 1.0), (10, 1.0)])
def test_soft_accuracy_follows_the_official_formula(n_match, expected):
    # 1 agreeing annotator -> 0.3, 2 -> 0.6, 3 -> 0.9, 4 or more -> 1.0
    assert vqa_soft_accuracy("no", answers(n_match)) == pytest.approx(expected)


def test_soft_accuracy_hand_computation():
    # 2 of 10 say "no": the two "no" annotators see 1 other match (1/3), the eight "yes" see 2 (2/3)
    assert vqa_soft_accuracy("no", answers(2)) == pytest.approx((2 * (1 / 3) + 8 * (2 / 3)) / 10)


def test_disagreeing_answers_normalise_the_prediction():
    gold = ["2"] * 6 + ["3"] * 4
    assert vqa_soft_accuracy("Two", gold) == pytest.approx(1.0)       # "Two" -> "2"
    assert vqa_soft_accuracy("the 3.", gold) == pytest.approx(1.0)    # article and period stripped


def test_unanimous_answers_are_compared_literally_like_the_official_script():
    assert vqa_soft_accuracy("Yes", ["yes"] * 10) == 0.0             # faithful quirk: no normalisation
    assert vqa_soft_accuracy(clean_vqa_prediction("Yes."), ["yes"] * 10) == 1.0


def test_wrong_answer_scores_zero():
    assert vqa_soft_accuracy("blue", ["red"] * 7 + ["orange"] * 3) == 0.0


def test_evaluate_vqa_breakdown():
    meta = [
        {"answers": ["yes"] * 10, "answer_type": "yes/no", "question_type": "is the"},
        {"answers": ["yes"] * 10, "answer_type": "yes/no", "question_type": "is the"},
        {"answers": ["2"] * 10, "answer_type": "number", "question_type": "how many"},
        {"answers": ["red"] * 6 + ["blue"] * 4, "answer_type": "other", "question_type": "what color is the"},
    ]
    preds = ["Yes.", "No", "2", "Red"]
    r = evaluate_vqa(preds, meta)
    assert r["n"] == 4 and r["accuracy"] == pytest.approx(100 * 3 / 4)
    assert r["by_answer_type"]["yes/no"] == {"accuracy": 50.0, "n": 2}
    assert r["by_answer_type"]["number"]["accuracy"] == 100.0
    assert r["by_question_type"]["how many"]["n"] == 1
    assert r["scores"] == [1.0, 0.0, 1.0, 1.0]


def test_evaluate_vqa_requires_one_prediction_per_question():
    with pytest.raises(AssertionError):
        evaluate_vqa(["yes"], [])


# ---------------------------------------------------------------- caption metrics

def test_tokenize():
    assert tokenize("A man's dog, running!") == ["a", "man's", "dog", "running"]
    assert tokenize("") == []


def test_bleu_perfect_and_disjoint():
    refs = [["a cat sits on the mat", "the cat is on the mat"], ["two dogs run in a park", "dogs run in the park"]]
    perfect = bleu([r[0] for r in refs], refs)
    assert all(p == pytest.approx(1.0, abs=1e-6) for p in perfect)
    disjoint = bleu(["xyz qrs tuv wxy zab", "mno pqr stu vwx yza"], refs)
    assert all(d < 1e-6 for d in disjoint)


def test_bleu_brevity_penalty():
    refs = [["a cat sits on the mat today"]]
    full = bleu(["a cat sits on the mat today"], refs)
    short = bleu(["a cat sits"], refs)
    assert short[0] < 1.0 and short[0] == pytest.approx(math.exp(1 - 7 / 3), rel=1e-6)   # all unigrams match
    assert full[0] > short[0]


def test_bleu_clips_repeated_words():
    refs = [["the cat"]]
    assert bleu(["the the the the"], refs, n=1)[0] == pytest.approx(0.25, abs=1e-6)       # "the" counted once


def test_cider_prefers_matching_captions():
    refs = [
        ["a cat sits on a mat", "a cat on the mat"],
        ["two dogs run in a park", "dogs running through the park"],
        ["a plane flies over the sea", "an airplane in the sky above water"],
    ]
    good, _ = cider_d([r[0] for r in refs], refs)
    shuffled = [refs[1][0], refs[2][0], refs[0][0]]
    bad, _ = cider_d(shuffled, refs)
    assert good > 1.0 and bad < 0.2 and good > bad


def test_cider_needs_a_corpus():
    score, _ = cider_d(["a cat"], [["a cat"]])
    assert score == 0.0                                  # one image: idf is log(1) = 0 for everything


def test_caption_report_fields():
    refs = [["a cat sits on a mat", "a cat on the mat"], ["two dogs run in a park", "dogs in the park"]]
    r = caption_report(["a cat on a mat", "two dogs in a park"], refs)
    assert set(r) == {"bleu1", "bleu2", "bleu3", "bleu4", "cider", "avg_words", "n"}
    assert r["n"] == 2 and 0 < r["bleu4"] <= r["bleu1"] <= 1


# ----------------------------------------------------- against the reference scorers

def reference_scores(hyps, refs):
    pytest.importorskip("pycocoevalcap.cider.cider")
    from pycocoevalcap.bleu.bleu import Bleu
    from pycocoevalcap.cider.cider import Cider

    tok = lambda s: " ".join(tokenize(s))
    gts = {i: [tok(r) for r in rs] for i, rs in enumerate(refs)}
    res = {i: [tok(h)] for i, h in enumerate(hyps)}
    bleus, _ = Bleu(4).compute_score(gts, res, verbose=0)
    cider, per_image = Cider().compute_score(gts, res)
    return bleus, cider, per_image


def check_against_reference(hyps, refs):
    ref_bleu, ref_cider, ref_per_image = reference_scores(hyps, refs)
    assert bleu(hyps, refs) == pytest.approx(ref_bleu, rel=1e-6, abs=1e-9)
    cider, per_image = cider_d(hyps, refs)
    assert cider == pytest.approx(ref_cider, rel=1e-9)
    assert np.allclose(per_image, ref_per_image, rtol=1e-9)


def test_matches_pycocoevalcap_on_synthetic_captions():
    rng = np.random.default_rng(0)
    vocab = "man woman dog cat red blue car tree sits stands runs on in a the with holding next to".split()
    sentence = lambda: " ".join(rng.choice(vocab, rng.integers(4, 12)))
    refs = [[sentence() for _ in range(5)] for _ in range(60)]
    hyps = [refs[i][0] if i % 2 == 0 else sentence() for i in range(60)]     # half exact, half random
    check_against_reference(hyps, refs)


def test_matches_pycocoevalcap_on_real_references():
    path = "data/manifests/caption_eval.jsonl"
    if not os.path.exists(path):
        pytest.skip("real data not downloaded")
    from src.data.manifest_dataset import read_jsonl
    rows = read_jsonl(path)[:200]
    refs = [r["captions"] for r in rows]
    hyps = [refs[i][1] if i % 3 else refs[(i + 1) % len(refs)][0] for i in range(len(refs))]
    check_against_reference(hyps, refs)


# --------------------------------------------------------------------------- latency

def test_count_generated_tokens_stops_at_eos():
    eos, pad = 9, 0
    generated = torch.tensor([[5, 6, 9, 0, 0], [5, 6, 7, 8, 9], [5, 6, 7, 8, 5]])
    assert count_generated_tokens(generated, eos, pad) == 3 + 5 + 5


def test_latency_tracker_accumulates():
    tracker = LatencyTracker("cpu")
    with tracker.measure(4) as m:
        sum(range(100000))
        m.tokens = 20
    with tracker.measure(2) as m:
        m.tokens = 10
    s = tracker.summary()
    assert s["samples"] == 6 and s["generated_tokens"] == 30 and s["seconds"] > 0
    assert s["ms_per_sample"] == pytest.approx(1000 * s["seconds"] / 6)
    assert s["tokens_per_second"] == pytest.approx(30 / s["seconds"])
    assert math.isnan(s["memory_gb"])                    # CPU
