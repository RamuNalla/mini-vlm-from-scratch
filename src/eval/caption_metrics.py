"""
Caption metrics: corpus BLEU-1..4 and CIDEr-D, written to match pycocoevalcap's scorers.

The official COCO pipeline tokenises with the Stanford PTB tokenizer (Java), which is not available
here. `tokenize` is a simple lowercase word tokenizer with punctuation removed, close to what PTBTokenizer
leaves. Scores are comparable between our models, but not to published numbers to the last digit.

CIDEr needs a corpus: document frequencies come from the reference captions of all images, so scoring a
single image gives 0.
"""
import math
import re
from collections import Counter, defaultdict
from typing import Dict, List, Sequence, Tuple

import numpy as np

TOKEN_RE = re.compile(r"[a-z0-9]+(?:'[a-z]+)?")


def tokenize(text: str) -> List[str]:
    return TOKEN_RE.findall(text.lower())


def ngram_counts(tokens: Sequence[str], n: int = 4) -> Counter:
    counts = Counter()
    for k in range(1, n + 1):
        for i in range(len(tokens) - k + 1):
            counts[tuple(tokens[i:i + k])] += 1
    return counts


# ------------------------------------------------------------------------------ BLEU

def bleu(hypotheses: Sequence[str], references: Sequence[Sequence[str]], n: int = 4) -> List[float]:
    """Corpus BLEU-1..n (closest reference length, as pycocoevalcap). Returns [bleu1, ..., bleun]."""
    assert len(hypotheses) == len(references)
    tiny, small = 1e-15, 1e-9
    hyp_len = ref_len = 0
    guess = [0] * n
    correct = [0] * n

    for hyp, refs in zip(hypotheses, references):
        h = tokenize(hyp)
        ref_tokens = [tokenize(r) for r in refs]
        hyp_len += len(h)
        ref_len += min((abs(len(r) - len(h)), len(r)) for r in ref_tokens)[1]

        max_ref = Counter()
        for r in ref_tokens:
            for gram, c in ngram_counts(r, n).items():
                max_ref[gram] = max(max_ref[gram], c)
        for gram, c in ngram_counts(h, n).items():
            correct[len(gram) - 1] += min(c, max_ref[gram])
        for k in range(n):
            guess[k] += max(0, len(h) - k)

    scores, product = [], 1.0
    for k in range(n):
        product *= (correct[k] + tiny) / (guess[k] + small)
        scores.append(product ** (1.0 / (k + 1)))
    ratio = (hyp_len + tiny) / (ref_len + small)
    if ratio < 1:                                    # brevity penalty
        scores = [s * math.exp(1 - 1 / ratio) for s in scores]
    return scores


# ----------------------------------------------------------------------------- CIDEr-D

def cider_d(
    hypotheses: Sequence[str], references: Sequence[Sequence[str]], n: int = 4, sigma: float = 6.0
) -> Tuple[float, np.ndarray]:
    """CIDEr-D (corpus mean, per-image scores), as pycocoevalcap's Cider."""
    assert len(hypotheses) == len(references)
    hyp_counts = [ngram_counts(tokenize(h), n) for h in hypotheses]
    ref_counts = [[ngram_counts(tokenize(r), n) for r in refs] for refs in references]

    document_frequency: Dict[tuple, float] = defaultdict(float)
    for refs in ref_counts:
        for gram in {g for r in refs for g in r}:
            document_frequency[gram] += 1.0
    log_num_images = np.log(float(len(ref_counts)))

    def to_vector(counts):
        vec = [defaultdict(float) for _ in range(n)]
        norm = [0.0] * n
        length = 0
        for gram, tf in counts.items():
            df = np.log(max(1.0, document_frequency[gram]))
            k = len(gram) - 1
            vec[k][gram] = float(tf) * (log_num_images - df)
            norm[k] += vec[k][gram] ** 2
            if k == 1:                              # the reference code counts bigrams as the "length"
                length += tf
        return vec, [np.sqrt(x) for x in norm], length

    def similarity(vh, vr, nh, nr, lh, lr):
        delta = float(lh - lr)
        val = np.zeros(n)
        for k in range(n):
            for gram in vh[k]:
                val[k] += min(vh[k][gram], vr[k][gram]) * vr[k][gram]
            if nh[k] != 0 and nr[k] != 0:
                val[k] /= nh[k] * nr[k]
            val[k] *= np.e ** (-(delta ** 2) / (2 * sigma ** 2))
        return val

    scores = []
    for h_counts, refs in zip(hyp_counts, ref_counts):
        vh, nh, lh = to_vector(h_counts)
        total = np.zeros(n)
        for r_counts in refs:
            vr, nr, lr = to_vector(r_counts)
            total += similarity(vh, vr, nh, nr, lh, lr)
        scores.append(np.mean(total) / len(refs) * 10.0)
    scores = np.array(scores)
    return float(scores.mean()), scores


def caption_report(hypotheses: Sequence[str], references: Sequence[Sequence[str]]) -> Dict[str, float]:
    b = bleu(hypotheses, references)
    cider, _ = cider_d(hypotheses, references)
    return {
        "bleu1": b[0], "bleu2": b[1], "bleu3": b[2], "bleu4": b[3],
        "cider": cider,
        "avg_words": float(np.mean([len(tokenize(h)) for h in hypotheses])),
        "n": len(hypotheses),
    }
