"""
Language-prior baseline for VQA: answer every question with the most common training answer for its question type
("how many" -> "2", "is the" -> "yes", ...). It sees no image, so whatever a model scores above it is not explained
by the question wording alone.
"""
from collections import Counter, defaultdict
from typing import Dict, Sequence


def fit_prior(train_records: Sequence[dict]) -> Dict[str, str]:
    """question_type -> most common training answer, plus "" -> the overall most common answer (the fallback)."""
    by_type = defaultdict(Counter)
    overall = Counter()
    for r in train_records:
        by_type[r["question_type"]][r["answer"]] += 1
        overall[r["answer"]] += 1
    prior = {qt: c.most_common(1)[0][0] for qt, c in by_type.items()}
    prior[""] = overall.most_common(1)[0][0]
    return prior


def prior_answer(prior: Dict[str, str], question_type: str) -> str:
    return prior.get(question_type, prior[""])
