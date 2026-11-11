"""
Language-prior check (no model runs): how much of the model's accuracy a question-type prior already explains.

    python -m scripts.analyze_priors      # writes results/failure_analysis/priors.{json,md}

For each question group, compares the full-recipe model with the text-only model and with a prior that answers every
question with the most common training answer for its question type, and measures how often the model's answer is
just that prior answer.
"""
import json
import os
from collections import Counter

from src.analysis.priors import fit_prior, prior_answer
from src.analysis.question_groups import GROUPS, question_group
from src.data.manifest_dataset import read_jsonl
from src.eval.vqa_accuracy import vqa_soft_accuracy

MODEL = "results/stage2"
OUT = "results/failure_analysis"


def main():
    train = read_jsonl("data/manifests/vqa_train.jsonl")
    prior = fit_prior(train)
    answers = {r["question_id"]: r["answers"] for r in read_jsonl("data/manifests/vqa_eval.jsonl")}
    rows = read_jsonl(os.path.join(MODEL, "vqa_predictions.jsonl"))
    text_only = {r["question_id"]: r["score"] for r in read_jsonl("results/text_only/vqa_predictions.jsonl")}

    stats = {g: {"n": 0, "model": 0.0, "text_only": 0.0, "prior": 0.0, "same_as_prior": 0, "model_when_same": 0.0,
                 "model_when_differs": 0.0, "answers": Counter(), "truth": Counter()} for g in GROUPS}
    for r in rows:
        g = question_group(r["question"], r["answer_type"])
        s = stats[g]
        p = prior_answer(prior, r["question_type"])
        prior_score = vqa_soft_accuracy(p, answers[r["question_id"]], normalize="always")
        same = r["cleaned"] == p
        s["n"] += 1
        s["model"] += r["score"]
        s["text_only"] += text_only[r["question_id"]]
        s["prior"] += prior_score
        s["same_as_prior"] += same
        s["model_when_same" if same else "model_when_differs"] += r["score"]
        s["answers"][r["cleaned"]] += 1
        s["truth"][r["answer"]] += 1

    summary, lines = {}, ["| Group | n | model | text-only | question-type prior | model answer = prior answer | model acc. when same / differs | top model answers |",
                          "|---|---|---|---|---|---|---|---|"]
    for g, s in stats.items():
        n, same = s["n"], s["same_as_prior"]
        row = {"n": n, "model": 100 * s["model"] / n, "text_only": 100 * s["text_only"] / n, "prior": 100 * s["prior"] / n,
               "same_as_prior_pct": 100 * same / n,
               "acc_when_same": 100 * s["model_when_same"] / max(1, same),
               "acc_when_differs": 100 * s["model_when_differs"] / max(1, n - same),
               "top_answers": s["answers"].most_common(5), "top_truth": s["truth"].most_common(5),
               "distinct_answers": len(s["answers"])}
        summary[g] = row
        top = ", ".join(f"{a} ({c})" for a, c in row["top_answers"][:4])
        lines.append(f"| {g} | {n} | {row['model']:.1f} | {row['text_only']:.1f} | {row['prior']:.1f} | {row['same_as_prior_pct']:.0f}% | "
                     f"{row['acc_when_same']:.1f} / {row['acc_when_differs']:.1f} | {top} |")
    text = "\n".join(lines)
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "priors.json"), "w") as f:
        json.dump(summary, f, indent=2)
    with open(os.path.join(OUT, "priors.md"), "w") as f:
        f.write(text + "\n")
    print(text)
    for g in ("counting", "yes/no"):
        print(f"\n{g}: top ground-truth answers {summary[g]['top_truth']}, distinct model answers {summary[g]['distinct_answers']}")


if __name__ == "__main__":
    main()
