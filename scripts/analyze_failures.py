"""
Per-question-group VQA accuracy for several saved evaluations (no model runs).

    python -m scripts.analyze_failures      # prints the table, writes results/failure_analysis/by_group.{json,md}

Reads results/<name>/vqa_predictions.jsonl, whose rows already hold each question's soft accuracy.
"""
import json
import math
import os

from src.analysis.question_groups import GROUPS, question_group

SYSTEMS = {
    "text-only": "results/text_only",
    "full recipe": "results/stage2",
    "ref (reduced)": "results/exp/ref",
    "16 tokens": "results/exp/tokens_16",
    "MLP": "results/exp/proj_mlp",
}
OUT = "results/failure_analysis"


def load_rows(directory):
    with open(os.path.join(directory, "vqa_predictions.jsonl")) as f:
        return [json.loads(line) for line in f]


def group_stats(rows):
    """{group: (n, mean accuracy in %, standard error in points)} plus an 'all' entry."""
    buckets = {g: [] for g in GROUPS}
    for r in rows:
        buckets[question_group(r["question"], r["answer_type"])].append(r["score"])
    buckets["all"] = [r["score"] for r in rows]
    out = {}
    for g, scores in buckets.items():
        n = len(scores)
        mean = sum(scores) / n
        var = sum((s - mean) ** 2 for s in scores) / max(1, n - 1)
        out[g] = {"n": n, "accuracy": 100 * mean, "se": 100 * math.sqrt(var / n)}
    return out


def main():
    stats = {name: group_stats(load_rows(path)) for name, path in SYSTEMS.items()}
    cols = GROUPS + ["all"]
    lines = ["| System | " + " | ".join(f"{g} (n={stats['full recipe'][g]['n']})" for g in cols) + " |",
             "|---|" + "---|" * len(cols)]
    for name in SYSTEMS:
        lines.append(f"| {name} | " + " | ".join(f"{stats[name][g]['accuracy']:.1f} ± {stats[name][g]['se']:.1f}" for g in cols) + " |")
    lines.append("")
    lines.append("| Gain over text-only (points) | " + " | ".join(cols) + " |")
    lines.append("|---|" + "---|" * len(cols))
    for name in SYSTEMS:
        if name != "text-only":
            lines.append(f"| {name} | " + " | ".join(f"{stats[name][g]['accuracy'] - stats['text-only'][g]['accuracy']:+.1f}" for g in cols) + " |")
    text = "\n".join(lines)
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "by_group.json"), "w") as f:
        json.dump(stats, f, indent=2)
    with open(os.path.join(OUT, "by_group.md"), "w") as f:
        f.write(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
