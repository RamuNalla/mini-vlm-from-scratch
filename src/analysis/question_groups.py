"""
Group VQAv2 questions by what they ask, for the failure analysis. The groups are mutually exclusive and
assigned by simple rules on the question text, in this order:

    counting  "how many ..."
    colour    mentions colour / color
    spatial   "where ..." or a position word (left, right, behind, next to, ...)
    yes/no    the remaining questions whose answers are yes or no
    other     everything else (what kind, what is, why, ...)
"""
import re

GROUPS = ["counting", "colour", "spatial", "yes/no", "other"]

SPATIAL_WORDS = re.compile(
    r"\b(left|right|behind|in front of|next to|beside|above|below|under|underneath|between|near|"
    r"top of|bottom of|on top|far|close to|closest|farthest|center|middle|corner)\b"
)


def question_group(question: str, answer_type: str) -> str:
    q = question.lower().strip()
    if q.startswith("how many"):
        return "counting"
    if re.search(r"\bcolou?rs?\b", q):
        return "colour"
    if q.startswith("where") or SPATIAL_WORDS.search(q):
        return "spatial"
    if answer_type == "yes/no":
        return "yes/no"
    return "other"
