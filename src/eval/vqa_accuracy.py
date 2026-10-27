"""
VQAv2 soft accuracy, following the official evaluation script (vqaEval.py).

For every question there are 10 human answers. A prediction scores min(1, matches / 3), averaged over
the 10 ways of leaving one human answer out, so full credit needs 4 agreeing annotators and a single
agreeing annotator gives 0.3.

The official script normalises text only when the 10 answers disagree (punctuation, number words,
articles, contractions) and compares unanimous answers literally. `vqa_soft_accuracy` reproduces that,
including the oddities (e.g. capitalised contraction keys never match because words are lowercased
first). Model outputs are therefore cleaned first with `clean_vqa_prediction`, as evaluation harnesses
for generative models do.
"""
import re
from collections import defaultdict
from typing import Dict, List, Sequence

CONTRACTIONS = {
    "aint": "ain't", "arent": "aren't", "cant": "can't", "couldve": "could've", "couldnt": "couldn't",
    "couldn'tve": "couldn't've", "couldnt've": "couldn't've", "didnt": "didn't", "doesnt": "doesn't",
    "dont": "don't", "hadnt": "hadn't", "hadnt've": "hadn't've", "hadn'tve": "hadn't've", "hasnt": "hasn't",
    "havent": "haven't", "hed": "he'd", "hed've": "he'd've", "he'dve": "he'd've", "hes": "he's",
    "howd": "how'd", "howll": "how'll", "hows": "how's", "Id've": "I'd've", "I'dve": "I'd've", "Im": "I'm",
    "Ive": "I've", "isnt": "isn't", "itd": "it'd", "itd've": "it'd've", "it'dve": "it'd've", "itll": "it'll",
    "let's": "let's", "maam": "ma'am", "mightnt": "mightn't", "mightnt've": "mightn't've",
    "mightn'tve": "mightn't've", "mightve": "might've", "mustnt": "mustn't", "mustve": "must've",
    "neednt": "needn't", "notve": "not've", "oclock": "o'clock", "oughtnt": "oughtn't",
    "ow's'at": "'ow's'at", "'ows'at": "'ow's'at", "'ow'sat": "'ow's'at", "shant": "shan't",
    "shed've": "she'd've", "she'dve": "she'd've", "she's": "she's", "shouldve": "should've",
    "shouldnt": "shouldn't", "shouldnt've": "shouldn't've", "shouldn'tve": "shouldn't've",
    "somebody'd": "somebodyd", "somebodyd've": "somebody'd've", "somebody'dve": "somebody'd've",
    "somebodyll": "somebody'll", "somebodys": "somebody's", "someoned": "someone'd",
    "someoned've": "someone'd've", "someone'dve": "someone'd've", "someonell": "someone'll",
    "someones": "someone's", "somethingd": "something'd", "somethingd've": "something'd've",
    "something'dve": "something'd've", "somethingll": "something'll", "thats": "that's",
    "thered": "there'd", "thered've": "there'd've", "there'dve": "there'd've", "therere": "there're",
    "theres": "there's", "theyd": "they'd", "theyd've": "they'd've", "they'dve": "they'd've",
    "theyll": "they'll", "theyre": "they're", "theyve": "they've", "twas": "'twas", "wasnt": "wasn't",
    "wed've": "we'd've", "we'dve": "we'd've", "weve": "we've", "werent": "weren't", "whatll": "what'll",
    "whatre": "what're", "whats": "what's", "whatve": "what've", "whens": "when's", "whered": "where'd",
    "wheres": "where's", "whereve": "where've", "whod": "who'd", "whod've": "who'd've",
    "who'dve": "who'd've", "wholl": "who'll", "whos": "who's", "whove": "who've", "whyll": "why'll",
    "whyre": "why're", "whys": "why's", "wont": "won't", "wouldve": "would've", "wouldnt": "wouldn't",
    "wouldnt've": "wouldn't've", "wouldn'tve": "wouldn't've", "yall": "y'all", "yall'll": "y'all'll",
    "y'allll": "y'all'll", "yall'd've": "y'all'd've", "y'alld've": "y'all'd've", "y'all'dve": "y'all'd've",
    "youd": "you'd", "youd've": "you'd've", "you'dve": "you'd've", "youll": "you'll", "youre": "you're",
    "youve": "you've",
}
NUMBER_WORDS = {
    "none": "0", "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4", "five": "5",
    "six": "6", "seven": "7", "eight": "8", "nine": "9", "ten": "10",
}
ARTICLES = ["a", "an", "the"]
PUNCT = [";", r"/", "[", "]", '"', "{", "}", "(", ")", "=", "+", "\\", "_", "-", ">", "<", "@", "`", ",", "?", "!"]
PERIOD_STRIP = re.compile(r"(?!<=\d)(\.)(?!\d)")
COMMA_STRIP = re.compile(r"(\d)(,)(\d)")


def process_punctuation(text: str) -> str:
    out = text
    for p in PUNCT:
        if (p + " " in text or " " + p in text) or (re.search(COMMA_STRIP, text) is not None):
            out = out.replace(p, "")
        else:
            out = out.replace(p, " ")
    # the official script passes re.UNICODE as `count` (a bug); kept for identical results
    return PERIOD_STRIP.sub("", out, re.UNICODE)


def process_digit_article(text: str) -> str:
    words = []
    for word in text.lower().split():
        word = NUMBER_WORDS.get(word, word)
        if word not in ARTICLES:
            words.append(word)
    return " ".join(CONTRACTIONS.get(w, w) for w in words)


def normalize_answer(text: str) -> str:
    return process_digit_article(process_punctuation(text))


def clean_vqa_prediction(text: str) -> str:
    """Make a generated answer comparable: first line, lowercase, no surrounding space or final period."""
    text = text.strip().split("\n")[0].strip().lower()
    return text[:-1].strip() if text.endswith(".") else text


def vqa_soft_accuracy(prediction: str, answers: Sequence[str]) -> float:
    """Official VQA accuracy of one prediction against the 10 human answers."""
    prediction = prediction.replace("\n", " ").replace("\t", " ").strip()
    answers = list(answers)
    if len(set(answers)) > 1:                       # the official script only normalises on disagreement
        answers = [process_punctuation(a) for a in answers]
        prediction = process_digit_article(process_punctuation(prediction))

    scores = []
    for i in range(len(answers)):
        matches = sum(1 for j, a in enumerate(answers) if j != i and a == prediction)
        scores.append(min(1.0, matches / 3))
    return sum(scores) / len(scores)


def evaluate_vqa(predictions: Sequence[str], meta: Sequence[dict], clean: bool = True) -> Dict[str, object]:
    """
    Accuracy overall and per answer type / question type.

    predictions: generated answers; meta: dataset `meta` dicts with `answers`, `answer_type`, `question_type`.
    Returns percentages (0-100) plus counts, and per-question scores under "scores".
    """
    assert len(predictions) == len(meta), "one prediction per question"
    scores = []
    by_answer, by_question = defaultdict(list), defaultdict(list)
    for pred, m in zip(predictions, meta):
        s = vqa_soft_accuracy(clean_vqa_prediction(pred) if clean else pred, m["answers"])
        scores.append(s)
        by_answer[m["answer_type"]].append(s)
        by_question[m["question_type"]].append(s)

    pct = lambda xs: 100.0 * sum(xs) / len(xs) if xs else float("nan")
    return {
        "accuracy": pct(scores),
        "n": len(scores),
        "by_answer_type": {k: {"accuracy": pct(v), "n": len(v)} for k, v in sorted(by_answer.items())},
        "by_question_type": {k: {"accuracy": pct(v), "n": len(v)} for k, v in sorted(by_question.items())},
        "scores": scores,
    }
