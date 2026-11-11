import pytest

from src.analysis.question_groups import GROUPS, question_group


@pytest.mark.parametrize("question, answer_type, expected", [
    ("How many people are there?", "number", "counting"),
    ("how many dogs?", "number", "counting"),
    ("What color is the bus?", "other", "colour"),
    ("Is the bus red? What colour?", "other", "colour"),
    ("Where is the cat?", "other", "spatial"),
    ("Is the dog on the left?", "yes/no", "spatial"),
    ("What is next to the table?", "other", "spatial"),
    ("Is this a real game?", "yes/no", "yes/no"),
    ("What sport is this?", "other", "other"),
    ("Why is he smiling?", "other", "other"),
])
def test_question_group(question, answer_type, expected):
    assert question_group(question, answer_type) == expected


def test_counting_wins_over_spatial_and_every_group_is_listed():
    assert question_group("How many people are on the left?", "number") == "counting"
    assert set(GROUPS) == {"counting", "colour", "spatial", "yes/no", "other"}
