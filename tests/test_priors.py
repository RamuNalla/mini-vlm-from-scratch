from src.analysis.priors import fit_prior, prior_answer


def test_prior_is_the_most_common_answer_per_question_type_with_a_fallback():
    train = [{"question_type": "how many", "answer": a} for a in ["2", "2", "3"]] + \
            [{"question_type": "is the", "answer": a} for a in ["yes", "no", "yes"]]
    prior = fit_prior(train)
    assert prior_answer(prior, "how many") == "2" and prior_answer(prior, "is the") == "yes"
    assert prior_answer(prior, "never seen") == prior[""] and prior[""] in {"2", "yes"}
