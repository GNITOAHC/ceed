"""The per-dataset answer metrics every Group is scored by.

The Phase 1 comparison is only meaningful if all arms are scored identically, so
these metrics are pinned here rather than left to the harness. Each dataset's
accepted convention is asserted at both ends — what must count as correct, and
what must not — because a metric that is merely lenient would manufacture the
small effects the plan is trying to measure.
"""

import pytest

from ceed_eval import anls, exact_match, normalize_answer, relaxed_accuracy, score_answer

# -- normalisation -----------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("A cat.", "cat"),
        ("  THE Dog  ", "dog"),
        ("an apple!", "apple"),
        ("10.5", "10.5"),  # a decimal point survives
        ("$1,200", "1200"),  # digit-grouping commas go, so 1,200 == 1200
    ],
)
def test_normalisation_strips_formatting_but_keeps_decimals(raw, expected):
    assert normalize_answer(raw) == expected


def test_a_grouped_and_ungrouped_number_are_the_same_answer():
    assert exact_match("1,200", ["1200"]) == 1.0


# -- DocVQA: ANLS ------------------------------------------------------------


def test_anls_is_one_for_an_exact_answer():
    assert anls("Total: 42", ["Total: 42"]) == 1.0


def test_anls_gives_partial_credit_for_a_near_miss():
    score = anls("Totl 42", ["Total 42"])
    assert 0.5 < score < 1.0


def test_anls_is_zero_below_the_similarity_threshold():
    # A wholly different string earns nothing, rather than a small similarity.
    assert anls("elephant", ["Total 42"]) == 0.0


def test_anls_takes_the_best_of_several_gold_answers():
    assert anls("blue", ["red", "blue", "green"]) == 1.0


# -- GQA: exact match --------------------------------------------------------


def test_exact_match_ignores_articles_and_punctuation():
    assert exact_match("A cat.", ["cat"]) == 1.0


def test_exact_match_rejects_a_near_miss():
    # GQA answers are short and closed, so "cats" is simply wrong.
    assert exact_match("cats", ["cat"]) == 0.0


# -- ChartQA: relaxed accuracy ----------------------------------------------


def test_relaxed_accuracy_accepts_a_value_within_five_percent():
    assert relaxed_accuracy("10.2", ["10.0"]) == 1.0


def test_relaxed_accuracy_rejects_a_value_outside_five_percent():
    assert relaxed_accuracy("12.0", ["10.0"]) == 0.0


def test_relaxed_accuracy_tolerates_currency_and_percent_formatting():
    assert relaxed_accuracy("$1,000", ["1000"]) == 1.0
    assert relaxed_accuracy("45%", ["45"]) == 1.0


def test_relaxed_accuracy_falls_back_to_exact_match_for_text():
    assert relaxed_accuracy("Canada", ["canada"]) == 1.0
    assert relaxed_accuracy("Mexico", ["canada"]) == 0.0


def test_a_zero_gold_value_is_compared_absolutely():
    # Relative tolerance is undefined at zero; the metric must not divide by it.
    assert relaxed_accuracy("0.0", ["0"]) == 1.0
    assert relaxed_accuracy("5", ["0"]) == 0.0


# -- the dataset -> metric routing ------------------------------------------


def test_each_dataset_is_scored_by_its_own_convention():
    # The same near-miss scores differently per dataset, which is the point.
    assert score_answer("docvqa", "Totl 42", ["Total 42"]) > 0.0
    assert score_answer("gqa", "cats", ["cat"]) == 0.0
    assert score_answer("chartqa", "10.2", ["10.0"]) == 1.0


def test_an_unknown_dataset_is_refused_rather_than_silently_scored():
    with pytest.raises(KeyError):
        score_answer("imagenet", "cat", ["cat"])
