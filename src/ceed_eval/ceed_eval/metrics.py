"""The per-dataset answer metrics B0 through B2 are scored on.

Each source dataset has its own accepted metric, and using the wrong one makes
numbers incomparable to the literature:

* **DocVQA** — ANLS (average normalised Levenshtein similarity), which tolerates
  the OCR-level near-misses that exact match would punish.
* **GQA** — exact match after normalisation; GQA answers are short and closed.
* **ChartQA** — relaxed accuracy, which accepts a numeric answer within 5% of
  the gold value and falls back to exact match for non-numeric answers.

Every metric takes the model's prediction and the gold answers (a question may
have several) and returns a score in ``[0, 1]``; a question's score is the best
over its gold answers. These are pure functions over strings, so they are tested
in the fast tier and behave identically on CPU and GPU.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

from ceed_data import CHARTQA, DOCVQA, GQA

# Punctuation and articles are stripped before comparison so that "a cat." and
# "cat" are the same answer.
_ARTICLES = re.compile(r"\b(a|an|the)\b")
_PUNCTUATION = re.compile(r"[^\w\s.]")
# A period survives only between digits, so "10.5" keeps its decimal point while
# the full stop in "a cat." is dropped.
_NON_DECIMAL_PERIOD = re.compile(r"(?<!\d)\.|\.(?!\d)")
# Digit-grouping commas are removed before anything else, so "1,200" and "1200"
# are the same answer rather than "1 200" and "1200".
_GROUPING_COMMA = re.compile(r"(?<=\d),(?=\d)")
_WHITESPACE = re.compile(r"\s+")


def normalize_answer(text: str) -> str:
    """Return a comparison-ready form of an answer string.

    Lowercases, strips punctuation and articles, and collapses whitespace, so
    that formatting differences do not count as wrong answers.

    Args:
        text: The raw answer text.

    Returns:
        The normalised answer.
    """
    lowered = text.lower().strip()
    lowered = _GROUPING_COMMA.sub("", lowered)
    lowered = _PUNCTUATION.sub(" ", lowered)
    lowered = _NON_DECIMAL_PERIOD.sub(" ", lowered)
    lowered = _ARTICLES.sub(" ", lowered)
    return _WHITESPACE.sub(" ", lowered).strip()


def _levenshtein(a: str, b: str) -> int:
    """Return the edit distance between two strings."""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        current = [i]
        for j, cb in enumerate(b, start=1):
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (ca != cb)))
        previous = current
    return previous[-1]


def anls(prediction: str, golds: Sequence[str], threshold: float = 0.5) -> float:
    """Return the ANLS score of a prediction against its gold answers.

    Normalised Levenshtein similarity is computed against each gold answer and
    the best is taken; a similarity at or below ``threshold`` scores zero, which
    is the standard DocVQA convention preventing loosely-similar strings from
    accruing partial credit.

    Args:
        prediction: The model's answer.
        golds: The accepted answers.
        threshold: The similarity below which the score is zero.

    Returns:
        The ANLS score in ``[0, 1]``.
    """
    predicted = normalize_answer(prediction)
    best = 0.0
    for gold in golds:
        target = normalize_answer(gold)
        if not predicted and not target:
            best = max(best, 1.0)
            continue
        longest = max(len(predicted), len(target))
        if longest == 0:
            continue
        similarity = 1.0 - _levenshtein(predicted, target) / longest
        best = max(best, similarity)
    return best if best > threshold else 0.0


def exact_match(prediction: str, golds: Sequence[str]) -> float:
    """Return 1.0 if the normalised prediction equals any gold answer, else 0.0."""
    predicted = normalize_answer(prediction)
    return float(any(predicted == normalize_answer(gold) for gold in golds))


def _as_number(text: str) -> float | None:
    """Parse a numeric answer, tolerating percent signs, currency, and commas."""
    cleaned = text.strip().replace(",", "").replace("$", "").replace("%", "")
    try:
        return float(cleaned)
    except ValueError:
        return None


def relaxed_accuracy(prediction: str, golds: Sequence[str], tolerance: float = 0.05) -> float:
    """Return ChartQA's relaxed accuracy for a prediction.

    A numeric prediction counts as correct when it is within ``tolerance``
    (relative) of a numeric gold answer — chart values are read off axes, so
    demanding exact equality would measure transcription rather than
    comprehension. Non-numeric answers fall back to exact match.

    Args:
        prediction: The model's answer.
        golds: The accepted answers.
        tolerance: The permitted relative deviation for numeric answers.

    Returns:
        1.0 if the answer is accepted, else 0.0.
    """
    predicted_number = _as_number(prediction)
    for gold in golds:
        gold_number = _as_number(gold)
        if predicted_number is not None and gold_number is not None:
            if gold_number == 0.0:
                if abs(predicted_number) <= tolerance:
                    return 1.0
            elif abs(predicted_number - gold_number) / abs(gold_number) <= tolerance:
                return 1.0
        elif normalize_answer(prediction) == normalize_answer(gold):
            return 1.0
    return 0.0


#: The metric each dataset is scored with, so a Group cannot be scored two ways.
DATASET_METRICS = {
    DOCVQA: anls,
    GQA: exact_match,
    CHARTQA: relaxed_accuracy,
}


def score_answer(dataset: str, prediction: str, golds: Sequence[str]) -> float:
    """Score one prediction with the metric its dataset is reported under.

    Args:
        dataset: The source dataset name.
        prediction: The model's answer.
        golds: The accepted answers.

    Returns:
        The score in ``[0, 1]``.

    Raises:
        KeyError: If the dataset has no registered metric.
    """
    return DATASET_METRICS[dataset](prediction, golds)
