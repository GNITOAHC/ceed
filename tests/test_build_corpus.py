"""The corpus builder's ``--limit`` flag.

The loaders stop at ``index >= limit``, so a literal ``0`` reaching them means
*zero examples* — an empty corpus, written without complaint. Nobody asks for an
empty corpus and everybody eventually asks for a whole one, so a non-positive
limit is read as "take everything". This is worth a test because the failure it
prevents is silent: a corpus of nothing looks like a corpus.
"""

import importlib.util
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


def _build_corpus_module():
    """Load scripts/build_corpus.py, which is a CLI rather than a package member."""
    spec = importlib.util.spec_from_file_location(
        "build_corpus", REPO_ROOT / "scripts" / "build_corpus.py"
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


build_corpus = _build_corpus_module()


@pytest.mark.parametrize("limit", [0, -1, -1000])
def test_a_non_positive_limit_means_the_whole_source_split(limit):
    # None is what the loaders take for "no cap".
    assert build_corpus.resolve_limit(limit) is None


@pytest.mark.parametrize("limit", [1, 200, 100_000])
def test_a_positive_limit_is_passed_through_unchanged(limit):
    assert build_corpus.resolve_limit(limit) == limit


def test_the_default_limit_is_a_small_positive_cap():
    # The default must stay small: the sources are gigabytes, and an accidental
    # full pull is expensive rather than merely slow.
    default = build_corpus.parse_args([]).limit
    assert default > 0
    assert build_corpus.resolve_limit(default) == default


def test_the_flag_reaches_resolve_limit_from_the_command_line():
    assert build_corpus.resolve_limit(build_corpus.parse_args(["--limit", "0"]).limit) is None
    assert build_corpus.resolve_limit(build_corpus.parse_args(["--limit", "50"]).limit) == 50
