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


# -- per-dataset limits -----------------------------------------------------
#
# One cap cannot suit three sources that differ by three orders of magnitude:
# ChartQA's usable split is ~2.5k rows, GQA's balanced train set is ~1M. Asking
# for "everything" from all three is a corpus nobody wanted and days of teacher
# extraction.


def test_without_overrides_every_dataset_takes_the_shared_limit():
    limits = build_corpus.resolve_dataset_limits(["docvqa", "gqa"], 200, [])
    assert limits == {"docvqa": 200, "gqa": 200}


def test_an_override_applies_to_only_the_dataset_it_names():
    limits = build_corpus.resolve_dataset_limits(["docvqa", "gqa", "chartqa"], 0, ["gqa=5000"])
    assert limits == {"docvqa": None, "gqa": 5000, "chartqa": None}


def test_an_override_of_zero_means_the_whole_source_split():
    limits = build_corpus.resolve_dataset_limits(["docvqa", "gqa"], 200, ["docvqa=0"])
    assert limits["docvqa"] is None
    assert limits["gqa"] == 200


def test_an_override_naming_a_dataset_not_being_built_is_refused():
    # Silently ignoring it would build the corpus with the wrong size for the
    # dataset the caller thought they were capping.
    with pytest.raises(SystemExit, match="chartqa"):
        build_corpus.resolve_dataset_limits(["docvqa"], 200, ["chartqa=10"])


def test_a_malformed_override_is_refused():
    with pytest.raises(SystemExit, match="NAME="):
        build_corpus.resolve_dataset_limits(["docvqa"], 200, ["docvqa=lots"])


def test_the_overrides_reach_resolve_from_the_command_line():
    args = build_corpus.parse_args(["--limit", "0", "--dataset-limit", "gqa=5000"])
    limits = build_corpus.resolve_dataset_limits(args.datasets, args.limit, args.dataset_limit)
    assert limits == {"docvqa": None, "gqa": 5000, "chartqa": None}


# -- a source that yields nothing is refused, not dropped -------------------


def test_chartqa_is_drawn_from_the_split_its_source_actually_publishes():
    # lmms-lab/ChartQA has no train split. Naming one meant every three-dataset
    # corpus was silently built without ChartQA.
    assert build_corpus.SOURCE_SPLITS["chartqa"] == "test"


def _stub_loaders(monkeypatch, working: str, empty: str) -> None:
    """Replace the streaming loaders so the build runs without downloading."""
    from ceed_data import make_docvqa_example

    def yields_examples(split, image_store, limit=None):
        for index in range(limit or 8):
            yield make_docvqa_example(
                source_id=f"{index}",
                question="q",
                answers=["a"],
                image_fingerprint="fp",
                ocr_words=[],
            )

    monkeypatch.setitem(build_corpus.LOADERS, working, yields_examples)
    monkeypatch.setitem(build_corpus.LOADERS, empty, lambda *a, **k: iter(()))


def test_a_dataset_that_contributes_nothing_fails_the_build(tmp_path, monkeypatch, capsys):
    _stub_loaders(monkeypatch, working="docvqa", empty="gqa")
    code = build_corpus.main(
        ["--output", str(tmp_path), "--datasets", "docvqa", "gqa", "--limit", "8"]
    )
    assert code == 1
    assert "gqa" in capsys.readouterr().err
    # Nothing may be written: a corpus on disk is taken as the corpus that was
    # asked for by everything downstream of it.
    assert not (tmp_path / "corpus_manifest.json").exists()


def test_allow_partial_writes_the_corpus_without_the_failed_dataset(tmp_path, monkeypatch):
    _stub_loaders(monkeypatch, working="docvqa", empty="gqa")
    code = build_corpus.main(
        [
            "--output",
            str(tmp_path),
            "--datasets",
            "docvqa",
            "gqa",
            "--limit",
            "8",
            "--allow-partial",
        ]
    )
    assert code == 0
    assert (tmp_path / "corpus_manifest.json").exists()
