"""A run's identity must cover the corpus it read, not just the corpus's name.

``CorpusConfig.name`` is a constant written into every Group's overlay
(``ceed-vqa``). Before the fingerprint existed, that made two runs over entirely
different corpora — DocVQA alone and DocVQA + GQA + ChartQA — hash to the same
value, and the consequences were silent rather than loud:

* ``runs/<hash>/run_record.json`` was overwritten, so the first corpus's result
  was replaced by the second's under a hash that claimed to describe both;
* the checkpoint directory is keyed by :func:`resume_key`, which is that same
  hash with the step budget normalised away, so the second run found a
  ``progress.json`` written by a matching configuration, adopted it, and — being
  already at its step budget — trained **zero steps** before being scored on the
  new corpus and reported as a new result.

The resume guard could not catch it: it compares configurations, and the
configurations really were identical. The corpus was the thing that differed, and
nothing in the configuration said so.

These tests pin the fix at the level the failure occurred: two corpora, two
hashes, two checkpoint directories.
"""

from pathlib import Path

import pytest
from ceed_core.config import GroupConfig, run_hash
from ceed_data.manifest import CorpusManifest, corpus_fingerprint, with_corpus_fingerprint
from ceed_student.training import resume_key

from ceed_data import make_chartqa_example, make_docvqa_example, split_corpus

FRACTIONS = {"train": 0.8, "validation": 0.1, "test": 0.1}


def a_manifest(datasets: tuple[str, ...] = ("docvqa",), n: int = 40) -> CorpusManifest:
    """Build a manifest over a synthetic corpus drawn from ``datasets``."""
    examples = []
    for dataset in datasets:
        for index in range(n):
            if dataset == "docvqa":
                examples.append(
                    make_docvqa_example(
                        source_id=f"{index}",
                        question="q",
                        answers=["a"],
                        image_fingerprint="fp",
                        ocr_words=[],
                    )
                )
            else:
                examples.append(
                    make_chartqa_example(
                        source_id=f"{index}",
                        question="q",
                        answers=["a"],
                        image_fingerprint="fp",
                    )
                )
    splits = split_corpus(examples, seed=0, fractions=FRACTIONS)
    return CorpusManifest.from_splits(splits, seed=0, fractions=FRACTIONS)


def a_config(fingerprint: str | None) -> GroupConfig:
    """A minimal training Group carrying ``fingerprint`` as its corpus identity."""
    return GroupConfig.model_validate(
        {
            "group_code": "B2",
            "seed": 0,
            "param_efficiency": "lora",
            "layer_mapping": {"kind": "proportional", "pairs": [[9, 12], [29, 39]]},
            "student": {"model": "gemma-4-E4B-it", "dtype": "float16"},
            "extraction": {
                "teacher_model": "gemma-4-26B-A4B-it",
                "model_revision": "main",
                "layers": [9, 19, 29],
                "ablation": "mean-of-active",
                "combine_weight": "effective",
            },
            "corpus": {"name": "ceed-vqa", "splits": ["train"], "fingerprint": fingerprint},
            "auxiliary_signals": [],
            "training": {
                "steps": 2000,
                "learning_rate": 1.0e-4,
                "batch_size": 8,
                "backbone": {"kd_weight": 1.0, "kd_temperature": 2.0},
            },
        }
    )


# -- the fingerprint describes the corpus -----------------------------------


def test_the_same_corpus_fingerprints_the_same_way():
    assert a_manifest().fingerprint() == a_manifest().fingerprint()


def test_a_corpus_with_different_datasets_fingerprints_differently():
    docvqa_only = a_manifest(("docvqa",))
    all_sources = a_manifest(("docvqa", "chartqa"))
    assert docvqa_only.fingerprint() != all_sources.fingerprint()


def test_a_corpus_with_different_examples_fingerprints_differently():
    assert a_manifest(n=40).fingerprint() != a_manifest(n=41).fingerprint()


def test_the_fingerprint_covers_which_split_an_example_landed_in():
    # Same examples, different seed: the membership moves, and a run trained on
    # one is not a run trained on the other.
    examples = [
        make_docvqa_example(
            source_id=f"{i}", question="q", answers=["a"], image_fingerprint="fp", ocr_words=[]
        )
        for i in range(40)
    ]
    first = CorpusManifest.from_splits(
        split_corpus(examples, seed=0, fractions=FRACTIONS), seed=0, fractions=FRACTIONS
    )
    second = CorpusManifest.from_splits(
        split_corpus(examples, seed=7, fractions=FRACTIONS), seed=7, fractions=FRACTIONS
    )
    assert first.fingerprint() != second.fingerprint()


# -- it reaches the run identity --------------------------------------------


def test_two_corpora_give_a_group_two_run_hashes():
    docvqa_only = a_config(a_manifest(("docvqa",)).fingerprint())
    all_sources = a_config(a_manifest(("docvqa", "chartqa")).fingerprint())
    assert run_hash(docvqa_only) != run_hash(all_sources)


def test_two_corpora_give_a_group_two_checkpoint_directories():
    # This is the failure in full: same Group, same seed, same budget, different
    # corpus. Sharing a resume key means the second run adopts the first's
    # completed checkpoint and trains nothing.
    docvqa_only = a_config(a_manifest(("docvqa",)).fingerprint())
    all_sources = a_config(a_manifest(("docvqa", "chartqa")).fingerprint())
    assert resume_key(docvqa_only) != resume_key(all_sources)


def test_raising_the_step_budget_still_resumes_within_one_corpus():
    # The fingerprint must not break the one continuation that is genuine.
    fingerprint = a_manifest().fingerprint()
    shorter = a_config(fingerprint)
    longer = shorter.model_copy(
        update={"training": shorter.training.model_copy(update={"steps": 4000})}
    )
    assert resume_key(shorter) == resume_key(longer)
    assert run_hash(shorter) != run_hash(longer)


# -- filling it in from the corpus on disk ----------------------------------


def test_the_fingerprint_is_read_from_the_corpus_directory(tmp_path: Path):
    a_manifest().write(tmp_path)
    assert corpus_fingerprint(tmp_path) == a_manifest().fingerprint()


def test_a_directory_with_no_manifest_has_no_fingerprint(tmp_path: Path):
    # The fast tier builds synthetic corpora by hand; they must stay usable.
    assert corpus_fingerprint(tmp_path) is None


def test_resolving_fills_the_fingerprint_in_from_disk(tmp_path: Path):
    a_manifest().write(tmp_path)
    merged = {"group_code": "B2", "corpus": {"name": "ceed-vqa", "splits": ["train"]}}
    resolved = with_corpus_fingerprint(merged, tmp_path)
    corpus = resolved["corpus"]
    assert isinstance(corpus, dict)
    assert corpus["fingerprint"] == a_manifest().fingerprint()


def test_resolving_does_not_mutate_what_it_was_given(tmp_path: Path):
    a_manifest().write(tmp_path)
    merged = {"group_code": "B2", "corpus": {"name": "ceed-vqa", "splits": ["train"]}}
    with_corpus_fingerprint(merged, tmp_path)
    assert "fingerprint" not in merged["corpus"]


@pytest.mark.parametrize("merged", [{"group_code": "B0"}, {"group_code": "B0", "corpus": None}])
def test_a_group_with_no_corpus_block_is_left_alone(merged, tmp_path: Path):
    a_manifest().write(tmp_path)
    assert with_corpus_fingerprint(merged, tmp_path) == merged


def test_the_entry_point_fills_it_in(tmp_path: Path):
    # The wiring, not just the helper: run_group.py must resolve against the
    # corpus it was pointed at, or none of the above protects a real run.
    import importlib.util

    a_manifest().write(tmp_path)
    spec = importlib.util.spec_from_file_location(
        "run_group_cli", Path(__file__).resolve().parents[1] / "scripts" / "run_group.py"
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    args = module.parse_args(["--group", "b2", "--corpus", str(tmp_path)])
    config = module.resolve_config(args)
    assert config.corpus is not None
    assert config.corpus.fingerprint == a_manifest().fingerprint()
