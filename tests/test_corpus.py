"""The corpus: one Example type, annotation-backed, deterministically split.

These tests drive the pure normalisation and assembly logic on synthetic rows.
The streaming loaders that download the real corpora are out of the fast tier.
"""

import pytest
from pydantic import ValidationError

from ceed_data import (
    BoundingBox,
    CorpusManifest,
    Example,
    ImageStore,
    OcrWord,
    assign_split,
    image_fingerprint,
    make_chartqa_example,
    make_docvqa_example,
    make_gqa_example,
    normalize_box,
    split_corpus,
)


def a_docvqa_example(source_id: str = "q1") -> Example:
    return make_docvqa_example(
        source_id=source_id,
        question="What is the total?",
        answers=["42"],
        image_fingerprint="fp",
        ocr_words=[OcrWord(text="Total", box=BoundingBox(x=0.1, y=0.1, width=0.2, height=0.05))],
    )


def a_gqa_example(source_id: str = "g1") -> Example:
    return make_gqa_example(
        source_id=source_id,
        question="What colour is the car?",
        answers=["red"],
        image_fingerprint="fp",
        answer_region=BoundingBox(x=0.3, y=0.3, width=0.2, height=0.2),
    )


def a_chartqa_example(source_id: str = "c1") -> Example:
    return make_chartqa_example(
        source_id=source_id, question="Peak value?", answers=["7"], image_fingerprint="fp"
    )


# -- one Example type, stable ids, annotations preserved --------------------


def test_all_three_datasets_produce_the_one_example_type():
    for example in (a_docvqa_example(), a_gqa_example(), a_chartqa_example()):
        assert isinstance(example, Example)


def test_identifiers_are_stable_and_dataset_prefixed():
    assert a_docvqa_example("q1").example_id == "docvqa:q1"
    assert a_gqa_example("g1").example_id == "gqa:g1"
    assert a_chartqa_example("c1").example_id == "chartqa:c1"


def test_docvqa_carries_ocr_words_and_boxes():
    example = a_docvqa_example()
    assert example.ocr_words[0].text == "Total"
    assert example.ocr_words[0].box.width == 0.2


def test_gqa_carries_the_answer_region():
    assert a_gqa_example().answer_region is not None


# -- intervention eligibility carried by the data ---------------------------


def test_docvqa_and_gqa_are_intervention_eligible():
    assert a_docvqa_example().intervention_eligible
    assert a_gqa_example().intervention_eligible


def test_chartqa_is_never_intervention_eligible():
    assert not a_chartqa_example().intervention_eligible


def test_a_chartqa_example_marked_eligible_is_rejected():
    with pytest.raises(ValidationError, match="never intervention-eligible"):
        Example(
            example_id="chartqa:x",
            dataset="chartqa",
            image_fingerprint="fp",
            question="q",
            answers=("a",),
            intervention_eligible=True,
        )


# -- normalisation ----------------------------------------------------------


def test_pixel_boxes_normalise_to_the_unit_square():
    box = normalize_box(50, 25, 100, 50, image_width=200, image_height=100)
    assert box.x == 0.25
    assert box.width == 0.5


def test_a_box_overrunning_the_image_is_clamped():
    box = normalize_box(180, 0, 100, 50, image_width=200, image_height=100)
    assert box.x + box.width <= 1.0


# -- deterministic, seed-recorded splits ------------------------------------

FRACTIONS = {"train": 0.8, "val": 0.2}


def test_split_assignment_is_deterministic_for_a_seed():
    assert assign_split("docvqa:q1", 7, FRACTIONS) == assign_split("docvqa:q1", 7, FRACTIONS)


def test_a_different_seed_can_reassign_an_example():
    seeds = {assign_split("docvqa:q1", s, FRACTIONS) for s in range(20)}
    assert len(seeds) == 2  # both splits are reachable across seeds


def test_split_is_independent_of_input_order():
    examples = [a_docvqa_example(f"q{i}") for i in range(50)]
    forward = split_corpus(examples, seed=3, fractions=FRACTIONS)
    backward = split_corpus(list(reversed(examples)), seed=3, fractions=FRACTIONS)
    assert {e.example_id for e in forward["train"]} == {e.example_id for e in backward["train"]}


def test_split_fractions_must_sum_to_one():
    with pytest.raises(ValueError, match="sum to 1"):
        assign_split("docvqa:q1", 0, {"train": 0.7, "val": 0.2})


# -- manifest ---------------------------------------------------------------


def test_manifest_records_examples_splits_and_counts(tmp_path):
    examples = [a_docvqa_example("q1"), a_gqa_example("g1"), a_chartqa_example("c1")]
    splits = split_corpus(examples, seed=1, fractions=FRACTIONS)
    manifest = CorpusManifest.from_splits(splits, seed=1, fractions=FRACTIONS)

    assert manifest.seed == 1
    assert sum(manifest.dataset_counts.values()) == 3
    recorded = {eid for ids in manifest.splits.values() for eid in ids}
    assert recorded == {"docvqa:q1", "gqa:g1", "chartqa:c1"}

    manifest.write(tmp_path)
    assert CorpusManifest.read(tmp_path) == manifest


# -- content-addressed images -----------------------------------------------


def test_identical_bytes_share_an_address_and_differ_from_others():
    assert image_fingerprint(b"image-a") == image_fingerprint(b"image-a")
    assert image_fingerprint(b"image-a") != image_fingerprint(b"image-b")


def test_the_store_returns_byte_identical_images(tmp_path):
    store = ImageStore(tmp_path)
    data = b"\x89PNG\r\n\x1a\n fake image bytes"
    fingerprint = store.put(data)
    assert store.get(fingerprint) == data
    assert store.contains(fingerprint)


def test_storing_the_same_image_twice_is_idempotent(tmp_path):
    store = ImageStore(tmp_path)
    data = b"same bytes"
    assert store.put(data) == store.put(data)
