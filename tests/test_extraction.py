"""Teacher extraction into the artifact store, on the synthetic teacher.

These drive the extraction engine end to end on CPU: teacher-forced artefacts for
every answer token, over-caching combine weights at every layer and hidden states
at candidate layers, a free-generation correctness flag, worker sharding, and
resume. The alignment test is the off-by-one guard the whole store depends on:
the cached answer-token index must equal the Student's cross-entropy target
index.
"""

import numpy as np
import torch

from ceed_teacher import (
    COMBINE_WEIGHTS,
    HIDDEN_STATES,
    TOP_K_LOGIT_IDS,
    TOP_K_LOGIT_VALUES,
    ExtractionSpec,
    SyntheticConfig,
    SyntheticHybridTeacher,
    create_store,
    extract_example,
    run_extraction,
    shard_examples,
)

VOCAB = SyntheticConfig().vocab_size


def a_spec() -> ExtractionSpec:
    # Combine weights at every layer (over-caching); hidden states at both
    # candidate layers; top-3 logits.
    return ExtractionSpec(top_k=3, combine_weight_layers=(0, 1), hidden_state_layers=(0, 1))


def a_source(ids: list[int], prompt_length: int, seed: int = 0) -> SyntheticHybridTeacher:
    teacher = SyntheticHybridTeacher(SyntheticConfig(seed=seed))
    teacher.run(torch.tensor(ids), prompt_length=prompt_length)
    return teacher


# -- teacher-forced artefacts per answer token ------------------------------


def test_one_row_per_answer_token():
    # Prompt of 3, gold answer of 2 -> two answer tokens.
    source = a_source([3, 1, 4, 1, 5], prompt_length=3)
    rows = extract_example(source, "docvqa:q1", a_spec())
    assert [r.answer_token_index for r in rows] == [3, 4]


def test_rows_carry_every_over_cached_artefact():
    source = a_source([3, 1, 4, 1, 5], prompt_length=3)
    spec = a_spec()
    row = extract_example(source, "docvqa:q1", spec)[0]

    assert row.vectors[TOP_K_LOGIT_IDS].shape == (spec.top_k,)
    assert row.vectors[TOP_K_LOGIT_VALUES].shape == (spec.top_k,)
    # Combine weights over every layer, hidden states over every candidate layer.
    assert row.vectors[COMBINE_WEIGHTS].shape == (2, source.n_experts)
    assert row.vectors[HIDDEN_STATES].shape == (2, source.hidden_size)


def test_cached_logits_are_the_top_k_of_the_teachers_prediction():
    source = a_source([3, 1, 4, 1, 5], prompt_length=3)
    row = extract_example(source, "docvqa:q1", a_spec())[0]
    # The first answer token (index 3) is predicted at measurement position 2.
    expected = torch.topk(source.logits(2), 3)
    assert np.array_equal(row.vectors[TOP_K_LOGIT_IDS], expected.indices.numpy())


# -- alignment: cached index == Student CE target index ---------------------


def test_cached_answer_token_index_aligns_with_the_student_ce_target():
    # A hand-checked example: prompt ids [7, 7, 7], gold answer ids [9, 2].
    ids = [7, 7, 7, 9, 2]
    prompt_length = 3
    source = a_source(ids, prompt_length=prompt_length)
    rows = extract_example(source, "docvqa:hand", a_spec())

    for row in rows:
        # The Student, teacher-forcing the same sequence, computes cross-entropy
        # at position `answer_token_index` against `ids[answer_token_index]`.
        student_ce_target = ids[row.answer_token_index]
        assert row.gold_token_id == student_ce_target
    # And the answer-token indices are exactly the gold-answer positions.
    assert [r.answer_token_index for r in rows] == [3, 4]


# -- correctness flag from a separate free-generation pass ------------------


def test_a_correctness_flag_is_produced_and_uniform_per_example():
    source = a_source([3, 1, 4, 1, 5], prompt_length=3)
    rows = extract_example(source, "docvqa:q1", a_spec())
    flags = {r.correct for r in rows}
    assert len(flags) == 1
    assert isinstance(next(iter(flags)), bool)


def test_correctness_matches_a_direct_free_generation():
    source = a_source([3, 1, 4, 1, 5], prompt_length=3)
    rows = extract_example(source, "docvqa:q1", a_spec())
    assert rows[0].correct == source.correctness()


# -- into the store, sharded and resumable ----------------------------------


def a_built_store(tmp_path):
    spec = a_spec()
    dims = SyntheticConfig()
    return create_store(tmp_path, "fp-extract", spec, dims.n_experts, dims.hidden_size), spec


def build_source_for(example_id: str) -> SyntheticHybridTeacher:
    # Deterministic per example: id text drives the token ids and the seed.
    n = sum(ord(c) for c in example_id)
    ids = [(n + i) % VOCAB for i in range(5)]
    return a_source(ids, prompt_length=3, seed=n % 7)


def test_extraction_writes_rows_readable_from_the_store(tmp_path):
    store, spec = a_built_store(tmp_path)
    run_extraction(store, ["docvqa:q1"], build_source_for, spec, worker_id="w0")

    back = store.vector("docvqa:q1", 3, COMBINE_WEIGHTS)
    assert back.shape == (2, SyntheticConfig().n_experts)


def test_workers_extract_disjoint_shards(tmp_path):
    store, spec = a_built_store(tmp_path)
    ids = [f"docvqa:q{i}" for i in range(6)]
    for worker in range(2):
        assigned = shard_examples(ids, n_workers=2, worker_index=worker)
        run_extraction(store, assigned, build_source_for, spec, worker_id=f"w{worker}")

    assert store.example_ids() == set(ids)
    assert len(list((store.root / "shards").glob("*.parquet"))) == 2


def test_extraction_resumes_and_skips_completed_examples(tmp_path):
    store, spec = a_built_store(tmp_path)
    first = run_extraction(
        store, ["docvqa:q1", "docvqa:q2"], build_source_for, spec, worker_id="w0"
    )
    assert first == 2

    # A resumed run over a superset re-extracts only the new example.
    again = run_extraction(
        store, ["docvqa:q1", "docvqa:q2", "docvqa:q3"], build_source_for, spec, worker_id="w0"
    )
    assert again == 1
    assert store.example_ids() == {"docvqa:q1", "docvqa:q2", "docvqa:q3"}


def test_shard_examples_partitions_disjointly():
    ids = [f"q{i}" for i in range(7)]
    parts = [shard_examples(ids, 3, w) for w in range(3)]
    covered = [e for part in parts for e in part]
    assert sorted(covered) == sorted(ids)
    assert len(covered) == len(set(covered))


def test_extraction_is_deterministic_across_reextraction(tmp_path):
    store_a, spec = a_built_store(tmp_path / "a")
    store_b, _ = a_built_store(tmp_path / "b")
    run_extraction(store_a, ["docvqa:q1"], build_source_for, spec, worker_id="w0")
    run_extraction(store_b, ["docvqa:q1"], build_source_for, spec, worker_id="w0")

    for kind in (TOP_K_LOGIT_VALUES, COMBINE_WEIGHTS, HIDDEN_STATES):
        assert np.array_equal(
            store_a.vector("docvqa:q1", 3, kind), store_b.vector("docvqa:q1", 3, kind)
        )
