"""The artifact store: the primary seam, tested by round-trips and identity.

A wrong store is silent: a mismatched key or a corrupted vector produces a Phase
1 table that looks reasonable and means nothing. These tests write, read back,
and assert exact identity of arrays and keys; assert that two extraction
configurations produce two roots; and assert the store fails fast when a required
artefact kind is absent.
"""

import numpy as np
import pytest

from ceed_core import (
    ArtifactRow,
    ArtifactStore,
    ArtifactStoreError,
    MissingArtifactKindError,
    StoreMetadata,
    VectorSpec,
    store_root,
)


def a_schema() -> dict[str, VectorSpec]:
    # Toy dimensions standing in for top-k logits, all-layer combine weights, and
    # candidate-layer hidden states.
    return {
        "top_k_logit_values": VectorSpec(dtype="float32", shape=(4,)),
        "combine_weights": VectorSpec(dtype="float16", shape=(2, 8)),
        "hidden_states": VectorSpec(dtype="float16", shape=(2, 16)),
    }


def a_metadata(fingerprint: str = "fp-abc") -> StoreMetadata:
    return StoreMetadata(
        extraction_fingerprint=fingerprint,
        vector_kinds=a_schema(),
        corpus_manifest={"seed": 7, "splits": {"train": ["docvqa:q1"]}},
    )


def a_row(example_id: str, token: int, *, correct: bool = True) -> ArtifactRow:
    rng = np.random.default_rng(abs(hash((example_id, token))) % (2**32))
    return ArtifactRow(
        example_id=example_id,
        answer_token_index=token,
        gold_token_id=100 + token,
        correct=correct,
        vectors={
            "top_k_logit_values": rng.standard_normal(4).astype(np.float32),
            "combine_weights": rng.standard_normal((2, 8)).astype(np.float16),
            "hidden_states": rng.standard_normal((2, 16)).astype(np.float16),
        },
    )


def a_store(tmp_path, fingerprint: str = "fp-abc") -> ArtifactStore:
    return ArtifactStore.create(store_root(tmp_path, fingerprint), a_metadata(fingerprint))


# -- root is the extraction fingerprint -------------------------------------


def test_store_root_is_the_extraction_fingerprint(tmp_path):
    assert store_root(tmp_path, "fp-abc") == tmp_path / "fp-abc"


def test_two_fingerprints_produce_two_roots(tmp_path):
    one = a_store(tmp_path, "fingerprint-one")
    two = a_store(tmp_path, "fingerprint-two")
    assert one.root != two.root


# -- one row per (example, answer token), round-tripped exactly -------------


def test_a_written_vector_round_trips_bit_exactly(tmp_path):
    store = a_store(tmp_path)
    row = a_row("docvqa:q1", 5)
    store.writer("w0").write([row])

    for kind, value in row.vectors.items():
        back = store.vector("docvqa:q1", 5, kind)
        assert back.dtype == value.dtype
        assert back.shape == value.shape
        assert np.array_equal(back, value)


def test_keys_and_scalars_round_trip(tmp_path):
    store = a_store(tmp_path)
    store.writer("w0").write([a_row("gqa:g1", 3, correct=False)])

    rows = store.query(
        "SELECT example_id, answer_token_index, gold_token_id, correct FROM artifacts"
    )
    assert rows == [("gqa:g1", 3, 103, False)]


def test_one_row_per_example_and_answer_token(tmp_path):
    store = a_store(tmp_path)
    store.writer("w0").write([a_row("docvqa:q1", 0), a_row("docvqa:q1", 1)])

    count = store.query("SELECT count(*) FROM artifacts")[0][0]
    assert count == 2


# -- metadata ----------------------------------------------------------------


def test_metadata_records_kinds_fingerprint_and_manifest(tmp_path):
    store = a_store(tmp_path)
    reopened = ArtifactStore(store.root)

    assert reopened.metadata.extraction_fingerprint == "fp-abc"
    assert set(reopened.metadata.vector_kinds) == {
        "top_k_logit_values",
        "combine_weights",
        "hidden_states",
    }
    assert reopened.metadata.corpus_manifest["seed"] == 7


def test_opening_a_directory_without_metadata_fails(tmp_path):
    with pytest.raises(ArtifactStoreError, match="no artifact store"):
        ArtifactStore(tmp_path / "nothing")


def test_recreating_with_different_metadata_is_refused(tmp_path):
    a_store(tmp_path, "fp-abc")
    clashing = StoreMetadata(extraction_fingerprint="fp-abc", vector_kinds={})
    with pytest.raises(ArtifactStoreError, match="different metadata"):
        ArtifactStore.create(store_root(tmp_path, "fp-abc"), clashing)


# -- fail fast on a missing artefact kind -----------------------------------


def test_require_kinds_passes_when_present(tmp_path):
    store = a_store(tmp_path)
    store.metadata.require_kinds(["combine_weights", "hidden_states"])


def test_require_kinds_names_what_is_missing(tmp_path):
    store = a_store(tmp_path)
    with pytest.raises(MissingArtifactKindError, match="attribution"):
        store.metadata.require_kinds(["combine_weights", "attribution"])


# -- SQL queryable -----------------------------------------------------------


def test_store_is_queryable_with_sql(tmp_path):
    store = a_store(tmp_path)
    store.writer("w0").write(
        [a_row("docvqa:q1", 0, correct=True), a_row("docvqa:q1", 1, correct=True)]
    )
    store.writer("w1").write([a_row("gqa:g1", 0, correct=False)])

    correct_examples = store.query(
        "SELECT DISTINCT example_id FROM artifacts WHERE correct ORDER BY example_id"
    )
    assert correct_examples == [("docvqa:q1",)]


def test_querying_an_empty_store_returns_nothing(tmp_path):
    store = a_store(tmp_path)
    assert store.query("SELECT * FROM artifacts") == []
    assert store.example_ids() == set()


# -- independent worker shards, no coordination -----------------------------


def test_several_workers_write_independent_shards(tmp_path):
    store = a_store(tmp_path)
    store.writer("w0").write([a_row("docvqa:q1", 0)])
    store.writer("w1").write([a_row("gqa:g1", 0)])

    shards = list((store.root / "shards").glob("*.parquet"))
    assert len(shards) == 2
    assert store.example_ids() == {"docvqa:q1", "gqa:g1"}


def test_a_worker_writes_a_distinct_shard_per_batch(tmp_path):
    store = a_store(tmp_path)
    writer = store.writer("w0")
    first = writer.write([a_row("docvqa:q1", 0)])
    second = writer.write([a_row("docvqa:q2", 0)])
    assert first != second


# -- resume ------------------------------------------------------------------


def test_completed_example_ids_support_resume(tmp_path):
    store = a_store(tmp_path)
    store.writer("w0").write([a_row("docvqa:q1", 0), a_row("docvqa:q2", 0)])

    done = store.example_ids()
    remaining = [e for e in ["docvqa:q1", "docvqa:q2", "docvqa:q3"] if e not in done]
    assert remaining == ["docvqa:q3"]


def test_reopening_a_partial_store_sees_prior_shards(tmp_path):
    store = a_store(tmp_path)
    store.writer("w0").write([a_row("docvqa:q1", 0)])

    resumed = ArtifactStore(store.root)
    assert resumed.example_ids() == {"docvqa:q1"}


# -- guards ------------------------------------------------------------------


def test_writing_an_empty_batch_is_refused(tmp_path):
    store = a_store(tmp_path)
    with pytest.raises(ArtifactStoreError, match="empty shard"):
        store.writer("w0").write([])


def test_a_row_with_the_wrong_vector_shape_is_refused(tmp_path):
    store = a_store(tmp_path)
    bad = ArtifactRow(
        example_id="docvqa:q1",
        answer_token_index=0,
        gold_token_id=1,
        correct=True,
        vectors={
            "top_k_logit_values": np.zeros(3, np.float32),  # schema wants 4
            "combine_weights": np.zeros((2, 8), np.float16),
            "hidden_states": np.zeros((2, 16), np.float16),
        },
    )
    with pytest.raises(ArtifactStoreError, match="shape"):
        store.writer("w0").write([bad])
