"""Teacher extraction: filling the artifact store the baseline Groups read from.

Extraction teacher-forces the gold answer with the thinking gate off (ADR-0002),
so an answer token is always a gold token and the teacher's positions align 1:1
with the Student's cross-entropy targets. For each answer token it caches the
top-k logits, the effective combine weights for *every* teacher layer, and the
hidden states for the candidate layers — the over-caching (ADR-0001) that lets
the eventual three-layer choice be a training-time selection rather than a
re-extraction. A separate free-generation pass yields one correctness flag per
example.

The answer-token index recorded is the *target* position ``t``: the position
whose gold token the Student predicts, so the cached key aligns with the
Student's loss target. The artefacts for token ``t`` are read at the measurement
position ``t - 1``, where the teacher-forced forward makes that prediction.

Extraction runs against any :class:`ExtractionSource`; the synthetic teacher
implements it for CPU tests, and the real hooked teacher implements it on a GPU.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import torch
from torch import Tensor

from ceed_core import ArtifactRow, ArtifactStore, StoreMetadata, VectorSpec, store_root

# Artefact-kind (column) names.
TOP_K_LOGIT_IDS = "top_k_logit_ids"
TOP_K_LOGIT_VALUES = "top_k_logit_values"
COMBINE_WEIGHTS = "combine_weights"
HIDDEN_STATES = "hidden_states"
VISUAL_ADVANTAGE = "visual_advantage"


@runtime_checkable
class ExtractionSource(Protocol):
    """A completed teacher-forced forward, exposing what extraction caches.

    Every query is indexed by a *measurement position* — the position at which
    the teacher predicts the next token. Answer-token index ``t`` is measured at
    position ``t - 1``.
    """

    n_experts: int
    hidden_size: int

    def answer_measurement_positions(self) -> list[int]:
        """Return the measurement positions whose next token is an answer token."""
        ...

    def gold_token_id(self, position: int) -> int:
        """Return the gold token id this position predicts (the Student's CE target)."""
        ...

    def logits(self, position: int) -> Tensor:
        """Return the next-token logits at this position."""
        ...

    def effective_combine_weights(self, layer: int, position: int) -> Tensor:
        """Return every expert's effective combine weight at this position and layer."""
        ...

    def hidden_state(self, layer: int, position: int) -> Tensor:
        """Return the residual-stream state leaving ``layer`` at this position."""
        ...

    def correctness(self) -> bool:
        """Return the free-generation correctness flag for this example."""
        ...


@dataclass(frozen=True)
class ExtractionSpec:
    """What to cache per answer token.

    An empty layer tuple means that kind is not cached at all, which is how a
    store built for the logit-KD Groups alone differs from one built for the
    whole baseline set.

    Attributes:
        top_k: How many top logits to cache for logit KD.
        combine_weight_layers: The layers to cache effective combine weights at —
            all teacher layers, so the three-layer choice never forces a
            re-extraction. Empty to cache none.
        hidden_state_layers: The candidate layers to cache hidden states at.
            Empty to cache none.
        visual_advantage: Whether to cache B4's per-token visual advantage.
    """

    top_k: int
    combine_weight_layers: tuple[int, ...] = ()
    hidden_state_layers: tuple[int, ...] = ()
    visual_advantage: bool = False


def store_schema(spec: ExtractionSpec, n_experts: int, hidden_size: int) -> dict[str, VectorSpec]:
    """Return the store's vector-column schema for a spec and model dimensions.

    This is the single definition of what a CEED store's columns are — the
    synthetic-teacher path and the real extraction script both derive their
    metadata from it, so the two cannot drift into disagreeing about a shape.

    The two layer-stacked kinds record *which* layers their leading axis holds.
    Over-caching (ADR-0001) means that is deliberately wider than the layer set a
    Group supervises, so a reader that assumed the extraction configuration's
    three layers would silently read the wrong ones.

    Args:
        spec: What the extraction caches per answer token.
        n_experts: The Teacher's experts per layer.
        hidden_size: The Teacher's residual width.

    Returns:
        The vector columns, omitting any kind the spec does not cache.
    """
    schema = {
        TOP_K_LOGIT_IDS: VectorSpec(dtype="int32", shape=(spec.top_k,)),
        TOP_K_LOGIT_VALUES: VectorSpec(dtype="float32", shape=(spec.top_k,)),
    }
    if spec.combine_weight_layers:
        schema[COMBINE_WEIGHTS] = VectorSpec(
            dtype="float16",
            shape=(len(spec.combine_weight_layers), n_experts),
            layers=spec.combine_weight_layers,
        )
    if spec.hidden_state_layers:
        schema[HIDDEN_STATES] = VectorSpec(
            dtype="float16",
            shape=(len(spec.hidden_state_layers), hidden_size),
            layers=spec.hidden_state_layers,
        )
    if spec.visual_advantage:
        schema[VISUAL_ADVANTAGE] = VectorSpec(dtype="float32", shape=(1,))
    return schema


def create_store(
    base_dir: Path,
    extraction_fingerprint: str,
    spec: ExtractionSpec,
    n_experts: int,
    hidden_size: int,
    corpus_manifest: dict[str, Any] | None = None,
) -> ArtifactStore:
    """Create (or open) the store for a fingerprint with the schema a spec implies."""
    metadata = StoreMetadata(
        extraction_fingerprint=extraction_fingerprint,
        vector_kinds=store_schema(spec, n_experts, hidden_size),
        corpus_manifest=corpus_manifest,
    )
    return ArtifactStore.create(store_root(base_dir, extraction_fingerprint), metadata)


def extract_example(
    source: ExtractionSource, example_id: str, spec: ExtractionSpec
) -> list[ArtifactRow]:
    """Extract one example's answer-token artefact rows.

    The teacher-forced artefacts are read first, then the free-generation
    correctness pass runs; the two never interleave, so the correctness pass may
    disturb and restore the source's forward state without affecting what was
    read.

    Args:
        source: The completed teacher-forced forward for this example.
        example_id: The example's stable identifier.
        spec: What to cache per answer token.

    Returns:
        One row per answer token, in answer-token order.
    """
    rows: list[ArtifactRow] = []
    for position in source.answer_measurement_positions():
        logits = source.logits(position)
        top = torch.topk(logits, spec.top_k)
        combine = torch.stack(
            [
                source.effective_combine_weights(layer, position)
                for layer in spec.combine_weight_layers
            ]
        )
        hidden = torch.stack(
            [source.hidden_state(layer, position) for layer in spec.hidden_state_layers]
        )
        rows.append(
            ArtifactRow(
                example_id=example_id,
                answer_token_index=position + 1,
                gold_token_id=source.gold_token_id(position),
                correct=False,  # filled once the correctness pass has run
                vectors={
                    TOP_K_LOGIT_IDS: top.indices.detach().cpu().numpy(),
                    TOP_K_LOGIT_VALUES: top.values.detach().cpu().numpy(),
                    COMBINE_WEIGHTS: combine.detach().cpu().numpy(),
                    HIDDEN_STATES: hidden.detach().cpu().numpy(),
                },
            )
        )

    correct = source.correctness()
    return [replace(row, correct=correct) for row in rows]


def shard_examples(example_ids: Sequence[str], n_workers: int, worker_index: int) -> list[str]:
    """Return the examples assigned to one worker by round-robin.

    The partition is disjoint and needs no coordination: worker ``i`` of ``n``
    takes every ``n``-th example, so the workers together cover the corpus once.
    """
    return list(example_ids[worker_index::n_workers])


def run_extraction(
    store: ArtifactStore,
    example_ids: Iterable[str],
    build_source: Callable[[str], ExtractionSource],
    spec: ExtractionSpec,
    worker_id: str,
    batch_size: int = 64,
) -> int:
    """Extract a worker's examples into the store, resuming and sharding.

    Examples already present in the store are skipped, so a preempted run
    resumes rather than restarting. Rows are written in batches, each its own
    shard file; only this ``worker_id`` writes these shards, so no coordination
    with other workers is needed.

    Args:
        store: The store to write into.
        example_ids: The examples assigned to this worker.
        build_source: Builds the completed teacher-forced forward for an example.
        spec: What to cache per answer token.
        worker_id: This worker's shard identifier.
        batch_size: How many rows to accumulate before writing a shard.

    Returns:
        The number of examples extracted (excluding those skipped by resume).
    """
    done = store.example_ids()
    writer = store.writer(worker_id)
    batch: list[ArtifactRow] = []
    extracted = 0
    for example_id in example_ids:
        if example_id in done:
            continue
        batch.extend(extract_example(build_source(example_id), example_id, spec))
        extracted += 1
        if len(batch) >= batch_size:
            writer.write(batch)
            batch = []
    if batch:
        writer.write(batch)
    return extracted
