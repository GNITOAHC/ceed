"""Group configuration schema, canonical hashing, and the extraction fingerprint.

Every arm of the CEED comparison — B0 through E4, at any seed, under either
parameter-efficiency mode — is one fully-resolved :class:`GroupConfig`. A
configuration is composed from a stack of YAML overlays (a base, a student
layer, a Group overlay, and an optional Phase 2 variant), validated once at load
time so that a malformed Group fails immediately rather than at hour six of a
run.

Two hashes are derived from a resolved configuration and they are deliberately
different:

* the **run hash** covers the whole configuration and is the run identifier —
  two runs claiming to be the same Group provably are;
* the **extraction fingerprint** covers only the extraction-relevant subset
  (see :class:`ExtractionConfig`) and identifies an artifact store, so artefacts
  produced under a different ablation or layer definition can never be mixed
  into a Group that did not extract them (see
  docs/adr/0001-answer-token-scoped-artifact-store.md).

Both hashes are taken over a *canonical* JSON encoding, so two configurations
that differ only in field order or whitespace hash identically, while any
meaningful difference changes the hash.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from enum import StrEnum
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator


class ParamEfficiencyMode(StrEnum):
    """How a Group's Student parameters are trained.

    ``full`` fine-tunes every Student parameter and is the only mode whose
    numbers may reach the Phase 1 headline table; ``lora`` is the cheap
    development path (see
    docs/adr/0005-lora-for-development-full-finetune-for-headline-results.md).
    """

    LORA = "lora"
    FULL = "full"


class _Frozen(BaseModel):
    """Base for every configuration model: immutable and closed to typos.

    ``frozen`` makes a resolved configuration hashable and safe to pass around
    without defensive copies; ``extra="forbid"`` turns an unrecognised YAML key
    into a load-time validation error rather than a silently ignored setting.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")


class LayerMapping(_Frozen):
    """The correspondence between Teacher and Student layers.

    Represented as a first-class object rather than an implicit convention
    because C2 is defined as a deliberately *mismatched* mapping and Phase 2
    reports variance across alternatives, so the mapping a run used must be
    recorded, not inferred.

    Attributes:
        kind: How the mapping was produced — ``proportional`` (the placeholder
            that unblocks B3), ``probe`` (the learned replacement), or
            ``mismatched`` (C2's control).
        pairs: The ``(teacher_layer, student_layer)`` correspondences.
    """

    kind: str
    pairs: tuple[tuple[int, int], ...]

    @field_validator("kind")
    @classmethod
    def _known_kind(cls, value: str) -> str:
        allowed = {"proportional", "probe", "mismatched"}
        if value not in allowed:
            raise ValueError(f"layer mapping kind must be one of {sorted(allowed)}, got {value!r}")
        return value


class StudentConfig(_Frozen):
    """The Student checkpoint and the precision it is loaded in.

    Attributes:
        model: The Student model identifier.
        dtype: The compute dtype (``float16`` on Volta, which has no bf16).
    """

    model: str
    dtype: str = "float16"


class CorpusConfig(_Frozen):
    """Which corpus a Group trains and evaluates on.

    ``None`` on a :class:`GroupConfig` denotes the null Group, which has no
    corpus at all. A real corpus is fleshed out in a later ticket; here it only
    needs to be nameable and hashable.

    Attributes:
        name: The corpus identifier.
        splits: The split names the run draws from.
    """

    name: str
    splits: tuple[str, ...] = ()


class ExtractionConfig(_Frozen):
    """The extraction-relevant subset of a configuration.

    The :func:`extraction_fingerprint` is a hash of exactly these fields, so two
    Groups whose extraction settings match share one artifact store while any
    change here forces a fresh store. The fields cover the extraction identity
    the CONTEXT glossary names — the layer set, the ablation and combine-weight
    definitions, the thinking gate, and the model revision — pinning the Teacher
    checkpoint by both its identifier and its revision so that two different
    Teachers at the same revision string cannot collide onto one store.

    Attributes:
        teacher_model: The Teacher model identifier.
        model_revision: The pinned Teacher checkpoint revision.
        layers: The Teacher layers artefacts are extracted at.
        ablation: The ablation definition (e.g. ``mean-of-active``).
        combine_weight: The combine-weight definition (e.g. ``effective``).
        thinking_enabled: Whether the Teacher's thinking gate is on. Disabled
            everywhere in CEED.
    """

    teacher_model: str
    model_revision: str
    layers: tuple[int, ...]
    ablation: str
    combine_weight: str
    thinking_enabled: bool = False


class AuxiliarySignalConfig(_Frozen):
    """One auxiliary supervision signal added on top of the shared backbone.

    A Group is the backbone plus zero or more of these; the null Group and B0
    have none, E4 has three. Later tickets give each signal its artefact-kind
    requirements and loss; here it only needs a stable name and weight so that
    Group composition and hashing can be exercised.

    Attributes:
        name: The signal identifier.
        weight: Its scalar loss weight.
    """

    name: str
    weight: float = 1.0


class Phase2Variant(_Frozen):
    """An optional Phase 2 variant overlay.

    Phase 2 experiments are out of scope for the current milestone, but the
    slot exists so a variant is a configuration field rather than a schema
    change.

    Attributes:
        name: The variant identifier.
    """

    name: str


class GroupConfig(_Frozen):
    """A fully-resolved arm of the CEED comparison.

    This is the single input to :func:`ceed_student.run.run_group`. It is
    produced by :func:`resolve_group_config` from a stack of YAML overlays and
    validated in one pass, so an invalid Group is rejected at load time.

    Attributes:
        group_code: The Group's code (``B0`` to ``E4``, or ``NULL`` for the null
            Group).
        seed: The random seed. Part of the run hash but not the extraction
            fingerprint.
        param_efficiency: Whether the Student is LoRA or fully fine-tuned.
        layer_mapping: The Teacher-to-Student layer correspondence used.
        student: The Student checkpoint and precision.
        extraction: The extraction-relevant settings (the fingerprint subset).
        corpus: The corpus, or ``None`` for the null Group.
        auxiliary_signals: The signals added on top of the backbone.
        phase2: An optional Phase 2 variant.
    """

    group_code: str
    seed: int = Field(ge=0)
    param_efficiency: ParamEfficiencyMode
    layer_mapping: LayerMapping
    student: StudentConfig
    extraction: ExtractionConfig
    corpus: CorpusConfig | None = None
    auxiliary_signals: tuple[AuxiliarySignalConfig, ...] = ()
    phase2: Phase2Variant | None = None

    @field_validator("group_code")
    @classmethod
    def _non_empty_code(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("group_code must be a non-empty string")
        return value


def canonical_json(model: BaseModel) -> str:
    """Serialise a model to a canonical JSON string.

    The encoding is independent of field declaration order and carries no
    incidental whitespace: keys are sorted and separators are tight. Two models
    with identical content therefore produce byte-identical output, which is the
    property the hashes rely on.

    Args:
        model: Any configuration model.

    Returns:
        The canonical JSON encoding.
    """
    return json.dumps(model.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def run_hash(config: GroupConfig) -> str:
    """Return the run identifier: a hash of the whole configuration.

    Args:
        config: A resolved Group configuration.

    Returns:
        The hex SHA-256 of the configuration's canonical JSON. Two
        configurations differing in any meaningful field produce different
        values; two differing only in field order or formatting produce the
        same value.
    """
    return _sha256(canonical_json(config))


def extraction_fingerprint(config: GroupConfig) -> str:
    """Return the extraction fingerprint: a hash of the extraction subset only.

    Unlike :func:`run_hash`, this ignores everything outside
    :class:`ExtractionConfig`, so two Groups that differ only in, say, their seed
    or auxiliary signals share one artifact store, while any change to the layer
    set, ablation, or combine-weight definition forces a fresh one.

    Args:
        config: A resolved Group configuration.

    Returns:
        The hex SHA-256 of the extraction configuration's canonical JSON.
    """
    return _sha256(canonical_json(config.extraction))


def _deep_merge(base: Mapping[str, Any], overlay: Mapping[str, Any]) -> dict[str, Any]:
    """Recursively merge ``overlay`` onto ``base``.

    Nested mappings are merged key by key; every other value (including lists)
    is replaced wholesale by the overlay. ``base`` and ``overlay`` are not
    mutated.
    """
    merged: dict[str, Any] = dict(base)
    for key, value in overlay.items():
        existing = merged.get(key)
        if isinstance(existing, Mapping) and isinstance(value, Mapping):
            merged[key] = _deep_merge(existing, value)
        else:
            merged[key] = value
    return merged


def load_overlay(path: Path) -> dict[str, Any]:
    """Read one YAML overlay file into a plain dictionary.

    Args:
        path: The overlay file.

    Returns:
        The overlay's top-level mapping.

    Raises:
        ValueError: If the file's top level is not a mapping.
    """
    raw = yaml.safe_load(path.read_text())
    if raw is None:
        return {}
    if not isinstance(raw, Mapping):
        raise ValueError(f"overlay {path} must contain a mapping at its top level")
    return {str(key): value for key, value in raw.items()}


def merge_overlays(layers: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Fold a stack of overlays into a single configuration mapping.

    Later overlays win over earlier ones, merging nested mappings and replacing
    everything else. This is the composition order a Group is built in: base,
    then student, then the Group overlay, then any Phase 2 variant.

    Args:
        layers: The overlays, in application order.

    Returns:
        The merged mapping, ready to validate into a :class:`GroupConfig`.
    """
    merged: dict[str, Any] = {}
    for layer in layers:
        merged = _deep_merge(merged, layer)
    return merged


def resolve_group_config(paths: Sequence[Path]) -> GroupConfig:
    """Compose and validate a Group configuration from a stack of YAML overlays.

    The overlays are merged in the given order and validated in one pass, so a
    malformed or incomplete Group is rejected here — with a Pydantic error
    naming the offending field — rather than partway through a run.

    Args:
        paths: The overlay files, in application order (base first).

    Returns:
        The resolved, validated configuration.

    Raises:
        pydantic.ValidationError: If the merged configuration is invalid.
        ValueError: If any overlay's top level is not a mapping.
    """
    merged = merge_overlays([load_overlay(path) for path in paths])
    return GroupConfig.model_validate(merged)
