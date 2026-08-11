"""The corpus manifest: exactly which examples and splits a run assembled.

The manifest is what makes two Groups trained weeks apart comparable. It records
the split seed, the split fractions, and the exact example ids in each split, so
a corpus can be checked for identity rather than assumed identical.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from ceed_data.example import Example

MANIFEST_FILENAME = "corpus_manifest.json"


def corpus_fingerprint(directory: Path) -> str | None:
    """Return the fingerprint of the corpus at ``directory``, if it has a manifest.

    Args:
        directory: A corpus directory written by ``scripts/build_corpus.py``.

    Returns:
        The manifest fingerprint, or ``None`` if no manifest is there — which is
        the case for the synthetic corpora the fast tier builds by hand.
    """
    if not (directory / MANIFEST_FILENAME).exists():
        return None
    return CorpusManifest.read(directory).fingerprint()


def with_corpus_fingerprint(merged: Mapping[str, object], directory: Path) -> dict[str, object]:
    """Return ``merged`` with the corpus fingerprint of ``directory`` filled in.

    Every entry point that resolves a Group against a corpus on disk calls this
    before validating, so the resulting run hash covers the corpus the run
    actually read. A Group with no ``corpus`` block, or a corpus directory with no
    manifest, is returned unchanged.

    Args:
        merged: The resolved overlay mapping, before validation.
        directory: The corpus directory the run was pointed at.

    Returns:
        A new mapping; ``merged`` is not mutated.
    """
    corpus = merged.get("corpus")
    if not isinstance(corpus, Mapping):
        return dict(merged)
    fingerprint = corpus_fingerprint(directory)
    if fingerprint is None:
        return dict(merged)
    return {**merged, "corpus": {**corpus, "fingerprint": fingerprint}}


class CorpusManifest(BaseModel):
    """A durable record of an assembled corpus.

    Attributes:
        seed: The split seed.
        fractions: The split fractions used.
        splits: Each split name mapped to its example ids, in assembly order.
        dataset_counts: How many examples came from each source dataset.
    """

    model_config = ConfigDict(frozen=True)

    seed: int
    fractions: dict[str, float]
    splits: dict[str, tuple[str, ...]]
    dataset_counts: dict[str, int]

    @classmethod
    def from_splits(
        cls,
        splits: Mapping[str, Sequence[Example]],
        seed: int,
        fractions: Mapping[str, float],
    ) -> CorpusManifest:
        """Build a manifest from an assembled, split corpus.

        Args:
            splits: Each split name mapped to its examples.
            seed: The split seed used.
            fractions: The split fractions used.

        Returns:
            The manifest describing this corpus.
        """
        dataset_counts: dict[str, int] = {}
        for examples in splits.values():
            for example in examples:
                dataset_counts[example.dataset] = dataset_counts.get(example.dataset, 0) + 1
        return cls(
            seed=seed,
            fractions=dict(fractions),
            splits={
                name: tuple(e.example_id for e in examples) for name, examples in splits.items()
            },
            dataset_counts=dataset_counts,
        )

    def write(self, directory: Path) -> Path:
        """Write the manifest as JSON into ``directory`` and return its path."""
        path = directory / MANIFEST_FILENAME
        path.write_text(self.model_dump_json(indent=2))
        return path

    @classmethod
    def read(cls, directory: Path) -> CorpusManifest:
        """Read a manifest previously written by :meth:`write`."""
        return cls.model_validate_json((directory / MANIFEST_FILENAME).read_text())

    def fingerprint(self) -> str:
        """Return a content hash of exactly which examples are in which split.

        This is what lets a run *state* the corpus it trained on rather than name
        it. ``CorpusConfig.name`` is a constant in the Group YAML, so without this
        two runs over entirely different corpora hash identically — and identical
        hashes mean the second run adopts the first's completed checkpoint, trains
        zero steps, and is reported under the new corpus's name.

        Covers the split membership, the seed, and the fractions. It deliberately
        does not cover the images or the question text: an example id names one
        question from one source, so two corpora agreeing on every id in every
        split are the same corpus.

        Returns:
            The first 16 hex characters of the SHA-256 of the canonical form.
        """
        payload = json.dumps(
            {
                "seed": self.seed,
                "fractions": {name: self.fractions[name] for name in sorted(self.fractions)},
                "splits": {name: list(self.splits[name]) for name in sorted(self.splits)},
            },
            separators=(",", ":"),
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode()).hexdigest()[:16]
