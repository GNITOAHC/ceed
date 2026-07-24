"""The corpus manifest: exactly which examples and splits a run assembled.

The manifest is what makes two Groups trained weeks apart comparable. It records
the split seed, the split fractions, and the exact example ids in each split, so
a corpus can be checked for identity rather than assumed identical.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from ceed_data.example import Example

MANIFEST_FILENAME = "corpus_manifest.json"


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
