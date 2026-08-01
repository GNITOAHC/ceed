"""The artifact store: the one seam between teacher measurement and student training.

Every teacher-side measurement is precomputed here and the teacher is never
loaded during student training (ADR-0001). The store is a directory of Parquet
shards read through DuckDB, keyed by ``(example_id, answer_token_index)`` with one
row per answer token. Vector artefacts — top-k logits, effective combine weights
for every teacher layer, hidden states for the candidate layers — are stored as
fixed-size binary columns; scalar artefacts such as the per-example correctness
flag are ordinary typed columns.

Three properties make the store safe to build a twelve-Group comparison on:

* its root is the **extraction fingerprint**, so artefacts produced under a
  different ablation or layer definition land in a different directory and can
  never be silently mixed;
* each extraction worker writes its own shards with no coordination, and a
  resumed run skips the examples already present rather than restarting;
* the store declares which artefact kinds it holds, so a Group that needs a kind
  the store lacks fails immediately with a clear message rather than at hour six.

This module depends on no model or training framework — arrays cross the boundary
as ``numpy`` and are encoded to bytes here.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from pydantic import BaseModel, ConfigDict, model_validator

METADATA_FILENAME = "store_metadata.json"
SHARDS_DIRNAME = "shards"

# The fixed key and scalar columns every store carries, alongside its vector
# artefact columns.
EXAMPLE_ID = "example_id"
ANSWER_TOKEN_INDEX = "answer_token_index"
GOLD_TOKEN_ID = "gold_token_id"
CORRECT = "correct"


class ArtifactStoreError(RuntimeError):
    """Raised when the store is used in a way that would corrupt or misread it."""


class MissingArtifactKindError(ArtifactStoreError):
    """Raised when a Group needs an artefact kind the store does not hold."""


class VectorSpec(BaseModel):
    """The dtype and shape of one vector artefact column.

    A vector artefact is stored as fixed-size binary — the raw bytes of a
    contiguous array — so its dtype and shape must be known to encode and decode
    it. They are recorded in the store metadata so a reader needs nothing but the
    store itself.

    Layer-stacked artefacts carry one more thing. Effective combine weights and
    hidden states are cached at several teacher layers and stacked along the
    leading axis, and over-caching (ADR-0001) means that axis is *not* the
    extraction configuration's layer set — it is whatever was cached, which is
    deliberately wider. ``layers`` names it, so a signal asking for teacher layer
    19 resolves it against the store rather than against a convention.

    Attributes:
        dtype: The numpy dtype name, e.g. ``float16`` or ``int32``.
        shape: The array shape of a single row's value.
        layers: For a layer-stacked artefact, the teacher layers its leading
            axis corresponds to, in order; ``None`` for artefacts that are not
            layer-stacked.
    """

    model_config = ConfigDict(frozen=True)

    dtype: str
    shape: tuple[int, ...]
    layers: tuple[int, ...] | None = None

    @model_validator(mode="after")
    def _layers_match_the_leading_axis(self) -> VectorSpec:
        if self.layers is not None and (not self.shape or len(self.layers) != self.shape[0]):
            raise ValueError(
                f"layers {self.layers} do not index the leading axis of shape {self.shape}"
            )
        return self

    def layer_index(self, layer: int) -> int:
        """Return the leading-axis index a teacher layer sits at.

        Args:
            layer: The teacher layer to locate.

        Returns:
            Its index along the artefact's leading axis.

        Raises:
            MissingArtifactKindError: If the artefact is not layer-stacked, or
                was not cached at ``layer`` — either way the Group asked for
                something this store does not hold.
        """
        if self.layers is None:
            raise MissingArtifactKindError(
                f"this artefact is not layer-stacked, so teacher layer {layer} has no index"
            )
        if layer not in self.layers:
            raise MissingArtifactKindError(
                f"teacher layer {layer} was not cached; this store holds {list(self.layers)}"
            )
        return self.layers.index(layer)

    @property
    def itemsize(self) -> int:
        """The number of bytes in one encoded value."""
        return int(np.prod(self.shape)) * np.dtype(self.dtype).itemsize

    def encode(self, array: np.ndarray) -> bytes:
        """Encode one array to its fixed-size byte string, checking dtype and shape."""
        if tuple(array.shape) != self.shape:
            raise ArtifactStoreError(f"expected shape {self.shape}, got {tuple(array.shape)}")
        return np.ascontiguousarray(array, dtype=self.dtype).tobytes()

    def decode(self, data: bytes) -> np.ndarray:
        """Decode one fixed-size byte string back to its array."""
        return np.frombuffer(data, dtype=self.dtype).reshape(self.shape)


class StoreMetadata(BaseModel):
    """A store's self-description: its identity, its schema, and its provenance.

    Attributes:
        extraction_fingerprint: The fingerprint the store root is named for.
        vector_kinds: The vector artefact columns present, mapped to their specs.
        corpus_manifest: The source corpus manifest, serialised, so the exact
            examples a store was built over are recorded with it.
    """

    model_config = ConfigDict(frozen=True)

    extraction_fingerprint: str
    vector_kinds: dict[str, VectorSpec]
    corpus_manifest: dict[str, Any] | None = None

    def require_kinds(self, required: Iterable[str]) -> None:
        """Raise if any required artefact kind is absent, naming what is missing.

        Args:
            required: The artefact-kind names a Group needs.

        Raises:
            MissingArtifactKindError: If any required kind is not present, with a
                message listing the missing kinds and what the store does hold.
        """
        missing = [kind for kind in required if kind not in self.vector_kinds]
        if missing:
            raise MissingArtifactKindError(
                f"store {self.extraction_fingerprint[:12]} is missing artefact "
                f"kind(s) {sorted(missing)}; it holds {sorted(self.vector_kinds)}"
            )


@dataclass(frozen=True)
class ArtifactRow:
    """One answer token's artefacts, ready to be written.

    Attributes:
        example_id: The example this token belongs to.
        answer_token_index: The token's position in the sequence — the same index
            the Student's cross-entropy target carries.
        gold_token_id: The gold token id at that position (the CE target).
        correct: Whether the teacher answered this example correctly under free
            generation.
        vectors: The vector artefacts for this token, keyed by artefact kind.
    """

    example_id: str
    answer_token_index: int
    gold_token_id: int
    correct: bool
    vectors: Mapping[str, np.ndarray] = field(default_factory=dict)


def store_root(base_dir: Path, extraction_fingerprint: str) -> Path:
    """Return the store directory for a fingerprint under ``base_dir``.

    The root is the fingerprint itself, so two different extraction
    configurations resolve to two different directories.
    """
    return base_dir / extraction_fingerprint


class ArtifactShardWriter:
    """Writes one worker's artefacts as independent Parquet shards.

    Each worker owns a distinct ``worker_id`` and writes shard files named for
    it, so several workers fill one store with no write coordination. Each
    :meth:`write` call produces one shard file, which is also the unit of resume:
    a batch that completed is on disk and its examples are skipped next time.
    """

    def __init__(self, store: ArtifactStore, worker_id: str) -> None:
        """Open a shard writer for ``worker_id`` against an existing ``store``.

        Sequence numbering continues past any shards this worker already wrote, so
        a resumed run appends new shards rather than overwriting earlier ones.
        """
        self._store = store
        self._worker_id = worker_id
        self._shards_dir = store.root / SHARDS_DIRNAME
        self._shards_dir.mkdir(parents=True, exist_ok=True)
        existing = self._shards_dir.glob(f"shard-{worker_id}-*.parquet")
        self._sequence = 1 + max(
            (int(path.stem.rsplit("-", 1)[1]) for path in existing), default=-1
        )

    def write(self, rows: Sequence[ArtifactRow]) -> Path:
        """Write a batch of rows to a new shard file and return its path.

        Args:
            rows: The answer-token rows to persist. Their vector kinds must match
                the store's schema exactly.

        Returns:
            The path of the shard written.

        Raises:
            ArtifactStoreError: If ``rows`` is empty or a row's vector kinds do
                not match the store schema.
        """
        if not rows:
            raise ArtifactStoreError("refusing to write an empty shard")
        table = self._store.rows_to_table(rows)
        path = self._shards_dir / f"shard-{self._worker_id}-{self._sequence:05d}.parquet"
        pq.write_table(table, path)
        self._sequence += 1
        return path


class ArtifactStore:
    """A fingerprinted, DuckDB-queryable store of teacher artefacts.

    Create one with :meth:`create` (which writes the metadata), then open it by
    construction. Writers append shards; readers query them through DuckDB or
    pull a single artefact back as an array.
    """

    def __init__(self, root: Path) -> None:
        """Open the store rooted at ``root``, loading its metadata.

        Raises:
            ArtifactStoreError: If ``root`` has no store metadata.
        """
        self.root = root
        metadata_path = root / METADATA_FILENAME
        if not metadata_path.exists():
            raise ArtifactStoreError(f"no artifact store at {root} (missing {METADATA_FILENAME})")
        self.metadata = StoreMetadata.model_validate_json(metadata_path.read_text())

    @classmethod
    def create(cls, root: Path, metadata: StoreMetadata) -> ArtifactStore:
        """Create a store at ``root``, writing its metadata, and open it.

        Creating an already-existing store re-reads it if the metadata matches
        and refuses if it does not, so a resumed run cannot append to a store
        built under a different schema.

        Raises:
            ArtifactStoreError: If ``root`` already holds a store with different
                metadata.
        """
        root.mkdir(parents=True, exist_ok=True)
        metadata_path = root / METADATA_FILENAME
        if metadata_path.exists():
            existing = StoreMetadata.model_validate_json(metadata_path.read_text())
            if existing != metadata:
                raise ArtifactStoreError(f"store at {root} already exists with different metadata")
        else:
            metadata_path.write_text(metadata.model_dump_json(indent=2))
        return cls(root)

    # -- writing --------------------------------------------------------------

    def rows_to_table(self, rows: Sequence[ArtifactRow]) -> pa.Table:
        """Encode rows into an Arrow table matching the store schema."""
        columns: dict[str, pa.Array] = {
            EXAMPLE_ID: pa.array([r.example_id for r in rows], pa.string()),
            ANSWER_TOKEN_INDEX: pa.array([r.answer_token_index for r in rows], pa.int64()),
            GOLD_TOKEN_ID: pa.array([r.gold_token_id for r in rows], pa.int64()),
            CORRECT: pa.array([r.correct for r in rows], pa.bool_()),
        }
        for kind, spec in self.metadata.vector_kinds.items():
            encoded = []
            for row in rows:
                if kind not in row.vectors:
                    raise ArtifactStoreError(f"row {row.example_id} is missing kind {kind!r}")
                encoded.append(spec.encode(row.vectors[kind]))
            columns[kind] = pa.array(encoded, pa.binary(spec.itemsize))
        return pa.table(columns)

    def writer(self, worker_id: str) -> ArtifactShardWriter:
        """Return a shard writer for ``worker_id``."""
        return ArtifactShardWriter(self, worker_id)

    # -- reading --------------------------------------------------------------

    def _shard_glob(self) -> str | None:
        shards_dir = self.root / SHARDS_DIRNAME
        if not shards_dir.exists() or not any(shards_dir.glob("*.parquet")):
            return None
        return str(shards_dir / "*.parquet")

    def query(self, sql: str) -> list[tuple[Any, ...]]:
        """Run a SQL query against the store's shards, returning rows.

        The shards are exposed to DuckDB as a view named ``artifacts``; a query
        over an empty store returns no rows.

        Args:
            sql: A SQL statement referencing the ``artifacts`` view.

        Returns:
            The result rows as tuples.
        """
        return self._read(sql)

    def _read(self, sql: str, params: Sequence[Any] | None = None) -> list[tuple[Any, ...]]:
        """Run one SQL statement against the shards, exposed as the ``artifacts`` view.

        This is the single place a DuckDB connection is opened over the store, so
        the shard glob, the empty-store view, and the connection lifecycle are
        defined once. An empty store presents an empty ``artifacts`` view rather
        than an error, so callers need no special case.
        """
        glob = self._shard_glob()
        connection = duckdb.connect()
        try:
            if glob is None:
                connection.execute(
                    f"CREATE TEMP VIEW artifacts AS SELECT NULL AS {EXAMPLE_ID} WHERE FALSE"
                )
            else:
                connection.execute(
                    f"CREATE TEMP VIEW artifacts AS SELECT * FROM read_parquet('{glob}')"
                )
            return connection.execute(sql, params or []).fetchall()
        finally:
            connection.close()

    def example_ids(self) -> set[str]:
        """Return the set of example ids present, for resume.

        A resumed extraction skips these rather than restarting.
        """
        return {row[0] for row in self._read(f"SELECT DISTINCT {EXAMPLE_ID} FROM artifacts")}

    def vector(self, example_id: str, answer_token_index: int, kind: str) -> np.ndarray:
        """Return one decoded vector artefact.

        Args:
            example_id: The example.
            answer_token_index: The answer-token position.
            kind: The vector artefact kind.

        Returns:
            The artefact as an array of its declared dtype and shape.

        Raises:
            MissingArtifactKindError: If ``kind`` is not in the store schema.
            ArtifactStoreError: If no such row exists.
        """
        if kind not in self.metadata.vector_kinds:
            raise MissingArtifactKindError(f"store does not hold kind {kind!r}")
        rows = self._read(
            f"SELECT {kind} FROM artifacts WHERE {EXAMPLE_ID} = ? AND {ANSWER_TOKEN_INDEX} = ?",
            [example_id, answer_token_index],
        )
        if not rows:
            raise ArtifactStoreError(f"no row for ({example_id!r}, {answer_token_index}) in store")
        return self.metadata.vector_kinds[kind].decode(rows[0][0])
