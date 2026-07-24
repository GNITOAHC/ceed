"""Metric emission: JSONL on disk is the source of truth.

Every metric a run emits is written as one JSON object per line to a file on
disk. That file, not a tracking service, is authoritative — results survive the
absence of an account or a network. A tracking service, when one is supplied, is
a passive mirror: it receives the same records, and its absence is never an
error.

A sink writes one run's metric stream from scratch: opening it truncates any
file left by a previous run of the same configuration, so re-running a Group
replaces its metrics rather than appending a second, duplicated stream.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from pathlib import Path
from types import TracebackType
from typing import Any

Tracker = Callable[[Mapping[str, Any]], None]
"""A tracking-service callback receiving each emitted metric record."""


class MetricsSink:
    """A JSONL metrics writer with an optional tracker mirror.

    The sink owns an open file handle for its lifetime; use it as a context
    manager, or call :meth:`close` when done. Each :meth:`log` call writes one
    line to disk and then, if a tracker was supplied, forwards the same record
    to it. Opening the sink truncates the file, so each run writes a clean
    metric stream rather than appending to a previous run's.
    """

    def __init__(self, path: Path, tracker: Tracker | None = None) -> None:
        """Open ``path`` for writing metric records, truncating any prior file.

        Args:
            path: The JSONL file to write. Its parent directory must exist.
            tracker: An optional callback mirrored with each record. When
                omitted, metrics are written to disk only, which is not an error.
        """
        self._path = path
        self._tracker = tracker
        self._handle = path.open("w", encoding="utf-8")

    @property
    def path(self) -> Path:
        """The file this sink writes to."""
        return self._path

    def log(self, record: Mapping[str, Any]) -> None:
        """Append one metric record to disk, then mirror it to the tracker.

        The disk write is the source of truth and happens first; the tracker,
        if present, is a best-effort mirror. A record is flushed immediately so
        a run killed mid-flight leaves every already-logged metric on disk.

        Args:
            record: A JSON-serialisable mapping of metric names to values.
        """
        self._handle.write(json.dumps(dict(record), sort_keys=True) + "\n")
        self._handle.flush()
        if self._tracker is not None:
            self._tracker(record)

    def close(self) -> None:
        """Close the underlying file handle."""
        self._handle.close()

    def __enter__(self) -> MetricsSink:
        """Enter the context manager, returning this sink."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Close the file handle on leaving the context."""
        self.close()
