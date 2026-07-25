"""Persisting an assembled corpus so every Group reads the identical examples.

The corpus is assembled once, written here, and then read back by teacher
extraction, training, and evaluation alike. That is what makes the plan's
"every Group trains on the identical corpus" claim checkable rather than
asserted: the file is the corpus, and the manifest beside it records the split
assignment that produced it.

Examples are stored as JSON Lines — one :class:`~ceed_data.example.Example` per
line — so the file streams, appends, and diffs, and a corpus of a few hundred
thousand examples costs nothing to open.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from pathlib import Path

from ceed_data.example import Example

CORPUS_FILENAME = "examples.jsonl"


def write_examples(path: Path, examples: Iterable[Example]) -> int:
    """Write examples to a JSONL file, one per line.

    Args:
        path: The file to write. Its parent is created if absent.
        examples: The examples to write, in order.

    Returns:
        How many examples were written.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w") as handle:
        for example in examples:
            handle.write(example.model_dump_json() + "\n")
            count += 1
    return count


def read_examples(path: Path) -> Iterator[Example]:
    """Stream examples back from a JSONL file written by :func:`write_examples`.

    Args:
        path: The corpus file.

    Yields:
        Each example, in file order.
    """
    with path.open() as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield Example.model_validate_json(line)
