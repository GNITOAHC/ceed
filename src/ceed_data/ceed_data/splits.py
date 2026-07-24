"""Deterministic, seed-recorded corpus splits.

A held-out set is only genuinely held out if every Group splits the corpus the
same way. Assignment is therefore a pure function of the example's stable id and
a recorded seed — not of iteration order, insertion order, or a global RNG — so
the same example lands in the same split on every machine and in every run.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Mapping

from ceed_data.example import Example


def _unit_hash(seed: int, example_id: str) -> float:
    """Map ``(seed, example_id)`` to a stable value in ``[0, 1)``."""
    digest = hashlib.sha256(f"{seed}:{example_id}".encode()).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


def assign_split(example_id: str, seed: int, fractions: Mapping[str, float]) -> str:
    """Assign one example to a split deterministically.

    Splits are laid end to end over ``[0, 1)`` in the order ``fractions`` is
    given, and the example's stable hash selects the interval it falls in.

    Args:
        example_id: The example's stable identifier.
        seed: The recorded split seed.
        fractions: Split names mapped to their fractions; must sum to 1.

    Returns:
        The name of the split this example belongs to.

    Raises:
        ValueError: If the fractions do not sum to 1.
    """
    total = sum(fractions.values())
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"split fractions must sum to 1, got {total}")
    position = _unit_hash(seed, example_id)
    cumulative = 0.0
    for name, fraction in fractions.items():
        cumulative += fraction
        if position < cumulative:
            return name
    return list(fractions)[-1]  # guard against floating-point drift at 1.0


def split_corpus(
    examples: Iterable[Example], seed: int, fractions: Mapping[str, float]
) -> dict[str, list[Example]]:
    """Partition examples into named splits deterministically.

    Args:
        examples: The examples to split.
        seed: The recorded split seed.
        fractions: Split names mapped to their fractions; must sum to 1.

    Returns:
        Each split name mapped to its examples, in input order.
    """
    result: dict[str, list[Example]] = {name: [] for name in fractions}
    for example in examples:
        result[assign_split(example.example_id, seed, fractions)].append(example)
    return result
