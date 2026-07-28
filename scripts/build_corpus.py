#!/usr/bin/env python
"""Assemble the CEED corpus once, so every Group reads the identical examples.

Streams DocVQA, GQA, and ChartQA from Hugging Face, normalises them into the one
:class:`~ceed_data.example.Example` type, stores each image by content address,
splits deterministically by example id, and writes the corpus plus its manifest.

Run it once. Every later step — teacher extraction, B1/B2 training, and
evaluation of all three Groups — reads this directory, which is what makes the
"identical corpus" claim structural rather than hoped for.

Example:
    uv run python scripts/build_corpus.py --output data/corpus --limit 200
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ceed_data.manifest import CorpusManifest

from ceed_data import (
    CORPUS_FILENAME,
    ImageStore,
    load_chartqa,
    load_docvqa,
    load_gqa,
    split_corpus,
    write_examples,
)

# The held-out fractions every Group splits by. Recorded in the manifest, so a
# corpus assembled in week two is checkably the same as one from week ten.
FRACTIONS = {"train": 0.8, "validation": 0.1, "test": 0.1}

LOADERS = {"docvqa": load_docvqa, "gqa": load_gqa, "chartqa": load_chartqa}

# The source split each dataset is drawn from.
SOURCE_SPLITS = {"docvqa": "validation", "gqa": "train", "chartqa": "train"}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the command line."""
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument(
        "--output", type=Path, default=Path("data/corpus"), help="corpus directory to write"
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["docvqa", "gqa", "chartqa"],
        choices=sorted(LOADERS),
        help="which source datasets to include",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=200,
        help=(
            "examples per dataset; zero or negative means the whole source split "
            "(keep small first — the sources are gigabytes)"
        ),
    )
    parser.add_argument("--seed", type=int, default=0, help="the recorded split seed")
    return parser.parse_args(argv)


def resolve_limit(limit: int) -> int | None:
    """Translate the ``--limit`` flag into the loaders' cap.

    The loaders take ``None`` for "no cap" and stop at ``index >= limit``
    otherwise, which makes a literal ``0`` mean *zero examples* — an empty
    corpus, silently. Since nobody asks for an empty corpus and everybody
    eventually asks for a whole one, any non-positive value is read as "take
    everything".

    Args:
        limit: The value passed on the command line.

    Returns:
        ``None`` to take the whole source split, or the positive cap.
    """
    return None if limit <= 0 else limit


def main(argv: list[str] | None = None) -> int:
    """Assemble the corpus and write it with its manifest."""
    args = parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)
    image_store = ImageStore(args.output / "images")

    limit = resolve_limit(args.limit)
    described = "no limit" if limit is None else f"limit {limit}"

    examples = []
    for dataset in args.datasets:
        source_split = SOURCE_SPLITS[dataset]
        print(f"[corpus] streaming {dataset} ({source_split}), {described} ...", flush=True)
        try:
            for example in LOADERS[dataset](source_split, image_store, limit=limit):
                examples.append(example)
        except Exception as error:
            print(f"[corpus] WARNING: {dataset} failed: {error}", file=sys.stderr)
            print(
                f"[corpus] continuing without {dataset}; check the dataset id and "
                "your Hugging Face access",
                file=sys.stderr,
            )
    if not examples:
        print("[corpus] no examples assembled; nothing written", file=sys.stderr)
        return 1

    splits = split_corpus(examples, seed=args.seed, fractions=FRACTIONS)
    written = write_examples(args.output / CORPUS_FILENAME, examples)
    manifest = CorpusManifest.from_splits(splits, seed=args.seed, fractions=FRACTIONS)
    manifest.write(args.output)

    print(f"[corpus] wrote {written} examples to {args.output / CORPUS_FILENAME}")
    for name, ids in manifest.splits.items():
        print(f"[corpus]   {name}: {len(ids)}")
    print(f"[corpus]   per dataset: {manifest.dataset_counts}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
