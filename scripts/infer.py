#!/usr/bin/env python
"""Ask a trained Group's Student a question about an image.

Loads the base Student plus the Group's trained adapter and decodes an answer
greedily, using the same prompt the Group was trained and scored under — so what
you see here is what the reported number was measured on.

Point it at a run record and it resolves the checkpoint itself:

    uv run python scripts/infer.py \
        --run runs/<config_hash> --image page.png --question "What is the total?"

Or at a checkpoint directly, and ask several questions at once:

    uv run python scripts/infer.py \
        --checkpoint runs/checkpoints/b1/checkpoint \
        --image page.png --question "What is the total?" --question "Who signed it?"

With no ``--image`` and no ``--question`` it replays examples from the corpus,
printing the prediction beside the gold answer and its score — the quickest way
to see what a Group actually learned:

    uv run python scripts/infer.py --run runs/<config_hash> --corpus data/corpus --limit 10
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ceed_core import DecodingConfig, RunRecord


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the command line."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--run", type=Path, help="a run directory holding run_record.json")
    source.add_argument("--checkpoint", type=Path, help="a checkpoint directory")
    parser.add_argument(
        "--model",
        default="google/gemma-4-e4b-it",
        help="the base Student the Group trained from",
    )
    parser.add_argument("--image", type=Path, help="an image to ask about")
    parser.add_argument("--question", action="append", default=[], help="a question (repeatable)")
    parser.add_argument("--corpus", type=Path, help="replay examples from this corpus instead")
    parser.add_argument("--split", default="validation")
    parser.add_argument("--limit", type=int, default=10, help="how many examples to replay")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--base",
        action="store_true",
        help="ignore the adapter and answer with the untrained Student (the B0 behaviour)",
    )
    return parser.parse_args(argv)


def resolve_checkpoint(args: argparse.Namespace) -> Path | None:
    """Return the checkpoint to load, from a run record or a direct path."""
    if args.base:
        return None
    if args.checkpoint is not None:
        return args.checkpoint
    record = RunRecord.read(args.run)
    if record.checkpoint_dir is None:
        print(
            f"[infer] {record.group_code} trained nothing (it is a zero-shot Group); "
            "answering with the base Student",
            file=sys.stderr,
        )
        return None
    print(f"[infer] {record.group_code}, trained {record.param_efficiency}")
    return Path(record.checkpoint_dir)


def main(argv: list[str] | None = None) -> int:
    """Load the Group's Student and answer questions or replay the corpus."""
    args = parse_args(argv)
    checkpoint = resolve_checkpoint(args)
    decoding = DecodingConfig(max_new_tokens=args.max_new_tokens)

    from ceed_student import load_student, load_trained_student

    if checkpoint is None:
        model, processor = load_student(args.model, decoding, dtype=args.dtype, device=args.device)
        model.eval()
    else:
        print(f"[infer] loading {args.model} + adapter from {checkpoint}")
        model, processor = load_trained_student(
            args.model, checkpoint, decoding, dtype=args.dtype, device=args.device
        )

    from ceed_eval.direct import generate_answer

    from ceed_data import Example, ImageStore
    from ceed_eval import score_answer

    if args.image is not None:
        if not args.question:
            print("[infer] --image needs at least one --question", file=sys.stderr)
            return 1
        # Put the image in a scratch content-addressed store so the shared prompt
        # builder — the one training and evaluation use — can render it.
        store = ImageStore(Path(".ceed-infer-images"))
        fingerprint = store.put(args.image.read_bytes())
        for question in args.question:
            example = Example(
                example_id=f"docvqa:{fingerprint[:8]}",
                dataset="docvqa",
                image_fingerprint=fingerprint,
                question=question,
                answers=("",),
            )
            answer = generate_answer(model, processor, example, store, decoding)
            print(f"\nQ: {question}\nA: {answer}")
        return 0

    if args.corpus is None:
        print("[infer] give either --image with --question, or --corpus", file=sys.stderr)
        return 1

    from ceed_student import load_corpus_split

    image_store = ImageStore(args.corpus / "images")
    examples = load_corpus_split(args.corpus, args.split)[: args.limit]
    print(f"[infer] replaying {len(examples)} examples from '{args.split}'\n")

    scores = []
    for example in examples:
        answer = generate_answer(model, processor, example, image_store, decoding)
        score = score_answer(example.dataset, answer, example.answers)
        scores.append(score)
        mark = "OK  " if score > 0.5 else "MISS"
        print(f"[{mark}] {example.example_id}")
        print(f"       Q:    {example.question}")
        print(f"       pred: {answer!r}")
        print(f"       gold: {list(example.answers)}   score {score:.3f}")
    if scores:
        print(f"\n[infer] mean score over {len(scores)}: {sum(scores) / len(scores):.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
