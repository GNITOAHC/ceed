#!/usr/bin/env python
"""Run one Group end to end: train it if it trains, then score it.

This is the command line over :func:`ceed_student.run.run_group`, the single
entry point every arm of the comparison goes through. The Group's YAML overlays
are resolved, the trainer and evaluator are built for it, and a run record lands
under the output directory named by the configuration hash.

Which pieces are built depends on the Group, not on the flags:

* **B0** declares an evaluation and no training, so only the evaluator is built
  and the shipped Student is scored.
* **B1** declares training with ``kd_weight: 0``, so it trains without a teacher
  and needs no artifact store.
* **B2** declares training with ``kd_weight: 1``, so it reads the Teacher's cached
  top-k logits from the store and refuses to start if they are absent.

Examples:
    # zero-shot baseline on 50 held-out examples per dataset
    uv run python scripts/run_group.py --group b0 --limit 50

    # supervised fine-tuning, no teacher
    uv run python scripts/run_group.py --group b1 --steps 200 --limit 50

    # the primary baseline: needs scripts/extract_teacher_logits.py to have run
    uv run python scripts/run_group.py --group b2 --steps 200 --limit 50
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from ceed_core.config import load_overlay, merge_overlays

from ceed_core import ArtifactStore, GroupConfig, extraction_fingerprint, store_root
from ceed_data import ImageStore
from ceed_student import AccelerateTrainer, CeedStudent, build_batches, run_group


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the command line."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--group", required=True, help="group overlay name, e.g. b0, b1, b2")
    parser.add_argument("--configs", type=Path, default=Path("configs"))
    parser.add_argument("--corpus", type=Path, default=Path("data/corpus"))
    parser.add_argument("--store", type=Path, default=Path("data/store"))
    parser.add_argument("--output", type=Path, default=Path("runs"))
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--eval-split", default="validation")
    parser.add_argument(
        "--limit", type=int, default=None, help="cap on evaluation examples per dataset"
    )
    parser.add_argument(
        "--train-limit", type=int, default=None, help="cap on training examples encoded"
    )
    parser.add_argument("--steps", type=int, default=None, help="override the config's step budget")
    parser.add_argument("--seed", type=int, default=None, help="override the config's seed")
    parser.add_argument(
        "--param-efficiency",
        choices=["lora", "full"],
        default=None,
        help="override the config's mode (ADR-0005: only 'full' may reach the Phase 1 table)",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--skip-eval", action="store_true", help="train only, do not score")
    return parser.parse_args(argv)


def resolve_config(args: argparse.Namespace) -> GroupConfig:
    """Resolve the Group's configuration, applying any command-line overrides."""
    merged = merge_overlays(
        [
            load_overlay(args.configs / "base.yaml"),
            load_overlay(args.configs / "student.yaml"),
            load_overlay(args.configs / "groups" / f"{args.group}.yaml"),
        ]
    )
    if args.seed is not None:
        merged["seed"] = args.seed
    if args.param_efficiency is not None:
        merged["param_efficiency"] = args.param_efficiency
    if args.steps is not None and merged.get("training") is not None:
        merged["training"] = {**merged["training"], "steps": args.steps}
    return GroupConfig.model_validate(merged)


def open_store(args: argparse.Namespace, config: GroupConfig) -> ArtifactStore | None:
    """Open the artifact store for this Group's fingerprint, if the Group distils."""
    if config.training is None or config.training.backbone.kd_weight == 0.0:
        return None
    root = store_root(args.store, extraction_fingerprint(config))
    if not (root / "store_metadata.json").exists():
        raise SystemExit(
            f"Group {config.group_code} distils from the Teacher but no artifact store "
            f"exists at {root}.\nRun: uv run python scripts/extract_teacher_logits.py "
            f"--corpus {args.corpus} --store {args.store} --group {args.group}"
        )
    return ArtifactStore(root)


def make_model_loader(args: argparse.Namespace):
    """Return a loader that builds the Student and processor for a Group, once."""
    cache: dict[str, tuple[Any, Any]] = {}

    def load(config: GroupConfig) -> tuple[Any, Any]:
        if "model" not in cache:
            from ceed_student import load_student

            decoding = config.evaluation.decoding if config.evaluation is not None else None
            from ceed_core import DecodingConfig

            cache["model"] = load_student(
                config.student.model,
                decoding or DecodingConfig(),
                dtype=config.student.dtype,
                device=args.device,
            )
        return cache["model"]

    return load


def main(argv: list[str] | None = None) -> int:
    """Resolve the Group, build its seams, run it, and report the record."""
    args = parse_args(argv)
    config = resolve_config(args)
    print(f"[run] group {config.group_code}  seed {config.seed}  mode {config.param_efficiency}")

    from ceed_student.dataset import load_corpus_split

    image_store = ImageStore(args.corpus / "images")
    load_model = make_model_loader(args)

    trainer = None
    if config.training is not None:
        store = open_store(args, config)
        train_examples = load_corpus_split(args.corpus, args.train_split)
        if args.train_limit is not None:
            train_examples = train_examples[: args.train_limit]
        print(f"[run] {len(train_examples)} training examples from '{args.train_split}'")

        def build_student(cfg: GroupConfig) -> Any:
            model, _ = load_model(cfg)
            return CeedStudent(model)

        def build_training_batches(cfg: GroupConfig) -> Any:
            _, processor = load_model(cfg)
            assert cfg.training is not None
            print("[run] encoding training batches ...", flush=True)
            return build_batches(
                processor,
                train_examples,
                image_store,
                store,
                kd_weight=cfg.training.backbone.kd_weight,
            )

        trainer = AccelerateTrainer(
            build_student=build_student,
            build_batches=build_training_batches,
            output_root=args.output / "checkpoints" / config.group_code.lower(),
        )

    evaluator = None
    if config.evaluation is not None and not args.skip_eval:
        from ceed_eval import DirectEvaluator

        eval_examples = load_corpus_split(args.corpus, args.eval_split)
        print(f"[run] {len(eval_examples)} evaluation examples from '{args.eval_split}'")
        checkpoint = (
            args.output / "checkpoints" / config.group_code.lower() / "checkpoint"
            if config.training is not None
            else None
        )
        evaluator = DirectEvaluator(
            examples=eval_examples,
            image_store=image_store,
            load_model=load_model,
            checkpoint_dir=checkpoint,
            limit=args.limit,
            progress=lambda done, total: (
                print(f"[eval] {done}/{total}", flush=True) if done % 25 == 0 else None
            ),
        )

    record = run_group(config, args.output, trainer=trainer, evaluator=evaluator)

    print("\n[run] === run record ===")
    print(f"[run] group            {record.group_code}")
    print(f"[run] param_efficiency {record.param_efficiency}")
    print(f"[run] config_hash      {record.config_hash}")
    print(f"[run] checkpoint       {record.checkpoint_dir}")
    print(f"[run] metrics          {json.dumps(record.metrics, indent=2)}")
    print(f"[run] written to       {args.output / record.config_hash}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
