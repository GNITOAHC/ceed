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
* **B3, B4 and B5** each declare one auxiliary signal on top of B2's objective,
  so each additionally reads the artefact kind its signal names — and refuses to
  start if the store never cached it.

Examples:
    # zero-shot baseline on 50 held-out examples per dataset
    uv run python scripts/run_group.py --group b0 --limit 50

    # supervised fine-tuning, no teacher
    uv run python scripts/run_group.py --group b1 --steps 200 --limit 50

    # the primary baseline: needs scripts/extract_teacher_artifacts.py to have run
    uv run python scripts/run_group.py --group b2 --steps 200 --limit 50

    # a Group with an auxiliary signal: needs that extraction to have run --all
    uv run python scripts/run_group.py --group b5 --steps 200 --limit 50
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from ceed_core.config import load_overlay, merge_overlays
from ceed_student.signals import build_signals, signal_artefact_kinds

from ceed_core import ArtifactStore, GroupConfig, extraction_fingerprint, store_root
from ceed_data import ImageStore
from ceed_student import AccelerateTrainer, CeedStudent, build_batches, run_group


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the command line."""
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
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
    """Open the artifact store for this Group's fingerprint, if the Group needs one.

    A Group needs the store if it distils (``kd_weight`` non-zero) or if any of
    its auxiliary signals reads a cached artefact — which all of B3, B4 and B5
    do.
    """
    needs_teacher = config.training is not None and config.training.backbone.kd_weight != 0.0
    if not needs_teacher and not config.auxiliary_signals:
        return None
    root = store_root(args.store, extraction_fingerprint(config))
    if not (root / "store_metadata.json").exists():
        extra = " --all" if config.auxiliary_signals else ""
        raise SystemExit(
            f"Group {config.group_code} reads the Teacher's cached artefacts but no "
            f"artifact store exists at {root}.\n"
            f"Run: uv run python scripts/extract_teacher_artifacts.py "
            f"--corpus {args.corpus} --store {args.store} --group {args.group}{extra}"
        )
    return ArtifactStore(root)


class StudentLoader:
    """Loads the Student once and hands the same one back, until released.

    Training and evaluation must **not** share a Student, and the reason is not
    tidiness. ``get_peft_model`` injects the LoRA layers into the module tree in
    place, so after training the loaded model already carries the adapter;
    handing that same object to the evaluator, which applies the adapter again
    from the checkpoint, stacks a second injection on the first. Whether that
    happens to compose or to double the adaptation, the number it produces is not
    the number the saved checkpoint would produce for anyone else.

    So the evaluator gets its own loader, and :meth:`release` frees the trained
    one first — two 8B Students in fp16 are 32 GB and will not sit on one V100
    together. Reloading also makes the score a statement about what is *on disk*,
    which is what a later run, a merge, or a reader downloading the checkpoint
    will actually get.
    """

    def __init__(self, args: argparse.Namespace, label: str) -> None:
        """Configure a loader; nothing is loaded until it is first asked for."""
        self.args = args
        self.label = label
        self._cache: tuple[Any, Any] | None = None

    def __call__(self, config: GroupConfig) -> tuple[Any, Any]:
        """Return the Student and processor, loading them on first use."""
        if self._cache is None:
            from ceed_core import DecodingConfig
            from ceed_student import load_student

            decoding = config.evaluation.decoding if config.evaluation is not None else None
            print(f"[run] loading the Student for {self.label} ...", flush=True)
            self._cache = load_student(
                config.student.model,
                decoding or DecodingConfig(),
                dtype=config.student.dtype,
                device=self.args.device,
            )
        return self._cache

    def release(self) -> None:
        """Drop the loaded Student and give its memory back to the device."""
        if self._cache is None:
            return
        self._cache = None
        import gc

        import torch

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f"[run] released the Student held for {self.label}", flush=True)


def main(argv: list[str] | None = None) -> int:
    """Resolve the Group, build its seams, run it, and report the record."""
    args = parse_args(argv)
    config = resolve_config(args)
    print(f"[run] group {config.group_code}  seed {config.seed}  mode {config.param_efficiency}")

    from ceed_student.dataset import load_corpus_split

    image_store = ImageStore(args.corpus / "images")
    load_for_training = StudentLoader(args, "training")
    load_for_evaluation = StudentLoader(args, "evaluation")

    trainer = None
    if config.training is not None:
        store = open_store(args, config)
        train_examples = load_corpus_split(args.corpus, args.train_split)
        if args.train_limit is not None:
            train_examples = train_examples[: args.train_limit]
        print(f"[run] {len(train_examples)} training examples from '{args.train_split}'")

        def build_student(cfg: GroupConfig) -> Any:
            model, _ = load_for_training(cfg)
            return CeedStudent(model)

        # The signals are built before anything expensive happens, so a Group
        # whose artefacts were never extracted fails here rather than at hour six.
        def make_signals(cfg: GroupConfig) -> Any:
            model, _ = load_for_training(cfg)
            return build_signals(cfg, store, int(model.config.text_config.hidden_size))

        signal_kinds = signal_artefact_kinds(config)
        if signal_kinds:
            print(f"[run] auxiliary signals read {signal_kinds} from {store.root}")

        def build_training_batches(cfg: GroupConfig) -> Any:
            _, processor = load_for_training(cfg)
            assert cfg.training is not None
            print("[run] encoding training batches ...", flush=True)
            return build_batches(
                processor,
                train_examples,
                image_store,
                store,
                kd_weight=cfg.training.backbone.kd_weight,
                artefact_kinds=signal_kinds,
            )

        trainer = AccelerateTrainer(
            build_student=build_student,
            build_batches=build_training_batches,
            build_signals=make_signals,
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

        # The trained Student is freed before the fresh one is loaded: two 8B
        # Students in fp16 do not fit on one card, and the point of reloading is
        # that the score describes the checkpoint on disk rather than the mutated
        # object training left behind.
        def load_fresh_student(cfg: GroupConfig) -> Any:
            load_for_training.release()
            return load_for_evaluation(cfg)

        evaluator = DirectEvaluator(
            examples=eval_examples,
            image_store=image_store,
            load_model=load_fresh_student,
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
