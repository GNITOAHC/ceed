"""Scoring a Group by decoding its Student over the held-out corpus.

``lmms-eval`` (:mod:`ceed_eval.harness`) exists so B0's zero-shot number is
comparable to published baselines. It cannot, however, score a Group's *trained*
checkpoint without a registered wrapper for every adapter variant, and — more
importantly — the Phase 1 table is a comparison *between* Groups. A comparison is
only valid if every arm is scored by the identical procedure on the identical
examples, so this evaluator is the one all Groups run through: same corpus split,
same greedy decoding (A9), same per-dataset metric (:mod:`ceed_eval.metrics`).

The Student is loaded from its base checkpoint and, when the Group trained one, a
LoRA adapter is applied on top — so B0 scores the shipped weights and B1/B2 score
what they actually trained.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from ceed_core import DecodingConfig, GroupConfig
from ceed_data import Example, ImageStore
from ceed_eval.metrics import score_answer
from ceed_student import EvaluationOutcome

#: Identifies numbers produced by this evaluator on a run record, so a direct
#: score is never mistaken for an lmms-eval score.
DIRECT_HARNESS_VERSION = "ceed-direct-1"


def generate_answer(
    model: Any,
    processor: Any,
    example: Example,
    image_store: ImageStore,
    decoding: DecodingConfig,
) -> str:  # pragma: no cover - needs the real Student and a GPU
    """Greedily decode one answer for an example.

    Args:
        model: The loaded Student.
        processor: Its processor.
        example: The example to answer.
        image_store: Where the example's image bytes live.
        decoding: The decoding settings; greedy is enforced regardless (A9).

    Returns:
        The decoded answer text, stripped.
    """
    import torch
    from ceed_student.dataset import chat_messages, load_image

    image = load_image(image_store, example)
    inputs = processor.apply_chat_template(
        chat_messages(example, image),
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)
    with torch.no_grad():
        generated = model.generate(
            **inputs,
            max_new_tokens=decoding.max_new_tokens,
            do_sample=False,  # greedy everywhere (A9)
            num_beams=1,
        )
    new_tokens = generated[0][inputs["input_ids"].shape[1] :]
    return processor.decode(new_tokens, skip_special_tokens=True).strip()


class DirectEvaluator:
    """Scores a Group by decoding its Student over a held-out corpus split."""

    def __init__(
        self,
        examples: Sequence[Example],
        image_store: ImageStore,
        load_model: Callable[[GroupConfig], tuple[Any, Any]],
        checkpoint_dir: Path | None = None,
        limit: int | None = None,
        progress: Callable[[int, int], None] | None = None,
    ) -> None:
        """Configure the evaluator.

        Args:
            examples: The held-out examples to score on. Every Group must be
                handed the same ones for the comparison to mean anything.
            image_store: Where example images live.
            load_model: Builds the Student and processor for a Group. Injected so
                this module stays importable without the weights.
            checkpoint_dir: A trained checkpoint whose LoRA adapter is applied
                before scoring, or ``None`` to score the base Student.
            limit: An optional cap on examples per dataset, for smoke runs.
            progress: Optional callback invoked with ``(done, total)``.
        """
        self.examples = examples
        self.image_store = image_store
        self.load_model = load_model
        self.checkpoint_dir = checkpoint_dir
        self.limit = limit
        self.progress = progress

    def _selected(self, datasets: Sequence[str]) -> dict[str, list[Example]]:
        """Group the held-out examples by dataset, applying any per-dataset cap."""
        by_dataset: dict[str, list[Example]] = {name: [] for name in datasets}
        for example in self.examples:
            bucket = by_dataset.get(example.dataset)
            if bucket is None:
                continue
            if self.limit is None or len(bucket) < self.limit:
                bucket.append(example)
        return by_dataset

    def evaluate(self, config: GroupConfig) -> EvaluationOutcome:  # pragma: no cover - needs GPU
        """Decode the Group's Student over the held-out split and score it.

        Args:
            config: A resolved Group whose ``evaluation`` names the datasets and
                decoding.

        Returns:
            Per-dataset accuracy, the harness identifier, and the decoding used.

        Raises:
            ValueError: If the Group declares no evaluation.
        """
        if config.evaluation is None:
            raise ValueError(f"Group {config.group_code} declares no evaluation")

        from ceed_student import apply_adapter

        model, processor = self.load_model(config)
        if self.checkpoint_dir is not None:
            model = apply_adapter(model, self.checkpoint_dir)
        model.eval()

        decoding = config.evaluation.decoding
        by_dataset = self._selected(config.evaluation.datasets)
        total = sum(len(v) for v in by_dataset.values())

        accuracies: dict[str, float] = {}
        extra: dict[str, float] = {}
        done = 0
        for dataset, examples in by_dataset.items():
            if not examples:
                # A dataset the split holds none of is left out of the record
                # rather than recorded as NaN: a missing number and a zero score
                # must not look alike, and NaN is not valid JSON.
                print(f"[eval] no {dataset} examples in this split; skipping")
                continue
            scores = []
            for example in examples:
                prediction = generate_answer(model, processor, example, self.image_store, decoding)
                scores.append(score_answer(dataset, prediction, example.answers))
                done += 1
                if self.progress is not None:
                    self.progress(done, total)
            accuracies[dataset] = sum(scores) / len(scores)
            extra[f"{dataset}.n"] = float(len(scores))

        return EvaluationOutcome(
            accuracies=accuracies,
            harness_version=DIRECT_HARNESS_VERSION,
            decoding=decoding,
            extra_metrics=extra,
        )
