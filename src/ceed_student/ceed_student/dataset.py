"""Turning the corpus and the artifact store into training batches.

This is the store-backed dataloader the training loop is handed: it reads the
assembled corpus, renders each example into the Student's chat format, tokenises
prompt and gold answer, and pairs each answer token with the teacher's cached
top-k logits for that token.

**Answer-token keying.** The store is keyed by ``(example_id,
answer_token_index)``. Here that index is the answer token's *ordinal* — its 0-based
position within the gold answer — not its absolute position in the sequence. The
Teacher and the Student share a tokenizer, so the answer's token sequence is
identical for both, but their prompts are not the same length (the vision towers
emit different numbers of image placeholder tokens). An ordinal is therefore the
only key that means the same thing on both sides. Extraction writes the same
ordinal, so the two agree by construction.

**Alignment.** Answer token ``t`` is *predicted* at the preceding position, so a
batch's ``answer_token_positions`` are the positions the Student makes those
predictions at and ``gold_token_ids`` are the tokens they should predict. This is
the same convention the teacher-side extraction uses.
"""

from __future__ import annotations

import io
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ceed_core import ArtifactStore
from ceed_data import Example, ImageStore
from ceed_student.training import TrainingBatch

# The artefact kinds the shared backbone reads. Named here rather than imported
# from ceed_teacher so that training never depends on the teacher package.
TOP_K_LOGIT_IDS = "top_k_logit_ids"
TOP_K_LOGIT_VALUES = "top_k_logit_values"


def answer_text(example: Example) -> str:
    """Return the gold answer a Group trains the Student on.

    An example may carry several accepted answers; the first is the one
    supervised, so every Group teacher-forces the identical string.
    """
    if not example.answers:
        raise ValueError(f"example {example.example_id} has no gold answer")
    return example.answers[0]


# The short-answer instruction every prompt carries. VQA gold answers are a word
# or a number, and an instruction-tuned model otherwise replies in a sentence —
# "The total written in the image is **28**." scores zero against gold "28" under
# every one of the three metrics. This is the standard VQA convention (lmms-eval
# appends the same instruction), and it lives here, in the one prompt builder
# training and evaluation share, so a Group can never train on one prompt and be
# scored on another.
SHORT_ANSWER_INSTRUCTION = "Answer the question using a single word or phrase."


def chat_messages(example: Example, image: Any) -> list[dict[str, Any]]:
    """Render one example as the chat turns the Student is prompted with.

    Args:
        example: The example to render.
        image: The decoded PIL image for the example.

    Returns:
        A single user turn carrying the image, the question, and the shared
        short-answer instruction.
    """
    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": f"{example.question}\n{SHORT_ANSWER_INSTRUCTION}"},
            ],
        }
    ]


def load_image(image_store: ImageStore, example: Example) -> Any:
    """Decode an example's image from the content-addressed store."""
    from PIL import Image

    return Image.open(io.BytesIO(image_store.get(example.image_fingerprint))).convert("RGB")


def encode_example(
    processor: Any, example: Example, image_store: ImageStore
) -> tuple[dict[str, torch.Tensor], int, torch.Tensor]:
    """Tokenise one example into Student inputs plus its answer-token targets.

    The prompt is rendered through the Student's chat template with the
    generation prompt appended, then the gold answer's tokens are concatenated —
    this is teacher forcing, and it makes the Student's targets exactly the gold
    answer tokens (ADR-0002).

    Args:
        processor: The Student's processor.
        example: The example to encode.
        image_store: Where the example's image bytes live.

    Returns:
        The Student inputs, the prompt length in tokens, and the gold answer's
        token ids.
    """
    image = load_image(image_store, example)
    prompt = processor.apply_chat_template(
        chat_messages(example, image),
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )
    prompt_ids = prompt["input_ids"][0]
    answer_ids = torch.tensor(
        processor.tokenizer(answer_text(example), add_special_tokens=False)["input_ids"],
        dtype=prompt_ids.dtype,
    )

    inputs = dict(prompt)
    inputs["input_ids"] = torch.cat([prompt_ids, answer_ids]).unsqueeze(0)
    if "attention_mask" in inputs:
        inputs["attention_mask"] = torch.ones_like(inputs["input_ids"])
    return inputs, int(prompt_ids.shape[0]), answer_ids


def topk_from(
    rows: Mapping[str, np.ndarray], example_id: str, n_answer_tokens: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return one example's cached teacher top-k logits, in answer-token order.

    Args:
        rows: The example's artefacts, as read in bulk from the store.
        example_id: The example, for the error message.
        n_answer_tokens: How many answer tokens the Student encoded.

    Returns:
        The top-k ids and values, each ``[answer_tokens, k]``.

    Raises:
        ValueError: If the store holds a different number of answer tokens for
            this example than the Student encoded — a tokenisation mismatch that
            would silently misalign every token's supervision after the first.
    """
    ids = rows.get(TOP_K_LOGIT_IDS)
    if ids is None:
        raise ValueError(
            f"the store holds no rows for {example_id!r}; it was built from a "
            "different corpus, or extraction has not covered this example"
        )
    if ids.shape[0] != n_answer_tokens:
        raise ValueError(
            f"the store holds {ids.shape[0]} answer tokens for {example_id!r} but the "
            f"Student encoded {n_answer_tokens}; the two were built from different "
            "prompts or tokenizers, and training would misalign supervision"
        )
    return (
        torch.from_numpy(ids.astype(np.int64)),
        torch.from_numpy(rows[TOP_K_LOGIT_VALUES].astype(np.float32)),
    )


def artefacts_from(rows: Mapping[str, np.ndarray], kinds: Sequence[str]) -> dict[str, torch.Tensor]:
    """Return the auxiliary artefacts a Group's signals read, as tensors.

    Half-precision artefacts are widened to float32: the loss is computed in the
    Student's compute dtype, and a half-precision target would cap the precision
    of every comparison against it.

    Args:
        rows: The example's artefacts, as read in bulk from the store.
        kinds: The artefact kinds the Group's signals declared.

    Returns:
        Each kind as ``[answer_tokens, ...]``, keyed by kind.
    """
    return {kind: torch.from_numpy(rows[kind].astype(np.float32)) for kind in kinds}


def placeholder_topk(answer_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return an inert top-k standing in for a teacher that was never consulted.

    B1 trains without a teacher, so it has no cached logits — but the backbone's
    signature is shared and fixed. The gold token alone is supplied as the
    "top-k", which is well-defined and contributes exactly nothing because B1's
    ``kd_weight`` is zero. A Group with a non-zero weight must never use this;
    :func:`build_batches` enforces that.
    """
    ids = answer_ids.unsqueeze(-1).to(torch.int64)
    return ids, torch.zeros_like(ids, dtype=torch.float32)


def build_batches(
    processor: Any,
    examples: Sequence[Example],
    image_store: ImageStore,
    store: ArtifactStore | None,
    kd_weight: float,
    artefact_kinds: Sequence[str] = (),
    progress: Callable[[int, int], None] | None = None,
) -> list[TrainingBatch]:
    """Build one training batch per example, pairing it with the teacher's cache.

    Args:
        processor: The Student's processor.
        examples: The training examples, already split and ordered.
        image_store: Where example images live.
        store: The artifact store holding the teacher's top-k logits, or ``None``
            for a teacher-free Group (B1).
        kd_weight: The Group's distillation weight, used to refuse a teacher-free
            build for a Group that actually distils.
        artefact_kinds: The further artefact kinds the Group's auxiliary signals
            declared — read into each batch alongside the backbone's targets.
        progress: Optional callback invoked with ``(done, total)``.

    Returns:
        One :class:`~ceed_student.training.TrainingBatch` per example.

    Raises:
        ValueError: If ``store`` is ``None`` while the Group needs one, either to
            distil or to feed an auxiliary signal.
    """
    if store is None and kd_weight != 0.0:
        raise ValueError(
            "a Group with kd_weight != 0 distils from the teacher and needs an "
            "artifact store; run teacher extraction first"
        )
    if store is None and artefact_kinds:
        raise ValueError(
            f"this Group's auxiliary signals read {sorted(artefact_kinds)} from the "
            "artifact store, but no store was supplied; run teacher extraction first"
        )

    # Every cached row this Group needs, in one query. Read a cell at a time and
    # each call scans the whole shard glob, so a corpus-sized dataloader spends
    # longer assembling batches than training on them.
    cached: dict[str, dict[str, np.ndarray]] = {}
    if store is not None:
        wanted = [TOP_K_LOGIT_IDS, TOP_K_LOGIT_VALUES, *artefact_kinds]
        cached = store.vectors_by_example(wanted, [e.example_id for e in examples])

    batches: list[TrainingBatch] = []
    for index, example in enumerate(examples):
        inputs, prompt_length, answer_ids = encode_example(processor, example, image_store)
        n_answer = int(answer_ids.shape[0])
        if n_answer == 0:
            continue
        # Answer token t is predicted at position t-1; the first answer token is
        # predicted at the prompt's final position.
        positions = torch.arange(prompt_length - 1, prompt_length - 1 + n_answer)
        artefacts: dict[str, torch.Tensor] = {}
        if store is None:
            topk_ids, topk_values = placeholder_topk(answer_ids)
        else:
            rows = cached.get(example.example_id, {})
            topk_ids, topk_values = topk_from(rows, example.example_id, n_answer)
            artefacts = artefacts_from(rows, artefact_kinds)
        batches.append(
            TrainingBatch(
                student_inputs=inputs,
                answer_token_positions=positions,
                gold_token_ids=answer_ids.to(torch.long),
                teacher_topk_ids=topk_ids,
                teacher_topk_values=topk_values,
                artefacts=artefacts,
            )
        )
        if progress is not None:
            progress(index + 1, len(examples))
    return batches


def load_corpus_split(
    corpus_dir: Path, split: str
) -> list[Example]:  # pragma: no cover - thin IO wrapper
    """Read one split's examples from an assembled corpus directory."""
    from ceed_data.corpus import CORPUS_FILENAME, read_examples
    from ceed_data.manifest import CorpusManifest

    manifest = CorpusManifest.read(corpus_dir)
    wanted = set(manifest.splits[split])
    return [e for e in read_examples(corpus_dir / CORPUS_FILENAME) if e.example_id in wanted]
