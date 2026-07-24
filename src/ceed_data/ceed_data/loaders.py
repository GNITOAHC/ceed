"""Normalising DocVQA, GQA, and ChartQA into the one Example type.

The pure builder functions — ``make_*_example`` and ``normalize_box`` — carry all
the normalisation logic and are exercised on synthetic rows in the fast test
tier. The streaming ``load_*`` generators wrap the Hugging Face ``datasets``
sources and are run against the real corpora out of the fast tier, since they
download gigabytes.
"""

from __future__ import annotations

import io
from collections.abc import Iterator, Sequence

from ceed_data.example import CHARTQA, DOCVQA, GQA, BoundingBox, Example, OcrWord
from ceed_data.images import ImageStore


def normalize_box(
    x: float | int,
    y: float | int,
    width: float | int,
    height: float | int,
    image_width: int,
    image_height: int,
) -> BoundingBox:
    """Convert a pixel-space box to the unit square, clamped to fit.

    Args:
        x: Left edge in pixels.
        y: Top edge in pixels.
        width: Box width in pixels.
        height: Box height in pixels.
        image_width: Source image width in pixels.
        image_height: Source image height in pixels.

    Returns:
        The box in normalised coordinates.

    Raises:
        ValueError: If the image dimensions are not positive.
    """
    if image_width <= 0 or image_height <= 0:
        raise ValueError("image dimensions must be positive")
    nx = min(max(x / image_width, 0.0), 1.0)
    ny = min(max(y / image_height, 0.0), 1.0)
    nw = min(width / image_width, 1.0 - nx)
    nh = min(height / image_height, 1.0 - ny)
    return BoundingBox(x=nx, y=ny, width=nw, height=nh)


def make_docvqa_example(
    source_id: str,
    question: str,
    answers: Sequence[str],
    image_fingerprint: str,
    ocr_words: Sequence[OcrWord],
) -> Example:
    """Build a DocVQA example, carrying its OCR words and boxes.

    DocVQA is intervention-eligible and supports text-substitution
    counterfactuals, so the OCR annotations are preserved verbatim.
    """
    return Example(
        example_id=f"{DOCVQA}:{source_id}",
        dataset=DOCVQA,
        image_fingerprint=image_fingerprint,
        question=question,
        answers=tuple(answers),
        ocr_words=tuple(ocr_words),
        intervention_eligible=True,
    )


def make_gqa_example(
    source_id: str,
    question: str,
    answers: Sequence[str],
    image_fingerprint: str,
    answer_region: BoundingBox,
) -> Example:
    """Build a GQA example, carrying the scene-graph object box tied to the answer.

    GQA is intervention-eligible; its relevant region is the answer-linked object
    box, so the region is annotation-backed rather than inferred.
    """
    return Example(
        example_id=f"{GQA}:{source_id}",
        dataset=GQA,
        image_fingerprint=image_fingerprint,
        question=question,
        answers=tuple(answers),
        answer_region=answer_region,
        intervention_eligible=True,
    )


def make_chartqa_example(
    source_id: str,
    question: str,
    answers: Sequence[str],
    image_fingerprint: str,
) -> Example:
    """Build a ChartQA example, which is never intervention-eligible (A8)."""
    return Example(
        example_id=f"{CHARTQA}:{source_id}",
        dataset=CHARTQA,
        image_fingerprint=image_fingerprint,
        question=question,
        answers=tuple(answers),
        intervention_eligible=False,
    )


# -- streaming loaders ------------------------------------------------------
#
# These wrap the Hugging Face sources and download gigabytes, so they run out of
# the fast tier. Their per-dataset field extraction is the documented assumption
# below; the normalisation they feed into is the tested builders above.


def _encode_png(image: object) -> bytes:  # pragma: no cover - trivial IO helper
    """Encode a PIL image to deterministic PNG bytes for content addressing."""
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")  # type: ignore[attr-defined]
    return buffer.getvalue()


def load_docvqa(
    split: str, image_store: ImageStore, limit: int | None = None
) -> Iterator[Example]:  # pragma: no cover - exercised out of the fast tier
    """Stream DocVQA into Examples, storing each image by content address.

    Assumes rows carry ``questionId``, ``question``, ``answers``, an ``image``,
    and parallel ``words``/``boxes`` OCR annotations in pixel coordinates.
    """
    from datasets import load_dataset

    rows = load_dataset("lmms-lab/DocVQA", "DocVQA", split=split, streaming=True)
    for index, row in enumerate(rows):
        if limit is not None and index >= limit:
            break
        image = row["image"]
        fingerprint = image_store.put(_encode_png(image))
        words = [
            OcrWord(
                text=text,
                box=normalize_box(box[0], box[1], box[2], box[3], image.width, image.height),
            )
            for text, box in zip(row.get("words", []), row.get("boxes", []), strict=False)
        ]
        yield make_docvqa_example(
            source_id=str(row["questionId"]),
            question=row["question"],
            answers=list(row["answers"]),
            image_fingerprint=fingerprint,
            ocr_words=words,
        )


def load_gqa(
    split: str, image_store: ImageStore, limit: int | None = None
) -> Iterator[Example]:  # pragma: no cover - exercised out of the fast tier
    """Stream GQA into Examples, carrying the answer-linked object box.

    Assumes rows carry ``id``, ``question``, ``answer``, an ``image``, and the
    answer object's pixel box as ``answer_box`` = ``(x, y, w, h)``.
    """
    from datasets import load_dataset

    rows = load_dataset("lmms-lab/GQA", "train_all_instructions", split=split, streaming=True)
    for index, row in enumerate(rows):
        if limit is not None and index >= limit:
            break
        image = row["image"]
        fingerprint = image_store.put(_encode_png(image))
        yield make_gqa_example(
            source_id=str(row["id"]),
            question=row["question"],
            answers=[row["answer"]],
            image_fingerprint=fingerprint,
            answer_region=normalize_box(
                row["answer_box"][0],
                row["answer_box"][1],
                row["answer_box"][2],
                row["answer_box"][3],
                image.width,
                image.height,
            ),
        )


def load_chartqa(
    split: str, image_store: ImageStore, limit: int | None = None
) -> Iterator[Example]:  # pragma: no cover - exercised out of the fast tier
    """Stream ChartQA into Examples (never intervention-eligible, A8).

    Assumes rows carry ``id``, ``question``, ``answer``, and an ``image``.
    """
    from datasets import load_dataset

    rows = load_dataset("lmms-lab/ChartQA", split=split, streaming=True)
    for index, row in enumerate(rows):
        if limit is not None and index >= limit:
            break
        fingerprint = image_store.put(_encode_png(row["image"]))
        yield make_chartqa_example(
            source_id=str(row["id"]),
            question=row["question"],
            answers=[row["answer"]],
            image_fingerprint=fingerprint,
        )
