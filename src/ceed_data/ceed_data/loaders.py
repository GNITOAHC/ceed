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
    answer_region: BoundingBox | None = None,
) -> Example:
    """Build a GQA example, carrying the scene-graph object box tied to the answer.

    GQA is intervention-eligible; its relevant region is the answer-linked object
    box, so the region is annotation-backed rather than inferred.

    The region is optional because the published source does not carry it. GQA's
    answer-linked boxes live in the scene-graph release, not in the question rows
    ``lmms-lab/GQA`` serves, so an example assembled from that source has
    ``answer_region=None``. That costs nothing for B0 to B5, which never read it,
    and it is the region-based interventions in the later phases that will have to
    supply the scene graph rather than assume it arrived with the questions.
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


#: How many questions the balanced GQA split carries per image, roughly. Used to
#: decide how many images to pull for a requested number of examples; being wrong
#: costs a few extra images or a short second pass, never correctness.
GQA_QUESTIONS_PER_IMAGE = 8


def load_gqa(
    split: str, image_store: ImageStore, limit: int | None = None
) -> Iterator[Example]:  # pragma: no cover - exercised out of the fast tier
    """Stream GQA into Examples by joining its questions to its images.

    Unlike DocVQA and ChartQA, ``lmms-lab/GQA`` does not serve one row per
    question-with-image. It splits into ``{split}_balanced_instructions`` — the
    questions, text only, carrying an ``imageId`` — and ``{split}_balanced_images``
    — the images, keyed by that same id. Neither is usable alone, so this joins
    them.

    The join is **image-major**: a prefix of the images is taken, then the
    questions are streamed and kept where their image is in hand. The other
    direction — take the first N questions, then hunt their images — reads the
    entire image config, because a question prefix references ids scattered
    through all ~72k of them. This way the images downloaded are exactly the
    images used.

    Assumes question rows carry ``id``, ``imageId``, ``question`` and ``answer``,
    and image rows carry ``id`` and ``image``. ``answer_region`` is left unset:
    the answer-linked box is in the scene-graph release, not in this source.
    """
    from datasets import load_dataset

    image_budget = None if limit is None else max(1, -(-limit // GQA_QUESTIONS_PER_IMAGE))
    image_rows = load_dataset(
        "lmms-lab/GQA", f"{split}_balanced_images", split=split, streaming=True
    )
    fingerprints: dict[str, str] = {}
    for index, row in enumerate(image_rows):
        if image_budget is not None and index >= image_budget:
            break
        fingerprints[str(row["id"])] = image_store.put(_encode_png(row["image"]))

    question_rows = load_dataset(
        "lmms-lab/GQA", f"{split}_balanced_instructions", split=split, streaming=True
    )
    yielded = 0
    for row in question_rows:
        if limit is not None and yielded >= limit:
            break
        fingerprint = fingerprints.get(str(row["imageId"]))
        if fingerprint is None:
            continue
        yielded += 1
        yield make_gqa_example(
            source_id=str(row["id"]),
            question=row["question"],
            answers=[row["answer"]],
            image_fingerprint=fingerprint,
        )


def load_chartqa(
    split: str, image_store: ImageStore, limit: int | None = None
) -> Iterator[Example]:  # pragma: no cover - exercised out of the fast tier
    """Stream ChartQA into Examples (never intervention-eligible, A8).

    Assumes rows carry ``question``, ``answer``, and an ``image``. They carry no
    identifier, so the example id is the row's position in the stream — stable for
    a fixed source revision, which is what the corpus manifest pins anyway, and
    checkable afterwards because the manifest records every id it assigned.
    """
    from datasets import load_dataset

    rows = load_dataset("lmms-lab/ChartQA", split=split, streaming=True)
    for index, row in enumerate(rows):
        if limit is not None and index >= limit:
            break
        fingerprint = image_store.put(_encode_png(row["image"]))
        yield make_chartqa_example(
            source_id=str(index),
            question=row["question"],
            answers=[row["answer"]],
            image_fingerprint=fingerprint,
        )
