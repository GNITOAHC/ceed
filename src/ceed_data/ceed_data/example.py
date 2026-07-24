"""The single Example type every Group trains and evaluates on.

DocVQA, GQA, and ChartQA are three different schemas; normalising them into one
Example type is what lets the plan claim every Group trains on the identical
corpus. The evidence annotations each dataset carries are preserved on the
Example — DocVQA's OCR words and boxes, GQA's answer-linked object box — so that
relevant regions and text-substitution counterfactuals are derivable downstream
without a second annotation pass.

Coordinates are normalised to the unit square so a box means the same thing
regardless of the source image's pixel size.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, model_validator

DOCVQA = "docvqa"
GQA = "gqa"
CHARTQA = "chartqa"

DATASETS = (DOCVQA, GQA, CHARTQA)

# ChartQA has no region annotations, so it never enters the intervention subset
# (plan amendment A8). Eligibility is carried by the data, not the operator.
INTERVENTION_INELIGIBLE_DATASETS = frozenset({CHARTQA})


class BoundingBox(BaseModel):
    """An axis-aligned box in normalised image coordinates.

    Attributes:
        x: Left edge, in ``[0, 1]``.
        y: Top edge, in ``[0, 1]``.
        width: Box width, in ``(0, 1]``.
        height: Box height, in ``(0, 1]``.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    x: float
    y: float
    width: float
    height: float

    @model_validator(mode="after")
    def _within_unit_square(self) -> BoundingBox:
        if self.width <= 0 or self.height <= 0:
            raise ValueError("bounding box must have positive width and height")
        if not (0.0 <= self.x <= 1.0 and 0.0 <= self.y <= 1.0):
            raise ValueError("bounding box origin must lie within the unit square")
        if self.x + self.width > 1.0 + 1e-6 or self.y + self.height > 1.0 + 1e-6:
            raise ValueError("bounding box must not extend past the unit square")
        return self


class OcrWord(BaseModel):
    """One recognised word and where it sits on the page.

    Attributes:
        text: The word's text.
        box: Its location in normalised coordinates.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    text: str
    box: BoundingBox


class Example(BaseModel):
    """One normalised vision-language example with its evidence annotations.

    Attributes:
        example_id: A stable identifier of the form ``"{dataset}:{source_id}"``,
            unchanged across runs so a Group trained in week two is comparable to
            one trained in week ten.
        dataset: The source dataset (``docvqa``, ``gqa``, or ``chartqa``).
        image_fingerprint: The content hash of the image bytes, so the teacher
            and the Student provably see byte-identical inputs.
        question: The question text.
        answers: The gold answers.
        ocr_words: DocVQA's recognised words and boxes; empty for other datasets.
        answer_region: GQA's answer-linked object box; ``None`` for others.
        intervention_eligible: Whether this example may be intervened on. Always
            ``False`` for ChartQA (A8).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    example_id: str
    dataset: str
    image_fingerprint: str
    question: str
    answers: tuple[str, ...]
    ocr_words: tuple[OcrWord, ...] = ()
    answer_region: BoundingBox | None = None
    intervention_eligible: bool = False

    @model_validator(mode="after")
    def _consistent(self) -> Example:
        if self.dataset not in DATASETS:
            raise ValueError(f"dataset must be one of {DATASETS}, got {self.dataset!r}")
        if not self.example_id.startswith(f"{self.dataset}:"):
            raise ValueError("example_id must be prefixed with its dataset")
        if self.dataset in INTERVENTION_INELIGIBLE_DATASETS and self.intervention_eligible:
            raise ValueError(f"{self.dataset} examples are never intervention-eligible (A8)")
        return self
