"""Corpus assembly: DocVQA, GQA, and ChartQA normalised into one Example type."""

from ceed_data.corpus import CORPUS_FILENAME, read_examples, write_examples
from ceed_data.example import (
    CHARTQA,
    DATASETS,
    DOCVQA,
    GQA,
    INTERVENTION_INELIGIBLE_DATASETS,
    BoundingBox,
    Example,
    OcrWord,
)
from ceed_data.images import ImageStore, image_fingerprint
from ceed_data.loaders import (
    load_chartqa,
    load_docvqa,
    load_gqa,
    make_chartqa_example,
    make_docvqa_example,
    make_gqa_example,
    normalize_box,
)
from ceed_data.manifest import CorpusManifest
from ceed_data.splits import assign_split, split_corpus

__version__ = "0.1.0"

__all__ = [
    "CHARTQA",
    "CORPUS_FILENAME",
    "DATASETS",
    "DOCVQA",
    "GQA",
    "INTERVENTION_INELIGIBLE_DATASETS",
    "BoundingBox",
    "CorpusManifest",
    "Example",
    "ImageStore",
    "OcrWord",
    "assign_split",
    "image_fingerprint",
    "load_chartqa",
    "load_docvqa",
    "load_gqa",
    "make_chartqa_example",
    "make_docvqa_example",
    "make_gqa_example",
    "normalize_box",
    "read_examples",
    "split_corpus",
    "write_examples",
]
