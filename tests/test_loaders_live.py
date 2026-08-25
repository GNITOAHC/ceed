"""The streaming loaders, against the sources they actually read.

These are marked ``slow`` and excluded from the fast tier because they download
from the Hugging Face Hub. They exist because the alternative — testing the pure
builders on synthetic rows and assuming the loaders feed them correctly — let two
of the three loaders sit broken indefinitely:

* ``load_gqa`` read ``train_all_instructions`` expecting an ``image`` and an
  ``answer_box``. That config is text only. Neither field has ever been there.
* ``load_chartqa`` read split ``train`` expecting an ``id``. ``lmms-lab/ChartQA``
  publishes only ``test``, and its rows have no identifier.

Both failures were invisible: ``build_corpus.py`` caught the exception per
dataset and continued, so asking for three datasets produced a DocVQA-only corpus
that looked exactly like a corpus. Run these before trusting a corpus build:

    uv run pytest -m slow tests/test_loaders_live.py
"""

import pytest

from ceed_data import CHARTQA, DOCVQA, GQA, ImageStore, load_chartqa, load_docvqa, load_gqa

pytestmark = pytest.mark.slow

# What scripts/build_corpus.py draws each source from. Kept here as literals
# rather than imported, so that a change to the script has to be made twice --
# once in the code and once against a source that is actually read.
SOURCE_SPLITS = {DOCVQA: "validation", GQA: "train", CHARTQA: "test"}

LOADERS = {DOCVQA: load_docvqa, GQA: load_gqa, CHARTQA: load_chartqa}


@pytest.mark.parametrize("dataset", sorted(LOADERS))
def test_each_loader_yields_usable_examples_from_its_real_source(dataset, tmp_path):
    """Every supported source must produce examples the rest of CEED can use."""
    image_store = ImageStore(tmp_path / "images")
    examples = list(LOADERS[dataset](SOURCE_SPLITS[dataset], image_store, limit=4))

    assert examples, f"{dataset} yielded nothing from split {SOURCE_SPLITS[dataset]!r}"
    for example in examples:
        assert example.dataset == dataset
        assert example.example_id.startswith(f"{dataset}:")
        assert example.question.strip()
        assert example.answers
        assert all(answer.strip() for answer in example.answers)
        # The image must have actually landed in the store: an example whose
        # fingerprint names nothing fails much later, inside training.
        assert image_store.contains(example.image_fingerprint)


@pytest.mark.parametrize("dataset", sorted(LOADERS))
def test_each_loader_honours_its_limit(dataset, tmp_path):
    image_store = ImageStore(tmp_path / "images")
    assert len(list(LOADERS[dataset](SOURCE_SPLITS[dataset], image_store, limit=2))) <= 2


def test_gqa_examples_carry_no_answer_region(tmp_path):
    """The published source has no answer box, and the loader must not invent one.

    ``lmms-lab/GQA`` serves the questions without the scene graph, so the
    answer-linked box is simply absent. Phase 2's region interventions have to
    supply it from the scene-graph release rather than assume it arrived here.
    """
    image_store = ImageStore(tmp_path / "images")
    examples = list(load_gqa("train", image_store, limit=4))
    assert examples
    assert all(example.answer_region is None for example in examples)


def test_gqa_joins_every_question_to_a_real_image(tmp_path):
    """The join is the whole loader: a question whose image is missing is dropped."""
    image_store = ImageStore(tmp_path / "images")
    examples = list(load_gqa("train", image_store, limit=16))
    assert examples
    fingerprints = {example.image_fingerprint for example in examples}
    # Balanced GQA asks several questions per image, so the join should reuse
    # images rather than store one per question.
    assert len(fingerprints) < len(examples)
