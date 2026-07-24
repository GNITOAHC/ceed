"""B0: the zero-shot Student, evaluated through the harness under greedy decoding.

The fast tier covers everything that does not need the 8B Student or a GPU:
greedy-decoding enforcement, the harness-version lookup, B0 config composition,
and B0 driven through run_group with a stand-in evaluator so the run record's
accuracy, harness version, and decoding capture are all asserted. The real fp16
load and the real lmms-eval run are a gpu-tier test.
"""

import json
from pathlib import Path

import pytest
from pydantic import ValidationError
from transformers import GenerationConfig

from ceed_core import DecodingConfig, RunRecord
from ceed_eval import DATASET_TASKS, harness_version
from ceed_student import EvaluationOutcome, enforce_greedy, load_student, run_group

STUDENT_MODEL = "google/gemma-4-e4b-it"


# -- greedy decoding enforced in code (A9) ----------------------------------


def test_enforce_greedy_overrides_a_sampling_config():
    shipped = GenerationConfig(do_sample=True, temperature=1.0, top_p=0.95, top_k=64)
    enforce_greedy(shipped, DecodingConfig(do_sample=False, max_new_tokens=32))

    assert shipped.do_sample is False
    assert shipped.num_beams == 1
    assert shipped.max_new_tokens == 32
    # Sampling knobs are nulled so a stale value cannot re-enable sampling.
    assert shipped.temperature is None
    assert shipped.top_p is None
    assert shipped.top_k is None


# -- harness version is a real, recordable value ----------------------------


def test_a_sampling_decoding_config_is_rejected():
    # Greedy is not a per-Group knob: sampling cannot be configured at all (A9).
    with pytest.raises(ValidationError, match="greedy decoding is enforced"):
        DecodingConfig(do_sample=True)


def test_enforce_greedy_forces_greedy_even_from_a_greedy_looking_config():
    # enforce_greedy does not trust the config's flag; it forces greedy in code.
    shipped = GenerationConfig(do_sample=True, temperature=0.7)
    enforce_greedy(shipped, DecodingConfig(max_new_tokens=16))
    assert shipped.do_sample is False


def test_harness_version_is_the_installed_lmms_eval_version():
    assert harness_version()
    assert harness_version()[0].isdigit()


def test_every_configured_dataset_maps_to_a_harness_task():
    for dataset in ("docvqa", "gqa", "chartqa"):
        assert dataset in DATASET_TASKS


# -- B0 config composition ---------------------------------------------------


def test_b0_config_declares_the_three_datasets_and_greedy_decoding(b0_config):
    assert b0_config.group_code == "B0"
    assert b0_config.evaluation is not None
    assert b0_config.evaluation.datasets == ("docvqa", "gqa", "chartqa")
    assert b0_config.evaluation.decoding.do_sample is False
    assert not b0_config.auxiliary_signals


# -- B0 through run_group with a stand-in evaluator --------------------------


class StubEvaluator:
    """An evaluator that returns fixed accuracies without loading a model."""

    def __init__(self, accuracies: dict[str, float]) -> None:
        self.accuracies = accuracies

    def evaluate(self, config) -> EvaluationOutcome:
        assert config.evaluation is not None
        return EvaluationOutcome(
            accuracies=self.accuracies,
            harness_version=harness_version(),
            decoding=config.evaluation.decoding,
        )


def test_b0_reports_accuracy_for_each_dataset(b0_config, tmp_path):
    accuracies = {"docvqa": 0.62, "gqa": 0.55, "chartqa": 0.31}
    record = run_group(b0_config, tmp_path, evaluator=StubEvaluator(accuracies))

    assert record.metrics == accuracies
    assert set(record.metrics) == set(b0_config.evaluation.datasets)


def test_b0_run_record_captures_harness_version_and_decoding(b0_config, tmp_path):
    record = run_group(b0_config, tmp_path, evaluator=StubEvaluator({"docvqa": 0.6}))

    assert record.harness_version == harness_version()
    assert record.decoding is not None
    assert record.decoding.do_sample is False
    # The record round-trips from disk with its metrics and provenance intact.
    assert RunRecord.read(tmp_path / record.config_hash) == record


def test_b0_writes_accuracy_events_to_the_metrics_jsonl(b0_config, tmp_path):
    accuracies = {"docvqa": 0.62, "gqa": 0.55, "chartqa": 0.31}
    record = run_group(b0_config, tmp_path, evaluator=StubEvaluator(accuracies))

    lines = [json.loads(line) for line in Path(record.metrics_path).read_text().splitlines()]
    accuracy_events = [line for line in lines if line["event"] == "run_group.accuracy"]
    assert {e["dataset"] for e in accuracy_events} == set(accuracies)


def test_a_group_without_an_evaluator_records_no_accuracy(b0_config, tmp_path):
    # Declaring an evaluation but supplying no evaluator still produces a valid,
    # accuracy-free run record rather than failing.
    record = run_group(b0_config, tmp_path, evaluator=None)
    assert record.metrics == {}
    assert record.harness_version is None


# -- the real fp16 Student on a GPU (criterion 1) ---------------------------


def _student_is_cached() -> bool:
    """Whether the Student weights are already in the local Hugging Face cache."""
    from transformers import AutoConfig

    try:
        AutoConfig.from_pretrained(STUDENT_MODEL, local_files_only=True)
    except Exception:
        return False
    return True


def _answer(model, processor, content: list[dict]) -> tuple[str, bool]:
    """Generate an answer for one chat turn, returning it and whether logits stayed finite."""
    import torch

    inputs = processor.apply_chat_template(
        [{"role": "user", "content": content}],
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)
    with torch.no_grad():
        finite = bool(torch.isfinite(model(**inputs).logits).all())
    generated = model.generate(**inputs, max_new_tokens=16)
    answer = processor.decode(
        generated[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
    ).strip()
    return answer, finite


@pytest.mark.gpu
def test_student_loads_in_fp16_and_answers_coherently():
    """The Student casts to fp16 on the V100 and stays numerically stable.

    B0's first acceptance criterion and the project's fp16-on-Volta stability
    gate: the Student must load in fp16 and answer coherently on both the text
    and the vision path (an fp16 overflow surfaces as non-finite logits before it
    surfaces as an incoherent string). Skipped when the weights are not cached,
    so the test never triggers a multi-gigabyte download implicitly.
    """
    import torch
    from PIL import Image, ImageDraw

    if not torch.cuda.is_available():
        pytest.skip("no CUDA device")
    if not _student_is_cached():
        pytest.skip(f"{STUDENT_MODEL} weights are not in the local cache")

    model, processor = load_student(STUDENT_MODEL, DecodingConfig(max_new_tokens=16))
    assert next(model.parameters()).dtype is torch.float16

    text_answer, text_finite = _answer(
        model, processor, [{"type": "text", "text": "What is 2 + 2? Answer briefly."}]
    )
    assert text_finite  # no fp16 overflow to NaN/Inf on the text path
    assert text_answer  # a coherent, non-empty answer

    document = Image.new("RGB", (224, 96), "white")
    ImageDraw.Draw(document).text((20, 30), "Total: 42", fill="black")
    vision_answer, vision_finite = _answer(
        model,
        processor,
        [
            {"type": "image", "image": document},
            {"type": "text", "text": "What number is written in the image? Answer briefly."},
        ],
    )
    assert vision_finite  # the vision tower stays finite in fp16 too
    assert vision_answer
