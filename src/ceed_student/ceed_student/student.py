"""Loading the Student and pinning its decoding to greedy.

Two things here are load-bearing for B0. The first is that the Student is cast to
fp16 — Volta has no bf16 and the checkpoint ships bf16 — which is the project's
first serious numerical risk and the reason B0 is built before any teacher work.
The second is that decoding is forced to greedy in code, overriding whatever
sampling the checkpoint's generation config ships, so that decoding variance
cannot swamp the small effects the plan predicts (plan amendment A9).

:func:`enforce_greedy` is pure and CPU-testable. :func:`load_student` needs the
real weights and a GPU and so runs out of the fast tier.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from ceed_core import DecodingConfig


def enforce_greedy(generation_config: Any, decoding: DecodingConfig) -> Any:
    """Overwrite a generation config so decoding is greedy, whatever it shipped.

    The checkpoint's own generation config may enable sampling (Gemma ships
    ``do_sample: true`` with a temperature and a top-p); this forces greedy
    decoding unconditionally — a single beam with sampling off and every
    sampling knob nulled — regardless of what the checkpoint or the config asks
    for, then applies the run's generation-length cap. It mutates and returns the
    object it is given.

    Args:
        generation_config: A Hugging Face ``GenerationConfig`` (or anything with
            the same attributes).
        decoding: The decoding settings; only ``max_new_tokens`` is read, since
            greedy is not negotiable (:class:`DecodingConfig` forbids sampling).

    Returns:
        The same generation config, now greedy.
    """
    generation_config.do_sample = False
    generation_config.num_beams = 1
    generation_config.max_new_tokens = decoding.max_new_tokens
    # Null out sampling knobs so a stale value can never re-enable sampling.
    generation_config.temperature = None
    generation_config.top_p = None
    generation_config.top_k = None
    return generation_config


def load_student(
    model_id: str, decoding: DecodingConfig, dtype: str = "float16", device: str = "cuda"
) -> tuple[Any, Any]:  # pragma: no cover - needs the real weights and a GPU
    """Load the Student in ``dtype`` on ``device`` with greedy decoding enforced.

    Args:
        model_id: The Student checkpoint identifier.
        decoding: The decoding settings to enforce on the loaded model.
        dtype: The compute dtype; ``float16`` on Volta, which has no bf16.
        device: The device to place the model on.

    Returns:
        The loaded model and its processor.
    """
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor

    compute_dtype = getattr(torch, dtype)
    model = AutoModelForImageTextToText.from_pretrained(model_id, dtype=compute_dtype)
    model = model.to(device)  # type: ignore[arg-type]
    enforce_greedy(model.generation_config, decoding)
    processor = AutoProcessor.from_pretrained(model_id)
    return model, processor


def apply_adapter(model: Any, checkpoint_dir: Path) -> Any:  # pragma: no cover - needs the weights
    """Apply a Group's trained LoRA adapter to a loaded Student, in place.

    The trainer saves the adapter from inside a :class:`CeedStudent`, so its
    recorded module paths carry that wrapper's prefix. The model is therefore
    re-wrapped here to make the trees line up. PEFT injects its layers into the
    existing modules, so the *inner* model is returned — adapted, and still
    carrying the ``generate`` that inference decodes through.

    A checkpoint with no adapter directory (a full fine-tune, whose weights live
    in the accelerate state) leaves the model unchanged rather than failing.

    Args:
        model: The loaded base Student.
        checkpoint_dir: The Group's checkpoint directory.

    Returns:
        The Student with the adapter applied.
    """
    adapter = Path(checkpoint_dir) / "adapter"
    if not adapter.is_dir():
        return model
    from peft import PeftModel

    wrapper = CeedStudent(model)
    PeftModel.from_pretrained(wrapper, str(adapter))
    return wrapper.model


def load_trained_student(
    model_id: str,
    checkpoint_dir: Path,
    decoding: DecodingConfig | None = None,
    dtype: str = "float16",
    device: str = "cuda",
) -> tuple[Any, Any]:  # pragma: no cover - needs the real weights and a GPU
    """Load a Group's trained Student: the base checkpoint plus its adapter.

    This is what to call to use a finished B1 or B2 run — the run record's
    ``checkpoint_dir`` is the argument. B0 trained nothing, so it is loaded with
    :func:`load_student` instead.

    Args:
        model_id: The base Student identifier the Group trained from.
        checkpoint_dir: The Group's checkpoint directory, as recorded on the run
            record.
        decoding: The decoding settings to enforce; greedy defaults apply if
            omitted.
        dtype: The compute dtype.
        device: The device to place the model on.

    Returns:
        The trained model and its processor, ready to generate.
    """
    model, processor = load_student(
        model_id, decoding or DecodingConfig(), dtype=dtype, device=device
    )
    model = apply_adapter(model, checkpoint_dir)
    model.eval()
    return model, processor


class CeedStudent(torch.nn.Module):
    """The real Student wired to the training loop's :class:`TrainableStudent` seam.

    The loop needs one thing from a Student — the logits at the answer-token
    positions under a given forward view — and this supplies it for the real
    vision-language model. Everything else (parameters, ``state_dict``, device
    placement) it inherits from being an ordinary module, which is what lets
    ``accelerate`` and PEFT wrap it unchanged.

    Only the original forward view is implemented. The intervened views belong to
    the E-Groups and arrive with the intervention pipeline; asking for one here
    fails loudly rather than silently scoring the wrong image.
    """

    def __init__(self, model: Any) -> None:
        """Wrap an already-loaded vision-language model."""
        super().__init__()
        self.model = model

    def answer_logits(self, batch: Any, view: Any) -> torch.Tensor:
        """Return the Student's logits at the batch's answer-token positions.

        Args:
            batch: The training batch, whose ``student_inputs`` are run through
                the model and whose ``answer_token_positions`` select the
                positions the answer tokens are predicted at.
            view: The forward view; only the original view is supported.

        Returns:
            The logits at the answer positions, ``[answer_tokens, vocab]``.

        Raises:
            NotImplementedError: If an intervened view is requested.
        """
        from ceed_student.auxiliary import ForwardView

        if view is not ForwardView.ORIGINAL:
            raise NotImplementedError(
                f"the Student has no {view} forward; intervened views arrive with "
                "the intervention pipeline"
            )
        outputs = self.model(**batch.student_inputs)
        # The batch holds one example, so the leading batch axis is dropped
        # before the answer positions are selected.
        return outputs.logits[0][batch.answer_token_positions]
