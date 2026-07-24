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

from typing import Any

from ceed_core import DecodingConfig


def enforce_greedy(generation_config: Any, decoding: DecodingConfig) -> Any:
    """Overwrite a generation config so decoding is greedy, whatever it shipped.

    The checkpoint's own generation config may enable sampling (Gemma ships
    ``do_sample: true`` with a temperature and a top-p); this clears every
    sampling knob and pins a single greedy beam, then applies the run's
    generation-length cap. It mutates and returns the object it is given.

    Args:
        generation_config: A Hugging Face ``GenerationConfig`` (or anything with
            the same attributes).
        decoding: The decoding settings to enforce.

    Returns:
        The same generation config, now greedy.
    """
    generation_config.do_sample = decoding.do_sample
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

    torch_dtype = getattr(torch, dtype)
    model = AutoModelForImageTextToText.from_pretrained(model_id, torch_dtype=torch_dtype)
    model = model.to(device)  # type: ignore[arg-type]
    enforce_greedy(model.generation_config, decoding)
    processor = AutoProcessor.from_pretrained(model_id)
    return model, processor
