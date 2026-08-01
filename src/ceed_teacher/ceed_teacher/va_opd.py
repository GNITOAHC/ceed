r"""The teacher-side half of the VA-OPD reproduction: degradation and advantage.

VA-OPD (*Visual-Advantage On-Policy Distillation for Vision-Language Models*,
arXiv:2605.21924) is B4, the plan's critical external baseline. Its premise is
that ordinary distillation improves a student's answers without strengthening
its *reliance on the image*, and its remedy is to find the tokens where the
teacher's own prediction depends on fine visual detail and concentrate the
distillation loss there.

Two quantities come from the teacher and are therefore cached here:

* the **degraded image** — the paper downsamples to 10% of the spatial
  resolution and upsamples back, destroying fine detail (small text, digits,
  symbols) while preserving global layout and colour. It does this rather than
  removing the image because the visual token count must stay identical for the
  two passes to align token-for-token;
* the **visual advantage** of an answer token — how much likelier the teacher
  makes that token when it can see the detail, rectified at zero.

The student-side half — grouping tokens by advantage and reweighting the loss —
is :mod:`ceed_student.signals`, because it is a loss and the teacher is never
loaded during training (ADR-0001). What is faithful to the paper and what
deviates from it is recorded in
docs/adr/0007-va-opd-reproduced-off-policy-on-gold-answers.md.
"""

from __future__ import annotations

from typing import Any

import torch
from jaxtyping import Float
from torch import Tensor

VISUAL_ADVANTAGE = "visual_advantage"

# The paper's degradation: down to a tenth of each spatial dimension and back.
DEGRADATION_FRACTION = 0.1


def degrade_image(image: Any, fraction: float = DEGRADATION_FRACTION) -> Any:
    """Return the image with its fine detail destroyed and its layout kept.

    Reproduces VA-OPD's degradation exactly: bilinear downsampling to
    ``fraction`` of each spatial dimension, then nearest-neighbour upsampling
    back to the original size. Bilinear on the way down averages detail away;
    nearest on the way up refuses to invent it back, leaving the blocky,
    layout-preserving image the paper's advantage is measured against.

    The output keeps the input's exact dimensions, which is load-bearing rather
    than tidy: the two teacher passes must emit the same number of image tokens,
    or the per-token advantage would compare different positions.

    Args:
        image: The PIL image to degrade.
        fraction: The share of each spatial dimension to downsample to.

    Returns:
        A new PIL image of the same size, carrying only coarse structure.

    Raises:
        ValueError: If ``fraction`` is not in ``(0, 1]``.
    """
    from PIL import Image

    if not 0.0 < fraction <= 1.0:
        raise ValueError(f"degradation fraction must be in (0, 1], got {fraction}")
    width, height = image.size
    small = (max(1, round(width * fraction)), max(1, round(height * fraction)))
    return image.resize(small, Image.Resampling.BILINEAR).resize(
        (width, height), Image.Resampling.NEAREST
    )


def visual_advantage(
    original_logprobs: Float[Tensor, " tokens"], degraded_logprobs: Float[Tensor, " tokens"]
) -> Float[Tensor, " tokens"]:
    r"""Return each answer token's visual advantage, rectified at zero.

    The paper's definition, verbatim:

    .. math:: a_t = \max\bigl(\log p_T(y_t \mid v, q, y_{<t})
              - \log p_T(y_t \mid \tilde v, q, y_{<t}),\ 0\bigr)

    where :math:`v` is the original image and :math:`\tilde v` the degraded one.
    The rectification is not a numerical guard: it discards the tokens the
    teacher predicts *better* without the detail, which carry no visual
    supervision and would otherwise pull the loss the wrong way.

    Args:
        original_logprobs: The teacher's log-probability of each gold answer
            token given the original image.
        degraded_logprobs: The same, given the degraded image.

    Returns:
        The per-token visual advantage, non-negative.
    """
    return torch.clamp(original_logprobs - degraded_logprobs, min=0.0)


def gold_logprobs(
    logits: Float[Tensor, "positions vocab"], gold_token_ids: Float[Tensor, " tokens"]
) -> Float[Tensor, " tokens"]:
    """Return the teacher's log-probability of each gold token at its position.

    Args:
        logits: The teacher's next-token logits at the answer-token measurement
            positions, in answer-token order.
        gold_token_ids: The gold token each of those positions predicts.

    Returns:
        One log-probability per answer token.
    """
    log_probs = torch.log_softmax(logits.float(), dim=-1)
    return log_probs.gather(-1, gold_token_ids.long().unsqueeze(-1)).squeeze(-1)
