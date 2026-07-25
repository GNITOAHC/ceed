"""The shared backbone objective: cross-entropy plus top-k logit distillation.

Every trained Group carries this one objective unchanged; the Groups differ only
in the auxiliary signals stacked on top (:mod:`ceed_student.auxiliary`). Fixing
it in a single function is what lets the plan claim no Group can accidentally
differ in its backbone.

Two terms make it up:

* **Cross-entropy** on the gold answer tokens — the ordinary supervised term.
* **Top-k logit distillation** against the teacher. The artifact store caches
  only the teacher's top-k logits per answer token (ADR-0001), so the teacher
  distribution is defined over that top-k support and the Student is matched to
  it there. Mass the Student places outside the cached support is untouched —
  the teacher never recorded a target for it.

Both are computed at a shared temperature (Hinton distillation), and the KD term
is scaled by ``temperature**2`` so its gradient magnitude does not collapse as
the temperature rises.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from jaxtyping import Float, Int
from torch import Tensor
from torch.nn import functional


@dataclass(frozen=True)
class BackboneLoss:
    """The backbone objective decomposed into its terms.

    The decomposition is kept rather than collapsed so the trainer can record
    the cross-entropy and distillation contributions separately — a run whose
    total is flat but whose KD is climbing is a different story from one where
    both fall.

    Attributes:
        total: The scalar optimised: ``cross_entropy + kd_weight * kd``.
        cross_entropy: The supervised cross-entropy on the gold answer tokens.
        kd: The top-k logit-distillation divergence, before weighting.
    """

    total: Tensor
    cross_entropy: Tensor
    kd: Tensor


def backbone_loss(
    student_logits: Float[Tensor, "tokens vocab"],
    gold_token_ids: Int[Tensor, " tokens"],
    teacher_topk_ids: Int[Tensor, "tokens k"],
    teacher_topk_values: Float[Tensor, "tokens k"],
    kd_weight: float,
    temperature: float = 1.0,
) -> BackboneLoss:
    """Compute the shared backbone loss for one batch of answer tokens.

    The cross-entropy is the mean over answer tokens of the Student's negative
    log-likelihood of the gold token. The distillation term is the mean over
    answer tokens of the KL divergence from the teacher's top-k distribution to
    the Student's, both softmaxed at ``temperature`` over the *teacher's* cached
    top-k support, and scaled by ``temperature**2``.

    Args:
        student_logits: The Student's next-token logits at each answer position.
        gold_token_ids: The gold token id at each answer position.
        teacher_topk_ids: The token ids of the teacher's top-k logits per answer
            position — the support the teacher was cached over.
        teacher_topk_values: The teacher's logit values at those ids.
        kd_weight: The weight of the distillation term relative to cross-entropy.
        temperature: The distillation softmax temperature.

    Returns:
        The total loss and its cross-entropy and distillation components.
    """
    cross_entropy = functional.cross_entropy(student_logits, gold_token_ids)

    # Restrict both distributions to the teacher's cached top-k support: the
    # Student's logits are gathered at exactly the ids the teacher recorded.
    student_on_support = torch.gather(student_logits, -1, teacher_topk_ids)
    teacher_p = functional.softmax(teacher_topk_values / temperature, dim=-1)
    student_log_q = functional.log_softmax(student_on_support / temperature, dim=-1)
    # KL(teacher || student) per token, averaged, with the standard T**2 scale.
    kd_per_token = functional.kl_div(student_log_q, teacher_p, reduction="none").sum(dim=-1)
    kd = kd_per_token.mean() * (temperature**2)

    total = cross_entropy + kd_weight * kd
    return BackboneLoss(total=total, cross_entropy=cross_entropy, kd=kd)
