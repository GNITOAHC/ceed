"""The shared backbone: cross-entropy plus top-k logit distillation.

The backbone is the one objective every trained Group carries unchanged, so its
numbers are asserted directly rather than through a training run. These tests pin
the decomposition (total = cross-entropy + kd_weight x kd), the two ends of the
KD term (zero when the Student matches the teacher's top-k, positive otherwise),
and the fact that the teacher supervises only the cached top-k support — a
Student may place any mass it likes outside the k the teacher recorded.
"""

import torch

from ceed_student import BackboneLoss, backbone_loss


def _teacher_topk(logits: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor]:
    top = torch.topk(logits, k, dim=-1)
    return top.indices, top.values


# -- decomposition -----------------------------------------------------------


def test_total_is_cross_entropy_plus_weighted_kd():
    torch.manual_seed(0)
    student = torch.randn(3, 8)
    teacher = torch.randn(3, 8)
    gold = torch.tensor([1, 4, 7])
    ids, values = _teacher_topk(teacher, 4)

    loss = backbone_loss(student, gold, ids, values, kd_weight=0.5)
    assert isinstance(loss, BackboneLoss)
    expected = loss.cross_entropy + 0.5 * loss.kd
    assert torch.isclose(loss.total, expected, atol=1e-6)


def test_zero_kd_weight_leaves_only_cross_entropy():
    torch.manual_seed(1)
    student = torch.randn(2, 6)
    teacher = torch.randn(2, 6)
    gold = torch.tensor([0, 5])
    ids, values = _teacher_topk(teacher, 3)

    loss = backbone_loss(student, gold, ids, values, kd_weight=0.0)
    assert torch.isclose(loss.total, loss.cross_entropy, atol=1e-7)
    ce = torch.nn.functional.cross_entropy(student, gold)
    assert torch.isclose(loss.cross_entropy, ce, atol=1e-6)


# -- the two ends of the KD term --------------------------------------------


def test_kd_is_zero_when_the_student_matches_the_teacher_on_the_top_k():
    # The Student's logits over the teacher's top-k support are an affine shift
    # of the teacher's, so their softmaxes over that support coincide and the KD
    # term vanishes. Mass outside the support is irrelevant.
    torch.manual_seed(2)
    teacher = torch.randn(4, 10)
    ids, values = _teacher_topk(teacher, 5)

    student = torch.full((4, 10), -30.0)
    for t in range(4):
        student[t, ids[t]] = values[t] + 3.0  # affine shift within the support

    loss = backbone_loss(student, torch.zeros(4, dtype=torch.long), ids, values, kd_weight=1.0)
    assert loss.kd.abs() < 1e-5


def test_kd_is_positive_when_the_student_disagrees_on_the_top_k():
    torch.manual_seed(3)
    teacher = torch.randn(4, 10)
    ids, values = _teacher_topk(teacher, 5)
    student = torch.randn(4, 10)  # unrelated

    loss = backbone_loss(student, torch.zeros(4, dtype=torch.long), ids, values, kd_weight=1.0)
    assert loss.kd > 0.0


def test_kd_ignores_student_mass_outside_the_cached_top_k():
    # Two Students that agree on the teacher's top-k support but differ wildly
    # off it produce the same KD — the teacher only cached the top-k (ADR-0001).
    torch.manual_seed(4)
    teacher = torch.randn(3, 12)
    ids, values = _teacher_topk(teacher, 4)
    gold = torch.zeros(3, dtype=torch.long)

    base = torch.randn(3, 12)
    other = base.clone()
    # Perturb only the off-support logits.
    mask = torch.ones(3, 12, dtype=torch.bool)
    for t in range(3):
        mask[t, ids[t]] = False
    other[mask] += 17.0

    a = backbone_loss(base, gold, ids, values, kd_weight=1.0)
    b = backbone_loss(other, gold, ids, values, kd_weight=1.0)
    assert torch.isclose(a.kd, b.kd, atol=1e-6)


# -- cross-entropy end behaviour --------------------------------------------


def test_cross_entropy_vanishes_for_a_confident_correct_student():
    student = torch.full((2, 5), -30.0)
    gold = torch.tensor([2, 4])
    student[0, 2] = 30.0
    student[1, 4] = 30.0
    ids, values = _teacher_topk(torch.randn(2, 5), 2)

    loss = backbone_loss(student, gold, ids, values, kd_weight=1.0)
    assert loss.cross_entropy < 1e-6


def test_the_loss_is_a_scalar_averaged_over_answer_tokens():
    torch.manual_seed(5)
    student = torch.randn(7, 9)
    teacher = torch.randn(7, 9)
    ids, values = _teacher_topk(teacher, 3)
    loss = backbone_loss(student, torch.zeros(7, dtype=torch.long), ids, values, kd_weight=1.0)
    assert loss.total.ndim == 0
    assert loss.total.requires_grad is False


def test_temperature_scaling_keeps_kd_finite_and_nonnegative():
    torch.manual_seed(6)
    student = torch.randn(3, 8)
    teacher = torch.randn(3, 8)
    ids, values = _teacher_topk(teacher, 4)
    gold = torch.zeros(3, dtype=torch.long)
    for temperature in (0.5, 1.0, 4.0):
        loss = backbone_loss(student, gold, ids, values, kd_weight=1.0, temperature=temperature)
        assert torch.isfinite(loss.kd)
        assert loss.kd >= -1e-6


def test_the_backbone_carries_gradients_when_the_student_requires_grad():
    # Under a real training step the Student logits require grad; the returned
    # total must be differentiable back to them.
    torch.manual_seed(7)
    student = torch.randn(3, 8, requires_grad=True)
    teacher = torch.randn(3, 8)
    ids, values = _teacher_topk(teacher, 4)
    loss = backbone_loss(student, torch.zeros(3, dtype=torch.long), ids, values, kd_weight=1.0)
    loss.total.backward()
    assert student.grad is not None
    assert torch.isfinite(student.grad).all()
