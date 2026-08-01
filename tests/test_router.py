"""Reading effective combine weights off the Teacher's routers.

B5's whole target is this number, and getting it subtly wrong — the raw logits,
or the softmax weights before ``per_expert_scale`` — produces a Group that trains
perfectly well against the wrong quantity. Plan amendment A3 fixes the definition
precisely because those three are *not* monotone transforms of one another, so
the tests here pin which one is captured.

The fixture mirrors gemma-4's router contract rather than mocking it: the same
three-tuple return, the same top-k-then-scale ordering, and a per-expert scale
that genuinely reorders experts.
"""

import pytest
import torch
from ceed_teacher.router import capture_combine_weights, hybrid_layers, stack_layers

N_EXPERTS = 6
TOP_K = 2


class FakeRouter(torch.nn.Module):
    """Gemma-4's router contract: normalise the top-k, then scale per expert."""

    def __init__(self, per_expert_scale: torch.Tensor) -> None:
        super().__init__()
        self.per_expert_scale = torch.nn.Parameter(per_expert_scale, requires_grad=False)
        self.proj = torch.nn.Linear(4, N_EXPERTS, bias=False)

    def forward(self, hidden_states: torch.Tensor):
        probabilities = torch.softmax(self.proj(hidden_states), dim=-1)
        weights, index = torch.topk(probabilities, TOP_K, dim=-1)
        weights = weights / weights.sum(dim=-1, keepdim=True)
        return probabilities, weights * self.per_expert_scale[index], index


class FakeHybridLayer(torch.nn.Module):
    """A hybrid layer: a shared dense path, a router, and an expert bank."""

    def __init__(self, layer_idx: int, scale: torch.Tensor) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.mlp = torch.nn.Linear(4, 4)  # the shared dense path
        self.router = FakeRouter(scale)
        self.experts = torch.nn.Linear(4, 4)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        self.router(hidden_states)
        return self.mlp(hidden_states)


class FakeTeacher(torch.nn.Module):
    """A stack of hybrid layers wrapped the way the real Teacher is."""

    def __init__(self, n_layers: int = 3) -> None:
        super().__init__()
        torch.manual_seed(0)
        # A scale that reorders experts: the largest softmax weight is not
        # necessarily the largest effective combine weight.
        scale = torch.linspace(0.2, 2.0, N_EXPERTS).flip(0)
        self.language_model = torch.nn.ModuleList(
            [FakeHybridLayer(i, scale.clone()) for i in range(n_layers)]
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        for layer in self.language_model:
            hidden_states = layer(hidden_states)
        return hidden_states


def a_forward(model: FakeTeacher, positions: int = 5) -> torch.Tensor:
    return torch.randn(positions, 4)


# -- finding the layers ------------------------------------------------------


def test_hybrid_layers_are_found_by_structure_and_ordered_by_depth():
    # Found by carrying both a router and experts, not by module path — the
    # stack moves when a text model is wrapped in a multimodal one.
    model = FakeTeacher(n_layers=4)
    assert [layer.layer_idx for layer in hybrid_layers(model)] == [0, 1, 2, 3]


def test_a_model_with_no_hybrid_layers_is_refused():
    # The Student is dense: asking it for combine weights is a wiring error, and
    # returning an empty capture would train B5 against nothing.
    dense = torch.nn.Sequential(torch.nn.Linear(4, 4))
    with pytest.raises(ValueError, match="no hybrid layers"):
        hybrid_layers(dense)


# -- what is captured --------------------------------------------------------


def test_capture_returns_one_dense_vector_per_layer_per_position():
    model = FakeTeacher()
    hidden = a_forward(model)
    with capture_combine_weights(model, N_EXPERTS) as captured:
        model(hidden)

    assert sorted(captured) == [0, 1, 2]
    assert captured[0].shape == (hidden.shape[0], N_EXPERTS)


def test_only_the_routed_experts_carry_weight_and_the_rest_carry_zero():
    model = FakeTeacher()
    with capture_combine_weights(model, N_EXPERTS) as captured:
        model(a_forward(model))

    routed = (captured[0] > 0).sum(dim=-1)
    assert torch.equal(routed, torch.full_like(routed, TOP_K))


def test_the_captured_weight_is_the_effective_one_not_the_softmax_weight():
    """A3: the number that actually multiplies the expert's output.

    The per-expert scale reorders experts, so the effective weight's argmax is
    not the router probability's argmax. Capturing the probabilities would look
    entirely reasonable and answer a different question from the one Phase 0.1
    asks.
    """
    model = FakeTeacher(n_layers=1)
    hidden = a_forward(model, positions=1)
    with capture_combine_weights(model, N_EXPERTS) as captured:
        model(hidden)

    layer = model.language_model[0]
    probabilities, weights, index = layer.router(hidden)
    expected = torch.zeros(1, N_EXPERTS).scatter(1, index, weights)

    assert torch.allclose(captured[0], expected)
    # ...and it is genuinely a different vector from the raw probabilities.
    assert not torch.allclose(captured[0], probabilities)


def test_the_hooks_are_removed_so_the_teacher_is_left_as_it_was_found():
    # The same Teacher runs the ordinary logit pass; a leaked hook would keep
    # allocating a dense per-expert tensor on every forward for the rest of the
    # extraction.
    model = FakeTeacher()
    with capture_combine_weights(model, N_EXPERTS):
        model(a_forward(model))
    assert all(not layer.router._forward_hooks for layer in hybrid_layers(model))


# -- selecting the layers a Group supervises --------------------------------


def test_stacking_selects_the_requested_layers_in_the_requested_order():
    model = FakeTeacher()
    with capture_combine_weights(model, N_EXPERTS) as captured:
        model(a_forward(model))

    stacked = stack_layers(captured, (2, 0), position=3)
    assert stacked.shape == (2, N_EXPERTS)
    assert torch.equal(stacked[0], captured[2][3])
    assert torch.equal(stacked[1], captured[0][3])


def test_stacking_a_layer_that_was_never_captured_is_refused():
    model = FakeTeacher(n_layers=2)
    with capture_combine_weights(model, N_EXPERTS) as captured:
        model(a_forward(model))
    with pytest.raises(KeyError, match="7"):
        stack_layers(captured, (7,), position=0)


def test_the_capture_is_keyed_by_the_layers_own_index_not_its_depth():
    """A Teacher whose layers are not all hybrid must still key by teacher layer.

    Gemma-4 makes every layer hybrid, so depth and `layer_idx` coincide and the
    distinction is invisible — until it is not, at which point `stack_layers`
    would read a different layer than the Group's mapping asked for and nothing
    would raise.
    """
    model = FakeTeacher(n_layers=2)
    # Stand in for a stack where the hybrid layers sit at 7 and 21.
    model.language_model[0].layer_idx = 7
    model.language_model[1].layer_idx = 21

    with capture_combine_weights(model, N_EXPERTS) as captured:
        model(a_forward(model))

    assert sorted(captured) == [7, 21]
