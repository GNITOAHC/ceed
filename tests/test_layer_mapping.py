"""The Teacher-to-Student layer mapping, as an object rather than a convention.

The mapping has to be first-class because two later results turn on it: C2 is
defined as a *deliberately mismatched* mapping, and Phase 2 reports variance
across alternatives. Neither is expressible if the correspondence is a rule
applied at the point of use. So the tests here assert the properties a reviewer
would check — that a mapping resolves, that a mismatched mapping is provably
mismatched against the proportional one it controls for, and that the mapping a
run used reaches the run record.
"""

import pytest
from pydantic import ValidationError

from ceed_core import LayerMapping, mismatched_mapping, proportional_mapping

# The real pair: the Teacher has 30 hybrid layers, the Student 42.
TEACHER_LAYERS = 30
STUDENT_LAYERS = 42


# -- resolution --------------------------------------------------------------


def test_a_mapping_resolves_a_teacher_layer_to_its_student_layer():
    mapping = LayerMapping(kind="proportional", pairs=((9, 12), (19, 26)))
    assert mapping.student_layer_for(9) == 12
    assert mapping.student_layer_for(19) == 26


def test_resolving_an_unmapped_teacher_layer_names_what_is_mapped():
    mapping = LayerMapping(kind="proportional", pairs=((9, 12),))
    with pytest.raises(KeyError, match="19"):
        mapping.student_layer_for(19)


def test_a_mapping_exposes_its_layers_in_declaration_order():
    mapping = LayerMapping(kind="proportional", pairs=((29, 39), (9, 12)))
    assert mapping.teacher_layers == (29, 9)
    assert mapping.student_layers == (39, 12)


# -- validation --------------------------------------------------------------


def test_a_teacher_layer_mapped_twice_is_refused():
    # Two student layers for one teacher layer makes student_layer_for ambiguous,
    # and would silently supervise only one of them.
    with pytest.raises(ValidationError, match="more than once"):
        LayerMapping(kind="proportional", pairs=((9, 12), (9, 26)))


def test_an_empty_mapping_is_refused():
    with pytest.raises(ValidationError, match="at least one"):
        LayerMapping(kind="proportional", pairs=())


def test_a_negative_layer_index_is_refused():
    with pytest.raises(ValidationError, match="non-negative"):
        LayerMapping(kind="proportional", pairs=((9, -1),))


def test_an_unknown_kind_is_refused():
    # The three kinds are a closed set: C2's control is defined by being one of
    # them, so a mapping labelled anything else has no defined meaning.
    with pytest.raises(ValidationError, match="handwritten"):
        LayerMapping(kind="handwritten", pairs=((9, 12),))


# -- the proportional placeholder -------------------------------------------


def test_a_proportional_mapping_places_layers_at_the_same_relative_depth():
    mapping = proportional_mapping((9, 19, 29), TEACHER_LAYERS, STUDENT_LAYERS)
    assert mapping.kind == "proportional"
    # floor(layer * 42 / 30): the same fraction of the way through each stack.
    assert mapping.pairs == ((9, 12), (19, 26), (29, 40))


def test_a_proportional_mapping_is_monotone_in_teacher_depth():
    mapping = proportional_mapping(tuple(range(TEACHER_LAYERS)), TEACHER_LAYERS, STUDENT_LAYERS)
    students = list(mapping.student_layers)
    assert students == sorted(students)


def test_a_proportional_mapping_never_leaves_the_student_stack():
    mapping = proportional_mapping(tuple(range(TEACHER_LAYERS)), TEACHER_LAYERS, STUDENT_LAYERS)
    assert all(0 <= layer < STUDENT_LAYERS for layer in mapping.student_layers)


def test_a_teacher_layer_outside_its_own_stack_is_refused():
    with pytest.raises(ValueError, match="teacher layer"):
        proportional_mapping((30,), TEACHER_LAYERS, STUDENT_LAYERS)


# -- C2's deliberately mismatched control -----------------------------------


def test_a_mismatched_mapping_agrees_with_its_source_nowhere():
    """C2's whole content: the same layers, paired wrongly on purpose.

    If any pair survived, C2 would be partly correct and its null result would
    be uninterpretable — the control has to be provably a control.
    """
    source = proportional_mapping((9, 19, 29), TEACHER_LAYERS, STUDENT_LAYERS)
    control = mismatched_mapping(source)

    assert control.kind == "mismatched"
    assert control.teacher_layers == source.teacher_layers
    assert all(
        control.student_layer_for(layer) != source.student_layer_for(layer)
        for layer in source.teacher_layers
    )


def test_a_mismatched_mapping_reuses_exactly_the_source_student_layers():
    # It is a permutation, not a different set of layers: C2 must differ from
    # its source in the correspondence alone, not in which layers are supervised.
    source = proportional_mapping((9, 19, 29), TEACHER_LAYERS, STUDENT_LAYERS)
    control = mismatched_mapping(source)
    assert sorted(control.student_layers) == sorted(source.student_layers)


def test_a_single_pair_cannot_be_mismatched():
    # With one pair there is no derangement, so there is no control to build —
    # better to refuse than to hand back the source relabelled 'mismatched'.
    source = LayerMapping(kind="proportional", pairs=((9, 12),))
    with pytest.raises(ValueError, match="at least two"):
        mismatched_mapping(source)


# -- what a reviewer checks --------------------------------------------------


def test_the_checked_in_base_mapping_is_recorded_explicitly(configs_dir):
    """base.yaml states the pairs; it does not leave them to a convention.

    Story 59 — a reviewer wants to see which layer mapping a reported B3 or E1
    used — is satisfied by the pairs being data, whatever rule produced them.
    """
    from ceed_core.config import load_overlay

    mapping = LayerMapping.model_validate(load_overlay(configs_dir / "base.yaml")["layer_mapping"])
    assert mapping.kind == "proportional"
    assert len(mapping.pairs) == 3
