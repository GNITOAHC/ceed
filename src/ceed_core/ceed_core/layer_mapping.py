"""The Teacher-to-Student layer correspondence, as a first-class object.

Placing hidden-state supervision or a probe at "the equivalent layer" is not a
convention that can live at the point of use, for two reasons the plan states
directly: C2 is defined as a *deliberately mismatched* mapping, and Phase 2
reports variance across alternative mappings. Both require the correspondence a
run used to be an object that is chosen, recorded, and compared — so it is one,
it reaches the run record, and a reviewer can read it off a finished run.

Three kinds exist. ``proportional`` places a teacher layer at the same relative
depth in the student stack and is the placeholder that unblocks B3; ``probe`` is
the learned replacement the layer-mapping study produces; ``mismatched`` is C2's
control, built from a source mapping by permuting its student layers so that no
pair survives.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, field_validator, model_validator

MAPPING_KINDS = ("proportional", "probe", "mismatched")


class LayerMapping(BaseModel):
    """The correspondence between Teacher and Student layers.

    Attributes:
        kind: How the mapping was produced — ``proportional`` (the placeholder
            that unblocks B3), ``probe`` (the learned replacement), or
            ``mismatched`` (C2's control).
        pairs: The ``(teacher_layer, student_layer)`` correspondences, in
            declaration order.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: str
    pairs: tuple[tuple[int, int], ...]

    @field_validator("kind")
    @classmethod
    def _known_kind(cls, value: str) -> str:
        if value not in MAPPING_KINDS:
            raise ValueError(
                f"layer mapping kind must be one of {sorted(MAPPING_KINDS)}, got {value!r}"
            )
        return value

    @model_validator(mode="after")
    def _well_formed(self) -> LayerMapping:
        if not self.pairs:
            raise ValueError("a layer mapping needs at least one (teacher, student) pair")
        if any(teacher < 0 or student < 0 for teacher, student in self.pairs):
            raise ValueError(f"layer indices must be non-negative, got {self.pairs}")
        teachers = [teacher for teacher, _ in self.pairs]
        if len(set(teachers)) != len(teachers):
            raise ValueError(
                f"a teacher layer is mapped more than once in {self.pairs}; "
                "the correspondence would be ambiguous"
            )
        return self

    @property
    def teacher_layers(self) -> tuple[int, ...]:
        """The teacher layers this mapping supervises, in declaration order."""
        return tuple(teacher for teacher, _ in self.pairs)

    @property
    def student_layers(self) -> tuple[int, ...]:
        """The student layers supervision lands on, in declaration order."""
        return tuple(student for _, student in self.pairs)

    def student_layer_for(self, teacher_layer: int) -> int:
        """Return the student layer a teacher layer's supervision is placed at.

        Args:
            teacher_layer: The teacher layer to resolve.

        Returns:
            The corresponding student layer.

        Raises:
            KeyError: If this mapping says nothing about ``teacher_layer``,
                naming what it does map — a signal asking for an unmapped layer
                is a configuration error, not a reason to guess.
        """
        for teacher, student in self.pairs:
            if teacher == teacher_layer:
                return student
        raise KeyError(
            f"teacher layer {teacher_layer} is not in this {self.kind} mapping, "
            f"which maps {list(self.teacher_layers)}"
        )


def proportional_mapping(
    teacher_layers: tuple[int, ...], n_teacher_layers: int, n_student_layers: int
) -> LayerMapping:
    """Map each teacher layer to the student layer at the same relative depth.

    ``floor(layer * n_student / n_teacher)``: layer 9 of 30 lands at layer 12 of
    42, a third of the way through both stacks. This is a placeholder and is
    labelled as one — it assumes the two stacks compute comparable things at
    comparable depths, which is exactly what the probe-based layer-mapping study
    is built to check.

    Args:
        teacher_layers: The teacher layers to map.
        n_teacher_layers: How many layers the Teacher has.
        n_student_layers: How many layers the Student has.

    Returns:
        The proportional mapping over ``teacher_layers``.

    Raises:
        ValueError: If a teacher layer lies outside the Teacher's own stack.
    """
    outside = [layer for layer in teacher_layers if not 0 <= layer < n_teacher_layers]
    if outside:
        raise ValueError(
            f"teacher layer(s) {outside} lie outside a {n_teacher_layers}-layer Teacher"
        )
    pairs = tuple(
        (layer, min(layer * n_student_layers // n_teacher_layers, n_student_layers - 1))
        for layer in teacher_layers
    )
    return LayerMapping(kind="proportional", pairs=pairs)


def mismatched_mapping(source: LayerMapping) -> LayerMapping:
    """Return C2's control: the same layers, deliberately paired wrongly.

    The student layers are rotated by one against the teacher layers, so the
    control supervises exactly the same student layers as ``source`` and differs
    from it only in the correspondence — which is the single thing C2 is a
    control for. Every pair changes, so no part of the control is accidentally
    correct.

    Args:
        source: The mapping to build a control against.

    Returns:
        A ``mismatched`` mapping over the same layers.

    Raises:
        ValueError: If ``source`` has fewer than two pairs (no derangement
            exists), or if rotating it would leave a pair unchanged because two
            of its student layers coincide.
    """
    if len(source.pairs) < 2:
        raise ValueError(
            "a mismatched mapping needs at least two pairs; with one pair there "
            "is no wrong pairing to make"
        )
    rotated = source.student_layers[1:] + source.student_layers[:1]
    pairs = tuple(zip(source.teacher_layers, rotated, strict=True))
    unchanged = [
        teacher for (teacher, student) in pairs if source.student_layer_for(teacher) == student
    ]
    if unchanged:
        raise ValueError(
            f"rotating {source.pairs} leaves teacher layer(s) {unchanged} mapped to the "
            "same student layer, so the result would not be a control"
        )
    return LayerMapping(kind="mismatched", pairs=pairs)
