"""Named array shapes for data that crosses CEED package boundaries.

CEED's arrays are distinguished mainly by their rank and by what each axis
means, not by their element type. An attribution vector, an ECC tensor and a
selectivity profile are all arrays of floats, so a plain ``ndarray`` annotation
lets any of them be passed where another is expected -- which type checks
cleanly and produces silently wrong science.

These aliases name the axes instead. Under pytest an import hook installs
``beartype`` over every CEED package, so the annotations below are enforced at
runtime; outside pytest they are ordinary annotations and cost nothing.

``ceed_core`` deliberately depends on no tensor framework, so the boundary type
is ``numpy.ndarray``; conversion to and from framework tensors happens in the
packages that own a model.
"""

from __future__ import annotations

import numpy as np
from jaxtyping import Float

AttributionVector = Float[np.ndarray, "tokens layers probed_experts"]
"""Measured Causal Expert Attribution, per answer token and probed layer.

``probed_experts`` covers the activated experts plus the near-miss experts, so
its length is fixed by the extraction configuration rather than by the router.
"""

EccTensor = Float[np.ndarray, "tokens regions layers probed_experts"]
"""Evidence-Computation Correspondence: attribution change per intervened region."""

SelectivityProfile = Float[np.ndarray, "tokens regions"]
"""Per-token output effect of each intervention, ordered relevant-then-control."""

CombineWeights = Float[np.ndarray, "tokens layers experts"]
"""Effective combine weights: softmax weight multiplied by the per-expert scale.

This is what CEED means by "gating score" throughout. It is the quantity that
actually multiplies an expert's output, and therefore the strongest comparator
against which routing-attribution divergence can be claimed.
"""


def echo_selectivity_profile(profile: SelectivityProfile) -> SelectivityProfile:
    """Return ``profile`` unchanged, having validated its shape.

    This function carries no domain logic. It exists so the test suite can prove
    that runtime shape enforcement is genuinely live rather than merely
    configured -- a guarantee worth checking explicitly, because every other
    shape annotation in the codebase silently depends on it.

    Args:
        profile: A two-dimensional array of per-token, per-region output
            effects.

    Returns:
        The same array, unmodified.

    Raises:
        jaxtyping.TypeCheckError: Under pytest, if ``profile`` does not have
            rank two. Outside pytest the import hook is absent and no check
            runs.
    """
    return profile
