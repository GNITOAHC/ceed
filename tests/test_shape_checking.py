"""Runtime shape enforcement is live under pytest, and absent outside it."""

import subprocess
import sys
from pathlib import Path

import jaxtyping
import numpy as np
import pytest
from ceed_core.shapes import echo_selectivity_profile

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_a_well_formed_selectivity_profile_is_accepted():
    profile = np.zeros((4, 2), dtype=np.float32)
    assert echo_selectivity_profile(profile) is profile


def test_a_wrongly_ranked_array_is_rejected():
    """An attribution-shaped array passed as a selectivity profile must fail.

    This is the error class the shape annotations exist to catch: both arrays
    hold floats, so nothing but the rank distinguishes them, and a static type
    checker sees no problem.
    """
    attribution_shaped = np.zeros((4, 3, 11), dtype=np.float32)
    with pytest.raises(jaxtyping.TypeCheckError, match="tokens regions"):
        echo_selectivity_profile(attribution_shaped)


def test_shape_enforcement_is_absent_outside_pytest():
    """Outside the test session the annotations are inert.

    The import hook is installed only by the root conftest, so a production run
    pays no per-call checking cost in a training loop.
    """
    probe = (
        "import numpy as np; from ceed_core.shapes import echo_selectivity_profile; "
        "echo_selectivity_profile(np.zeros((4, 3, 11), dtype=np.float32)); "
        "print('not enforced')"
    )
    result = subprocess.run(
        [sys.executable, "-c", probe],
        capture_output=True,
        text=True,
        check=True,
        cwd=REPO_ROOT,
    )
    assert result.stdout.strip() == "not enforced"
