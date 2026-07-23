"""The three test tiers exist, and the fast tier excludes the other two.

The fast tier is what runs in CI and on every change, so it must stay CPU-only
and quick. The marked tests here are deliberately trivial: they exist so the
tier configuration is exercised, and so a later ticket adding a genuinely slow
or GPU-bound test has a working example to copy.
"""

import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_fast_tier_runs_by_default():
    assert True


@pytest.mark.gpu
def test_gpu_tier_is_available_when_selected():
    assert True


@pytest.mark.slow
def test_slow_tier_is_available_when_selected():
    assert True


def test_default_selection_deselects_the_marked_tiers():
    """Collecting with default options must skip the gpu and slow tests.

    Run as a subprocess because a test cannot observe its own session's
    deselection.
    """
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "--collect-only", "-q", str(Path(__file__))],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )
    collected = result.stdout
    assert "test_fast_tier_runs_by_default" in collected
    assert "test_gpu_tier_is_available_when_selected" not in collected
    assert "test_slow_tier_is_available_when_selected" not in collected


@pytest.mark.parametrize("marker", ["gpu", "slow"])
def test_a_marked_tier_can_be_selected_explicitly(marker):
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "--collect-only", "-q", "-m", marker, str(Path(__file__))],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )
    assert f"test_{marker}_tier_is_available_when_selected" in result.stdout


def test_an_unregistered_marker_is_an_error(tmp_path):
    """A typo'd marker must fail loudly rather than silently never matching.

    Without --strict-markers, `@pytest.mark.gpuu` is silently accepted and the
    test quietly runs in the fast tier forever.
    """
    offending = tmp_path / "test_typo_marker.py"
    offending.write_text(
        "import pytest\n\n\n@pytest.mark.gpuu\ndef test_typo():\n    assert True\n"
    )
    result = subprocess.run(
        # -c is required: the offending file lives outside the repo, so without
        # it pytest infers its rootdir from the temporary directory and never
        # loads the configuration under test.
        [
            sys.executable,
            "-m",
            "pytest",
            "-c",
            str(REPO_ROOT / "pyproject.toml"),
            "--collect-only",
            "-q",
            str(offending),
        ],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )
    assert result.returncode != 0
    assert "gpuu" in result.stdout + result.stderr
