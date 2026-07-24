"""Shared fixtures for the test suite.

The null Group's overlay stack is used by both the configuration tests and the
run_group seam tests, so its paths live here rather than being repeated in each.
"""

from pathlib import Path

import pytest

from ceed_core import GroupConfig, resolve_group_config

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def configs_dir() -> Path:
    """The repository's configuration overlay directory."""
    return REPO_ROOT / "configs"


@pytest.fixture
def null_group_overlays(configs_dir: Path) -> list[Path]:
    """The overlay stack composing the null Group: base, student, then group."""
    return [
        configs_dir / "base.yaml",
        configs_dir / "student.yaml",
        configs_dir / "groups" / "null.yaml",
    ]


@pytest.fixture
def null_config(null_group_overlays: list[Path]) -> GroupConfig:
    """The resolved null Group configuration."""
    return resolve_group_config(null_group_overlays)
