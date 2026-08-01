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


def _resolve(configs_dir: Path, group: str) -> GroupConfig:
    return resolve_group_config(
        [
            configs_dir / "base.yaml",
            configs_dir / "student.yaml",
            configs_dir / "groups" / f"{group}.yaml",
        ]
    )


@pytest.fixture
def b0_config(configs_dir: Path) -> GroupConfig:
    """The resolved B0 (zero-shot Student) configuration."""
    return _resolve(configs_dir, "b0")


@pytest.fixture
def b1_config(configs_dir: Path) -> GroupConfig:
    """The resolved B1 (supervised fine-tuning, no teacher) configuration."""
    return _resolve(configs_dir, "b1")


@pytest.fixture
def b2_config(configs_dir: Path) -> GroupConfig:
    """The resolved B2 (primary baseline: CE + logit KD) configuration."""
    return _resolve(configs_dir, "b2")


@pytest.fixture
def b3_config(configs_dir: Path) -> GroupConfig:
    """The resolved B3 (hidden-state projection distillation) configuration."""
    return _resolve(configs_dir, "b3")


@pytest.fixture
def b4_config(configs_dir: Path) -> GroupConfig:
    """The resolved B4 (VA-OPD reproduction) configuration."""
    return _resolve(configs_dir, "b4")


@pytest.fixture
def b5_config(configs_dir: Path) -> GroupConfig:
    """The resolved B5 (router combine-weight distillation) configuration."""
    return _resolve(configs_dir, "b5")
