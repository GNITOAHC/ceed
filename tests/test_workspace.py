"""The workspace resolves, every member is importable, and the seams hold."""

import importlib
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]

CEED_PACKAGES = [
    "ceed_core",
    "ceed_data",
    "ceed_eval",
    "ceed_interventions",
    "ceed_modes",
    "ceed_student",
    "ceed_teacher",
]

# Dependencies that ADR-0004 keeps out of ceed_core: model runtimes, dataset
# machinery, and training frameworks.
HEAVY_DEPENDENCIES = {
    "accelerate",
    "datasets",
    "diffusers",
    "lmms-eval",
    "peft",
    "torch",
    "transformers",
}


def declared_dependencies(package: str) -> set[str]:
    manifest = tomllib.loads((REPO_ROOT / "src" / package / "pyproject.toml").read_text())
    return {
        dependency.split(">")[0].split("=")[0].split("<")[0].strip()
        for dependency in manifest["project"]["dependencies"]
    }


@pytest.mark.parametrize("package", CEED_PACKAGES)
def test_package_is_importable(package):
    assert importlib.import_module(package) is not None


@pytest.mark.parametrize("package", CEED_PACKAGES)
def test_package_declares_a_version(package):
    assert importlib.import_module(package).__version__ == "0.1.0"


@pytest.mark.parametrize("package", CEED_PACKAGES)
def test_package_has_a_module_docstring(package):
    assert importlib.import_module(package).__doc__


def test_ceed_core_declares_no_heavy_dependencies():
    assert not declared_dependencies("ceed_core") & HEAVY_DEPENDENCIES


@pytest.mark.parametrize("package", [p for p in CEED_PACKAGES if p != "ceed_core"])
def test_every_other_package_depends_on_ceed_core(package):
    assert "ceed-core" in declared_dependencies(package)


def test_importing_ceed_core_pulls_in_no_heavy_dependencies():
    """Importing ceed_core must not drag a model or training framework into memory.

    Checked in a subprocess because the pytest interpreter has already imported
    torch by way of other packages' tests, which would mask the very thing this
    asserts.
    """
    probe = (
        "import sys, ceed_core, ceed_core.shapes; "
        "roots = {name.split('.')[0] for name in sys.modules}; "
        f"print(','.join(sorted(roots & {HEAVY_DEPENDENCIES!r})))"
    )
    result = subprocess.run(
        [sys.executable, "-c", probe],
        capture_output=True,
        text=True,
        check=True,
        cwd=REPO_ROOT,
    )
    assert result.stdout.strip() == ""
