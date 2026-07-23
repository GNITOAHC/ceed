"""Pytest configuration for the CEED workspace.

Installs the jaxtyping import hook over every CEED package before those packages
are imported, so that the shape annotations in ``ceed_core.shapes`` are enforced
at runtime for the duration of the test session.

The hook is installed here, and only here, so that shape checking is active
under pytest and absent in production runs -- where the overhead would be paid
on every call in a training loop.
"""

from jaxtyping import install_import_hook

CEED_PACKAGES = (
    "ceed_core",
    "ceed_data",
    "ceed_eval",
    "ceed_interventions",
    "ceed_modes",
    "ceed_student",
    "ceed_teacher",
)

_import_hook = install_import_hook(CEED_PACKAGES, "beartype.beartype")
