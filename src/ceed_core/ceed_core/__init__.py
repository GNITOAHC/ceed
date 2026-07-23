"""CEED domain types, configuration schema, artifact store, and extraction fingerprinting.

This package is the shared vocabulary every other CEED package imports. It
deliberately depends on no model, dataset, or training framework, so that the
heavy and mutually hostile dependency trees of the other packages cannot reach
it (see docs/adr/0004-seven-package-uv-workspace.md).
"""

__version__ = "0.1.0"
