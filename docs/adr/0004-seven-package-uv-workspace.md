---
status: accepted
---

# `src/` is a seven-member uv workspace split along artifact-flow boundaries

The codebase is a uv workspace of seven packages — `ceed_core`, `ceed_data`, `ceed_interventions`, `ceed_teacher`, `ceed_modes`, `ceed_student`, `ceed_eval` — each owning exactly one stage of the pipeline, with the on-disk artifact store as the seam between stages.

Seven packages is more than a project of this size would normally justify, and the reason is **dependency isolation**, not code organisation. The dependency sets here are heterogeneous and actively hostile to one another: the intervention pipeline wants `diffusers`, teacher extraction wants a pinned pre-release `transformers` with forward hooks, student training wants `accelerate`, mode analysis wants `scikit-learn`/`scipy`, evaluation wants `lmms-eval`. In a single resolver environment, a `diffusers` pin can silently move the `transformers` version that the teacher extraction correctness depends on. A workspace gives each package its own dependency set resolved against one shared lockfile, so the environments cannot drift in the dependencies they *do* share.

The split follows artifact flow rather than layering, because the artifact boundaries are already the places where the pipeline can be stopped, inspected, and restarted — which makes them the natural test seams too.

## Consequences

- Seven `pyproject.toml` files to maintain, and cross-package refactors carry more ceremony than moving a file.
- `ceed_core` owns the shared vocabulary (domain types, config schema, artifact store read/write) and is the only package every other package depends on. It must stay free of model, framework, and dataset dependencies.
- A stage can be run without installing the others' dependency trees, which is what makes toy-scale end-to-end runs on constrained hardware practical.
