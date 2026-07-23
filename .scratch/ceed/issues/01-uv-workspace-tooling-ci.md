# 01 — uv workspace, tooling, and CI

**What to build:** A researcher can clone the repo, run one sync command, and get a working environment on the Volta box — with linting, type checking, shape checking, and the test tiers all runnable and green. Pure prefactor: no research behaviour yet, but every later ticket lands on this.

**Blocked by:** None — can start immediately.

**Status:** ready-for-agent

- [ ] `src/` is a uv workspace with all seven members present and importable: `ceed_core`, `ceed_data`, `ceed_interventions`, `ceed_teacher`, `ceed_modes`, `ceed_student`, `ceed_eval`
- [ ] Each member declares its own dependencies; `ceed_core` depends on no model, dataset, or training framework, and every other member depends on it
- [ ] One lockfile carries both hardware targets via conflicting extras; syncing the Volta extra resolves torch from the sm70-capable index, and syncing the scale-out extra resolves a newer torch (see ADR-0006)
- [ ] Python is pinned to 3.13
- [ ] ruff lint and format run clean, with the Google docstring convention enforced on public APIs only
- [ ] mypy runs strict over `ceed_core` and lenient elsewhere
- [ ] `jaxtyping` is available to all members, with `beartype` runtime shape enforcement active under pytest and inactive outside it
- [ ] pytest is configured with three tiers — a fast CPU default, a `gpu` marker, and a `slow` marker — and the default tier excludes the other two
- [ ] CI runs ruff, mypy, and the fast tier on every change
