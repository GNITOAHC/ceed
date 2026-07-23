# CEED

Causal Expert–Evidence Distillation: distilling the measured division of computational labour from a sparse MoE vision-language teacher (`gemma-4-26B-A4B-it`) into a compute-matched dense student (`gemma-4-E4B-it`).

The research plan is `ceed.md`, amended by `docs/plan-amendments-v2.1.md`. The implementation spec is `.scratch/ceed/spec.md`.

## Development

Everything runs through `uv`; never invoke `python` or `pytest` directly.

```bash
uv sync                     # Volta box: resolves torch from the cu126 index (sm70)
uv run pytest               # fast tier only (gpu and slow are deselected)
uv run pytest -m gpu        # real weights on a CUDA device
uv run ruff check . && uv run ruff format --check .
uv run mypy                 # strict over ceed_core, lenient elsewhere
```

The scale-out target is the explicit opt-in, and is what CI uses:

```bash
uv sync --no-default-groups --group dev --group modern
```

`src/` is a seven-member uv workspace. Runtime shape checking is installed by the
root `conftest.py` and is therefore active only under pytest — see
`ceed_core.shapes` for the named axis aliases that cross package boundaries.

## Agent skills

### Issue tracker

Local markdown — issues and specs live under `.scratch/<feature>/`. See `docs/agents/issue-tracker.md`.

### Triage labels

The five canonical roles, using their default label strings. See `docs/agents/triage-labels.md`.

### Domain docs

Single-context — root `CONTEXT.md` plus `docs/adr/`. See `docs/agents/domain.md`.
