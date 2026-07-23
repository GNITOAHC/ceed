# Domain Docs

How the engineering skills should consume this repo's domain documentation when exploring the codebase.

## Before exploring, read these

- **`CONTEXT.md`** at the repo root — the CEED glossary.
- **`docs/adr/`** — read ADRs that touch the area you're about to work in.

If any of these files don't exist, **proceed silently**. Don't flag their absence; don't suggest creating them upfront. The `/domain-modeling` skill (reached via `/grill-with-docs` and `/improve-codebase-architecture`) creates them lazily when terms or decisions actually get resolved.

## File structure

This repo is **single-context**:

```
/
├── CONTEXT.md
├── docs/adr/
│   ├── 0001-answer-token-scoped-artifact-store.md
│   └── ...
└── src/
```

**Do not convert this to a multi-context layout.** `src/` is a seven-member uv workspace (`ceed_core`, `ceed_data`, `ceed_interventions`, `ceed_teacher`, `ceed_modes`, `ceed_student`, `ceed_eval`), and the package count is a dependency-isolation decision, not a domain boundary — see ADR-0004. There is one CEED domain and one glossary; `ceed_teacher` and `ceed_student` share the same vocabulary rather than each owning a private one. A `CONTEXT-MAP.md` here would fragment a single ubiquitous language across seven files for no benefit.

## Use the glossary's vocabulary

When your output names a domain concept (in an issue title, a refactor proposal, a hypothesis, a test name), use the term as defined in `CONTEXT.md`. Don't drift to synonyms the glossary explicitly avoids.

This matters unusually much here: several avoided terms are avoided because they are *wrong*, not merely off-style. Writing "MoE layer" hides the shared dense path that ADR-0003 turns on; writing "gating score" without meaning the effective combine weight makes the routing–attribution divergence claim ambiguous; writing "generated token" contradicts ADR-0002's teacher forcing.

If the concept you need isn't in the glossary yet, that's a signal — either you're inventing language the project doesn't use (reconsider) or there's a real gap (note it for `/domain-modeling`).

## Flag ADR conflicts

If your output contradicts an existing ADR, surface it explicitly rather than silently overriding:

> _Contradicts ADR-0007 (event-sourced orders) — but worth reopening because…_
