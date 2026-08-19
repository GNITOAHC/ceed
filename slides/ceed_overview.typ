// CEED overview deck - built on the CP2 slide-kit (Forest Teal theme).
// Covers: background, motivation, related work, methodology, experiment
// design, and implementation status (from .scratch/ceed/issues/).

#import "/slide-kit/core.typ": *
#import "/slide-kit/themes.typ": forest-teal

#let theme = forest-teal
#let title-slide = title-slide.with(theme: theme)
#let contents-slide = contents-slide.with(theme: theme)
#let section-slide = section-slide.with(theme: theme)
#let content-slide = content-slide.with(theme: theme)
#let callout = callout.with(theme: theme)
#let card = card.with(theme: theme)
#show: body => deck(theme: theme, body)

// ---------------------------------------------------------------
// Small local diagram components (kept in this file, not the kit)
// ---------------------------------------------------------------

#let node(title, sub: none, color: theme.primary, w: 1fr, fill: theme.surface) = box(
  width: w,
  inset: 9pt,
  radius: 6pt,
  fill: fill,
  stroke: 1pt + color,
)[
  #align(center)[
    #text(size: 14pt, weight: "bold", fill: color)[#title]
    #if sub != none {
      v(2pt)
      text(size: 10.5pt, fill: theme.muted)[#sub]
    }
  ]
]

#let step-arrow(size: 22pt, color: theme.muted, label: none) = align(center + horizon)[
  #text(size: size, fill: color)[$arrow.r$]
  #if label != none {
    v(-4pt)
    text(size: 9pt, fill: theme.muted)[#label]
  }
]

#let down-arrow(size: 22pt, color: theme.muted) = align(center)[
  #text(size: size, fill: color)[$arrow.b$]
]

#let pipeline(..items) = {
  let seq = items.pos()
  let cols = ()
  for i in range(seq.len()) {
    cols.push(3fr)
    if i < seq.len() - 1 { cols.push(0.5fr) }
  }
  let cells = ()
  for i in range(seq.len()) {
    cells.push(seq.at(i))
    if i < seq.len() - 1 { cells.push(step-arrow()) }
  }
  grid(columns: cols, column-gutter: 6pt, align: horizon, ..cells)
}

#let dot(color, label) = box[
  #box(width: 9pt, height: 9pt, radius: 999pt, fill: color)
  #h(4pt)
  #text(size: 11.5pt, fill: theme.text)[#label]
]

#let status-chip(n, done: false) = box(
  width: 26pt,
  height: 20pt,
  radius: 3pt,
  inset: (y: 3pt),
  fill: if done { theme.primary } else { theme.surface },
  stroke: 0.7pt + (if done { theme.primary } else { theme.border }),
)[
  #align(center)[
    #text(size: 10pt, weight: "bold", fill: if done { white } else { theme.muted })[#n]
  ]
]

// ---------------------------------------------------------------
// 1. Title
// ---------------------------------------------------------------

#title-slide(
  course: [CEED],
  title: [Causal Expert–Evidence Distillation],
  details: [Distilling a sparse MoE VLM teacher into a compute-matched dense student · 2026/08/19],
  eyebrow: [RESEARCH OVERVIEW],
)

// ---------------------------------------------------------------
// 2. Contents
// ---------------------------------------------------------------

#contents-slide(items: (
  ([Background], [3]),
  ([Motivation], [6]),
  ([Related Work], [9]),
  ([Our Methodology], [12]),
  ([Experiment Design], [20]),
  ([Implementation Status], [26]),
  ([Risks & Summary], [29]),
))

// ---------------------------------------------------------------
// Background
// ---------------------------------------------------------------

#section-slide(
  title: [Background],
  subtitle: [A sparse teacher, a dense student, and the gap between them.],
)

#content-slide(title: [The teacher–student pair])[
  #two-columns(
    [
      #node([Teacher], sub: [gemma-4-26B-A4B-it], color: theme.accent)
      #v(6pt)
      #grid(
        columns: (1fr, 1fr, 1fr, 1fr),
        column-gutter: 4pt,
        row-gutter: 4pt,
        ..range(8).map(i => box(
          height: 22pt, radius: 3pt, fill: theme.surface, stroke: 0.6pt + theme.border,
        )[#align(center + horizon)[#text(size: 9pt, fill: theme.muted)[E#i]]])
      )
      #v(4pt)
      #text(size: 12.5pt, fill: theme.muted)[Every layer routes each token to a few of 128 experts — 26B total, ~4B active.]
    ],
    [
      #node([Student], sub: [gemma-4-E4B-it], color: theme.primary)
      #v(6pt)
      #box(
        width: 100%, height: 22pt, radius: 3pt, fill: theme.surface, stroke: 0.6pt + theme.border,
      )[#align(center + horizon)[#text(size: 10pt, fill: theme.muted)[dense · router-free]]]
      #v(4pt)
      #text(size: 12.5pt, fill: theme.muted)[Effective 4B via Per-Layer Embeddings — matched active compute, no routing at all.]
    ],
  )
  #v(0.5em)
  #callout(title: [Why this pair])[
    Active-compute is matched, so any gain is attributable to *transferred structure*, not extra capacity.
  ]
]

#content-slide(title: [The standard transfer signal: logit KD])[
  #pipeline(
    node([Teacher forward pass], sub: [answer tokens]),
    node([Top-k logits], sub: [per token]),
    node([KD loss], sub: [\+ CE on gold answer], color: theme.accent),
    node([Student], sub: [backbone for every group]),
  )
  #v(0.8em)
  #callout(kind: "accent", title: [The gap])[
    Logit KD teaches the student *what* the teacher said. It says nothing about *which internal
    computation* produced it — that structure is what CEED targets.
  ]
]

// ---------------------------------------------------------------
// Motivation
// ---------------------------------------------------------------

#section-slide(
  title: [Motivation],
  subtitle: [The router predicts computation; it does not report it.],
)

#content-slide(title: [The router is an unreliable witness])[
  #two-columns(
    [
      == What the router reports
      #node([Effective combine weight], sub: [predicted before compute runs], color: theme.muted)
      #v(0.6em)
      == What actually happened
      #node([Causal Expert Attribution], sub: [measured — expert ablated, ΔlogP recorded], color: theme.primary)
    ],
    [
      #callout(title: [Empirical])[
        Gating weight and ablated ΔNLL correlate only weakly (arXiv:2606.18304). Frequently-fired experts can be
        near-removable; some ablations *reduce* loss.
      ]
      #v(0.4em)
      #callout(kind: "accent", title: [Consequence])[
        Distilling router scores risks teaching the student the router's blind spots, not the teacher's
        division of labour.
      ]
    ],
  )
]

#content-slide(title: [Research question])[
  #callout(title: [Thesis])[
    #text(size: 17pt)[
      Does the *causally verified* division of computational labour in a MoE teacher — and the visual
      evidence driving it — transfer to a compute-matched dense student, when made an explicit training
      signal?
    ]
  ]
  #v(0.5em)
  #grid(
    columns: (1fr, 1fr),
    column-gutter: 16pt,
    card(title: [CEA])[Per-token, per-expert *measured* causal attribution — not router probability.],
    card(title: [ECC])[How that attribution reorganises under targeted, controlled visual interventions.],
  )
  #v(0.35em)
  #text(size: 12pt, fill: theme.muted)[All CEED machinery is removed after training — the deployed student is architecturally unchanged.]
]

// ---------------------------------------------------------------
// Related work
// ---------------------------------------------------------------

#section-slide(
  title: [Related Work],
  subtitle: [Precedent map: what is taken, what supports the case, what remains open.],
)

#content-slide(title: [Precedent map])[
  #set text(size: 13.5pt)
  #grid(
    columns: (1fr, 1fr, 1fr),
    column-gutter: 12pt,
    card(title: [Taken])[
      Sparse→dense KD (OneS); inactive-expert KD; counterfactual-image KD (VA-OPD); router-to-router KD;
      input-attribution KD (AD-KD, TSD); causal-abstraction distillation (*DIITO*).
    ],
    card(title: [Supports the case])[
      Routing–attribution misalignment in pruning literature; dormant/correlational routing; per-expert
      ablation maps as interpretability; implicit-leakage risk (Shadow-MoE).
    ],
    card(title: [Remains open])[
      Module-level causal attribution as a *distillation target*; coupling input interventions to internal
      attribution reorganisation; the compute-matched MoE→router-free setting.
    ],
  )
]

#content-slide(title: [Positioning against DIITO])[
  #text(size: 12.5pt)[DIITO (NAACL 2022) owns "distill causal structure, not just outputs" — CEED sits inside that family, not against it.]
  #v(0.4em)
  #set text(size: 12pt)
  #grid(
    columns: (1fr, 1fr),
    column-gutter: 20pt,
    [
      #node([DIITO], sub: [causal abstraction], color: theme.muted)
      #v(3pt)
      - Intervenes on *activations* (swaps hidden states)
      - Requires layer/width-aligned dense pairs
      - Matches counterfactual outputs
    ],
    [
      #node([CEED], sub: [this work], color: theme.primary)
      #v(3pt)
      - Intervenes on *inputs* (image regions) and *modules* (experts)
      - No activation alignment needed
      - Matches attribution structure, not activations
    ],
  )
  #v(0.35em)
  #callout(kind: "accent", title: [Why it matters])[
    #set text(size: 12.5pt)
    Activation interchange is ill-defined across the MoE→PLE architecture gap — CEED's input/module-level
    interventions are the reason this pair is tractable at all.
  ]
]

// ---------------------------------------------------------------
// Methodology
// ---------------------------------------------------------------

#section-slide(
  title: [Our Methodology],
  subtitle: [CEA and ECC: two causally-grounded objects, distilled through three auxiliary losses.],
)

#content-slide(title: [Causal Expert Attribution (CEA)])[
  #pipeline(
    node([Run teacher], sub: [record activated + near-miss experts]),
    node([Ablate expert $e$], sub: [zero / mean-replace, re-run from layer m]),
    node([ΔlogP], sub: [attribution of expert $e$], color: theme.accent),
  )
  #v(0.7em)
  #two-columns(
    [
      #callout(title: [Near-miss experts])[
        Top-k activated *plus* the next 2–3 by gating score — causally decisive experts can sit just below
        threshold.
      ]
    ],
    [
      #callout(kind: "accent", title: [Result])[
        A per-(token, layer) *attribution vector* — comparable across contexts, unlike expert IDs.
      ]
    ],
  )
]

#content-slide(title: [Functional modes])[
  #grid(
    columns: (1fr, 1fr, 1fr),
    column-gutter: 10pt,
    box(inset: 10pt, radius: 6pt, fill: theme.surface, stroke: 1pt + theme.border)[
      #align(center)[
        #text(size: 12pt, fill: theme.muted)[raw attribution vectors]
        #v(4pt)
        #grid(columns: (1fr,1fr,1fr,1fr), column-gutter: 3pt, row-gutter: 3pt,
          ..range(8).map(i => box(width: 10pt, height: 10pt, radius: 999pt,
            fill: (theme.primary, theme.accent, theme.muted, theme.primary-dark).at(calc.rem(i,4)))))
      ]
    ],
    step-arrow(label: [cluster / layer]),
    box(inset: 10pt, radius: 6pt, fill: theme.surface, stroke: 1pt + theme.primary)[
      #align(center)[
        #text(size: 12pt, weight: "bold", fill: theme.primary)[8–32 modes]
        #v(4pt)
        #text(size: 10.5pt, fill: theme.muted)[e.g. visual extraction, numeric manipulation, surface realization]
      ]
    ],
  )
  #v(0.8em)
  #callout(title: [Why modes])[
    Coarser and more stable than raw vectors, invariant to expert-ID permutation — the supervision unit for
    the student's probe. Labels come from post-hoc analysis, never assumption.
  ]
]

#content-slide(title: [Evidence–Computation Correspondence (ECC)])[
  #two-columns(
    [
      #node([Relevant region], sub: [answer-critical, inpainted away], color: theme.accent)
      #v(0.15em)
      #down-arrow(size: 14pt)
      #v(0.15em)
      #node([Large attribution reorganisation], sub: [\+ large output effect], color: theme.accent)
    ],
    [
      #node([Control region], sub: [matched size/texture, unrelated], color: theme.muted)
      #v(0.15em)
      #down-arrow(size: 14pt)
      #v(0.15em)
      #node([Near-zero reorganisation], sub: [\+ near-zero output effect], color: theme.muted)
    ],
  )
  #v(0.3em)
  #callout(kind: "accent", title: [The desired shape])[
    #set text(size: 12.5pt)
    Selectivity profile = large relevant effect, near-zero control effect. Coupling strength gates which
    tokens receive the coupling loss.
  ]
]

#content-slide(title: [Four-cell diagnosis])[
  #v(0.4em)
  #grid(
    columns: (110pt, 1fr, 1fr),
    column-gutter: 8pt,
    row-gutter: 8pt,
    align: horizon,
    [], node([Output changed], color: theme.muted), node([Output stable], color: theme.muted),
    node([Attribution changed], color: theme.muted, w: 110pt),
    node([Strong coupling], sub: [selects tokens for coupling loss], color: theme.accent),
    node([Internal compensation], sub: [genuine, not router noise]),
    node([Attribution stable], color: theme.muted, w: 110pt),
    node([Ambiguous], sub: [rare — investigate]),
    node([Invariance], sub: [defines the control mask], color: theme.primary),
  )
]

#content-slide(title: [Student objectives])[
  #node([Backbone: CE + top-k logit KD], sub: [every group shares this], color: theme.muted, w: 100%)
  #v(0.5em)
  #down-arrow()
  #v(0.5em)
  #grid(
    columns: (1fr, 1fr, 1fr),
    column-gutter: 12pt,
    card(title: [1. Mode probing])[Removable linear probes predict per-token teacher mode. Deleted after training.],
    card(title: [2. Selectivity matching])[Match the student's per-token effect vector (relevant vs. control) to the teacher's.],
    card(title: [3. Coupling consistency])[On strong-coupling tokens, the probed mode shift must track the teacher's attribution shift.],
  )
  #v(0.5em)
  #text(size: 12pt, fill: theme.muted)[Small gated weights (≈0.05–0.1 × CE), warm-up schedule (CE+KD first).]
]

#content-slide(title: [Architecture: the artifact store as the seam])[
  #pipeline(
    node([Teacher], sub: [measured once, offline], color: theme.accent),
    node([Artifact store], sub: [CEA + ECC on disk, fingerprinted], color: theme.muted),
    node([Student training], sub: [teacher never loaded], color: theme.primary),
    node([Deployed E4B], sub: [probes removed, zero added cost]),
  )
  #v(0.8em)
  #callout(title: [Extraction fingerprint])[
    A content hash of layer set, ablation definition, combine-weight definition, thinking gate, and model
    revision — artefacts from different definitions can never silently mix.
  ]
]

// ---------------------------------------------------------------
// Experiment design
// ---------------------------------------------------------------

#section-slide(
  title: [Experiment Design],
  subtitle: [Twelve groups, one shared corpus and backbone, one varying signal.],
)

#content-slide(title: [The groups])[
  #set text(size: 12.5pt)
  #table(
    columns: (auto, 1fr, 1.6fr),
    stroke: 0.5pt + theme.border,
    inset: 6pt,
    fill: (col, row) => if row == 0 { theme.surface } else { white },
    table.header([*Group*], [*Auxiliary signal*], [*Role*]),
    [B0], [none — zero-shot], [floor],
    [B1], [none — SFT, no teacher], [isolates teacher value],
    [B2], [CE + logit KD], [*primary baseline*],
    [B3], [\+ hidden-state projection KD], [standard alternative],
    [B4], [\+ VA-OPD reproduction], [critical external baseline],
    [B5], [\+ router-score distillation], [routing vs. attribution ablation],
    [C1 / C2], [shuffled labels / mismatched layer], [controls],
    [E1], [\+ CEA mode probing], [experimental component 1],
    [E2], [\+ selectivity-profile matching], [experimental component 2],
    [E3 / E4], [E1+E2 / +coupling consistency], [additivity / full CEED],
  )
]

#content-slide(title: [Phase 0 — go/no-go (teacher characterization)])[
  #set text(size: 13pt)
  - *0.1* Gating score vs. measured attribution correlation → expect ρ ≈ 0.3–0.6 (misalignment)
  - *0.2* Mode assignment MI vs. token-function, vs. router/hidden-state clusters → CEA must carry more
  - *0.3* Mode stability across paraphrases and image pairs → modes track function, not token identity
  - *0.4* ECC responsiveness: relevant vs. control reorganisation → expect ≥2–3×
  #v(0.6em)
  #callout(kind: "accent", title: [Fallback])[
    ρ > 0.9 everywhere → CEA ≈ routing; abandon the pivot, fall back to a router-signal design (B5) with
    reduced claims.
  ]
]

#content-slide(title: [Phase 1 — decision logic])[
  #set text(size: 13pt)
  #dot(theme.primary, [*E1 > B5 > B2* — gap concentrated on high-divergence tokens (+0.5–1.5 aggregate)])
  #v(6pt)
  #dot(theme.primary, [*E2 > B4* on grounding selectivity + hallucination, even at comparable accuracy])
  #v(6pt)
  #dot(theme.accent, [*E1 ≫ C1* — non-negotiable; shuffled-label parity would mean regularization, not transfer])
  #v(6pt)
  #dot(theme.muted, [*E4 vs. E3* — coupling super-additivity, smallest and least certain effect])
  #v(0.8em)
  #callout(title: [The paper survives on E1 / E2 + Phase 0])[
    Even if E4 ≈ E3 (coupling is a null result), the mode-probing and selectivity-matching results stand on
    their own.
  ]
]

#content-slide(title: [Phase 2 — fairness and robustness])[
  #grid(
    columns: (1fr, 1fr),
    column-gutter: 14pt,
    row-gutter: 10pt,
    card(title: [Matched-FLOPs re-run])[Grant B2 the ablation/intervention compute as extra vanilla-KD data. Edge should shrink ~⅓ but survive.],
    card(title: [Held-out generalization])[Gains should persist, attenuated, outside the training task families.],
    card(title: [Layer-mapping robustness])[Repeat E1 under two alternative teacher↔student mappings.],
    card(title: [Inference verification])[Byte-identical E4B architecture; zero latency delta after probe removal.],
  )
]

#content-slide(title: [Execution schedule and parallelism])[
  #set text(size: 11.5pt)
  #table(
    columns: (1.3fr, 3fr, auto),
    stroke: 0.5pt + theme.border,
    inset: 6pt,
    fill: (col, row) => if row == 0 { theme.surface } else { white },
    table.header([*Track*], [*What*], [*Weeks*]),
    [Immediate], [B0–B4, B5, shared intervention pipeline, corpus, teacher logits], [1–2],
    [Gated on pipeline], [C1, C2, E1 (mode labels) · E2 (needs validated interventions)], [4–5],
    [Gated on conclusions], [E3, E4 — only if Phase 0 passes go-criteria], [6–12],
    [Fairness], [Phase 2: matched-FLOPs, generalization, robustness], [12–16],
  )
  #v(0.6em)
  #callout(title: [Key point])[Baseline checkpoints are frozen and reused everywhere — no baseline is ever retrained.]
]

// ---------------------------------------------------------------
// Implementation status
// ---------------------------------------------------------------

#section-slide(
  title: [Implementation Status],
  subtitle: [Twenty-one tracked issues, from tooling to the full E4 pipeline.],
)

#content-slide(title: [Issue dependency map])[
  #set text(size: 10.5pt)
  #grid(
    columns: (auto, 1fr),
    column-gutter: 8pt,
    row-gutter: 5pt,
    text(fill: theme.muted)[Infra], grid(columns: (auto, auto), column-gutter: 4pt, status-chip("01", done: true), status-chip("02", done: true)),
    text(fill: theme.muted)[Corpus / B0–B2], grid(columns: (auto,auto,auto,auto), column-gutter: 4pt,
      status-chip("03", done: true), status-chip("04", done: true), status-chip("05", done: true), status-chip("06", done: true)),
    text(fill: theme.muted)[Baselines], grid(columns: (auto,auto), column-gutter: 4pt, status-chip("07", done: true), status-chip("08")),
    text(fill: theme.muted)[Phase 0], grid(columns: (auto,auto,auto,auto), column-gutter: 4pt,
      status-chip("09"), status-chip("10"), status-chip("11"), status-chip("13")),
    text(fill: theme.muted)[Interventions], grid(columns: (auto,auto), column-gutter: 4pt, status-chip("12"), status-chip("17")),
    text(fill: theme.muted)[E-track], grid(columns: (auto,auto,auto,auto,auto), column-gutter: 4pt,
      status-chip("14"), status-chip("15"), status-chip("18"), status-chip("19"), status-chip("20")),
    text(fill: theme.muted)[Wrap-up], grid(columns: (auto,auto), column-gutter: 4pt, status-chip("16"), status-chip("21")),
  )
  #v(0.6em)
  #grid(columns: (auto, auto), column-gutter: 16pt,
    dot(theme.primary, [done]), dot(theme.surface.darken(0%), [ready-for-agent, not started]))
]

#content-slide(title: [Status snapshot])[
  #grid(
    columns: (1fr, 1fr, 1fr),
    column-gutter: 12pt,
    card(title: [Done])[
      #text(size: 13pt)[Workspace, CI, `run_group` spine, synthetic-teacher fixture, corpora (DocVQA/GQA/ChartQA), B0 zero-shot, artifact store + extraction, B1/B2 backbone.]
    ],
    card(title: [In progress])[
      #text(size: 13pt)[07 (B1/B2 backbone) at 7/8 checklist items.]
    ],
    card(title: [Not started])[
      #text(size: 13pt)[B3–B5; the full Phase 0 sweep (09–11, 13); intervention pipeline (12); the entire E-track (14–20); layer-mapping study and the Phase 1 table (16, 21).]
    ],
  )
  #v(0.6em)
  #callout(kind: "accent", title: [Reading the map])[
    Everything through B2 — the primary baseline — is built. The critical path from here is the
    intervention pipeline (12) and the Phase 0 sweep (09–11, 13): both gate the entire E-track.
  ]
]

// ---------------------------------------------------------------
// Risks and summary
// ---------------------------------------------------------------

#section-slide(
  title: [Risks & Summary],
  subtitle: [What could go wrong, and what this work claims.],
)

#content-slide(title: [Risks and mitigations])[
  #set text(size: 12.5pt)
  #table(
    columns: (1.1fr, 1.6fr),
    stroke: 0.5pt + theme.border,
    inset: 6pt,
    fill: (col, row) => if row == 0 { theme.surface } else { white },
    table.header([*Risk*], [*Mitigation*]),
    [Ablation cost], [3 layers, top-k+2 experts, activation caching, answer tokens only],
    [Attribution noise], [Distill modes, not raw vectors; report Phase-0 reliability],
    [Region-identification errors], [Teacher-verified effect gaps; annotation-backed core results],
    [PLE layer mapping], [Empirical selection + two-mapping robustness report],
    [Teacher errors], [Correctness-flag gating],
    [Auxiliary-loss interference], [Warm-up schedule, small gated weights, early stopping],
  )
]

#content-slide(title: [Contributions])[
  #set text(size: 13pt)
  + Causal-attribution characterization of a production multimodal MoE
  + CEA as a distillation target — module-level *measured* attribution, not router scores or outputs
  + Evidence–computation coupling transfer — how structure reorganises under visual interventions
  + Inference-free transfer via removable probes — zero added cost
  + A grounding-selectivity evaluation protocol
  #v(0.7em)
  #callout(title: [Defensible statement])[
    #text(size: 13.5pt, style: "italic")[
      "We show that a multimodal MoE's router is an unreliable witness to its own computation and distill
      what the model measurably does instead — transferring evidence–computation correspondence into a
      router-free, compute-matched student at zero added inference cost."
    ]
  ]
]
