---
status: accepted
---

# Attribution is measured over teacher-forced gold answers, with the thinking gate disabled

All cached teacher artefacts are produced by teacher-forcing the gold answer with the teacher's thinking gate off, so an answer token is a gold token, never a generated one. A separate, cheap free-generation pass per example produces nothing but a correctness flag.

The research plan describes attribution over "the emitted token", which implies free generation, while simultaneously specifying a student backbone of cross-entropy on gold answers plus logit KD, which implies gold. These are different sequences. Under free generation the teacher's token positions do not align with the student's targets, so every per-token loss would need a teacher→student alignment step first — survivable for a three-token GQA answer, unreliable for DocVQA. Teacher forcing makes positions align 1:1 across every group, which is precisely what makes the twelve-group comparison commensurable, and makes extraction deterministic and reproducible.

The thinking gate stays off because turning it on means the attributed sequence contains hundreds of rationale tokens the student is never trained to produce; the student would have to be trained to think for the probes to have matching positions, roughly doubling scope for a benefit the plan describes only as "for analysis".

## Consequences

- The plan's §2.1 wording must be restated as "the gold token under teacher forcing"; the paper cannot claim attribution over what the teacher actually emits at inference.
- Rationale spans are unavailable for mode labelling and Phase 0 analysis in v1.
- Attribution is measured on a distribution the teacher is not, at inference time, in — a limitation worth stating explicitly, and the free-generation correctness pass is what keeps supervision away from examples where teacher forcing is papering over a wrong answer.
