# 04 — Corpora: DocVQA, GQA, and ChartQA

**What to build:** A researcher can assemble the training corpus once and know that every Group provably trains on the identical examples, with the evidence annotations each dataset carries available downstream and ChartQA's intervention ineligibility enforced by the data rather than remembered by the operator.

**Blocked by:** 02

**Status:** ready-for-agent

- [x] DocVQA, GQA, and ChartQA all load into one Example type with stable identifiers
- [x] DocVQA examples carry their OCR words and boxes, so relevant regions and text-substitution counterfactuals are derivable without a second annotation pass
- [x] GQA examples carry the scene-graph object box tied to the answer
- [x] Each example declares whether it is eligible for interventions; ChartQA examples are ineligible (see plan amendment A8)
- [x] Splits are deterministic and their seed is recorded
- [x] A corpus manifest records exactly which examples and splits were assembled, so a Group trained in week two is comparable to one trained in week ten
- [x] Images are addressable such that the teacher and the Student provably see byte-identical inputs for the same example
