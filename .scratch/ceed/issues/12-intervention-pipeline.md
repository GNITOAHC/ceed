# 12 — The intervention pipeline

**What to build:** A researcher can turn an eligible example into a relevant/control intervention pair — the answer-critical region removed, and a matched but unrelated region removed the same way — and trust that the pair differs in evidence and not in artefact.

The control region is what makes the whole grounding claim work: global degradation cannot teach invariance to irrelevant regions, and a control that isn't genuinely matched is just a second relevant region.

**Blocked by:** 04

**Status:** ready-for-agent

- [ ] Relevant regions come from dataset annotations — DocVQA's OCR boxes located against the answer string, GQA's scene-graph object box tied to the answer
- [ ] Control regions are matched to their relevant region in size and texture, and are unrelated to the answer
- [ ] Regions are removed by inpainting rather than rectangle masking, so the Student cannot learn to detect the intervention instead of losing the evidence
- [ ] Text-substitution counterfactuals are produced for document examples, altering only the intended span
- [ ] Intervened images are persisted and content-addressed, so the teacher's and the Student's intervened passes see byte-identical inputs
- [ ] ChartQA examples are never intervened on, enforced by the eligibility flag carried in the corpus
- [ ] Property tests hold: a control region matches its relevant region in area within tolerance, does not overlap it, and regeneration is reproducible
