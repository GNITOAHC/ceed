# 18 — E2: selectivity-profile matching

**What to build:** The second experimental component, and the first that requires the Student to be run more than once per training step. The Student sees the original image and both intervened images, and its vector of per-token output effects is matched to the teacher's — with a ranking constraint that the relevant-region effect must exceed the control-region effect by the teacher's own margin.

**Blocked by:** 13, 14, 17

**Status:** ready-for-agent

- [ ] The training step runs the Student on the original image and on both intervened images, within one step
- [ ] The Student's per-token selectivity profile is matched to the teacher's with a robust loss, plus a ranking constraint enforcing the teacher's relevant-over-control margin
- [ ] The loss is gated per token by teacher-measured coupling strength and follows the warm-up schedule
- [ ] Memory and step time under the three-view forward are measured and recorded, since this is the most expensive Group to train
- [ ] E2 runs through `run_group`, reports accuracy, and reports grounding selectivity on held-out interventions
- [ ] E2 is compared against B4 on grounding selectivity and hallucination, which is the comparison this component exists to make
