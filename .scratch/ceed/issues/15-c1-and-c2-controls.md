# 15 — C1 and C2: the controls

**What to build:** The two Groups that decide whether E1's gain is knowledge transfer or merely the regularising effect of having an auxiliary task at all. C1 probes on shuffled Mode labels; C2 probes under a deliberately mismatched layer mapping.

These are the first controls a reviewer will demand, and the plan calls E1 ≫ C1 non-negotiable. The related literature's finding that routing habits leak implicitly through vanilla distillation makes it doubly necessary.

**Blocked by:** 08, 14

**Status:** ready-for-agent

- [ ] **C1** trains the same probe against shuffled Mode labels, with the auxiliary-loss budget matched to E1
- [ ] The shuffle is recorded and reproducible from the run's seed, so C1 is rerunnable
- [ ] **C2** trains the same probe under a deliberately mismatched layer mapping, exercising the layer mapping as a first-class object
- [ ] Both run through `run_group` and report accuracy, with the run record naming the shuffle and the mapping used
- [ ] Neither control required a change to the training loop or to the E1 component
