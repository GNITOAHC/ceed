# 07 — B1 and B2: the training loop and shared backbone

**What to build:** A researcher can train the Student. Groups B1 (supervised fine-tuning, no teacher) and B2 (the primary baseline: cross-entropy plus top-k logit distillation) both run to completion and report numbers, and the machinery that every later Group plugs into exists.

**Blocked by:** 05, 06

**Status:** ready-for-agent

- [x] A training loop built on `accelerate` handles checkpointing and resumption, so a multi-day Group survives preemption — `AccelerateTrainer` in `ceed_student/training.py`; `save_state`/`load_state` + a `progress.json` step counter. `test_training.py` proves a second run resumes from the checkpoint and runs only the remaining steps.
- [x] The backbone — cross-entropy on gold answers plus top-k logit distillation — is implemented once and shared, so no Group can accidentally differ in it — `ceed_student/backbone.py::backbone_loss`, the one objective the loop calls. `test_backbone.py` pins the decomposition and the top-k-support behaviour.
- [x] The auxiliary-signal contract exists: a signal declares which artefact kinds it requires and which forward views it needs, and contributes a per-token loss that the trainer gates and schedules. B1 and B2 attach zero signals — `ceed_student/auxiliary.py` (`AuxiliarySignal`, `ForwardView`, `required_artefact_kinds`/`required_forward_views`, `verify_store_supports` fail-fast, `coupling_gate`, `training_loss`). B1/B2 overlays carry `auxiliary_signals: []`.
- [x] Auxiliary terms are introduced on a warm-up schedule after the backbone, so they cannot destabilise early training — `WarmupSchedule`; the loop scales aux terms by `schedule.scale(step)` via `training_loss`. Tested in `test_auxiliary.py`.
- [x] LoRA and full fine-tuning are selected by one configuration flag, and the run record states unambiguously which was used, so a LoRA number can never reach the Phase 1 table (see ADR-0005) — `param_efficiency` selects the path; `TrainingOutcome.param_efficiency` and `RunRecord.param_efficiency` record it. `test_training.py` shows LoRA trains only the adapter.
- [~] A LoRA run of B2 completes on the Volta box at toy scale — gpu-tier test `test_b1_b2.py::test_b2_lora_completes_at_toy_scale_on_the_volta_box` is wired but skips: it needs the real Student builder and a store-backed dataloader (the same gpu-tier follow-on ticket 06 deferred). The loop itself is proven end-to-end under LoRA on the tiny CPU Student.
- [x] B1 and B2 both run through `run_group` and report accuracy through the evaluation harness — `run_group` gained an injected `Trainer` seam alongside the `Evaluator`; `test_b1_b2.py` drives both Groups train-then-evaluate and asserts the record carries training metrics, the checkpoint, and accuracy.
- [x] Baseline checkpoints are frozen and reusable, so no baseline is ever retrained for a later comparison — a completed checkpoint (`progress.completed_steps >= steps`) is reused: `test_a_completed_baseline_is_not_retrained` asserts `steps_run == 0` on a second run. `RunRecord.checkpoint_dir` is the durable handle.

## Deferred (gpu-tier follow-on)

Criterion 6's real B2 LoRA run needs two pieces that are gpu-tier and shared with ticket 06's deferred real-teacher work: a real Student builder (`build_student`) that loads gemma-4-e4b-it in fp16 with a probe-free head, and a store-backed dataloader (`build_batches`) that tokenises the corpus and reads the teacher's cached top-k / gold ids from the artifact store. The CPU seam (`TrainableStudent`, `TrainingBatch`) is exactly what those two builders must satisfy, and the whole loop — backbone, checkpoint, resume, LoRA — is proven against it. Evaluating the *trained checkpoint* (rather than the base Student) is the matching evaluator-side follow-on; the checkpoint path is already recorded on the run record for the real evaluator to resolve.
