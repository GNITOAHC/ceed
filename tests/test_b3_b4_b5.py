"""B3, B4 and B5 through run_group: the baseline Group set completed.

Each of the three is the same shape — the shared backbone plus one auxiliary
signal — and each is driven here end to end through the spine with a real
training loop, real signals built from a real store's metadata, and a tiny CPU
Student. What is asserted is what the plan rests on:

* the three Groups differ from B2 in their auxiliary signal **and nothing else**,
  because the comparison is the contribution and an accidental asymmetry would
  invalidate the result rather than break the run;
* the signal that actually trained reaches the run record, so a Phase 1 row
  states its own independent variable;
* probes are deletable — the Student a B5 run leaves behind is architecturally
  identical to the one it started from.
"""

import json
from pathlib import Path

import pytest
import torch
from ceed_student.signals import COMBINE_WEIGHTS, HIDDEN_STATES, VISUAL_ADVANTAGE, build_signals
from ceed_student.training import PROBES_DIRNAME, AccelerateTrainer

from ceed_core import ArtifactStore, StoreMetadata, VectorSpec
from ceed_eval import harness_version
from ceed_student import EvaluationOutcome, TrainingBatch, run_group
from test_training import TinyStudent

CACHED_LAYERS = (4, 9, 14, 19, 24, 29)
TEACHER_HIDDEN = 6
N_EXPERTS = 5
VOCAB = 8
N_ANSWER_TOKENS = 3


class StubEvaluator:
    """Scores a Group without a model, so the spine is exercised on CPU."""

    def evaluate(self, config) -> EvaluationOutcome:
        return EvaluationOutcome(
            accuracies={"docvqa": 0.5},
            harness_version=harness_version(),
            decoding=config.evaluation.decoding,
        )


def a_store(tmp_path: Path) -> ArtifactStore:
    """A store declaring every artefact kind the three Groups read.

    One extraction fills all of it, which is the point of over-caching
    (ADR-0001): the layer axis is wider than any Group supervises, so choosing
    the CEA layers later never forces a re-extraction.
    """
    return ArtifactStore.create(
        tmp_path / "store",
        StoreMetadata(
            extraction_fingerprint="fp",
            vector_kinds={
                HIDDEN_STATES: VectorSpec(
                    dtype="float32",
                    shape=(len(CACHED_LAYERS), TEACHER_HIDDEN),
                    layers=CACHED_LAYERS,
                ),
                COMBINE_WEIGHTS: VectorSpec(
                    dtype="float32", shape=(len(CACHED_LAYERS), N_EXPERTS), layers=CACHED_LAYERS
                ),
                VISUAL_ADVANTAGE: VectorSpec(dtype="float32", shape=(1,)),
            },
        ),
    )


def a_batch(seed: int = 0) -> TrainingBatch:
    """One toy example carrying every artefact the three signals read."""
    gen = torch.Generator().manual_seed(seed)
    gold = torch.randint(0, VOCAB, (N_ANSWER_TOKENS,), generator=gen)
    combine = torch.zeros(N_ANSWER_TOKENS, len(CACHED_LAYERS), N_EXPERTS)
    combine[:, :, :2] = torch.rand(N_ANSWER_TOKENS, len(CACHED_LAYERS), 2, generator=gen)
    return TrainingBatch(
        student_inputs={"input_ids": torch.randint(0, VOCAB, (5,), generator=gen)},
        answer_token_positions=torch.tensor([2, 3, 4]),
        gold_token_ids=gold,
        teacher_topk_ids=torch.stack([gold, (gold + 1) % VOCAB], dim=1),
        teacher_topk_values=torch.tensor([[4.0, 1.0]] * N_ANSWER_TOKENS),
        artefacts={
            HIDDEN_STATES: torch.randn(
                N_ANSWER_TOKENS, len(CACHED_LAYERS), TEACHER_HIDDEN, generator=gen
            ),
            COMBINE_WEIGHTS: combine,
            VISUAL_ADVANTAGE: torch.rand(N_ANSWER_TOKENS, 1, generator=gen),
        },
    )


def a_trainer(tmp_path: Path, store: ArtifactStore, output: str = "out") -> AccelerateTrainer:
    """The same trainer B1 and B2 use, handed this Group's signals.

    Nothing about the loop changes per Group: the factory returns whatever the
    overlay declares, which is the property that makes adding a Group a
    component and a YAML file.
    """
    return AccelerateTrainer(
        build_student=lambda cfg: TinyStudent(vocab=VOCAB),
        build_batches=lambda cfg: [a_batch(0), a_batch(1)],
        build_signals=lambda cfg: build_signals(cfg, store, TinyStudent.HIDDEN),
        output_root=tmp_path / output,
        lora_targets=["proj"],
        cpu=True,
    )


def a_short_run(config, steps: int = 4):
    """The Group, with its step budget cut to something a CPU test can run."""
    training = config.training.model_copy(update={"steps": steps, "checkpoint_every": 0})
    return config.model_copy(update={"training": training})


# -- the three Groups differ from B2 in one thing ----------------------------


@pytest.mark.parametrize("group", ["b3", "b4", "b5"])
def test_each_group_is_b2_plus_exactly_one_auxiliary_signal(group, b2_config, request):
    """The single independent variable of the comparison, asserted as one.

    If a Group also differed in its corpus, its backbone weighting, its decoding
    or its evaluation, its delta against B2 would measure that difference too —
    and the run would still complete and still produce a plausible number.
    """
    config = request.getfixturevalue(f"{group}_config")

    assert len(config.auxiliary_signals) == 1
    assert not b2_config.auxiliary_signals
    assert config.corpus == b2_config.corpus
    assert config.training.backbone == b2_config.training.backbone
    assert config.training.steps == b2_config.training.steps
    assert config.training.learning_rate == b2_config.training.learning_rate
    assert config.evaluation == b2_config.evaluation
    assert config.layer_mapping == b2_config.layer_mapping


@pytest.mark.parametrize("group", ["b3", "b4", "b5"])
def test_each_group_hashes_differently_so_none_reads_anothers_checkpoint(group, b2_config, request):
    from ceed_core import run_hash

    config = request.getfixturevalue(f"{group}_config")
    assert run_hash(config) != run_hash(b2_config)


def test_the_three_groups_share_one_extraction_fingerprint(b3_config, b4_config, b5_config):
    """One extraction feeds all three: they differ only outside the fingerprint.

    This is what makes the baseline set affordable — and what makes it valid,
    since three Groups reading three separately-extracted stores would differ in
    their teacher measurement as well as their signal.
    """
    from ceed_core import extraction_fingerprint

    fingerprints = {extraction_fingerprint(c) for c in (b3_config, b4_config, b5_config)}
    assert len(fingerprints) == 1


# -- through run_group -------------------------------------------------------


@pytest.mark.parametrize(
    ("group", "signal"),
    [
        ("b3", "hidden_state_projection"),
        ("b4", "visual_advantage_reweighting"),
        ("b5", "combine_weight_probe"),
    ],
)
def test_each_group_trains_then_evaluates_and_records_its_signal(group, signal, tmp_path, request):
    config = a_short_run(request.getfixturevalue(f"{group}_config"))
    store = a_store(tmp_path)

    record = run_group(
        config,
        tmp_path / "runs",
        trainer=a_trainer(tmp_path, store),
        evaluator=StubEvaluator(),
    )

    assert record.metrics["docvqa"] == 0.5  # the Group reports accuracy
    assert record.checkpoint_dir is not None  # from a frozen, reusable checkpoint
    # The auxiliary signal's own contribution is recorded, by name.
    assert f"train.aux.{signal}" in record.metrics


@pytest.mark.parametrize("group", ["b3", "b4", "b5"])
def test_the_signal_that_trained_reaches_the_metrics_log(group, tmp_path, request):
    config = a_short_run(request.getfixturevalue(f"{group}_config"))
    record = run_group(
        config,
        tmp_path / "runs",
        trainer=a_trainer(tmp_path, a_store(tmp_path)),
    )
    lines = [json.loads(line) for line in Path(record.metrics_path).read_text().splitlines()]
    trained = next(line for line in lines if line["event"] == "run_group.trained")
    assert trained["signal_names"] == [config.auxiliary_signals[0].name]


@pytest.mark.parametrize("group", ["b3", "b4", "b5"])
def test_the_auxiliary_signal_actually_changes_the_optimised_loss(group, tmp_path, request):
    """A signal that trains nothing would produce a Group identical to B2.

    The step loss must differ from the backbone's total, or the Group is B2 under
    another name and its row in the Phase 1 table is a duplicate.
    """
    config = a_short_run(request.getfixturevalue(f"{group}_config"), steps=1)
    # No warm-up, so the signal acts from the first step and the difference is
    # visible in a one-step run.
    config = config.model_copy(
        update={
            "training": config.training.model_copy(
                update={
                    "warmup": config.training.warmup.model_copy(
                        update={"backbone_only_steps": 0, "ramp_steps": 0}
                    )
                }
            )
        }
    )
    outcome = a_trainer(tmp_path, a_store(tmp_path)).train(config)

    backbone_total = outcome.backbone_metrics["cross_entropy"] + outcome.backbone_metrics["kd"]
    assert outcome.final_loss != pytest.approx(backbone_total)
    # B4's term is a signed redistribution, so magnitude is what matters here.
    assert any(value != 0.0 for value in outcome.auxiliary_metrics.values())


def test_the_warmup_holds_a_probing_signal_out_of_the_first_steps(b5_config, tmp_path):
    """An untrained probe is noise; it must not reach the Student at step zero."""
    config = a_short_run(b5_config, steps=1)  # b5.yaml holds aux out for 200 steps
    outcome = a_trainer(tmp_path, a_store(tmp_path)).train(config)

    backbone_total = outcome.backbone_metrics["cross_entropy"] + outcome.backbone_metrics["kd"]
    assert outcome.final_loss == pytest.approx(backbone_total, rel=1e-5)


# -- probes are deletable (story 47) ----------------------------------------


def test_a_probing_group_leaves_the_student_architecturally_unchanged(
    b2_config, b5_config, tmp_path
):
    """The deployed Student must be identical to the base model, not nearly so.

    The zero-added-inference-cost claim is only literally true if nothing has to
    be stripped out afterwards. So attaching B5's probe must change nothing about
    the Student: the same trainer over the same Student trains exactly the same
    parameters whether the Group probes or not, and the probe is reported apart
    from them.
    """
    store = a_store(tmp_path)
    probing = a_trainer(tmp_path, store, output="b5").train(a_short_run(b5_config))
    plain = a_trainer(tmp_path, store, output="b2").train(a_short_run(b2_config))

    assert probing.trainable_parameters == plain.trainable_parameters
    assert probing.total_parameters == plain.total_parameters
    assert probing.probe_parameters > 0  # a probe really was trained
    assert plain.probe_parameters == 0


def test_the_probe_is_written_beside_the_checkpoint_not_into_it(b5_config, tmp_path):
    config = a_short_run(b5_config)
    outcome = a_trainer(tmp_path, a_store(tmp_path)).train(config)

    probes = Path(outcome.checkpoint_dir) / PROBES_DIRNAME / "probes.pt"
    assert probes.is_file()
    # Whatever a server loads from the checkpoint, it cannot pick the probe up.
    state = torch.load(probes, weights_only=True)
    assert state
    assert all("probes" in key for key in state)


def test_a_group_with_no_probe_writes_none(b4_config, tmp_path):
    # B4 is pure reweighting: no parameters, so nothing to save and nothing to
    # delete afterwards.
    config = a_short_run(b4_config)
    outcome = a_trainer(tmp_path, a_store(tmp_path)).train(config)

    assert outcome.probe_parameters == 0
    assert not (Path(outcome.checkpoint_dir) / PROBES_DIRNAME).exists()


# -- the baseline set is complete -------------------------------------------


def test_every_baseline_group_from_b0_to_b5_resolves(configs_dir):
    """B0 through B5 all exist as Groups the spine can run.

    Whether they have *numbers* is a matter of having run them on a GPU; that
    they are all runnable through one entry point is what this ticket delivers.
    """
    from ceed_core import resolve_group_config

    codes = []
    for group in ("b0", "b1", "b2", "b3", "b4", "b5"):
        config = resolve_group_config(
            [
                configs_dir / "base.yaml",
                configs_dir / "student.yaml",
                configs_dir / "groups" / f"{group}.yaml",
            ]
        )
        codes.append(config.group_code)
        assert config.evaluation is not None  # every baseline reports accuracy
    assert codes == ["B0", "B1", "B2", "B3", "B4", "B5"]


# -- the already-trained baselines keep their identity -----------------------

# Every baseline Group's run hash. A run hash is a pure function of the whole
# configuration, so *any* new field with a default, anywhere in the schema,
# silently changes them — and a changed hash means a completed baseline is no
# longer found, is retrained from zero, and no longer matches the provenance
# recorded beside its weights.
#
# These were last rolled deliberately, when the supervised span was corrected to
# include the token that ends the assistant's turn. Before that, base.yaml's
# layer mapping was corrected to the proportional rule it claims to follow, and
# `batch_size` became a setting the loop honours rather than one it ignored.
# Each changes what a run *is*, so every baseline was retrained.
BASELINE_HASHES = {
    "b0": "4936b71280131892723b66cef21ad0535366597955a8021e8d3626e8ca97de1f",
    "b1": "4cc9ec6e275678d5da0f4bc4a63ac6e1d23c1ef93ebaee739293fbb81e7b6455",
    "b2": "b549d25703206bce7cea95110cd22bd5b333b97091f7a1af9f1517861e1a2321",
    "b3": "43c86901570689bfe239587660aa77f095ff80d1b4ff258505f3d377fb27f624",
    "b4": "c4c052901e2ffcb0e3287b557155b515046f641063428ccd49103cf3d692167e",
    "b5": "effd5294fe8cf1a05800ce00e48ad0700dcae4d5fbdd5731928c238b493914b0",
}


@pytest.mark.parametrize(("group", "expected"), sorted(BASELINE_HASHES.items()))
def test_a_trained_baselines_run_hash_does_not_drift(group, expected, configs_dir):
    """A schema change must not silently change the identity of a trained Group.

    A hash that moves without anyone deciding it should costs a full retrain and
    orphans whatever provenance was published beside the old weights. This is the
    only thing that says so before that bill arrives.

    If this fails, the fix is almost never to update the constant. It is either
    to stop the new field reaching a Group that does not use it, or — if the
    change really does alter what a run means — to roll these deliberately and
    retrain, as ticket 08 did.
    """
    from ceed_core import resolve_group_config, run_hash

    config = resolve_group_config(
        [
            configs_dir / "base.yaml",
            configs_dir / "student.yaml",
            configs_dir / "groups" / f"{group}.yaml",
        ]
    )
    assert run_hash(config) == expected
