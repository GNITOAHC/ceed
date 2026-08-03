"""The store-backed dataloader: corpus in, aligned training batches out.

The alignment asserted here is the one that silently ruins a distillation run if
it is wrong. Answer token ``t`` is *predicted* at the position before it, so a
batch's positions must be the prompt's last position onward and its gold ids must
be the answer's tokens. The teacher's cached logits are keyed by the answer
token's ordinal — the Teacher and Student share a tokenizer but not a prompt
length — and a store that disagrees about how many answer tokens an example has
must fail rather than quietly misalign supervision.

A fake processor stands in for the real one, so this runs on CPU in the fast tier.
"""

import numpy as np
import pytest
import torch
from ceed_student.dataset import (
    TOP_K_LOGIT_IDS,
    TOP_K_LOGIT_VALUES,
    answer_text,
    build_batches,
    encode_example,
)

from ceed_core import ArtifactRow, ArtifactStore, StoreMetadata, VectorSpec, store_root
from ceed_data import CORPUS_FILENAME, Example, ImageStore, read_examples, write_examples

TOP_K = 3
# The fake tokenizer maps each character to its ordinal, so an answer's token
# count is its character count — easy to reason about in assertions.
PROMPT_TOKENS = [1, 2, 3, 4, 5]
# The token this fake template ends an assistant turn with, and which generation
# would stop on. Supervising it is what teaches the Student to stop.
TERMINATOR = 999


class FakeTokenizer:
    """Char-per-token, with a chat template that terminates an assistant turn."""

    all_special_ids: tuple[int, ...] = (TERMINATOR,)

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [ord(c) for c in text]}

    def apply_chat_template(self, messages, add_generation_prompt=False, **kwargs):
        ids = list(PROMPT_TOKENS)
        if not add_generation_prompt:
            for message in messages:
                if message["role"] == "assistant":
                    ids += [ord(c) for c in message["content"]] + [TERMINATOR]
        return {"input_ids": ids}


class FakeProcessor:
    """Stands in for the Student's processor: fixed prompt, char-per-token answers."""

    def __init__(self):
        self.tokenizer = FakeTokenizer()

    def apply_chat_template(self, messages, **kwargs):
        return {
            "input_ids": torch.tensor([PROMPT_TOKENS]),
            "attention_mask": torch.ones(1, len(PROMPT_TOKENS), dtype=torch.long),
        }


def an_example(example_id="docvqa:q1", answer="ab") -> Example:
    return Example(
        example_id=example_id,
        dataset="docvqa",
        image_fingerprint="fp",
        question="what?",
        answers=(answer,),
    )


@pytest.fixture
def image_store(tmp_path, monkeypatch) -> ImageStore:
    """An image store whose decode step is stubbed — no real image bytes needed."""
    store = ImageStore(tmp_path / "images")
    monkeypatch.setattr("ceed_student.dataset.load_image", lambda _store, _example: object())
    return store


def a_teacher_store(tmp_path, example_id="docvqa:q1", n_tokens=3) -> ArtifactStore:
    """A store holding top-k logits for ``n_tokens`` answer tokens, keyed by ordinal."""
    metadata = StoreMetadata(
        extraction_fingerprint="fp-test",
        vector_kinds={
            TOP_K_LOGIT_IDS: VectorSpec(dtype="int32", shape=(TOP_K,)),
            TOP_K_LOGIT_VALUES: VectorSpec(dtype="float32", shape=(TOP_K,)),
        },
    )
    store = ArtifactStore.create(store_root(tmp_path / "store", "fp-test"), metadata)
    rows = [
        ArtifactRow(
            example_id=example_id,
            answer_token_index=ordinal,
            gold_token_id=0,
            correct=True,
            vectors={
                TOP_K_LOGIT_IDS: np.array([ordinal, ordinal + 1, ordinal + 2], np.int32),
                TOP_K_LOGIT_VALUES: np.array([3.0, 2.0, 1.0], np.float32),
            },
        )
        for ordinal in range(n_tokens)
    ]
    store.writer("w0").write(rows)
    return store


# -- corpus round-trip -------------------------------------------------------


def test_the_corpus_round_trips_through_jsonl(tmp_path):
    examples = [an_example("docvqa:q1"), an_example("docvqa:q2", answer="xyz")]
    path = tmp_path / CORPUS_FILENAME
    assert write_examples(path, examples) == 2
    assert list(read_examples(path)) == examples


# -- the shared prompt -------------------------------------------------------


def test_the_prompt_asks_for_a_short_answer():
    # VQA gold answers are a word or a number. Without this instruction an
    # instruction-tuned Student answers in a sentence and scores zero under every
    # metric, which would sink all Groups equally and hide any real difference.
    from ceed_student.dataset import SHORT_ANSWER_INSTRUCTION, chat_messages

    text = chat_messages(an_example(), object())[0]["content"][1]["text"]
    assert "what?" in text
    assert SHORT_ANSWER_INSTRUCTION in text


def test_the_prompt_carries_the_image_before_the_question():
    from ceed_student.dataset import chat_messages

    image = object()
    content = chat_messages(an_example(), image)[0]["content"]
    assert content[0] == {"type": "image", "image": image}
    assert content[1]["type"] == "text"


# -- encoding and alignment --------------------------------------------------


def test_the_gold_answer_supervised_is_the_first_accepted_answer():
    example = Example(
        example_id="gqa:g1",
        dataset="gqa",
        image_fingerprint="fp",
        question="q",
        answers=("cat", "a cat"),
    )
    assert answer_text(example) == "cat"


def test_encoding_appends_the_gold_answer_to_the_prompt(image_store):
    inputs, prompt_length, answer_ids = encode_example(FakeProcessor(), an_example(), image_store)

    assert prompt_length == len(PROMPT_TOKENS)
    assert answer_ids.tolist() == [ord("a"), ord("b"), TERMINATOR]
    # Teacher forcing: the sequence is prompt, gold answer, end of turn.
    assert inputs["input_ids"][0].tolist() == [*PROMPT_TOKENS, ord("a"), ord("b"), TERMINATOR]
    assert inputs["attention_mask"].shape == inputs["input_ids"].shape


def test_the_supervised_span_ends_with_the_token_that_stops_generation():
    """Supervising the answer alone teaches what to say and never when to stop.

    Measured, not hypothesised: B1 — supervised fine-tuning with no teacher —
    reached a training cross-entropy of 0.13 over 3.7 epochs and a DocVQA score
    of 0.32, emitting the right answer and then transcribing the rest of the
    page. The distilling Groups were shielded by the teacher's distribution, so
    the defect presented as a B1 problem rather than a corpus-wide one.
    """
    from ceed_student.dataset import answer_target_ids, turn_terminator_id

    processor = FakeProcessor()
    terminator = turn_terminator_id(processor)
    assert terminator == TERMINATOR
    assert answer_target_ids(processor, an_example(answer="ab"), terminator) == [
        ord("a"),
        ord("b"),
        TERMINATOR,
    ]


def test_a_template_that_terminates_nothing_supervises_the_answer_alone():
    # Not every Student's template appends a stop token; one that does not must
    # not have a spurious id invented for it.
    from ceed_student.dataset import answer_target_ids

    assert answer_target_ids(FakeProcessor(), an_example(answer="ab"), None) == [
        ord("a"),
        ord("b"),
    ]


def test_answer_positions_are_where_the_answer_tokens_are_predicted(tmp_path, image_store):
    store = a_teacher_store(tmp_path, n_tokens=3)
    batches = build_batches(FakeProcessor(), [an_example()], image_store, store, kd_weight=1.0)

    batch = batches[0]
    # Three supervised tokens, predicted from the prompt's last position onward.
    assert batch.answer_token_positions.tolist() == [
        len(PROMPT_TOKENS) - 1,
        len(PROMPT_TOKENS),
        len(PROMPT_TOKENS) + 1,
    ]
    assert batch.gold_token_ids.tolist() == [ord("a"), ord("b"), TERMINATOR]


def test_the_teacher_topk_is_read_in_answer_token_order(tmp_path, image_store):
    store = a_teacher_store(tmp_path, n_tokens=3)
    batch = build_batches(FakeProcessor(), [an_example()], image_store, store, kd_weight=1.0)[0]

    assert batch.teacher_topk_ids.shape == (3, TOP_K)
    assert batch.teacher_topk_values.shape == (3, TOP_K)
    # Ordinal 0's cached ids come first, ordinal 1's second.
    assert batch.teacher_topk_ids[0].tolist() == [0, 1, 2]
    assert batch.teacher_topk_ids[1].tolist() == [1, 2, 3]


def test_positions_gold_and_teacher_rows_all_have_one_entry_per_answer_token(tmp_path, image_store):
    store = a_teacher_store(tmp_path, example_id="docvqa:q1", n_tokens=4)
    example = an_example(answer="abc")
    batch = build_batches(FakeProcessor(), [example], image_store, store, kd_weight=1.0)[0]

    n = len(batch.gold_token_ids)
    assert n == 4  # three answer tokens and the terminator
    assert len(batch.answer_token_positions) == n
    assert batch.teacher_topk_ids.shape[0] == n
    assert batch.teacher_topk_values.shape[0] == n


# -- the teacher-free path (B1) ---------------------------------------------


def test_a_teacher_free_group_builds_without_a_store(image_store):
    # B1 has kd_weight 0, so it needs no cached teacher logits at all.
    batches = build_batches(FakeProcessor(), [an_example()], image_store, None, kd_weight=0.0)

    batch = batches[0]
    assert batch.teacher_topk_ids.shape == (3, 1)  # gold-only placeholder
    assert torch.equal(batch.teacher_topk_values, torch.zeros(3, 1))
    assert batch.teacher_topk_ids[:, 0].tolist() == batch.gold_token_ids.tolist()


def test_a_distilling_group_without_a_store_is_refused(image_store):
    # B2 with no extraction run must fail here, not silently train on nothing.
    with pytest.raises(ValueError, match="artifact store"):
        build_batches(FakeProcessor(), [an_example()], image_store, None, kd_weight=1.0)


def test_a_store_missing_an_answer_token_fails_rather_than_misaligning(tmp_path, image_store):
    # The store holds two tokens but the Student encodes three: a tokenisation
    # disagreement that must surface, since misaligned supervision is silent.
    store = a_teacher_store(tmp_path, n_tokens=2)
    with pytest.raises(ValueError, match=r"store 2, Student 4"):
        build_batches(
            FakeProcessor(), [an_example(answer="abc")], image_store, store, kd_weight=1.0
        )


def test_an_example_the_store_never_covered_fails_rather_than_training_on_nothing(
    tmp_path, image_store
):
    # A corpus example with no rows at all: the store was built from a different
    # corpus, or extraction did not finish. Either way the Group must not quietly
    # train on the examples that happen to be present.
    store = a_teacher_store(tmp_path, example_id="docvqa:q1", n_tokens=3)
    with pytest.raises(ValueError, match="no rows for 1 of 1"):
        build_batches(
            FakeProcessor(),
            [an_example(example_id="docvqa:missing")],
            image_store,
            store,
            kd_weight=1.0,
        )


# -- the placeholder is genuinely inert -------------------------------------


def test_the_placeholder_topk_contributes_nothing_to_the_backbone(image_store):
    from ceed_student import backbone_loss

    batch = build_batches(FakeProcessor(), [an_example()], image_store, None, kd_weight=0.0)[0]
    logits = torch.randn(len(batch.gold_token_ids), 1000)
    loss = backbone_loss(
        logits,
        batch.gold_token_ids,
        batch.teacher_topk_ids,
        batch.teacher_topk_values,
        kd_weight=0.0,
    )
    assert torch.isclose(loss.total, loss.cross_entropy)


# -- auxiliary artefacts, read by the same ordinal key -----------------------


def a_signal_store(tmp_path, example_id="docvqa:q1", n_tokens=3) -> ArtifactStore:
    """A store also holding the per-token artefacts B3, B4 and B5 read.

    Each token's values are distinct and derived from its ordinal, so a read that
    returned the wrong token's artefacts is visible in the assertion rather than
    plausible.
    """
    metadata = StoreMetadata(
        extraction_fingerprint="fp-aux",
        vector_kinds={
            TOP_K_LOGIT_IDS: VectorSpec(dtype="int32", shape=(TOP_K,)),
            TOP_K_LOGIT_VALUES: VectorSpec(dtype="float32", shape=(TOP_K,)),
            "hidden_states": VectorSpec(dtype="float16", shape=(2, 3), layers=(9, 19)),
            "visual_advantage": VectorSpec(dtype="float32", shape=(1,)),
        },
    )
    store = ArtifactStore.create(store_root(tmp_path / "store", "fp-aux"), metadata)
    rows = [
        ArtifactRow(
            example_id=example_id,
            answer_token_index=ordinal,
            gold_token_id=0,
            correct=True,
            vectors={
                TOP_K_LOGIT_IDS: np.array([ordinal, ordinal + 1, ordinal + 2], np.int32),
                TOP_K_LOGIT_VALUES: np.array([3.0, 2.0, 1.0], np.float32),
                "hidden_states": np.full((2, 3), ordinal, np.float16),
                "visual_advantage": np.array([ordinal * 0.5], np.float32),
            },
        )
        for ordinal in range(n_tokens)
    ]
    store.writer("w0").write(rows)
    return store


def test_auxiliary_artefacts_are_stacked_in_answer_token_order(tmp_path, image_store):
    store = a_signal_store(tmp_path)
    batches = build_batches(
        FakeProcessor(),
        [an_example(answer="ab")],
        image_store,
        store,
        kd_weight=1.0,
        artefact_kinds=["hidden_states", "visual_advantage"],
    )
    hidden = batches[0].artefacts["hidden_states"]
    advantage = batches[0].artefacts["visual_advantage"]

    # One row per supervised token, in ordinal order, with the store's own shapes.
    assert hidden.shape == (3, 2, 3)
    assert torch.equal(advantage.reshape(-1), torch.tensor([0.0, 0.5, 1.0]))
    assert torch.equal(hidden[0], torch.zeros(2, 3))
    assert torch.equal(hidden[1], torch.ones(2, 3))


def test_half_precision_artefacts_are_widened_before_they_reach_a_loss(tmp_path, image_store):
    # The store holds hidden states in float16 to fit on disk; a loss computed
    # against a half-precision target would cap the precision of the comparison.
    store = a_signal_store(tmp_path)
    batches = build_batches(
        FakeProcessor(),
        [an_example(answer="ab")],
        image_store,
        store,
        kd_weight=1.0,
        artefact_kinds=["hidden_states"],
    )
    assert batches[0].artefacts["hidden_states"].dtype is torch.float32


def test_a_group_needing_artefacts_without_a_store_is_refused(image_store):
    # The same fail-fast as a distilling Group with no store: better here than
    # at the first training step.
    with pytest.raises(ValueError, match="hidden_states"):
        build_batches(
            FakeProcessor(),
            [an_example()],
            image_store,
            None,
            kd_weight=0.0,
            artefact_kinds=["hidden_states"],
        )


def test_a_backbone_only_group_carries_no_artefacts(tmp_path, image_store):
    batches = build_batches(
        FakeProcessor(), [an_example()], image_store, a_teacher_store(tmp_path), kd_weight=1.0
    )
    assert batches[0].artefacts == {}


# -- the corpus is encoded a batch at a time, not all at once ----------------


def test_encoding_costs_scale_with_batches_asked_for_not_corpus_size(tmp_path, image_store):
    """Encoding every example up front is what took the host's OOM killer.

    A page's pixel_values alone is ~7.7 MB, so a 4,282-example corpus is tens of
    gigabytes per process, and four Groups training in parallel does not fit in
    329 GB. What must hold is that the work of building a corpus does not grow
    with the corpus — so a 200-example corpus, indexed three times, encodes a
    handful of examples and not two hundred.

    (The slack is for the test tier only: beartype samples one element when it
    checks a value against ``Sequence[TrainingBatch]``, and it is installed under
    pytest alone.)
    """
    processor = FakeProcessor()
    encoded: list[str] = []
    original = processor.apply_chat_template
    processor.apply_chat_template = lambda messages, **kw: (  # type: ignore[method-assign]
        encoded.append("x") or original(messages, **kw)
    )

    corpus = build_batches(
        processor,
        [an_example(f"docvqa:q{i}") for i in range(200)],
        image_store,
        None,
        kd_weight=0.0,
    )
    assert len(corpus) == 200
    built = len(encoded)
    assert built <= 2  # not 200

    for index in (0, 7, 13):
        corpus[index]
    assert len(encoded) - built <= 4  # one per access, plus sampling slack


def test_indexing_the_same_example_twice_gives_the_same_batch(tmp_path, image_store):
    # Re-encoding is the price of not holding the corpus in memory; it is only
    # safe because it is deterministic, so an example means the same thing in
    # epoch one and epoch four.
    corpus = build_batches(
        FakeProcessor(), [an_example(answer="abc")], image_store, None, kd_weight=0.0
    )
    first, second = corpus[0], corpus[0]
    assert torch.equal(first.gold_token_ids, second.gold_token_ids)
    assert torch.equal(first.answer_token_positions, second.answer_token_positions)
    assert torch.equal(first.student_inputs["input_ids"], second.student_inputs["input_ids"])
