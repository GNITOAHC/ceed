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


class FakeProcessor:
    """Stands in for the Student's processor: fixed prompt, char-per-token answers."""

    class _Tok:
        def __call__(self, text, add_special_tokens=False):
            return {"input_ids": [ord(c) for c in text]}

    def __init__(self):
        self.tokenizer = self._Tok()

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


def a_teacher_store(tmp_path, example_id="docvqa:q1", n_tokens=2) -> ArtifactStore:
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
    assert answer_ids.tolist() == [ord("a"), ord("b")]
    # Teacher forcing: the sequence is prompt then gold answer.
    assert inputs["input_ids"][0].tolist() == [*PROMPT_TOKENS, ord("a"), ord("b")]
    assert inputs["attention_mask"].shape == inputs["input_ids"].shape


def test_answer_positions_are_where_the_answer_tokens_are_predicted(tmp_path, image_store):
    store = a_teacher_store(tmp_path, n_tokens=2)
    batches = build_batches(FakeProcessor(), [an_example()], image_store, store, kd_weight=1.0)

    batch = batches[0]
    # Two answer tokens, predicted at the prompt's last position and the one after.
    assert batch.answer_token_positions.tolist() == [len(PROMPT_TOKENS) - 1, len(PROMPT_TOKENS)]
    assert batch.gold_token_ids.tolist() == [ord("a"), ord("b")]


def test_the_teacher_topk_is_read_in_answer_token_order(tmp_path, image_store):
    store = a_teacher_store(tmp_path, n_tokens=2)
    batch = build_batches(FakeProcessor(), [an_example()], image_store, store, kd_weight=1.0)[0]

    assert batch.teacher_topk_ids.shape == (2, TOP_K)
    assert batch.teacher_topk_values.shape == (2, TOP_K)
    # Ordinal 0's cached ids come first, ordinal 1's second.
    assert batch.teacher_topk_ids[0].tolist() == [0, 1, 2]
    assert batch.teacher_topk_ids[1].tolist() == [1, 2, 3]


def test_positions_gold_and_teacher_rows_all_have_one_entry_per_answer_token(tmp_path, image_store):
    store = a_teacher_store(tmp_path, example_id="docvqa:q1", n_tokens=3)
    example = an_example(answer="abc")
    batch = build_batches(FakeProcessor(), [example], image_store, store, kd_weight=1.0)[0]

    n = len(batch.gold_token_ids)
    assert n == 3
    assert len(batch.answer_token_positions) == n
    assert batch.teacher_topk_ids.shape[0] == n
    assert batch.teacher_topk_values.shape[0] == n


# -- the teacher-free path (B1) ---------------------------------------------


def test_a_teacher_free_group_builds_without_a_store(image_store):
    # B1 has kd_weight 0, so it needs no cached teacher logits at all.
    batches = build_batches(FakeProcessor(), [an_example()], image_store, None, kd_weight=0.0)

    batch = batches[0]
    assert batch.teacher_topk_ids.shape == (2, 1)  # gold-only placeholder
    assert torch.equal(batch.teacher_topk_values, torch.zeros(2, 1))
    assert batch.teacher_topk_ids[:, 0].tolist() == batch.gold_token_ids.tolist()


def test_a_distilling_group_without_a_store_is_refused(image_store):
    # B2 with no extraction run must fail here, not silently train on nothing.
    with pytest.raises(ValueError, match="artifact store"):
        build_batches(FakeProcessor(), [an_example()], image_store, None, kd_weight=1.0)


def test_a_store_missing_an_answer_token_fails_rather_than_misaligning(tmp_path, image_store):
    # The store holds two tokens but the Student encodes three: a tokenisation
    # disagreement that must surface, since misaligned supervision is silent.
    store = a_teacher_store(tmp_path, n_tokens=2)
    with pytest.raises(Exception, match="no row"):
        build_batches(
            FakeProcessor(), [an_example(answer="abc")], image_store, store, kd_weight=1.0
        )


# -- the placeholder is genuinely inert -------------------------------------


def test_the_placeholder_topk_contributes_nothing_to_the_backbone(image_store):
    from ceed_student import backbone_loss

    batch = build_batches(FakeProcessor(), [an_example()], image_store, None, kd_weight=0.0)[0]
    logits = torch.randn(2, 300)
    loss = backbone_loss(
        logits,
        batch.gold_token_ids,
        batch.teacher_topk_ids,
        batch.teacher_topk_values,
        kd_weight=0.0,
    )
    assert torch.isclose(loss.total, loss.cross_entropy)
