#!/usr/bin/env python
"""Cache the Teacher's per-answer-token artefacts — everything B2 to B5 read.

The artifact store is the single seam between teacher measurement and student
training (ADR-0001): the Teacher is never loaded during training, so everything a
Group's objective needs is measured once and cached here. Four kinds are
available, and which of them this script writes is chosen by flag:

* ``top_k_logit_ids`` / ``top_k_logit_values`` — the shared backbone's
  distillation targets. **B2** and every Group above it.
* ``hidden_states`` — the residual state leaving each candidate layer. **B3**.
* ``combine_weights`` — every expert's effective combine weight, read off the
  router (plan amendment A3). **B5**.
* ``visual_advantage`` — how much likelier the Teacher makes each gold token when
  it can see the image's fine detail, from a second pass over a degraded copy.
  **B4**, reproducing VA-OPD; see ADR-0007.

None of this is the Causal Expert Attribution extraction, which needs the hooked
forward that can ablate an expert and re-run the tail. The combine weights are a
byproduct of the ordinary forward, which is exactly why B5 ships with the
baselines while the CEA Groups wait for the ablation engine.

**Over-cache.** Combine weights are written for all 30 teacher layers and hidden
states for 6 candidates by default, because the three CEA layers are chosen by
Phase 0.0 and re-extracting the corpus to change that choice would cost days. The
store records which layers its stacked artefacts hold, so a Group resolves a
layer against the store rather than against a convention.

**Keying.** Rows are keyed by the answer token's *ordinal* within the gold
answer, not its absolute sequence position. The Teacher and Student share a
tokenizer but emit different numbers of image placeholder tokens, so an absolute
position would mean different things on the two sides; an ordinal means the same
thing on both. The dataloader reads the same ordinal.

Following ADR-0002 the gold answer is teacher-forced with the thinking gate off,
so an answer token is always a gold token and the Teacher's positions align 1:1
with the Student's cross-entropy targets. A separate greedy free-generation pass
records one correctness flag per example.

Examples:
    # B2 only (what a logit-KD Group needs)
    uv run python scripts/extract_teacher_artifacts.py --corpus data/corpus --store data/store

    # everything B2 through B5 need, in one pass over the corpus
    uv run python scripts/extract_teacher_artifacts.py \
        --corpus data/corpus --store data/store-full --all
"""

from __future__ import annotations

import argparse
import io
from collections.abc import Callable
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import torch
from ceed_core.config import (
    GroupConfig,
    load_overlay,
    merge_overlays,
)
from ceed_data.manifest import CorpusManifest
from ceed_teacher.extraction import (
    COMBINE_WEIGHTS,
    HIDDEN_STATES,
    TOP_K_LOGIT_IDS,
    TOP_K_LOGIT_VALUES,
    VISUAL_ADVANTAGE,
    ExtractionSpec,
    store_schema,
)
from ceed_teacher.router import capture_combine_weights, hybrid_layers, stack_layers
from ceed_teacher.va_opd import degrade_image, gold_logprobs, visual_advantage

from ceed_core import ArtifactRow, ArtifactStore, StoreMetadata, store_root
from ceed_data import CORPUS_FILENAME, ImageStore, read_examples

# The six candidate layers hidden states are cached at: spread across the
# Teacher's 30 and including all three of base.yaml's CEA layers, so B3 and the
# layer-mapping study can both proceed before the layer sweep concludes.
DEFAULT_HIDDEN_STATE_LAYERS = (4, 9, 14, 19, 24, 29)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the command line."""
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--corpus", type=Path, default=Path("data/corpus"))
    parser.add_argument("--store", type=Path, default=Path("data/store"))
    parser.add_argument("--configs", type=Path, default=Path("configs"))
    parser.add_argument("--group", default="b2", help="the Group whose fingerprint names the store")
    parser.add_argument("--split", default="train")
    parser.add_argument("--top-k", type=int, default=64)
    parser.add_argument("--limit", type=int, default=None, help="cap on examples, for smoke runs")
    parser.add_argument("--batch-size", type=int, default=16, help="rows per Parquet shard")
    parser.add_argument("--worker-id", default="w0")
    parser.add_argument("--n-workers", type=int, default=1)
    parser.add_argument("--worker-index", type=int, default=0)
    parser.add_argument(
        "--max-new-tokens", type=int, default=32, help="cap for the correctness pass"
    )
    parser.add_argument(
        "--device-map",
        default="auto",
        help="how to shard the Teacher across GPUs (the 26B needs several)",
    )
    parser.add_argument(
        "--skip-correctness",
        action="store_true",
        help="skip the free-generation pass (much faster; the flag is stored False)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="cache every artefact kind, so one pass feeds B2 through B5",
    )
    parser.add_argument(
        "--with-hidden-states", action="store_true", help="cache hidden states (B3)"
    )
    parser.add_argument(
        "--with-combine-weights", action="store_true", help="cache effective combine weights (B5)"
    )
    parser.add_argument(
        "--with-visual-advantage",
        action="store_true",
        help="cache per-token visual advantage, doubling the forward cost (B4)",
    )
    parser.add_argument(
        "--hidden-state-layers",
        type=int,
        nargs="+",
        default=list(DEFAULT_HIDDEN_STATE_LAYERS),
        help="teacher layers to cache hidden states at",
    )
    parser.add_argument(
        "--combine-weight-layers",
        type=int,
        nargs="+",
        default=None,
        help="teacher layers to cache combine weights at (default: every layer)",
    )
    args = parser.parse_args(argv)
    if args.all:
        args.with_hidden_states = True
        args.with_combine_weights = True
        args.with_visual_advantage = True
    return args


def resolve_config(configs_dir: Path, group: str) -> GroupConfig:
    """Resolve a Group configuration from the repository's overlay stack."""
    layers = [
        load_overlay(configs_dir / "base.yaml"),
        load_overlay(configs_dir / "student.yaml"),
        load_overlay(configs_dir / "groups" / f"{group}.yaml"),
    ]
    return GroupConfig.model_validate(merge_overlays(layers))


def load_teacher(model_id: str, device_map: str, dtype: str) -> tuple[Any, Any]:
    """Load the Teacher sharded across the available GPUs."""
    from transformers import AutoModelForImageTextToText, AutoProcessor

    model = AutoModelForImageTextToText.from_pretrained(
        model_id, dtype=getattr(torch, dtype), device_map=device_map
    )
    model.eval()
    return model, AutoProcessor.from_pretrained(model_id)


def build_inputs(
    processor: Any,
    example: Any,
    image_store: ImageStore,
    transform: Callable[[Any], Any] | None = None,
) -> tuple[dict, int, Any]:
    """Teacher-force the gold answer: prompt + answer, with the prompt length.

    The prompt is built by the *same* builder the Student's dataloader and the
    evaluator use, so the Teacher is measured on the question the Student is
    trained and scored on. A prompt that drifted between the two sides would
    misattribute the difference to distillation.

    Args:
        processor: The Teacher's processor.
        example: The example to encode.
        image_store: Where the example's image bytes live.
        transform: An optional edit applied to the image before it is prompted
            with — the degradation of the visual-advantage pass.

    Returns:
        The Teacher inputs, the prompt length in tokens, and the gold answer ids.
    """
    from ceed_student.dataset import chat_messages
    from PIL import Image

    image = Image.open(io.BytesIO(image_store.get(example.image_fingerprint))).convert("RGB")
    if transform is not None:
        image = transform(image)
    prompt = processor.apply_chat_template(
        chat_messages(example, image),
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )
    prompt_ids = prompt["input_ids"][0]
    answer_ids = torch.tensor(
        processor.tokenizer(example.answers[0], add_special_tokens=False)["input_ids"],
        dtype=prompt_ids.dtype,
    )
    inputs = dict(prompt)
    inputs["input_ids"] = torch.cat([prompt_ids, answer_ids]).unsqueeze(0)
    if "attention_mask" in inputs:
        inputs["attention_mask"] = torch.ones_like(inputs["input_ids"])
    return inputs, int(prompt_ids.shape[0]), answer_ids


def to_device(inputs: dict, device: Any) -> dict:
    """Move every tensor in a processor's output onto the model's device."""
    return {k: (v.to(device) if hasattr(v, "to") else v) for k, v in inputs.items()}


def answer_slice(prompt_length: int, n_answer: int) -> slice:
    """Return the measurement positions of the answer tokens.

    Answer token ``i`` sits at ``prompt_length + i`` and is predicted at the
    position before it, so the first answer token is measured at the prompt's
    final position.
    """
    return slice(prompt_length - 1, prompt_length - 1 + n_answer)


def degraded_gold_logprobs(
    model: Any,
    processor: Any,
    example: Any,
    image_store: ImageStore,
    prompt_length: int,
    answer_ids: Any,
) -> torch.Tensor:
    """Return the Teacher's gold-token log-probabilities on the degraded image.

    The degraded image keeps its dimensions, so the Teacher emits the same number
    of image tokens and the two passes align token for token. That is checked
    rather than assumed: a mismatch would compare different positions and yield a
    plausible, wrong advantage.

    Args:
        model: The loaded Teacher.
        processor: Its processor.
        example: The example being measured.
        image_store: Where the example's image bytes live.
        prompt_length: The prompt length of the *original* pass.
        answer_ids: The gold answer token ids.

    Returns:
        One log-probability per gold answer token.

    Raises:
        ValueError: If degrading the image changed the prompt's token count.
    """
    inputs, degraded_prompt_length, _ = build_inputs(
        processor, example, image_store, transform=degrade_image
    )
    if degraded_prompt_length != prompt_length:
        raise ValueError(
            f"degrading {example.example_id}'s image changed the prompt from "
            f"{prompt_length} to {degraded_prompt_length} tokens; the two passes "
            "would not align token for token"
        )
    with torch.no_grad():
        logits = model(**to_device(inputs, model.device)).logits[0].float()
    positions = answer_slice(prompt_length, int(answer_ids.shape[0]))
    return gold_logprobs(logits[positions], answer_ids.to(logits.device))


def extract_example(
    model: Any,
    processor: Any,
    example: Any,
    image_store: ImageStore,
    spec: ExtractionSpec,
    n_experts: int,
    max_new_tokens: int,
    skip_correctness: bool,
) -> list[ArtifactRow]:
    """Return one artefact row per answer token of this example.

    What is cached is entirely ``spec``: a kind whose layer tuple is empty is not
    computed, so a store built for the logit-KD Groups pays nothing for B3's
    hidden states or B4's second forward.
    """
    inputs, prompt_length, answer_ids = build_inputs(processor, example, image_store)
    n_answer = int(answer_ids.shape[0])
    if n_answer == 0:
        return []

    on_device = to_device(inputs, model.device)
    capture: Any = (
        capture_combine_weights(model, n_experts) if spec.combine_weight_layers else nullcontext({})
    )
    with capture as captured, torch.no_grad():
        outputs = model(**on_device, output_hidden_states=bool(spec.hidden_state_layers))
    logits = outputs.logits[0].float()

    advantage = None
    if spec.visual_advantage:
        original = gold_logprobs(
            logits[answer_slice(prompt_length, n_answer)], answer_ids.to(logits.device)
        )
        degraded = degraded_gold_logprobs(
            model, processor, example, image_store, prompt_length, answer_ids
        )
        advantage = visual_advantage(original, degraded.to(original.device)).cpu()

    rows: list[ArtifactRow] = []
    for ordinal in range(n_answer):
        measurement = prompt_length - 1 + ordinal
        top = torch.topk(logits[measurement], spec.top_k)
        vectors: dict[str, np.ndarray] = {
            TOP_K_LOGIT_IDS: top.indices.cpu().numpy().astype(np.int32),
            TOP_K_LOGIT_VALUES: top.values.cpu().numpy().astype(np.float32),
        }
        if spec.hidden_state_layers:
            # hidden_states[0] is the embedding output, so layer L is index L + 1.
            # Each layer's state is moved to the CPU before stacking: a Teacher
            # sharded across GPUs returns them on the device its layer sits on.
            stacked = torch.stack(
                [
                    outputs.hidden_states[layer + 1][0][measurement].float().cpu()
                    for layer in spec.hidden_state_layers
                ]
            )
            vectors[HIDDEN_STATES] = stacked.numpy().astype(np.float16)
        if spec.combine_weight_layers:
            weights = stack_layers(captured, spec.combine_weight_layers, measurement)
            vectors[COMBINE_WEIGHTS] = weights.numpy().astype(np.float16)
        if advantage is not None:
            vectors[VISUAL_ADVANTAGE] = advantage[ordinal].reshape(1).numpy().astype(np.float32)
        rows.append(
            ArtifactRow(
                example_id=example.example_id,
                answer_token_index=ordinal,
                gold_token_id=int(answer_ids[ordinal]),
                correct=False,  # filled by the correctness pass below
                vectors=vectors,
            )
        )

    correct = False
    if not skip_correctness:
        prompt_only = {
            k: (v[:, :prompt_length] if k in {"input_ids", "attention_mask"} else v)
            for k, v in on_device.items()
        }
        with torch.no_grad():
            generated = model.generate(
                **prompt_only, max_new_tokens=max_new_tokens, do_sample=False, num_beams=1
            )
        answer = processor.decode(generated[0][prompt_length:], skip_special_tokens=True).strip()
        correct = answer.lower() == example.answers[0].strip().lower()

    from dataclasses import replace

    return [replace(row, correct=correct) for row in rows]


def build_spec(args: argparse.Namespace, n_hybrid_layers: int) -> ExtractionSpec:
    """Turn the command line into the one description of what this run caches.

    The ``--with-*`` flags exist because they read well on a command line; from
    here on there is a single :class:`~ceed_teacher.extraction.ExtractionSpec`,
    and both the store's schema and the per-example extraction are derived from
    it — so the columns declared and the columns written cannot disagree.
    """
    return ExtractionSpec(
        top_k=args.top_k,
        combine_weight_layers=(
            tuple(
                args.combine_weight_layers
                if args.combine_weight_layers is not None
                else range(n_hybrid_layers)
            )
            if args.with_combine_weights
            else ()
        ),
        hidden_state_layers=tuple(args.hidden_state_layers) if args.with_hidden_states else (),
        visual_advantage=args.with_visual_advantage,
    )


def main(argv: list[str] | None = None) -> int:
    """Extract the Teacher's answer-token artefacts into the artifact store."""
    args = parse_args(argv)
    config = resolve_config(args.configs, args.group)

    from ceed_core import extraction_fingerprint

    fingerprint = extraction_fingerprint(config)
    manifest = CorpusManifest.read(args.corpus)
    wanted = list(manifest.splits[args.split])
    wanted = wanted[args.worker_index :: args.n_workers]
    if args.limit is not None:
        wanted = wanted[: args.limit]
    selected = set(wanted)

    image_store = ImageStore(args.corpus / "images")
    examples = [e for e in read_examples(args.corpus / CORPUS_FILENAME) if e.example_id in selected]
    order = {eid: i for i, eid in enumerate(wanted)}
    examples.sort(key=lambda e: order[e.example_id])

    model, processor = load_teacher(
        config.extraction.teacher_model, args.device_map, config.student.dtype
    )
    text_config = model.config.text_config
    n_experts = int(getattr(text_config, "num_experts", 0))
    spec = build_spec(args, len(hybrid_layers(model)))

    metadata = StoreMetadata(
        extraction_fingerprint=fingerprint,
        vector_kinds=store_schema(spec, n_experts, int(text_config.hidden_size)),
        corpus_manifest=manifest.model_dump(mode="json"),
    )
    print(f"[extract] caching {sorted(metadata.vector_kinds)}")
    store = ArtifactStore.create(store_root(args.store, fingerprint), metadata)
    done = store.example_ids()
    todo = [e for e in examples if e.example_id not in done]
    print(f"[extract] store {store.root}")
    print(f"[extract] {len(todo)} to do, {len(examples) - len(todo)} already present")
    if not todo:
        return 0

    writer = store.writer(args.worker_id)
    batch: list[ArtifactRow] = []
    for index, example in enumerate(todo, start=1):
        batch.extend(
            extract_example(
                model,
                processor,
                example,
                image_store,
                spec,
                n_experts,
                args.max_new_tokens,
                args.skip_correctness,
            )
        )
        if len(batch) >= args.batch_size:
            writer.write(batch)
            batch = []
        if index % 10 == 0 or index == len(todo):
            print(f"[extract] {index}/{len(todo)} examples", flush=True)
    if batch:
        writer.write(batch)
    print(f"[extract] done; store holds {len(store.example_ids())} examples")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
