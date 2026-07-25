#!/usr/bin/env python
"""Cache the Teacher's top-k answer-token logits — the artefacts B2 distils from.

B2's backbone needs one thing from the Teacher: its top-k next-token logits at
each gold answer token. That comes from an ordinary teacher-forced forward pass
and needs none of the MoE hooking (router internals, expert ablation, effective
combine weights) that the Causal Expert Attribution work requires. This script
therefore unblocks the primary baseline today, and is deliberately *not* the full
CEA extraction — that lands with the hooked teacher.

Following ADR-0002 the gold answer is teacher-forced with the thinking gate off,
so an answer token is always a gold token and the Teacher's positions align 1:1
with the Student's cross-entropy targets. A separate greedy free-generation pass
records one correctness flag per example.

**Keying.** Rows are keyed by the answer token's *ordinal* within the gold answer,
not its absolute sequence position. The Teacher and Student share a tokenizer but
emit different numbers of image placeholder tokens, so an absolute position would
mean different things on the two sides; an ordinal means the same thing on both.
The dataloader reads the same ordinal.

Example:
    uv run python scripts/extract_teacher_logits.py \
        --corpus data/corpus --store data/store --split train --limit 50
"""

from __future__ import annotations

import argparse
import io
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

from ceed_core import ArtifactRow, ArtifactStore, StoreMetadata, VectorSpec, store_root
from ceed_data import CORPUS_FILENAME, ImageStore, read_examples

TOP_K_LOGIT_IDS = "top_k_logit_ids"
TOP_K_LOGIT_VALUES = "top_k_logit_values"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the command line."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
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
    return parser.parse_args(argv)


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


def build_inputs(processor: Any, example: Any, image_store: ImageStore) -> tuple[dict, int, Any]:
    """Teacher-force the gold answer: prompt + answer, with the prompt length.

    The prompt is built by the *same* builder the Student's dataloader and the
    evaluator use, so the Teacher is measured on the question the Student is
    trained and scored on. A prompt that drifted between the two sides would
    misattribute the difference to distillation.
    """
    from ceed_student.dataset import chat_messages
    from PIL import Image

    image = Image.open(io.BytesIO(image_store.get(example.image_fingerprint))).convert("RGB")
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


def extract_example(
    model: Any,
    processor: Any,
    example: Any,
    image_store: ImageStore,
    top_k: int,
    max_new_tokens: int,
    skip_correctness: bool,
) -> list[ArtifactRow]:
    """Return one artefact row per answer token of this example."""
    inputs, prompt_length, answer_ids = build_inputs(processor, example, image_store)
    n_answer = int(answer_ids.shape[0])
    if n_answer == 0:
        return []

    device = model.device
    on_device = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in inputs.items()}
    with torch.no_grad():
        logits = model(**on_device).logits[0].float()

    # Answer token with ordinal i sits at sequence position prompt_length + i and
    # is predicted at the position before it.
    rows: list[ArtifactRow] = []
    for ordinal in range(n_answer):
        measurement = prompt_length - 1 + ordinal
        top = torch.topk(logits[measurement], top_k)
        rows.append(
            ArtifactRow(
                example_id=example.example_id,
                answer_token_index=ordinal,
                gold_token_id=int(answer_ids[ordinal]),
                correct=False,  # filled by the correctness pass below
                vectors={
                    TOP_K_LOGIT_IDS: top.indices.cpu().numpy().astype(np.int32),
                    TOP_K_LOGIT_VALUES: top.values.cpu().numpy().astype(np.float32),
                },
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


def main(argv: list[str] | None = None) -> int:
    """Extract the Teacher's top-k answer-token logits into the artifact store."""
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

    metadata = StoreMetadata(
        extraction_fingerprint=fingerprint,
        vector_kinds={
            TOP_K_LOGIT_IDS: VectorSpec(dtype="int32", shape=(args.top_k,)),
            TOP_K_LOGIT_VALUES: VectorSpec(dtype="float32", shape=(args.top_k,)),
        },
        corpus_manifest=manifest.model_dump(mode="json"),
    )
    store = ArtifactStore.create(store_root(args.store, fingerprint), metadata)
    done = store.example_ids()
    todo = [e for e in examples if e.example_id not in done]
    print(f"[extract] store {store.root}")
    print(f"[extract] {len(todo)} to do, {len(examples) - len(todo)} already present")
    if not todo:
        return 0

    model, processor = load_teacher(
        config.extraction.teacher_model, args.device_map, config.student.dtype
    )
    writer = store.writer(args.worker_id)
    batch: list[ArtifactRow] = []
    for index, example in enumerate(todo, start=1):
        batch.extend(
            extract_example(
                model,
                processor,
                example,
                image_store,
                args.top_k,
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
