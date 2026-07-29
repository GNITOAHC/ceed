#!/usr/bin/env python
r"""Fold a Group's LoRA adapter into the base weights, producing a standalone model.

A LoRA run leaves a ~9 MB adapter that only means anything applied to its base
Student. This folds ``W' = W + (alpha/r) * BA`` into the weights themselves and
writes an ordinary Hugging Face checkpoint — same architecture, same config, same
processor as the base model, different weight values. The result is a drop-in
replacement for ``google/gemma-4-e4b-it``: uploadable to the Hub, loadable with
``AutoModelForImageTextToText.from_pretrained``, and servable by any backend that
can serve the base model.

Two details make this correct rather than merely plausible:

* **The merge runs in fp32 on CPU.** Folding ``BA`` into ``W`` at fp16 accumulates
  rounding error across 132 projections; the weights are cast to the serving
  dtype only once, on save.
* **The adapter is applied through the same wrapper it was saved from.** The
  trainer saved it from inside a :class:`~ceed_student.student.CeedStudent`, so
  its recorded module paths carry that prefix; the merged *inner* model is what
  gets written.

Merging does not change what the run was: a merged LoRA model is still a LoRA
result (ADR-0005). The source run's identity is written to the output directory
so that provenance travels with the weights.

Example:
    uv run python scripts/merge_adapter.py --run runs/<config_hash> \
        --output merged/ceed-b1 --verify --corpus data/corpus
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from ceed_core import DecodingConfig, RunRecord

PROVENANCE_FILENAME = "ceed_provenance.json"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the command line."""
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--run", type=Path, help="a run directory holding run_record.json")
    source.add_argument("--checkpoint", type=Path, help="a checkpoint directory")
    parser.add_argument("--output", type=Path, required=True, help="where to write the model")
    parser.add_argument(
        "--model",
        default="google/gemma-4-e4b-it",
        help="the base Student the Group trained from",
    )
    parser.add_argument(
        "--save-dtype",
        default="float16",
        help="dtype the merged weights are written in (the merge itself is fp32)",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="reload the merged model and re-score it, to prove the merge preserved behaviour",
    )
    parser.add_argument("--corpus", type=Path, help="corpus to verify against")
    parser.add_argument("--split", default="validation")
    parser.add_argument("--limit", type=int, default=10, help="examples to verify on")
    parser.add_argument("--device", default="cuda", help="device used for verification only")
    parser.add_argument(
        "--overwrite", action="store_true", help="write into a non-empty output directory"
    )
    return parser.parse_args(argv)


def resolve_source(args: argparse.Namespace) -> tuple[Path, RunRecord | None]:
    """Return the checkpoint to merge and the run record behind it, if there is one."""
    if args.checkpoint is not None:
        return args.checkpoint, None
    record = RunRecord.read(args.run)
    if record.checkpoint_dir is None:
        raise SystemExit(
            f"Group {record.group_code} trained nothing, so there is no adapter to merge."
        )
    return Path(record.checkpoint_dir), record


def merge(base_model_id: str, checkpoint: Path, save_dtype: str) -> tuple[Any, Any]:
    """Load the base model, fold the adapter into its weights, and return it.

    The model is loaded in fp32 on CPU so the merge arithmetic is done at full
    precision, then cast to ``save_dtype`` once the adapter has been folded in.

    Args:
        base_model_id: The base Student the Group trained from.
        checkpoint: The Group's checkpoint directory, holding ``adapter/``.
        save_dtype: The dtype the merged weights are cast to.

    Returns:
        The merged model and its processor.

    Raises:
        SystemExit: If the checkpoint holds no adapter.
    """
    import torch
    from peft import PeftModel
    from transformers import AutoModelForImageTextToText, AutoProcessor

    from ceed_student import CeedStudent

    adapter = checkpoint / "adapter"
    if not adapter.is_dir():
        raise SystemExit(
            f"no adapter at {adapter}.\nOnly a LoRA run writes one; a full fine-tune's "
            "weights are already whole and need no merge."
        )

    print(f"[merge] loading {base_model_id} in float32 on CPU ...", flush=True)
    model = AutoModelForImageTextToText.from_pretrained(
        base_model_id, dtype=torch.float32, device_map="cpu"
    )
    processor = AutoProcessor.from_pretrained(base_model_id)

    # The adapter's module paths were recorded through this wrapper, so the trees
    # must match before PEFT can find anything to merge into.
    print(f"[merge] applying adapter from {adapter} ...", flush=True)
    wrapper = CeedStudent(model)
    peft_model = PeftModel.from_pretrained(wrapper, str(adapter), torch_dtype=torch.float32)

    print("[merge] folding LoRA weights into the base weights ...", flush=True)
    merged_wrapper = peft_model.merge_and_unload()  # type: ignore[reportCallIssue]
    merged = merged_wrapper.model  # unwrap CeedStudent: a plain HF model again

    print(f"[merge] casting merged weights to {save_dtype}")
    return merged.to(getattr(torch, save_dtype)), processor


def write_provenance(
    output: Path, record: RunRecord | None, base_model_id: str, checkpoint: Path
) -> None:
    """Record which run these weights came from, beside the weights.

    A merged LoRA checkpoint is indistinguishable from a fully fine-tuned one by
    inspection, and ADR-0005 turns on that distinction. Writing the source run's
    identity next to the weights keeps it attached to them.
    """
    provenance: dict[str, Any] = {
        "produced_by": "scripts/merge_adapter.py",
        "base_model": base_model_id,
        "source_checkpoint": str(checkpoint),
    }
    if record is not None:
        provenance.update(
            {
                "group_code": record.group_code,
                "seed": record.seed,
                "param_efficiency": record.param_efficiency.value,
                "config_hash": record.config_hash,
                "extraction_fingerprint": record.extraction_fingerprint,
                "metrics": record.metrics,
            }
        )
    (output / PROVENANCE_FILENAME).write_text(json.dumps(provenance, indent=2))


def verify(args: argparse.Namespace, expected: dict[str, float] | None) -> int:
    """Reload the merged model as a plain HF checkpoint and re-score it.

    This is the check that the merge preserved behaviour: the model is loaded the
    way any downstream user would load it — no adapter, no wrapper — and scored on
    the same corpus with the same metric.
    """
    if args.corpus is None:
        print("[verify] --verify needs --corpus", file=sys.stderr)
        return 1

    import torch
    from ceed_eval.direct import generate_answer
    from transformers import AutoModelForImageTextToText, AutoProcessor

    from ceed_data import ImageStore
    from ceed_eval import score_answer
    from ceed_student import load_corpus_split

    print(f"\n[verify] reloading {args.output} as a plain checkpoint ...", flush=True)
    model = AutoModelForImageTextToText.from_pretrained(
        args.output, dtype=getattr(torch, args.save_dtype)
    ).to(args.device)
    model.eval()
    processor = AutoProcessor.from_pretrained(args.output)

    decoding = DecodingConfig(max_new_tokens=64)
    image_store = ImageStore(args.corpus / "images")
    examples = load_corpus_split(args.corpus, args.split)[: args.limit]

    scores = []
    for example in examples:
        answer = generate_answer(model, processor, example, image_store, decoding)
        score = score_answer(example.dataset, answer, example.answers)
        scores.append(score)
        print(f"[verify] {'OK  ' if score > 0.5 else 'MISS'} {example.example_id}  {answer!r}")

    if scores:
        mean = sum(scores) / len(scores)
        print(f"\n[verify] merged model scores {mean:.4f} over {len(scores)} examples")
        if expected:
            for dataset, value in expected.items():
                if not dataset.startswith("train.") and not dataset.endswith(".n"):
                    print(f"[verify] the source run recorded {dataset} = {value:.4f}")
            print(
                "[verify] these are comparable only if --split and --limit match the "
                "run's evaluation; a small sample will differ."
            )
    return 0


def main(argv: list[str] | None = None) -> int:
    """Merge a Group's adapter into its base weights and write a standalone model."""
    args = parse_args(argv)
    checkpoint, record = resolve_source(args)

    if args.output.exists() and any(args.output.iterdir()) and not args.overwrite:
        print(
            f"[merge] {args.output} is not empty; pass --overwrite to write into it",
            file=sys.stderr,
        )
        return 1
    args.output.mkdir(parents=True, exist_ok=True)

    if record is not None:
        print(f"[merge] {record.group_code}, trained {record.param_efficiency.value}")

    merged, processor = merge(args.model, checkpoint, args.save_dtype)

    print(f"[merge] writing to {args.output} ...", flush=True)
    merged.save_pretrained(args.output)
    processor.save_pretrained(args.output)
    write_provenance(args.output, record, args.model, checkpoint)

    print(f"[merge] done. {args.output} is a standalone {args.model.split('/')[-1]}-shaped model.")
    print("[merge] load it with AutoModelForImageTextToText.from_pretrained(...), or upload:")
    print(f"[merge]   uv run hf upload <your-repo> {args.output} .")
    if record is not None and record.param_efficiency.value == "lora":
        print(
            "[merge] NOTE: these weights came from a LoRA run. Merging does not make "
            "them a full fine-tune (ADR-0005); see ceed_provenance.json."
        )

    if args.verify:
        return verify(args, record.metrics if record is not None else None)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
