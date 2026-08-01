"""Reading the real Teacher's effective combine weights off its routers.

This is deliberately *not* the CEA ablation engine. Causal Expert Attribution
needs the hooked forward that can replace one expert's contribution and re-run
the tail; the effective combine weight needs only to watch what the router
already computes, so it is — as the plan says of B5 — a trivial byproduct of the
same teacher-forced forward that produces B2's logits. That is why B5 is
deliverable with the baselines and the CEA groups are not.

**What is captured** (plan amendment A3). Gemma-4's router returns the softmax
probabilities, the top-k weights *after* normalisation and after multiplication
by ``per_expert_scale``, and the expert indices. The third of those is the
number that actually multiplies an expert's output in the residual stream, so it
is the effective combine weight and it is what is scattered into a dense
per-expert vector here. The raw logits and the unscaled softmax weights are not
monotone transforms of it — ``per_expert_scale`` reorders experts — so caching
either of the others would answer a different question from the one Phase 0.1
asks.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import torch
from jaxtyping import Float
from torch import Tensor


def hybrid_layers(model: Any) -> list[Any]:
    """Return the Teacher's hybrid decoder layers, in depth order.

    Found by structure rather than by module path: a hybrid layer is one holding
    both a router and an expert bank. That survives the layer stack moving inside
    the model (as it does between a text model and its multimodal wrapper).

    Args:
        model: The loaded Teacher.

    Returns:
        The hybrid layers, ordered by their position in the stack.

    Raises:
        ValueError: If the model has no hybrid layers, which means it is not the
            sparse Teacher this extraction is defined against.
    """
    found = [
        module
        for _, module in model.named_modules()
        if hasattr(module, "router") and hasattr(module, "experts")
    ]
    if not found:
        raise ValueError(
            "this model has no hybrid layers (no module carrying both a router and "
            "experts); effective combine weights are only defined for the sparse Teacher"
        )
    return sorted(found, key=lambda layer: getattr(layer, "layer_idx", 0))


def _layer_index(layer: Any, depth: int) -> int:
    """Return a layer's own index in the Teacher's stack, or its depth if it has none."""
    return int(getattr(layer, "layer_idx", depth))


@contextmanager
def capture_combine_weights(model: Any, n_experts: int) -> Iterator[dict[int, Tensor]]:
    """Capture every layer's effective combine weights for the next forward.

    Yields a dictionary that the forward fills: **teacher layer index** to a
    dense ``[positions, experts]`` tensor in which the routed experts carry their
    effective combine weight and every other expert carries zero. The key is the
    layer's own ``layer_idx``, not its position among the hybrid layers — those
    coincide for a Teacher whose every layer is hybrid, and diverge silently for
    one whose layers are not, which is the sort of thing that reads the wrong
    layer without ever raising. The hooks are removed on exit, so the Teacher is
    left exactly as it was found — it is also used for the ordinary logit pass.

    Args:
        model: The loaded Teacher.
        n_experts: The number of experts per layer, used to size the dense
            vector the sparse top-k is scattered into.

    Yields:
        The per-layer capture, empty until a forward has run.
    """
    captured: dict[int, Tensor] = {}
    handles = []

    def hook_for(index: int):
        def hook(_module: Any, _inputs: Any, output: Any) -> None:
            # Gemma-4's router returns (probabilities, top_k_weights, top_k_index);
            # the weights are already normalised and scaled by per_expert_scale.
            _, top_k_weights, top_k_index = output
            weights = top_k_weights.detach().float()
            index_ = top_k_index.detach().long()
            dense = torch.zeros(
                weights.shape[0], n_experts, dtype=weights.dtype, device=weights.device
            )
            # Brought to the CPU as it is captured: the Teacher is sharded across
            # GPUs, so layer 0's weights and layer 29's live on different devices
            # and could not otherwise be stacked. They are small, and this keeps
            # the capture off the cards that are holding the model.
            captured[index] = dense.scatter(1, index_, weights).cpu()

        return hook

    for depth, layer in enumerate(hybrid_layers(model)):
        handles.append(layer.router.register_forward_hook(hook_for(_layer_index(layer, depth))))
    try:
        yield captured
    finally:
        for handle in handles:
            handle.remove()


def stack_layers(
    captured: dict[int, Tensor], layers: tuple[int, ...], position: int
) -> Float[Tensor, "layers experts"]:
    """Stack one position's combine weights across the requested layers.

    Args:
        captured: What :func:`capture_combine_weights` collected.
        layers: The teacher layers to stack, in the order the store records.
        position: The sequence position to read.

    Returns:
        The combine weights, ``[len(layers), experts]``.

    Raises:
        KeyError: If a requested layer was never captured.
    """
    missing = [layer for layer in layers if layer not in captured]
    if missing:
        raise KeyError(f"no combine weights captured for layer(s) {missing}")
    return torch.stack([captured[layer][position] for layer in layers])
