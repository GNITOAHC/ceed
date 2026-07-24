"""What the CEA ablation engine needs from a model, and nothing more.

The ablation engine is the highest-risk logic in the repository, so it is written
against a narrow protocol rather than against the real teacher. Any model that
can expose its hybrid-layer internals for one answer token — the activated
experts, the effective combine weights, the per-expert outputs, the shared dense
path's output, and the combined feed-forward output — and can resume its forward
pass from a hybrid layer with that token's feed-forward output overridden, can be
attributed. The tiny synthetic fixture and the real teacher are two
implementations of this one protocol.

All tensors are indexed by a single ``(layer, token)`` pair, where ``token`` is
the position of an answer token under teacher forcing. Shapes are named with
jaxtyping and, under pytest, enforced at runtime.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from jaxtyping import Float, Int
from torch import Tensor


@runtime_checkable
class HybridForward(Protocol):
    """A completed teacher-forced forward exposing hybrid-layer internals.

    Implementations hold the state of one forward pass over one example and
    answer the queries below for any hybrid ``layer`` and answer-``token``
    position. The engine treats them as read-only apart from :meth:`resume`,
    which is a pure function of its ``ffn_outputs`` argument.

    Attributes:
        n_layers: The number of hybrid layers.
        n_experts: The number of routed experts per layer.
        top_k: The number of experts the router activates per token.
        hidden_size: The residual-stream width.
    """

    n_layers: int
    n_experts: int
    top_k: int
    hidden_size: int

    def effective_combine_weights(self, layer: int, token: int) -> Float[Tensor, " experts"]:
        """Return the effective combine weight of every expert for this token.

        The effective combine weight is the softmax router weight multiplied by
        the per-expert scale — the quantity that actually multiplies an expert's
        output — defined for all experts so that near-miss experts can be ranked
        by it (see docs/adr/0003 and CONTEXT.md).
        """
        ...

    def activated_experts(self, layer: int, token: int) -> Int[Tensor, " top_k"]:
        """Return the indices of the experts the router activated for this token."""
        ...

    def expert_outputs(self, layer: int, token: int) -> Float[Tensor, "experts hidden"]:
        """Return every expert's output vector for this token."""
        ...

    def shared_dense_output(self, layer: int, token: int) -> Float[Tensor, " hidden"]:
        """Return the shared dense path's output for this token.

        This path fires for every token regardless of routing and is never an
        ablation target.
        """
        ...

    def ffn_output(self, layer: int, token: int) -> Float[Tensor, " hidden"]:
        """Return the combined feed-forward output for this token.

        This is the shared dense output plus the combine-weighted sum of the
        activated experts' outputs — the quantity added back to the residual
        stream.
        """
        ...

    def gold_token_id(self, token: int) -> int:
        """Return the vocabulary id whose log-probability is attributed at this token."""
        ...

    def resume(
        self, layer: int, token: int, ffn_outputs: Float[Tensor, "variants hidden"]
    ) -> Float[Tensor, "variants vocab"]:
        """Resume the forward from ``layer`` for ``token`` under overridden FFN outputs.

        For each variant feed-forward output supplied, the forward is completed
        from ``layer`` onward with this token's layer-``layer`` feed-forward
        output replaced by that variant, and the resulting logits at ``token``
        are returned. Every variant is evaluated in a single batched pass.

        Because attention is causal, overriding this token's feed-forward output
        can only affect tokens at or after it; tokens before it are unchanged,
        which is the cached-key/value assumption the cost model depends on.

        Args:
            layer: The hybrid layer to resume from.
            token: The answer-token position whose feed-forward output is
                overridden.
            ffn_outputs: One replacement feed-forward output per variant.

        Returns:
            The logits at ``token`` for each variant.
        """
        ...
