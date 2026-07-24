"""A tiny synthetic teacher with the real teacher's hybrid structure.

The real teacher is 49 GB and needs a GPU, so no meaningful test could run
against it on every change. This fixture has the *same structure* — hybrid
layers in which a shared dense path runs in parallel with a top-k expert mixture,
and a router carrying both an input scale and a per-expert scale — at a size that
runs on CPU in milliseconds. It exists so the ablation engine's numerical
invariants are checked on every test run.

It is a genuine causal transformer, not a mock: attention is causal, so the claim
that ablating one answer token leaves earlier tokens bit-identical is a real
property of this model rather than an asserted one.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from jaxtyping import Float, Int
from torch import Tensor


@dataclass(frozen=True)
class SyntheticConfig:
    """Shape of the synthetic teacher.

    The defaults mirror the real teacher's structure at toy scale: several hybrid
    layers, many experts, a small top-k, and a shared dense path in every layer.

    Attributes:
        n_layers: Number of hybrid decoder layers.
        n_experts: Routed experts per layer.
        top_k: Experts activated per token.
        hidden_size: Residual-stream width.
        shared_intermediate: Hidden width of the shared dense MLP.
        expert_intermediate: Hidden width of each expert MLP.
        vocab_size: Output vocabulary size.
        seed: Seed for deterministic weight initialisation.
    """

    n_layers: int = 2
    n_experts: int = 8
    top_k: int = 2
    hidden_size: int = 64
    shared_intermediate: int = 32
    expert_intermediate: int = 16
    vocab_size: int = 32
    seed: int = 0


def _linear(gen: torch.Generator, out_features: int, in_features: int) -> Tensor:
    return torch.randn(out_features, in_features, generator=gen) / (in_features**0.5)


class SyntheticHybridTeacher:
    """A CPU-sized causal transformer of hybrid layers, implementing HybridForward.

    Call :meth:`run` once with an input sequence to populate the forward state,
    then query the hybrid-layer internals or :meth:`resume` the forward under
    overridden feed-forward outputs. All state belongs to the most recent
    :meth:`run`.
    """

    def __init__(self, config: SyntheticConfig | None = None) -> None:
        """Initialise deterministic weights from ``config`` (defaults if omitted)."""
        self.config = config or SyntheticConfig()
        c = self.config
        self.n_layers = c.n_layers
        self.n_experts = c.n_experts
        self.top_k = c.top_k
        self.hidden_size = c.hidden_size

        gen = torch.Generator().manual_seed(c.seed)
        h = c.hidden_size
        self._embed = _linear(gen, c.vocab_size, h)  # [vocab, hidden]
        self._wq = [_linear(gen, h, h) for _ in range(c.n_layers)]
        self._wk = [_linear(gen, h, h) for _ in range(c.n_layers)]
        self._wv = [_linear(gen, h, h) for _ in range(c.n_layers)]
        self._wo = [_linear(gen, h, h) for _ in range(c.n_layers)]
        # Shared dense path: h -> shared_intermediate -> h, one per layer.
        self._shared_up = [_linear(gen, c.shared_intermediate, h) for _ in range(c.n_layers)]
        self._shared_down = [_linear(gen, h, c.shared_intermediate) for _ in range(c.n_layers)]
        # Experts: h -> expert_intermediate -> h, per expert per layer.
        self._expert_up = [
            [_linear(gen, c.expert_intermediate, h) for _ in range(c.n_experts)]
            for _ in range(c.n_layers)
        ]
        self._expert_down = [
            [_linear(gen, h, c.expert_intermediate) for _ in range(c.n_experts)]
            for _ in range(c.n_layers)
        ]
        self._router = [_linear(gen, c.n_experts, h) for _ in range(c.n_layers)]
        self._input_scale = 1.7
        # per-expert scale reorders experts, so it is not a monotone rescale.
        self._per_expert_scale = [
            torch.linspace(0.5, 1.5, c.n_experts)[torch.randperm(c.n_experts, generator=gen)]
            for _ in range(c.n_layers)
        ]
        self._head = _linear(gen, c.vocab_size, h)  # [vocab, hidden]

        self._input_ids: Tensor | None = None
        self._layer_input: list[Tensor] = []  # x entering each layer, [seq, hidden]
        self._post_attn: list[Tensor] = []  # h after attention, pre-FFN, [seq, hidden]
        self._shared: list[Tensor] = []  # [seq, hidden]
        self._expert_out: list[Tensor] = []  # [seq, experts, hidden]
        self._eff_weights: list[Tensor] = []  # [seq, experts]
        self._activated: list[Tensor] = []  # [seq, top_k]
        self._ffn: list[Tensor] = []  # [seq, hidden]

    # -- forward -------------------------------------------------------------

    def _attention(self, layer: int, x: Tensor) -> Tensor:
        """Causal single-head attention over ``x`` of shape ``[..., seq, hidden]``."""
        q = x @ self._wq[layer].T
        k = x @ self._wk[layer].T
        v = x @ self._wv[layer].T
        scores = q @ k.transpose(-2, -1) / (self.hidden_size**0.5)
        seq = x.shape[-2]
        mask = torch.triu(torch.ones(seq, seq, dtype=torch.bool), diagonal=1)
        scores = scores.masked_fill(mask, float("-inf"))
        attn = torch.softmax(scores, dim=-1) @ v
        return attn @ self._wo[layer].T

    def _hybrid_ffn(self, layer: int, n: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Compute the hybrid FFN for a ``[seq, hidden]`` normalised input.

        Returns the shared output, all experts' outputs, the effective combine
        weights, the activated expert indices, and the combined FFN output.
        """
        shared = torch.tanh(n @ self._shared_up[layer].T) @ self._shared_down[layer].T
        expert_outs = torch.stack(
            [
                torch.tanh(n @ self._expert_up[layer][e].T) @ self._expert_down[layer][e].T
                for e in range(self.n_experts)
            ],
            dim=-2,
        )  # [..., experts, hidden]
        logits = (n @ self._router[layer].T) * self._input_scale  # [..., experts]
        eff_w = torch.softmax(logits, dim=-1) * self._per_expert_scale[layer]  # [..., experts]
        activated = torch.topk(eff_w, self.top_k, dim=-1).indices  # [..., top_k]
        gate = torch.zeros_like(eff_w).scatter(-1, activated, eff_w.gather(-1, activated))
        routed = (gate.unsqueeze(-1) * expert_outs).sum(dim=-2)  # [..., hidden]
        ffn = shared + routed
        return shared, expert_outs, eff_w, activated, ffn

    def run(self, input_ids: Int[Tensor, " seq"]) -> None:
        """Run the teacher-forced forward and cache every hybrid-layer internal.

        Args:
            input_ids: The token ids of one example, prompt and gold answer
                concatenated.
        """
        self._input_ids = input_ids
        self._layer_input = []
        self._post_attn = []
        self._shared = []
        self._expert_out = []
        self._eff_weights = []
        self._activated = []
        self._ffn = []

        x = self._embed[input_ids]  # [seq, hidden]
        for layer in range(self.n_layers):
            self._layer_input.append(x)
            h = x + self._attention(layer, x)
            self._post_attn.append(h)
            shared, expert_outs, eff_w, activated, ffn = self._hybrid_ffn(layer, h)
            self._shared.append(shared)
            self._expert_out.append(expert_outs)
            self._eff_weights.append(eff_w)
            self._activated.append(activated)
            self._ffn.append(ffn)
            x = h + ffn

    # -- HybridForward queries ------------------------------------------------

    def effective_combine_weights(self, layer: int, token: int) -> Float[Tensor, " experts"]:
        """Return every expert's effective combine weight for this token."""
        return self._eff_weights[layer][token]

    def activated_experts(self, layer: int, token: int) -> Int[Tensor, " top_k"]:
        """Return the activated expert indices for this token."""
        return self._activated[layer][token]

    def expert_outputs(self, layer: int, token: int) -> Float[Tensor, "experts hidden"]:
        """Return every expert's output vector for this token."""
        return self._expert_out[layer][token]

    def shared_dense_output(self, layer: int, token: int) -> Float[Tensor, " hidden"]:
        """Return the shared dense path's output for this token."""
        return self._shared[layer][token]

    def ffn_output(self, layer: int, token: int) -> Float[Tensor, " hidden"]:
        """Return the combined feed-forward output for this token."""
        return self._ffn[layer][token]

    def gold_token_id(self, token: int) -> int:
        """Return the next token id, which this position predicts under teacher forcing."""
        assert self._input_ids is not None, "run() must be called before querying"
        return int(self._input_ids[token + 1])

    def _forward_tail(self, x: Tensor, start: int) -> Tensor:
        """Run layers ``start..`` on ``x`` of shape ``[..., seq, hidden]`` to logits."""
        for layer in range(start, self.n_layers):
            h = x + self._attention(layer, x)
            _, _, _, _, ffn = self._hybrid_ffn(layer, h)
            x = h + ffn
        return x @ self._head.T

    def resume_full(
        self, layer: int, token: int, ffn_outputs: Float[Tensor, "variants hidden"]
    ) -> Float[Tensor, "variants seq vocab"]:
        """Resume the forward under overridden FFN outputs, returning all positions.

        The token's layer-``layer`` output is rebuilt as its post-attention state
        plus each supplied feed-forward variant; the rest of the sequence keeps
        its baseline layer output. The tail layers then run once over the whole
        batch. All positions are returned so that the causal-isolation invariant
        — earlier tokens unchanged — is observable without reaching into
        internals.
        """
        assert self._input_ids is not None, "run() must be called before resuming"
        variants = ffn_outputs.shape[0]
        baseline_out = self._post_attn[layer] + self._ffn[layer]  # [seq, hidden]
        batch = baseline_out.unsqueeze(0).repeat(variants, 1, 1)  # [variants, seq, hidden]
        batch[:, token, :] = self._post_attn[layer][token] + ffn_outputs
        return self._forward_tail(batch, start=layer + 1)  # [variants, seq, vocab]

    def resume(
        self, layer: int, token: int, ffn_outputs: Float[Tensor, "variants hidden"]
    ) -> Float[Tensor, "variants vocab"]:
        """Resume the forward from ``layer`` for ``token``, returning the token's logits.

        This is :meth:`resume_full` sliced to the attributed position, and is the
        method the ablation engine uses through the :class:`HybridForward`
        protocol.
        """
        return self.resume_full(layer, token, ffn_outputs)[:, token, :]
