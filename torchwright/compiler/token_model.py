"""Token-I/O wrapper over :class:`CompiledHeadless`.

A vanilla autoregressive transformer's runtime looks like::

    id_out = argmax(W_embed.T @ transformer(W_embed[id_in] + bypasses))

``CompiledHeadless`` handles the middle — the residual-stream forward
pass, KV-cached `.step()`, raw `(inputs, outputs)` flat-tensor I/O.
It is deliberately headless: no embedding, no LM head.

``CompiledToken`` is the surrounding two slices.  It wraps a headless
module together with the graph-level :class:`Embedding` node and:

* accepts integer ``token_ids`` plus a bypass-field dict on input,
* packs them into the raw column layout ``CompiledHeadless`` expects,
* invokes ``CompiledHeadless.step`` (KV-cached forward),
* pulls out the designated ``logit_output_name`` slice
  (``(n_new, d_embed)``),
* computes logits via dot product with the embedding table
  (``W_embed.T``) and argmaxes to emit ``next_token_ids``,
* returns every other declared output field as a name → tensor
  dict, alongside the new KV cache.

The factory :func:`compile_token` mirrors
``torchwright.compiler.export.compile_headless``'s signature and
tacks on the embedding + logit slot at the end.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch

from torchwright.compiler.export import CompiledHeadless, compile_headless
from torchwright.graph import Node
from torchwright.graph.embedding import Embedding
from torchwright.graph.pos_encoding import PosEncoding


class CompiledToken:
    """Token-I/O wrapper over :class:`CompiledHeadless`.

    API mirrors a real LLM step: integer IDs in, integer IDs out, plus
    any configured bypass fields.  See module docstring for the
    full rationale.
    """

    def __init__(
        self,
        headless: CompiledHeadless,
        embedding: Embedding,
        *,
        token_id_input_name: str = "token_ids",
        logit_output_name: str = "next_token_embedding",
    ) -> None:
        self._headless = headless
        self._embedding = embedding
        self._token_id_input_name = token_id_input_name
        self._logit_output_name = logit_output_name

        input_specs = headless._input_specs
        output_specs = headless._output_specs

        input_names = {n for n, _, _ in input_specs}
        output_names = {n for n, _, _ in output_specs}
        assert token_id_input_name in input_names, (
            f"CompiledToken expected input slot {token_id_input_name!r}; "
            f"available: {sorted(input_names)}"
        )
        assert logit_output_name in output_names, (
            f"CompiledToken expected output slot {logit_output_name!r}; "
            f"available: {sorted(output_names)}"
        )
        for n, _, w in input_specs:
            if n == token_id_input_name:
                assert w == 1, f"{token_id_input_name!r} slot must be 1 wide (got {w})"
        for n, _, w in output_specs:
            if n == logit_output_name:
                assert w == embedding.d_embed, (
                    f"{logit_output_name!r} slot width {w} must match "
                    f"embedding.d_embed={embedding.d_embed}"
                )

        d_input = max((s + w) for _, s, w in input_specs)
        self._d_input = d_input
        self._input_specs = list(input_specs)
        self._output_specs = list(output_specs)
        self._input_by_name = {n: (s, w) for n, s, w in input_specs}
        self._output_by_name = {n: (s, w) for n, s, w in output_specs}

    # ---- pass-through surface ------------------------------------------

    @property
    def metadata(self) -> dict:
        return self._headless.metadata

    def empty_past(self) -> tuple:
        return self._headless.empty_past()

    def eval(self) -> "CompiledToken":
        self._headless.eval()
        return self

    @property
    def device(self) -> torch.device:
        return self._headless._net.device

    @property
    def embedding(self) -> Embedding:
        return self._embedding

    @property
    def input_names(self) -> List[str]:
        return [n for n, _, _ in self._input_specs]

    @property
    def output_names(self) -> List[str]:
        return [n for n, _, _ in self._output_specs]

    # ---- core step ------------------------------------------------------

    def step(
        self,
        inputs: Dict[str, torch.Tensor],
        past: tuple,
        past_len: Optional[int] = None,
        debug: bool = False,
        debug_atol: float = 1e-7,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], tuple]:
        """Run one KV-cached autoregressive step.

        Args:
            inputs: Field-name → tensor.  Must include ``token_ids``
                (``(n,)`` or ``(n, 1)`` integer) — the IDs the host is
                feeding this step.  Any other declared input slot not
                present in the dict is zeroed.
            past: KV cache from :meth:`empty_past` or a prior call.
            past_len: Absolute query position for the new rows.  See
                :meth:`CompiledHeadless.step`.
            debug / debug_atol: Pass-through to
                :meth:`CompiledHeadless.step` for assert/watch checks.

        Returns:
            ``(next_token_ids, overflow, new_past)`` where

            * ``next_token_ids`` is ``(n,)`` int64 — the argmax of
              ``out_emb @ W_embed.T`` at each position.
            * ``overflow`` is a dict of all other declared output
              fields, each a ``(n, width)`` tensor at its declared
              column slot.
            * ``new_past`` is the updated KV cache.
        """
        if self._token_id_input_name not in inputs:
            raise KeyError(
                f"CompiledToken.step requires {self._token_id_input_name!r} "
                f"in inputs dict"
            )
        token_ids = inputs[self._token_id_input_name]
        if token_ids.ndim == 1:
            n = int(token_ids.shape[0])
            ids_col = token_ids.view(n, 1)
        elif token_ids.ndim == 2:
            assert (
                token_ids.shape[1] == 1
            ), f"token_ids must be (n,) or (n, 1); got {tuple(token_ids.shape)}"
            n = int(token_ids.shape[0])
            ids_col = token_ids
        else:
            raise AssertionError(
                f"token_ids must be 1-D or 2-D; got {token_ids.ndim}-D"
            )

        device = self.device
        flat = torch.zeros((n, self._d_input), dtype=torch.float32, device=device)

        # Token IDs occupy a 1-wide slot — written as float32 (the host
        # machinery routes them through float residual columns but
        # Embedding.compute casts back to long internally).
        tid_start, _ = self._input_by_name[self._token_id_input_name]
        flat[:, tid_start : tid_start + 1] = ids_col.to(
            dtype=torch.float32, device=device
        )

        for name, tensor in inputs.items():
            if name == self._token_id_input_name:
                continue
            if name not in self._input_by_name:
                raise KeyError(
                    f"CompiledToken.step: unknown input field {name!r}; "
                    f"known: {sorted(self._input_by_name)}"
                )
            start, width = self._input_by_name[name]
            v = tensor
            if v.ndim == 1:
                v = v.unsqueeze(0).expand(n, -1)
            assert v.shape == (n, width), (
                f"input {name!r} expected shape ({n}, {width}); got "
                f"{tuple(v.shape)}"
            )
            flat[:, start : start + width] = v.to(dtype=torch.float32, device=device)

        outputs, new_past = self._headless.step(
            flat, past, past_len=past_len, debug=debug, debug_atol=debug_atol
        )

        # Argmax the logit-source slice against W_embed.T.
        logit_start, logit_width = self._output_by_name[self._logit_output_name]
        out_emb = outputs[:, logit_start : logit_start + logit_width]
        table = self._embedding.table.to(out_emb.device)
        logits = out_emb @ table.T  # (n, V)
        next_token_ids = logits.argmax(dim=-1).to(dtype=torch.int64)

        # Build overflow dict of every OTHER output field.
        overflow: Dict[str, torch.Tensor] = {}
        for name, start, width in self._output_specs:
            if name == self._logit_output_name:
                continue
            overflow[name] = outputs[:, start : start + width]

        return next_token_ids, overflow, new_past


def compile_token(
    pos_encoding: PosEncoding,
    embedding: Embedding,
    *,
    io: Dict[str, Tuple[Optional[Node], Optional[Node]]],
    d: int,
    d_head: int = 16,
    max_layers: int = 400,
    device: str = "auto",
    verbose: bool = True,
    extra_metadata: Optional[dict] = None,
    d_hidden: Optional[int] = None,
    token_id_input_name: str = "token_ids",
    logit_output_name: str = "next_token_embedding",
) -> CompiledToken:
    """Compile a token-I/O graph and wrap it in a :class:`CompiledToken`.

    Same shape as :func:`torchwright.compiler.export.compile_headless`
    with two extra arguments:

    * ``embedding`` — the graph's :class:`Embedding` leaf.
    * ``token_id_input_name`` / ``logit_output_name`` — slot names in
      the ``io`` dict that mark the token-ID input and the d_embed-wide
      output the wrapper should argmax.

    The ``io`` dict must declare both slots; every other entry is a
    bypass field and is preserved verbatim as a headless I/O slot.
    """
    headless = compile_headless(
        pos_encoding,
        io=io,
        d=d,
        d_head=d_head,
        max_layers=max_layers,
        device=device,
        verbose=verbose,
        extra_metadata=extra_metadata,
        d_hidden=d_hidden,
    )
    return CompiledToken(
        headless,
        embedding,
        token_id_input_name=token_id_input_name,
        logit_output_name=logit_output_name,
    )
