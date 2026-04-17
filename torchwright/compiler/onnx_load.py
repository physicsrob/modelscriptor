"""Runtime loader for headless ONNX models.

Provides ``OnnxHeadlessModule`` — an ``onnxruntime``-backed callable
that speaks the KV-cache prefill/decode protocol produced by
:func:`torchwright.compiler.export.compile_headless_to_onnx` — plus a
:class:`HeadlessRuntime` :class:`typing.Protocol` that describes the
shared interface with the in-memory
:class:`torchwright.compiler.export.CompiledHeadless`.

Two usage shapes:

- **Independent per-query** (e.g. per-frame DOOM rendering):
  ``module(inputs)`` runs one prefill call and returns outputs. Each
  call is stateless — the KV cache built during the call is discarded.

- **Autoregressive decode**: ``past = module.empty_past()`` for the
  initial state, then ``outputs, past = module.step(inputs, past)``
  repeatedly to extend the cached context one chunk at a time.

``.eval()`` returns ``self`` for PyTorch drop-in symmetry.
"""

import json
import os
from typing import List, Optional, Protocol, Tuple

import numpy as np
import torch

from torchwright.compiler.export import HEADLESS_META_FORMAT, meta_path_for

# Past state is a pair (past_K_tuple, past_V_tuple) where each inner
# tuple has one torch.Tensor of shape (n_heads, n_past, d_head) per
# layer.  Both CompiledHeadless and OnnxHeadlessModule use this exact
# representation so past tensors can be threaded between runtimes.
PastKV = Tuple[Tuple[torch.Tensor, ...], Tuple[torch.Tensor, ...]]


class HeadlessRuntime(Protocol):
    """Structural type for any headless runtime (in-memory or ONNX).

    Lets callers type-hint "either :class:`CompiledHeadless` or
    :class:`OnnxHeadlessModule`" without importing both — a function
    that renders a frame or runs a decode step only needs to know that
    it has ``input_names``, ``metadata``, and the three callables below.
    """

    input_names: List[str]
    metadata: dict

    def __call__(self, inputs: torch.Tensor) -> torch.Tensor: ...

    def step(
        self,
        inputs: torch.Tensor,
        past: PastKV,
        past_len: Optional[int] = None,
    ) -> Tuple[torch.Tensor, PastKV]: ...

    def empty_past(self) -> PastKV: ...

    def eval(self) -> "HeadlessRuntime": ...


class OnnxHeadlessModule:
    """Loads a headless cached ONNX model and exposes it as a callable.

    Args:
        onnx_path: Path to the ``.onnx`` file.  A sidecar
            ``<stem>.meta.json`` with format ``torchwright.headless.v1``
            must exist alongside it.
        providers: ``onnxruntime`` execution providers list.  Defaults
            to CPU.
    """

    def __init__(self, onnx_path: str, providers=None) -> None:
        import onnxruntime as ort

        meta_path = meta_path_for(onnx_path)
        if not os.path.exists(meta_path):
            raise FileNotFoundError(
                f"Missing sidecar {meta_path}. Re-export with "
                f"compile_headless_to_onnx to produce it."
            )
        with open(meta_path) as f:
            meta = json.load(f)

        fmt = meta.get("format")
        if fmt != HEADLESS_META_FORMAT:
            raise ValueError(
                f"{meta_path}: unexpected format {fmt!r}, "
                f"expected {HEADLESS_META_FORMAT!r}"
            )
        self.input_names: List[str] = list(meta["input_names"])
        self.metadata: dict = dict(meta.get("extra") or {})

        self._session = ort.InferenceSession(
            onnx_path,
            providers=providers or ["CPUExecutionProvider"],
        )

        # Discover KV cache topology from the ONNX graph's input spec.
        # past_K_i inputs have shape (n_heads_i, n_past, d_head); after
        # head trimming each layer may have a different head count.
        inputs = {inp.name: inp for inp in self._session.get_inputs()}
        self._n_layers = sum(1 for name in inputs if name.startswith("past_K_"))
        assert (
            self._n_layers > 0
        ), f"{onnx_path}: no past_K_* inputs — is this a cached-protocol model?"
        self._per_layer_n_heads = [
            int(inputs[f"past_K_{i}"].shape[0]) for i in range(self._n_layers)
        ]
        self._d_head = int(inputs["past_K_0"].shape[2])

        # Cache the list of output names in the protocol order so we
        # can unpack session.run() results without another dict lookup.
        self._out_names = ["outputs"]
        for i in range(self._n_layers):
            self._out_names += [f"new_K_{i}", f"new_V_{i}"]

    def empty_past(self) -> PastKV:
        """Zero-length past tensors suitable for a first prefill call."""
        past_K = tuple(
            torch.zeros(nh, 0, self._d_head) for nh in self._per_layer_n_heads
        )
        past_V = tuple(
            torch.zeros(nh, 0, self._d_head) for nh in self._per_layer_n_heads
        )
        return (past_K, past_V)

    def step(
        self,
        inputs: torch.Tensor,
        past: PastKV,
        past_len: Optional[int] = None,
    ) -> Tuple[torch.Tensor, PastKV]:
        """Run one cached-protocol call and return (outputs, new_past).

        Args:
            inputs: ``(n_new, d_input)`` float tensor.
            past: ``(past_K_tuple, past_V_tuple)`` from a prior step or
                :meth:`empty_past`.  Each tuple has length ``n_layers``
                and each entry is a ``(n_heads, n_past, d_head)`` torch
                tensor.
            past_len: Optional absolute query position for the new rows.
                When ``None`` (default), derived from
                ``past_K[0].shape[1]``.  Callers using a sliding-window
                runtime may pass the true global position here while
                handing over a trimmed cache; the graph's pos-encoding
                slice uses this value while the attention mask uses the
                cache's actual shape.

        Returns:
            ``(outputs, new_past)`` where ``outputs`` is a
            ``(n_new, d_output)`` torch tensor and ``new_past`` matches
            the shape of ``past`` with the new rows appended.
        """
        past_K, past_V = past
        assert len(past_K) == self._n_layers
        assert len(past_V) == self._n_layers

        inputs_np = inputs.detach().cpu().numpy().astype(np.float32, copy=False)
        if past_len is None:
            past_len = int(past_K[0].shape[1])

        feeds: dict = {
            "inputs": inputs_np,
            "past_len": np.array(past_len, dtype=np.int64),
        }
        for i in range(self._n_layers):
            feeds[f"past_K_{i}"] = (
                past_K[i].detach().cpu().numpy().astype(np.float32, copy=False)
            )
            feeds[f"past_V_{i}"] = (
                past_V[i].detach().cpu().numpy().astype(np.float32, copy=False)
            )

        results = self._session.run(self._out_names, feeds)
        outputs = torch.from_numpy(results[0])

        new_K = tuple(
            torch.from_numpy(results[1 + 2 * i]) for i in range(self._n_layers)
        )
        new_V = tuple(
            torch.from_numpy(results[1 + 2 * i + 1]) for i in range(self._n_layers)
        )

        return outputs, (new_K, new_V)

    def __call__(self, inputs: torch.Tensor) -> torch.Tensor:
        """Convenience: stateless prefill that discards the cache.

        Equivalent to ``self.step(inputs, self.empty_past())[0]``. Use
        this for independent per-query inference (e.g. per-frame DOOM
        rendering) where no state is carried between calls.
        """
        outputs, _ = self.step(inputs, self.empty_past())
        return outputs

    def eval(self) -> "OnnxHeadlessModule":
        return self
