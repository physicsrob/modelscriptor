"""Runtime loader for headless ONNX models.

Provides a thin ``OnnxHeadlessModule`` that wraps an ``onnxruntime``
session behind the same ``module(inputs) -> outputs`` callable interface
as ``HeadlessTransformerModule``, so it can be dropped directly into
code like ``step_frame_compiled`` / ``render_frame_compiled`` without
any other changes.

The goal is "build once, run anywhere": produce the ``.onnx`` with
``torchwright.doom.to_onnx`` and then play or walkthrough from the saved
model, skipping the multi-second compile on every launch.
"""

import json
import os
from typing import List

import numpy as np
import torch

from torchwright.compiler.export import HEADLESS_META_FORMAT, _meta_path_for


class OnnxHeadlessModule:
    """onnxruntime-backed drop-in for ``HeadlessTransformerModule``.

    Loads a ``.onnx`` produced by one of the feed-forward headless
    exporters (:func:`emit_headless_onnx` or
    :func:`compile_and_emit_onnx`) plus its ``<stem>.meta.json`` sidecar
    and exposes a ``__call__`` that accepts a ``(seq_len, d_input)``
    ``torch.Tensor`` and returns a ``(seq_len, d_output)`` ``torch.Tensor``.

    KV-cached models produced by :func:`compile_headless_to_onnx` are
    rejected with a clear error — this loader only speaks the plain
    feed-forward protocol.
    """

    def __init__(self, onnx_path: str, providers=None) -> None:
        import onnxruntime as ort

        meta_path = _meta_path_for(onnx_path)
        if not os.path.exists(meta_path):
            raise FileNotFoundError(
                f"Missing sidecar {meta_path}. Re-export with a current "
                f"torchwright exporter to produce it."
            )
        with open(meta_path) as f:
            meta = json.load(f)

        fmt = meta.get("format")
        if fmt != HEADLESS_META_FORMAT:
            raise ValueError(
                f"{meta_path}: unexpected format {fmt!r}, "
                f"expected {HEADLESS_META_FORMAT!r}"
            )
        if meta.get("cached", False):
            raise NotImplementedError(
                f"{onnx_path} is a KV-cached headless model; "
                f"OnnxHeadlessModule only supports the feed-forward protocol. "
                f"Run inference via onnxruntime directly with the "
                f"prefill/decode inputs (past_K_i, past_V_i, past_len)."
            )
        self.input_names: List[str] = meta["input_names"]

        self._session = ort.InferenceSession(
            onnx_path,
            providers=providers or ["CPUExecutionProvider"],
        )
        self._input_name = self._session.get_inputs()[0].name
        self._output_name = self._session.get_outputs()[0].name

    def __call__(self, inputs: torch.Tensor) -> torch.Tensor:
        arr = inputs.detach().cpu().numpy().astype(np.float32, copy=False)
        out = self._session.run([self._output_name], {self._input_name: arr})[0]
        return torch.from_numpy(out)

    def eval(self) -> "OnnxHeadlessModule":
        return self
