"""Runtime loader for headless ONNX models produced by ``emit_headless_onnx``.

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
from typing import List

import numpy as np
import torch


class OnnxHeadlessModule:
    """onnxruntime-backed drop-in for ``HeadlessTransformerModule``.

    Loads a ``.onnx`` produced by :func:`emit_headless_onnx` plus its
    ``<path>.input_names.json`` sidecar and exposes a ``__call__`` that
    accepts a ``(seq_len, d_input)`` ``torch.Tensor`` and returns a
    ``(seq_len, d_output)`` ``torch.Tensor``.
    """

    def __init__(self, onnx_path: str, providers=None) -> None:
        import onnxruntime as ort

        self._session = ort.InferenceSession(
            onnx_path,
            providers=providers or ["CPUExecutionProvider"],
        )
        self._input_name = self._session.get_inputs()[0].name
        self._output_name = self._session.get_outputs()[0].name

        with open(onnx_path + ".input_names.json") as f:
            self.input_names: List[str] = json.load(f)["input_names"]

    def __call__(self, inputs: torch.Tensor) -> torch.Tensor:
        arr = inputs.detach().cpu().numpy().astype(np.float32, copy=False)
        out = self._session.run([self._output_name], {self._input_name: arr})[0]
        return torch.from_numpy(out)

    def eval(self) -> "OnnxHeadlessModule":
        return self
