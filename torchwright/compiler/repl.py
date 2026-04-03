"""Interactive REPL that runs inference on an ONNX model.

Standalone -- does not depend on any torchwright graph code.
Requires: onnxruntime, numpy.
"""

import json
import os
from typing import List

import numpy as np


class _Vocab:
    """Minimal tokenizer reconstructed from a vocab list."""

    def __init__(self, vocab: List[str]):
        self.vocab = vocab
        self._token_to_id = {t: i for i, t in enumerate(vocab)}

    def token_to_id(self, token: str) -> int:
        return self._token_to_id.get(token, 0)  # 0 = <unk>

    def id_to_token(self, token_id: int) -> str:
        if 0 <= token_id < len(self.vocab):
            return self.vocab[token_id]
        return self.vocab[0]


def generate(
    session,
    vocab: _Vocab,
    input_text: str,
    max_new_tokens: int = 10,
    bos_token: str = "<bos",
    eos_token: str = "<eos>",
) -> str:
    """Run autoregressive generation on an ONNX model.

    Args:
        session: An onnxruntime.InferenceSession.
        vocab: Vocabulary for token ID conversion.
        input_text: The input string (e.g. "12+34=").
        max_new_tokens: Maximum tokens to generate.
        bos_token: Beginning-of-sequence token.
        eos_token: End-of-sequence token.

    Returns:
        The generated output string (excluding input and special tokens).
    """
    tokens = [bos_token] + list(input_text)
    token_ids = np.array([vocab.token_to_id(t) for t in tokens], dtype=np.int64)

    for _ in range(max_new_tokens):
        logits = session.run(None, {"token_ids": token_ids})[0]
        next_id = int(logits[-1].argmax())
        next_token = vocab.id_to_token(next_id)
        if next_token == eos_token:
            break
        token_ids = np.append(token_ids, next_id)

    return "".join(vocab.id_to_token(int(tid)) for tid in token_ids[len(tokens) :])


def run_repl(
    onnx_path: str,
    max_new_tokens: int = 20,
) -> None:
    """Load an ONNX model and run an interactive REPL.

    Expects a vocab file at <onnx_path_without_ext>.vocab.json.

    Args:
        onnx_path: Path to the .onnx model file.
        max_new_tokens: Maximum tokens to generate per query.
    """
    import onnxruntime  # type: ignore[import-untyped]

    base, _ = os.path.splitext(onnx_path)
    vocab_path = base + ".vocab.json"

    with open(vocab_path) as f:
        vocab_data = json.load(f)
    vocab = _Vocab(vocab_data["vocab"])

    session = onnxruntime.InferenceSession(onnx_path)

    print(f"Loaded {onnx_path} ({len(vocab.vocab)} tokens). Type 'q' to quit.")
    while True:
        text = input("> ")
        if text.lower() == "q":
            print("Bye")
            break
        result = generate(session, vocab, text, max_new_tokens)
        print(result)
