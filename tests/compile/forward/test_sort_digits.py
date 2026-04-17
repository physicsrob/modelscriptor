"""End-to-end compile tests for the sort_digits example variants.

Each variant compiles its ``create_network_parts`` through the forward
compiler, runs autoregressive decoding on a battery of inputs, and
asserts the decoded output matches the ascending sort.

V4 is the primary variant and supports duplicates. V1 only supports
distinct-digit inputs. V2 / V3 support duplicates (but aren't built
yet in this file; they'll be added once those example files exist).
"""

import torch

from torchwright.compiler.forward.compile import forward_compile
from torchwright.compiler.transformer import HeadlessTransformer
from torchwright.graph import Embedding, Node

D_HEAD = 32


def _decode_token(embedding: Embedding, vector: torch.Tensor) -> str:
    dists = torch.cdist(vector.unsqueeze(0).cpu(), embedding.table)
    return embedding.tokenizer.decode_id(int(dists.argmin().item()))


def _run_autoregressive(
    net: HeadlessTransformer,
    output_node: Node,
    embedding: Embedding,
    input_tokens: list,
    max_new_tokens: int,
) -> str:
    """Autoregressively decode ``max_new_tokens`` tokens after the input.

    The sort examples emit their sorted output starting at the trigger
    position (the ``"\\n"``), so ``max_new_tokens`` is the number of
    emitted tokens we read from positions *at and after* the trigger.
    """
    tokens = list(input_tokens)
    emitted = []
    for _ in range(max_new_tokens):
        result = net.compute(
            n_pos=len(tokens), input_values={"embedding_input": tokens}
        )
        next_token = _decode_token(embedding, result[output_node][-1])
        if next_token == "<eos>":
            break
        emitted.append(next_token)
        tokens.append(next_token)
    return "".join(emitted)


def _build_v4():
    from examples.sort_digits_v4 import create_network_parts, D_MODEL

    output_node, pos_encoding, embedding = create_network_parts()
    net = forward_compile(
        d=D_MODEL,
        d_head=D_HEAD,
        output_node=output_node,
        pos_encoding=pos_encoding,
        verbose=False,
    )
    return net, output_node, embedding


def _build_v1():
    from examples.sort_digits_v1 import create_network_parts, D_MODEL

    output_node, pos_encoding, embedding = create_network_parts()
    net = forward_compile(
        d=D_MODEL,
        d_head=D_HEAD,
        output_node=output_node,
        pos_encoding=pos_encoding,
        verbose=False,
    )
    return net, output_node, embedding


def _build_v2():
    from examples.sort_digits_v2 import create_network_parts, D_MODEL

    output_node, pos_encoding, embedding = create_network_parts()
    net = forward_compile(
        d=D_MODEL,
        d_head=D_HEAD,
        output_node=output_node,
        pos_encoding=pos_encoding,
        verbose=False,
        max_layers=200,  # V2's MLP chain is deep (~20 layers per slot).
    )
    return net, output_node, embedding


# Cases shared by all variants that support only distinct digits.
_DISTINCT_CASES = [
    ("9583", "3589"),
    ("1", "1"),
    ("5432", "2345"),
    ("1234", "1234"),
    ("9876543210", "0123456789"),
]

# Additional cases with duplicates, for variants that handle them.
_DUPLICATE_CASES = [
    ("1111", "1111"),
    ("1121", "1112"),
    ("3131", "1133"),
    ("2211", "1122"),
    ("223331", "122333"),
]


def _check_case(net, output_node, embedding, input_str: str, expected: str):
    tokens = ["<bos>"] + list(input_str) + ["\n"]
    decoded = _run_autoregressive(
        net, output_node, embedding, tokens, max_new_tokens=len(input_str)
    )
    assert (
        decoded == expected
    ), f"input={input_str!r} expected={expected!r} got={decoded!r}"


# ---------------------------------------------------------------------------
# V4 — primary variant, handles duplicates.
# ---------------------------------------------------------------------------


def test_sort_digits_v4_distinct_battery():
    net, output_node, embedding = _build_v4()
    for inp, expected in _DISTINCT_CASES:
        _check_case(net, output_node, embedding, inp, expected)


def test_sort_digits_v4_duplicate_battery():
    net, output_node, embedding = _build_v4()
    for inp, expected in _DUPLICATE_CASES:
        _check_case(net, output_node, embedding, inp, expected)


# ---------------------------------------------------------------------------
# V1 — distinct digits only. Does not support duplicates by design.
# ---------------------------------------------------------------------------


def test_sort_digits_v1_distinct_battery():
    net, output_node, embedding = _build_v1()
    for inp, expected in _DISTINCT_CASES:
        _check_case(net, output_node, embedding, inp, expected)


# ---------------------------------------------------------------------------
# V2 — rank-lookup (MLP selection brain, attention is a lookup).
# ---------------------------------------------------------------------------


_V2_DISTINCT_CASES = [
    ("9583", "3589"),
    ("1", "1"),
    ("5432", "2345"),
    ("1234", "1234"),
]

_V2_DUPLICATE_CASES = [
    ("1111", "1111"),
    ("1121", "1112"),
    ("3131", "1133"),
    ("2211", "1122"),
]


def test_sort_digits_v2_distinct_battery():
    net, output_node, embedding = _build_v2()
    for inp, expected in _V2_DISTINCT_CASES:
        _check_case(net, output_node, embedding, inp, expected)


def test_sort_digits_v2_duplicate_battery():
    net, output_node, embedding = _build_v2()
    for inp, expected in _V2_DUPLICATE_CASES:
        _check_case(net, output_node, embedding, inp, expected)
