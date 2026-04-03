"""Digit-by-digit arithmetic in embedding space via lookup tables.

These functions implement school-math algorithms — carry-propagation addition,
borrowing subtraction, lexicographic comparison, and long multiplication —
entirely in embedding space. Every digit pair is enumerated in a map_to_table
lookup, making the approach exhaustive but exact.

This is the embedding-space counterpart to scalar arithmetic via
scalar_encoding. Use embedding-space when the number of possible values is
small (e.g., single digits 0-9), or when you want to avoid the thermometer
threshold count that scalar-space requires. Use scalar-space (scalar_encoding
+ arithmetic_ops) when operating on numbers as wholes is simpler than
digit-by-digit propagation.

All sequences are MSB-first: seq[0] is the most significant digit.
"""

from typing import Tuple, List

import torch

from torchwright.graph import Node, Embedding
from torchwright.ops.arithmetic_ops import concat, compare, sum_nodes
from torchwright.ops.inout_nodes import create_constant
from torchwright.ops.map_select import map_to_table, select


# ---------------------------------------------------------------------------
# Addition
# ---------------------------------------------------------------------------


def sum_digits(
    embedding: Embedding, num1: Node, num2: Node, carry_in: Node
) -> Tuple[Node, Node]:
    """Add two single-digit embeddings plus a carry bit.

    Args:
        embedding: The embedding table (must contain "0"-"9").
        num1: Embedding-valued node for the first digit.
        num2: Embedding-valued node for the second digit.
        carry_in: Boolean node (1.0 = carry, -1.0 = no carry).

    Returns:
        (result_digit, carry_out): result_digit is an embedding-valued node,
        carry_out is a boolean node.
    """
    sum_out_table = {}
    carry_out_table = {}

    for A in range(10):
        for B in range(10):
            for C in [0, 1]:
                entry_key = torch.cat(
                    [
                        embedding.get_embedding(str(A)),
                        embedding.get_embedding(str(B)),
                        torch.tensor([1.0 if C else -1.0]),
                    ]
                )
                sum_out_table[entry_key] = embedding.get_embedding(
                    str((A + B + C) % 10)
                )
                carry_out_table[entry_key] = torch.tensor(
                    [1.0 if (A + B + C) >= 10 else -1.0]
                )

    key = concat([num1, num2, carry_in])

    return (
        map_to_table(
            inp=key,
            key_to_value=sum_out_table,
            default=embedding.get_embedding("0"),
        ),
        map_to_table(key, key_to_value=carry_out_table, default=torch.tensor([-1.0])),
    )


def sum_digit_seqs(
    embedding: Embedding, seq1: List[Node], seq2: List[Node]
) -> List[Node]:
    """Add two digit sequences with carry propagation, right-to-left.

    Sequences are MSB-first: seq[0] is the most significant digit,
    seq[-1] is the least significant.
    """
    carry = create_constant(torch.tensor([-1.0]))
    out = []
    for digit1, digit2 in reversed(list(zip(seq1, seq2))):
        sum, carry = sum_digits(embedding, digit1, digit2, carry)
        out.append(sum)

    return list(reversed(out))


# ---------------------------------------------------------------------------
# Subtraction
# ---------------------------------------------------------------------------


def subtract_digits(
    embedding: Embedding, num1: Node, num2: Node, borrow_in: Node
) -> Tuple[Node, Node]:
    """Subtract num2 + borrow from num1. Returns (result_digit, borrow_out).

    borrow_in: 1.0 = borrow, -1.0 = no borrow (same convention as carry).
    """
    result_table = {}
    borrow_table = {}

    for A in range(10):
        for B in range(10):
            for borrow in [0, 1]:
                entry_key = torch.cat(
                    [
                        embedding.get_embedding(str(A)),
                        embedding.get_embedding(str(B)),
                        torch.tensor([1.0 if borrow else -1.0]),
                    ]
                )
                diff = A - B - borrow
                result_table[entry_key] = embedding.get_embedding(str(diff % 10))
                borrow_table[entry_key] = torch.tensor([1.0 if diff < 0 else -1.0])

    key = concat([num1, num2, borrow_in])
    return (
        map_to_table(key, result_table, default=embedding.get_embedding("0")),
        map_to_table(key, borrow_table, default=torch.tensor([-1.0])),
    )


def subtract_digit_seqs(
    embedding: Embedding, seq1: List[Node], seq2: List[Node]
) -> List[Node]:
    """Subtract seq2 from seq1 digit by digit with borrow propagation.

    Assumes seq1 >= seq2 (no sign handling). Processes right-to-left.
    """
    borrow = create_constant(torch.tensor([-1.0]))
    out = []
    for d1, d2 in reversed(list(zip(seq1, seq2))):
        result, borrow = subtract_digits(embedding, d1, d2, borrow)
        out.append(result)
    return list(reversed(out))


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------


def compare_digit_pair(embedding: Embedding, a: Node, b: Node) -> Node:
    """Compare two single-digit embeddings.

    Returns 1.0 if a > b, -1.0 if a < b, 0.0 if equal.
    """
    table = {}
    for i in range(10):
        for j in range(10):
            key = torch.cat(
                [
                    embedding.get_embedding(str(i)),
                    embedding.get_embedding(str(j)),
                ]
            )
            if i > j:
                table[key] = torch.tensor([1.0])
            elif i < j:
                table[key] = torch.tensor([-1.0])
            else:
                table[key] = torch.tensor([0.0])
    return map_to_table(concat([a, b]), table, default=torch.tensor([0.0]))


def compare_digit_seqs(
    embedding: Embedding, seq1: List[Node], seq2: List[Node]
) -> Node:
    """Lexicographic comparison of two digit sequences.

    Returns 1.0 if seq1 >= seq2, -1.0 if seq1 < seq2.

    Folds from MSB to LSB: the first non-equal digit determines the result.
    """
    # result_so_far: 1.0 (a>b), -1.0 (a<b), 0.0 (equal so far)
    # combine(prev, current): if prev != 0 keep prev, else take current
    combine_table = {}
    for prev_val in [-1.0, 0.0, 1.0]:
        for cur_val in [-1.0, 0.0, 1.0]:
            key = torch.tensor([prev_val, cur_val])
            result = prev_val if prev_val != 0.0 else cur_val
            combine_table[key] = torch.tensor([result])

    result = compare_digit_pair(embedding, seq1[0], seq2[0])
    for a_i, b_i in zip(seq1[1:], seq2[1:]):
        cmp_i = compare_digit_pair(embedding, a_i, b_i)
        result = map_to_table(
            concat([result, cmp_i]),
            combine_table,
            default=torch.tensor([0.0]),
        )

    # Collapse to boolean: >= 0 means a >= b (treat equal as positive)
    return compare(result, thresh=-0.5, true_level=1.0, false_level=-1.0)


# ---------------------------------------------------------------------------
# Multiplication
# ---------------------------------------------------------------------------


def multiply_digit_pair(embedding: Embedding, a: Node, b: Node) -> Tuple[Node, Node]:
    """Multiply two single-digit embeddings.

    Returns (tens_digit, ones_digit) as embedding-valued nodes.
    """
    tens_table = {}
    ones_table = {}
    for i in range(10):
        for j in range(10):
            key = torch.cat(
                [
                    embedding.get_embedding(str(i)),
                    embedding.get_embedding(str(j)),
                ]
            )
            product = i * j
            tens_table[key] = embedding.get_embedding(str(product // 10))
            ones_table[key] = embedding.get_embedding(str(product % 10))
    inp = concat([a, b])
    return (
        map_to_table(inp, tens_table, default=embedding.get_embedding("0")),
        map_to_table(inp, ones_table, default=embedding.get_embedding("0")),
    )


def multiply_digit_seqs(
    embedding: Embedding, seq1: List[Node], seq2: List[Node]
) -> List[Node]:
    """Long multiplication in embedding space: seq1 * seq2.

    Returns up to 2*n digit embeddings (MSB-first). Uses partial product rows
    added pairwise with sum_digit_seqs, mirroring pencil-and-paper long
    multiplication.
    """
    n = len(seq1)
    zero = create_constant(embedding.get_embedding("0"))

    # Step 1: Compute all digit*digit products (parallel in the graph)
    products = {}
    for i in range(n):
        for j in range(n):
            products[i, j] = multiply_digit_pair(embedding, seq1[i], seq2[j])

    # Step 2: Build partial product rows.
    # Row j = seq1 * seq2[j], shifted left by (n-1-j) positions.
    rows = []
    for j in range(n):
        carry = create_constant(torch.tensor([-1.0]))
        row_digits = []
        for i in reversed(range(n)):
            tens, ones = products[i, j]
            digit_sum, carry = sum_digits(
                embedding, ones, carry_digit if i < n - 1 else zero, carry
            )
            row_digits.append(digit_sum)
            carry_digit = tens
        # Final carry + last tens digit
        final_digit, _ = sum_digits(embedding, tens, zero, carry)
        row_digits.append(final_digit)
        row_digits = list(reversed(row_digits))

        # Pad with trailing zeros for shift (row j shifted by n-1-j)
        shift = n - 1 - j
        row_padded = row_digits + [zero] * shift
        # Pad with leading zeros to get 2*n digits total
        while len(row_padded) < 2 * n:
            row_padded = [zero] + row_padded
        rows.append(row_padded)

    # Step 3: Sum rows pairwise
    result = rows[0]
    for row in rows[1:]:
        result = sum_digit_seqs(embedding, result, row)

    return result
