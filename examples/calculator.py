"""Calculator: supports +, -, * on positive integers up to max_digits digits.

V1: Embedding-space implementation. All digit manipulation uses embedding-valued
nodes and map_to_table lookups, consistent with the adder.
"""

from typing import Tuple, List

import torch

from modelscriptor.graph import Node, Embedding, PosEncoding
from modelscriptor.modelscript.arithmetic_ops import concat, sum_nodes
from modelscriptor.modelscript.inout_nodes import (
    create_constant,
    create_embedding,
    create_pos_encoding,
)
from modelscriptor.modelscript.logic_ops import (
    equals_vector,
    cond_gate,
    bool_not,
    bool_all_true,
    bool_any_true,
)
from modelscriptor.modelscript.map_select import map_to_table, select, switch

from examples.adder import (
    check_is_digit,
    sum_digits,
    sum_digit_seqs,
    remove_leading_0s,
    output_sequence,
    NumericSequence,
)

max_digits = 3


# ---------------------------------------------------------------------------
# Subtraction helpers
# ---------------------------------------------------------------------------


def subtract_digits(
    embedding: Embedding, num1: Node, num2: Node, borrow_in: Node
) -> Tuple[Node, Node]:
    """Subtracts num2 + borrow from num1. Returns (result_digit, borrow_out).

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
    """Subtracts seq2 from seq1 digit by digit (assumes seq1 >= seq2)."""
    borrow = create_constant(torch.tensor([-1.0]))
    out = []
    for d1, d2 in reversed(list(zip(seq1, seq2))):
        result, borrow = subtract_digits(embedding, d1, d2, borrow)
        out.append(result)
    return list(reversed(out))


def compare_digit_pair(embedding: Embedding, a: Node, b: Node) -> Node:
    """Returns 1.0 if a > b, -1.0 if a < b, 0.0 if equal."""
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
    """Returns 1.0 if seq1 >= seq2, -1.0 if seq1 < seq2.

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
    from modelscriptor.modelscript.arithmetic_ops import compare

    return compare(result, thresh=-0.5, true_level=1.0, false_level=-1.0)


# ---------------------------------------------------------------------------
# Multiplication helpers
# ---------------------------------------------------------------------------


def multiply_digit_pair(embedding: Embedding, a: Node, b: Node) -> Tuple[Node, Node]:
    """Returns (tens_digit, ones_digit) embeddings for a*b."""
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
    """Long multiplication: seq1 * seq2. Returns up to 2*n digit embeddings.

    Uses partial product rows added pairwise with sum_digit_seqs.
    seq1/seq2 are MSB-first (seq[0] = most significant digit).
    """
    n = len(seq1)
    zero = create_constant(embedding.get_embedding("0"))

    # Step 1: Compute all digit×digit products (parallel in the graph)
    # products[i][j] = (tens, ones) of seq1[i] * seq2[j]
    products = {}
    for i in range(n):
        for j in range(n):
            products[i, j] = multiply_digit_pair(embedding, seq1[i], seq2[j])

    # Step 2: Build partial product rows.
    # Row j = seq1 * seq2[j], shifted left by (n-1-j) positions.
    # Each row has n+1 digits (including carry from digit products).
    rows = []
    for j in range(n):
        # Multiply seq1 by single digit seq2[j] with carry propagation
        carry = create_constant(torch.tensor([-1.0]))  # no carry
        row_digits = []
        for i in reversed(range(n)):
            tens, ones = products[i, j]
            # Add ones digit to running sum with carry
            digit_sum, carry = sum_digits(
                embedding, ones, carry_digit if i < n - 1 else zero, carry
            )
            row_digits.append(digit_sum)
            carry_digit = tens
        # Final carry + last tens digit
        final_digit, _ = sum_digits(embedding, tens, zero, carry)
        row_digits.append(final_digit)
        row_digits = list(reversed(row_digits))  # MSB first

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


# ---------------------------------------------------------------------------
# Main network
# ---------------------------------------------------------------------------


def create_network_parts() -> Tuple[Node, PosEncoding, Embedding]:
    """Build the calculator graph and return (output_node, pos_encoding, embedding)."""
    vocab = list(
        " 0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!@#$%^&*()-+="
    ) + ["<bos", "<eos>", "default"]
    embedding = create_embedding(vocab=vocab)
    pos_encoding = create_pos_encoding()

    # --- Parsing ---
    num_seq = NumericSequence(pos_encoding, embedding, max_digits)

    is_plus = equals_vector(embedding, embedding.get_embedding("+"))
    is_minus = equals_vector(embedding, embedding.get_embedding("-"))
    is_times = equals_vector(embedding, embedding.get_embedding("*"))
    is_operator = bool_any_true([is_plus, is_minus, is_times])
    is_equals = equals_vector(embedding, embedding.get_embedding("="))

    # Restrict operator detection to before "=" — output tokens like "-" must
    # not re-trigger operator signals during autoregressive decoding.
    has_seen_equals = pos_encoding.get_prev_value(is_equals, is_equals)
    before_equals = bool_not(has_seen_equals)
    is_operator_input = bool_all_true([is_operator, before_equals])

    # Latch which operator was used (captured at the operator position)
    which_plus = pos_encoding.get_prev_value(is_plus, is_operator_input)
    which_minus = pos_encoding.get_prev_value(is_minus, is_operator_input)
    which_times = pos_encoding.get_prev_value(is_times, is_operator_input)

    # Extract operand digits (MSB first)
    first_num_digits = num_seq.get_digits_at_event(is_operator_input)
    second_num_digits = num_seq.get_digits_at_event(is_equals)

    zero_embed = create_constant(embedding.get_embedding("0"))
    eos_embed = create_constant(embedding.get_embedding("<eos>"))
    minus_embed = create_constant(embedding.get_embedding("-"))

    # Multiplication produces up to 2*max_digits digits, so seq_len must
    # accommodate that. Addition and subtraction produce shorter results
    # but are padded with eos to match.
    seq_len = 2 * max_digits + 2  # max result digits + eos padding

    # --- Addition ---
    add_result = sum_digit_seqs(embedding, first_num_digits, second_num_digits)
    add_seq = add_result + [eos_embed] * (max_digits + 2)
    add_seq = remove_leading_0s(embedding, add_seq, max_removals=max_digits - 1)

    # --- Subtraction ---
    is_a_gte_b = compare_digit_seqs(embedding, first_num_digits, second_num_digits)
    bigger = [
        select(is_a_gte_b, a, b) for a, b in zip(first_num_digits, second_num_digits)
    ]
    smaller = [
        select(is_a_gte_b, b, a) for a, b in zip(first_num_digits, second_num_digits)
    ]
    sub_abs = subtract_digit_seqs(embedding, bigger, smaller)
    # Positive case: digits + eos padding
    sub_pos = sub_abs + [eos_embed] * (max_digits + 2)
    sub_pos = remove_leading_0s(embedding, sub_pos, max_removals=max_digits - 1)
    # Negative case: "-" then clean absolute value + eos padding
    sub_abs_clean = remove_leading_0s(
        embedding, sub_abs + [eos_embed] * (max_digits + 1), max_removals=max_digits - 1
    )
    sub_neg = [minus_embed] + sub_abs_clean[: seq_len - 1]
    # Select between positive and negative
    sub_seq = [select(is_a_gte_b, sub_pos[i], sub_neg[i]) for i in range(seq_len)]

    # --- Multiplication ---
    mul_result = multiply_digit_seqs(embedding, first_num_digits, second_num_digits)
    mul_seq = mul_result + [eos_embed, eos_embed]
    mul_seq = remove_leading_0s(embedding, mul_seq, max_removals=2 * max_digits - 1)

    # --- Dispatch ---
    result_digits = []
    for i in range(seq_len):
        result_digits.append(
            switch(
                [which_plus, which_minus, which_times],
                [add_seq[i], sub_seq[i], mul_seq[i]],
            )
        )

    output_node = output_sequence(
        pos_encoding,
        is_equals,
        result_digits,
        embedding.get_embedding(" "),
    )
    return output_node, pos_encoding, embedding
