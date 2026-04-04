"""Calculator using embedding-space arithmetic: supports +, -, * on positive
integers up to max_digits digits.

All digit manipulation uses embedding-valued nodes and map_to_table lookups.
Addition uses carry propagation, subtraction uses borrow propagation, and
multiplication uses long multiplication with partial product rows — all
mirroring pencil-and-paper algorithms.

See calculator_v2 for the scalar-space alternative, which converts digits to
scalars and uses thermometer-coded arithmetic.
"""

from typing import Tuple, List

import torch

from torchwright.graph import Node, Embedding, PosEncoding
from torchwright.ops.inout_nodes import (
    create_constant,
    create_embedding,
    create_pos_encoding,
)
from torchwright.ops.logic_ops import (
    equals_vector,
    bool_not,
    bool_all_true,
    bool_any_true,
)
from torchwright.ops.map_select import select, switch
from torchwright.ops.embedding_arithmetic import (
    sum_digit_seqs,
    subtract_digit_seqs,
    compare_digit_seqs,
    multiply_digit_seqs,
)
from torchwright.ops.sequence_ops import (
    NumericSequence,
    output_sequence,
    remove_leading_0s,
)

max_digits = 3
D_MODEL = 1536


def create_network_parts() -> Tuple[Node, PosEncoding, Embedding]:
    """Build the calculator graph and return (output_node, pos_encoding, embedding)."""
    vocab = list(
        " 0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!@#$%^&*()-+="
    ) + ["\n", "<bos", "<eos>", "default"]
    embedding = create_embedding(vocab=vocab)
    pos_encoding = create_pos_encoding()

    # --- Phase 1: Parse operands and detect operator ---
    num_seq = NumericSequence(pos_encoding, embedding, max_digits)

    is_plus = equals_vector(embedding, embedding.get_embedding("+"))
    is_minus = equals_vector(embedding, embedding.get_embedding("-"))
    is_times = equals_vector(embedding, embedding.get_embedding("*"))
    is_operator = bool_any_true([is_plus, is_minus, is_times])
    is_equals = equals_vector(embedding, embedding.get_embedding("\n"))

    # Guard: only detect operators before "=" — output tokens like "-" must
    # not re-trigger operator signals during autoregressive decoding.
    has_seen_equals = pos_encoding.get_prev_value(is_equals, is_equals)
    before_equals = bool_not(has_seen_equals)
    is_operator_input = bool_all_true([is_operator, before_equals])

    # Latch which operator was used (captured at the operator position,
    # carried forward to all later positions via attention).
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
    seq_len = 2 * max_digits + 2

    # --- Phase 2a: Addition ---
    add_result = sum_digit_seqs(embedding, first_num_digits, second_num_digits)
    add_seq = add_result + [eos_embed] * (max_digits + 2)
    add_seq = remove_leading_0s(embedding, add_seq, max_removals=max_digits - 1)

    # --- Phase 2b: Subtraction ---
    # Compute |A - B| and determine sign separately.
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
    # Negative case: "-" prefix then absolute value
    sub_abs_clean = remove_leading_0s(
        embedding, sub_abs + [eos_embed] * (max_digits + 1), max_removals=max_digits - 1
    )
    sub_neg = [minus_embed] + sub_abs_clean[: seq_len - 1]
    sub_seq = [select(is_a_gte_b, sub_pos[i], sub_neg[i]) for i in range(seq_len)]

    # --- Phase 2c: Multiplication (long multiplication) ---
    mul_result = multiply_digit_seqs(embedding, first_num_digits, second_num_digits)
    mul_seq = mul_result + [eos_embed, eos_embed]
    mul_seq = remove_leading_0s(embedding, mul_seq, max_removals=2 * max_digits - 1)

    # --- Phase 3: Dispatch by operator and output ---
    # switch() selects the result corresponding to whichever operator was used.
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
