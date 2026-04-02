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
    compare_to_vector,
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


def create_network_parts() -> Tuple[Node, PosEncoding, Embedding]:
    """Build the calculator graph and return (output_node, pos_encoding, embedding)."""
    vocab = list(
        " 0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!@#$%^&*()-+="
    ) + ["<bos", "<eos>", "default"]
    embedding = create_embedding(vocab=vocab)
    pos_encoding = create_pos_encoding()

    # --- Parsing ---
    num_seq = NumericSequence(pos_encoding, embedding, max_digits)

    is_plus = compare_to_vector(embedding, embedding.get_embedding("+"))
    is_minus = compare_to_vector(embedding, embedding.get_embedding("-"))
    is_times = compare_to_vector(embedding, embedding.get_embedding("*"))
    is_operator = bool_any_true([is_plus, is_minus, is_times])
    is_equals = compare_to_vector(embedding, embedding.get_embedding("="))

    # Latch which operator was used (captured at the operator position)
    which_plus = pos_encoding.get_prev_value(is_plus, is_operator)
    which_minus = pos_encoding.get_prev_value(is_minus, is_operator)
    which_times = pos_encoding.get_prev_value(is_times, is_operator)

    # Extract operand digits
    first_num_digits = num_seq.get_digits_at_event(is_operator)
    second_num_digits = num_seq.get_digits_at_event(is_equals)

    zero_embed = create_constant(embedding.get_embedding("0"))
    eos_embed = create_constant(embedding.get_embedding("<eos>"))

    # --- Addition ---
    add_result = sum_digit_seqs(embedding, first_num_digits, second_num_digits)

    # Pad addition result to 2*max_digits + 1 with leading zeros
    # Addition produces max_digits digits; pad to 2*max_digits+1
    result_len = 2 * max_digits + 1
    add_padded = [zero_embed] * (result_len - len(add_result)) + add_result

    # --- Subtraction (placeholder: output zeros) ---
    sub_padded = [zero_embed] * result_len

    # --- Multiplication (placeholder: output zeros) ---
    mul_padded = [zero_embed] * result_len

    # --- Dispatch ---
    result_digits = []
    for i in range(result_len):
        result_digits.append(
            switch(
                [which_plus, which_minus, which_times],
                [add_padded[i], sub_padded[i], mul_padded[i]],
            )
        )

    # Remove leading zeros and append <eos>
    result_digits = result_digits + [eos_embed]
    result_digits = remove_leading_0s(
        embedding, result_digits, max_removals=result_len - 1
    )

    output_node = output_sequence(
        pos_encoding,
        is_equals,
        result_digits,
        embedding.get_embedding(" "),
    )
    return output_node, pos_encoding, embedding
