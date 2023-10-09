from typing import Tuple, Optional, List

import torch

from modelscriptor.graph import Node, Embedding, PosEncoding
from modelscriptor.graph.embedding import Unembedding
from modelscriptor.modelscript.arithmetic_ops import concat, sum_nodes
from modelscriptor.modelscript.inout_nodes import (
    create_constant,
    create_embedding,
    create_pos_encoding,
    create_unembedding,
)
from modelscriptor.modelscript.logic_ops import (
    compare_to_vector,
    cond_gate,
    bool_not,
    bool_all_true,
)
from modelscriptor.modelscript.map_select import map_to_table, select

max_digits = 3


def check_is_digit(embedding: Embedding) -> Node:
    """
    Check if the current embedding value is a digits.
    """
    return map_to_table(
        inp=embedding,
        key_to_value={
            embedding.get_embedding(str(i)): torch.tensor([1.0]) for i in range(10)
        },
        default=torch.tensor([-1.0]),
    )


def sum_digits(
    embedding: Embedding, num1: Node, num2: Node, carry_in: Node
) -> Tuple[Node, Node]:
    """
    Adds num1 + num2 + carry_in.
    Assumes num1 and num2 are both embedding-valued nodes.
    Assumes carry_in is a boolean node.
    return result as embedding-valued node and carry as boolean.
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
    """
    Sums a sequence of digits. The order of the sequence is greatest to least significance,
    so the first element is the 100s digit, the second element is the 10s digit, and the third
    element is the 1s digit.
    """

    carry = create_constant(torch.tensor([-1.0]))
    out = []
    # We add from right to left.
    for digit1, digit2 in reversed(list(zip(seq1, seq2))):
        sum, carry = sum_digits(embedding, digit1, digit2, carry)
        out.append(sum)

    return list(reversed(out))


def remove_leading_0s(
    embedding: Embedding, seq: List[Node], max_removals: int
) -> List[Node]:
    """
    Removes leading 0s from a sequence of digits.
    """
    if max_removals == 0:
        return seq

    is_leading_zero = compare_to_vector(inp=seq[0], vector=embedding.get_embedding("0"))

    out = []
    seq = seq + [seq[-1]]
    for i, _ in enumerate(seq[:-1]):
        out.append(
            select(cond=is_leading_zero, true_node=seq[i + 1], false_node=seq[i])
        )
    return remove_leading_0s(embedding, out, max_removals - 1)


def output_sequence(
    pos_encoding: PosEncoding,
    trigger_condition: Node,
    seq: List[Node],
    default_output: torch.Tensor,
):
    # Add <eos> token ot seq.
    seq = seq

    # has_triggered will be true for all positions starting when trigger_condition is true.
    has_triggered = pos_encoding.get_prev_value(trigger_condition, trigger_condition)

    out_values = []
    for i, value in enumerate(seq):
        delta = -i
        trigger = pos_encoding.get_last_value(trigger_condition, delta_pos=delta)
        out_values.append(cond_gate(trigger, value))

    return select(
        cond=has_triggered,
        true_node=sum_nodes(out_values),
        false_node=create_constant(default_output),
    )


class NumericSequence:
    def __init__(self, pos_encoding: PosEncoding, embedding: Embedding, digits: int):
        self.pos_encoding = pos_encoding
        zero_constant = create_constant(embedding.get_embedding("0"))
        is_digit = check_is_digit(embedding)
        is_num_start = bool_all_true(
            [is_digit, bool_not(pos_encoding.get_last_value(is_digit))]
        )

        current_digits = [embedding]
        for i in range(digits - 1):
            current_digits.append(
                select(
                    cond=is_num_start,
                    true_node=zero_constant,
                    false_node=pos_encoding.get_last_value(current_digits[-1]),
                )
            )

        # We always reference digit values one sequence position after.
        self.digit_values = [
            pos_encoding.get_last_value(digit) for digit in current_digits
        ]

    def get_digits_at_event(self, termination_event: Node) -> List[Node]:
        # Get the number that was just completed when termination_event is true.
        return [
            self.pos_encoding.get_prev_value(digit, termination_event)
            for digit in reversed(self.digit_values)
        ]


def create_network() -> Unembedding:
    # Define our vocabulary -- these are the tokens that will be used for our netowrk.
    vocab = list(
        " 0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!@#$%^&*()-+="
    ) + ["<bos", "<eos>", "default"]
    embedding = create_embedding(vocab=vocab)
    pos_encoding = create_pos_encoding()

    num_seq = NumericSequence(pos_encoding, embedding, max_digits)

    is_end_of_first_num = compare_to_vector(
        inp=embedding, vector=embedding.get_embedding("+")
    )

    # Define a flag for the end of the second number (when we hit the = symbol).
    is_end_of_second_num = compare_to_vector(
        inp=embedding, vector=embedding.get_embedding("=")
    )

    first_num_digits = num_seq.get_digits_at_event(is_end_of_first_num)
    second_num_digits = num_seq.get_digits_at_event(is_end_of_second_num)
    sum_digits = sum_digit_seqs(embedding, first_num_digits, second_num_digits) + [
        create_constant(embedding.get_embedding("<eos>"))
    ]
    sum_digits = remove_leading_0s(embedding, sum_digits, max_removals=max_digits - 1)

    return create_unembedding(
        output_sequence(
            pos_encoding,
            is_end_of_second_num,
            sum_digits,
            embedding.get_embedding(" "),
        ),
        embedding,
    )
