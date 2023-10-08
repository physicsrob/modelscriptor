from typing import Tuple, Optional

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


def check_is_num(embedding_value: Node, embedding: Embedding) -> Node:
    return map_to_table(
        inp=embedding_value,
        key_to_value={
            embedding.get_embedding(str(i)): torch.tensor([1.0]) for i in range(10)
        },
        default=torch.tensor([-1.0]),
    )


def sum_numbers(
    embedding: Embedding, num1: Node, num2: Node, carry_in: Optional[Node] = None
) -> Tuple[Node, Node]:
    """
    Adds num1 + num2 + carry_in.
    Assumes num1 and num2 are both embedding-valued nodes.
    Assumes carry_in is a boolean node.
    return result as embedding-valued node and carry as boolean.
    """

    sum_out_table = {}
    carry_out_table = {}

    # If carry_in is specified, we use concat([num1, num2, carry_in]) as the key.
    # If carry_in is not specified, we use concat([num1, num2]) as the key.
    if carry_in:
        key = concat([num1, num2, carry_in])
    else:
        key = concat([num1, num2])

    for A in range(10):
        for B in range(10):
            for C in [0, 1] if carry_in else [0]:
                if carry_in:
                    carry_tensor = torch.tensor([1.0 if C else -1.0])
                    entry_key = torch.cat(
                        [
                            embedding.get_embedding(str(A)),
                            embedding.get_embedding(str(B)),
                            carry_tensor,
                        ]
                    )
                else:
                    entry_key = torch.cat(
                        [
                            embedding.get_embedding(str(A)),
                            embedding.get_embedding(str(B)),
                        ]
                    )
                sum_out_table[entry_key] = embedding.get_embedding(
                    str((A + B + C) % 10)
                )
                carry_out_table[entry_key] = torch.tensor(
                    [1.0 if (A + B + C) >= 10 else -1.0]
                )

    return (
        map_to_table(
            inp=key,
            key_to_value=sum_out_table,
            default=embedding.get_embedding("0"),
        ),
        map_to_table(key, key_to_value=carry_out_table, default=torch.tensor([-1.0])),
    )


def output_sequence(
    pos_encoding: PosEncoding,
    trigger_condition: Node,
    value1: Node,
    value2: Node,
    value3: Node,
    default_output: torch.Tensor,
    eos_output: torch.Tensor,
):
    # has_triggered will be true for all positions starting when trigger_condition is true.
    has_triggered = pos_encoding.get_prev_value(trigger_condition, trigger_condition)

    # If trigger_condition is true, output value1.
    # Next, output value 2.
    trigger_condition2 = pos_encoding.get_last_value(trigger_condition, delta_pos=-1)
    trigger_condition3 = pos_encoding.get_last_value(trigger_condition, delta_pos=-2)
    trigger_condition4 = pos_encoding.get_last_value(trigger_condition, delta_pos=-3)

    out_value1 = cond_gate(trigger_condition, value1)
    out_value2 = cond_gate(trigger_condition2, value2)
    out_value3 = cond_gate(trigger_condition3, value3)
    out_value4 = cond_gate(trigger_condition4, create_constant(eos_output))

    return select(
        cond=has_triggered,
        true_node=sum_nodes([out_value1, out_value2, out_value3, out_value4]),
        false_node=create_constant(default_output),
    )


def create_network() -> Unembedding:
    # Define our vocabulary -- these are the tokens that will be used for our netowrk.
    vocab = list(
        " 0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!@#$%^&*()-+="
    ) + ["<bos", "<eos>", "default"]
    embedding = create_embedding(vocab=vocab)
    pos_encoding = create_pos_encoding()

    #
    # Make network that adds 2 digit numbers
    #

    # Define current number.
    zero_constant = create_constant(embedding.get_embedding("0"))
    is_num = check_is_num(embedding_value=embedding, embedding=embedding)

    is_num_start = bool_all_true(
        [is_num, bool_not(pos_encoding.get_last_value(is_num, delta_pos=-1))]
    )

    # current_num is the embedding of the current character if it is a number,
    # otherwise it is the embedding of 0.
    current_num = select(cond=is_num, true_node=embedding, false_node=zero_constant)

    # Define a flag for the end of the first number (when we hit the + symbol).
    is_end_of_first_num = compare_to_vector(
        inp=embedding, vector=embedding.get_embedding("+")
    )

    # Define a flag for the end of the second number (when we hit the = symbol).
    is_end_of_second_num = compare_to_vector(
        inp=embedding, vector=embedding.get_embedding("=")
    )

    current_num_1s = current_num
    current_num_10s = select(
        cond=is_num_start,
        true_node=zero_constant,
        false_node=pos_encoding.get_last_value(current_num_1s, delta_pos=-1),
    )
    current_num_100s = select(
        cond=is_num_start,
        true_node=zero_constant,
        false_node=pos_encoding.get_last_value(current_num_10s, delta_pos=-1),
    )

    just_completed_num_1s = pos_encoding.get_last_value(current_num, delta_pos=-1)
    just_completed_num_10s = pos_encoding.get_last_value(current_num_10s, delta_pos=-1)
    just_completed_num_100s = pos_encoding.get_last_value(
        current_num_100s, delta_pos=-1
    )

    first_num_1s = pos_encoding.get_prev_value(
        just_completed_num_1s, is_end_of_first_num
    )
    first_num_10s = pos_encoding.get_prev_value(
        just_completed_num_10s, is_end_of_first_num
    )
    first_num_100s = pos_encoding.get_prev_value(
        just_completed_num_100s, is_end_of_first_num
    )
    second_num_1s = pos_encoding.get_prev_value(
        just_completed_num_1s, is_end_of_second_num
    )
    second_num_10s = pos_encoding.get_prev_value(
        just_completed_num_10s, is_end_of_second_num
    )
    second_num_100s = pos_encoding.get_prev_value(
        just_completed_num_100s, is_end_of_second_num
    )

    # Figure out how to calculate output index.
    sum_1s, carry_1s = sum_numbers(embedding, first_num_1s, second_num_1s)
    sum_10s, carry_10s = sum_numbers(embedding, first_num_10s, second_num_10s, carry_1s)
    sum_100s, carry_100s = sum_numbers(
        embedding, first_num_100s, second_num_100s, carry_10s
    )
    # return create_unembedding(sum_100s, embedding)
    return create_unembedding(
        output_sequence(
            pos_encoding,
            is_end_of_second_num,
            sum_100s,
            sum_10s,
            sum_1s,
            embedding.get_embedding(" "),
            embedding.get_embedding("<eos>"),
        ),
        embedding,
    )
