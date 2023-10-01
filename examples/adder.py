from typing import Tuple

import torch

from modelscriptor.graph import Node, Embedding
from modelscriptor.graph.embedding import Unembedding
from modelscriptor.modelscript.arithmetic_ops import concat
from modelscriptor.modelscript.inout_nodes import (
    create_constant,
    create_embedding,
    create_pos_encoding,
    create_unembedding,
)
from modelscriptor.modelscript.logic_ops import compare_to_vector
from modelscriptor.modelscript.map_select import map_to_table, select


def check_is_num(embedding_value: Node, embedding: Embedding) -> Node:
    return map_to_table(
        inp=embedding_value,
        key_to_value={
            embedding.get_embedding(str(i)): torch.tensor([1.0]) for i in range(10)
        },
        default=torch.tensor([-1.0]),
    )


def sum_numbers(embedding: Embedding, num1: Node, num2: Node) -> Tuple[Node, Node]:
    """
    Adds num1 with num2.
    Assumes num1 and num2 are both embedding-valued nodes.
    return result as embedding-valued node and carry as boolean.
    """
    result_table = {}
    carry_table = {}
    for A in range(10):
        for B in range(10):
            numcat = torch.cat(
                [embedding.get_embedding(str(A)), embedding.get_embedding(str(B))]
            )
            result_table[numcat] = embedding.get_embedding(str((A + B) % 10))
            carry_table[numcat] = torch.tensor([1.0 if (A + B) >= 10 else -1.0])

    num1_num2 = concat([num1, num2])
    return (
        map_to_table(
            inp=num1_num2,
            key_to_value=result_table,
            default=embedding.get_embedding("0"),
        ),
        map_to_table(num1_num2, key_to_value=carry_table, default=torch.tensor([-1.0])),
    )


# def output_sequence(
#     pos_encoding: PosEncoding,
#     trigger_condition: Node,
#     value1: Node,
#     value2: Node,
#     default_output: torch.Tensor,
# ):
#     # If trigger_condition is true, output value1.
#     # Next, output value 2.
#     trigger_condition2 = pos_encoding.get_last_value(trigger_condition, delta_pos=-1)
#     trigger_condition3 = pos_encoding.get_last_value(trigger_condition, delta_pos=-2)
#     output = switch_table(
#         [(trigger_condition, value1), (trigger_condition2, value2)],
#         default=default_output,
#     )
#


def create_network() -> Unembedding:
    vocab = list(
        "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!@#$%^&*()-+="
    ) + ["<eos>", "default"]
    embedding = create_embedding(vocab=vocab)
    pos_encoding = create_pos_encoding()

    # Make network that adds 1 digit numbers
    zero_constant = create_constant(embedding.get_embedding("0"))
    is_num = check_is_num(embedding_value=embedding, embedding=embedding)
    current_num = select(cond=is_num, true_node=embedding, false_node=zero_constant)
    just_completed_num = pos_encoding.get_last_value(current_num, delta_pos=-1)
    is_first_num = compare_to_vector(inp=embedding, vector=embedding.get_embedding("+"))
    is_second_num = compare_to_vector(
        inp=embedding, vector=embedding.get_embedding("=")
    )

    first_num = pos_encoding.get_prev_value(just_completed_num, is_first_num)
    second_num = pos_encoding.get_prev_value(just_completed_num, is_second_num)

    # Figure out how to calculate output index.
    summed, carry = sum_numbers(embedding, first_num, second_num)
    return create_unembedding(summed, embedding)
