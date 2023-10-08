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


def create_network() -> Unembedding:
    # Define our vocabulary -- these are the tokens that will be used for our netowrk.
    vocab = list(
        "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!@#$%^&*()-+="
    ) + ["<eos>", "default"]
    embedding = create_embedding(vocab=vocab)
    pos_encoding = create_pos_encoding()

    #
    # Make network that adds 1 digit numbers
    #

    # Define current number.
    zero_constant = create_constant(embedding.get_embedding("0"))
    is_num = check_is_num(embedding_value=embedding, embedding=embedding)

    # current_num is the embedding of the current character if it is a number,
    # otherwise it is the embedding of 0.
    current_num = select(cond=is_num, true_node=embedding, false_node=zero_constant)

    # Define a flag for the end of the first number (when we hit the + symbol).
    is_first_num = compare_to_vector(inp=embedding, vector=embedding.get_embedding("+"))

    # Define a flag for the end of the second number (when we hit the = symbol).
    is_second_num = compare_to_vector(
        inp=embedding, vector=embedding.get_embedding("=")
    )

    just_completed_num = pos_encoding.get_last_value(current_num, delta_pos=-1)
    first_num = pos_encoding.get_prev_value(just_completed_num, is_first_num)
    second_num = pos_encoding.get_prev_value(just_completed_num, is_second_num)

    # Figure out how to calculate output index.
    summed, carry = sum_numbers(embedding, first_num, second_num)
    return create_unembedding(summed, embedding)
