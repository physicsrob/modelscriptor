"""Adder V2: Scalar-space addition.

Converts digit embeddings to scalar numbers, adds them, then converts
the result back to digit embeddings. Eliminates carry propagation entirely.
"""

from typing import List

import torch

from modelscriptor.graph import Node, Embedding
from modelscriptor.graph.embedding import Unembedding
from modelscriptor.modelscript.arithmetic_ops import (
    add,
    add_scaled_nodes,
    sum_nodes,
)
from modelscriptor.modelscript.const import turn_on_speed
from modelscriptor.modelscript.ffn_layer import ffn_layer
from modelscriptor.modelscript.inout_nodes import (
    create_constant,
    create_embedding,
    create_pos_encoding,
    create_unembedding,
)
from modelscriptor.modelscript.logic_ops import compare_to_vector
from modelscriptor.modelscript.map_select import map_to_table

from examples.adder import (
    check_is_digit,
    remove_leading_0s,
    output_sequence,
    NumericSequence,
)


def digit_to_scaled_scalar(
    embedding: Embedding, digit_node: Node, place_value: float
) -> Node:
    """Convert an embedding-valued digit node to a scalar multiplied by place_value.

    Input: 8D embedding node representing a digit 0-9.
    Output: 1D scalar node with value digit * place_value.
    """
    table = {}
    for i in range(10):
        table[embedding.get_embedding(str(i))] = torch.tensor([float(i) * place_value])
    return map_to_table(inp=digit_node, key_to_value=table, default=torch.tensor([0.0]))


def digits_to_number(embedding: Embedding, digit_nodes: List[Node]) -> Node:
    """Convert a list of embedding-valued digit nodes (MSB first) to a scalar number.

    E.g., digit nodes for [1, 2, 3] -> scalar 123.0
    """
    num_digits = len(digit_nodes)
    scaled = []
    for i, digit in enumerate(digit_nodes):
        place_value = 10.0 ** (num_digits - 1 - i)
        scaled.append(digit_to_scaled_scalar(embedding, digit, place_value))
    return sum_nodes(scaled)


def thermometer_floor_div(inp: Node, divisor: int, max_value: int) -> Node:
    """Compute floor(inp / divisor) using thermometer coding in a single FFN layer.

    Places n threshold comparisons at k*divisor - 0.5 for k=1..n, where
    n = max_value // divisor. Each crossed threshold contributes 1.0 to the output.
    Uses half-integer thresholds for numerical safety.
    """
    assert len(inp) == 1, "Input must be a 1D scalar node"
    n = max_value // divisor
    if n == 0:
        return create_constant(torch.tensor([0.0]))

    d_int = 2 * n
    input_proj = torch.zeros(d_int, 1)
    input_bias = torch.zeros(d_int)
    output_proj = torch.zeros(d_int, 1)

    for k in range(n):
        threshold = (k + 1) * divisor - 0.5
        row = 2 * k
        input_proj[row, 0] = turn_on_speed
        input_proj[row + 1, 0] = turn_on_speed
        input_bias[row] = -turn_on_speed * threshold
        input_bias[row + 1] = -turn_on_speed * threshold - 1.0
        output_proj[row, 0] = 1.0
        output_proj[row + 1, 0] = -1.0

    return ffn_layer(
        input_node=inp,
        input_proj=input_proj,
        input_bias=input_bias,
        output_proj=output_proj,
        output_bias=torch.zeros(1),
    )


def number_to_digit_scalars(inp: Node, num_digits: int, max_value: int) -> List[Node]:
    """Extract individual digit scalars (0.0-9.0) from a scalar number, MSB first.

    Uses sequential thermometer-based division: extract the most significant digit,
    subtract it, then extract the next digit from the remainder.
    """
    digits = []
    remainder = inp
    for i in range(num_digits):
        place = 10 ** (num_digits - 1 - i)
        if place == 1:
            digits.append(remainder)
        else:
            digit = thermometer_floor_div(remainder, place, max_value)
            digits.append(digit)
            remainder = add_scaled_nodes(1.0, remainder, -float(place), digit)
            max_value = place - 1
    return digits


def scalar_to_embedding(inp: Node, embedding: Embedding) -> Node:
    """Convert a scalar digit (0.0-9.0) to its 8D embedding vector.

    Uses thermometer coding with 9 comparisons at half-integer thresholds.
    Crossing threshold k+0.5 adds embed(k+1) - embed(k) to the output,
    so for integer input d the result is embed(d).
    """
    assert len(inp) == 1, "Input must be a 1D scalar node"
    d_embed = embedding.d_embed
    n_thresholds = 9
    d_int = 2 * n_thresholds

    input_proj = torch.zeros(d_int, 1)
    input_bias = torch.zeros(d_int)
    output_proj = torch.zeros(d_int, d_embed)

    for k in range(n_thresholds):
        threshold = k + 0.5
        row = 2 * k
        input_proj[row, 0] = turn_on_speed
        input_proj[row + 1, 0] = turn_on_speed
        input_bias[row] = -turn_on_speed * threshold
        input_bias[row + 1] = -turn_on_speed * threshold - 1.0

        delta = embedding.get_embedding(str(k + 1)) - embedding.get_embedding(str(k))
        output_proj[row, :] = delta
        output_proj[row + 1, :] = -delta

    output_bias = embedding.get_embedding(str(0)).clone()

    return ffn_layer(
        input_node=inp,
        input_proj=input_proj,
        input_bias=input_bias,
        output_proj=output_proj,
        output_bias=output_bias,
    )


def create_network(max_digits: int = 3) -> Unembedding:
    """Create a v2 adder network that adds two numbers up to max_digits digits.

    Uses scalar-space arithmetic: converts digit embeddings to numbers,
    adds them, then converts back to digit embeddings.
    """
    vocab = list(
        " 0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!@#$%^&*()-+="
    ) + ["<bos", "<eos>", "default"]
    embedding = create_embedding(vocab=vocab)
    pos_encoding = create_pos_encoding()

    # Reuse v1's input parsing infrastructure
    num_seq = NumericSequence(pos_encoding, embedding, max_digits)

    is_end_of_first_num = compare_to_vector(
        inp=embedding, vector=embedding.get_embedding("+")
    )
    is_end_of_second_num = compare_to_vector(
        inp=embedding, vector=embedding.get_embedding("=")
    )

    first_num_digits = num_seq.get_digits_at_event(is_end_of_first_num)
    second_num_digits = num_seq.get_digits_at_event(is_end_of_second_num)

    # V2: Convert digit embeddings to scalar numbers
    number_a = digits_to_number(embedding, first_num_digits)
    number_b = digits_to_number(embedding, second_num_digits)

    # Add as plain numbers
    result = add(number_a, number_b)

    # Convert result back to digit embeddings
    max_result = 2 * (10**max_digits - 1)  # e.g., 999+999=1998
    num_output_digits = max_digits + 1  # Extra digit for overflow
    digit_scalars = number_to_digit_scalars(result, num_output_digits, max_result)
    result_digits = [scalar_to_embedding(d, embedding) for d in digit_scalars]

    # Append <eos> and remove leading zeros
    result_digits = result_digits + [create_constant(embedding.get_embedding("<eos>"))]
    result_digits = remove_leading_0s(embedding, result_digits, max_removals=max_digits)

    return create_unembedding(
        output_sequence(
            pos_encoding,
            is_end_of_second_num,
            result_digits,
            embedding.get_embedding(" "),
        ),
        embedding,
    )
