"""Adder V2: Scalar-space addition.

Instead of propagating carries digit-by-digit (V1), this converts digit
embeddings into plain numbers, adds them as scalars, then converts back.

  "123" + "456"
  → [embed("1"), embed("2"), embed("3")]   input: 8D embedding per digit
  → 123.0                                  digits_to_number
  + 456.0                                  digits_to_number
  = 579.0                                  scalar addition — one Add node
  → [5.0, 7.0, 9.0]                       number_to_digit_scalars
  → [embed("5"), embed("7"), embed("9")]   scalar_to_embedding

The scalar↔digit conversions use thermometer coding: a ReLU-based technique
that counts how many thresholds an input crosses, like mercury rising in a
thermometer. See thermometer_floor_div for the detailed explanation.
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
    """Convert a digit embedding to a scalar multiplied by place_value.

    Example: embed("5") with place_value=100 → 500.0

    Uses a 10-entry lookup table: embed("0")→0, embed("1")→place_value, ...,
    embed("9")→9*place_value.
    """
    table = {}
    for i in range(10):
        table[embedding.get_embedding(str(i))] = torch.tensor([float(i) * place_value])
    return map_to_table(inp=digit_node, key_to_value=table, default=torch.tensor([0.0]))


def digits_to_number(embedding: Embedding, digit_nodes: List[Node]) -> Node:
    """Convert digit embeddings (MSB first) to a single scalar.

    Example: [embed("1"), embed("2"), embed("3")]
           → 1*100 + 2*10 + 3*1 = 123.0
    """
    num_digits = len(digit_nodes)
    scaled = []
    for i, digit in enumerate(digit_nodes):
        place_value = 10.0 ** (num_digits - 1 - i)
        scaled.append(digit_to_scaled_scalar(embedding, digit, place_value))
    return sum_nodes(scaled)


def thermometer_floor_div(inp: Node, divisor: int, max_value: int) -> Node:
    """Compute floor(inp / divisor) using thermometer coding in a single FFN layer.

    Example: floor(x / 10) for x in [0, 99]

    Place a detector at each multiple of the divisor. Each detector outputs
    1.0 when x crosses that multiple. Sum them — like mercury in a thermometer.

        threshold  9.5 → fires when x >= 10  (contributes 1.0)
        threshold 19.5 → fires when x >= 20  (contributes 1.0)
        ...
        threshold 89.5 → fires when x >= 90  (contributes 1.0)

        x = 35 → first 3 detectors fire → output = 3 = floor(35/10) ✓

    Half-integer thresholds (9.5 not 10.0) ensure clean separation for
    integer inputs: x=9 is well below 9.5, x=10 is well above.

    Each detector is a paired ReLU that produces a 0-or-1 step:

        detector(x) = ReLU(s*(x - threshold)) - ReLU(s*(x - threshold) - 1)

        x < threshold → both ReLUs output 0              → 0
        x > threshold → both ramp up equally, offset by 1 → 1

    The speed s (turn_on_speed=10) makes the ramp steep so it saturates
    quickly. Two ReLU units per detector → d_int = 2*n.
    """
    assert len(inp) == 1, "Input must be a 1D scalar node"
    n = max_value // divisor
    if n == 0:
        return create_constant(torch.tensor([0.0]))

    d_int = 2 * n  # Two ReLU units per threshold detector
    input_proj = torch.zeros(d_int, 1)
    input_bias = torch.zeros(d_int)
    output_proj = torch.zeros(d_int, 1)

    for k in range(n):
        threshold = (k + 1) * divisor - 0.5  # Half-integer: 9.5, 19.5, ...
        row = 2 * k

        # Paired ReLU: ReLU(s*x - s*threshold) - ReLU(s*x - s*threshold - 1)
        input_proj[row, 0] = turn_on_speed
        input_proj[row + 1, 0] = turn_on_speed
        input_bias[row] = -turn_on_speed * threshold
        input_bias[row + 1] = -turn_on_speed * threshold - 1.0

        # First ReLU contributes +1, second cancels the ramp → net step of 1.0
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

    Greedy extraction: peel off the most significant digit, subtract it,
    repeat on the remainder.

    Example: 579 → 3 digits

        digit[0] = floor(579 / 100) = 5    remainder = 579 - 5*100 = 79
        digit[1] = floor(79 / 10)   = 7    remainder = 79  - 7*10  = 9
        digit[2] = remainder         = 9

    max_value shrinks at each step (999→99→9) so the thermometer FFNs
    only need thresholds for the remaining range.
    """
    digits = []
    remainder = inp
    for i in range(num_digits):
        place = 10 ** (num_digits - 1 - i)
        if place == 1:
            # Last digit: just the remainder, no division needed
            digits.append(remainder)
        else:
            digit = thermometer_floor_div(remainder, place, max_value)
            digits.append(digit)
            # remainder = remainder - digit * place
            remainder = add_scaled_nodes(1.0, remainder, -float(place), digit)
            max_value = place - 1  # Tighten range for next digit
    return digits


def scalar_to_embedding(inp: Node, embedding: Embedding) -> Node:
    """Convert a scalar digit (0.0-9.0) back to its 8D embedding vector.

    Same paired-ReLU thermometer as thermometer_floor_div, but instead of
    each threshold contributing 1.0, it contributes an embedding delta.
    This reconstructs the embedding by telescoping from embed(0):

        input=0 → embed(0)
        input=1 → embed(0) + [embed(1) - embed(0)]              = embed(1)
        input=2 → embed(0) + [embed(1)-embed(0)] + [embed(2)-embed(1)] = embed(2)
        ...
        input=d → embed(d)

    9 thresholds at 0.5, 1.5, ..., 8.5 cover digits 0-9.
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

        # Same paired-ReLU step function as thermometer_floor_div
        input_proj[row, 0] = turn_on_speed
        input_proj[row + 1, 0] = turn_on_speed
        input_bias[row] = -turn_on_speed * threshold
        input_bias[row + 1] = -turn_on_speed * threshold - 1.0

        # Instead of contributing 1.0, contribute the embedding delta
        delta = embedding.get_embedding(str(k + 1)) - embedding.get_embedding(str(k))
        output_proj[row, :] = delta
        output_proj[row + 1, :] = -delta

    # Start from embed("0"); deltas telescope up to embed(d)
    output_bias = embedding.get_embedding(str(0)).clone()

    return ffn_layer(
        input_node=inp,
        input_proj=input_proj,
        input_bias=input_bias,
        output_proj=output_proj,
        output_bias=output_bias,
    )


def create_network(max_digits: int = 3) -> Unembedding:
    """Create a v2 adder: parses "A+B=", outputs result digits autoregressively.

    Three phases:
      1. Parse: extract digit embeddings for A and B from the token stream
      2. Compute: embeddings → scalars → add → scalars → embeddings
      3. Output: emit result digits left-to-right, then <eos>
    """
    vocab = list(
        " 0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!@#$%^&*()-+="
    ) + ["<bos", "<eos>", "default"]
    embedding = create_embedding(vocab=vocab)
    pos_encoding = create_pos_encoding()

    # --- Phase 1: Parse operand digits from the token stream ---
    # NumericSequence tracks a sliding window of digits as tokens arrive.
    # get_digits_at_event captures the window when the trigger token appears.
    num_seq = NumericSequence(pos_encoding, embedding, max_digits)

    is_end_of_first_num = compare_to_vector(
        inp=embedding, vector=embedding.get_embedding("+")
    )
    is_end_of_second_num = compare_to_vector(
        inp=embedding, vector=embedding.get_embedding("=")
    )

    first_num_digits = num_seq.get_digits_at_event(is_end_of_first_num)
    second_num_digits = num_seq.get_digits_at_event(is_end_of_second_num)

    # --- Phase 2: Scalar-space addition ---
    # Convert digit embeddings to numbers, add, convert back.
    number_a = digits_to_number(embedding, first_num_digits)
    number_b = digits_to_number(embedding, second_num_digits)
    result = add(number_a, number_b)

    # Result can overflow by one digit (e.g. 999+999=1998)
    max_result = 2 * (10**max_digits - 1)
    num_output_digits = max_digits + 1
    digit_scalars = number_to_digit_scalars(result, num_output_digits, max_result)
    result_digits = [scalar_to_embedding(d, embedding) for d in digit_scalars]

    # --- Phase 3: Format output ---
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
