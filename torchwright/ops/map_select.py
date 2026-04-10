from torchwright.graph import Node, Concatenate
from typing import List, Dict
import torch

from torchwright.ops.const import (
    big_offset,
    step_sharpness,
    embedding_step_sharpness,
)
from torchwright.ops.logic_ops import cond_add_vector, cond_gate
from torchwright.ops.arithmetic_ops import sum_nodes
from torchwright.ops.linear_relu_linear import linear_relu_linear


def map_to_table(
    inp: Node, key_to_value: Dict[torch.Tensor, torch.Tensor], default: torch.Tensor
) -> Node:
    """
    Maps the value of the input node to a lookup table.

    Args:
        inp (Node): Node whose values will be looked up.
        key_to_value (Dict[torch.Tensor, torch.Tensor]): Lookup table mapping from keys to values.
        default (torch.Tensor): Default tensor to return if the input value doesn't exist in the table.

    Returns:
        Node: Output node with mapped values.
    """
    d_keys = {len(x) for x in key_to_value.keys()}
    d_values = {len(x) for x in key_to_value.values()}
    assert len(d_keys) == 1
    assert len(d_values) == 1
    d_key = d_keys.pop()
    d_value = d_values.pop()
    assert len(inp) == d_key
    assert len(default) == d_value

    d_hidden = len(key_to_value)
    speed = embedding_step_sharpness
    # We'll use 1 MLP entry per item in the table, and an overall output bias of the default value
    # So roughly speaking:
    # input_proj will be (d_hidden x d_key), where input_proj[i, :] = table.keys()[i]
    # input_bias will be (d_hidden), where input_bias[i] = 1.0/speed - (table.keys()[i] @ table.keys()[i])
    # output_proj will be (d_hidden, d_value), where output_proj[i, :] = speed * (table.values()[i] - default)
    # output_bias will be (d_value), equal to default

    input_proj = torch.zeros(d_hidden, d_key)
    input_bias = torch.zeros(d_hidden)
    output_proj = torch.zeros(d_hidden, d_value)

    for i, (key, value) in enumerate(key_to_value.items()):
        input_proj[i, :] = key
        input_bias[i] = 1.0 / speed - (key @ key)
        output_proj[i, :] = speed * (value - default)

    return linear_relu_linear(
        input_node=inp,
        input_proj=input_proj,
        input_bias=input_bias,
        output_proj=output_proj,
        output_bias=default,
    )


def switch(conditions: List[Node], values: List[Node]) -> Node:
    """
    Select one of N values based on which condition is true.

    Assumes exactly one condition is true (1.0), rest are false (-1.0).

    Args:
        conditions (List[Node]): Boolean condition nodes (each length 1).
        values (List[Node]): Value nodes (all same length).

    Returns:
        Node: The value whose corresponding condition is true.
    """
    return sum_nodes([cond_gate(c, v) for c, v in zip(conditions, values)])


def select(cond: Node, true_node: Node, false_node: Node) -> Node:
    """
    Outputs one of two nodes based on a boolean condition.

    Args:
        cond (Node): Condition node that outputs either true or false.
        true_node (Node): Node to be outputted if the condition is true.
        false_node (Node): Node to be outputted if the condition is false.

    Returns:
        Node: Either true_node or false_node based on the condition.
    """
    assert len(cond) == 1  # Condition must be length 1
    assert len(true_node) == len(false_node)

    d = len(true_node)
    # Fused single L→R→L reading [cond, true_node, false_node]:
    #   unit_a[j] = ReLU(big_offset*cond + true_j)    -- alive when cond=+1
    #   unit_b[j] = ReLU(-big_offset*cond + false_j)  -- alive when cond=-1
    #   out_j     = unit_a[j] + unit_b[j] - big_offset
    d_hidden = 2 * d
    input_proj = torch.zeros(d_hidden, 1 + 2 * d)
    input_bias = torch.zeros(d_hidden)
    output_proj = torch.zeros(d_hidden, d)
    output_bias = torch.full((d,), -big_offset)

    for j in range(d):
        a = j
        b = d + j
        input_proj[a, 0] = big_offset
        input_proj[a, 1 + j] = 1.0
        input_proj[b, 0] = -big_offset
        input_proj[b, 1 + d + j] = 1.0
        output_proj[a, j] = 1.0
        output_proj[b, j] = 1.0

    x = Concatenate([cond, true_node, false_node])
    return linear_relu_linear(
        input_node=x,
        input_proj=input_proj,
        input_bias=input_bias,
        output_proj=output_proj,
        output_bias=output_bias,
        name="select",
    )


def in_range(lower: Node, upper: Node, n_slots: int) -> Node:
    """Test each integer position against a runtime interval.

    For each position i in {0, 1, ..., n_slots-1}, returns 1.0 (true)
    if lower <= i + 0.5 < upper, or -1.0 (false) otherwise.

    The +0.5 offset means position i is "in range" when the interval
    covers the center of the integer bin.

    Args:
        lower: Scalar node, lower bound of the interval.
        upper: Scalar node, upper bound of the interval.
        n_slots: Number of integer positions to test (0 through n_slots-1).

    Returns:
        Node of width n_slots, each value 1.0 (in range) or -1.0 (out of range).
    """
    assert len(lower) == 1
    assert len(upper) == 1

    S = step_sharpness
    d_hidden = 4 * n_slots  # 4 neurons per position

    # Input is [lower, upper], d_input=2
    inp = Concatenate([lower, upper])

    input_proj = torch.zeros(d_hidden, 2)
    input_bias = torch.zeros(d_hidden)
    output_proj = torch.zeros(d_hidden, n_slots)
    output_bias = torch.full((n_slots,), -1.0)

    for i in range(n_slots):
        center = i + 0.5
        base = 4 * i

        # step_past_lower: step(center - lower) using 2 neurons
        # Unit 0: ReLU(S*center - S*lower) = ReLU(-S*lower + S*center)
        input_proj[base, 0] = -S  # reads lower
        input_bias[base] = S * center
        # Unit 1: ReLU(S*center - S*lower - 1) = ReLU(-S*lower + S*center - 1)
        input_proj[base + 1, 0] = -S
        input_bias[base + 1] = S * center - 1.0

        # step_past_upper: step(center - upper) using 2 neurons
        # Unit 2: ReLU(S*center - S*upper)
        input_proj[base + 2, 1] = -S  # reads upper
        input_bias[base + 2] = S * center
        # Unit 3: ReLU(S*center - S*upper - 1)
        input_proj[base + 3, 1] = -S
        input_bias[base + 3] = S * center - 1.0

        # output_i = 2*(step_past_lower - step_past_upper) - 1
        # step_past_lower = unit0 - unit1, step_past_upper = unit2 - unit3
        # output_i = 2*((u0 - u1) - (u2 - u3)) - 1
        #          = 2*u0 - 2*u1 - 2*u2 + 2*u3 - 1
        output_proj[base, i] = 2.0
        output_proj[base + 1, i] = -2.0
        output_proj[base + 2, i] = -2.0
        output_proj[base + 3, i] = 2.0
        # output_bias[i] = -1.0 (already set)

    return linear_relu_linear(
        input_node=inp,
        input_proj=input_proj,
        input_bias=input_bias,
        output_proj=output_proj,
        output_bias=output_bias,
        name="in_range",
    )


def broadcast_select(
    masks: Node,
    true_value: Node,
    false_value: Node,
    n_slots: int,
    d_fill: int,
) -> Node:
    """Select between two values at each of N slots, based on per-slot masks.

    This is a vectorized version of select(). Each slot independently
    picks true_value or false_value based on its mask. Values can be
    broadcast (same for all slots) or per-slot (different per slot).

    Args:
        masks: Node of width n_slots. Each value is 1.0 (true) or -1.0 (false).
        true_value: Node of width d_fill (broadcast to all slots)
            or n_slots*d_fill (per-slot values).
        false_value: Same shape options as true_value.
        n_slots: Number of slots.
        d_fill: Width of the value at each slot.

    Returns:
        Node of width n_slots * d_fill.
    """
    assert len(masks) == n_slots
    true_is_broadcast = len(true_value) == d_fill
    false_is_broadcast = len(false_value) == d_fill
    assert true_is_broadcast or len(true_value) == n_slots * d_fill
    assert false_is_broadcast or len(false_value) == n_slots * d_fill

    half_big = big_offset / 2.0
    d_hidden = 2 * n_slots * d_fill
    inp = Concatenate([masks, true_value, false_value])
    d_input = len(inp)

    # Offsets into the concatenated input
    mask_offset = 0
    true_offset = n_slots
    false_offset = n_slots + len(true_value)

    input_proj = torch.zeros(d_hidden, d_input)
    input_bias = torch.zeros(d_hidden)
    output_proj = torch.zeros(d_hidden, n_slots * d_fill)
    output_bias = torch.full((n_slots * d_fill,), -half_big)

    for i in range(n_slots):
        for j in range(d_fill):
            out_idx = i * d_fill + j
            unit_a = 2 * out_idx  # routes true_value
            unit_b = 2 * out_idx + 1  # routes false_value

            # Unit A: ReLU(half_big * mask_i + true_ij)
            input_proj[unit_a, mask_offset + i] = half_big
            if true_is_broadcast:
                input_proj[unit_a, true_offset + j] = 1.0
            else:
                input_proj[unit_a, true_offset + i * d_fill + j] = 1.0

            # Unit B: ReLU(-half_big * mask_i + false_ij)
            input_proj[unit_b, mask_offset + i] = -half_big
            if false_is_broadcast:
                input_proj[unit_b, false_offset + j] = 1.0
            else:
                input_proj[unit_b, false_offset + i * d_fill + j] = 1.0

            # Output: rA + rB - half_big
            output_proj[unit_a, out_idx] = 1.0
            output_proj[unit_b, out_idx] = 1.0

    return linear_relu_linear(
        input_node=inp,
        input_proj=input_proj,
        input_bias=input_bias,
        output_proj=output_proj,
        output_bias=output_bias,
        name="broadcast_select",
    )
