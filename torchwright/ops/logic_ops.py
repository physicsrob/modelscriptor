from typing import List

from torchwright.graph import Node, Add, Concatenate
from torchwright.ops.linear_relu_linear import linear_relu_linear

import torch

from torchwright.ops.arithmetic_ops import sum_nodes, compare
from torchwright.ops.const import (
    step_sharpness,
    embedding_step_sharpness,
    big_offset,
)


def bool_any_true(inp_list: List[Node]) -> Node:
    """
    Returns a node that evaluates to True if any of the input nodes are true.

    Args:
        inp_list (List[Node]): List of nodes to be evaluated.

    Returns:
        Node: Output node that is True if any input nodes are true, otherwise False.
    """
    # Strategy:
    # Convert all the values to 1.0 if they're > 0.0 and 0.0 otherwise
    # then sum them, and if the sum is > 0.5, return 1.0, otherwise -1.0
    sum_node = sum_nodes(
        [compare(n, thresh=0.0, true_level=1.0, false_level=0.0) for n in inp_list]
    )
    return compare(sum_node, thresh=0.5, true_level=1.0, false_level=-1.0)


def bool_all_true(inp_list: List[Node]) -> Node:
    """
    Returns a node that evaluates to True if all of the input nodes are true.

    Inputs must be clean ±1.0 booleans (as produced by compare/bool_* ops).
    Sum of N such inputs is +N only when all are +1; otherwise ≤ N-2.
    A threshold at N-1 cleanly separates the two cases.

    Args:
        inp_list (List[Node]): List of nodes to be evaluated.

    Returns:
        Node: Output node that is True if all input nodes are true, otherwise False.
    """
    return compare(
        sum_nodes(inp_list),
        thresh=len(inp_list) - 1.0,
        true_level=1.0,
        false_level=-1.0,
    )


def bool_not(inp: Node) -> Node:
    """
    Returns a node that evaluates to 1.0 if the input node is false, and -1.0 if the input node is true.

    Args:
        inp: Input node to be evaluated

    Returns:
        Node: Output node that is 1.0 if the input node is false, and -1.0 if the input node is true.
    """
    return compare(inp, thresh=0.0, true_level=-1.0, false_level=1.0)


def equals_vector(inp: Node, vector: torch.Tensor) -> Node:
    """
    Compares a node's value to a vector tensor.

    Args:
        inp (Node): The node to be compared.
        vector (torch.Tensor): The vector tensor for comparison.

    Returns:
        Node: Node with the result of the comparison.
    """
    # If value1 == c, result is 1
    # else result is -1
    # We'll use an MLP:
    # y = 2.0*speed * max(1.0/speed + c @ value - c @ c, 0) - 1.0
    # d_hidden = 1
    speed = embedding_step_sharpness
    input_proj = vector.unsqueeze(0)  # We're dotting vector into value
    input_bias = 1.0 / speed - vector @ vector
    output_proj = torch.tensor([[2.0 * speed]])
    output_bias = torch.tensor([-1.0])
    return linear_relu_linear(
        input_node=inp,
        input_proj=input_proj,
        input_bias=input_bias,
        output_proj=output_proj,
        output_bias=output_bias,
    )


def cond_add_vector(
    cond: Node, inp: Node, true_vector: torch.Tensor, false_vector: torch.Tensor
) -> Node:
    """
    Conditionally adds a vector to the input node based on the value of the condition node.

    If the value from the `cond` node is true, this function adds the `true_vector` to the `input_node`.
    If the value from the `cond` node is false, it adds the `false_vector` to the `input_node`.

    Args:
        cond (Node): A boolean input node that determines which vector gets added.
        inp (Node): The node whose values are to be modified based on the condition.
        true_vector (torch.Tensor): The vector to add if the condition is true.
        false_vector (torch.Tensor): The vector to add if the condition is false.

    Returns:
        Node: A new node with the modified values based on the condition and input vectors.
    """
    assert len(cond) == 1
    assert len(true_vector) == len(false_vector) == len(inp)

    # We need 2 MLP entries, we'll use the equation:
    # y= c * [max(step_sharpness*x, 0) - max(step_sharpness*x - 1, 0)]
    # And rely on the residual connection

    d_input = len(inp)

    input_proj = torch.tensor([[step_sharpness], [step_sharpness]])
    input_bias = torch.tensor([0.0, -1.0])
    output_proj = torch.zeros((2, d_input))
    output_bias = false_vector

    for d in range(d_input):
        output_proj[0, d] = true_vector[d] - false_vector[d]
        output_proj[1, d] = -(true_vector[d] - false_vector[d])

    return Add(
        inp,
        linear_relu_linear(
            input_node=cond,
            input_proj=input_proj,
            input_bias=input_bias,
            output_proj=output_proj,
            output_bias=output_bias,
        ),
    )


def cond_gate(cond: Node, inp: Node) -> Node:
    """
    Gates the value of a node based on a condition. If the condition is true,
    outputs the value. If false, outputs a zero tensor of the same shape as value.

    Args:
        cond (Node): Condition node.
        inp (Node): The node whose value is to be gated.

    Returns:
        Node: Output node after applying the gate based on condition.
    """
    assert len(cond) == 1

    d = len(inp)
    # Fused single L->ReLU->L reading [cond, inp]:
    #   unit_a[j] = ReLU( big_offset * cond + inp[j])  -- alive when cond=+1
    #   unit_b[j] = ReLU(-big_offset * cond)            -- alive when cond=-1
    #   out[j]    = unit_a[j] + unit_b[j] - big_offset
    #
    # cond=+1: out = inp,  cond=-1: out = 0
    d_hidden = 2 * d
    input_proj = torch.zeros(d_hidden, 1 + d)
    input_bias = torch.zeros(d_hidden)
    output_proj = torch.zeros(d_hidden, d)
    output_bias = torch.full((d,), -big_offset)

    for j in range(d):
        a = j
        b = d + j
        input_proj[a, 0] = big_offset
        input_proj[a, 1 + j] = 1.0
        input_proj[b, 0] = -big_offset
        output_proj[a, j] = 1.0
        output_proj[b, j] = 1.0

    x = Concatenate([cond, inp])
    return linear_relu_linear(
        input_node=x,
        input_proj=input_proj,
        input_bias=input_bias,
        output_proj=output_proj,
        output_bias=output_bias,
        name="cond_gate",
    )
