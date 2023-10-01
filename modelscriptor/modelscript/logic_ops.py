from typing import List

from modelscriptor.graph import Node, Add
from modelscriptor.modelscript.ffn_layer import ffn_layer

import torch

from modelscriptor.modelscript.arithmetic_ops import relu
from modelscriptor.modelscript.const import turn_on_speed, big_offset


def bool_any_true(inp_list: List[Node]) -> Node:
    """
    Returns a node that evaluates to True if any of the input nodes are true.

    Args:
        inp_list (List[Node]): List of nodes to be evaluated.

    Returns:
        Node: Output node that is True if any input nodes are true, otherwise False.
    """
    raise NotImplementedError()


def bool_all_true(inp_list: List[Node]) -> Node:
    """
    Returns a node that evaluates to True if all of the input nodes are true.

    Args:
        inp_list (List[Node]): List of nodes to be evaluated.

    Returns:
        Node: Output node that is True if all input nodes are true, otherwise False.
    """
    raise NotImplementedError()


def compare_to_vector(inp: Node, vector: torch.Tensor) -> Node:
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
    # We'll use a FFN:
    # y = 2.0*turn_on_speed * max(1.0/turn_on_speed + c @ value - c @ c, 0) - 1.0
    # d_int = 1
    input_proj = vector.unsqueeze(0)  # We're dotting vector into value
    input_bias = 1.0 / turn_on_speed - vector @ vector
    output_proj = torch.tensor([[2.0 * turn_on_speed]])
    output_bias = torch.tensor([-1.0])
    return ffn_layer(
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

    Parameters:
    - cond (Node): A boolean input node that determines which vector gets added to the `input_node`.
    - inp (Node): The node whose values are to be modified based on the condition.
    - true_vector (torch.Tensor): The vector to add if the condition is true.
    - false_vector (torch.Tensor): The vector to add if the condition is false.

    Returns:
    - Node: A new node with the modified values based on the condition and input vectors.
    """
    assert len(cond) == 1
    assert len(true_vector) == len(false_vector) == len(inp)

    # We need 2 FFN entries, we'll use the equation:
    # y= c * [max(turn_on_speed*x, 0) - max(turn_on_speed*x - 1, 0)]
    # And rely on the residual connection

    d_input = len(inp)

    input_proj = torch.tensor([[turn_on_speed], [turn_on_speed]])
    input_bias = torch.tensor([0.0, -1.0])
    output_proj = torch.zeros((2, d_input))
    output_bias = false_vector

    for d in range(d_input):
        output_proj[0, d] = true_vector[d] - false_vector[d]
        output_proj[1, d] = -(true_vector[d] - false_vector[d])

    return Add(
        inp,
        ffn_layer(
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
    # Strategy:
    # Add offset if cond is true, -offset if cond is false
    # Rectify
    # Add -offset if cond is true, 0 if cond is false
    x = cond_add_vector(
        cond=cond,
        inp=inp,
        true_vector=torch.tensor([big_offset] * len(inp)),
        false_vector=torch.tensor([big_offset] * len(inp)),
    )
    x = relu(x)
    x = cond_add_vector(
        cond=cond,
        inp=x,
        true_vector=torch.tensor([-big_offset] * len(inp)),
        false_vector=torch.zeros(len(inp)),
    )
    return x
