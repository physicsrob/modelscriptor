from typing import List

from modelscriptor.graph import Node, Add, Concatenate, Linear
import torch

from modelscriptor.graph.relu import ReLU
from modelscriptor.modelscript.ffn_layer import ffn_layer

from modelscriptor.modelscript.const import turn_on_speed, big_offset


def add_scalar(inp: Node, scalar: float) -> Node:
    """
    Adds a scalar value to each entry of the input node.

    Args:
        inp (Node): Node whose values will have the scalar added.
        scalar (float): Scalar value to add.

    Returns:
        Node: Output node with scalar added to each entry.
    """
    return Add(
        inp,
        ffn_layer(
            input_node=inp,
            input_proj=torch.tensor([0.0] * len(inp)),
            input_bias=torch.zeros(1),
            output_proj=torch.tensor([0.0] * len(inp)),
            output_bias=torch.tensor([scalar] * len(inp)),
        ),
    )


def add(inp1: Node, inp2: Node) -> Node:
    """
    Performs element-wise addition of two input nodes.

    Args:
        inp1 (Node): First node for addition.
        inp2 (Node): Second node for addition.

    Returns:
        Node: Node resulting from element-wise addition.
    """
    return Add(inp1, inp2)


def compare(
    inp: Node, thresh: float, true_level: float = 1.0, false_level: float = -1.0
) -> Node:
    """
    Compare input with threshold and return boolean valued node (1.0 for true, -1.0 for false)

    Args:
        inp: Node to compare. Must be length 1.
        thresh: Threshold to use.
        true_level: Value to return if inp is greater than thresh.
        false_level: Value to return if inp is less than thresh.


    Returns:
        Node: Node with a value of true_level if inp is greater than thresh, false_level otherwise.
    """

    # We need 2 FFN entries, we'll use the equation:
    # y= (true_level-false_level) * [
    #   max(turn_on_speed*x - turn_on_speed*thresh, 0) - max(turn_on_speed*x - turn_on_speed*thresh - 1, 0)
    # ] + false_level

    d_input = len(inp)

    input_proj = torch.tensor([[turn_on_speed], [turn_on_speed]])
    input_bias = torch.tensor([-turn_on_speed * thresh, -turn_on_speed * thresh - 1.0])
    output_proj = torch.tensor([[true_level - false_level], [false_level - true_level]])
    output_bias = false_level * torch.ones(1)

    return ffn_layer(
        input_node=inp,
        input_proj=input_proj,
        input_bias=input_bias,
        output_proj=output_proj,
        output_bias=output_bias,
    )


def concat(inp_list: List[Node]) -> Node:
    """
    Concatenates all the Nodes in inp_list.

    Args:
        inp_list (List[Node]): List of nodes to concatenate

    Returns:
        Node: Node resulting from concatenation
    """
    return Concatenate(inp_list)


def add_scaled_nodes(c1: float, inp1: Node, c2: float, inp2: Node) -> Node:
    """
    Computes the linear combination of two nodes using specified coefficients.

    Args:
        c1 (float): Coefficient for the first node.
        inp1 (Node): First node.
        c2 (float): Coefficient for the second node.
        inp2 (Node): Second node.

    Returns:
        Node: Node resulting from the linear combination of input nodes.
    """
    assert len(inp1) == len(inp2)
    d = len(inp1)

    concat = Concatenate([inp1, inp2])
    M = torch.zeros(len(concat), d)
    for i in range(d):
        M[i, i] = c1
        M[d + i, i] = c2

    return Linear(concat, M)


def sum_nodes(inp_list: List[Node]) -> Node:
    """
    Computes the sum of all input nodes.

    Args:
        inp_list (List[Node]): List of nodes to be summed.

    Returns:
        Node: Node with the summed value of input nodes.
    """
    d_values = {len(node) for node in inp_list}
    assert len(d_values) == 1
    d = d_values.pop()
    x = Concatenate(inp_list)
    output_matrix = torch.zeros(len(x), d)
    for i in range(len(x)):
        output_matrix[i, i % d] = 1.0

    return Linear(input_node=x, output_matrix=output_matrix)


def relu(inp: Node) -> Node:
    """
    Applies the Rectified Linear Unit (ReLU) function to the input node.

    Args:
        inp (Node): Node to apply ReLU.

    Returns:
        Node: Node with ReLU applied.
    """
    return ReLU(inp)


def relu_add(inp1: Node, inp2: Node) -> Node:
    """
    Applies the ReLU function to both input nodes and then adds them together.

    Args:
        inp1 (Node): First node for ReLU and addition.
        inp2 (Node): Second node for ReLU and addition.

    Returns:
        Node: Node resulting from ReLU application and addition.
    """
    # Rectifies val1 and val2 and then adds them together.
    # Equivalent to torch.clamp(val1, min=0) + torch.clamp(val2, min=0)
    assert len(inp1) == len(inp2)
    x = Concatenate([inp1, inp2])

    input_proj = torch.eye(len(x))
    input_bias = torch.zeros(len(x))
    output_proj = torch.zeros((len(x), len(inp1)))
    output_bias = torch.zeros(len(inp1))

    for i in range(len(inp1)):
        output_proj[i, i] = 1.0
        output_proj[len(inp1) + i, i] = 1.0

    return ffn_layer(
        input_node=x,
        input_proj=input_proj,
        input_bias=input_bias,
        output_proj=output_proj,
        output_bias=output_bias,
    )
