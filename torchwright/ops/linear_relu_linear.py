from torchwright.graph import Node, Linear

import torch

from torchwright.graph.relu import ReLU


def linear_relu_linear(
    input_node: Node,
    input_proj: torch.Tensor,
    input_bias: torch.Tensor,
    output_proj: torch.Tensor,
    output_bias: torch.Tensor,
    name: str = "",
) -> Node:
    """Build a ``Linear -> ReLU -> Linear`` subgraph.

    This is the fundamental building block for piecewise-linear
    functions in the computation graph. The compiler maps each call
    to one FFN sublayer in the compiled transformer.

    Args:
        input_node: Upstream node whose output feeds into the first Linear.
        input_proj: First-layer weight matrix, shape ``(d_intermediate, d_input)``.
        input_bias: First-layer bias, shape ``(d_intermediate,)``.
        output_proj: Second-layer weight matrix, shape ``(d_intermediate, d_output)``.
        output_bias: Second-layer bias, shape ``(d_output,)``.
        name: Label prefix for the sub-nodes (for debugging).

    Returns:
        The final Linear node (output of the two-layer subgraph).
    """
    if len(input_proj.shape) == 1:
        input_proj = input_proj.unsqueeze(0)
    if len(input_bias.shape) == 0:
        input_bias = input_bias.unsqueeze(0)
    if len(output_proj.shape) == 1:
        output_proj = output_proj.unsqueeze(0)

    d_int = input_proj.shape[0]
    d_input = input_proj.shape[1]

    assert input_proj.shape == (d_int, d_input)
    assert input_bias.shape == (d_int,)

    d_output = output_proj.shape[1]
    assert output_proj.shape == (d_int, d_output)
    assert output_bias.shape == (d_output,)

    linear1 = Linear(input_node, input_proj.t(), input_bias, name=f"{name}_linear1")
    relu_out = ReLU(linear1, name=f"{name}_relu")
    linear2 = Linear(relu_out, output_proj, output_bias, name=f"{name}_linear2")
    return linear2
