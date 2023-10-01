from modelscriptor.graph import Node, Linear

import torch

from modelscriptor.graph.relu import ReLU


def ffn_layer(
    input_node: Node,
    input_proj: torch.Tensor,
    input_bias: torch.Tensor,
    output_proj: torch.Tensor,
    output_bias: torch.Tensor,
) -> Node:
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

    linear1 = Linear(input_node, input_proj.t(), input_bias)
    relu_out = ReLU(linear1)
    linear2 = Linear(relu_out, output_proj, output_bias)
    return linear2
