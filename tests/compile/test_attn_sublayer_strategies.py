from typing import Set

import torch

from modelscriptor.compiler.feature_assignment import (
    FeatureAssignmentConstraints,
    solve,
    FeatureAssignment,
)
from modelscriptor.compiler.groups.attn_sublayer import AttnSubLayer
from modelscriptor.graph import Linear, Node
from modelscriptor.modelscript.arithmetic_ops import add
from modelscriptor.modelscript.inout_nodes import (
    create_input,
    create_constant,
    create_pos_encoding,
)


def compile_attn(layer: AttnSubLayer, output_node: Node) -> FeatureAssignment:
    constraint = FeatureAssignmentConstraints()
    constraint.add_node_to_state(output_node, layer.out_state)
    strategies = layer.get_strategies_for_node(output_node)
    strategy = strategies[0]
    print("Best strategy:")
    strategy.print()
    constraint.update(layer.get_constraints(strategy))
    feature_assignment = solve(constraint)
    feature_assignment.print()
    layer.apply_strategy(feature_assignment, strategy)
    return feature_assignment
    # return feature_assignment.get_nodes(layer.in_state)


def test_pass_through():
    # Input node on output should be passed through to input.
    input_node = create_input("test", 1)
    attn_sublayer = AttnSubLayer(d=128, d_head=64)

    feature_assignment = compile_attn(attn_sublayer, input_node)
    assert input_node in feature_assignment.get_nodes(attn_sublayer.in_state)


def test_add_compiled():
    # Add(input, input) should be compiled if there is a position encoding.
    input_node1 = create_input("test1", 1)
    input_node2 = create_input("test2", 1)
    added = add(input_node1, input_node2)
    pos = create_pos_encoding()
    attn_sublayer = AttnSubLayer(d=128, d_head=32, pos_encoding=pos)
    feature_assignment = compile_attn(attn_sublayer, added)
    assert input_node1 in feature_assignment.get_nodes(attn_sublayer.in_state)
    assert input_node2 in feature_assignment.get_nodes(attn_sublayer.in_state)


def test_add_pass_through():
    # Add(input, input) should be passed through to input if there is no position encoding.
    # The reason this gets passed through is that without a position layer, the attention
    # layer has no way of attending to the current token, so the only option is to use
    # the skip layer.
    input_node1 = create_input("test1", 1)
    input_node2 = create_input("test2", 1)
    added = add(input_node1, input_node2)
    attn_sublayer = AttnSubLayer(d=128, d_head=32)
    feature_assignment = compile_attn(attn_sublayer, added)
    assert added in feature_assignment.get_nodes(attn_sublayer.in_state)


def test_posencoding_pass_through():
    # Test that pos encoding is passed through the attention layer
    pos_encoding = create_pos_encoding()
    attn_sublayer = AttnSubLayer(d=128, d_head=32)
    feature_assignment = compile_attn(attn_sublayer, pos_encoding)
    assert pos_encoding in feature_assignment.get_nodes(attn_sublayer.in_state)


def test_attn():
    value_input = create_input("value", 1)
    pos_encoding = create_pos_encoding()
    last_input = pos_encoding.get_last_value(value_input, delta_pos=-1)

    attn_sublayer = AttnSubLayer(d=128, d_head=32)

    feature_assignment = compile_attn(attn_sublayer, last_input)
    assert last_input not in feature_assignment.get_nodes(attn_sublayer.in_state)
    assert value_input in feature_assignment.get_nodes(attn_sublayer.in_state)


def test_linear_on_attn():
    # Test the ability of an attention sublayer to compile a linear node
    value_input = create_input("value", 8)
    pos_encoding = create_pos_encoding()
    transformed = Linear(
        input_node=value_input,
        output_matrix=torch.rand(8, 8),
        output_bias=torch.zeros(8),
    )
    n_pos = 4
    input_value = torch.rand(4, 8)
    expected_output = transformed.compute(n_pos, {"value": input_value})

    attn_sublayer = AttnSubLayer(d=128, d_head=32, pos_encoding=pos_encoding)

    feature_assignment = compile_attn(attn_sublayer, transformed)
    assert value_input in feature_assignment.get_nodes(attn_sublayer.in_state)
    assert transformed not in feature_assignment.get_nodes(attn_sublayer.in_state)

    out_indices = feature_assignment.get_node_indices(
        attn_sublayer.out_state, transformed
    )
    in_indices = feature_assignment.get_node_indices(
        attn_sublayer.in_state, value_input
    )
    in_pos_indices = feature_assignment.get_node_indices(
        attn_sublayer.in_state, pos_encoding
    )
    inp = torch.zeros((n_pos, 128))
    inp[:, in_indices] = input_value
    inp[:, in_pos_indices] = pos_encoding.get_pos_encoding(n_pos)
    x = attn_sublayer.forward(inp)
    result = x[:, out_indices]
    assert torch.allclose(result, expected_output)


def test_add_on_attn():
    # Test the ability of an attention sublayer to add two variables
    # This is possible because one will be passed through an identity attention
    # and the other will pass through the skip connection.
    value_input1 = create_input("value1", 8)
    value_input2 = create_input("value2", 8)
    pos_encoding = create_pos_encoding()
    added = add(value_input1, value_input2)
    n_pos = 4
    input_value1 = torch.rand(4, 8)
    input_value2 = torch.rand(4, 8)
    expected_output = input_value1 + input_value2

    attn_sublayer = AttnSubLayer(d=128, d_head=32, pos_encoding=pos_encoding)

    feature_assignment = compile_attn(attn_sublayer, added)
    assert value_input1 in feature_assignment.get_nodes(attn_sublayer.in_state)
    assert value_input2 in feature_assignment.get_nodes(attn_sublayer.in_state)
    assert added not in feature_assignment.get_nodes(attn_sublayer.in_state)

    out_indices = feature_assignment.get_node_indices(attn_sublayer.out_state, added)
    in_indices1 = feature_assignment.get_node_indices(
        attn_sublayer.in_state, value_input1
    )
    in_indices2 = feature_assignment.get_node_indices(
        attn_sublayer.in_state, value_input2
    )
    in_pos_indices = feature_assignment.get_node_indices(
        attn_sublayer.in_state, pos_encoding
    )
    inp = torch.zeros((n_pos, 128))
    inp[:, in_indices1] = input_value1
    inp[:, in_indices2] = input_value2
    inp[:, in_pos_indices] = pos_encoding.get_pos_encoding(n_pos)
    x = attn_sublayer.forward(inp)
    result = x[:, out_indices]
    assert torch.allclose(result, expected_output)
