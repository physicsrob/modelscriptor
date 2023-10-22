from typing import Set

import torch

from modelscriptor.compiler.feature_assignment import (
    FeatureAssignmentConstraints,
    solve,
    FeatureAssignment,
)
from modelscriptor.compiler.groups.ffn_sublayer import FFNSubLayer

from modelscriptor.graph import Linear, ReLU, Node
from modelscriptor.modelscript.arithmetic_ops import add, relu
from modelscriptor.modelscript.inout_nodes import (
    create_input,
    create_constant,
    create_pos_encoding,
)


def compile_ffn(layer: FFNSubLayer, output_node: Node) -> FeatureAssignment:
    constraint = FeatureAssignmentConstraints()
    constraint.add_node_to_state(output_node, layer.out_state)
    strategies = layer.get_strategies_for_node(output_node)
    strategy = strategies[0]
    strategy.print()
    constraint.update(layer.get_constraints(strategy))
    feature_assignment = solve(constraint)
    layer.apply_strategy(feature_assignment, strategy)
    return feature_assignment


def test_net1():
    input_node = create_input("test", 10)
    linear1 = Linear(input_node, torch.rand(10, 10), torch.zeros(10))
    relu_out = ReLU(linear1)
    linear2 = Linear(relu_out, torch.rand(10, 10), torch.zeros(10))

    ffn_sublayer = FFNSubLayer(d=20)
    feature_assignment = compile_ffn(ffn_sublayer, linear2)
    result = feature_assignment.get_nodes(ffn_sublayer.in_state)
    assert input_node in result


def test_net2():
    input_node = create_input("test", 3)
    constant = create_constant(torch.ones(3))
    linear1 = Linear(input_node, torch.zeros(3, 3), torch.zeros(3))
    relu_out = ReLU(linear1)
    linear2 = Linear(relu_out, torch.zeros(3, 3), torch.zeros(3))
    add_out = add(linear2, constant)
    ffn_sublayer = FFNSubLayer(d=10)

    feature_assignment = compile_ffn(ffn_sublayer, add_out)
    result = feature_assignment.get_nodes(ffn_sublayer.in_state)
    assert input_node in result
    assert constant in result


def test_input_pass_through():
    # Test that input node is passed through FFN layer
    input_node = create_input("test", 1)
    ffn_sublayer = FFNSubLayer(d=10)
    feature_assignment = compile_ffn(ffn_sublayer, input_node)
    result = feature_assignment.get_nodes(ffn_sublayer.in_state)
    assert input_node in result


def test_posencoding_pass_through():
    # Test that pos encoding is passed through FFN layer
    pos_encoding = create_pos_encoding()
    ffn_sublayer = FFNSubLayer(d=30)
    feature_assignment = compile_ffn(ffn_sublayer, pos_encoding)
    result = feature_assignment.get_nodes(ffn_sublayer.in_state)
    assert pos_encoding in result


def compile_ffn_multinode(layer: FFNSubLayer, nodes: Set[Node]) -> FeatureAssignment:
    constraint = FeatureAssignmentConstraints()
    for node in nodes:
        constraint.add_node_to_state(node, layer.out_state)
    strategies = layer.get_strategies(nodes)
    strategy = strategies[0]
    strategy.print()
    constraint.update(layer.get_constraints(strategy))
    feature_assignment = solve(constraint)
    layer.apply_strategy(feature_assignment, strategy)
    return feature_assignment
