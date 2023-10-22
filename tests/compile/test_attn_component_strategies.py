from typing import Set

import torch

from modelscriptor.compiler.components.attn import AttnLayerComponent
from modelscriptor.compiler.components.component import Component
from modelscriptor.compiler.feature_assignment import (
    FeatureAssignmentConstraints,
    solve,
)
from modelscriptor.compiler.groups.attn_sublayer import AttnSubLayer
from modelscriptor.graph import Linear, Node
from modelscriptor.modelscript.arithmetic_ops import add, sum_nodes
from modelscriptor.modelscript.inout_nodes import (
    create_input,
    create_constant,
    create_pos_encoding,
)


def compile_component(component: Component, output_node: Node) -> Set[Node]:
    constraint = FeatureAssignmentConstraints()
    constraint.add_node_to_state(output_node, component.out_state)
    strategies = component.get_strategies(output_node)
    print(f"{len(strategies)} strategies.")
    # for strategy in strategies:
    #     strategy.print()

    strategy = strategies[0]

    # print("Best strategy:")
    # strategy.print()
    constraint.update(component.get_constraints_for_strategy(strategy))
    feature_assignment = solve(constraint)
    component.apply_strategy(feature_assignment, strategy)
    return feature_assignment.get_nodes(component.in_state)


def test_pass_through():
    # Input node on output should be passed through to input.
    input_node = create_input("test", 1)
    pos = create_pos_encoding()
    attn = AttnLayerComponent(d=128, d_head=64, pos_encoding=pos)
    result = compile_component(attn, input_node)
    assert input_node in result


def test_add():
    # Input node on output should be passed through to input.
    input_node1 = create_input("test1", 1)
    input_node2 = create_input("test2", 1)
    pos = create_pos_encoding()
    a1 = sum_nodes([input_node1, input_node2])
    attn = AttnLayerComponent(d=128, d_head=64, pos_encoding=pos)
    result = compile_component(attn, a1)
    assert input_node1 in result
    assert input_node2 in result
