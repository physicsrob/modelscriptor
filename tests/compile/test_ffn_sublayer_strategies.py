import torch

from modelscriptor.compiler.groups.ffn_sublayer import FFNSubLayer

from modelscriptor.graph import Linear, ReLU
from modelscriptor.modelscript.arithmetic_ops import add
from modelscriptor.modelscript.inout_nodes import (
    create_input,
    create_constant,
    create_pos_encoding,
)


def test_net1():
    input_node = create_input("test", 10)
    linear1 = Linear(input_node, torch.rand(10, 10), torch.zeros(10))
    relu_out = ReLU(linear1)
    linear2 = Linear(relu_out, torch.rand(10, 10), torch.zeros(10))

    ffn_sublayer = FFNSubLayer(d=20)
    ffn_sublayer.out_state.allocate_node(linear2)
    strategies = ffn_sublayer.get_strategies(linear2)
    ffn_sublayer.apply_strategy(strategies[0])
    ffn_sublayer.print()
    assert ffn_sublayer.in_state.has_node_indices(input_node)


def test_net2():
    input_node = create_input("test", 3)
    constant = create_constant(torch.ones(3))
    linear1 = Linear(input_node, torch.zeros(3, 3), torch.zeros(3))
    relu_out = ReLU(linear1)
    linear2 = Linear(relu_out, torch.zeros(3, 3), torch.zeros(3))
    add_out = add(linear2, constant)

    ffn_sublayer = FFNSubLayer(d=10)
    ffn_sublayer.out_state.allocate_node(add_out)
    strategies = ffn_sublayer.get_strategies(add_out)
    print(strategies)
    ffn_sublayer.apply_strategy(strategies[0])
    ffn_sublayer.print()
    assert ffn_sublayer.in_state.has_node_indices(input_node)
    assert ffn_sublayer.in_state.has_node_indices(constant)


def test_input_pass_through():
    # Test that input node is passed through FFN layer
    input_node = create_input("test", 1)
    ffn_sublayer = FFNSubLayer(d=10)
    ffn_sublayer.out_state.allocate_node(input_node)
    strategies = ffn_sublayer.get_strategies(input_node)
    print(strategies)
    ffn_sublayer.apply_strategy(strategies[0])
    ffn_sublayer.print()
    assert ffn_sublayer.in_state.has_node_indices(input_node)


def test_posencoding_pass_through():
    # Test that pos encoding is passed through FFN layer
    pos_encoding = create_pos_encoding()
    ffn_sublayer = FFNSubLayer(d=30)
    ffn_sublayer.out_state.allocate_node(pos_encoding)
    strategies = ffn_sublayer.get_strategies(pos_encoding)
    print(strategies)
    ffn_sublayer.apply_strategy(strategies[0])
    ffn_sublayer.print()
    assert ffn_sublayer.in_state.has_node_indices(pos_encoding)
