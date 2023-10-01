import torch

from modelscriptor.compiler.compile import compile_ffn_network
from modelscriptor.compiler.groups.ffn_sublayer import FFNSubLayer

from modelscriptor.graph import Linear, ReLU
from modelscriptor.modelscript.arithmetic_ops import add
from modelscriptor.modelscript.inout_nodes import create_input, create_constant


def test_net1():
    input_node = create_input("test", 10)
    linear1 = Linear(input_node, torch.zeros(10, 10), torch.zeros(10))
    relu_out = ReLU(linear1)
    linear2 = Linear(relu_out, torch.zeros(10, 10), torch.zeros(10))

    ffn_sublayer = FFNSubLayer(d=20)
    ffn_sublayer.out_state.allocate_node(linear2)
    strategies = ffn_sublayer.get_strategies(linear2)
    print(strategies)
    ffn_sublayer.apply_strategy(strategies[0])
    ffn_sublayer.print()
    assert input_node in ffn_sublayer.in_state.nodes


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
    assert input_node in ffn_sublayer.in_state.nodes
    assert constant in ffn_sublayer.in_state.nodes
