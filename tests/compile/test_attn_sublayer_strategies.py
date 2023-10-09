import torch

from modelscriptor.compiler.groups.attn_sublayer import AttnSubLayer
from modelscriptor.modelscript.inout_nodes import (
    create_input,
    create_constant,
    create_pos_encoding,
)


def test_pass_through():
    # Input node on output should be passed through to input.
    input_node = create_input("test", 1)
    attn_sublayer = AttnSubLayer(d=128)
    attn_sublayer.out_state.allocate_node(input_node)
    strategies = attn_sublayer.get_strategies(input_node)
    print(strategies)
    attn_sublayer.apply_strategy(strategies[0])
    assert attn_sublayer.in_state.has_node(input_node)


def test_zero():
    # Layer should be able to compile zero constants
    zero_node = create_constant(torch.zeros(8))
    attn_sublayer = AttnSubLayer(d=128)
    attn_sublayer.out_state.allocate_node(zero_node)
    strategies = attn_sublayer.get_strategies(zero_node)
    attn_sublayer.apply_strategy(strategies[0])
    assert not attn_sublayer.in_state.has_node(zero_node)


def test_attn():
    value_input = create_input("value", 1)
    pos_encoding = create_pos_encoding()
    last_input = pos_encoding.get_last_value(value_input, delta_pos=-1)

    attn_sublayer = AttnSubLayer(d=128)
    attn_sublayer.out_state.allocate_node(last_input)
    strategies = attn_sublayer.get_strategies(last_input)
    attn_sublayer.apply_strategy(strategies[0])
    assert not attn_sublayer.in_state.has_node(last_input)
    assert attn_sublayer.in_state.has_node(value_input)
