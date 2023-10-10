import torch

from modelscriptor.compiler.groups.attn_sublayer import AttnSubLayer
from modelscriptor.modelscript.arithmetic_ops import add
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


def test_add_pass_through():
    # Add(input, input) should be passed through to input.
    # (This might seem like a weird test, but an earlier implementation of the skip layer
    # didn't support this)
    input_node1 = create_input("test1", 1)
    input_node2 = create_input("test2", 1)
    added = add(input_node1, input_node2)
    attn_sublayer = AttnSubLayer(d=128)
    attn_sublayer.out_state.allocate_node(added)
    strategies = attn_sublayer.get_strategies(added)
    print(strategies)
    attn_sublayer.apply_strategy(strategies[0])
    assert attn_sublayer.in_state.has_node(added)


def test_posencoding_pass_through():
    # Test that pos encoding is passed through the attention layer
    pos_encoding = create_pos_encoding()
    attn_sublayer = AttnSubLayer(d=128)
    attn_sublayer.out_state.allocate_node(pos_encoding)
    strategies = attn_sublayer.get_strategies(pos_encoding)
    print(strategies)
    attn_sublayer.apply_strategy(strategies[0])
    assert attn_sublayer.in_state.has_node(pos_encoding)


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
