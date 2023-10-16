import torch

from modelscriptor.compiler.groups.attn_sublayer import AttnSubLayer
from modelscriptor.graph import Linear
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
    assert attn_sublayer.in_state.has_node_indices(input_node)


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
    assert attn_sublayer.in_state.has_node_indices(added)


def test_posencoding_pass_through():
    # Test that pos encoding is passed through the attention layer
    pos_encoding = create_pos_encoding()
    attn_sublayer = AttnSubLayer(d=128)
    attn_sublayer.out_state.allocate_node(pos_encoding)
    strategies = attn_sublayer.get_strategies(pos_encoding)
    print(strategies)
    attn_sublayer.apply_strategy(strategies[0])
    assert attn_sublayer.in_state.has_node_indices(pos_encoding)


def test_attn():
    value_input = create_input("value", 1)
    pos_encoding = create_pos_encoding()
    last_input = pos_encoding.get_last_value(value_input, delta_pos=-1)

    attn_sublayer = AttnSubLayer(d=128)
    attn_sublayer.out_state.allocate_node(last_input)
    strategies = attn_sublayer.get_strategies(last_input)
    attn_sublayer.apply_strategy(strategies[0])
    assert not attn_sublayer.in_state.has_node_indices(last_input)
    assert attn_sublayer.in_state.has_node_indices(value_input)


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

    attn_sublayer = AttnSubLayer(d=128, pos_encoding=pos_encoding)
    attn_sublayer.out_state.allocate_node(transformed)
    strategies = attn_sublayer.get_strategies(transformed)
    print("Strategies found:")
    for s in strategies:
        print(s.sub_strategies)
        attn_sublayer.print_strategy(s)
        print("Score: ", s.get_score())
        # s.print([sublayer], [sublayer_type])
    attn_sublayer.apply_strategy(strategies[0])
    assert attn_sublayer.in_state.has_node_indices(value_input)
    assert not attn_sublayer.in_state.has_node_indices(transformed)

    out_indices = attn_sublayer.out_state.get_node_indices(transformed)
    in_indices = attn_sublayer.in_state.get_node_indices(value_input)
    in_pos_indices = attn_sublayer.in_state.get_node_indices(pos_encoding)
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

    attn_sublayer = AttnSubLayer(d=128, pos_encoding=pos_encoding)
    attn_sublayer.out_state.allocate_node(added)
    strategies = attn_sublayer.get_strategies(added)
    print("Strategies found:")
    for s in strategies:
        print(s.sub_strategies)
        attn_sublayer.print_strategy(s)
        print("Score: ", s.get_score())
        # s.print([sublayer], [sublayer_type])
    attn_sublayer.apply_strategy(strategies[0])
    assert attn_sublayer.in_state.has_node_indices(
        value_input1
    ) and attn_sublayer.in_state.has_node_indices(value_input2)
    assert not attn_sublayer.in_state.has_node_indices(added)

    out_indices = attn_sublayer.out_state.get_node_indices(added)
    in_indices1 = attn_sublayer.in_state.get_node_indices(value_input1)
    in_indices2 = attn_sublayer.in_state.get_node_indices(value_input2)
    in_pos_indices = attn_sublayer.in_state.get_node_indices(pos_encoding)
    inp = torch.zeros((n_pos, 128))
    inp[:, in_indices1] = input_value1
    inp[:, in_indices2] = input_value2
    inp[:, in_pos_indices] = pos_encoding.get_pos_encoding(n_pos)
    x = attn_sublayer.forward(inp)
    result = x[:, out_indices]
    assert torch.allclose(result, expected_output)
