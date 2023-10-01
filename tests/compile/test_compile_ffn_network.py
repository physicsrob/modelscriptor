import torch

from modelscriptor.compiler.compile import compile_ffn_network
from modelscriptor.compiler.groups.ffn_sublayer import FFNSubLayer

# from modelscriptor.compiler.plan.layer_plan import FFNPlan
from modelscriptor.graph import Linear, ReLU
from modelscriptor.modelscript.arithmetic_ops import add, add_scalar
from modelscriptor.modelscript.inout_nodes import create_input, create_constant


def test_compile_1layer():
    input_node = create_input("test", 10)
    linear1 = Linear(input_node, torch.rand(10, 10), torch.rand(10))
    relu_out = ReLU(linear1)
    linear2 = Linear(relu_out, torch.rand(10, 10), torch.rand(10))
    # Now try using compiler
    net = compile_ffn_network(20, linear2)
    net.print()
    n_pos = 2
    in_value = torch.rand(n_pos, 10)
    # expected_output = net.compute(n_pos=2, input_values={"test": in_value})
    expected_output = linear2.compute(n_pos=n_pos, input_values={"test": in_value})
    output = net.compute(n_pos=2, input_values={"test": in_value})
    assert torch.allclose(output[linear2], expected_output)


def test_compile_add_scalar():
    offset = 100.0
    value_input = create_input("value", 1)
    n = add_scalar(value_input, offset)

    # Calculate expected_value traversing modelscript graph
    expected_output = n.compute(n_pos=1, input_values={"value": torch.tensor([[1.0]])})
    # expected_output == [[101.0]]
    net = compile_ffn_network(20, n)
    result = net.compute(1, input_values={"value": torch.tensor([[1.0]])})
    print(result)
    assert result[n] == expected_output
