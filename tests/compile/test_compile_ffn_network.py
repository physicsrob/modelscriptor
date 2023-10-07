from typing import Dict

import torch

from modelscriptor.compiler.compile import compile_ffn_network
from modelscriptor.compiler.groups.ffn_sublayer import FFNSubLayer

# from modelscriptor.compiler.plan.layer_plan import FFNPlan
from modelscriptor.graph import Linear, ReLU, Node
from modelscriptor.modelscript.arithmetic_ops import add, add_scalar, relu_add
from modelscriptor.modelscript.inout_nodes import create_input, create_constant
from modelscriptor.modelscript.logic_ops import cond_add_vector
from modelscriptor.modelscript.map_select import select


def compiler_test(
    output_node: Node, n_pos: int, input_values: Dict[str, torch.Tensor], d: int = 20
):
    net = compile_ffn_network(d, output_node)
    all_pass = True

    # Run a forward pass preserving all intermediate states
    inp = net.get_input_res_stream(n_pos, input_values)
    res, states = net.forward(inp, return_states=True)

    for state_name, (res_state, x) in states.items():
        for node in res_state.get_distinct_nodes():
            indices = res_state.get_node_indices(node)
            expected_output = node.compute(n_pos, input_values)
            result = x[:, indices]
            if not torch.allclose(expected_output, result):
                all_pass = False
                print(
                    f"   Failed. Expected {expected_output}, got {result}, at {state_name}"
                )
    assert all_pass

    final_result = net.compute(n_pos, input_values)[output_node]
    expected_output = output_node.compute(n_pos, input_values)
    assert torch.allclose(final_result, expected_output), "Final output"


def test_compile_1layer():
    input_node = create_input("test", 10)
    linear1 = Linear(input_node, torch.rand(10, 10), torch.rand(10))
    relu_out = ReLU(linear1)
    linear2 = Linear(relu_out, torch.rand(10, 10), torch.rand(10))
    # Now try using compiler
    n_pos = 2
    compiler_test(linear2, n_pos=n_pos, input_values={"test": torch.rand(n_pos, 10)})


def test_compile_add_scalar():
    offset = 100.0
    value_input = create_input("value", 1)
    n = add_scalar(value_input, offset)

    n_pos = 1
    in_value = torch.tensor([[1.0]])
    compiler_test(n, n_pos=n_pos, input_values={"value": in_value})


def test_compile_cond_add_vector():
    base_value = torch.tensor([15.0, 25.0])
    true_offset = torch.tensor([100.0, 0.0])
    false_offset = torch.tensor([0.0, 100.0])

    cond_input = create_input("cond", 1)
    x = create_constant(base_value)
    x = cond_add_vector(cond_input, x, true_offset, false_offset)
    for cond_value in [-1.0, 1.0]:
        compiler_test(x, n_pos=1, input_values={"cond": torch.tensor([[cond_value]])})


def test_compile_select():
    start = 100.0
    offset = 123.0
    cond_input = create_input("cond", 1)
    x = create_constant(torch.tensor([start]))
    x = select(cond=cond_input, true_node=add_scalar(x, offset), false_node=x)
    for cond in [1.0, -1.0]:
        compiler_test(x, n_pos=1, input_values={"cond": torch.tensor([[cond]])})


def test_compile_relu_add():
    input_val1 = create_input("val1", 3)
    input_val2 = create_input("val2", 3)
    x = relu_add(input_val1, input_val2)
    for i in range(10):
        val1 = 100.0 * (torch.rand(3) - 0.5)
        val2 = 100.0 * (torch.rand(3) - 0.5)
        compiler_test(
            x,
            n_pos=1,
            input_values={"val1": val1.unsqueeze(0), "val2": val2.unsqueeze(0)},
        )
