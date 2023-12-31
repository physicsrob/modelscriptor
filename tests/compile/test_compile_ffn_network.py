import inspect
from typing import Dict, Optional

import torch

from modelscriptor.compiler.compile import compile_network
from modelscriptor.compiler.groups.ffn_sublayer import FFNSubLayer

# from modelscriptor.compiler.plan.layer_plan import FFNPlan
from modelscriptor.graph import Linear, ReLU, Node, PosEncoding
from modelscriptor.modelscript.arithmetic_ops import add, add_scalar, relu_add, relu
from modelscriptor.modelscript.inout_nodes import (
    create_input,
    create_constant,
    create_pos_encoding,
)
from modelscriptor.modelscript.logic_ops import cond_add_vector
from modelscriptor.modelscript.map_select import select


def current_test_name():
    stack = inspect.stack()
    for frame in stack:
        if frame.function.startswith("test_"):
            return frame.function
    return "unknown"


def compiler_test(
    output_node: Node,
    n_pos: int,
    input_values: Dict[str, torch.Tensor],
    d: int = 256,
    pos_encoding: Optional[PosEncoding] = None,
):
    net = compile_network(
        d, 64, output_node, report_name=current_test_name(), pos_encoding=pos_encoding
    )
    all_pass = True

    # Run a forward pass preserving all intermediate states
    inp = net.get_input_res_stream(n_pos, input_values)
    res, states = net.forward(inp, return_states=True)

    for state_name, (res_state, x) in states.items():
        for node in res_state.get_nodes():
            indices = res_state.get_node_indices(node)
            expected_output = node.compute(n_pos, input_values)
            result = x[:, indices]
            if not torch.allclose(expected_output, result):
                all_pass = False
                print(
                    f"   Failed. Expected {expected_output}, got {result}, at {state_name}"
                )
                breakpoint()
                print("Here")
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


def test_compile_repeated_adds():
    c1 = create_constant(torch.tensor([1.0]))
    c2 = create_constant(torch.tensor([1.0]))
    c3 = create_constant(torch.tensor([1.0]))
    c4 = create_constant(torch.tensor([1.0]))
    pos = create_pos_encoding()
    a1 = add(c1, c2)
    a2 = add(c3, c4)
    a3 = add(a1, a2)
    # Not compilable because Add(Add, Add) requires one of the Adds to be compiled by the Linear->Relu->Linear,
    # which can't happen.
    # This necessitates the creation of other Add strategies (e.g. in Attention with a simple linear addition)
    # But that requires a position encoding to be accessible by the attention layer
    compiler_test(a3, n_pos=2, input_values={}, pos_encoding=pos)


def test_compile_add_relu():
    value_input1 = create_input("value1", 4)
    value_input2 = create_input("value2", 4)
    inp1 = torch.rand(2, 4) - 0.5
    inp2 = torch.rand(2, 4) - 0.5
    pos = create_pos_encoding()
    a1 = add(value_input1, value_input2)
    x = relu(a1)
    x2 = relu(x)
    # Not compilable because Add(Add, Add) requires one of the Adds to be compiled by the Linear->Relu->Linear,
    # which can't happen.
    # This necessitates the creation of other Add strategies (e.g. in Attention with a simple linear addition)
    # But that requires a position encoding to be accessible by the attention layer
    compiler_test(
        x2,
        n_pos=2,
        input_values={"value1": inp1, "value2": inp2},
        pos_encoding=pos,
    )


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


def test_compile_get_prev_value():
    input_values = torch.tensor([[1.0], [2.0], [3.0], [4.0], [5.0]])
    cond_values = torch.tensor([[1.0], [0.0], [0.0], [1.0], [0.0]])

    value_input = create_input("value", 1)
    cond_input = create_input("cond", 1)
    pos_encoding = create_pos_encoding()
    last_input = pos_encoding.get_prev_value(value_input, cond_input)
    compiler_test(
        last_input, n_pos=5, input_values={"value": input_values, "cond": cond_values}
    )


def test_compile_get_last_value():
    input_values = torch.tensor([[1.0], [2.0], [3.0], [4.0], [5.0]])
    value_input = create_input("value", 1)
    pos_encoding = create_pos_encoding()
    last_input = pos_encoding.get_last_value(value_input, delta_pos=-1)
    compiler_test(last_input, n_pos=5, input_values={"value": input_values})
