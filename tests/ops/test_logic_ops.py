from torchwright.graph.asserts import assert_matches_value_type
from torchwright.graph.spherical_codes import index_to_vector
from torchwright.graph.value_type import NodeValueType, Range
from torchwright.ops.inout_nodes import create_literal_value, create_input
from torchwright.ops.logic_ops import (
    equals_vector,
    cond_add_vector,
    cond_gate,
    bool_any_true,
    bool_all_true,
    bool_not,
)

import torch


def test_equals_vector():
    for i in range(10):
        for j in range(10):
            x = create_literal_value(index_to_vector(i))
            c = index_to_vector(j)
            y = equals_vector(x, c)
            output = y.compute(n_pos=1, input_values={})
            expected_output = torch.tensor(1.0 if i == j else -1.0)
            assert torch.allclose(output, expected_output, atol=1.0e-3)


def test_cond_add_vector():
    base_value = torch.tensor([15.0, 25.0])
    true_offset = torch.tensor([100.0, 0.0])
    false_offset = torch.tensor([0.0, 100.0])

    cond_input = create_input("cond", 1)
    x = create_literal_value(base_value)
    x = cond_add_vector(cond_input, x, true_offset, false_offset)
    for cond_value in [-1.0, 1.0]:
        output = x.compute(n_pos=1, input_values={"cond": torch.tensor([[cond_value]])})
        if cond_value > 0.0:
            expected_value = base_value + true_offset
        else:
            expected_value = base_value + false_offset
        assert (output == expected_value.unsqueeze(0)).all()


def test_cond_gate():
    x = create_input("x", 1)
    x_bounded = assert_matches_value_type(
        x, NodeValueType(value_range=Range(-2.0, 2.0))
    )
    cond_input = create_input("cond", 1)
    out = cond_gate(cond_input, x_bounded)
    for cond_value in [-1.0, 1.0]:
        for x_value in [-2.0, -1.0, 0.0, 1.0, 2.0]:
            output = out.compute(
                n_pos=1,
                input_values={
                    "cond": torch.tensor([[cond_value]]),
                    "x": torch.tensor([[x_value]]),
                },
            )
            if cond_value > 0.0:
                expected_value = x_value
            else:
                expected_value = 0.0
            assert output.item() == expected_value


def test_cond_gate_exact_branch_passes_clean_cond():
    """Two-sublayer branch matches the single-sublayer branch under clean ±1 cond."""
    x = create_input("x", 1)
    x_bounded = assert_matches_value_type(
        x, NodeValueType(value_range=Range(-2.0, 2.0))
    )
    cond_input = create_input("cond", 1)
    out = cond_gate(cond_input, x_bounded, approximate=False)
    for cond_value in [-1.0, 1.0]:
        for x_value in [-2.0, -1.0, 0.0, 1.0, 2.0]:
            output = out.compute(
                n_pos=1,
                input_values={
                    "cond": torch.tensor([[cond_value]]),
                    "x": torch.tensor([[x_value]]),
                },
            )
            expected_value = x_value if cond_value > 0.0 else 0.0
            assert output.item() == expected_value


def test_cond_gate_exact_branch_preserves_small_inputs():
    """With a bounded inp range the cancellation-free branch is float-exact on pass-through."""
    x = create_input("x", 1)
    # Wrap in Assert to declare a bounded range; cond_gate reads this range to set M.
    x_bounded = assert_matches_value_type(
        x, NodeValueType(value_range=Range(-1.0, 1.0))
    )
    cond_input = create_input("cond", 1)
    out = cond_gate(cond_input, x_bounded, approximate=False)

    x_tensor = torch.tensor([[1.0e-5]])
    output = out.compute(
        n_pos=1,
        input_values={
            "cond": torch.tensor([[1.0]]),
            "x": x_tensor,
        },
    )
    # False branch on-path is structurally exact: ReLU(v) - ReLU(-v) = v with no
    # large-constant cancellation. Output should equal the input bit-for-bit.
    assert torch.equal(output, x_tensor)


def test_cond_gate_adaptive_M_uses_value_range():
    """Single-sublayer (approximate=True) branch picks M from inp.value_type.value_range,
    so small inputs survive that would be lost under the old global big_offset=1000."""
    x = create_input("x", 1)
    x_bounded = assert_matches_value_type(
        x, NodeValueType(value_range=Range(-1.0, 1.0))
    )
    cond_input = create_input("cond", 1)
    out = cond_gate(cond_input, x_bounded, approximate=True)

    small = 1.0e-5
    output = out.compute(
        n_pos=1,
        input_values={
            "cond": torch.tensor([[1.0]]),
            "x": torch.tensor([[small]]),
        },
    )
    # M=1 here, so ULP(M)≈1.2e-7; 1e-5 survives cancellation cleanly.
    assert abs(output.item() - small) < 1.0e-6


def test_cond_gate_defers_unbounded_inp():
    """Unbounded inp range returns a placeholder instead of raising."""
    from torchwright.graph.placeholders import CondGatePlaceholder

    x = create_input("x", 1)
    cond_input = create_input("cond", 1)
    result = cond_gate(cond_input, x)
    assert isinstance(result, CondGatePlaceholder)


def test_bool_any_true():
    x = create_input("x", 1)
    y = create_input("y", 1)
    z = create_input("z", 1)
    out = bool_any_true([x, y, z])
    for x_value in [-1.0, 1.0]:
        for y_value in [-1.0, 1.0]:
            for z_value in [-1.0, 1.0]:
                output = out.compute(
                    n_pos=1,
                    input_values={
                        "x": torch.tensor([[x_value]]),
                        "y": torch.tensor([[y_value]]),
                        "z": torch.tensor([[z_value]]),
                    },
                )
                expected_value = (
                    1.0 if (x_value > 0.0 or y_value > 0.0 or z_value > 0.0) else -1.0
                )
                assert output.item() == expected_value


def test_bool_all_true():
    x = create_input("x", 1)
    y = create_input("y", 1)
    z = create_input("z", 1)
    out = bool_all_true([x, y, z])
    for x_value in [-1.0, 1.0]:
        for y_value in [-1.0, 1.0]:
            for z_value in [-1.0, 1.0]:
                output = out.compute(
                    n_pos=1,
                    input_values={
                        "x": torch.tensor([[x_value]]),
                        "y": torch.tensor([[y_value]]),
                        "z": torch.tensor([[z_value]]),
                    },
                )
                expected_value = (
                    1.0 if (x_value > 0.0 and y_value > 0.0 and z_value > 0.0) else -1.0
                )
                assert output.item() == expected_value


def test_bool_not():
    x = create_input("x", 1)
    out = bool_not(x)
    for x_value in [-1.0, 1.0]:
        output = out.compute(
            n_pos=1,
            input_values={
                "x": torch.tensor([[x_value]]),
            },
        )
        expected_value = 1.0 if x_value < 0.0 else -1.0
        assert output.item() == expected_value
