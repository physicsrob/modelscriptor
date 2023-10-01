from modelscriptor.modelscript.arithmetic_ops import add_scalar, relu_add
from modelscriptor.modelscript.inout_nodes import create_input
import torch


def test_add_scalar():
    offset = 100.0
    value_input = create_input("value", 1)
    n = add_scalar(value_input, offset)
    output = n.compute(n_pos=1, input_values={"value": torch.tensor([[1.0]])})
    assert output.tolist() == [[101.0]]


def test_relu_add():
    input_val1 = create_input("val1", 3)
    input_val2 = create_input("val2", 3)
    x = relu_add(input_val1, input_val2)
    for i in range(10):
        val1 = 100.0 * (torch.rand(3) - 0.5)
        val2 = 100.0 * (torch.rand(3) - 0.5)
        output = x.compute(
            n_pos=1, input_values={"val1": val1.unsqueeze(0), "val2": val2.unsqueeze(0)}
        )
        expected_value = torch.clamp(val1, min=0) + torch.clamp(val2, min=0)
        assert (output == expected_value).all()
