from modelscriptor.graph.utils import index_to_vector
from modelscriptor.modelscript.inout_nodes import create_constant, create_input
from modelscriptor.modelscript.logic_ops import compare_to_vector, cond_add_vector

import torch


def test_compare_to_vector():
    for i in range(10):
        for j in range(10):
            x = create_constant(index_to_vector(i))
            c = index_to_vector(j)
            y = compare_to_vector(x, c)
            output = y.compute(n_pos=1, input_values={})
            expected_output = torch.tensor(1.0 if i == j else -1.0)
            assert torch.allclose(output, expected_output, atol=1.0e-3)


def test_cond_add_vector():
    base_value = torch.tensor([15.0, 25.0])
    true_offset = torch.tensor([100.0, 0.0])
    false_offset = torch.tensor([0.0, 100.0])

    cond_input = create_input("cond", 1)
    x = create_constant(base_value)
    x = cond_add_vector(cond_input, x, true_offset, false_offset)
    for cond_value in [-1.0, 1.0]:
        output = x.compute(n_pos=1, input_values={"cond": torch.tensor([[cond_value]])})
        if cond_value > 0.0:
            expected_value = base_value + true_offset
        else:
            expected_value = base_value + false_offset
        assert (output == expected_value.unsqueeze(0)).all()
