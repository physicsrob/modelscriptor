from modelscriptor.graph import PosEncoding, InputNode
from modelscriptor.graph.pos_encoding import get_pos_delta_matrix
import torch

atol = 1.0e-4  # Absolute tolerance for comparing tensors

d_pos = 32
N_pos = 16

pos_encoding_node = PosEncoding(d_pos)
pos_encoding = pos_encoding_node.compute(n_pos=N_pos, input_values={})


def test_get_prev_value():
    input_values = torch.tensor([[1.0], [2.0], [3.0], [4.0], [5.0]])
    cond_values = torch.tensor([[1.0], [0.0], [0.0], [1.0], [0.0]])
    expected_prev_values = torch.tensor([[1.0], [1.0], [1.0], [4.0], [4.0]])

    value_input = InputNode("value", 1)
    cond_input = InputNode("cond", 1)
    pos_encoding = PosEncoding(16)
    last_input = pos_encoding.get_prev_value(value_input, cond_input)
    output = last_input.compute(
        n_pos=5, input_values={"value": input_values, "cond": cond_values}
    )
    assert torch.allclose(output, expected_prev_values)


def test_get_last_value():
    input_values = torch.tensor([[1.0], [2.0], [3.0], [4.0], [5.0]])
    value_input = InputNode("value", 1)
    pos_encoding = PosEncoding(16)
    last_input = pos_encoding.get_last_value(value_input, delta_pos=-1)
    output = last_input.compute(n_pos=5, input_values={"value": input_values})
    assert torch.allclose(output[:, 1:], input_values[:, :-1])


def test_delta_matrix():
    delta_matrix = get_pos_delta_matrix(delta_pos=1, d=d_pos)
    delta_pos_encoding = pos_encoding @ delta_matrix
    assert torch.allclose(delta_pos_encoding[1:, :], pos_encoding[:-1, :], atol=atol)


def test_delta_matrix2():
    delta_matrix = get_pos_delta_matrix(delta_pos=2, d=d_pos)
    delta_pos_encoding = pos_encoding @ delta_matrix
    assert torch.allclose(delta_pos_encoding[2:, :], pos_encoding[:-2, :], atol=atol)


def test_delta_matrix_neg():
    delta_matrix = get_pos_delta_matrix(delta_pos=-1, d=d_pos)
    delta_pos_encoding = pos_encoding @ delta_matrix
    assert torch.allclose(delta_pos_encoding[:-1, :], pos_encoding[1:, :], atol=atol)

    assert torch.allclose(delta_pos_encoding[1], pos_encoding[2], atol=atol)
