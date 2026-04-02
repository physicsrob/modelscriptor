from modelscriptor.modelscript.arithmetic_ops import add_scalar
from modelscriptor.modelscript.inout_nodes import (
    create_input,
    create_constant,
    create_embedding,
)
from modelscriptor.modelscript.map_select import select, map_to_table, switch

import torch


def test_select():
    start = 100.0
    offset = 123.0
    cond_input = create_input("cond", 1)
    x = create_constant(torch.tensor([start]))
    x = select(cond=cond_input, true_node=add_scalar(x, offset), false_node=x)
    for cond in [1.0, -1.0]:
        print("\n\n")
        output = x.compute(n_pos=1, input_values={"cond": torch.tensor([[cond]])})
        expected_value = start + (offset if cond > 0 else 0)
        assert output.tolist() == [[expected_value]]


def test_switch():
    """Select one of three values based on which condition is true."""
    c1 = create_input("c1", 1)
    c2 = create_input("c2", 1)
    c3 = create_input("c3", 1)
    v1 = create_constant(torch.tensor([10.0, 20.0]))
    v2 = create_constant(torch.tensor([30.0, 40.0]))
    v3 = create_constant(torch.tensor([50.0, 60.0]))
    out = switch([c1, c2, c3], [v1, v2, v3])

    # Condition 1 true
    result = out.compute(
        n_pos=1,
        input_values={
            "c1": torch.tensor([[1.0]]),
            "c2": torch.tensor([[-1.0]]),
            "c3": torch.tensor([[-1.0]]),
        },
    )
    assert torch.allclose(result, torch.tensor([[10.0, 20.0]]), atol=1e-4)

    # Condition 2 true
    result = out.compute(
        n_pos=1,
        input_values={
            "c1": torch.tensor([[-1.0]]),
            "c2": torch.tensor([[1.0]]),
            "c3": torch.tensor([[-1.0]]),
        },
    )
    assert torch.allclose(result, torch.tensor([[30.0, 40.0]]), atol=1e-4)

    # Condition 3 true
    result = out.compute(
        n_pos=1,
        input_values={
            "c1": torch.tensor([[-1.0]]),
            "c2": torch.tensor([[-1.0]]),
            "c3": torch.tensor([[1.0]]),
        },
    )
    assert torch.allclose(result, torch.tensor([[50.0, 60.0]]), atol=1e-4)


def test_map_to_table():
    table_values = {
        "x": torch.tensor([10.0, 1.0, 1.0]),
        "y": torch.tensor([10.0, 2.0, 2.0]),
        "z": torch.tensor([10.0, 3.0, 3.0]),
    }
    default_value = torch.tensor([-1.0, -1.0, -1.0])

    embedding = create_embedding(vocab=["x", "y", "z"])
    x = map_to_table(
        inp=embedding,
        key_to_value={
            embedding.get_embedding(text): value for text, value in table_values.items()
        },
        default=default_value,
    )
    for text in ["x", "y", "z", "other"]:
        output = x.compute(n_pos=1, input_values={"embedding_input": [text]})
        if text in table_values:
            assert torch.allclose(output[0], table_values[text], atol=1.0e-2)
        else:
            assert torch.allclose(output[0], default_value, atol=1.0e-2)
