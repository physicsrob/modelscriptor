import pytest
import torch

from torchwright.ops.arithmetic_ops import add_const, compare
from torchwright.ops.inout_nodes import create_input, create_literal_value
from torchwright.ops.loop_ops import unrolled_loop


def test_count_to_target():
    """Counter increments by 1, stops at 3 with 3 max iters."""
    counter_input = create_input("counter", 1)

    def step_fn(state):
        return {"counter": add_const(state["counter"], 1.0)}

    def done_fn(state):
        return compare(state["counter"], 2.5)  # true when counter >= 3

    state, done = unrolled_loop(
        n_iters=3,
        state={"counter": counter_input},
        step_fn=step_fn,
        done_fn=done_fn,
    )

    result = state["counter"].compute(
        n_pos=1, input_values={"counter": torch.tensor([[0.0]])}
    )
    assert result.tolist() == [[3.0]]

    done_result = done.compute(
        n_pos=1, input_values={"counter": torch.tensor([[0.0]])}
    )
    assert done_result.tolist() == [[1.0]]


def test_already_done():
    """done_fn returns true initially — state should be unchanged after 3 iters."""
    x_input = create_input("x", 1)

    def step_fn(state):
        return {"x": add_const(state["x"], 100.0)}

    def done_fn(state):
        return compare(state["x"], 0.5)  # x starts at 50, already done

    state, done = unrolled_loop(
        n_iters=3,
        state={"x": x_input},
        step_fn=step_fn,
        done_fn=done_fn,
    )

    result = state["x"].compute(
        n_pos=1, input_values={"x": torch.tensor([[50.0]])}
    )
    assert result.tolist() == [[50.0]]

    done_result = done.compute(
        n_pos=1, input_values={"x": torch.tensor([[50.0]])}
    )
    assert done_result.tolist() == [[1.0]]


def test_zero_iterations():
    """n_iters=0 returns initial state unchanged; done_fn evaluated once."""
    x_input = create_input("x", 1)

    def step_fn(state):
        return {"x": add_const(state["x"], 1.0)}

    def done_fn(state):
        return compare(state["x"], 999.0)  # never done

    state, done = unrolled_loop(
        n_iters=0,
        state={"x": x_input},
        step_fn=step_fn,
        done_fn=done_fn,
    )

    result = state["x"].compute(
        n_pos=1, input_values={"x": torch.tensor([[5.0]])}
    )
    assert result.tolist() == [[5.0]]

    done_result = done.compute(
        n_pos=1, input_values={"x": torch.tensor([[5.0]])}
    )
    assert done_result.tolist() == [[-1.0]]


def test_never_done():
    """done_fn always false — all 3 iters run, done=false at end."""
    counter_input = create_input("counter", 1)

    def step_fn(state):
        return {"counter": add_const(state["counter"], 1.0)}

    def done_fn(state):
        return compare(state["counter"], 999.0)  # never triggers

    state, done = unrolled_loop(
        n_iters=3,
        state={"counter": counter_input},
        step_fn=step_fn,
        done_fn=done_fn,
    )

    result = state["counter"].compute(
        n_pos=1, input_values={"counter": torch.tensor([[0.0]])}
    )
    assert result.tolist() == [[3.0]]

    done_result = done.compute(
        n_pos=1, input_values={"counter": torch.tensor([[0.0]])}
    )
    assert done_result.tolist() == [[-1.0]]


def test_multi_variable_freeze():
    """x increments by 1, y by 10 — both freeze together when x >= 3."""
    x_input = create_input("x", 1)
    y_input = create_input("y", 1)

    def step_fn(state):
        return {
            "x": add_const(state["x"], 1.0),
            "y": add_const(state["y"], 10.0),
        }

    def done_fn(state):
        return compare(state["x"], 2.5)  # true when x >= 3

    state, done = unrolled_loop(
        n_iters=3,
        state={"x": x_input, "y": y_input},
        step_fn=step_fn,
        done_fn=done_fn,
    )

    inputs = {"x": torch.tensor([[0.0]]), "y": torch.tensor([[0.0]])}

    x_result = state["x"].compute(n_pos=1, input_values=inputs)
    assert x_result.tolist() == [[3.0]]

    y_result = state["y"].compute(n_pos=1, input_values=inputs)
    assert y_result.tolist() == [[30.0]]

    done_result = done.compute(n_pos=1, input_values=inputs)
    assert done_result.tolist() == [[1.0]]


def test_mismatched_keys():
    """step_fn returns wrong keys — should raise ValueError."""
    x_input = create_input("x", 1)

    def step_fn(state):
        return {"wrong_key": add_const(state["x"], 1.0)}

    def done_fn(state):
        return compare(state["x"], 999.0)

    with pytest.raises(ValueError, match="mismatched keys"):
        unrolled_loop(
            n_iters=1,
            state={"x": x_input},
            step_fn=step_fn,
            done_fn=done_fn,
        )
