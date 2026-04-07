from typing import Callable, Dict, Tuple

from torchwright.graph import Node
from torchwright.ops.map_select import select


def unrolled_loop(
    n_iters: int,
    state: Dict[str, Node],
    step_fn: Callable[[Dict[str, Node]], Dict[str, Node]],
    done_fn: Callable[[Dict[str, Node]], Node],
) -> Tuple[Dict[str, Node], Node]:
    """Unroll a fixed number of iterations with conditional state freezing.

    At each iteration, evaluates done_fn on the current state *before* calling
    step_fn.  If done, select keeps the old state and the step output is
    discarded.  Once done becomes true it stays true (sticky).

    Args:
        n_iters: Number of iterations to unroll (>= 0).
        state: Dict of named state nodes.
        step_fn: Takes current state dict, returns new state dict with same keys.
        done_fn: Takes current state dict, returns boolean Node (1.0 = done).

    Returns:
        (final_state, final_done) — state dict after all iterations and a
        boolean Node indicating whether the loop terminated early.
    """
    if n_iters < 0:
        raise ValueError(f"n_iters must be >= 0, got {n_iters}")

    keys = set(state.keys())
    done = done_fn(state)

    for i in range(n_iters):
        new_state = step_fn(state)

        if set(new_state.keys()) != keys:
            raise ValueError(
                f"step_fn returned mismatched keys at iteration {i}: "
                f"expected {keys}, got {set(new_state.keys())}"
            )

        # Freeze: done=true keeps old state, done=false takes new state
        state = {
            k: select(cond=done, true_node=state[k], false_node=new_state[k])
            for k in state
        }

        # Sticky done: once true, stays true
        new_done = done_fn(state)
        done = select(cond=done, true_node=done, false_node=new_done)

    return state, done
