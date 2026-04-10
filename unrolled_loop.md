# `unrolled_loop` Op

## Motivation

The DDA raycasting algorithm steps a ray through a grid until it hits a wall.
This requires ~8 state variables (ray position, side distances, grid cell, hit
flag, wall distance, etc.) that all must freeze together once the hit condition
fires. Today, this "iterate with early termination" pattern must be hand-wired
with per-variable `select` calls at every iteration — error-prone and noisy.

`unrolled_loop` abstracts this into a single call: give it initial state, a step
function, a done condition, and an iteration count, and it unrolls the graph
nodes with correct freeze-on-done semantics for all state variables.

## Design

### Signature

```python
def unrolled_loop(
    n_iters: int,
    state: Dict[str, Node],
    step_fn: Callable[[Dict[str, Node]], Dict[str, Node]],
    done_fn: Callable[[Dict[str, Node]], Node],
) -> Tuple[Dict[str, Node], Node]:
```

### Parameters

- **`n_iters`**: number of iterations to unroll (>= 0). This is a
  graph-construction-time constant — the loop is fully unrolled into nodes.
- **`state`**: initial state as a dict mapping variable names to Nodes.
- **`step_fn`**: takes the current state dict, returns a new state dict with the
  same keys. Called once per unrolled iteration (always creates nodes, even when
  done — suppressed by `select`).
- **`done_fn`**: takes the current state dict, returns a scalar boolean Node
  (1.0 = done, -1.0 = not done).

### Returns

`(final_state, final_done)` — the state dict after all iterations, and a
boolean Node indicating whether the loop terminated early.

Returning the done flag is important: for DDA, it tells the caller whether the
ray actually hit a wall within the max iterations.

### Semantics

- **Done checked before step**: each iteration evaluates `done_fn` on the
  current state before calling `step_fn`. If done, `select` keeps the old state
  and the step output is discarded.
- **Sticky done**: once done becomes true, it stays true. Implemented by
  `select(done, done, new_done)` — when done=1.0, the true branch returns
  done (1.0), so it can never revert.
- **All variables freeze together**: the same done flag gates every state
  variable via `select`.

### Implementation

```python
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
```

## Design Decisions

**Dict vs List for state**: DDA has 8+ named variables. Positional `List[Node]`
would be unreadable. `Dict[str, Node]` is new for op signatures but justified
by the complexity of the use case.

**Done before step (not after)**: if the initial state is already "done" (e.g.,
player starts inside a wall), no stepping should occur.

**Suppressed nodes are acceptable**: `step_fn` always creates graph nodes even
for iterations after done. Those nodes exist in the compiled transformer but
their outputs are masked by `select`. This is unavoidable in a static graph and
matches the existing carry-chain pattern.

## Cost Model

Each unrolled iteration adds:

- 1 `done_fn` call (typically 1 MLP sublayer for a `compare`)
- 1 `step_fn` call (user-defined; for DDA this is several MLP sublayers)
- N `select` calls for N state variables (1 MLP sublayer each)
- 1 `select` for the sticky done flag (1 MLP sublayer)

**Overhead per iteration**: N + 2 MLP sublayers beyond the step function cost.

For DDA with 8 state variables and ~16 max iterations (diagonal of 8x8 grid):
~16 x (8 + 2 + step_cost) layers. With step_cost ~5-10 for DDA, total is
~200-300 layers.

## File Location

`torchwright/ops/loop_ops.py` — new file. This is a control flow combinator,
not arithmetic or logic. Imports `select` from `map_select.py`.

## Tests

File: `tests/ops/test_loop_ops.py`

1. **Count to target**: counter increments by 1, stops at 3 with 10 max iters.
   Verify counter=3 and done=true.
2. **Already done**: done_fn returns true initially. Verify state unchanged
   after 5 iterations.
3. **Zero iterations**: n_iters=0. Verify initial state returned unchanged.
4. **Never done**: done_fn always false. Verify all iterations run (counter=5
   after 5 iters) and done=false.
5. **Multi-variable freeze**: x increments by 1, y by 10, stop when x>=3.
   Verify both freeze together (x=3, y=30).
6. **Mismatched keys**: step_fn returns wrong keys. Verify ValueError raised.

## Edge Cases

| Case | Behavior |
|------|----------|
| `n_iters=0` | Returns initial state; done_fn evaluated once |
| Done from start | All iterations create nodes but select always picks old state |
| Never done | All iterations run; final done = false |
| Mismatched keys | ValueError at graph-construction time |
