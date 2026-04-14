# TODO

## Move ceiling/floor into the transformer

The host currently decides ceiling vs floor per pixel (`ceil if y < center_y
else floor_c`). This is computation on the host side. The transformer should
emit ceiling/floor pixels itself — either as part of each render chunk (fill
non-wall rows within the chunk) or as a separate post-render pass.

Files: `torchwright/doom/compile.py` (step_frame render loop),
`torchwright/doom/game_graph.py` (render output)

## Clip column iteration to screen bounds

The state machine iterates col_lo..col_hi, which can include negative columns
and columns >= W (e.g., cols=[-2, 122) with W=120). The host safely skips
out-of-bounds columns, but each is still a full autoregressive step (~70ms).
At 120x100 with 4 full-screen walls, ~16 steps are wasted per frame.

Fix: clamp col_lo/col_hi to [0, W) in the graph, or have the state machine
skip out-of-bounds columns via the feedback loop.

Files: `torchwright/doom/game_graph.py` (state machine col iteration)

## Make graph-side done_flag work when N < max_walls

The graph checks `mask_sum > max_walls - 0.5`, which only fires when all
max_walls slots are masked. When the scene has fewer walls, the host
terminates via bit-count instead. The graph's done_flag computation runs on
every RENDER token but never triggers — wasted graph nodes.

Options:
- Feed N as a graph input so the threshold is dynamic
- Detect sentinel wall data (score=99) after all real walls are masked
- Accept the current host-side fix and remove the graph-side done_flag entirely

Files: `torchwright/doom/game_graph.py` (state_transitions),
`torchwright/doom/compile.py` (host-side termination)

---

# Known test flakiness & intermittent failures

The full `make test` run has several failures that look like flakiness on
the surface but resolve into three distinct failure modes with different
root causes.  Each should be triaged/fixed independently — don't bundle
them into one "fix all flakiness" pass.

## Mode A — `test_fuse_chain_of_three` (order-dependent bug, not a flake)

Test: `tests/graph/test_optimize.py::test_fuse_chain_of_three`

Passes in isolation (`make test FILE=tests/graph/test_optimize.py`),
fails in full `make test` with `assert 3 == 2`.

Root cause: `fuse_consecutive_linears` at `torchwright/graph/optimize.py`
collects all `(L1, L2)` fusion candidates in one pass (lines 56-85),
then mutates `L2` in a second loop (lines 92-125).  For an `L1→L2→L3`
chain, *both* `(L1, L2)` and `(L2, L3)` are candidates; after the first
fusion mutates `L2` in place, the second tuple now points at a mutated
node, and the outer `while True` loop in the test drives either 1 or 2
passes depending on the set-iteration order of the `for node in
all_nodes` walk.  The global node-id counter shifts during full-suite
runs (earlier tests allocate nodes), which shuffles set order, which
flips the fusion count.

Fix direction: either (a) change the outer loop to fuse one pair per
invocation and rely on the caller's retry loop for convergence, or
(b) topologically order the candidate list so upstream fusions happen
first, deterministically.  Option (b) matches the existing ABI (callers
expect a fusion count) and is probably cleaner.

Scope: ~20 lines, one file, one test.  Add a regression test that
loops through different `global_node_id` offsets to catch the
ordering dependency explicitly.

Files: `torchwright/graph/optimize.py` (fuse_consecutive_linears),
`tests/graph/test_optimize.py` (add ordering regression test).

## Mode B — pixel-precision cluster (compile-layout drift)

Tests:
- `tests/doom/test_game_graph.py::test_renders_from_angle[192]`
- `tests/doom/test_game_graph.py::test_renders_oblique_angle[20/100/160/210]`
- `tests/doom/test_game_graph.py::test_renders_off_center_oblique[3.0-2.0-20]`
- `tests/doom/test_game_graph.py::test_renders_off_center_oblique[1.0--3.0-50]`

Max pixel error sits right against the 0.45 tolerance (set in commit
`981e20a`: "tighten render tolerances to catch BSP sort regressions",
`tests/doom/test_game_graph.py:150`).  Some full-suite runs pass all of
these (seen: 634 pass, 1 fail); other runs fail 7 of them (seen on the
next run with no code changes).  The failure pattern is "max_err
observed = 0.50, threshold = 0.45" — a ~10 % drift, not a structural
error.

Suspected root cause: residual-column allocation in
`torchwright/compiler/forward/graph_analysis.py` and
`torchwright/compiler/residual_assignment.py` iterates graphs through
`set` / `dict` iteration.  `Node.__hash__` uses `node_id`, which is
monotonically assigned from a global counter — any earlier test that
allocates nodes shifts the counter, which shifts iteration order, which
shifts column assignments, which shifts compiled weight matrices enough
to drift compiled pixel values by a few percent.

Fix directions (pick one):
1. **Determinism**: sort every set/dict iteration in the compiler by
   `node_id` before consuming it, so ordering is independent of hash
   collisions.  Cleanest but needs a careful audit of every iteration
   site.
2. **Reset the counter per test** via a pytest fixture that resets
   `torchwright.graph.node.global_node_id` to 0 at session start.
   Cheap, plausibly eliminates all counter-shift flakiness at the cost
   of making `node_id` no longer globally unique across the process
   (likely fine).
3. **Relax tolerance to 0.55**: pragmatic but reopens the window that
   `981e20a` was trying to close.

Option 2 is probably the right first move — test-only change, no
product-code risk, likely fixes A and B at once.

Files: `torchwright/compiler/forward/graph_analysis.py`,
`torchwright/compiler/residual_assignment.py`,
`tests/conftest.py` (if going with the fixture approach).

## Mode C — `test_renders_in_all_four_directions[192]` (real bug, always fails)

Test: `tests/doom/test_bsp_rank_integration.py::TestBspIntegration::test_renders_in_all_four_directions[192]`

Always fails, every run.  **Not flaky** — a genuine bug.  Angles 0,
64, 128 pass cleanly; 192 produces `max pixel error 0.500` in a stable,
reproducible way.

Observed payload at the SORTED argmin (from `step_frame` debug print):
```
sort[2]: wall=[4.86,5.00,-5.00,4.86] tex=1.0 rank=2
```

Expected: the real wall at `[5, 5, -5, 5]`.  The `4.86` is 97 % × 5 +
3 % × 0 — a softmax that concentrated only to ~97 % on the correct key
and spread ~3 % across non-wall positions.  Either the BSP rank
computation produces too-close scores at angle 192, or the tiebreak
offset (`wall_index * 0.1` in `torchwright/doom/stages/wall.py::_compute_bsp_rank`)
isn't big enough to survive compiled softmax drift at this particular
orientation.

Recommended first step: use the new `assert_picked_from` annotation
(already placed at `torchwright/doom/stages/sorted.py::_argmin_and_derive`
and exposed via `module.metadata["asserts"]`) to localize.  Extend
`torchwright/debug/probe.py` to accept a `step_frame`-style input
sequence, or instrument `step_frame` to call `check_asserts_on_compiled`
after each step, so angle 192 fires the predicate with the exact row
that blended.

Fix direction once the layer is identified: likely either widen the
tiebreak offset from `0.1` to `0.3` (still fits under the `1.0` real-
rank spacing), or raise the `nonwall_sentinel` (currently 99.9) further
from the `bsp_sentinel` (99.0) so non-wall positions lose the softmax
by a larger margin.

Files: `torchwright/doom/stages/wall.py::_compute_bsp_rank`,
`torchwright/doom/stages/sorted.py::_argmin_and_derive`,
`torchwright/debug/probe.py` (optional: step_frame instrumentation hook).
