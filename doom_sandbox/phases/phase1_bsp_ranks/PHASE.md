# Phase 1 — BSP Ranks

## Goal

Given a DOOM scene (`MapSubset`) and a player state (`GameState`),
compute the **BSP rank** of every wall — an integer in `[0, N)` where
`N` is the number of walls, encoding the wall's order in a BSP
traversal from the player's position. 0 = first in traversal order
(closest), `N-1` = last (farthest).

This is the smallest meaningful phase exercising the sandbox: the
prefill format, the autoregressive thinking-token mechanism,
parameterized tokens, and at least one cross-position attention pattern.

## Success criterion

The phase passes when this test passes for every fixture, at every
pose in `scene.test_poses`:

```python
def test_phase1_box_room():
    scene = load_fixture("box_room")
    for state in scene.test_poses:
        expected = expected_bsp_ranks(scene, state)

        config = setup()
        prefill = get_prefill(scene, state)
        output = run(config, prefill, forward)
        actual = extract_bsp_ranks(output)

        assert actual == expected, f"mismatch at pose ({state.x}, {state.y})"
```

Exact integer match — no tolerance, no near-equality. Quantization
through `IntSlot`s is exact; `FloatSlot` quantization rounds back to
integer at extract time.

`scene.test_poses` is a fixture-supplied list of player states picked
to land clearly off every BSP plane, so the PWL-approximated `side_P`
bits are robust against framework numerical noise. If you add new
fixtures or roll your own pose, call
`assert_pose_clear_of_planes(scene, state)` (in
`doom_sandbox.fixtures`) to surface a near-plane pose loudly rather
than as a downstream rank mismatch — `compare_const`'s deadband and
FloatSlot quantization can otherwise flip a `side_P` bit silently.

## Required functions

```python
# reference.py
def expected_bsp_ranks(scene: MapSubset, state: GameState) -> list[int]:
    """Pure-Python ground truth, in ascending wall_index order."""

# extract.py
def extract_bsp_ranks(output: RunOutput) -> list[int]:
    """Pull computed BSP ranks from the autoregressive run output, in
    ascending wall_index order."""

# setup.py
def setup() -> Config: ...

# prefill.py
def get_prefill(scene: MapSubset, state: GameState) -> list[Token]: ...

# forward.py
def forward(input_vec: Vec, past: Past) -> ForwardOutput: ...
```

The reference function is straightforward — see `MapSubset`'s docstring
for the formula:

```
rank(s) = round(seg_bsp_coeffs[s] · side_P_vec + seg_bsp_consts[s])
```

where `side_P_vec[i] ∈ {0, 1}` records whether the player is on the
FRONT side of BSP plane `i` (`sign(nx_i·player_x + ny_i·player_y + d_i) > 0`).

## Scope

**In phase 1:**
- BSP plane data (`MapSubset.bsp_nodes`).
- Per-wall BSP coefficients (`MapSubset.seg_bsp_coeffs`) — these are
  always integer-valued (counts of walls in subtrees).
- Per-wall BSP constants (`MapSubset.seg_bsp_consts`) — also integer.
- Player position (`GameState.x, .y`).
- One identifier emission per wall: `BSP_RANK` carrying **both the
  wall index and the integer rank**. The emission must be
  self-describing so `extract_bsp_ranks` can sort by wall index
  without depending on emission order. Either a single
  `BSP_RANK(wall_index, rank)` token or a marker+VALUE pair like
  `BSP_RANK(wall_index) → VALUE(rank)` works; the load-bearing
  property is that the wall index travels with the value.

**Deferred to later phases:**
- Wall geometry (`ax`, `ay`, `bx`, `by`, `tex_id`) — not needed for rank.
- Other thinking identifiers (`IS_RENDERABLE`, `CROSS_*`, `DOT_*`, etc.).
- `SORTED` stage (picking walls in BSP order).
- `RENDER` stage (pixel emission, chunks, textures).
- Texture data, collisions, multi-frame state.

## Constraints

- **`M ≤ 16` BSP nodes per scene.** Phase 1 commits to small fixtures
  to keep dense BSP coefficient storage bounded. Larger scenes are a
  later-phase concern.
- **`N ≤ 8` walls per scene.** Same reason.
- **Frame budget: 8k tokens.** Total prefill + autoregressive ≤ 8192.

## Fixed dimensions and padding

Phase 1 commits to **fixed maximums**, not "whatever this scene
happens to have." Pick `M_MAX = 16` and `N_MAX = 8` as Python
constants at module load, and pad every scene to those sizes with
neutral values. The cross-position aggregation patterns in the API
require this — `Past.mean(name)` divides by the count of contributing
positions, so recovering a sum (e.g. `coeffs · side_P_vec`) needs a
`× count` multiplication, and `count` must be a fixed Python int
known at module load (PWL weights, `linear` matrices, and any
counts-as-constants are all frozen before `forward()` runs).

Padding pattern in `get_prefill`:

```python
M_MAX = 16
N_MAX = 8

def get_prefill(scene, state):
    tokens = []
    # ... player and other tokens ...
    # Real BSP nodes
    for j, node in enumerate(scene.bsp_nodes):
        tokens.append(make_bsp_node(j, nx=node.nx, ny=node.ny, d=node.d))
    # Padding BSP nodes — pick (nx, ny, d) so the half-plane test
    # produces a constant 0 (BACK) for any in-room player. (0, 0, -1)
    # works: 0·x + 0·y + (-1) = -1 < 0 always, so side_P_j = 0.
    for j in range(len(scene.bsp_nodes), M_MAX):
        tokens.append(make_bsp_node(j, nx=0.0, ny=0.0, d=-1.0))
    # Real wall coefficients (dense over (i, j))
    for i, coeffs in enumerate(scene.seg_bsp_coeffs):
        for j, c in enumerate(coeffs):
            tokens.append(make_bsp_coeff(i, j, c=c))
        for j in range(len(coeffs), M_MAX):
            tokens.append(make_bsp_coeff(i, j, c=0))
    # Padding walls — coefficients = 0, const = 0 contributes nothing.
    for i in range(len(scene.seg_bsp_coeffs), N_MAX):
        for j in range(M_MAX):
            tokens.append(make_bsp_coeff(i, j, c=0))
    # ... etc ...
    return tokens
```

`extract_bsp_ranks` returns ranks for the **real** walls only — slice
the autoregressive output by `len(scene.segments)`, not by `N_MAX`.

Two architectures both work over fixed-dim padded scenes:

1. **`mean × M_MAX` aggregation.** Each `THINK_PARTIAL(i, j)`
   position publishes `c_{i,j} · side_P_j` slotted into a width-`N_MAX`
   one-hot at slot `i` (zero elsewhere). At `BSP_RANK(i)` the
   per-wall sum is `past.mean("partial_vec") × M_MAX`, then a
   `one_hot(i, N_MAX)` + `multiply` + `reduce_sum` extracts wall `i`'s
   sum. Faithful to the transformer's natural averaging primitive.
2. **Per-position width-M lookup.** At `BSP_RANK(i)`, do M_MAX
   sequential `past.lookup`s — one per `j` — to fetch each
   `side_P_j` into slot `j` of a width-`M_MAX` Vec, similarly fetch
   the M_MAX coefficients for wall `i`, then `multiply` +
   `reduce_sum` for the dot product locally. Trades aggregation
   depth for sequential-lookup depth; no `× M_MAX` multiplication.

Both are first-class. Pick whichever reads better; the rank test
doesn't care.

## Hints (not prescriptions)

Everything below is a starting-point sketch — the agent designs the
actual vocabulary, prefill structure, and forward implementation. If
a different shape works, use it.

### Token shapes

The autoregressive thinking phase already uses a marker + VALUE pattern
(emit a `BSP_RANK` marker, then a `VALUE` token carrying the rank as
a quantized scalar). Extending the same pattern to prefill keeps token
shapes uniform: each scene-data marker has one or two `IntSlot`
parameters, then one or more `VALUE` tokens carrying its data.

For example, BSP plane `i` could be encoded as:
```
BSP_NODE(i)  →  VALUE(nx_i)  →  VALUE(ny_i)  →  VALUE(d_i)
```

And per-wall coefficient `c_{i,j}`:
```
BSP_COEFF(wall_index=i, bsp_node_index=j)  →  VALUE(c_{i,j})
```

Dense emission (every `(i, j)` pair) keeps consumer logic simple at
the cost of `M·N` (marker, value) pairs in prefill. Sparse emission
(only nonzero coefficients) saves prefill tokens but the consumer has
to handle missing entries explicitly. For phase 1's small fixtures
either works; dense is probably the easier first cut.

### Coefficient and constant magnitudes

Both `seg_bsp_coeffs` and `seg_bsp_consts` are always integer-valued
in `[-N, N]`. They're stored as floats in `MapSubset` for generality,
but if you encode them as `IntSlot(lo=-N, hi=N+1)` rather than
`FloatSlot`, you skip quantization noise entirely (exact integer
encoding) and shrink the slot's required precision.

### `side_P_vec` and aggregation

`side_P_vec` doesn't exist directly in prefill — it has to be computed
from the BSP planes plus player position, which only arrives
autoregressively. Some options to consider:

- An intermediate thinking phase that emits `SIDE_P` VALUE tokens
  (one per BSP node) after `PLAYER_X` / `PLAYER_Y`. Each thinking
  position attends to its corresponding `BSP_NODE` data and the
  player position to compute its bit. The per-wall `BSP_RANK`
  computation then reads these via `past.pick_most_recent` or
  `past.mean`.
- Inline computation at each `BSP_RANK` position — read all BSP planes
  and the player position via attention, compute side bits inline. Costs
  more depth per `BSP_RANK` step.

### Aggregating the dot product

`bsp_coeffs[N] · side_P_vec` is a sum over BSP nodes. With dense
coefficient emission, you have `M` `(BSP_COEFF, VALUE)` pairs per
wall. `past.mean` over matching positions gives `(sum) / count`; to
recover the sum, the agent needs to know the count (which equals `M`
for dense — known at module load).

### Termination

The model needs to emit `DONE` after the last wall's rank. It needs
some way to know "is this the last wall?" — maybe a `TOTAL_WALLS`
token in prefill, maybe by reading the count of `BSP_CONST` markers
implicitly, maybe by structural choice in the autoregressive loop.
The agent picks.

### Prefill → autoregressive transition

See "Prefill vs. autoregressive transition" in `doom_sandbox/CLAUDE.md`.
The recommended pattern — dedicated marker token (e.g. `BEGIN`) as the
final prefill token, only its `forward()` branch produces the AR-start
token — keeps prefill ordering decoupled from AR-start logic. Without
the marker, *some* structural type ends up last in prefill and its
branch silently becomes load-bearing for AR boot, which is brittle.

### Phases don't import each other

If you find yourself wanting to import a previous phase's setup or
forward, rewrite it instead. Patterns carry forward through reading
prior phase code, not through code reuse.
