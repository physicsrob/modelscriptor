# DOOM Sandbox

A constrained Python environment for designing DOOM-rendering algorithms.
Once a phase here passes its tests, the design will be ported to a
transformer-based renderer in the parent project — but you do not need to
know anything about transformers. The rules below are what the analogy
buys you, expressed as a small Python API.

You only need what's in this file plus the per-phase `PHASE.md` for the
phase you're working on. Ignore everything outside `doom_sandbox/`.

## What you build

A function that, position by position, consumes one input token and
produces one output token. The framework runs your `forward()` once per
position. Position N can read state that positions 0..N have explicitly
published, including the current position N's own publishes earlier in
the same `forward()` call. There is no other cross-position channel.

## Phases

Each phase is a self-contained re-implementation of some subset of the
DOOM renderer. A phase lives in `phases/<phase_name>/` and contains:

| File | What it owns |
|------|--------------|
| `PHASE.md` | What this phase computes, what the test asserts, the extract function's signature |
| `setup.py` | Vocab declaration + embedding config + (optional) per-position pixel decoder |
| `prefill.py` | `get_prefill(map, state) -> list[Token]` |
| `forward.py` | `forward(input_vec, past) -> ForwardOutput` |
| `extract.py` | `extract_<thing>(outputs) -> <result>` — pull the phase's output from the autoregressive run |
| `reference.py` | `expected_<thing>(map, state) -> <result>` — pure-Python ground truth |
| `test_<phase>.py` | Tie reference + extract together; the phase passes when this passes |

You write all of these. The framework runs the autoregressive loop and
calls your code at each position.

**Phases do not import from each other.** Each is its own implementation.
What carries forward is patterns you've internalized, not code reuse. If
you find yourself wanting to import a previous phase's setup or forward,
rewrite it instead.

## Vec

The data type that carries computed values through your code. Python
ints, strings, `Token`s, and `Pixel`s flow alongside as metadata, but
they don't participate in PWL or `past.*` calls.

A Vec exposes two attributes: `.shape` (length, an int) and `.depth`
(longest op chain that produced it). The underlying numpy values are
private — to inspect them during development, use `print_vec` and the
assertion helpers (see "Debugging and assertions" below).

Vecs are not directly constructible — they come from:

- The framework (input embedding, slot extractions, `past.*` results).
- `PWLDef.__call__(...)`.
- `concat(*vecs)` and `split(vec, sizes)` — free utilities, no compute.

The `input_vec` passed to your `forward()` has `depth = 0` at every
position; depth accumulates from there as you compute.

## PWL — the only compute primitive you define

Other things in this doc — `past.*`, `extract_*_slot`, `make_token` —
also do compute and cost depth, but they're framework-provided
primitives. PWL is where your own functions go.

PWL definitions are created **at module level** (declared once, like model
weights). They are applied many times inside `forward()`.

```python
# At module level
SQUARE = pwl_def(lambda x: x * x,
                 breakpoints=64,
                 input_range=(-10.0, 10.0))
RAMP   = pwl_def(lambda x: max(0.0, x),
                 breakpoints=2,
                 input_range=(-1.0, 1.0))

# Inside forward()
def forward(input_vec, past):
    y = SQUARE(input_vec)   # elementwise; same shape as input
    z = RAMP(y)
```

Rules:

- `pwl_def(fn, breakpoints, input_range)` must run at module load, not
  inside `forward()`. The framework freezes PWL construction during the
  autoregressive loop and raises if you try.
- `fn` takes **one** scalar input (applied elementwise to a Vec input).
  For 2-input operations (products, bilinear functions, etc.), use the
  named primitives in `doom_sandbox.api.std` rather than `pwl_def`.
  See the "2D operations" note below.
- `breakpoints` is an int — the grid resolution along the input axis.
  Must be ≤ 1024.
- `input_range=(lo, hi)` declares the domain the breakpoint grid spans.
  Inputs outside this range are clamped at runtime. Tighter
  `input_range` means a denser grid, which reduces the linear-interp
  residual on nonlinear functions.
- The framework approximates `fn` by sampling it on the breakpoint
  grid and running real linear interpolation on inputs. Affine
  functions land on the grid exactly, so the interp residual is zero
  for them at any breakpoint count. Nonlinear functions accumulate
  the genuine interp error; more breakpoints and a tighter input
  range reduce it.
- On top of the interp result, every `PWLDef` call adds a small
  per-element gaussian (`σ = 1e-6 · |value|`) to represent FP32
  accumulation and other transformer-runtime noise. This noise is
  always present, including for affine functions; size it into your
  tolerance budgets.
- Each `PWLDef.__call__` adds 1 to the result Vec's depth.

**2D operations.** There's no unique definition of a 2D piecewise-linear
function — bilinear interpolation, triangulated linear, and tensor
products are all reasonable choices, with different precision
characteristics. So `pwl_def` is 1D only; 2D operations live as named
primitives in `doom_sandbox.api.std` (e.g., `multiply`,
`piecewise_linear_2d`), each with a specific approximation strategy
that mirrors a corresponding real-project op. If you need a 2D op
that isn't in the stdlib, ask — adding one is a platform-side decision
about which approximation strategy to commit to.

Pre-built primitives covering common patterns are in
`doom_sandbox.api.std` (e.g. `RELU`, `compare_const(c)`, `tan`, `cos`).
Use them. If something you need isn't there, **ask before adding a new
primitive** — new primitives carry a porting cost.

## Free utilities

```python
concat(*vecs: Vec) -> Vec
split(vec: Vec, sizes: list[int]) -> list[Vec]
```

No compute. `concat`'s result has `depth = max(v.depth for v in vecs)`;
each piece returned by `split` carries the parent Vec's `depth`. Use
these to extract specific scalar fields from a packed Vec or to combine
pieces.

## Debugging and assertions

The framework provides two channels for inspecting Vecs without making
their values available to your computation: printing for ad-hoc
inspection, and assertions for invariants you want enforced at runtime.
Both have access to `Vec.data` internally; your code does not.

### Printing

```python
print_vec(vec, label="...")                # always prints
debug_watch(vec, predicate, label="...")   # prints only when predicate fires
```

`print_vec` is the dumb tool — fine for one-off debugging or when
`forward()` runs at only a few positions. In long autoregressive loops
it floods. For ongoing observation across many positions, prefer
`debug_watch` with a predicate that fires only on values worth seeing:

```python
debug_watch(x, lambda data: data[0] > 100, label="x_overflow")
```

The framework cannot expose a position index to your code (that would
let you branch compute on it). Filter via `debug_watch` predicates or
on `vec.shape`, not by counting positions.

### Assertions

Named helpers — these mirror the existing project's assertion API, so
porting recognizes them directly:

```python
assert_in_range(vec, lo, hi)         # bounds check
assert_close(vec, expected, atol)    # value match within tolerance
assert_bool(vec)                     # values near 0 or 1
assert_integer(vec)                  # near-integer values
```

For unusual checks, an escape hatch:

```python
assert_(vec, predicate, message="...")   # arbitrary predicate over data
```

Lean on the named helpers — they're more disciplined and read better.
All assertions raise immediately on failure, returning nothing.

## Tokens — types and slots

The vocabulary is a set of `TokenType`s. Each type has named typed
parameter slots. The embedding encodes (type, slot values) compositionally;
deembed recovers them via per-slot argmax.

```python
RENDER = TokenType("RENDER", slots={
    "col":   IntSlot(lo=0, hi=320),
    "chunk": IntSlot(lo=0, hi=16),
})

VALUE = TokenType("VALUE", slots={
    "v": FloatSlot(lo=-40.0, hi=40.0, levels=65536),
})

THINKING_WALL = TokenType("THINKING_WALL", slots={
    "wall_index": IntSlot(lo=0, hi=8),
})
```

Slot semantics:

- `IntSlot(lo, hi)` — integer range `[lo, hi)`. Recovered exactly.
- `FloatSlot(lo, hi, levels=65536)` — float quantized to `levels`
  evenly-spaced steps over `[lo, hi]`. Round-trip introduces ~one LSB of
  quantization error. Default 65536 levels gives ~`(hi-lo)/2^16` precision.

## ForwardOutput

```python
@dataclass
class ForwardOutput:
    next_token: Vec                          # required
    pixels: Vec | None = None                # only at render-emitting positions
```

- `next_token` — the framework deembeds this to a discrete `Token`. During
  the autoregressive phase, it re-embeds the deembedded token as the input
  for position N+1. During prefill, the next position's input is the next
  prefill token (not your prediction) — your `next_token` at prefill
  positions is computed but not used as feedback. In both phases, the
  deembedded token's typed slots are exposed under `past` (see
  *Auto-published input slots* below).
- `pixels` — the framework calls your phase's `decode_pixels(input_tok,
  pixels)` at every position. Your `decode_pixels` returns `[]` at
  positions that don't render and a list of `Pixel`s at ones that do.
  Pixels are blitted, never re-embedded, never visible to future
  positions. If you pass `pixels=None` and `decode_pixels` tries to read
  it, raise from inside `decode_pixels` to surface the bug early.

To make a Vec visible to other positions, call `past.publish(name, vec)`
during your `forward()` — see *Cross-position attention via past* below.

## input_vec — the per-position input

`input_vec` is a Vec encoding the input token at the current position —
both its type and its slot values, packed together by the framework.
Its shape is fixed (chosen by the framework based on your declared
vocab); you don't read the layout, you read through primitives.

### Inspecting the input

```python
is_render = is_type(input_vec, RENDER)             # 1-Vec: 1.0 if RENDER, else 0.0
col       = extract_int_slot(input_vec, "col")     # 1-Vec carrying the int value
v         = extract_float_slot(input_vec, "v")     # 1-Vec carrying the float
```

`is_type` returns a 1-Vec that's 1.0 when the input is the named type
and 0.0 otherwise. Types are mutually exclusive — exactly one
`is_type(input_vec, T)` returns 1.0 across the vocab.

If a slot isn't declared on the current input's type,
`extract_*_slot` returns 0. Use `is_type` masks to ignore extractions
whose type doesn't have that slot.

### Dispatching by token type

Your `forward()` runs the same Python code at every position — you
can't `if`-branch on type. The pattern is: compute each token type's
branch unconditionally, then mask-and-sum with `is_type`:

```python
render_out   = ...   # PWL chain handling RENDER inputs
thinking_out = ...   # PWL chain handling THINKING_WALL inputs
value_out    = ...   # PWL chain handling VALUE inputs

next_tok = type_switch(
    (is_render,   render_out),
    (is_thinking, thinking_out),
    (is_value,    value_out),
)
```

`type_switch` (in `doom_sandbox.api.std`) selects the branch whose
mask is 1.0; the others contribute zero. Branches whose token type
isn't in the current vocab subset can be omitted; you only need
branches for types the framework will actually feed in.

### Building the next token

```python
next_tok = make_token(RENDER, col=next_col_vec, chunk=next_chunk_vec)
```

Slot values are Vecs (computed from your PWL chains, or extracted from
`input_vec`, or returned from `past.*`). Unspecified slots default to
0 for `IntSlot` and 0.0 for `FloatSlot`.

`is_type`, `extract_*_slot`, and `make_token` each cost depth +1.

## Cross-position attention via `past`

You attend to other positions only via `past`, only by named values you
have published. `past` searches every position from the start of the
run up to and including the current one.

### Publishing

```python
past.publish(name: str, vec: Vec) -> None
```

Inside `forward()`, call `past.publish(name, vec)` to make `vec`
queryable as `name` at this position and every later position. The
order matters: a `past.*` call only sees `name` from the current
position if you've already published it earlier in the same `forward()`.

This mirrors the transformer rule that an attention layer reads from
positions' residual streams as of the most recent prior write.
Publishing is the analog of writing a column to the residual stream;
re-publishing under the same name overwrites (analog of a residual-stream
column being rewritten by a later layer).

Names starting with `input.` are reserved — see *Auto-published input
slots* below.

### Querying

```python
# Score = query · key at each position. The picked value is returned.
past.pick_argmax(query: Vec, key_name: str, value_name: str) -> Vec
past.pick_argmin(query: Vec, key_name: str, value_name: str) -> Vec
past.pick_above_argmin(query, key_name, value_name, threshold: Vec) -> Vec

# Equality lookup — same Q·K scoring as pick_*, but raises if no
# position scores clearly above the rest, or if the search set is empty.
# Use when you're certain exactly one position should match.
past.lookup(query: Vec, key_name: str, value_name: str) -> Vec

# Recency-biased lookup — among matching positions, return the
# value at the most recent. Used for reading prior identifier emissions.
past.pick_most_recent(query: Vec, key_name: str, value_name: str) -> Vec

# Score = a precomputed scalar exported by the producer (no Q·K).
past.pick_argmax_by(score_name: str, value_name: str) -> Vec
past.pick_argmin_by(score_name: str, value_name: str) -> Vec
past.pick_above_argmin_by(score_name, value_name, threshold: Vec) -> Vec

# Aggregation: elementwise mean of value_name across all positions
# that published it. Use for broadcast (single contributor — mean of
# one value is the value) and for multi-position aggregation (e.g.
# combining M positions' slot-vectors into an M-wide result, where
# each slot lands at value/M; multiply by M downstream if you need
# unscaled sums).
past.mean(value_name: str) -> Vec
```

**Self-attention works naturally via publish-then-attend.** After
`past.publish("X", vec)`, a subsequent `past.*` call at the same
position can find `X` (with the current position's value as one of the
candidates). To aggregate over all positions including yourself, just
publish before you query.

**Auto-published input slots.** At every position the framework
auto-publishes the input token's data:

- `input.type` — a one-hot Vec of width `len(vocab)` with the bit set
  for the input token's type.
- `input.<slot_name>` — a 1-shape Vec carrying the input token's value
  for each declared slot, e.g. `input.col`, `input.x`.

These are queryable like anything else. At prefill positions, the
input is the prefill token; at autoregressive positions, the input is
the previous position's deembedded `next_token`.

**Depth of past.\* results.** A `past.*` call returns a Vec with
`depth = (max of all input depths) + 1`. The exact set of inputs varies
by primitive:
- `pick_argmax` / `pick_argmin` / `lookup` / `pick_most_recent`: query
  depth, deepest key across all candidates, deepest value across all
  candidates.
- `pick_above_argmin`: same plus `threshold.depth`.
- `pick_argmax_by` / `pick_argmin_by`: deepest score, deepest value
  (no query).
- `pick_above_argmin_by`: same plus `threshold.depth`.
- `mean`: deepest contributor.

Every position that contributes (under either name) affects the
result's depth, not just the one whose key actually matched — the
framework has to wait for all candidates' published values to be
ready before deciding. Chaining `past.*` calls in series compounds
depth, since each consumer inherits its producer's depth.

**No separate namespace for emitted slots.** A `next_token` you emit
at position N has no direct queryable form at later positions other
than through `past.input.*` at position N+1 (and only when the
autoregressive loop fed it forward — at prefill positions, the
predicted `next_token` is discarded and never becomes anyone's
`input.*`). If a value needs to be visible to many later positions
independent of the autoregressive chain, publish it.

**Near-tie blending.** `pick_*` and `lookup` share a uniqueness
threshold of `1.0`: when the top score beats the second-best by 1.0 or
more, `pick_*` returns a clean pick; when the gap is smaller, `pick_*`
blends linearly between the top two candidates. `lookup` is the strict
variant — it raises in the blend zone instead of returning a blend,
with a diagnostic naming the two close scores. Designs that depend on
near-ties will fail their tests. Either tighten the scoring (different
key, more separation) or pick a different primitive.

## setup()

```python
from doom_sandbox.api import Config, TokenVocab

DONE = TokenType("DONE", slots={})

def setup() -> Config:
    vocab = TokenVocab([RENDER, VALUE, THINKING_WALL, NO_OP, DONE, ...])

    def decode_pixels(input_tok: Token, pixels_vec: Vec) -> list[Pixel]:
        # Called at every position. Return [] for non-render positions
        # or for phases that don't render. For render positions, decode
        # pixels_vec into Pixel(x, y, color) entries.
        return []

    return Config(
        vocab=vocab,
        decode_pixels=decode_pixels,
        terminal_token_types={DONE},
    )
```

## get_prefill()

```python
from doom_sandbox.types import MapSubset, GameState

def get_prefill(map: MapSubset, state: GameState) -> list[Token]:
    """Encode the scene + initial player state into the prefill token sequence."""
    tokens = []
    # ... append scene tokens ...
    return tokens
```

The framework feeds these tokens through the embedding and runs your
`forward()` at each position, accumulating outputs. The prefill tokens
are the prefix; later positions are model-decided (your `forward()`'s
`next_token` becomes the next position's input).

## How the run terminates

The framework runs `forward()` at every prefill position, then continues
autoregressively. The loop stops when either:

1. The deembedded `next_token` is a **terminal token type** declared in
   your `setup()` (the framework checks each step against this set), or
2. A hard `max_positions` cap is hit (declared in `setup()`, default
   8192).

The terminal token type and `max_positions` are both phase-level
decisions you make in `setup()`. PHASE.md will say what's expected for
the phase you're working on.

## Tests

```bash
make test                                         # full sandbox suite
make test FILE=phases/phase1_bsp_ranks/test.py   # single phase
```

Pure Python. No GPU. Every phase test follows this pattern:

```python
from doom_sandbox.api import run
from doom_sandbox.types import GameState
from doom_sandbox.fixtures import load_fixture
from .setup import setup
from .prefill import get_prefill
from .forward import forward
from .extract import extract_bsp_ranks
from .reference import expected_bsp_ranks

def test_phase1_box_room():
    map = load_fixture("box_room")
    state = GameState(x=0.0, y=0.0, angle=0)

    expected = expected_bsp_ranks(map, state)

    config = setup()
    prefill_tokens = get_prefill(map, state)
    outputs = run(config, prefill_tokens, forward)   # framework drives the loop

    actual = extract_bsp_ranks(outputs)
    assert actual == expected
```

`outputs.layer_count` is the renderer's total depth — the max `.depth`
across every Vec your `forward()` returns at any position (`next_token`,
`pixels` if present, and every value passed to `past.publish`). A returned Vec's
`.depth` already includes the depth of every intermediate that fed into
it, so you don't need to track intermediates separately. Aim to keep
`layer_count` bounded — it's a floor on how deep a transformer this
would compile to (compiled depth can exceed it when residual width is
tight; it doesn't come in lower).

## Don't

- Import from `torchwright/`, `doom_sandbox.runtime`, or another phase.
  Allowed: `doom_sandbox.api`, `doom_sandbox.types`,
  `doom_sandbox.fixtures`, modules within your own phase, and the
  Python standard library.
- Read `Vec.data` directly.
- Branch on Vec values with Python `if`/`while`. Conditional logic goes
  through PWL. Python `if` on `v.shape` (an int) is fine.
- Worry about transformers, attention heads, residual streams, GPU
  numerical precision, or compilation. The sandbox simulates what
  matters.
