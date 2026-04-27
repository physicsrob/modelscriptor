# Chunk 7 plan — Implementing the framework runtime

This document is the executable plan for chunk 7. It assumes the
sandbox API skeleton already exists at `doom_sandbox/api/` (with every
function and method body raising `NotImplementedError("framework
runtime not yet implemented")`) and the types module is complete at
`doom_sandbox/types/`. The job of chunk 7 is to fill in those stubs
with real implementations and tests so the sandbox actually runs.

## Current state (where this plan picks up)

**Done:**
- `doom_sandbox/CLAUDE.md` — sandbox agent's complete context.
- `doom_sandbox/api/` — all stubs declared:
  - `vec.py` — `Vec` dataclass, `constant()` factory (stub).
  - `pwl.py` — `PWLDef`, `pwl_def()`. Construction-time validation works; `__call__` raises.
  - `utils.py` — `concat`, `split` (stubs).
  - `tokens.py` — `IntSlot`, `FloatSlot`, `Slot`, `TokenType`, `Token`, `make_token`, `extract_int_slot`, `extract_float_slot`, `is_type` (stubs).
  - `past.py` — `Past` class with `pick_argmax`, `pick_argmin`, `pick_above_argmin`, `lookup`, `pick_most_recent`, `mean`, `pick_argmax_by`, `pick_argmin_by`, `pick_above_argmin_by` (all stubs). Module docstring documents the 1.0 margin threshold for `pick_*` blending and `lookup` raising.
  - `forward.py` — `ForwardOutput`, `Pixel`, `TokenVocab`, `Config`, `RunOutput`, `run()` (run is a stub).
  - `debug.py` — `print_vec`, `debug_watch`, `assert_in_range`, `assert_close`, `assert_bool`, `assert_integer`, `assert_` (stubs).
  - `std.py` — `type_switch`, `relu`, `saturate`, `compare_const`, `piecewise_linear`, `multiply`, `piecewise_linear_2d` (stubs).
  - `_runtime.py` — exists, holds `_FORWARD_RUNNING` flag (referenced from `pwl.py` and `vec.py`).
- `doom_sandbox/types/` — pydantic schemas:
  - `MapSubset`, `Segment`, `BSPNode`, `Texture`, `GameState`, `FrameInput` (pydantic).
  - `Frame` (plain dataclass, not pydantic).
- `doom_sandbox/phases/phase1_bsp_ranks/` — only `PHASE.md` and `reference.py` exist. `setup.py`, `prefill.py`, `forward.py`, `extract.py`, `test_phase1.py` are agent's job in chunk 9.
- `docs/sandbox/overview.md` — platform docs.
- `docs/sandbox/translation_table.md` — primitive → real-graph mapping.

**Not done:**
- Any framework runtime implementation.
- Any tests.
- `doom_sandbox/fixtures/` (empty — chunk 8 territory).
- `doom_sandbox/runtime/` (does it even exist yet? May need creation depending on internal-module decisions below).

## Goals for chunk 7

By the end:

1. Every stub in `doom_sandbox/api/` does the right thing.
2. The framework can run a phase end-to-end given valid setup/prefill/forward functions.
3. Each primitive is covered by a focused unit test in `tests/sandbox/`.
4. `make test FILE=tests/sandbox/...` runs successfully.
5. The phase 1 implementation (chunk 9) can be written against a working framework.

## Sub-chunk breakdown

The runtime is too big for a single sitting. Order matters because later
primitives compose earlier ones. Recommended:

### 7a — Vec, PWLDef, concat/split

Smallest viable foundation. Once green, every later sub-chunk has a
working substrate.

- **Vec construction**: the framework needs a private way to construct
  Vecs (the agent can't). Add `Vec._make(data, depth)` or similar
  internal constructor used by all primitives.
- **`constant(value)` impl**: builds a 1-shape (or len(value)-shape) Vec with `depth=0`. Honors the freeze flag.
- **`PWLDef.__call__(input)` impl**:
  - Clamp input to `input_range`.
  - Apply `fn` elementwise to clamped values.
  - Add deterministic noise (see "Noise model" below).
  - Return Vec with `depth = input.depth + 1`.
- **`concat(*vecs)`**: numpy concatenate, depth = max input depth.
- **`split(vec, sizes)`**: numpy split, each piece carries vec.depth.
- **Tests**:
  - `test_vec.py` — Vec equality, immutability, `_data` private.
  - `test_pwl.py` — freeze raises; affine functions noise-free at any breakpoint count; depth increments by 1; `breakpoints` validation; `input_range` validation; clamping works.
  - `test_concat_split.py` — depth = max for concat; carry-through for split; shape mismatch raises.

**Open Q for 7a**: noise model details. Pin down before implementing PWLDef.__call__ — see "Open questions" below.

### 7b — Token system (encoding, slot extraction, is_type)

This is the trickiest design decision in chunk 7 because it determines
the internal layout of Vecs that hold tokens.

- **Encoding scheme**: Pick a layout for `(type, slot_values) → Vec`. Recommendation:
  - Reserve `N_TYPES` columns for type one-hot at the start of the Vec.
  - For each slot across all types, reserve `slot_width` columns. Width depends on slot kind:
    - `IntSlot(lo, hi)`: 1 column carrying the integer literally.
    - `FloatSlot(lo, hi, levels)`: 1 column carrying the float (we don't need Gray-code in sandbox — just store the value).
  - Total Vec shape = `N_TYPES + sum(slot widths)`.
  - The framework computes the layout once from `Config.vocab` and caches it.
- **`make_token(type, **slot_values)`**: builds a Vec with the type one-hot set and slot columns populated. Slots not provided default to 0.
- **`extract_int_slot(vec, name)`**: reads the named slot's column (returns 1-shape Vec). If the current input's type doesn't have that slot, return a 1-shape Vec containing 0 (the default).
- **`extract_float_slot(vec, name)`**: same but for FloatSlot.
- **`is_type(vec, token_type)`**: reads the type one-hot column for the given type, returns 1-shape Vec.
- **Each cost**: depth +1. Implementation-side this is a single numpy operation, but we set the depth to `input.depth + 1` to match the per-op cost convention.
- **Deembed `Vec → Token`**: needed by `run()`. Argmax over type one-hot columns to identify type, read slot columns to recover values, dequantize floats per `FloatSlot.levels`.
- **Tests**:
  - `test_tokens.py` — `make_token` round-trips through deembed; `extract_*_slot` returns the right value; missing-slot returns 0; `is_type` is mutually exclusive across types; depth correctness.
  - `test_dispatch.py` — `is_type` masks behave correctly when used with `pwl_def(lambda x: x * mask, ...)` — sandbox doesn't have a `multiply`-by-Vec primitive yet at this point, so this test may need to wait for 7e or use `pwl_def` 1-input-with-mask-as-input.

**Decision needed**: pick the `Config`-to-Vec-layout function's home. Probably `doom_sandbox/runtime/embedding.py`, called once at `Config` construction.

### 7c — Past primitives

The biggest sub-chunk. All 9 attention methods plus the underlying
storage.

- **`Past` storage**: a list of per-position records. Each record contains:
  - The deembedded input `Token` (for `input.type`, `input.<slot>` auto-exports).
  - The position's `exports` dict (Vec values keyed by name).
  - The position's `next_token` Vec (for slot exposure to position N+1, but autoregressive only — at prefill positions this is computed but not used).
- **`Past` is constructed and managed by `run()`**. After each position N's forward call, `run()` appends position N's record to the Past. The Past passed to position N+1 has records 0..N inclusive.
- **Auto-export resolution**: when a query asks for `key_name="input.col"`, the framework looks up the input Token at each past position and reads the `col` slot from its values. Same for `input.type`.
- **`pick_argmax(query, key_name, value_name)`**:
  - For each past position, get its key Vec and value Vec under those names. (Skip positions that don't have either.)
  - Compute `query · key` per past position.
  - Pick the position with the highest score.
  - Implement margin check: if top - second_best < 1.0, blend linearly between the top two; else clean pick.
  - Return Vec with `depth = max(query.depth, max(k.depth across past), max(v.depth across past)) + 1`.
- **`pick_argmin`**: same as above but pick lowest. Margin check still uses |gap| ≥ 1.0.
- **`pick_above_argmin(query, key_name, value_name, threshold)`**: filter to past positions with `query·key > threshold[0]`, then pick lowest among those. Same margin behavior.
- **`lookup(query, key_name, value_name)`**: same scoring as pick_argmax, but raise if margin < 1.0 OR if no past position exports both key and value names. Error message names the two close-scoring positions if margin fails.
- **`pick_most_recent(query, key_name, value_name)`**: among past positions where `query·key` is "high" (above some threshold; or above the same 1.0 margin from second-best?), return the value at the most recent. Need to pin down: is "matching" just argmax, or is there a separate threshold? **Open Q**.
- **`pick_argmax_by(score_name, value_name)`**: same as `pick_argmax` with `query=ones_vec` and `key_name=score_name`. The score_name slot must be 1-shape.
- **`pick_argmin_by`, `pick_above_argmin_by`**: analogous.
- **`mean(value_name)`**: numpy mean across all past positions that exported value_name. Depth = max contributor depth + 1.
- **Tests**: `test_past.py` — one test per primitive, plus integration tests:
  - `pick_argmax` finds clear winner.
  - `pick_argmax` blends on near-tie (verify the blend ratio).
  - `lookup` raises on near-tie with a useful message.
  - `pick_most_recent` finds the most recent matching position.
  - `mean` aggregates correctly across multiple contributors.
  - `pick_*_by` works with precomputed scores.
  - Auto-export `input.<slot>` queries find values from past positions' input tokens.
  - Depth propagation correct across all primitives.

### 7d — `run()` driver and lifecycle

- **`run(config, prefill_tokens, forward)`**:
  1. Initialize empty `Past`.
  2. For each prefill token: embed → call `forward(input_vec, past)` → record output → update Past with input + exports + next_token.
  3. After prefill: enter autoregressive mode. For each subsequent position:
     - Take the previous position's `next_token` Vec, deembed to a Token.
     - If the token type is in `config.terminal_token_types`: stop.
     - Otherwise re-embed it as `input_vec`, call forward, record, update Past.
  4. If `len(positions) >= config.max_positions`: raise (overrun).
  5. Compute `layer_count = max(v.depth for fwd in outputs for v in [fwd.next_token, fwd.pixels, *fwd.exports.values()] if v is not None)`.
  6. Decode pixels: at every position, call `config.decode_pixels(input_tok, fwd.pixels)`; accumulate into `RunOutput.pixels`.
  7. Return `RunOutput(forward_outputs=..., pixels=..., layer_count=...)`.
- **The freeze flag**: set `_FORWARD_RUNNING = True` for the duration of the autoregressive loop. Reset on exit (try/finally).
- **Tests**: `test_run.py` — minimal phase that emits a designated terminal token after N positions; verify run terminates correctly; verify `max_positions` cap raises; verify pixels are collected; verify layer_count is the max-of-returned-Vecs.

### 7e — Stdlib + debug

- **`debug.py`**:
  - `print_vec(vec, label)`: prints `label: <vec._data>` to stdout.
  - `debug_watch(vec, predicate, label)`: same but only if `predicate(vec._data)` is True.
  - `assert_in_range(vec, lo, hi)`: numpy check.
  - `assert_close(vec, expected, atol)`: numpy check on shape and values.
  - `assert_bool(vec, atol)`: values within atol of {0.0, 1.0}.
  - `assert_integer(vec, atol)`: values within atol of nearest integer.
  - `assert_(vec, predicate, message)`: `if not predicate(vec._data): raise AssertionError(message)`.
- **`std.py`**:
  - `type_switch(*branches)`: `result = sum(mask._data * value._data for mask, value in branches)`. Wrap in Vec with depth = max(...) + 1.
  - `relu(input_range)`: returns `pwl_def(lambda x: max(0.0, x), breakpoints=2, input_range=input_range)`.
  - `saturate(input_range)`: returns `pwl_def(lambda x: min(1.0, max(0.0, x)), breakpoints=3, input_range=input_range)`.
  - `compare_const(c, input_range)`: returns `pwl_def(lambda x: 1.0 if x > c else 0.0, breakpoints=...)`.
  - `piecewise_linear(fn, breakpoints, input_range)`: alias for `pwl_def`.
  - `multiply(a, b)`: bilinear PWL approximation via numpy elementwise multiply + noise. Depth = max(a.depth, b.depth) + 1.
  - `piecewise_linear_2d(fn, breakpoints, input_range)`: returns a callable that applies fn pairwise via bilinear approximation.
- **Tests**: one test per primitive. `test_std.py` and `test_debug.py`.

## Open questions to resolve before starting

These are decisions I haven't pinned down. Worth a chat with the user
before writing the corresponding code.

1. **Noise model specifics.** "Simplified" — confirmed earlier — but the
   exact formula isn't agreed. Suggested:
   - For nonlinear PWLs: noise sampled from `N(0, σ)` where
     `σ = SCALE * (input_range_width / breakpoints)²`.
   - `SCALE` is a tunable constant, default `0.01` (rough match to
     PL-drift scales we've seen in `op_noise_data.json`).
   - For affine functions: `σ = 0` (exact).
   - Determinism: seed from `hash((id(pwl_def), tuple(input.tobytes())))`
     so identical inputs give identical outputs across runs.
   - Need user sign-off.

2. **`pick_most_recent` matching threshold.** The current docstring
   says "matching the query (high `query·key` score)" without a
   threshold. Should it be:
   - "Use the same 1.0 margin from `pick_argmax`" (i.e., must be
     a clear winner)?
   - "Take any positive `query·key`"?
   - "Threshold passed in"?
   Need to pin before implementing.

3. **Internal module layout.** `_runtime.py` exists in `api/`. Do we
   add more private modules (`api/_embedding.py`, `api/_loop.py`) or
   create `doom_sandbox/runtime/` as a sibling directory? The CLAUDE.md
   forbids importing from `doom_sandbox.runtime`, so the directory
   name is reserved. Recommendation: use `doom_sandbox/runtime/` as
   the home for non-trivial framework code (loop, embedding, noise,
   state). Keep `_runtime.py` in api/ minimal — just the freeze flag
   and similar tiny shared state.

4. **Token encoding scheme**. Sketched in 7b but not finalized. Concretely:
   - Type one-hot or type index? Pros/cons?
   - Slot widths fixed (1 col each) or variable?
   - How are auto-exports `input.<slot>` resolved at query time —
     re-deembed the past Vec, or store the deembedded Token alongside
     and look up directly? Latter is simpler/faster; former is more
     "graph-faithful." Recommend latter for sandbox.

5. **`extract_*_slot` for missing slots — return 0 always, or raise on
   intentional misuse?** Sandbox semantics: returns 0, agent uses
   `is_type` masks to filter. Confirmed in CLAUDE.md. Just noting it
   here so we don't second-guess during impl.

6. **Tests location and runner.** I'd suggested `tests/sandbox/` outside
   the sandbox tree (mirrors `docs/sandbox/`). Confirm at start of 7.
   Also: do these tests run via the existing `make test` (which is
   GPU-sharded on Modal), or a new `make test-sandbox` target (pure
   Python, local)? Pure-Python local makes sense for sandbox tests;
   may need Makefile addition.

## Files to be created/modified

**Modified** (filling stubs):
- `doom_sandbox/api/vec.py`
- `doom_sandbox/api/pwl.py`
- `doom_sandbox/api/utils.py`
- `doom_sandbox/api/tokens.py`
- `doom_sandbox/api/past.py`
- `doom_sandbox/api/forward.py`
- `doom_sandbox/api/debug.py`
- `doom_sandbox/api/std.py`
- `doom_sandbox/api/_runtime.py` (likely small additions)

**Created**:
- `doom_sandbox/runtime/__init__.py`
- `doom_sandbox/runtime/embedding.py` (token ↔ Vec encode/deembed)
- `doom_sandbox/runtime/loop.py` (run() internals)
- `doom_sandbox/runtime/noise.py` (PWL noise simulation)
- `doom_sandbox/runtime/state.py` (per-position record types for Past)
- `tests/sandbox/conftest.py`
- `tests/sandbox/test_vec.py`
- `tests/sandbox/test_pwl.py`
- `tests/sandbox/test_concat_split.py`
- `tests/sandbox/test_tokens.py`
- `tests/sandbox/test_dispatch.py`
- `tests/sandbox/test_past.py`
- `tests/sandbox/test_run.py`
- `tests/sandbox/test_debug.py`
- `tests/sandbox/test_std.py`

Estimated line counts: ~800-1200 lines of impl + ~600-1000 lines of tests.

## Recommended workflow for the fresh session

1. **Re-read** `doom_sandbox/CLAUDE.md` (it's the contract these
   primitives have to satisfy).
2. **Re-read** the API stubs in `doom_sandbox/api/` to refresh on the
   exact signatures and docstrings.
3. **Re-read** this plan (especially Open Questions).
4. **Resolve open questions with the user** before writing any code.
   Don't start implementing until #1, #2, #4 are pinned down.
5. **Implement 7a** (Vec, PWLDef, concat/split) plus tests. Confirm
   green before moving on.
6. **Implement 7b** (tokens). Confirm green.
7. **Implement 7c** (past). Confirm green.
8. **Implement 7d** (run). Confirm green.
9. **Implement 7e** (stdlib + debug). Confirm green.
10. **Integration smoke test**: write a tiny phase (no fixtures, in-memory `MapSubset`) that exercises end-to-end. Verify `run()` works.

After chunk 7 lands, chunk 8 (fixture serialization) and chunk 9 (phase
1 implementation) follow naturally.

## Things to remember

- The sandbox does *not* need to be fast. Pure Python is fine. Numpy
  for vector math, but no jit / no parallelism / no GPU.
- **Determinism is mandatory.** Tests will be flaky otherwise. Noise
  is seeded from input values, attention scoring is exact, etc.
- The phase 1 reference function (`expected_bsp_ranks`) is already
  written. After chunk 7 + chunk 8 + chunk 9, the phase 1 test should
  pass when the agent's `forward()` correctly computes BSP ranks.
- The translation table (`docs/sandbox/translation_table.md`) is the
  porting bible — when adding/changing a primitive's behavior, update
  the table in the same commit.
