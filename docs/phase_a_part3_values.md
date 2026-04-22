# Phase A, Part 3 — Per-value math

With the 16-wide state machine in place (Part 2), this part fills in
the 10 stub slots with real math: BSP_RANK, IS_RENDERABLE, the four
rotation values (CROSS_A, DOT_A, CROSS_B, DOT_B), T_LO, T_HI,
VIS_LO, VIS_HI.

Crucially, Part 3 is also where the core motivation of the thinking-
token architecture lands in code: **derived values attend to prior
thinking-value tokens at layer 0**, dequantize from the embedding
bits, and use the result in their own math. Thinking tokens here
function as an arithmetic substrate that other thinking tokens
consume, exactly as the design intends.

The rendered frame behavior does not change from Part 2. SORTED
still reads bsp_rank / is_renderable / vis_lo / vis_hi from the
prefill WALL stage; RENDER still reads player_x/y from the PLAYER
broadcast. No downstream stage outside of thinking_wall consumes
the 10 new values in Phase A. Their purpose is (a) dual-path
testing, (b) exercising the thinking-value-readback mechanism, and
(c) being correct ahead of the future phase that deletes prefill
WALL.

## Strong opinions this part commits to

### Base values compute from first principles; derived values read upstream thinking values

The 10 value types split by whether they have upstream thinking values
to consume:

**Base** — no upstream thinking values, compute from attended wall
geometry + broadcast values:

- BSP_RANK
- IS_RENDERABLE
- CROSS_A, DOT_A, CROSS_B, DOT_B
- HIT_FULL, HIT_X, HIT_Y (Part 2 carries over)

**Derived** — consume prior thinking-value tokens via layer-0
attention:

- T_LO, T_HI (read CROSS_A, DOT_A, CROSS_B, DOT_B)
- VIS_LO, VIS_HI (read T_LO, T_HI, CROSS_A, DOT_A, CROSS_B, DOT_B)

This is the load-bearing decision. It's what differentiates Phase A
from "parallel-computing-everything-at-every-step" — the chain
demonstrates that thinking tokens carry arithmetic forward through
the token stream.

### Cross-step thinking-value reads go through one helper

All reads of prior thinking values go through a single,
well-tested helper:

```python
readback = build_thinking_readback(
    embedding=embedding,
    prev_id_slots=prev_id_slots,
    is_value_category=is_value_category,
    pos_encoding=pos_encoding,
)

t_lo_cross_a = readback.get_value_after_last("CROSS_A")  # → Node[float]
```

`get_value_after_last(name)` returns a Node holding the dequantized
float from the most recent VALUE position whose input token was the
VALUE that followed an identifier of the given `name`. No caller
builds attention + dequantization inline; no caller manages the
is_X_value flag construction by hand.

The helper is a single module (`torchwright/doom/thinking_readback.py`)
with dedicated unit tests. See **Testability** below.

### VIS values gate on locally-computed is_renderable

`_compute_visibility_columns` in `wall.py` zeroes its output on
non-renderable walls. The VIS identifier step computes
is_renderable locally (fresh from sort_den / sort_num_t) and gates
its output. It does *not* read IS_RENDERABLE via the helper.

Rationale: is_renderable's computation is shallow (~3 layers),
cheaper to redo than to route through an attention + dequant. This
is a pragmatic exception to the "attend to upstream values"
default — applied once, for the one value that's cheap enough to
recompute that routing through the helper adds depth with no
benefit.

### Math ports 1:1 from wall.py for base values

- BSP_RANK: `_compute_bsp_rank` — dot over `bsp_coeffs × side_P_vec +
  bsp_const`.
- IS_RENDERABLE: the renderability subset of `_compute_bsp_rank`.
- CROSS/DOT: `_rotate_into_player_frame` on endpoint offsets.
- HIT_*: `_compute_hit_flags` (unchanged from M4/Part 2).

### Math ports mostly from wall.py for derived values, with attention reads replacing recomputation

- T_LO/T_HI: `_plane_clip_contribs` + aggregation. Inputs
  (cross_a, dot_a, cross_b, dot_b) come from `readback.get_value_after_last(...)`.
- VIS_LO/VIS_HI: `_compute_visibility_columns`'s projection step.
  Inputs (t_lo, t_hi, cross/dot endpoints) come from the helper.
  is_renderable gate is recomputed locally.

### Values quantize uniformly to [0, 65535]

As in Part 1 — BSP_RANK (integer 0..7), IS_RENDERABLE (0/1), and
HIT_* (0/1) are sparse-domain values in a full-range encoding. The
design-doc `VALUE_RANGE_BY_IDX` table defines each value's
`(lo, hi)` float range for quantize / dequantize. No special-casing.

### Per-step depth ceiling is still 15 layers

Derived values are depth-tighter than base values. Rough estimates:

| Step                       | Depth (compute) | Depth (emit) | Total |
|----------------------------|-----------------|--------------|-------|
| Base values (BSP, CROSS)   | ~2–4            | ~4           | ~6–8  |
| IS_RENDERABLE              | ~3              | ~4           | ~7    |
| HIT_*                      | ~4              | ~4           | ~8    |
| T_LO/T_HI (derived)        | ~1 (attn) + ~1–2 (dequant) + ~5 (clip) | ~4 | ~11–12 |
| VIS_LO/VIS_HI (derived)    | ~1 (attn) + ~1–2 (dequant) + ~7 (project) | ~4 | ~13–14 |

VIS steps are the closest to the ceiling. Measure with `make graph-stats`;
flag if any step exceeds 15.

## The thinking_readback helper

### API

```python
class ThinkingReadback:
    def get_value_after_last(self, name: str) -> Node:
        """Return the dequantized float for the most recent VALUE
        token that followed an identifier named `name`.

        `name` is one of the 16 identifier names (BSP_RANK, T_LO, etc.).
        The returned Node is a 1-wide float in the identifier's
        design-doc range.

        Caches the built flag + attention per name — repeated calls
        for the same name return the same Node.
        """

def build_thinking_readback(
    embedding: Node,
    prev_id_slots: List[Node],      # 16-wide one-hot components from thinking_wall
    is_value_category: Node,        # ±1 flag, true at VALUE positions
    pos_encoding: PosEncoding,
) -> ThinkingReadback: ...
```

### Internals (expected shape)

For each requested name `X`:

1. Build `is_X_value = bool_all_true([is_value_category, compare(prev_id_slots[IDX_X], 0.5)])` if not cached.
2. Attention: `attend_most_recent_matching(query = +1 const, key = is_X_value, value = embedding_bit_cols)` returns the 64-wide one-hot region of the matching VALUE position's embedding.
3. Decode: one Linear mapping `(h3, h2, h1, h0)` one-hots to integer ID = `4096·h3 + 256·h2 + 16·h1 + h0`.
4. Dequantize: one Linear mapping ID to float using `(lo, hi) = VALUE_RANGE_BY_IDX[IDX_X]`.

### Testability

`tests/doom/test_thinking_readback.py`:

1. **Round-trip correctness** — emit a known quantized float at a VALUE position; `get_value_after_last` returns it within one LSB. Run once per representative value range (CROSS_A's `[-40, 40]`, T_LO's `[0, 1]`, BSP_RANK's `[0, 7]`, RESOLVED_X's `[-20, 20]`).
2. **Independence** — multiple identifiers emitted in a single sequence; each `get_value_after_last(X)` returns X's value, not any other's.
3. **Recency** — two instances of the same identifier (wall 0's CROSS_A, wall 1's CROSS_A); the consumer at wall 1's downstream position reads wall 1's value.
4. **Hardness** — attention's `assert_hardness_gt(0.99)` passes on a well-formed input sequence.
5. **Behavior on empty cache** — consumer fires before any instance of the identifier appears. Expected behavior is defined (either returns 0 or some default; caller must gate on validity if it matters). Decide during implementation.

Tests are standalone — a minimal graph with hand-constructed token sequence, not dependent on the full DOOM compile.

## Scope

- Write `torchwright/doom/thinking_readback.py` with
  `build_thinking_readback` + `ThinkingReadback.get_value_after_last`.
- Write `tests/doom/test_thinking_readback.py` covering the 5 cases
  above.
- Replace the 10 stub slots in `thinking_wall.py` (or wherever they
  live post-Part-2):
  - Base values: BSP_RANK, IS_RENDERABLE, CROSS/DOT, HIT_* —
    compute from first principles via ports from `wall.py`.
  - Derived values: T_LO, T_HI, VIS_LO, VIS_HI — read upstream
    values via `readback.get_value_after_last(...)`, then apply
    the ported wall.py math.
- Add one dual-path test per new value type. For derived values,
  the reference uses exact float math (no quantization), and the
  assertion tolerance accounts for accumulated LSBs through the
  chain.

## Not in scope

- RESOLVED_X/Y/ANGLE math (Part 4; reuses the same helper).
- EOS gutting, RENDER migration (Part 4).
- Consuming the 10 values in SORTED / RENDER (future phase, out of
  Phase A).
- Updating the numerical noise pipeline (`make measure-noise`) —
  probably not needed since no new primitive ops are introduced, but
  see open question.

## Acceptance criteria

1. `test_thinking_readback.py` (the 5 cases above) passes.
2. Each of the 10 new value types has a dual-path test passing
   within the design-doc quantization tolerance for its range
   (accounting for accumulated LSBs on derived values).
3. HIT_FULL/HIT_X/HIT_Y dual-path tests (from M4/Part 2) continue
   to pass.
4. `make walkthrough ARGS="--scene box --frames 3"` renders the
   same output as Part 2 (unchanged — 10 values still have no
   downstream consumers in Phase A).
5. `make graph-stats` shows no identifier step exceeding 15 layers.
6. No regressions in `make test` outside the thinking-token area.

## Open questions

### Helper API shape

Registry object (the sketch above) vs. pair of free functions
(`build_flag(name, ...)` + `read_value(flag, name, ...)`). Registry
is cleaner; free functions are more explicit about what's being
built. Decide during implementation.

### Behavior when the referenced identifier is absent from the KV cache

For Phase A's actual use (derived values reading upstream base
values within the same wall, RENDER reading RESOLVED after all
walls), the cache always contains the referenced identifier when
the consumer fires. But if we later use the helper in a context
where the reference might be missing, behavior needs to be defined —
return zero, raise, or have a "valid" bit. Decide based on the
Phase A consumers' actual needs.

### Accumulated quantization error

Derived values compound LSBs:
- VIS_LO = f(T_LO, CROSS_A, DOT_A, ...)
- T_LO = f(CROSS_A, DOT_A, ...)
- CROSS_A has 1 LSB at emit.

Through the chain, VIS_LO accumulates ~3 LSBs (1 per quantization
boundary crossed). The design doc's per-value ranges and per-value
LSBs mean the total accumulated error in screen columns for VIS_LO
may approach or exceed the ~0.003 screen-column budget. Flag as a
measurement task during implementation. Mitigation if tight: widen
T_LO/T_HI to 32-bit (current 16-bit at LSB 0.000015 amplifies
through projection). Do the widening only if measurements show it's
needed.

### Numerical noise pipeline

No new primitive ops are introduced (all math uses existing
`piecewise_linear_2d`, `reciprocal`, `low_rank_2d`, etc.), but new
call-site distributions may warrant entries in
`docs/op_noise_data.json`. If `make measure-noise` output changes
after Part 3, update `docs/numerical_noise_findings.md`. If
unchanged, nothing to do.

### Stage-file organization

Whether Part 3's math lives in `thinking_wall.py` inline, in a
`thinking_identifier.py` monolith, or split per-cluster
(`thinking_bsp.py`, `thinking_rotation.py`, `thinking_clip.py`,
`thinking_vis.py`, `thinking_hit.py`). Cosmetic; decide during
implementation based on code volume.

## High-level task list (not exhaustive)

1. Write `torchwright/doom/thinking_readback.py` — helper module with
   `build_thinking_readback` + `ThinkingReadback.get_value_after_last`.
2. Write `tests/doom/test_thinking_readback.py` — 5 unit tests.
3. Replace stub slots with real computations:
   - Base values (port from `wall.py`).
   - Derived values (attend via helper + port clip / projection math).
4. Write dual-path tests per value. For derived values, compute the
   reference with exact float math; the tolerance accounts for chain
   LSBs.
5. Run `make graph-stats`, `make test`, `make walkthrough` to
   confirm no regressions and depth within budget.
6. Measure accumulated quantization error on the derived chain
   (box scene + multi scene). If near the ~0.003 screen-column
   budget, widen T_LO/T_HI to 32-bit.
