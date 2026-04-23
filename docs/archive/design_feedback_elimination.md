# Feedback Elimination: Toward a Pure-Autoregressive Game Graph

## Status

Future direction. Not yet implemented.

## Motivation

The current game graph carries two feedback vectors — `sort_feedback`
(16+ wide) and `render_feedback` (27+ wide) — that smuggle internal
state machine state from one token to the next via the input overlay.
This has no analog in how language models work. An LLM's context is its
previous outputs (the KV cache) and the current token's identity.
Hidden side channels are an implementation artifact, not an
architectural necessity.

This document proposes eliminating both feedback vectors so that every
token's computation is a pure function of:

1. Its **type and identity** (which token am I?).
2. **Host-provided ground truth** (player state, wall geometry, textures).
3. **Attention over previous tokens' outputs** (the KV cache).

No side channels. The result is a game graph that is structurally
identical to how a normal autoregressive transformer generates tokens.

## Current Architecture

```
EOS → SORTED×N → (THINKING → RENDER×k)×N
        ↑ sort_feedback    ↑ render_feedback (27 values)
        └─ prev_bsp_rank   └─ wall identity, precomputes,
           sel_onehot         mask, column, chunk position
```

THINKING exists because RENDER doesn't know which wall it's rendering.
The render_feedback vector carries that identity plus the selected
wall's precomputed render parameters plus the column/chunk state
machine position. The sort_feedback carries the previous BSP rank
threshold so each SORTED token can find the next wall.

## Proposed Architecture

### Wall-typed RENDER tokens

Introduce `RENDER_WALL_i` token types that encode wall identity
directly:

```
token_type = [E8_RENDER, one_hot(i, max_walls)]
```

Each RENDER_WALL_i token knows which wall it's rendering from its own
type. It fetches that wall's render precomputes (sort_den, C, D, E,
H_inv, tex_id, vis_lo, vis_hi) via attention to the corresponding
SORTED position. No feedback needed to carry wall identity or
precomputes.

This eliminates the THINKING token type entirely. The token sequence
becomes:

```
EOS → SORTED×N → RENDER_WALL_1×k₁ → RENDER_WALL_2×k₂ → ... → RENDER_WALL_N×kₙ
```

When a RENDER_WALL_i token finishes its last column, it emits
`token_type = RENDER_WALL_{i+1}` as its output. The host feeds it
back unchanged. When wall N finishes, it emits `done = +1`.

### What happens to render_feedback

| Field | Current width | Disposition |
|-------|--------------|-------------|
| render_mask | max_walls | Gone — wall selection is explicit in token type |
| fb_sort_den, fb_C, fb_D, fb_E, fb_H_inv, fb_tex_id | 6 | Gone — fetched via attention |
| fb_col_lo, fb_col_hi | 2 | Gone — fetched via attention |
| fb_onehot | max_walls | Gone — in the token type |
| render_col | 1 | Already an output (`col`) — read back from previous token |
| render_chunk_start | 1 | Already an output (`start`) — read back from previous token |
| render_is_new_wall | 1 | Derivable from token type transition |

The key observation: `render_col` and `render_chunk_start` are not
hidden state. They are the previous token's public output — the
same `col` and `start` values the host bitblits to the framebuffer.
Reading them back is "look at what I just said," which is how
autoregressive generation works.

render_feedback is eliminated entirely.

### What happens to sort_feedback

The sort_feedback's load-bearing field is `prev_bsp_rank` — the
threshold for the next SORTED token's argmin-above query. This is
just the previous SORTED token's selected BSP rank, i.e., its output.

If each SORTED token reads the previous SORTED token's output BSP
rank via attention (from the KV cache), no feedback is needed. The
sort_feedback vector is eliminated.

A simpler alternative: derive the threshold from position. BSP ranks
are clean integers 0..N-1, and SORTED tokens are emitted in order. If
SORTED token at position i uses threshold `i - 1`, the sort becomes a
prefill (all SORTED tokens computed in parallel) with no autoregressive
dependency at all.

## The Dumb-Host Constraint

The host remains a dumb token feeder and pixel bitblitter. It does not
decide which wall to render or what the next token type is. The
transformer's output `token_type` tells the host what to feed next.
The host copies it back verbatim, the same as today.

## What This Costs

- **token_type widens** from 8 to 8 + max_walls dimensions to carry
  the wall one-hot.
- **One extra attention head** per RENDER position to fetch wall
  precomputes from the SORTED sequence.
- The attention pattern for precompute fetch is a simple one-hot
  selection — the kind `attend_argmax_dot` or `attend_mean_where`
  already handles.

## What This Gains

- **THINKING tokens eliminated** — N fewer tokens per frame.
- **render_feedback eliminated** — 27+ input dimensions removed.
- **sort_feedback eliminated** (or sort phase becomes prefill).
- **Simpler output assembly** — no thinking_render_fb construction,
  no feedback packing/unpacking.
- **Architectural clarity** — the compiled transformer is legible as
  a transformer. Every token's inputs are self-describing: its type,
  the world state, and what previous tokens emitted. No mysterious
  27-wide side channels.
- **Philosophical alignment** — the game graph becomes structurally
  identical to autoregressive text generation. Context is the KV cache.
  State is the output sequence. There is nothing else.
