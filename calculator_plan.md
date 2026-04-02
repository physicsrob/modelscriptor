# Calculator Implementation Plan

## Goal

Build a calculator that supports `+`, `-`, `*` on positive integers up to `max_digits` digits. Input format: `A op B=`, output: result digits (autoregressively). Subtraction can produce negative results (prefixed with `-`).

Example inputs and outputs (max_digits=3):
```
123+456=  →  579
500-123=  →  377
100-999=  →  -899
12*34=    →  408
123*456=  →  56088
0*999=    →  0
```

---

## Current Status

**Addition: DONE.** Compiles and passes all tests (1-digit: 18 layers, 3-digit: 32 layers).

**Subtraction: DONE.** Including negative results with "-" prefix. All tests pass.

**Multiplication: BLOCKED.** The V1 embedding-space approach fails due to chained `map_to_table` numerical amplification (Blocker #1). The multiplication graph code exists in `calculator.py` (`multiply_digit_pair`, `multiply_digit_seqs`) but is disabled — the calculator currently outputs "0" for `*`.

**Test suite: 101 passed, 2 expected failures** (the multiplication placeholder tests).

### Files

| File | Status |
|------|--------|
| `examples/calculator.py` | V1: +/- working, * placeholder |
| `tests/compile/forward/test_forward_calculator.py` | 7 addition tests, 5 subtraction tests, 2 multiplication tests (failing) |
| `modelscriptor/compiler/forward/scheduler.py` | Bug fix for shared add_into addends |
| `modelscriptor/modelscript/arithmetic_ops.py` | New: `negate`, `subtract`, `multiply_scalar` |
| `modelscriptor/modelscript/map_select.py` | New: `switch` |

### Bugs Fixed During Implementation

1. **Shared add_into addend reassignment** (scheduler.py): When multiple `add_into` ops share an addend, the step 2a loop re-evaluated `_is_dead_for_add` using mutating `computed_nodes`, causing a shared node to flip from "live" to "dead" mid-batch. Fix: snapshot `computed_nodes` before the loop. Tests: `test_add_into_shared_addend_not_reassigned`, `test_compile_multi_switch_shared_constants`.

2. **Operator re-trigger during autoregressive output** (calculator.py): The "-" token in subtraction results re-triggered `is_operator` at positions after "=", corrupting digit extraction and comparison signals. Fix: `before_equals` guard restricts operator detection to pre-"=" positions.

---

## Blockers

### Blocker #1: Chained `map_to_table` Numerical Amplification

**Severity:** Blocks multiplication entirely. Also limits `remove_leading_0s` to ≤1 recursive level.

**What happens:** When the output of one `map_to_table` feeds as input to another `map_to_table`, the second lookup produces garbage. Verified with minimal reproduction:

```python
# Works (single lookup):
tens, ones = multiply_digit_pair(embedding, a, b)  # dist=0.06 from true embedding

# Fails (chained lookup):
digit_sum, carry = sum_digits(embedding, ones, zero, no_carry)  # dist=1245 — completely wrong
```

**Root cause:** `map_to_table` uses `turn_on_speed = 1000` to create sharp activation boundaries. The output of a lookup is close to the true embedding but not exact (`dist ≈ 0.06`). When this slightly-off embedding is fed into a second lookup, the `turn_on_speed` scaling amplifies the error:

```
ffn_layer linear1: output = turn_on_speed * (key · input - key · key + 1/turn_on_speed)
```

For a perfect match: `key · input = key · key`, output = 1.0. For a slight mismatch where `key · input = key · key + 0.01`: output = `1000 * (0.01 + 0.001)` = 11.0. This 11x amplification cascades through the ReLU and output projection, producing wildly wrong values.

**Impact on multiplication:** Long multiplication requires `multiply_digit_pair` (map_to_table) → `sum_digits` (map_to_table) to propagate carries within partial product rows. This is an unavoidable chain.

**Impact on remove_leading_0s:** Each recursive level applies `compare_to_vector` (which uses the same `turn_on_speed` mechanism) to the output of a `select`. After ≥2 levels, the comparison result blows up. Verified: `remove_leading_0s` with `max_removals=1` works, `max_removals=2` fails.

**Possible fixes:**

1. **Clamp `compare_to_vector` output to [-1, 1]:** Add a second ReLU unit to cap the output. This is `compare_to_vector` in `logic_ops.py`. The current implementation uses 1 FFN unit; it would need 2 (same trick as the `compare` function which already clamps). This fixes `remove_leading_0s` chaining and `compare_to_vector` chaining but NOT arbitrary `map_to_table` chaining.

2. **"Snap" `map_to_table` output to nearest valid embedding:** After a lookup, project the result onto the nearest embedding vector. This could be done as a post-processing step that normalizes the output. However, this is hard to implement in the current graph framework since "nearest embedding" is a non-linear operation over a discrete set.

3. **Reduce `turn_on_speed`:** Lower values reduce amplification but also reduce discrimination sharpness — nearby embeddings might both activate, producing blended outputs instead of sharp lookups. This is a global tradeoff.

4. **Avoid chaining entirely — use scalar representation for arithmetic:** The V2 approach: convert embeddings to scalars (0-9) via a single lookup, do arithmetic with `sum_nodes` (Linear — exact), then convert back. Chaining only occurs at the embed→scalar and scalar→embed boundaries, never in the arithmetic core. The multiplication column accumulation becomes `sum_nodes` (no lookup chain). The `divmod10` lookup takes a scalar sum (exact from Linear addition), so its input is exact and the lookup works.

5. **Increase embedding distance:** If embeddings are further apart in vector space, the `turn_on_speed` amplification matters less because the error-to-distance ratio is smaller. Currently `d_embed=8` with embeddings having norm ~40 and inter-embedding distances of ~5-10. Increasing `d_embed` or using spherical codes with maximum separation would help.

**Recommendation:** Fix #1 (clamp compare_to_vector) is a quick targeted fix that unblocks `remove_leading_0s` chaining. Fix #4 (scalar representation) is the clean solution for multiplication. Fix #5 (embedding distance) is a longer-term improvement to the foundation.

### Blocker #2: Multiplication Result Length vs Output Sequence

**Severity:** Design issue, not yet hit in practice (multiplication is disabled).

**What happens:** Multiplication of two n-digit numbers produces up to 2n digits. The output sequence length must accommodate this. Currently `seq_len = max_digits + 2` (sufficient for addition/subtraction). For multiplication, `seq_len` would need to be `2 * max_digits + 2`.

**Why it's a problem:** Increasing `seq_len` means `output_sequence` creates more `cond_gate` + `sum_nodes` terms. Each extra element adds ~20 nodes. For 3-digit multiplication: `seq_len = 8` (vs 5 for add/sub). The `sum_nodes` of 8 gated values may introduce numerical cross-talk. More importantly, the longer sequence may need more `remove_leading_0s` levels, which hits Blocker #1.

**Possible fixes:**
- Fix Blocker #1 first (allows deeper `remove_leading_0s`).
- Or: use different `seq_len` per operation and have the switch select between the full sequences (not element-by-element). This requires restructuring the dispatch.

---

## Architecture (current implementation)

### Parsing

```python
is_plus  = compare_to_vector(embedding, embedding.get_embedding("+"))
is_minus = compare_to_vector(embedding, embedding.get_embedding("-"))
is_times = compare_to_vector(embedding, embedding.get_embedding("*"))
is_operator = bool_any_true([is_plus, is_minus, is_times])
is_equals = compare_to_vector(embedding, embedding.get_embedding("="))

# Guard: only detect operators before "="
has_seen_equals = pos_encoding.get_prev_value(is_equals, is_equals)
before_equals = bool_not(has_seen_equals)
is_operator_input = bool_all_true([is_operator, before_equals])

which_plus  = pos_encoding.get_prev_value(is_plus, is_operator_input)
which_minus = pos_encoding.get_prev_value(is_minus, is_operator_input)
which_times = pos_encoding.get_prev_value(is_times, is_operator_input)
```

### Dispatch

Each operation produces a `seq_len`-element sequence (digits + eos, leading zeros removed). The `switch` selects element-by-element:

```python
result_digits = [switch([which_plus, which_minus, which_times],
                        [add_seq[i], sub_seq[i], mul_seq[i]])
                 for i in range(seq_len)]
```

### Key constraint: `remove_leading_0s` max_removals ≤ max_digits - 1

Due to Blocker #1, we cannot apply more than `max_digits - 1` levels of leading zero removal (same as the adder). Each operation must produce its digits in a format where this suffices.

---

## Recommended Attack Order

1. **Fix `compare_to_vector` output clamping** (Blocker #1, fix #1). Surgical change to `logic_ops.py` — add a second ReLU unit to cap output at 1.0 (same pattern as the existing `compare` function). This unblocks `remove_leading_0s` chaining and `compare_to_vector` chaining. Test by verifying `remove_leading_0s` with `max_removals=2+` compiles correctly.

2. **Test whether clamped `compare_to_vector` is sufficient for multiplication.** Re-enable the existing `multiply_digit_seqs` code in `calculator.py` and run the multiplication tests. The carry propagation in partial product rows chains `multiply_digit_pair` → `sum_digits`, which is `map_to_table` → `map_to_table`. If the amplification there is also caused by `compare_to_vector`-like patterns, the clamp fix may help. If not, proceed to step 3.

3. **If chained `map_to_table` still fails: implement scalar multiplication.** Convert digit embeddings to scalars (0-9), do column accumulation with `sum_nodes` (exact Linear addition), use `divmod10` lookup for carry extraction, convert back to embeddings. This avoids chained lookups entirely. Only the multiplication path needs the scalar approach — addition and subtraction remain in embedding space.

4. **Increase `seq_len` for multiplication** (Blocker #2). With `remove_leading_0s` chaining fixed, set `seq_len = 2 * max_digits + 2` to accommodate multiplication results up to 2n digits.

5. **Integration testing.** All three operations in one compiled network, autoregressive decoding, resource usage comparison.

---

## Remaining Work

### Phase 3: Multiplication

Blocked on Blocker #1. Once resolved, the implementation path depends on which fix is used:

**If Blocker #1 is fixed at the `map_to_table` level (fixes #1-3, #5):** The existing V1 `multiply_digit_seqs` code in `calculator.py` can be re-enabled. It uses partial product rows with `sum_digits` carry propagation, then adds rows with `sum_digit_seqs`. The `seq_len` needs to increase to `2 * max_digits + 2`, and `remove_leading_0s` needs to handle `2 * max_digits - 1` levels (requires Blocker #1 fix).

**If using scalar representation (fix #4):** Implement `multiply_digit_seqs_scalar` that converts to scalars, does column accumulation via `sum_nodes`, and uses `divmod10` for carry extraction. Then convert result back to embeddings. This avoids chained lookups entirely.

### Phase 4: Integration

All three operations compiled into one network. Autoregressive tests with mixed operations. Resource usage logging.

**Compile targets:** `d=1024, d_head=16` (same as adder).
