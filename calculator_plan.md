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

**All three operations implemented and passing.** 133 tests, 0 failures.

| Operation | Status | 1-digit layers | 3-digit layers |
|-----------|--------|---------------|---------------|
| Addition | DONE | 18 | 38 |
| Subtraction | DONE | 18 | 38 |
| Multiplication | DONE | 18 | 38 |

**Compile targets:** 1-digit: d=1024, 3-digit: d=2048 (d_head=16).

### Files

| File | Status |
|------|--------|
| `examples/calculator.py` | V1: +/-/* all working |
| `tests/compile/forward/test_forward_calculator.py` | 7 tests (module-scoped fixtures) |
| `tests/modelscript/test_numerical_robustness.py` | 14 tests for chaining robustness |
| `modelscriptor/modelscript/const.py` | `embedding_turn_on_speed = 1.0` |
| `modelscriptor/modelscript/arithmetic_ops.py` | `negate`, `subtract`, `multiply_scalar` |
| `modelscriptor/modelscript/map_select.py` | `switch` |
| `modelscriptor/compiler/utils.py` | Fixed exponential BFS |

### Key Fixes During Implementation

1. **Shared add_into addend reassignment** (scheduler.py): When multiple `add_into` ops share an addend, the step 2a loop re-evaluated `_is_dead_for_add` using mutating `computed_nodes`, causing a shared node to flip from "live" to "dead" mid-batch. Fix: snapshot `computed_nodes` before the loop.

2. **Operator re-trigger during autoregressive output** (calculator.py): The "-" token in subtraction results re-triggered `is_operator` at positions after "=", corrupting digit extraction and comparison signals. Fix: `before_equals` guard restricts operator detection to pre-"=" positions.

3. **`embedding_turn_on_speed`** (const.py, map_select.py, logic_ops.py): The global `turn_on_speed=10.0` gave a margin of only 0.1 for embedding-space operations (`map_to_table`, `compare_to_vector`). With embedding norms of 40 (self-dot=1600), even tiny Euclidean errors from `map_to_table` output (~0.012) produced dot-product errors (~0.293) exceeding the margin. This caused chained `map_to_table` lookups to fail — the correct key's ReLU activation was killed, returning the default value. Fix: separate `embedding_turn_on_speed=1.0` (margin=1.0, ~30x safety factor) for embedding-space ops. Scalar operations keep `turn_on_speed=10.0`.

4. **Exponential `get_ancestor_nodes`** (utils.py): Recursive traversal without a visited set re-traversed shared subgraphs exponentially. For 1544 nodes: 8 seconds → 0.5 milliseconds.

---

## Architecture

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
```

### Dispatch

Each operation produces a `seq_len`-element sequence (seq_len = 2*max_digits+2 to accommodate multiplication results up to 2n digits). Addition and subtraction pad with eos. The `switch` selects element-by-element:

```python
result_digits = [switch([which_plus, which_minus, which_times],
                        [add_seq[i], sub_seq[i], mul_seq[i]])
                 for i in range(seq_len)]
```

### Multiplication

Long multiplication in embedding space: `multiply_digit_pair` produces (tens, ones) via `map_to_table`, partial product rows accumulate carries via `sum_digits`, rows are summed pairwise via `sum_digit_seqs`. This chains `map_to_table` → `map_to_table`, which works thanks to `embedding_turn_on_speed=1.0`.

---

## Performance

| Metric | 1-digit | 3-digit |
|--------|---------|---------|
| Graph nodes | 345 | 1544 |
| Layers | 18 | 38 |
| d | 1024 | 2048 |
| Compile time | ~0.3s | ~1.8s |
| Inference per case | ~0.5s | ~4.3s |

Inference is the bottleneck — each autoregressive step runs a full forward pass through all layers at full d. Compile time is fast after the `get_ancestor_nodes` fix.
