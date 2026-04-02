# Adder Graph Optimizations

## Current State

| Digits | Nodes | Layers | Peak cols (of 1024) |
|--------|-------|--------|---------------------|
| 1      | 51    | 6      | ~67                 |
| 3      | 202   | 20     | ~297                |
| 5      | 398   | 30     | ~297                |

Two sequential bottlenecks in the adder graph dominate the critical path: the carry chain in `sum_digit_seqs` and the recursive `remove_leading_0s`. Both are O(n) depth for n digits. Both can be reduced to O(log n).

---

## 1. Carry-Lookahead Addition

**What:** Replace sequential carry propagation in `sum_digit_seqs` with a parallel prefix carry computation.

**Current approach** (`adder.py:80-96`): A right-to-left loop where each digit's sum depends on the carry from the previous digit. Depth: O(n) where n = max_digits.

```
carry_0 = -1 (no carry)
sum_2, carry_2 = sum_digits(d1_2, d2_2, carry_0)
sum_1, carry_1 = sum_digits(d1_1, d2_1, carry_2)   # waits for carry_2
sum_0, carry_0 = sum_digits(d1_0, d2_0, carry_1)   # waits for carry_1
```

**Proposed approach:**

1. Compute Generate/Propagate signals for each digit pair (all parallel):
   - G_i = 1 if a_i + b_i >= 10 (generates carry regardless of carry-in)
   - P_i = 1 if a_i + b_i == 9 (propagates carry-in to carry-out)
   - One `map_to_table` per digit pair, from `concat([digit_a, digit_b])` to `(G, P)`

2. Parallel prefix scan to resolve all carries in O(log n) depth:
   - Combine rule: G_ij = G_i OR (P_i AND G_j), P_ij = P_i AND P_j
   - Use existing `bool_all_true` (for AND) and `bool_any_true` (for OR)
   - Tree structure: log2(n) levels of combines

3. Compute final digit sums in parallel using resolved carries:
   - Same `map_to_table(concat([digit_a, digit_b, carry_in]))` as now
   - But all digits computed simultaneously since all carries are known

**Depth comparison:**

| Digits | Sequential | Carry-lookahead | Savings |
|--------|-----------|-----------------|---------|
| 3      | 3 maps    | 2 maps + 2 bool | ~0      |
| 5      | 5 maps    | 2 maps + 3 bool | small   |
| 10     | 10 maps   | 2 maps + 4 bool | ~50%    |
| 100    | 100 maps  | 2 maps + 7 bool | ~93%    |

**Implementation:** Replace `sum_digit_seqs` (~20 lines). All needed primitives (`map_to_table`, `bool_all_true`, `bool_any_true`, `concat`) already exist. Estimated ~30 lines of new code.

**Verdict:** Marginal at 3-5 digits. Essential at 10+. Implement when scaling to larger digit counts.

---

## 2. Parallel Leading-Zero Removal

**What:** Replace recursive leading-zero removal with a flat parallel structure.

**Current approach** (`adder.py:99-116`): Recursive with depth `max_digits - 1`. Each level checks if the first element is "0", and if so, shifts everything left by one position using `select`. After k levels, up to k leading zeros are removed.

```python
# Level 1: if seq[0]=="0", shift left
# Level 2: if new seq[0]=="0", shift left again
# ... max_digits-1 levels
```

Depth: O(max_digits), with each level adding a `compare_to_vector` + `select` chain.

**Proposed approach:**

1. Compute `is_zero[i] = compare_to_vector(seq[i], "0")` for each position (parallel)

2. Compute prefix-AND to get "all zeros up to position i":
   - `all_zero_to[0] = is_zero[0]`
   - `all_zero_to[1] = bool_all_true([is_zero[0], is_zero[1]])`
   - `all_zero_to[k] = bool_all_true([all_zero_to[k-1], is_zero[k]])` (or use tree for O(log n))

3. Flat multiplexer at each output position (all positions parallel):
   ```python
   # For 3 digits + eos:
   out[0] = select(all_zero_to[1], seq[2], select(all_zero_to[0], seq[1], seq[0]))
   out[1] = select(all_zero_to[1], seq[3], select(all_zero_to[0], seq[2], seq[1]))
   out[2] = select(all_zero_to[0], seq[3], seq[2])
   out[3] = seq[3]  # eos passthrough
   ```

**Depth comparison:**

| Digits | Recursive levels | Parallel depth | Savings |
|--------|-----------------|----------------|---------|
| 3      | 2               | ~2             | ~0      |
| 5      | 4               | ~3             | ~25%    |
| 10     | 9               | ~4             | ~55%    |
| 100    | 99              | ~7             | ~93%    |

**Implementation:** Replace `remove_leading_0s` (~18 lines). Estimated ~20 lines of new code. For small n, the prefix-AND can be done sequentially (still O(n) but with lower constant). For large n, use a tree.

**Verdict:** Similar to carry-lookahead — marginal at small n, essential at large n.

---

## Estimated Combined Impact

| Configuration | 3-digit layers | 10-digit layers (est.) |
|---------------|---------------|----------------------|
| Current | 20 | ~55 |
| + Carry-lookahead | ~19 | ~35 |
| + Parallel leading-zero | ~18 | ~30 |
| + Both | ~17 | ~28 |

The real payoff is at scale. For a 100-digit adder, these reduce the graph's critical path from O(n) to O(log n), which is the difference between ~200 layers and ~30.
