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

Two implementations: an **embedding-space** version (V1, consistent with the adder) and a **numeric-scalar** version (V2, exploring the alternative representation). Both share parsing, output formatting, and test infrastructure.

## Architecture

| File | Purpose |
|------|---------|
| `examples/calculator.py` | V1: embedding-space calculator |
| `examples/calculator_scalar.py` | V2: numeric-scalar calculator |
| `tests/compile/forward/test_forward_calculator.py` | Tests for both versions |

Both import shared infrastructure from `examples/adder.py`: `NumericSequence`, `check_is_digit`, `output_sequence`, `remove_leading_0s`.

## Parsing (shared by V1 and V2)

Reuse `NumericSequence` from adder for digit tracking. Detect operator and equals:

```python
is_plus  = compare_to_vector(embedding, embedding.get_embedding("+"))
is_minus = compare_to_vector(embedding, embedding.get_embedding("-"))
is_times = compare_to_vector(embedding, embedding.get_embedding("*"))
is_operator = bool_any_true([is_plus, is_minus, is_times])
is_equals = compare_to_vector(embedding, embedding.get_embedding("="))

# Capture which operator was used (latched at the = position)
which_plus  = pos_encoding.get_prev_value(is_plus, is_equals)
which_minus = pos_encoding.get_prev_value(is_minus, is_equals)
which_times = pos_encoding.get_prev_value(is_times, is_equals)
```

Extract operands using `is_operator` as the end-of-first-number event and `is_equals` as end-of-second-number (same as adder but with `is_operator` instead of `is_end_of_first_num`).

## Operation Dispatch (shared by V1 and V2)

Compute all three results in parallel, select with `switch`:

```python
result = switch(
    [which_plus, which_minus, which_times],
    [add_result, sub_result, mul_result],
)
```

The transformer can't branch — all paths execute regardless. `switch` picks the right output at O(1) depth.

## Result Sizing (shared by V1 and V2)

- Addition: up to `max_digits + 1` result digits (carry overflow)
- Subtraction: up to `max_digits` result digits (+ optional `-` sign)
- Multiplication: up to `2 * max_digits` result digits

All results are zero-padded to `2 * max_digits + 1` positions, then leading zeros are removed.

---

# V1: Embedding-Space Calculator

All digit manipulation uses embedding-valued nodes and `map_to_table` lookups, same as the adder. Every arithmetic step is a table lookup.

## V1 Phase 1: Skeleton + Addition

**Goal:** Working calculator that only does addition.

1. Parse input, detect operator, extract digit embeddings
2. Compute addition result using `sum_digit_seqs` from adder
3. Pad to `2 * max_digits + 1` with zero-embedding, remove leading zeros, append `<eos>`
4. `output_sequence` to emit result after `=`

**Tests:** `1+1=` → `2`, `123+456=` → `579`, `0+0=` → `0`

## V1 Phase 2: Subtraction

### Step 1: Digit comparison

Compare operands digit-by-digit from MSB to LSB:

```python
def compare_digit_pair(embedding, a, b):
    """Returns 1.0 if a > b, -1.0 if a < b, 0.0 if a == b."""
    return map_to_table(
        concat([a, b]),
        {key: torch.tensor([1.0 if i > j else (-1.0 if i < j else 0.0)])
         for i in range(10) for j in range(10) ...},
        default=torch.tensor([0.0]),
    )
```

### Step 2: Sign propagation

Fold from MSB to LSB: the first non-equal digit determines the overall comparison. Sequential O(n) depth, same structure as carry propagation.

```python
# Fold: if previous comparison is nonzero, keep it; else take current
result = cmp_digits[0]
for cmp in cmp_digits[1:]:
    result = map_to_table(concat([result, cmp]), ...)  # if result != 0: result, else: cmp
is_a_gte_b = compare(result, thresh=0.0)
```

### Step 3: Conditional swap

If `a < b`, swap operands so we always subtract the larger minus the smaller:

```python
first  = [select(is_a_gte_b, a_i, b_i) for ...]
second = [select(is_a_gte_b, b_i, a_i) for ...]
```

### Step 4: Digit subtraction with borrow

Same structure as `sum_digit_seqs` but with subtraction tables:

```python
def subtract_digits(embedding, num1, num2, borrow_in):
    # For A, B in 0-9, borrow in {0,1}:
    #   result = (A - B - borrow) mod 10
    #   borrow_out = 1 if (A - B - borrow) < 0 else 0
```

### Step 5: Output sign

If `a < b`, prepend `-` embedding to the output sequence. Leading zero removal handles the `0` placeholder when the result is positive.

**Tests:** `5-3=` → `2`, `456-123=` → `333`, `100-100=` → `0`, `1-5=` → `-4`, `100-999=` → `-899`

## V1 Phase 3: Multiplication

Uses partial product rows, then adds rows using `sum_digit_seqs`.

### Step 1: Digit×digit products

For each pair (a_i, b_j), compute the product as (tens_digit, ones_digit) embeddings:

```python
def multiply_digit_pair(embedding, a, b):
    """Two map_to_table lookups: one for tens, one for ones."""
```

For max_digits=n, this produces n² products, all in parallel.

### Step 2: Partial product rows

Each digit b_j of the second operand produces a row: `b_j × A`, which is n+1 digits (n digit×digit products with carry propagation). This uses the same right-to-left carry fold as addition, but starting from the digit products.

For max_digits=3:
```
Row 0: b_0 × [a_2, a_1, a_0]  →  4 digits (shifted by 0)
Row 1: b_1 × [a_2, a_1, a_0]  →  4 digits (shifted by 1)
Row 2: b_2 × [a_2, a_1, a_0]  →  4 digits (shifted by 2)
```

Each row has carry propagation (O(n) sequential depth). Rows are independent so their internal carry chains can run in parallel.

### Step 3: Sum partial product rows

Add rows pairwise using `sum_digit_seqs`:
```
temp = sum_digit_seqs(row_0_padded, row_1_padded)
result = sum_digit_seqs(temp, row_2_padded)
```

Each addition is O(n) sequential depth. For n rows, this is O(n) total (or O(log n) with tree reduction).

### Step 4: Result formatting

Pad to `2 * max_digits`, remove leading zeros, append `<eos>`.

**Tests:** `2*3=` → `6`, `9*9=` → `81`, `12*34=` → `408`, `123*456=` → `56088`, `0*5=` → `0`

## V1 Phase 4: Integration

All three operations compiled into one network. Autoregressive tests with mixed operations. Resource usage logging.

**Compile targets:** `d=1024, d_head=16`.

---

# V2: Numeric-Scalar Calculator

Alternative representation: convert digit embeddings to scalar values (0.0–9.0) early, do arithmetic with scalar node operations, convert back to embeddings at the end. This explores whether a numeric representation produces a smaller/shallower graph, particularly for multiplication.

## V2 Representation

```python
def embed_to_scalar(embedding, digit_node):
    """Convert digit embedding → scalar node (value 0.0-9.0)."""
    return map_to_table(digit_node,
        {embedding.get_embedding(str(i)): torch.tensor([float(i)])
         for i in range(10)},
        default=torch.tensor([0.0]))

def scalar_to_embed(embedding, scalar_node):
    """Convert scalar node (integer 0-9) → digit embedding."""
    return map_to_table(scalar_node,
        {torch.tensor([float(i)]): embedding.get_embedding(str(i))
         for i in range(10)},
        default=embedding.get_embedding("0"))
```

Two additional utility tables for extracting digits from sums:

```python
def divmod10(scalar_node, max_val=50):
    """Returns (quotient, remainder) for integer scalar node."""
    digit = map_to_table(scalar_node,
        {torch.tensor([float(v)]): torch.tensor([float(v % 10)])
         for v in range(max_val + 1)}, ...)
    carry = map_to_table(scalar_node,
        {torch.tensor([float(v)]): torch.tensor([float(v // 10)])
         for v in range(max_val + 1)}, ...)
    return carry, digit
```

## V2 Phase 1: Skeleton + Addition

Same parsing as V1. After extracting digit embeddings, convert to scalars:

```python
a_scalars = [embed_to_scalar(embedding, d) for d in a_digits]
b_scalars = [embed_to_scalar(embedding, d) for d in b_digits]
```

Addition with carry:
```python
carry = create_constant(torch.tensor([0.0]))
result_scalars = []
for a_i, b_i in reversed(zip(a_scalars, b_scalars)):
    total = sum_nodes([a_i, b_i, carry])  # scalar addition, no lookup!
    carry, digit = divmod10(total)
    result_scalars.append(digit)
```

Convert back: `result_embeds = [scalar_to_embed(embedding, s) for s in result_scalars]`

**Key difference from V1:** The digit addition is `sum_nodes` (a Linear node) instead of a 200-entry `map_to_table`. The `divmod10` is still a lookup, but only ~20 entries (for sums 0-19) instead of 200.

## V2 Phase 2: Subtraction

Comparison is simpler in scalar space:

```python
# Per-digit comparison: just subtract
cmp_i = subtract(a_scalar_i, b_scalar_i)  # positive if a > b
```

Sign propagation still needs a fold, but the fold table is simpler (keyed on scalar differences instead of embedding pairs).

Subtraction with borrow:
```python
diff = subtract(a_i, subtract(b_i, borrow))  # scalar subtraction
# If diff < 0, add 10 and set borrow = 1
is_negative = compare(diff, thresh=-0.5)
digit = select(is_negative, add_scalar(diff, 10.0), diff)
borrow = select(is_negative, create_constant(torch.tensor([1.0])),
                              create_constant(torch.tensor([0.0])))
```

Wait — this still needs `select`, which isn't free. But the overall structure is cleaner: no 200-entry subtraction table.

## V2 Phase 3: Multiplication

This is where scalar space really shines. Column accumulation becomes trivial:

```python
# Compute all n² digit products as scalars (parallel)
products = {}
for i in range(n):
    for j in range(n):
        products[i, j] = multiply_digit_pair_scalar(a_scalars[i], b_scalars[j])
        # This is still a map_to_table (100 entries) → scalar product (0-81)

# Accumulate columns with plain addition
for col in range(2 * n):
    contributors = [products[i, j] for i, j where i + j == col]
    column_sum = sum_nodes(contributors + [carry])  # one Linear, no lookup chain!
    carry, result_scalars[col] = divmod10(column_sum, max_val=...)
```

**Key advantage:** Column accumulation is a single `sum_nodes` per column — O(1) depth regardless of contributor count. In V1, accumulating a column with 5 contributors requires 4 sequential `sum_digits` lookups.

The `divmod10` lookup handles larger ranges (column sums up to ~45 + carry), so tables have ~50 entries instead of ~200.

## V2 Phase 4: Integration

Same as V1. Compare the two implementations:
- Node count
- Layer count
- Peak column usage
- Compilation time

---

## Estimated Complexity Comparison (max_digits=3)

| | V1 (embedding) | V2 (scalar) |
|---|---|---|
| **Representation** | Digit embeddings (8-dim) | Scalar nodes (1-dim) |
| **Addition depth** | O(n) × map_to_table | O(n) × (sum + divmod10) |
| **Subtraction depth** | O(n) compare + O(n) borrow | O(n) compare + O(n) borrow |
| **Multiply accumulation** | O(n) per column (repeated sum_digits) | O(1) per column (sum_nodes) |
| **Multiply total depth** | O(n²) accumulate + O(n) carry | O(n) products + O(n) carry |
| **Largest lookup tables** | 200 entries (digit+digit+carry) | 100 entries (digit×digit) |
| **Extra overhead** | None | embed↔scalar conversion (20 entries × 2n digits) |
| **Est. total nodes** | ~530 | ~400 (fewer accumulation nodes) |
| **Est. layers** | ~40-50 | ~30-40 (shallower multiply) |

V2's main advantage is multiplication: column accumulation is O(1) instead of O(n) depth per column. V1's advantage is simplicity and consistency with the adder — no representation boundary.

## Implementation Order

1. **V1 Phase 1-4** — embedding-space calculator, all operations
2. **V2 Phase 1-4** — scalar-space calculator, same tests, compare results
3. **Analysis** — compare node counts, layer counts, compilation stats

## Phased Testing

Tests are shared — both versions must produce identical outputs for all inputs:

**Phase 1 (Addition):** `1+1=` → `2`, `123+456=` → `579`, `0+0=` → `0`

**Phase 2 (Subtraction):** `5-3=` → `2`, `456-123=` → `333`, `100-100=` → `0`, `1-5=` → `-4`, `100-999=` → `-899`

**Phase 3 (Multiplication):** `2*3=` → `6`, `9*9=` → `81`, `12*34=` → `408`, `123*456=` → `56088`, `0*5=` → `0`

**Phase 4 (Integration):** Mixed operations, autoregressive, resource logging, V1 vs V2 comparison.
