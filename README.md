![CI](https://github.com/physicsrob/torchwright/actions/workflows/ci.yml/badge.svg)

# TorchWright

TorchWright is a compiler that transforms computation graphs into transformer
neural network weights. You define what you want the transformer to compute
using high-level operations (arithmetic, logic, attention, table lookups), and
the compiler produces actual weight matrices that run in a standard transformer
architecture.

The key insight is that transformers are not just learnable architectures --
they are a fixed computational substrate that can be *programmed* by setting
weights directly, without any training.


## Getting Started

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then:

```bash
uv sync          # install dependencies
make test        # run tests
make lint        # run black + mypy
```


## Examples

TorchWright has been used to compile several non-trivial programs into
transformer weights:

**Multi-digit adder** (`examples/adder.py`) -- Parses "123+456=" and outputs
"579" autoregressively. Digit-by-digit addition with carry propagation, exactly
like pencil-and-paper arithmetic. The 3-digit version compiles to a 29-layer
transformer with d=1024.

**Calculator** (`examples/calculator.py`) -- Supports +, -, * on positive
integers up to 3 digits. Subtraction handles negative results; multiplication
uses long multiplication with partial product rows. Compiles to 38 layers with
d=2048.

Both examples can be compiled to ONNX for portable inference:

```bash
make compile     # produces adder.onnx and calculator.onnx
```


## Key Concepts

### Nodes

A node represents a computed value in the computation graph. It *is* its
output -- a `Linear` node that produces 10 floats *is* those 10 floats.
Nodes know their size (number of output floats) and their inputs (other
nodes they depend on).

For example:

```
input_node         →  10 floats  (no inputs -- comes from the user)
linear_result      →  10 floats  (input: input_node)
relu_result        →  10 floats  (input: linear_result)
```

The computation graph is just nodes pointing to their inputs. The compiler's
job is to figure out how to pack all these values into a transformer's residual
stream and set the weights so the transformer computes them correctly.

### States

A state is a label for a point in the transformer's forward pass. It names a
specific moment -- "the residual stream right after attention finishes" or "the
residual stream right before the FFN layer." It is not the vector itself, and
it is not a slice of the vector. It's a label for *when*.

A simple 1-layer transformer has states at each boundary:

```
→ attn_in_state    (residual stream before attention)
    [attention runs]
→ ffn_in_state     (residual stream after attention, before FFN)
    [FFN runs]
→ ffn_out_state    (residual stream after FFN)
```

At each state, some set of nodes are **alive** -- their values are sitting in
the residual stream, either because they were just computed or because they're
being passed through for later use. The compiler tracks which nodes are alive at
each state, and assigns each alive node to specific columns in the residual
stream at that state.


## The Residual Stream Allocation Problem

A transformer's residual stream is a fixed-width vector (say, 1024 floats) that
carries all information between layers. Think of it as a shared whiteboard with
numbered columns. At every point between layers, everything the transformer
"knows" must be written somewhere on this whiteboard.

Each value in the computation graph occupies a set of columns. For example, if
your computation involves an input (10 floats), a linear transformation result
(10 floats), a ReLU result (10 floats), and a final output (10 floats), each
of these needs its own non-overlapping columns in the residual stream.

### The rules

**No overlaps.** Two values that are alive at the same point in the network
can't share columns. If both `input` and `linear_result` exist in the residual
stream at the same time, they need different columns.

**Skip connections preserve positions.** A transformer's skip connection is just
addition: `x + f(x)`. It doesn't rearrange anything. So if `input` is at
columns [0-9] before the FFN layer, it's still at columns [0-9] after the skip
connection. Any value that flows through a skip connection must keep its column
assignment.

**Concatenation is logical, not physical.** When the computation graph
concatenates two values (e.g., to feed them into a Linear node), they do *not*
need to be adjacent in the residual stream. The compiler scatters the Linear
node's weight matrix to whatever columns the inputs occupy. This is a key
simplification -- it means concatenation adds no allocation constraints.

**Columns need not be contiguous.** A node's columns can be scattered anywhere
in the residual stream. The weight writer gathers and scatters via index lists,
so physical adjacency is never required. This eliminates fragmentation entirely.


## How Compilation Works

The compiler works **forward** from inputs to outputs, building the transformer
one layer at a time. At each layer, a scheduler decides what to compute using
the attention sublayer and FFN sublayer, then a weight writer sets the
corresponding weight matrices.

### The skip connection

Every sublayer in a transformer computes `out = in + f(in)`. The sublayer
output gets *added* to whatever is already in the residual stream. This single
fact determines how values enter, persist, and leave:

- **Values persist automatically.** They are the `in` that passes through
  unchanged. A value written at layer 3 is still there at layer 30 unless
  something explicitly removes it.

- **A sublayer can only add.** It cannot overwrite or rearrange columns. What
  happens depends on what is already in the target columns:
  - If the columns are **empty** (zeroed out), the new value appears:
    `0 + new = new`. This is how new computations enter the residual stream.
  - If the columns hold a **dead value** (nothing downstream needs it anymore),
    write its negation: `v + (-v) = 0`. This frees those columns for reuse.
  - If the columns hold a value you want to **add to**, write the other addend
    and the skip connection produces the sum. This is how the compiler
    implements Add nodes without using any extra columns.

### What each sublayer computes

**Attention.** Each attention head applies a weight matrix and writes to
d_head columns -- so anything that is a matrix multiply or a targeted write
to specific columns maps here. This includes:
- Attention nodes (cross-position lookups -- the thing only attention can do)
- Linear nodes without bias (a head applies the weight matrix at the
  current position)
- Column management: cancellations and additions into existing columns are
  just targeted writes with the right values

**FFN.** The FFN sublayer's internal structure is Linear → ReLU → Linear,
so it directly handles graph operations with the same shape:
- Linear → ReLU → Linear chains
- Standalone ReLU nodes
- Constants (via the output bias)
- Bias terms for Linear nodes that were computed in the attention sublayer

### Scheduling

The compiler picks what to compute at each layer based on what is ready (all
inputs alive in the residual stream) and what is most urgent (longest
dependency chain to an output). When free columns run low, the scheduler
shifts to prioritizing operations that free space -- cancellations and
additions into existing columns -- over new computations. Column assignment
is greedy: nodes get whatever columns are free when they are computed.


## The Ops Layer

The `torchwright/ops/` package provides high-level operations built
on top of the raw graph nodes. These are what examples typically use:

- **Arithmetic**: `add_scalar`, `negate`, `subtract`, `multiply_scalar`
- **Logic**: `equals_vector`, `bool_not`, `bool_all_true`, `bool_any_true`
- **Table lookups**: `map_to_table` -- maps an embedding-valued input to an
  embedding-valued output via an FFN lookup table
- **Selection**: `select` (if/else), `switch` (multi-way dispatch)
- **Sequence operations**: `output_sequence`, `remove_leading_0s`
- **Embedding arithmetic**: `sum_digits`, `sum_digit_seqs` -- digit-by-digit
  addition with carry propagation in embedding space


## Architecture

```
torchwright/
├── graph/              # Computation graph: nodes, embeddings, attention, etc.
├── ops/                # High-level operations (arithmetic, logic, tables)
└── compiler/
    ├── forward/        # The forward compiler
    │   ├── compile.py        # Main entry point (forward_compile)
    │   ├── graph_analysis.py # Topological order, consumers, critical paths
    │   ├── residual_map.py   # Greedy column allocator
    │   ├── scheduler.py      # Layer-by-layer operation scheduling
    │   └── weight_writer.py  # Writes weight matrices into transformer layers
    ├── components/     # Transformer component abstractions (attn, linear, relu)
    ├── groups/         # Sublayer and layer groupings
    ├── export.py       # ONNX export
    └── transformer.py  # HeadlessTransformer (the compiled artifact)
```
