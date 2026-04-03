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

### The three primitives

The transformer skip connection `out = in + f(in)` enables three operations:

| Operation | Mechanism | Cost |
|-----------|-----------|------|
| **Write** to free columns | `0 + f(in) = f(in)` | 1 attention head or FFN slots |
| **Add** into existing columns | `dead + live = Add(dead, live)` via skip | 1 attention head |
| **Cancel** a dead node | `v + (-v) = 0` | 1 attention head |

### What each sublayer computes

**Attention sublayer** (budget: d/d_head heads per layer):
- Attention nodes (cross-position lookups)
- Zero-bias Linear nodes (current-position attention applies the weight matrix)
- Add-into operations (reuses a dead addend's columns)
- Cancellations (frees columns for reuse)

**FFN sublayer** (budget: d internal slots):
- Linear → ReLU → Linear chains
- Standalone ReLU nodes
- Constants (via output bias, no slot cost)
- Bias terms for Linear nodes computed in the attention sublayer

### Column allocation

The compiler uses a simple greedy allocator -- no constraint solver needed.
Nodes are assigned to whatever columns are free at the time they're computed.
When a node is dead (all downstream consumers have been computed), its columns
are either cancelled (freeing them) or reused in-place by an Add operation.

### Pressure-aware scheduling

When free columns drop below 25% of the residual stream width, the scheduler
switches from critical-path priority to a pressure mode that prioritizes
operations which free columns (cancellations and add-into) over new
computations.


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
