![CI](https://github.com/physicsrob/modelscriptor/actions/workflows/ci.yml/badge.svg)

# ModelScriptor

ModelScriptor is a compiler that transforms computation graphs into transformer
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
each state, and the solver assigns each alive node to specific columns in the
residual stream at that state.


## The Residual Stream Allocation Problem

A transformer's residual stream is a fixed-width vector (say, 256 floats) that
carries all information between layers. Think of it as a shared whiteboard with
numbered columns. At every point between layers, everything the transformer
"knows" must be written somewhere on this whiteboard.

Each value in the computation graph occupies a contiguous range of columns. For
example, if your computation involves an input (10 floats), a linear
transformation result (10 floats), a ReLU result (10 floats), and a final
output (10 floats), each of these needs its own non-overlapping slice of the
residual stream.

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

**Minimize width.** We want the narrowest residual stream possible -- fewer
columns means a smaller, faster transformer.

### Why this is hard

With one layer and one state, you'd just pack things greedily. But a real
compilation involves:

- Multiple layers, each with multiple states (before attention, after
  attention, before FFN, after FFN)
- Skip connections that lock values to the same columns across states
- The same value appearing in many states (it stays alive until consumed)

These cross-state constraints create a coupled packing problem. A column
assignment that looks fine for one state might make another state impossible to
pack. The assignment must be solved globally, considering all states and
constraints simultaneously.

The compiler uses Google's OR-Tools CP-SAT solver to find valid allocations.
Each node is modeled as an interval variable, and the solver's native
`AddNoOverlap` constraint ensures no two nodes share columns within a state.


## Compilation: Strategies and the Beam Search

The compiler works backwards from the output node. At each layer, it asks:
"How should I use this sublayer (FFN or Attention) to compute the nodes I need?"

### What is a strategy?

A **strategy** is a plan for how a single sublayer computes one node. Each
sublayer is made of components wired in sequence:

- **FFN sublayer:** linear1 → relu → linear2 → skip
- **Attention sublayer:** attn → skip

For each node, each component offers possible ways to handle it:

- **Skip component:** Route the node through the residual skip connection
  (bypass the computation path), or if the node is an `Add`, decompose it
  into skip + computation. Always has options.
- **Attention component:** Compute an `Attn` node with an attention head.
  Pass any node through as identity (using an attention head that copies from
  the current position). Handle zero constants for free.
- **Linear component:** Apply the node's weight matrix, or pass through as
  identity.
- **ReLU component:** Apply ReLU to a `ReLU` node. No other options -- this
  component is a bottleneck for non-ReLU nodes passing through the FFN.

A strategy for one node is a choice at each component: "this node goes through
the skip connection" or "this node gets computed by attention head #3" etc.

### Combining strategies across nodes

A sublayer must handle **all** its assigned nodes simultaneously. If there are
22 nodes at a given layer, the compiler needs a joint strategy that satisfies
all constraint-solver requirements (no feature index conflicts, etc.).

`get_combined_strategies` does this via **beam search**. It processes nodes one
at a time:

1. Pick a node, take its top K strategies (sorted by a score that counts
   ancestor nodes -- simpler dependencies are preferred)
2. Recursively combine strategies for all remaining nodes
3. Cross-product the two sets, check each combination against the OR-Tools
   constraint solver
4. Keep the top K results

The beam width K (`max_strategies`, default 2) controls the trade-off between
compilation speed and reliability. Each node typically has 3-5 strategies, so
with 22 nodes the search space is ~4^22 ≈ 10^13. The beam search explores a
tiny fraction.

### Why this can fail

The beam search is sensitive to **node processing order**. Nodes are currently
processed in arbitrary dict-iteration order. If the globally-viable strategy
happens to rank 3rd for the first node processed, a beam width of 2 prunes it
and compilation fails with "No strategies found."

Empirically, for a 22-node case from a 2-digit adder:
- Random ordering at beam width 2: **1 in 20** orderings find a solution
- Beam width 3: **all 20** orderings find a solution
- "Fewest strategies first" ordering at beam width 2: succeeds deterministically

The cost of increasing beam width is linear -- each increment of K adds
roughly 80 constraint-solver calls (~2 seconds) at this graph size. But the
constraint solver itself gets slower as more layers accumulate constraints, so
full compilation time is super-linear in the number of layers.


## TODO List
This list is a non-exhaustive list of all the things that needs to be done before this project is relatively feature complete.

Immediately:
- Remove group strategy optimization -- it's only necessary because we aren't compiling zero nodes.
- Compile zero nodes to free up space -- can get away with just doing it in the attention layers
- Copy the output to the input in the last attention layer


- [ ] Implement embedding / deembedding compilation
- [ ] Separate tokenizer from embedding
- [ ] Export to ONNX

Recently complete:
- [x] Implement attn compilation in attention sublayer
- [x] Refactor compilation scoring
- [x] Include compilation statistics (number of parameters, efficiency, etc)
- [x] Better pass-through support for attention layers; We shouldn't require one head per zero.
