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

**Concatenations must be contiguous.** When the computation graph concatenates
two values, they must occupy adjacent columns in the residual stream, and the
combined range must match across states.

**Minimize width.** We want the narrowest residual stream possible -- fewer
columns means a smaller, faster transformer.

### Why this is hard

With one layer and one state, you'd just pack things greedily. But a real
compilation involves:

- Multiple layers, each with multiple states (before attention, after
  attention, before FFN, after FFN)
- Skip connections that lock values to the same columns across states
- The same value appearing in many states (it stays alive until consumed)
- Concatenation constraints that force groups of values to be contiguous

These cross-state constraints create a coupled packing problem. A column
assignment that looks fine for one state might make another state impossible to
pack. The assignment must be solved globally, considering all states and
constraints simultaneously.

This is the same class of problem as **register allocation** in a traditional
compiler, where CPU registers must be assigned to program variables subject to
liveness and interference constraints. Register allocation is NP-hard.


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
