# The Residual Stream Allocation Solver

## The Problem

The compiler must assign every computed value (node) to specific columns in the
transformer's residual stream at every point in the forward pass (state) where
that value is alive. The assignment must satisfy two types of constraints:

1. **Non-overlapping.** Two nodes alive at the same state can't share columns.

2. **Skip connection pinning.** A value that passes through a skip connection
   keeps its column assignment. Since skip connections are just addition, they
   can't rearrange the residual stream.

The solver takes a `FeatureAssignmentConstraints` (the problem) and returns a
`FeatureAssignment` (the solution: a mapping from every (state, node) pair to a
list of column indices), or `None` if no valid assignment exists.


## What about concatenation?

Concatenation is a **logical** grouping, not a physical layout constraint. When
`Concatenate([A, B])` feeds into a `Linear` node, A and B do not need to be
adjacent in the residual stream. The compiler scatters the Linear node's weight
matrix to whatever columns A and B occupy.

This was a key simplification made during development. The original system
treated concatenation as a physical adjacency constraint, which required Z3's
`ForAll` quantifier to solve and caused the solver to hang on graphs with ~100
nodes. Recognizing that the transformer hardware doesn't require adjacency
eliminated the hardest constraint entirely.

The one exception: when a `Concatenate` node appears as a skip connection input
(element-wise addition requires matching columns), the children must be
contiguous to match the single output node. This is a simple contiguity
constraint handled by OR-Tools without any special machinery.


## Current implementation

### OR-Tools CP-SAT Solver (`solve()`)

The sole solver. Uses Google's CP-SAT constraint programming engine.

- Creates interval variables for each (state, node) pair
- Uses native `AddNoOverlap()` for non-overlapping constraints within each state
- Expresses skip pinning as start-position equality between states
- Minimizes total width as an optimization objective
- Handles the rare skip-with-Concatenate case via simple contiguity constraints

**Pros:** Fast. Native interval scheduling designed for this class of problem.
Produces optimal (minimum-width) solutions. Handles all remaining constraint
types.

**Cons:** Solves the entire network in one call, which may become slow for very
large graphs (thousands of nodes). The adder test (~100 nodes) takes ~12 seconds.


## Historical context

The project previously used Z3 (an SMT solver) to handle a more complex
constraint set that included physical concatenation alignment. That constraint
required Z3's `ForAll` quantifier, which made the solver hang on ~100 nodes.

The progression was:
1. OR-Tools (original) -- fast but couldn't express concat-vs-concat alignment
2. Z3 (migration) -- expressive but couldn't scale
3. Logical concatenation reformulation -- eliminated the hard constraint
4. OR-Tools (current) -- fast and sufficient for all remaining constraints

The argument files (`argument1.md`, `argument2.md`, and their rebuttals) capture
the solver-vs-heuristic debate that preceded the reformulation. The debate became
moot once concatenation was made logical.


## Scaling considerations

The current single-call solver may struggle at Doom-scale (thousands of nodes).
Options if this becomes a bottleneck, in order of effort:

1. **Disable optimization.** Tell OR-Tools to find any feasible solution instead
   of minimizing width. Compact unused columns afterward. This is one line of
   code.

2. **Hierarchical decomposition.** Solve each layer independently. Nodes that
   cross layer boundaries (via skip connections) get fixed column assignments
   from the previous layer. Each solve call handles ~10-20 nodes instead of
   hundreds.

3. **Greedy heuristic allocator.** Walk states in order, assign nodes to lowest
   free columns, respect skip pinning. O(N) time. The remaining constraints are
   simple enough that this would be correct without backtracking.


## Architecture

The solver interface is:

```python
def solve(
    constraints: FeatureAssignmentConstraints, max_d: int = 1000
) -> Optional[FeatureAssignment]
```

The `FeatureAssignmentConstraints` describe the problem declaratively. The
`FeatureAssignment` maps every (state, node) pair to column indices.
`check_solution()` validates any solution regardless of how it was produced.

This decoupling means the solver is pluggable -- a heuristic allocator with
the same interface could be swapped in without changing any other code.
