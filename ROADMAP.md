# ModelScriptor Roadmap

The goal of this roadmap is to build a solid, proven foundation. Each phase
proves a specific capability of the compiler before moving on.


## Phase 1: End-to-End Adder

Fix the broken 1-digit adder test so it compiles a transformer and produces
correct output (1+1=2). This proves the full pipeline works: computation graph
→ constraint solver → weight assignment → forward pass → correct answer.

The test currently compiles successfully but the compute call uses the wrong
API (HeadlessTransformer vs Transformer).


## Phase 2: Compiler Hardening

Complete the unfinished parts of the compiler core:

- **Zero node compilation.** Zero nodes waste residual stream space. Compile
  them in attention layers to free up space. This removes the need for the
  "group strategy optimization" workaround noted in the old TODO list.

- **Embedding / deembedding compilation.** Currently handled separately from
  the main compilation pipeline. Integrate them so the full input → output
  flow goes through the compiler.

- **Separate tokenizer from embedding.** The embedding class currently bundles
  tokenization. These are distinct concerns.


## Phase 3: ONNX Export

Export compiled transformers to ONNX format. Verify the exported model runs
in ONNX runtime and produces identical output to the internal forward pass.

This is infrastructure that pays off for every subsequent demo — each one
automatically gets a portable artifact that anyone can run without
ModelScriptor installed.


## Phase 4: Intermediate Demos

A series of progressively harder compilations, each exercising different
compiler capabilities. If any expose bugs or scaling issues, fix them before
moving on.

- **Balanced parentheses checker.** Input a token sequence, output whether
  parentheses are balanced. Tests attention (looking back in sequence) and
  counting. A classic RASP example — proves we can do what Tracr did.

- **String pattern matcher.** Input a token sequence, output whether it matches
  a pattern. Tests attention, logic ops, and conditional branching.

- **Multi-digit adder (3+ digits).** Extends the Phase 1 adder with carry
  propagation across multiple positions. Stresses the compiler with a larger
  graph (~500+ nodes) and tests positional encoding.

- **Simple calculator.** Parse "3*4+2=" and output 14. Tests tokenization,
  operator precedence via attention, and arithmetic. Combines many primitives.

- **Sorting a short list.** Classic algorithmic task. Tests heavy use of
  attention (comparisons between positions) and multiple layers of computation.
