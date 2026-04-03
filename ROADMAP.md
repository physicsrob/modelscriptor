# ModelScriptor Roadmap

The goal of this roadmap is to build a solid, proven foundation. Each phase
proves a specific capability of the compiler before moving on.


## Phase 1: End-to-End Adder -- DONE

Fix the broken 1-digit adder test so it compiles a transformer and produces
correct output (1+1=2). This proves the full pipeline works: computation graph
→ scheduling → weight assignment → forward pass → correct answer.

The 3-digit adder also compiles and passes all arithmetic tests (202 nodes,
29 layers at d=1024).


## Phase 2: Compiler Hardening -- DONE

The original backward compiler (strategy search, beam search, CP-SAT solver)
was replaced entirely with a forward compiler that works from inputs toward
outputs. This eliminated zero-node accumulation, removed the constraint solver
dependency, and replaced beam search with greedy layer-by-layer scheduling.

- Embedding / deembedding handled via `CompiledTransformerModule` in the export
  pipeline.
- Tokenizer separated from Embedding into its own class.


## Phase 3: ONNX Export -- DONE

Compiled transformers export to ONNX format via `compile_to_onnx()`. Both the
adder and calculator have ONNX compilation targets in the Makefile (`make
compile`).


## Phase 4: Intermediate Demos

A series of progressively harder compilations, each exercising different
compiler capabilities.

- [x] **Multi-digit adder (3+ digits).** Carry propagation across positions.
  Stresses the compiler with 200+ nodes and tests positional encoding.

- [x] **Simple calculator.** Parses "3*4+2=" and computes the answer. Supports
  +, -, *. Tests tokenization, operator dispatch, and arithmetic. Compiles to
  38 layers at d=2048.

- [ ] **Balanced parentheses checker.** Input a token sequence, output whether
  parentheses are balanced. Tests attention (looking back in sequence) and
  counting.

- [ ] **String pattern matcher.** Input a token sequence, output whether it
  matches a pattern. Tests attention, logic ops, and conditional branching.

- [ ] **Sorting a short list.** Classic algorithmic task. Tests heavy use of
  attention (comparisons between positions) and multiple layers of computation.
