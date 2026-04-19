# TODO

## fp16 inference for Flash Attention

The render decode loop runs at ~35ms/step (GPU-bound). The bottleneck is SDPA
over long sequences (~800 positions): each step attends over an 800×800-ish
matrix using PyTorch's materialized ("math") softmax kernel.

**Why the math kernel is used:** PyTorch's SDPA dispatcher only enables Flash
Attention for fp16/bf16 inputs. All model weights and activations are currently
fp32, so every SDPA call falls back to the materialized kernel regardless of
whether an `attn_mask` is provided.

**The opportunity:** Casting weights and activations to fp16 at inference time
(e.g., `.half()` on `CompiledHeadless._net`) would let SDPA dispatch to Flash
Attention on A100. Flash Attention is O(N) in memory and substantially faster
for long sequences — potentially 3–5× faster SDPA, which could bring decode
from ~35ms to ~15–20ms per step.

The dynamic decode path (`forward_cached`, `is_causal=False`, no explicit mask)
already satisfies all Flash Attention preconditions except dtype. The static
KV cache path (added in this branch) uses a float `attn_mask` and would need
to switch to a boolean mask or be dropped in favour of the dynamic path.

**Risk:** Weights are compiled in fp32 and encode precise numerical thresholds
(comparison results, attention argmin/argmax patterns). fp16 has ~3 significant
decimal digits vs ~7 for fp32. It's unknown whether the compiled logic survives
the precision loss without correctness failures. A test that compares fp32 vs
fp16 `step_frame` outputs would answer this.

**Suggested experiment:**
1. `module._net.to(torch.float16)` after `compile_game()`
2. Cast inputs to fp16 in `_build_res_stream`
3. Run `make walkthrough ARGS="--frames 5"` and compare to fp32 reference
4. If pixel errors are within tolerance, fp16 is viable

Files: `torchwright/compiler/export.py` (`_build_res_stream`, `CompiledHeadless`),
`torchwright/compiler/components/attn.py` (weights), `torchwright/doom/compile.py`

## Move ceiling/floor into the transformer

The host currently decides ceiling vs floor per pixel (`ceil if y < center_y
else floor_c`). This is computation on the host side. The transformer should
emit ceiling/floor pixels itself — either as part of each render chunk (fill
non-wall rows within the chunk) or as a separate post-render pass.

Files: `torchwright/doom/compile.py` (step_frame render loop),
`torchwright/doom/game_graph.py` (render output)

## Clip column iteration to screen bounds

The state machine iterates col_lo..col_hi, which can include negative columns
and columns >= W (e.g., cols=[-2, 122) with W=120). The host safely skips
out-of-bounds columns, but each is still a full autoregressive step (~70ms).
At 120x100 with 4 full-screen walls, ~16 steps are wasted per frame.

Fix: clamp col_lo/col_hi to [0, W) in the graph, or have the state machine
skip out-of-bounds columns via the feedback loop.

Files: `torchwright/doom/game_graph.py` (state machine col iteration)

## Make graph-side done_flag work when N < max_walls

The graph checks `mask_sum > max_walls - 0.5`, which only fires when all
max_walls slots are masked. When the scene has fewer walls, the host
terminates via bit-count instead. The graph's done_flag computation runs on
every RENDER token but never triggers — wasted graph nodes.

Options:
- Feed N as a graph input so the threshold is dynamic
- Detect sentinel wall data (score=99) after all real walls are masked
- Accept the current host-side fix and remove the graph-side done_flag entirely

Files: `torchwright/doom/game_graph.py` (state_transitions),
`torchwright/doom/compile.py` (host-side termination)

## Tests that validate Asserts on the compiled graph

`check_asserts_on_compiled` (`torchwright/debug/probe.py:964`) already
runs each `Assert`'s predicate against the compiled transformer's
residual stream, but no tests currently call it. Asserts today only
fire during reference evaluation — the whole point of checking them
post-compile is to catch invariants that exact math satisfies but
compiled approximations violate (the class of bug Mode C was).

Add tests that exercise it on the DOOM graph and on smaller stage-level
graphs.  At minimum: collect the asserts via
`torchwright.graph.asserts.collect_asserts(output_node)` before
`compile_headless`, compile, then call `check_asserts_on_compiled` on
a representative input battery and assert no violations.

Files: `torchwright/debug/probe.py` (`check_asserts_on_compiled`,
`collect_asserts`), new tests under `tests/doom/` and
`tests/doom/stages/`.

## Softmax hardness assertion on `attend_*` primitives

Every `attend_*` op in `torchwright/ops/attention_ops.py`
(`attend_argmin`, `attend_argmax`, `attend_argmin_where`,
`attend_argmax_where`, `attend_argmin_above_integer`,
`attend_argmin_unmasked`, `attend_argmin_valid_unmasked`,
`attend_mean_where`, `attend_argmax_dot`) should accept a
`assert_hardness_gt=<float>` kwarg that checks, at reference-eval time
and on the compiled graph, that the softmax weight on the selected key
is at least that threshold.  E.g. `assert_hardness_gt=0.99` means
"≥99 % of the attention mass must land on the argmin/argmax key."

Today, concentration is verified indirectly via
`assert_distinct_across` / `assert_score_gap_at_least`
(`torchwright/graph/asserts.py:490,565`), which reason about
*score separation* and rely on the caller knowing the op's
`_QUERY_GAIN` to translate a margin into a hardness.  A direct
hardness check is (a) easier to reason about at the call site,
(b) closes the loop without the caller needing to know the gain,
and (c) catches post-compile precision loss that score-gap asserts
miss when the gap survives but the logits get squashed.

Files: `torchwright/ops/attention_ops.py` (add kwarg + hardness
probe), `torchwright/graph/asserts.py` (new
`assert_softmax_hardness`), `torchwright/debug/probe.py` (wire into
`check_asserts_on_compiled` path).
