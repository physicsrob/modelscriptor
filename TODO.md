# TODO

## Tests that validate Asserts on the compiled graph

`check_asserts_on_compiled` (`torchwright/debug/probe.py:964`) already
runs each `Assert`'s predicate against the compiled transformer's
residual stream, but no tests currently call it. Asserts today only
fire during reference evaluation — the whole point of checking them
post-compile is to catch invariants that exact math satisfies but
compiled approximations violate.

Add tests that exercise it on representative compiled graphs (the
example transformers under `examples/`, plus stage-level synthetic
graphs).  At minimum: collect the asserts via
`torchwright.graph.asserts.collect_asserts(output_node)` before
`compile_headless`, compile, then call `check_asserts_on_compiled` on
a representative input battery and assert no violations.

Files: `torchwright/debug/probe.py` (`check_asserts_on_compiled`,
`collect_asserts`), new tests under `tests/`.

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
