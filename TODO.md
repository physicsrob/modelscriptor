# TODO

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
