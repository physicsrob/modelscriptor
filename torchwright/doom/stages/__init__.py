"""Per-token-type stage builders for the DOOM game graph.

Each stage file defines up to four dataclasses:

* ``<Stage>Token`` — the discrete fields the host sets at this position
  (scene data for prefill tokens, overlay copies for autoregressive).
* ``<Stage>KVInput`` — data arriving from other positions via attention
  (broadcasts, argmin/argmax lookups).  Omitted for stages that don't
  read from other positions.
* ``<Stage>TokenOutput`` — overlay + overflow values the host reads
  from this position.  Omitted for stages whose output is only visible
  to other positions through attention.
* ``<Stage>KVOutput`` — values this stage makes available for other
  positions to attend to.  Omitted for stages that aren't attended to.

Token-type flags (``is_wall``, ``is_render``, etc.) and
``pos_encoding`` are direct keyword arguments to ``build_<stage>``,
not dataclass fields.

Stages are composed in ``torchwright.doom.game_graph.build_game_graph``.
"""
