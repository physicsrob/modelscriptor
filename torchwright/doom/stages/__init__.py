"""Per-token-type stage builders for the DOOM game graph.

Each stage file defines:

* An ``<Stage>Inputs`` dataclass — everything the stage consumes.
* An ``<Stage>Outputs`` dataclass — everything the stage produces for
  downstream stages or the orchestrator.
* A public ``build_<stage>`` function implementing the stage's graph.
* Private ``_compute_*`` helpers for sub-computations.

Stages are composed in ``torchwright.doom.game_graph.build_game_graph``.
"""
