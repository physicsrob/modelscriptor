"""DOOM sandbox — a constrained Python environment for designing
DOOM-rendering algorithms before porting to the transformer pipeline.

Phases import from `doom_sandbox.api`, `doom_sandbox.types`,
`doom_sandbox.fixtures`, and modules within their own phase. Framework-
internal modules (notably `doom_sandbox.runtime`) must not be imported
by phases.
"""
