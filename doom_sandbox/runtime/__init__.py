"""Internal framework implementation. Not importable by phases.

The agent-facing API lives in `doom_sandbox.api`. This package holds
the framework's private machinery: noise simulation, embedding layout,
the autoregressive loop, etc. Phases must not import from here.
"""
