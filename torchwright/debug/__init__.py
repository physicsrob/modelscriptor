"""Debug tooling for torchwright graphs.

:mod:`probe` provides a divergence locator for compile-vs-semantics
bugs: it runs a compiled :class:`HeadlessTransformer` alongside a
direct recursive evaluation of the source graph and reports the first
graph node whose compiled value diverges from the reference.
"""
