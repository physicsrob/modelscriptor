"""TEX_COL stage: per-position column one-hot for texture attention.

Every TEX_COL token carries one column of one texture.  The stage
turns ``tex_col_input`` (the host-fed column index within its texture)
into a one-hot over ``[0, tex_w)`` used as the attention key match
bits by the RENDER stage.
"""

from dataclasses import dataclass

from torchwright.graph import Node, annotate
from torchwright.ops.arithmetic_ops import add_const, bool_to_01
from torchwright.ops.map_select import in_range


@dataclass
class TexColToken:
    tex_col_input: Node  # host-fed column index (meaningful at TEX_COL)


@dataclass
class TexColKVOutput:
    tc_onehot_01: Node  # {0,1} one-hot over tex_w columns


def build_tex_col(token: TexColToken, *, tex_w: int) -> TexColKVOutput:
    with annotate("tex_col"):
        tc_p1 = add_const(token.tex_col_input, 1.0)
        tc_onehot_01 = bool_to_01(in_range(token.tex_col_input, tc_p1, tex_w))
    return TexColKVOutput(tc_onehot_01=tc_onehot_01)
