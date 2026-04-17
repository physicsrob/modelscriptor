"""Authoritative layout for the WALL→SORTED sort payload.

The WALL stage builds this payload per-wall token; the SORTED stage
retrieves it through ``attend_argmin_unmasked``.  Defining the layout
in one module prevents the packer (WALL) and unpacker (SORTED) from
silently disagreeing on offsets.

Layout
------
Offsets into the packed payload (all widths in floats):

    [0 .. 5)            wall geometry       (ax, ay, bx, by, tex_id)
    [5 .. 10)           render precomputed  (sort_den, C, D, E, H_inv)
    [10 .. 11)          bsp_rank            (the sort score — preserved in
                                             the payload so SORTED can
                                             forward it to downstream
                                             consumers like THINKING)
    [11 .. 13)          visibility columns  (vis_lo, vis_hi — screen-column
                                             range, gated to 0 for
                                             non-renderable walls)
    [13 .. 13+max_walls) position onehot    (sort-mask)

Total width: ``payload_width(max_walls) == 13 + max_walls``.
"""

from dataclasses import dataclass

from torchwright.graph import Concatenate, Node

from torchwright.doom.graph_utils import extract_from


# ---------------------------------------------------------------------------
# Section widths and offsets
# ---------------------------------------------------------------------------

GEOMETRY_WIDTH = 5
RENDER_WIDTH = 5
BSP_RANK_WIDTH = 1
VISIBILITY_WIDTH = 2

GEOMETRY_OFFSET = 0
RENDER_OFFSET = GEOMETRY_OFFSET + GEOMETRY_WIDTH
BSP_RANK_OFFSET = RENDER_OFFSET + RENDER_WIDTH
VISIBILITY_OFFSET = BSP_RANK_OFFSET + BSP_RANK_WIDTH
ONEHOT_OFFSET = VISIBILITY_OFFSET + VISIBILITY_WIDTH

# Sub-offsets within the geometry block.
GEOMETRY_FIELD_OFFSETS = {
    "ax": 0,
    "ay": 1,
    "bx": 2,
    "by": 3,
    "tex_id": 4,
}


def payload_width(max_walls: int) -> int:
    return ONEHOT_OFFSET + max_walls


# ---------------------------------------------------------------------------
# Pack / unpack
# ---------------------------------------------------------------------------


def pack_wall_payload(
    wall_ax: Node,
    wall_ay: Node,
    wall_bx: Node,
    wall_by: Node,
    wall_tex_id: Node,
    sort_den: Node,
    precomp_C: Node,
    precomp_D: Node,
    precomp_E: Node,
    precomp_H_inv: Node,
    bsp_rank: Node,
    vis_lo: Node,
    vis_hi: Node,
    position_onehot: Node,
) -> Node:
    """Concatenate per-WALL values into the canonical sort payload."""
    return Concatenate([
        wall_ax, wall_ay, wall_bx, wall_by, wall_tex_id,
        sort_den, precomp_C, precomp_D, precomp_E, precomp_H_inv,
        bsp_rank,
        vis_lo, vis_hi,
        position_onehot,
    ])


@dataclass
class UnpackedWallPayload:
    """Top-level sections of the packed wall payload.

    ``wall_data`` is 5-wide and can be further decomposed via
    ``extract_geometry_field``.  ``render_data`` is the 5-wide
    [sort_den, C, D, E, H_inv] block.  ``bsp_rank`` is 1-wide.
    ``vis_cols`` is 2-wide [vis_lo, vis_hi].
    ``onehot`` is max_walls-wide.
    """

    wall_data: Node
    render_data: Node
    bsp_rank: Node
    vis_cols: Node
    onehot: Node


def unpack_wall_payload(node: Node, max_walls: int) -> UnpackedWallPayload:
    """Split the packed payload into its logical sections."""
    d_total = payload_width(max_walls)
    return UnpackedWallPayload(
        wall_data=extract_from(
            node, d_total, GEOMETRY_OFFSET, GEOMETRY_WIDTH, "wall_data"),
        render_data=extract_from(
            node, d_total, RENDER_OFFSET, RENDER_WIDTH, "render_data"),
        bsp_rank=extract_from(
            node, d_total, BSP_RANK_OFFSET, BSP_RANK_WIDTH, "bsp_rank"),
        vis_cols=extract_from(
            node, d_total, VISIBILITY_OFFSET, VISIBILITY_WIDTH, "vis_cols"),
        onehot=extract_from(
            node, d_total, ONEHOT_OFFSET, max_walls, "onehot"),
    )


def extract_geometry_field(wall_data: Node, field: str) -> Node:
    """Pull a single 1-wide field from an unpacked ``wall_data`` block.

    ``field`` is one of ``"ax"``, ``"ay"``, ``"bx"``, ``"by"``, ``"tex_id"``.
    """
    if field not in GEOMETRY_FIELD_OFFSETS:
        raise ValueError(
            f"unknown geometry field: {field!r} "
            f"(expected one of {sorted(GEOMETRY_FIELD_OFFSETS)})"
        )
    return extract_from(
        wall_data, GEOMETRY_WIDTH, GEOMETRY_FIELD_OFFSETS[field], 1, f"geom_{field}"
    )
