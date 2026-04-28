"""MapSubset and supporting types — schema for a DOOM scene fixture.

These mirror the parent project's `MapSubset` / `BspNodeSubset` /
`Segment` data shapes. Pydantic models for JSON round-trip; matrices
and texture pixels are nested-list rather than numpy arrays so JSON
serialization is straightforward.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .game_state import GameState


class Segment(BaseModel):
    """A wall segment — line from `(ax, ay)` to `(bx, by)`.

    Color is RGB in [0, 1]. `texture_id` indexes the `MapSubset.textures`
    list (not yet schematized in the sandbox); -1 means no texture
    (fall back to solid color).
    """

    model_config = ConfigDict(extra="ignore")

    ax: float
    ay: float
    bx: float
    by: float
    color: tuple[float, float, float] = (1.0, 1.0, 1.0)
    texture_id: int = -1


class Texture(BaseModel):
    """A texture image — RGB pixel data, dimensions `(height, width)`.

    Pixels are organized as `pixels[y][x] = (r, g, b)`, all values in
    `[0, 1]`.
    """

    model_config = ConfigDict(extra="ignore")

    height: int
    width: int
    pixels: list[list[tuple[float, float, float]]]

    @model_validator(mode="after")
    def _check_shape(self) -> "Texture":
        if len(self.pixels) != self.height:
            raise ValueError(
                f"pixels has {len(self.pixels)} rows but height={self.height}"
            )
        for y, row in enumerate(self.pixels):
            if len(row) != self.width:
                raise ValueError(
                    f"pixels[{y}] has {len(row)} columns but width={self.width}"
                )
        return self


class BSPNode(BaseModel):
    """A BSP splitting plane: `nx*x + ny*y + d = 0`.

    Convention: `side_P = 1` if `nx*player_x + ny*player_y + d > 0`
    (player on FRONT side), else `0` (BACK). The plane normal `(nx, ny)`
    is conventionally a unit vector.
    """

    model_config = ConfigDict(extra="ignore")

    nx: float
    ny: float
    d: float


class MapSubset(BaseModel):
    """A transformer-ready slice of a DOOM map.

    Holds wall segments (sorted by distance from the player, closest
    first), BSP splitting planes, and precomputed BSP-rank coefficients
    that turn `side_P_vec` (an `M`-wide bit vector recording the
    player's side of every plane) into a per-segment integer rank.

    The rank formula is

        rank(s) = round(seg_bsp_coeffs[s] · side_P_vec + seg_bsp_consts[s])

    where `s` is a segment index, `seg_bsp_coeffs[s]` is a length-`M`
    vector, and `side_P_vec[i] ∈ {0, 1}` is the player's side of plane
    `i`. The precomputation bakes the BSP-tree topology and each
    segment's traversal path into the coefficients.

    Attributes
    ----------
    segments : list[Segment]
        Wall segments, distance-sorted (closest first).
    bsp_nodes : list[BSPNode]
        BSP splitting planes used by the rank computation.
    seg_bsp_coeffs : list[list[float]]
        Shape `(N, M)` — per-segment linear-combination weights for
        the rank formula. `N = len(segments)`, `M = len(bsp_nodes)`.
    seg_bsp_consts : list[float]
        Length `N` — per-segment constant term in the rank formula.
    scene_origin : tuple[float, float]
        Per-scene coordinate shift; subtracted from positions and the
        BSP plane `d`-coefficient before the graph sees them, then
        added back on outputs. Default `(0, 0)` (no shift).
    original_seg_indices : list[int]
        Segment indices in the original WAD's seg array, for cross-
        checking against reference traversal. Optional.
    test_poses : list[GameState]
        Recommended player states for sandbox tests. Each pose is far
        enough from every BSP plane that PWL-approximated `side_P`
        bits are robust against the framework's numerical noise — the
        fixture builder picks them so phase tests don't have to do
        their own geometric reasoning. Verify with
        `assert_pose_clear_of_planes` (in `doom_sandbox.fixtures`) if
        you roll your own. Optional; default empty.
    """

    model_config = ConfigDict(extra="ignore")

    segments: list[Segment]
    bsp_nodes: list[BSPNode]
    seg_bsp_coeffs: list[list[float]]
    seg_bsp_consts: list[float]
    scene_origin: tuple[float, float] = (0.0, 0.0)
    textures: list[Texture] = Field(default_factory=list)
    tex_name_to_id: dict[str, int] = Field(default_factory=dict)
    original_seg_indices: list[int] = Field(default_factory=list)
    test_poses: list[GameState] = Field(default_factory=list)

    @model_validator(mode="after")
    def _check_shapes(self) -> "MapSubset":
        n = len(self.segments)
        m = len(self.bsp_nodes)
        n_tex = len(self.textures)
        if len(self.seg_bsp_coeffs) != n:
            raise ValueError(
                f"seg_bsp_coeffs has {len(self.seg_bsp_coeffs)} rows but "
                f"there are {n} segments"
            )
        if len(self.seg_bsp_consts) != n:
            raise ValueError(
                f"seg_bsp_consts has length {len(self.seg_bsp_consts)} "
                f"but there are {n} segments"
            )
        for i, row in enumerate(self.seg_bsp_coeffs):
            if len(row) != m:
                raise ValueError(
                    f"seg_bsp_coeffs[{i}] has length {len(row)} but "
                    f"there are {m} BSP nodes"
                )
        for i, seg in enumerate(self.segments):
            if seg.texture_id != -1 and not (0 <= seg.texture_id < n_tex):
                raise ValueError(
                    f"segments[{i}].texture_id={seg.texture_id} is out of "
                    f"range for {n_tex} textures (use -1 for no texture)"
                )
        for name, idx in self.tex_name_to_id.items():
            if idx != -1 and not (0 <= idx < n_tex):
                raise ValueError(
                    f"tex_name_to_id[{name!r}]={idx} is out of range for "
                    f"{n_tex} textures (use -1 for not loaded)"
                )
        return self
