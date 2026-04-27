"""Generate sandbox fixtures from torchwright source-of-truth scenes.

Platform-side script — run once when scenes change. Imports
`torchwright.reference_renderer.scenes` (hand-authored scenes) and
`torchwright.doom.map_subset.build_scene_subset` (real BSP-rank
coefficient computation), then converts the result to the sandbox's
JSON schema and writes it under `doom_sandbox/fixtures/`.

Usage:

    uv run python -m scripts.build_sandbox_fixtures

The sandbox runtime never imports torchwright; the dependency is
contained to this script. The committed JSON is the single source of
truth at sandbox load time.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from doom_sandbox.types import BSPNode, MapSubset, Segment
from torchwright.doom.map_subset import build_scene_subset
from torchwright.reference_renderer.scenes import box_room_textured


_FIXTURE_DIR = Path(__file__).resolve().parent.parent / "doom_sandbox" / "fixtures"


def _to_sandbox_subset(ts) -> MapSubset:
    """Convert a torchwright `MapSubset` (numpy arrays) to the sandbox's
    pydantic `MapSubset` (nested lists, BSP coefficient matrix
    truncated to the actual BSP node count rather than padded to
    `max_bsp_nodes`)."""
    n_nodes = len(ts.bsp_nodes)
    coeffs = np.asarray(ts.seg_bsp_coeffs)[:, :n_nodes]
    consts = np.asarray(ts.seg_bsp_consts)
    # Strip texture_id (set to -1) — phase 1 doesn't use textures, and
    # carrying the texture atlas through the schema would inflate JSON
    # size. Later phases that need textures will regenerate fixtures
    # with the texture pixel data included.
    segments = [
        Segment(
            ax=float(s.ax),
            ay=float(s.ay),
            bx=float(s.bx),
            by=float(s.by),
            color=tuple(float(c) for c in s.color),
            texture_id=-1,
        )
        for s in ts.segments
    ]
    bsp_nodes = [BSPNode(nx=float(n.nx), ny=float(n.ny), d=float(n.d)) for n in ts.bsp_nodes]
    return MapSubset(
        segments=segments,
        bsp_nodes=bsp_nodes,
        seg_bsp_coeffs=coeffs.tolist(),
        seg_bsp_consts=consts.tolist(),
        scene_origin=tuple(float(v) for v in ts.scene_origin),
        original_seg_indices=list(ts.original_seg_indices),
        # Textures omitted: phase 1 doesn't use them, and pydantic's
        # default empty list keeps the JSON small.
    )


def _write(name: str, subset: MapSubset) -> None:
    path = _FIXTURE_DIR / f"{name}.json"
    path.write_text(subset.model_dump_json(indent=2))
    print(f"wrote {path} ({path.stat().st_size} bytes)")


def main() -> None:
    _FIXTURE_DIR.mkdir(parents=True, exist_ok=True)

    # box_room: 4 walls. build_scene_subset constructs a real balanced
    # BSP (3 internal nodes) and the rank-coefficient matrix.
    segments, textures = box_room_textured()
    ts = build_scene_subset(segments, textures, max_bsp_nodes=8)
    _write("box_room", _to_sandbox_subset(ts))


if __name__ == "__main__":
    main()
