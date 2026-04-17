"""Extract textures, map geometry, and BSP data from DOOM WAD files.

The WAD (Where's All the Data?) format stores DOOM's game assets:
textures, maps, sounds, etc.  This module handles:

- Texture extraction: composite multi-patch textures from the WAD's
  column-based picture format using the PLAYPAL palette.
- Map geometry: parse a map's VERTEXES, LINEDEFS, SIDEDEFS, SECTORS
  lumps into structured dataclasses.
- BSP tree: parse a map's SEGS, SSECTORS, NODES lumps for
  front-to-back rendering order.

Textures are returned as ``(width, height, 3)`` float64 numpy arrays
in column-major layout (matching the renderer's convention).

Map geometry uses DOOM's native integer coordinate system (int16).
Vertex coordinates typically range from about -4000 to +4000.
"""

import colorsys
import struct
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# High bit of a BSP node's child index indicates a subsector (leaf).
# Clear this bit to get the subsector index; otherwise the index points
# to another BSP node.
SUBSECTOR_FLAG = 0x8000

# Lump names that belong to a map (follow the map marker lump).
_MAP_LUMP_NAMES = frozenset(
    {
        "THINGS",
        "LINEDEFS",
        "SIDEDEFS",
        "VERTEXES",
        "SEGS",
        "SSECTORS",
        "NODES",
        "SECTORS",
        "REJECT",
        "BLOCKMAP",
    }
)


# ---------------------------------------------------------------------------
# Map data dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Vertex:
    """A point in the map's 2D coordinate space (int16 per axis)."""

    x: int
    y: int


@dataclass(frozen=True)
class Linedef:
    """An authored wall between two vertices.

    ``front_sidedef`` and ``back_sidedef`` are indices into SIDEDEFS,
    or ``-1`` if unused.  A one-sided wall (like a solid room wall)
    has ``back_sidedef == -1``.
    """

    v1: int
    v2: int
    flags: int
    special: int
    tag: int
    front_sidedef: int
    back_sidedef: int


@dataclass(frozen=True)
class Sidedef:
    """Visual properties for one side of a linedef.

    ``upper``, ``lower``, ``middle`` are texture names (or ``"-"`` for
    none).  ``sector`` is the index into SECTORS that this side faces.
    """

    x_offset: int
    y_offset: int
    upper: str
    lower: str
    middle: str
    sector: int


@dataclass(frozen=True)
class Sector:
    """A floor/ceiling region with uniform lighting.

    Floor and ceiling textures are *flat* names (64x64 flat lumps),
    not wall textures.
    """

    floor_h: int
    ceiling_h: int
    floor_tex: str
    ceiling_tex: str
    light: int
    special: int
    tag: int


@dataclass(frozen=True)
class Seg:
    """A BSP-split wall segment.

    Each SEG is a fragment of a LINEDEF after BSP partitioning.  It's
    what DOOM actually renders (as opposed to whole linedefs).

    ``side`` is 0 for the front side of the linedef, 1 for the back.
    ``angle`` is the seg's orientation as a BAM (binary angle, 0-65535
    mapped over a full turn).
    """

    v1: int
    v2: int
    angle: int
    linedef: int
    side: int
    offset: int


@dataclass(frozen=True)
class Subsector:
    """A BSP leaf: a convex region made up of a contiguous run of SEGS.

    ``first_seg`` and ``seg_count`` describe the range in SEGS that
    belongs to this subsector.
    """

    seg_count: int
    first_seg: int


@dataclass(frozen=True)
class BspNode:
    """An internal BSP tree node.

    The splitting line passes through ``(px, py)`` with direction
    ``(dx, dy)``.  ``front_child`` and ``back_child`` are encoded
    indices: if the ``SUBSECTOR_FLAG`` bit is set, clear it and use as
    a subsector index; otherwise use as another node index.

    Bounding boxes are ``(top, bottom, left, right)`` in DOOM's
    coordinate system (top > bottom in world coordinates).
    """

    px: int
    py: int
    dx: int
    dy: int
    front_bbox: Tuple[int, int, int, int]
    back_bbox: Tuple[int, int, int, int]
    front_child: int
    back_child: int


@dataclass
class MapData:
    """All geometry + BSP data parsed from a single map (e.g. E1M1)."""

    name: str
    vertices: List[Vertex]
    linedefs: List[Linedef]
    sidedefs: List[Sidedef]
    sectors: List[Sector]
    segs: List[Seg]
    subsectors: List[Subsector]
    nodes: List[BspNode]


# ---------------------------------------------------------------------------
# WAD reader
# ---------------------------------------------------------------------------


class WADReader:
    """Read lumps, textures, and map data from a DOOM WAD file."""

    def __init__(self, path: str):
        with open(path, "rb") as f:
            self._data = f.read()

        # Header
        self.wad_id = self._data[:4]
        numlumps = struct.unpack_from("<I", self._data, 4)[0]
        dir_offset = struct.unpack_from("<I", self._data, 8)[0]

        # Lump directory — keep ordered list (needed for map markers
        # whose following lumps have duplicate names across maps) and
        # first-occurrence dict (used by texture/palette lookup).
        self._lumps: Dict[str, tuple] = {}
        self._lump_order: List[Tuple[str, int, int]] = []
        for i in range(numlumps):
            base = dir_offset + i * 16
            offset = struct.unpack_from("<I", self._data, base)[0]
            size = struct.unpack_from("<I", self._data, base + 4)[0]
            name = self._data[base + 8 : base + 16].rstrip(b"\x00").decode("ascii")
            self._lump_order.append((name, offset, size))
            if name not in self._lumps:
                self._lumps[name] = (offset, size)

        # Palette (first of 14 palettes in PLAYPAL)
        pal_off, _ = self._lumps["PLAYPAL"]
        self.palette = np.frombuffer(
            self._data[pal_off : pal_off + 768],
            dtype=np.uint8,
        ).reshape(256, 3)

        # Patch names
        pn_off, _ = self._lumps["PNAMES"]
        self._num_pnames = struct.unpack_from("<I", self._data, pn_off)[0]
        self._pnames = []
        for i in range(self._num_pnames):
            base = pn_off + 4 + i * 8
            self._pnames.append(
                self._data[base : base + 8].rstrip(b"\x00").decode("ascii")
            )

        # Texture definitions (TEXTURE1)
        self._tex_defs = self._parse_texture_lump("TEXTURE1")
        if "TEXTURE2" in self._lumps:
            self._tex_defs.update(self._parse_texture_lump("TEXTURE2"))

    def _parse_texture_lump(self, lump_name: str) -> dict:
        off, _ = self._lumps[lump_name]
        num = struct.unpack_from("<I", self._data, off)[0]
        offsets = [
            struct.unpack_from("<I", self._data, off + 4 + i * 4)[0] for i in range(num)
        ]
        defs = {}
        for tex_off in offsets:
            base = off + tex_off
            name = self._data[base : base + 8].rstrip(b"\x00").decode("ascii")
            width = struct.unpack_from("<H", self._data, base + 12)[0]
            height = struct.unpack_from("<H", self._data, base + 14)[0]
            np_count = struct.unpack_from("<H", self._data, base + 20)[0]
            patches = []
            for j in range(np_count):
                pb = base + 22 + j * 10
                ox = struct.unpack_from("<h", self._data, pb)[0]
                oy = struct.unpack_from("<h", self._data, pb + 2)[0]
                pidx = struct.unpack_from("<H", self._data, pb + 4)[0]
                patches.append((ox, oy, self._pnames[pidx]))
            defs[name] = (width, height, patches)
        return defs

    def _read_patch(self, name: str) -> Optional[np.ndarray]:
        """Read a patch lump into a (width, height, 3) array.

        Transparent pixels are set to -1.
        """
        if name not in self._lumps:
            return None
        off, _ = self._lumps[name]
        width = struct.unpack_from("<H", self._data, off)[0]
        height = struct.unpack_from("<H", self._data, off + 2)[0]
        col_offsets = [
            struct.unpack_from("<I", self._data, off + 8 + c * 4)[0]
            for c in range(width)
        ]
        pixels = np.full((width, height, 3), -1, dtype=np.int16)
        for col in range(width):
            pos = off + col_offsets[col]
            while True:
                row_start = self._data[pos]
                pos += 1
                if row_start == 0xFF:
                    break
                count = self._data[pos]
                pos += 2  # count + padding
                for j in range(count):
                    row = row_start + j
                    if row < height:
                        pixels[col, row] = self.palette[self._data[pos]]
                    pos += 1
                pos += 1  # padding
        return pixels

    def get_texture(self, name: str) -> Optional[np.ndarray]:
        """Compose a texture by name.

        Returns a ``(width, height, 3)`` float64 array with values in
        [0, 1], or ``None`` if the texture is not found.
        """
        if name not in self._tex_defs:
            return None
        width, height, patches = self._tex_defs[name]
        canvas = np.zeros((width, height, 3), dtype=np.float64)
        for ox, oy, patch_name in patches:
            patch = self._read_patch(patch_name)
            if patch is None:
                continue
            pw, ph = patch.shape[0], patch.shape[1]
            for col in range(pw):
                for row in range(ph):
                    if patch[col, row, 0] >= 0:
                        dx, dy = ox + col, oy + row
                        if 0 <= dx < width and 0 <= dy < height:
                            canvas[dx, dy] = patch[col, row] / 255.0
        return canvas

    def get_textures(self, names: List[str]) -> Dict[str, np.ndarray]:
        """Load multiple textures by name."""
        result = {}
        for name in names:
            tex = self.get_texture(name)
            if tex is not None:
                result[name] = tex
        return result

    def list_textures(self) -> List[str]:
        """Return all available texture names."""
        return sorted(self._tex_defs.keys())

    # -----------------------------------------------------------------
    # Map geometry + BSP loading
    # -----------------------------------------------------------------

    def _find_map_lumps(self, map_name: str) -> Dict[str, Tuple[int, int]]:
        """Locate the geometry lumps that belong to a map marker.

        A map marker is a zero-size lump with the map's name (e.g.
        ``E1M1``).  The lumps that follow — until the next marker or
        a non-map-lump name is encountered — are the map's data.

        Returns a dict mapping lump name (e.g. ``"VERTEXES"``) to
        ``(offset, size)``.  Raises ``KeyError`` if the map marker
        is not found.
        """
        found_marker = False
        result: Dict[str, Tuple[int, int]] = {}
        for name, off, size in self._lump_order:
            if not found_marker:
                if name == map_name and size == 0:
                    found_marker = True
                continue
            if name in _MAP_LUMP_NAMES:
                # Only take the first occurrence of each lump type.
                if name not in result:
                    result[name] = (off, size)
            else:
                # Hit a non-map lump — end of this map's data.
                break
        if not found_marker:
            raise KeyError(f"Map marker {map_name!r} not found in WAD")
        return result

    def _parse_vertexes(self, off: int, size: int) -> List[Vertex]:
        """Parse a VERTEXES lump — 4 bytes per vertex (int16 x, int16 y)."""
        buf = self._data[off : off + size]
        return [Vertex(x, y) for x, y in struct.iter_unpack("<hh", buf)]

    def _parse_linedefs(self, off: int, size: int) -> List[Linedef]:
        """Parse a LINEDEFS lump — 14 bytes per linedef.

        Format: ``<HHHHHHH`` — v1, v2, flags, special, tag, front_side, back_side.
        Sidedef value ``0xFFFF`` (65535) is converted to ``-1`` for "no sidedef".
        """
        buf = self._data[off : off + size]
        result = []
        for v1, v2, flags, special, tag, fs, bs in struct.iter_unpack("<HHHHHHH", buf):
            front = -1 if fs == 0xFFFF else fs
            back = -1 if bs == 0xFFFF else bs
            result.append(
                Linedef(
                    v1=v1,
                    v2=v2,
                    flags=flags,
                    special=special,
                    tag=tag,
                    front_sidedef=front,
                    back_sidedef=back,
                )
            )
        return result

    @staticmethod
    def _decode_name(raw: bytes) -> str:
        """Strip NUL padding and decode an 8-byte lump name."""
        return raw.rstrip(b"\x00").decode("ascii", errors="replace")

    def _parse_sidedefs(self, off: int, size: int) -> List[Sidedef]:
        """Parse a SIDEDEFS lump — 30 bytes per sidedef.

        Format: ``<hh8s8s8sH`` — x_off, y_off, upper, lower, middle, sector.
        Texture names are 8-byte ASCII, NUL-padded.
        """
        buf = self._data[off : off + size]
        result = []
        for xo, yo, u, lo, mi, sec in struct.iter_unpack("<hh8s8s8sH", buf):
            result.append(
                Sidedef(
                    x_offset=xo,
                    y_offset=yo,
                    upper=self._decode_name(u),
                    lower=self._decode_name(lo),
                    middle=self._decode_name(mi),
                    sector=sec,
                )
            )
        return result

    def _parse_sectors(self, off: int, size: int) -> List[Sector]:
        """Parse a SECTORS lump — 26 bytes per sector.

        Format: ``<hh8s8shhh`` — floor_h, ceil_h, floor_tex, ceil_tex,
        light, special, tag.
        """
        buf = self._data[off : off + size]
        result = []
        for fh, ch, ft, ct, light, special, tag in struct.iter_unpack(
            "<hh8s8shhh", buf
        ):
            result.append(
                Sector(
                    floor_h=fh,
                    ceiling_h=ch,
                    floor_tex=self._decode_name(ft),
                    ceiling_tex=self._decode_name(ct),
                    light=light,
                    special=special,
                    tag=tag,
                )
            )
        return result

    def _parse_segs(self, off: int, size: int) -> List[Seg]:
        """Parse a SEGS lump — 12 bytes per seg.

        Format: ``<HHhHhh`` — v1, v2, angle, linedef, side, offset.
        """
        buf = self._data[off : off + size]
        result = []
        for v1, v2, angle, ld, side, offset in struct.iter_unpack("<HHhHhh", buf):
            result.append(
                Seg(
                    v1=v1,
                    v2=v2,
                    angle=angle,
                    linedef=ld,
                    side=side,
                    offset=offset,
                )
            )
        return result

    def _parse_subsectors(self, off: int, size: int) -> List[Subsector]:
        """Parse an SSECTORS lump — 4 bytes per subsector.

        Format: ``<HH`` — seg_count, first_seg.
        """
        buf = self._data[off : off + size]
        return [
            Subsector(seg_count=count, first_seg=first)
            for count, first in struct.iter_unpack("<HH", buf)
        ]

    def _parse_nodes(self, off: int, size: int) -> List[BspNode]:
        """Parse a NODES lump — 28 bytes per node.

        Format: ``<hhhh`` (splitting line: px, py, dx, dy),
        followed by 8 × int16 (two bboxes: top, bottom, left, right each),
        followed by ``<HH`` (front_child, back_child).  Total 28 bytes.
        """
        buf = self._data[off : off + size]
        result = []
        # 14 int16 fields + no extra padding (14*2 = 28 bytes).
        for fields in struct.iter_unpack("<hhhh" "hhhh" "hhhh" "HH", buf):
            (
                px,
                py,
                dx,
                dy,
                f_top,
                f_bot,
                f_left,
                f_right,
                b_top,
                b_bot,
                b_left,
                b_right,
                front_child,
                back_child,
            ) = fields
            result.append(
                BspNode(
                    px=px,
                    py=py,
                    dx=dx,
                    dy=dy,
                    front_bbox=(f_top, f_bot, f_left, f_right),
                    back_bbox=(b_top, b_bot, b_left, b_right),
                    front_child=front_child,
                    back_child=back_child,
                )
            )
        return result

    def get_map(self, map_name: str) -> MapData:
        """Parse all seven geometry lumps for a map.

        Returns a :class:`MapData` containing vertices, linedefs,
        sidedefs, sectors, segs, subsectors, and BSP nodes.

        Raises ``KeyError`` if the map marker is not found, or if
        any required lump is missing.
        """
        lumps = self._find_map_lumps(map_name)
        required = (
            "VERTEXES",
            "LINEDEFS",
            "SIDEDEFS",
            "SECTORS",
            "SEGS",
            "SSECTORS",
            "NODES",
        )
        missing = [n for n in required if n not in lumps]
        if missing:
            raise KeyError(f"Map {map_name!r} missing required lumps: {missing}")
        return MapData(
            name=map_name,
            vertices=self._parse_vertexes(*lumps["VERTEXES"]),
            linedefs=self._parse_linedefs(*lumps["LINEDEFS"]),
            sidedefs=self._parse_sidedefs(*lumps["SIDEDEFS"]),
            sectors=self._parse_sectors(*lumps["SECTORS"]),
            segs=self._parse_segs(*lumps["SEGS"]),
            subsectors=self._parse_subsectors(*lumps["SSECTORS"]),
            nodes=self._parse_nodes(*lumps["NODES"]),
        )

    def get_map_segments(
        self,
        map_name: str,
        tex_size: int = 8,
    ) -> Tuple[List["Segment"], List[np.ndarray], Dict[str, int]]:
        """Load renderable segments and their texture atlas from a map.

        Uses the map's SEGS (BSP-split wall fragments) — these are what
        DOOM actually renders.  Each seg becomes one
        :class:`~torchwright.reference_renderer.types.Segment`.  Texture
        names from each seg's sidedef are collected, loaded from the
        WAD, and downscaled to ``tex_size`` x ``tex_size``.

        Segments referencing invalid sidedefs are silently skipped.
        Texture names not found in the WAD are also dropped (segments
        get ``texture_id = -1``).

        Coordinates are returned in DOOM's native integer scale (int16,
        roughly ±4000 across a full map).  The reference renderer and
        compiled transformer each consume these directly; callers that
        want smaller numeric ranges should rescale themselves at the
        point of use.

        Returns:
            ``(segments, textures, name_to_id)``:

            - ``segments``: list of ``Segment`` with native DOOM coords.
            - ``textures``: list of ``(tex_size, tex_size, 3)`` float
              arrays, indexed by the texture_id assigned in
              ``name_to_id``.
            - ``name_to_id``: dict mapping wall-texture name to its
              index in ``textures``.
        """
        # Local import to avoid pulling reference_renderer at module
        # load time (and to mirror how scenes.py imports it).
        from torchwright.reference_renderer.textures import downscale_texture
        from torchwright.reference_renderer.types import Segment

        md = self.get_map(map_name)
        segments, name_to_id = seg_list_to_segments(md)

        # Load textures in name_to_id order (a dict preserves insertion
        # order in Python 3.7+).  Drop any not found in the WAD.
        textures: List[np.ndarray] = []
        valid_name_to_id: Dict[str, int] = {}
        dropped_names: set = set()
        for name in name_to_id:
            tex = self.get_texture(name)
            if tex is None:
                dropped_names.add(name)
                continue
            valid_name_to_id[name] = len(textures)
            textures.append(downscale_texture(tex, tex_size, tex_size))

        # Remap segments: any texture_id pointing at a dropped name
        # becomes -1; otherwise rewrite to the compacted index.
        if dropped_names:
            # Build old_id -> new_id mapping
            old_to_new: Dict[int, int] = {}
            for old_name, old_id in name_to_id.items():
                if old_name in valid_name_to_id:
                    old_to_new[old_id] = valid_name_to_id[old_name]
            remapped: List[Segment] = []
            for seg in segments:
                if seg.texture_id < 0:
                    remapped.append(seg)
                elif seg.texture_id in old_to_new:
                    remapped.append(
                        Segment(
                            ax=seg.ax,
                            ay=seg.ay,
                            bx=seg.bx,
                            by=seg.by,
                            color=seg.color,
                            texture_id=old_to_new[seg.texture_id],
                        )
                    )
                else:
                    remapped.append(
                        Segment(
                            ax=seg.ax,
                            ay=seg.ay,
                            bx=seg.bx,
                            by=seg.by,
                            color=seg.color,
                            texture_id=-1,
                        )
                    )
            segments = remapped

        return segments, textures, valid_name_to_id


# ---------------------------------------------------------------------------
# Standalone helpers (factored out for unit testing)
# ---------------------------------------------------------------------------


def sector_color(sector_index: int) -> Tuple[float, float, float]:
    """Deterministic per-sector RGB color via HSV hash.

    Used as a fallback color when no texture is assigned and to
    visually distinguish sectors in untextured debug renders.
    """
    hue = (sector_index * 0.137) % 1.0
    r, g, b = colorsys.hsv_to_rgb(hue, 0.6, 0.8)
    return (float(r), float(g), float(b))


def _assign_tex_id(
    name: str,
    name_to_id: Dict[str, int],
) -> int:
    """Return the texture id for *name*, assigning a new one if unseen.

    ``"-"`` is DOOM's sentinel for "no texture" and returns ``-1``.
    Mutates ``name_to_id`` to record new assignments.
    """
    if name == "-" or not name:
        return -1
    if name in name_to_id:
        return name_to_id[name]
    new_id = len(name_to_id)
    name_to_id[name] = new_id
    return new_id


def _pick_seg_texture(sidedef: Sidedef) -> str:
    """Pick the drawable texture name for a seg's sidedef.

    Prefers the middle texture (solid walls); falls back to lower,
    then upper.  Returns ``"-"`` if none are set.
    """
    if sidedef.middle != "-":
        return sidedef.middle
    if sidedef.lower != "-":
        return sidedef.lower
    if sidedef.upper != "-":
        return sidedef.upper
    return "-"


def seg_list_to_segments(
    md: MapData,
) -> Tuple[List["Segment"], Dict[str, int]]:
    """Convert a map's SEGS to a list of renderable ``Segment`` objects.

    Each seg becomes one Segment with:

    - Coordinates looked up from ``md.vertices`` (DOOM-native int16,
      cast to float)
    - Texture id assigned on first sight of the texture name
    - Color derived from the seg's sector index (HSV hash)

    Segs with invalid vertex/linedef/sidedef refs are silently skipped.
    Returns ``(segments, name_to_id)`` — caller loads textures using
    the name→id map.
    """
    from torchwright.reference_renderer.types import Segment

    segments: List[Segment] = []
    name_to_id: Dict[str, int] = {}
    nv = len(md.vertices)
    nl = len(md.linedefs)
    ns = len(md.sidedefs)

    for seg in md.segs:
        if seg.v1 >= nv or seg.v2 >= nv or seg.v1 < 0 or seg.v2 < 0:
            continue
        if seg.linedef < 0 or seg.linedef >= nl:
            continue
        ld = md.linedefs[seg.linedef]
        sd_idx = ld.front_sidedef if seg.side == 0 else ld.back_sidedef
        if sd_idx < 0 or sd_idx >= ns:
            continue
        sd = md.sidedefs[sd_idx]
        tex_name = _pick_seg_texture(sd)
        tex_id = _assign_tex_id(tex_name, name_to_id)
        color = sector_color(sd.sector)
        v1 = md.vertices[seg.v1]
        v2 = md.vertices[seg.v2]
        segments.append(
            Segment(
                ax=float(v1.x),
                ay=float(v1.y),
                bx=float(v2.x),
                by=float(v2.y),
                color=color,
                texture_id=tex_id,
            )
        )
    return segments, name_to_id
