"""Extract textures and other data from DOOM WAD files.

The WAD (Where's All the Data?) format stores DOOM's game assets:
textures, maps, sounds, etc.  This module handles texture extraction
— compositing multi-patch textures from the WAD's column-based
picture format using the PLAYPAL palette.

Textures are returned as ``(width, height, 3)`` float64 numpy arrays
in column-major layout (matching the renderer's convention).
"""

import struct
from typing import Dict, List, Optional

import numpy as np


class WADReader:
    """Read lumps and textures from a DOOM WAD file."""

    def __init__(self, path: str):
        with open(path, "rb") as f:
            self._data = f.read()

        # Header
        self.wad_id = self._data[:4]
        numlumps = struct.unpack_from("<I", self._data, 4)[0]
        dir_offset = struct.unpack_from("<I", self._data, 8)[0]

        # Lump directory
        self._lumps: Dict[str, tuple] = {}
        for i in range(numlumps):
            base = dir_offset + i * 16
            offset = struct.unpack_from("<I", self._data, base)[0]
            size = struct.unpack_from("<I", self._data, base + 4)[0]
            name = self._data[base + 8 : base + 16].rstrip(b"\x00").decode("ascii")
            self._lumps[name] = (offset, size)

        # Palette (first of 14 palettes in PLAYPAL)
        pal_off, _ = self._lumps["PLAYPAL"]
        self.palette = np.frombuffer(
            self._data[pal_off : pal_off + 768], dtype=np.uint8,
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
            struct.unpack_from("<I", self._data, off + 4 + i * 4)[0]
            for i in range(num)
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
