"""Sandbox fixtures — JSON-serialized `MapSubset`s.

Fixtures are produced by `scripts/build_sandbox_fixtures.py` (which
reads from torchwright source-of-truth scenes) and committed as JSON
under this directory. The sandbox runtime never imports torchwright;
phases load fixtures via `load_fixture(name)`.
"""

from __future__ import annotations

from pathlib import Path

from ..types import MapSubset


_FIXTURE_DIR = Path(__file__).resolve().parent


def load_fixture(name: str) -> MapSubset:
    """Load a fixture by name. Returns the deserialized `MapSubset`.

    Looks for `<name>.json` in the fixtures directory. Raises
    `FileNotFoundError` listing the available fixtures if no match.
    """
    path = _FIXTURE_DIR / f"{name}.json"
    if not path.exists():
        available = sorted(p.stem for p in _FIXTURE_DIR.glob("*.json"))
        raise FileNotFoundError(
            f"No fixture named {name!r}. Available: {available}"
        )
    return MapSubset.model_validate_json(path.read_text())


def available_fixtures() -> list[str]:
    """Return the names of every fixture present on disk, sorted."""
    return sorted(p.stem for p in _FIXTURE_DIR.glob("*.json"))
