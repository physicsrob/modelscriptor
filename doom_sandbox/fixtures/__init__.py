"""Sandbox fixtures — JSON-serialized `MapSubset`s.

Fixtures are produced by `scripts/build_sandbox_fixtures.py` (which
reads from torchwright source-of-truth scenes) and committed as JSON
under this directory. The sandbox runtime never imports torchwright;
phases load fixtures via `load_fixture(name)`.

Each fixture may carry a `test_poses` list — known-safe player states
for sandbox tests, picked to land clearly off every BSP plane so PWL
approximation noise can't flip a `side_P` bit. Phases should prefer
`scene.test_poses[0]` over hand-rolled coordinates; if they roll their
own, `assert_pose_clear_of_planes` raises loudly when a pose is too
close to any plane.
"""

from __future__ import annotations

from pathlib import Path

from ..types import GameState, MapSubset


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


def assert_pose_clear_of_planes(
    scene: MapSubset, state: GameState, atol: float = 0.1
) -> None:
    """Assert the player position is at distance ≥ `atol` from every BSP plane.

    A pose lying near a plane (`|nx·x + ny·y + d| < atol`) is fragile:
    PWL approximation noise on the dot product can flip the
    corresponding `side_P` bit, breaking the rank computation's exact
    integer-match contract. This helper raises `AssertionError` naming
    the offending plane(s) so the test fails on the pose, not on a
    downstream rank mismatch.

    Use it on any pose you didn't pick from `scene.test_poses` — the
    fixture builder guarantees the committed test poses are well
    clear, but a hand-rolled pose has no such guarantee.

    The default `atol=0.1` leaves comfortable margin above the
    deadband of `compare_const` (≈ 0.001 · input-range span) and
    above FloatSlot quantization on the published x/y. Loosen it if
    your design needs tighter geometry; tighten it if you've squeezed
    the dot-product input range very small.
    """
    violations: list[str] = []
    for i, node in enumerate(scene.bsp_nodes):
        signed = node.nx * state.x + node.ny * state.y + node.d
        if abs(signed) < atol:
            violations.append(
                f"plane {i} (nx={node.nx}, ny={node.ny}, d={node.d}): "
                f"signed distance {signed:.6f} < atol {atol}"
            )
    if violations:
        raise AssertionError(
            f"player pose ({state.x}, {state.y}) is too close to "
            f"{len(violations)} BSP plane(s):\n  "
            + "\n  ".join(violations)
        )
