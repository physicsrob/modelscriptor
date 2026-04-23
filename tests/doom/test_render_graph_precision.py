"""Per-node precision regression tests for the compiled DOOM renderer.

These tests walk every node in the compiled graph and assert each stays
within declared numerical envelopes. They encode what Phase E taught us
about precision discipline:

1. **fp64 CPU replay** — the compiler wires weights correctly
   (construction correctness, independent of fp32 accumulation drift).
2. **Gate-M audit** — no approximate-cond_gate site has an amplifier
   constant large enough to turn compare-noise into render-breaking
   error (this is the direct structural invariant Phase E violated).
3. **Per-op-class absolute bounds** — clean ops (pre-amplifier) stay
   at compare-noise floor; tail ops (approximate-gate outputs) stay
   within their M-amplified noise budget.

Tests are split into two tiers:

- **Ratchet** tests pass today and gate future regressions.
- **Stretch** tests are ``xfail(strict=True)`` with reasons that name
  the specific debt blocking them. When someone tightens enough to
  make a stretch test pass, the ``strict=True`` flips it to XPASS and
  forces the threshold to be updated — the test file becomes the
  punch-list for remaining precision work.

See ``docs/postmortems/phase_e_xfail.md`` for the investigation that
motivated this file.
"""

from __future__ import annotations

import copy
import math
from typing import Dict, List, Tuple

import pytest
import torch

from torchwright.compiler.export import compile_headless
from torchwright.compiler.utils import get_ancestor_nodes
from torchwright.debug.probe import probe_compiled
from torchwright.doom.compile import _build_inputs, _stack_inputs
from torchwright.doom.embedding import vocab_id
from torchwright.doom.game_graph import (
    TEX_E8_OFFSET,
    build_game_graph,
)
from torchwright.doom.map_subset import build_scene_subset
from torchwright.graph import Concatenate, Linear
from torchwright.graph.attn import Attn
from torchwright.graph.optimize import fuse_consecutive_linears
from torchwright.graph.relu import ReLU
from torchwright.graph.spherical_codes import index_to_vector
from torchwright.reference_renderer.textures import default_texture_atlas
from torchwright.reference_renderer.trig import generate_trig_table
from torchwright.reference_renderer.types import RenderConfig, Segment

_SCENES = [
    (0.0, 0.0, 0.0),  # axis-aligned, centered
    (0.0, 0.0, 45.0),  # oblique, centered
    (3.0, 2.0, 20.0),  # the Phase E scene
    (-2.0, 3.0, 240.0),  # off-center + oblique
]
_FP64_SCENE = (3.0, 2.0, 20.0)


# ── Graph construction helpers (shared with dump_phase_e_allocator) ──


def _config() -> RenderConfig:
    return RenderConfig(
        screen_width=16,
        screen_height=20,
        fov_columns=16,
        trig_table=generate_trig_table(),
        ceiling_color=(0.2, 0.2, 0.2),
        floor_color=(0.4, 0.4, 0.4),
    )


def _segments(half: float = 5.0) -> List[Segment]:
    return [
        Segment(
            ax=half, ay=-half, bx=half, by=half, color=(0.8, 0.2, 0.1), texture_id=0
        ),
        Segment(
            ax=-half, ay=-half, bx=-half, by=half, color=(0.8, 0.2, 0.1), texture_id=1
        ),
        Segment(
            ax=-half, ay=half, bx=half, by=half, color=(0.8, 0.2, 0.1), texture_id=2
        ),
        Segment(
            ax=-half, ay=-half, bx=half, by=-half, color=(0.8, 0.2, 0.1), texture_id=3
        ),
    ]


def _build_prefill(module, subset, *, px, py, angle):
    max_bsp_nodes = int(module.metadata["max_bsp_nodes"])
    common = dict(
        player_x=torch.tensor([px]),
        player_y=torch.tensor([py]),
        player_angle=torch.tensor([float(angle)]),
    )
    tex_w = subset.textures[0].shape[0]
    rows = []
    for tex_idx in range(len(subset.textures)):
        tex_e8 = index_to_vector(tex_idx + TEX_E8_OFFSET)
        for col in range(tex_w):
            pixel_data = subset.textures[tex_idx][col].flatten()
            rows.append(
                _build_inputs(
                    module,
                    token_id=vocab_id("TEX_COL"),
                    texture_id_e8=tex_e8,
                    tex_col_input=torch.tensor([float(col)]),
                    tex_pixels=torch.tensor(pixel_data, dtype=torch.float32),
                    **common,
                )
            )
    rows.append(_build_inputs(module, token_id=vocab_id("INPUT"), **common))
    for i in range(max_bsp_nodes):
        onehot = torch.zeros(max_bsp_nodes)
        onehot[i] = 1.0
        if i < len(subset.bsp_nodes):
            plane = subset.bsp_nodes[i]
            nx, ny, d = plane.nx, plane.ny, plane.d
        else:
            nx, ny, d = 0.0, 0.0, 0.0
        rows.append(
            _build_inputs(
                module,
                token_id=vocab_id("BSP_NODE"),
                bsp_plane_nx=torch.tensor([nx], dtype=torch.float32),
                bsp_plane_ny=torch.tensor([ny], dtype=torch.float32),
                bsp_plane_d=torch.tensor([d], dtype=torch.float32),
                bsp_node_id_onehot=onehot,
                **common,
            )
        )
    for i, seg in enumerate(subset.segments):
        coeffs = torch.tensor(
            subset.seg_bsp_coeffs[i, :max_bsp_nodes],
            dtype=torch.float32,
        )
        const = torch.tensor([float(subset.seg_bsp_consts[i])], dtype=torch.float32)
        rows.append(
            _build_inputs(
                module,
                token_id=vocab_id("WALL"),
                wall_ax=torch.tensor([float(seg.ax)]),
                wall_ay=torch.tensor([float(seg.ay)]),
                wall_bx=torch.tensor([float(seg.bx)]),
                wall_by=torch.tensor([float(seg.by)]),
                wall_tex_id=torch.tensor([float(seg.texture_id)]),
                wall_index=torch.tensor([float(i)]),
                wall_bsp_coeffs=coeffs,
                wall_bsp_const=const,
                **common,
            )
        )
    rows.append(_build_inputs(module, token_id=vocab_id("EOS"), **common))
    return _stack_inputs(rows)


def _classify(node) -> str:
    """Tag a node into one of {clean, tail, other}.

    - ``clean``: pre-amplifier ops whose error floor is compare-noise
      (~0.005 abs). If these grow, the compiler or an op regressed.
    - ``tail``: approximate-cond_gate output chain (``select_linear2``,
      ``cond_gate_linear2``, the ``negate`` Linear that immediately
      follows, and ``Add`` nodes receiving the cancellation output).
      Error floor is M × ε_cond.
    - ``other``: Linears that don't fit either bucket (e.g. utility
      linears like ``multiply_const``, ``t_star_pos_*``). These are
      checked only against the global rel/decl bound, not per-class.
    """
    if isinstance(node, (ReLU, Attn)):
        return "clean"
    if isinstance(node, Linear):
        name = node.name or ""
        if (
            "select_linear2" in name
            or "cond_gate_linear2" in name
            or name == "negate"
            or name == "_linear2"
        ):
            return "tail"
        if "linear1" in name or name.startswith("bsp_"):
            return "clean"
        return "other"
    # Add nodes inside the approximate-gate cancellation chain read the
    # tail-noisy gate output and carry its noise forward. Rather than
    # classify by topology, we treat Add as tail by default — the only
    # way an Add accumulates >1 abs error today is from such a chain.
    if type(node).__name__ == "Add":
        return "tail"
    return "other"


# ── Approximate-gate M audit ─────────────────────────────────────────


def _gate_M(node: Linear) -> float:
    """Recover the M baked into a ``_linear2`` gate-output node.

    cond_gate's output range is ``[min(0, inp.lo), max(0, inp.hi)]``, so
    ``max|output_range|`` recovers ``max|inp|``. ``M = SAFETY × max|inp|``.
    For ``select`` the output range spans both branches; same proxy.
    """
    from torchwright.ops.logic_ops import _GATE_OFFSET_SAFETY_FACTOR as SAFETY

    r = node.value_type.value_range
    if not (math.isfinite(r.lo) and math.isfinite(r.hi)):
        return float("inf")
    return SAFETY * max(abs(r.lo), abs(r.hi))


def _approx_gate_sites(graph_root) -> List[Dict]:
    """Enumerate all approximate-gate output Linear nodes and their M."""
    sites: List[Dict] = []
    for n in get_ancestor_nodes({graph_root}):
        if not isinstance(n, Linear):
            continue
        name = n.name or ""
        if "select_linear2" not in name and "cond_gate_linear2" not in name:
            continue
        sites.append({"node": n, "M": _gate_M(n), "name": name})
    return sites


# ── Shared fixture: compile once per class ───────────────────────────


class TestRenderGraphPrecision:
    """Per-node precision regression tests for the compiled DOOM renderer.

    All tests in this class share a single compiled module via the
    ``compiled`` class-scoped fixture. Compilation takes ~17s on A100;
    each probe run adds ~2s.
    """

    @pytest.fixture(scope="class")
    def compiled(self):
        """Compile the DOOM graph once; share the module across tests."""
        config = _config()
        textures = default_texture_atlas()
        subset = build_scene_subset(_segments(), textures)

        graph_io, pos_encoding = build_game_graph(
            config,
            textures,
            max_walls=8,
            max_coord=20.0,
            move_speed=0.3,
            turn_speed=4,
            chunk_size=20,
            max_bsp_nodes=48,
        )
        output_nodes = set(graph_io.overlaid_outputs.values())
        output_nodes.update(graph_io.overflow_outputs.values())
        output_nodes.add(pos_encoding)
        while fuse_consecutive_linears(output_nodes, verbose=False) > 0:
            pass

        io = {}
        for name, node in graph_io.inputs.items():
            io[name] = (node, graph_io.overlaid_outputs.get(name))
        for name, node in graph_io.overflow_outputs.items():
            io[name] = (None, node)

        module = compile_headless(
            pos_encoding,
            io=io,
            d=2048,
            d_head=32,
            max_layers=400,
            verbose=False,
            extra_metadata={
                "chunk_size": 20,
                "max_walls": 8,
                "max_bsp_nodes": 48,
                "tex_h": textures[0].shape[1],
            },
        )
        module.eval()

        probe_root = Concatenate(
            list(graph_io.overlaid_outputs.values())
            + list(graph_io.overflow_outputs.values())
        )
        return module, probe_root, subset

    @pytest.fixture(scope="class")
    def fp32_reports(self, compiled):
        """Dict[scene_tuple, ProbeReport] — full per-node walk per scene."""
        module, probe_root, subset = compiled
        reports = {}
        for px, py, angle in _SCENES:
            prefill = _build_prefill(module, subset, px=px, py=py, angle=angle)
            n_pos = prefill["token_ids"].shape[0]
            rep = probe_compiled(
                module,
                probe_root,
                prefill,
                n_pos=n_pos,
                atol=1e9,
            )
            reports[(px, py, angle)] = rep
        return reports

    @pytest.fixture(scope="class")
    def fp64_report(self, compiled):
        """Compiled module converted to fp64 on CPU; probe once at the
        Phase E scene. Catches compiler-construction bugs separately
        from fp32 accumulation."""
        module, probe_root, subset = compiled
        px, py, angle = _FP64_SCENE
        prefill = _build_prefill(module, subset, px=px, py=py, angle=angle)

        # The compiled module is deep-copied and its deepcopy is converted
        # to fp64. But the *graph* (shared via probe_root) is not copied —
        # reference_eval walks the same Linear/Attn nodes the fp32 tests
        # also walk. We must convert the graph weights to fp64 for the
        # oracle, then restore them to fp32 before returning, so
        # fp32_reports (which may run after this fixture) sees fp32
        # weights.
        all_graph_nodes = get_ancestor_nodes({probe_root})
        _prev = torch.get_default_dtype()
        torch.set_default_dtype(torch.float64)
        try:
            module_cpu = copy.deepcopy(module)
            module_cpu._net.to("cpu")
            for _layer in module_cpu._net.layers:
                for _attr in (
                    "query_matrix",
                    "key_matrix",
                    "value_matrix",
                    "output_matrix",
                ):
                    setattr(
                        _layer.attn.attn,
                        _attr,
                        getattr(_layer.attn.attn, _attr).to(torch.float64),
                    )
                for _comp in (_layer.mlp.linear1, _layer.mlp.linear2):
                    _comp.output_matrix = _comp.output_matrix.to(torch.float64)
                    _comp.output_bias = _comp.output_bias.to(torch.float64)

            for _n in all_graph_nodes:
                if isinstance(_n, Linear):
                    _n.output_matrix = _n.output_matrix.to(torch.float64)
                    _n.output_bias = _n.output_bias.to(torch.float64)
                if isinstance(_n, Attn):
                    _n.query_matrix = _n.query_matrix.to(torch.float64)
                    _n.key_matrix = _n.key_matrix.to(torch.float64)
                    _n.value_matrix = _n.value_matrix.to(torch.float64)
                    _n.output_matrix = _n.output_matrix.to(torch.float64)
                if hasattr(_n, "value") and isinstance(_n.value, torch.Tensor):
                    _n.value = _n.value.to(torch.float64)

            inputs_f64 = {
                name: v.to(device="cpu", dtype=torch.float64)
                for name, v in prefill.items()
            }
            n_pos = prefill["token_ids"].shape[0]
            rep = probe_compiled(
                module_cpu,
                probe_root,
                inputs_f64,
                n_pos=n_pos,
                atol=1e9,
            )
        finally:
            # Restore graph-node weights to fp32 so fp32 tests using the
            # same graph see the original dtype.
            for _n in all_graph_nodes:
                if isinstance(_n, Linear):
                    _n.output_matrix = _n.output_matrix.to(torch.float32)
                    _n.output_bias = _n.output_bias.to(torch.float32)
                if isinstance(_n, Attn):
                    _n.query_matrix = _n.query_matrix.to(torch.float32)
                    _n.key_matrix = _n.key_matrix.to(torch.float32)
                    _n.value_matrix = _n.value_matrix.to(torch.float32)
                    _n.output_matrix = _n.output_matrix.to(torch.float32)
                if hasattr(_n, "value") and isinstance(_n.value, torch.Tensor):
                    _n.value = _n.value.to(torch.float32)
            torch.set_default_dtype(_prev)
        return rep

    # ── Helpers accessing fixture data ──────────────────────────────

    @staticmethod
    def _combined_worst(reports):
        """Per node_id, the worst NodeDivergence record across scenes."""
        combined: Dict[int, object] = {}
        for rep in reports.values():
            for rec in rep.per_node.values():
                nid = rec.node.node_id
                prev = combined.get(nid)
                if prev is None or rec.max_abs_error > prev.max_abs_error:  # type: ignore[union-attr]
                    combined[nid] = rec
        return combined

    @staticmethod
    def _format_offenders(offenders, limit=5, extra_fmt=None) -> str:
        """Format up to `limit` worst offenders for an assertion message."""
        lines = []
        for item in offenders[:limit]:
            if extra_fmt:
                lines.append("  " + extra_fmt(item))
            else:
                node, err = item[0], item[1]
                lines.append(
                    f"  id={node.node_id} {type(node).__name__}:{node.name!r} "
                    f"|Δ|={err:.4g}"
                )
        return "\n".join(lines)

    # ── Ratchet tests (must pass today) ─────────────────────────────

    def test_fp64_construction_correct(self, fp64_report):
        """fp64 CPU replay matches oracle to ~fp64 precision everywhere.

        Construction-correctness gate: if this fails, the compiler is
        wiring weights wrong (Q/K/V row scatter, bias width mismatch,
        Linear transpose bug) — a different bug class from fp32 drift.
        """
        worst = max(fp64_report.per_node.values(), key=lambda r: r.max_abs_error)
        assert worst.max_abs_error < 1e-5, (
            f"fp64 replay divergence exceeds 1e-5 — compiler may be wiring "
            f"weights incorrectly (not an fp32 accumulation issue).\n"
            f"  Worst node: {worst.summary()}"
        )

    def test_gate_M_below_20k(self, compiled):
        """No approximate-cond_gate site has M ≥ 20,000.

        This is the direct structural invariant Phase E violated. Even
        a single over-loose declared range on an upstream op inflates M
        at every downstream gate, and M amplifies compare-noise (~0.005)
        into M·ε_cond units of fp32 error. Keep M bounded and the
        rel/decl test above stays meaningful.
        """
        _, probe_root, _ = compiled
        sites = _approx_gate_sites(probe_root)
        assert sites, "no approximate-gate sites found — graph changed shape?"
        worst = max(sites, key=lambda s: s["M"])
        assert worst["M"] < 20_000, (
            f"Approximate-gate M = {worst['M']:.0f} at node "
            f"id={worst['node'].node_id} name={worst['name']!r} — "
            f"exceeds 20K ceiling.\n"
            f"Back-trace this node's cond input's value_range to the "
            f"upstream op that over-declares. See "
            f"docs/postmortems/phase_e_xfail.md for the Phase E template."
        )

    def test_clean_ops_abs_below_0_02(self, fp32_reports):
        """Pre-amplifier ops (linear1, ReLU, Attn, bsp_*) stay near the
        compare-noise floor.

        These ops run *before* any approximate-gate amplification. If
        they grow above 0.02 abs, it points at upstream op regression
        or a new amplifier being introduced before the gate.
        """
        combined = self._combined_worst(fp32_reports)
        offenders = []
        for rec in combined.values():
            if _classify(rec.node) != "clean":
                continue
            if rec.max_abs_error >= 0.02:
                offenders.append((rec.node, rec.max_abs_error))
        offenders.sort(key=lambda t: -t[1])
        assert not offenders, (
            f"{len(offenders)} clean-op nodes exceed 0.02 abs. Top 5:\n"
            + self._format_offenders(offenders)
        )

    def test_tail_abs_below_65(self, fp32_reports):
        """Approximate-gate output chain stays below 65 abs.

        Ceiling derivation: M ceiling (20K) × c_tol (0.005) × typical
        cancellation factor ≈ 65. A failure here means either M blew up
        (covered by the M-audit test) or condition noise regressed
        (covered by clean-ops test) — those should fire first.
        """
        combined = self._combined_worst(fp32_reports)
        offenders = []
        for rec in combined.values():
            if _classify(rec.node) != "tail":
                continue
            if rec.max_abs_error >= 65:
                offenders.append((rec.node, rec.max_abs_error))
        offenders.sort(key=lambda t: -t[1])
        assert not offenders, (
            f"{len(offenders)} tail nodes exceed 65 abs. Top 5:\n"
            + self._format_offenders(offenders)
        )

    # ── Stretch tests (xfail today, encode remaining debt) ───────────
    #
    # These tests represent what the graph *should* look like to meet
    # the rendering pipeline's default 0.99/0.15 match thresholds
    # (currently loosened to 0.96/0.30 at tests/doom/test_game_graph.py).
    # Each xfail reason names the specific debt blocking it.

    @pytest.mark.xfail(
        strict=True,
        reason=(
            "stretch: gate M < 5000. Currently ~16,822 worst-case at one of "
            "the wall-geometry select sites. Tightening requires either (a) "
            "auditing and declaring tight output ranges on remaining "
            "piecewise / multiply ops (same fix pattern as Phase E's "
            "piecewise_linear_2d), or (b) switching selected sites to "
            "approximate=False cond_gate at a cost of +2 layers per site."
        ),
    )
    def test_gate_M_below_5k(self, compiled):
        """Stretch target for gate M."""
        _, probe_root, _ = compiled
        sites = _approx_gate_sites(probe_root)
        assert sites
        worst = max(sites, key=lambda s: s["M"])
        assert worst["M"] < 5_000, f"worst M = {worst['M']:.0f} at {worst['name']!r}"

    @pytest.mark.xfail(
        strict=True,
        reason=(
            "stretch: tail abs < 15. Currently ~52 worst-case. This is the "
            "headline precision goal — meeting it should let the oblique "
            "rendering tests pass at the library's default 0.99 / 0.15 "
            "thresholds (they're currently loosened to 0.96 / 0.30 at "
            "tests/doom/test_game_graph.py). Derivation: 5K-M × 0.005 "
            "compare × 0.6 sign-cancellation = 15. Gating on this test and "
            "the gate-M stretch test together motivates the M-tightening "
            "work."
        ),
    )
    def test_tail_abs_below_15(self, fp32_reports):
        """Stretch target for approximate-gate tail."""
        combined = self._combined_worst(fp32_reports)
        offenders = []
        for rec in combined.values():
            if _classify(rec.node) != "tail":
                continue
            if rec.max_abs_error >= 15:
                offenders.append(rec.node)
        assert not offenders, f"{len(offenders)} tail nodes exceed 15 abs"
