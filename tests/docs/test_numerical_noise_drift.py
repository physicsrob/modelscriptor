"""Drift test for the per-op noise measurement pipeline.

Re-runs `_measure_all()` on CPU and compares the freshly-regenerated JSON
to the committed `docs/op_noise_data.json`. Fails if any op's measured
numbers have drifted from what's been committed — the canonical signal
that an op's implementation, breakpoint grid, or measurement distribution
has changed without `make measure-noise` being re-run.

Pairs with `test_numerical_noise_consistency.py`:
  - `test_numerical_noise_consistency.py` (~30ms): verifies JSON, markdown,
    and docstring footers agree with each other. Format/schema drift.
  - `test_numerical_noise_drift.py` (this file, ~15s): verifies the JSON
    matches what the current code actually measures. Number drift.

Together they close the gap CLAUDE.md § "Numerical noise" used to
document as a manual obligation.
"""

from __future__ import annotations

import json

from scripts.measure_op_noise import (
    DOCS_JSON,
    _measure_all,
    render_json,
)


def _strip_metadata(text: str) -> dict:
    """Drop `commit` and `measured_at` from the JSON envelope.

    These fields change on every re-run and are noise for drift detection;
    the interesting content is the per-op measurement data.
    """
    data = json.loads(text)
    data.pop("commit", None)
    data.pop("measured_at", None)
    return data


def test_committed_measurements_match_current_code() -> None:
    """Re-measure every op and fail if the numbers no longer match the
    committed ``docs/op_noise_data.json``.

    To fix a failure of this test:
        1. Run ``make measure-noise`` to regenerate the JSON, markdown,
           and per-op docstring footers from a fresh measurement.
        2. Commit the regenerated files:
            - ``docs/op_noise_data.json``
            - ``docs/numerical_noise.md``
            - the updated ``.. noise-footer::`` blocks in
              ``torchwright/ops/*.py``.
        3. Per CLAUDE.md § "Numerical noise", diff the new JSON against
           the prior commit and update ``docs/numerical_noise_findings.md``
           for any findings-worthy changes.

    See CLAUDE.md § "Numerical noise" for the full workflow.
    """
    measurements = _measure_all()
    regenerated = render_json(
        measurements,
        commit="<ignored>",
        measured_at="<ignored>",
    )
    committed = DOCS_JSON.read_text()

    regenerated_data = _strip_metadata(regenerated)
    committed_data = _strip_metadata(committed)

    assert regenerated_data == committed_data, (
        "Per-op noise measurements have drifted from the committed values in "
        f"{DOCS_JSON.name}. Run `make measure-noise` to regenerate the noise "
        "artefacts (JSON, markdown, docstring footers), then commit the diff. "
        "See CLAUDE.md section 'Numerical noise' for the full workflow."
    )
