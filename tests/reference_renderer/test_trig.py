import numpy as np
import pytest

from torchwright.reference_renderer.trig import generate_trig_table


@pytest.fixture
def table():
    return generate_trig_table()


def test_shape(table):
    assert table.shape == (256, 2)


def test_cardinal_directions(table):
    """cos/sin at 0, 90, 180, 270 degrees (indices 0, 64, 128, 192)."""
    atol = 1e-15
    # 0 degrees: cos=1, sin=0
    assert abs(table[0, 0] - 1.0) < atol
    assert abs(table[0, 1] - 0.0) < atol
    # 90 degrees: cos=0, sin=1
    assert abs(table[64, 0]) < atol
    assert abs(table[64, 1] - 1.0) < atol
    # 180 degrees: cos=-1, sin=0
    assert abs(table[128, 0] - (-1.0)) < atol
    assert abs(table[128, 1]) < atol
    # 270 degrees: cos=0, sin=-1
    assert abs(table[192, 0]) < atol
    assert abs(table[192, 1] - (-1.0)) < atol


def test_sin_cos_relationship(table):
    """sin(i) == cos(i - 64) for all i (quarter-turn phase shift)."""
    for i in range(256):
        j = (i - 64) % 256
        assert abs(table[i, 1] - table[j, 0]) < 1e-12


def test_unit_circle(table):
    """cos^2 + sin^2 == 1 for all entries."""
    norms = table[:, 0] ** 2 + table[:, 1] ** 2
    np.testing.assert_allclose(norms, 1.0, atol=1e-14)
