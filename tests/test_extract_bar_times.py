"""Tests for data/scripts/extract_bar_times.py — extract_bar_times()."""
import pytest
from conftest import REAL_XML

from data.scripts.extract_bar_times import extract_bar_times


@pytest.fixture(scope="module")
def bar_times_real():
    return extract_bar_times(REAL_XML)


def test_returns_list(bar_times_real):
    assert isinstance(bar_times_real, list)


def test_nonempty(bar_times_real):
    assert len(bar_times_real) > 0, "Expected at least one bar"


def test_first_bar_at_zero(bar_times_real):
    """First bar starts at time 0."""
    assert bar_times_real[0] == pytest.approx(0.0), "First bar should start at t=0"


def test_monotonically_increasing(bar_times_real):
    """Bar onset times must strictly increase."""
    for i in range(1, len(bar_times_real)):
        assert bar_times_real[i] > bar_times_real[i - 1], (
            f"Bar {i + 1} onset ({bar_times_real[i]:.3f}s) ≤ "
            f"bar {i} onset ({bar_times_real[i - 1]:.3f}s)"
        )


def test_all_floats(bar_times_real):
    assert all(isinstance(t, float) for t in bar_times_real)


def test_reasonable_duration(bar_times_real):
    """A Bach chorale shouldn't be shorter than 10s or longer than 10 minutes."""
    total = bar_times_real[-1]
    assert 10.0 < total < 600.0, f"Total duration {total:.1f}s looks wrong"
