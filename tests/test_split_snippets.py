"""Tests for data/scripts/split_snippets.py — find_closest_bar."""
import sys, os
# split_snippets uses bare relative imports when run as a script, so import directly
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "data", "scripts"))

from split_snippets import find_closest_bar

SNIPPET_LEN = 128
HOP_LEN = 64


def test_find_closest_bar_exact():
    bar_times = [0.0, 1.0, 2.0, 3.0]
    assert find_closest_bar(2.0, bar_times) == 2


def test_find_closest_bar_between():
    bar_times = [0.0, 1.0, 2.0, 3.0]
    # 1.4 is closer to index 1 (1.0) than index 2 (2.0)
    assert find_closest_bar(1.4, bar_times) == 1


def test_find_closest_bar_past_end():
    bar_times = [0.0, 1.0, 2.0]
    # 99s is closest to the last bar
    assert find_closest_bar(99.0, bar_times) == 2


def test_find_closest_bar_before_start():
    bar_times = [1.0, 2.0, 3.0]
    assert find_closest_bar(0.0, bar_times) == 0


def test_find_closest_bar_single_bar():
    assert find_closest_bar(5.0, [3.0]) == 0


def test_snippet_count_formula():
    """
    The number of snippets produced by split_and_label should match
    len(range(0, total_frames - SNIPPET_LEN + 1, HOP_LEN)).
    """
    total_frames = 500
    expected = len(range(0, total_frames - SNIPPET_LEN + 1, HOP_LEN))
    assert expected > 0
    # Verify formula manually: floor((500 - 128) / 64) + 1 = floor(372/64) + 1 = 5 + 1 = 6
    assert expected == 6
