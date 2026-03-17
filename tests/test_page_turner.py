"""Tests for display/page_turner.py — _build_page_first_and_last and PageTurner."""
import time
import threading
import pytest

from display.page_turner import PageTurner, _build_page_first_and_last


# ---------------------------------------------------------------------------
# _build_page_first_and_last
# ---------------------------------------------------------------------------

def test_first_and_last_basic(bar_to_page):
    """Bars 1-10 on page 0, 11-20 on page 1, 21-30 on page 2."""
    first, last = _build_page_first_and_last(bar_to_page)
    assert first[0] == 1
    assert last[0] == 10
    assert first[1] == 11
    assert last[1] == 20
    assert first[2] == 21
    assert last[2] == 30


def test_single_page_single_bar():
    b2p = {1: 0}
    first, last = _build_page_first_and_last(b2p)
    assert first[0] == 1
    assert last[0] == 1


def test_all_bars_on_one_page():
    b2p = {i: 0 for i in range(1, 11)}
    first, last = _build_page_first_and_last(b2p)
    assert first[0] == 1
    assert last[0] == 10


# ---------------------------------------------------------------------------
# PageTurner — reactive turns
# ---------------------------------------------------------------------------

def _make_turner(bar_to_page, bar_times, callback, lead_time=0.0):
    return PageTurner(bar_to_page, bar_times, callback, lead_time=lead_time)


def test_reactive_turn_fires_callback(bar_to_page, bar_times):
    """Pushing a bar on page 1 while on page 0 triggers callback(1)."""
    events = []
    turner = _make_turner(bar_to_page, bar_times, lambda p: events.append(p), lead_time=0.0)
    turner.start()
    turner.push_prediction(11)  # bar 11 → page 1
    time.sleep(0.2)
    turner.stop()
    assert 1 in events, f"Expected page 1 in events, got {events}"


def test_no_spurious_turn_on_same_page(bar_to_page, bar_times):
    """Predictions within the same page should not fire a callback."""
    events = []
    turner = _make_turner(bar_to_page, bar_times, lambda p: events.append(p), lead_time=0.0)
    turner.start()
    for bar in range(1, 10):
        turner.push_prediction(bar)
    time.sleep(0.2)
    turner.stop()
    assert events == [], f"Unexpected page changes: {events}"


def test_skipped_page_jump(bar_to_page, bar_times):
    """Jumping from page 0 directly to bar 25 (page 2) fires callback with 2."""
    events = []
    turner = _make_turner(bar_to_page, bar_times, lambda p: events.append(p), lead_time=0.0)
    turner.start()
    turner.push_prediction(25)
    time.sleep(0.2)
    turner.stop()
    assert 2 in events


def test_current_page_tracks_state(bar_to_page, bar_times):
    """current_page attribute stays in sync with the last page change."""
    turner = _make_turner(bar_to_page, bar_times, lambda p: None, lead_time=0.0)
    turner.start()
    turner.push_prediction(15)  # → page 1
    time.sleep(0.2)
    assert turner.current_page == 1
    turner.push_prediction(25)  # → page 2
    time.sleep(0.2)
    assert turner.current_page == 2
    turner.stop()


# ---------------------------------------------------------------------------
# PageTurner — predictive turns
# ---------------------------------------------------------------------------

def test_predictive_turn_fires_early(bar_to_page):
    """With lead_time = 5s, a bar 4s before the page boundary triggers early turn."""
    # bar_times: bar N starts at N-1 seconds (so page boundary at bar 11 = t=10)
    bar_times = [float(i) for i in range(30)]
    events = []

    turner = _make_turner(bar_to_page, bar_times, lambda p: events.append(p), lead_time=5.0)
    turner.start()
    # Bar 7 is at t=6s; next page starts at bar 11 (t=10s); time_remaining = 4s < lead_time=5s
    turner.push_prediction(7)
    time.sleep(0.2)
    turner.stop()
    assert 1 in events, f"Expected predictive turn to page 1, events={events}"


def test_predictive_turn_does_not_fire_too_early(bar_to_page):
    """With lead_time = 2s, a bar with 6s remaining should NOT trigger early turn."""
    bar_times = [float(i) for i in range(30)]
    events = []

    turner = _make_turner(bar_to_page, bar_times, lambda p: events.append(p), lead_time=2.0)
    turner.start()
    # Bar 4 at t=3s; next page boundary at t=10s; time_remaining = 7s > 2s
    turner.push_prediction(4)
    time.sleep(0.2)
    turner.stop()
    assert events == [], f"Should not have turned yet, events={events}"
