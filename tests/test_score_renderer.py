"""Tests for display/score_renderer.py — build_bar_to_page (and render_score_pages stub)."""
import os
import pytest
from unittest.mock import patch, MagicMock
from conftest import REAL_XML

from display.score_renderer import build_bar_to_page, render_score_pages


# ---------------------------------------------------------------------------
# build_bar_to_page
# ---------------------------------------------------------------------------

def test_all_bars_assigned(bar_to_page):
    """Every bar number in the fixture has a page assignment."""
    for bar in range(1, 31):
        assert bar in bar_to_page


def test_page_indices_nonnegative(bar_to_page):
    assert all(p >= 0 for p in bar_to_page.values())


def test_page_indices_contiguous(bar_to_page):
    """Pages should be 0, 1, 2 with no gap."""
    pages = sorted(set(bar_to_page.values()))
    assert pages == list(range(len(pages)))


def test_build_bar_to_page_real_xml():
    """build_bar_to_page covers all unique bar numbers in the score."""
    from music21 import converter
    score = converter.parse(REAL_XML)
    measures = list(score.parts[0].getElementsByClass("Measure"))
    # bar_map is keyed by measure.number, which may not be unique (pickup bars,
    # repeated sections), so we compare against unique numbers.
    unique_bar_numbers = {
        m.number if m.number else (i + 1) for i, m in enumerate(measures)
    }

    fake_pages = {0: "/fake/page0.png"}
    bar_map = build_bar_to_page(REAL_XML, fake_pages)

    assert len(bar_map) == len(unique_bar_numbers), (
        f"Expected {len(unique_bar_numbers)} unique bars in map, got {len(bar_map)}"
    )


def test_build_bar_to_page_values_within_page_count():
    """Page indices should not exceed the total number of pages."""
    fake_pages = {0: "a.png", 1: "b.png"}
    bar_map = build_bar_to_page(REAL_XML, fake_pages)
    n_pages = len(fake_pages)
    assert all(p < n_pages for p in bar_map.values()), (
        "A bar was assigned to a page index ≥ total page count"
    )


def test_build_bar_to_page_empty_pages():
    """With no rendered pages, all bars fall on page 0."""
    bar_map = build_bar_to_page(REAL_XML, {})
    assert all(p == 0 for p in bar_map.values())


# ---------------------------------------------------------------------------
# render_score_pages — subprocess is mocked; we verify the logic, not MuseScore
# ---------------------------------------------------------------------------

def test_render_score_pages_collects_numbered_pngs(tmp_path):
    """render_score_pages picks up {base}-1.png, {base}-2.png, … after rendering."""
    base = os.path.splitext(os.path.basename(REAL_XML))[0]

    # Pre-create fake PNG files that MuseScore would have written
    for i in range(1, 4):
        (tmp_path / f"{base}-{i}.png").write_bytes(b"fake-png")

    with patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0)
        pages = render_score_pages(REAL_XML, str(tmp_path))

    assert len(pages) == 3
    assert set(pages.keys()) == {0, 1, 2}


def test_render_score_pages_returns_empty_when_no_pngs(tmp_path):
    """If MuseScore produces nothing (or is absent), an empty dict is returned."""
    with patch("subprocess.run", side_effect=FileNotFoundError):
        pages = render_score_pages(REAL_XML, str(tmp_path))
    assert pages == {}


def test_render_score_pages_falls_back_to_unsuffixed_png(tmp_path):
    """If only {base}.png exists (no suffix), it maps to page index 0."""
    base = os.path.splitext(os.path.basename(REAL_XML))[0]
    (tmp_path / f"{base}.png").write_bytes(b"fake-png")

    with patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0)
        pages = render_score_pages(REAL_XML, str(tmp_path))

    assert pages == {0: str(tmp_path / f"{base}.png")}
