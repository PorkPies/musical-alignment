"""
Score renderer: converts a MusicXML file to per-page PNG images using MuseScore,
and builds a bar_number → page_index mapping via music21.
"""
import os
import sys
import subprocess
from music21 import converter

# Fallback: if MuseScore can't determine page layout, assume this many measures per page.
DEFAULT_MEASURES_PER_PAGE = 8

# MuseScore executable names to try in order.
MUSESCORE_CANDIDATES = ["mscore4", "mscore3", "mscore", "musescore", "MuseScore4", "MuseScore3"]


def render_score_pages(xml_path, output_dir):
    """
    Render a MusicXML file to per-page PNG images using MuseScore CLI.

    MuseScore exports pages as {base}-1.png, {base}-2.png, …

    Parameters:
        xml_path (str): Path to the MusicXML file.
        output_dir (str): Directory where PNGs are written.

    Returns:
        dict[int, str]: {page_index (0-based): png_path}
    """
    os.makedirs(output_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(xml_path))[0]
    out_prefix = os.path.join(output_dir, base)

    rendered = False
    for cmd in MUSESCORE_CANDIDATES:
        try:
            result = subprocess.run(
                [cmd, xml_path, "-o", out_prefix + ".png"],
                capture_output=True,
                text=True,
                timeout=120,
            )
            if result.returncode == 0:
                rendered = True
                break
            # Some MuseScore versions exit 0 even on soft errors; continue if no PNGs appear.
        except (FileNotFoundError, subprocess.TimeoutExpired):
            continue

    if not rendered:
        print(
            "Warning: MuseScore not found or failed. No page images rendered. "
            "Install MuseScore and ensure it is on PATH."
        )

    # Collect paginated PNGs: {base}-1.png, {base}-2.png, …
    page_images = {}
    i = 1
    while True:
        png = f"{out_prefix}-{i}.png"
        if os.path.exists(png):
            page_images[i - 1] = png
            i += 1
        else:
            break

    # Fallback: single-page export (no suffix)
    if not page_images:
        single = out_prefix + ".png"
        if os.path.exists(single):
            page_images[0] = single

    return page_images


def build_bar_to_page(xml_path, page_images, measures_per_page=None):
    """
    Build a mapping from bar number (1-indexed, as used by extract_bar_times)
    to page index (0-indexed).

    If the total number of rendered pages is known, the measures are divided
    evenly. Otherwise DEFAULT_MEASURES_PER_PAGE is used.

    Parameters:
        xml_path (str): Path to the MusicXML file.
        page_images (dict): Output of render_score_pages (may be empty).
        measures_per_page (int | None): Override; inferred when None.

    Returns:
        dict[int, int]: {bar_number: page_index}
    """
    score = converter.parse(xml_path)
    measures = list(score.parts[0].getElementsByClass("Measure"))
    total_measures = len(measures)
    total_pages = max(len(page_images), 1)

    if measures_per_page is None:
        measures_per_page = max(1, round(total_measures / total_pages))

    bar_to_page = {}
    for i, measure in enumerate(measures):
        bar_num = measure.number if measure.number else (i + 1)
        page_idx = min(i // measures_per_page, total_pages - 1)
        bar_to_page[bar_num] = page_idx

    return bar_to_page


if __name__ == "__main__":
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    xml = (
        sys.argv[1]
        if len(sys.argv) > 1
        else os.path.join(base_dir, "data", "scores", "bach_chorale_0.musicxml")
    )
    out_dir = os.path.join(base_dir, "display", "rendered")

    pages = render_score_pages(xml, out_dir)
    print(f"Rendered {len(pages)} page(s):")
    for idx, path in sorted(pages.items()):
        print(f"  Page {idx + 1}: {path}")

    bar_map = build_bar_to_page(xml, pages)
    print(f"Bar-to-page mapping ({len(bar_map)} bars):", bar_map)
