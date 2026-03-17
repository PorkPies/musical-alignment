#!/usr/bin/env python3
"""
End-to-end PoC demo: synthesize a Bach chorale, run alignment inference,
and automatically advance the score display.

Usage:
    python demo/run_demo.py [path/to/score.musicxml] [options]

The demo:
  1. Pre-renders the MusicXML score to PNG pages (via MuseScore).
  2. Synthesizes the corresponding MIDI to a WAV file (via FluidSynth).
  3. Starts the score display window (tkinter).
  4. Runs offline inference on the synthesized WAV in a background thread.
  5. Feeds predicted bar numbers into the page turner, which drives the display.

For demo reliability, audio is synthesized to a WAV file first, so inference
does not depend on real-time microphone capture. Microphone/live-pipe modes
are available in models/inference.py for later use with real recordings.
"""
import os
import sys
import argparse
import subprocess
import tempfile
import threading

base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, base_dir)

from display.score_renderer import render_score_pages, build_bar_to_page
from display.page_turner import PageTurner
from display.app import ScoreDisplay
from data.scripts.extract_bar_times import extract_bar_times
from models.inference import run_offline

# Defaults — all relative to project root
DEFAULT_XML = os.path.join(base_dir, "data", "scores", "bach_chorale_0.musicxml")
DEFAULT_MIDI = os.path.join(base_dir, "data", "raw", "bach_chorale_0.mid")
DEFAULT_CHECKPOINT = os.path.join(base_dir, "models", "checkpoint.pt")

# Common soundfont locations
SOUNDFONT_CANDIDATES = [
    os.path.join(base_dir, "FluidR3_GM.sf2"),
    "/usr/share/sounds/sf2/FluidR3_GM.sf2",
    "/usr/share/soundfonts/FluidR3_GM.sf2",
    os.path.expanduser("~/FluidR3_GM.sf2"),
]


def find_soundfont():
    for path in SOUNDFONT_CANDIDATES:
        if os.path.exists(path):
            return path
    return None


def synthesize_to_wav(midi_path, wav_path, soundfont_path, sr=22050):
    """Use FluidSynth to synthesize MIDI → WAV at the inference sample rate."""
    cmd = [
        "fluidsynth", "-ni",
        soundfont_path, midi_path,
        "-F", wav_path,
        "-r", str(sr),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"FluidSynth failed:\n{result.stderr}")


def main():
    parser = argparse.ArgumentParser(
        description="Musical alignment PoC demo",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "xml",
        nargs="?",
        default=DEFAULT_XML,
        help="MusicXML score file",
    )
    parser.add_argument("--midi", default=DEFAULT_MIDI, help="MIDI file to synthesize")
    parser.add_argument(
        "--checkpoint",
        default=DEFAULT_CHECKPOINT,
        help="Trained model checkpoint (.pt)",
    )
    parser.add_argument(
        "--wav",
        default=None,
        help="Pre-synthesized WAV to use instead of calling FluidSynth",
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Disable the score display window (inference output only)",
    )
    parser.add_argument(
        "--lead-time",
        type=float,
        default=2.5,
        help="Seconds before page boundary to trigger predictive page turn",
    )
    args = parser.parse_args()

    # ------------------------------------------------------------------ #
    # 1. Validate inputs
    # ------------------------------------------------------------------ #
    if not os.path.exists(args.xml):
        print(f"ERROR: MusicXML file not found: {args.xml}")
        sys.exit(1)

    if not os.path.exists(args.checkpoint):
        print(f"ERROR: No model checkpoint found at {args.checkpoint}")
        print("Run `python models/train.py` first to train and save the model.")
        sys.exit(1)

    # ------------------------------------------------------------------ #
    # 2. Pre-render score pages
    # ------------------------------------------------------------------ #
    print("Rendering score pages...")
    rendered_dir = os.path.join(base_dir, "display", "rendered")
    page_images = render_score_pages(args.xml, rendered_dir)
    print(f"  {len(page_images)} page(s) rendered.")

    bar_to_page = build_bar_to_page(args.xml, page_images)
    bar_times = extract_bar_times(args.xml)
    print(f"  {len(bar_times)} bars parsed from score.")

    # ------------------------------------------------------------------ #
    # 3. Synthesize audio (if not supplied)
    # ------------------------------------------------------------------ #
    wav_path = args.wav
    _tmp_wav = None  # keep reference so it isn't GC'd before we finish

    if wav_path is None:
        soundfont = find_soundfont()
        if soundfont is None:
            print(
                "ERROR: No soundfont found. Place FluidR3_GM.sf2 in the project "
                "root or install it system-wide."
            )
            sys.exit(1)
        if not os.path.exists(args.midi):
            print(f"ERROR: MIDI file not found: {args.midi}")
            sys.exit(1)
        print(f"Synthesizing {os.path.basename(args.midi)} …")
        _tmp_wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        _tmp_wav.close()
        wav_path = _tmp_wav.name
        synthesize_to_wav(args.midi, wav_path, soundfont)
        print(f"  Synthesized to {wav_path}")

    # ------------------------------------------------------------------ #
    # 4. Set up page turner
    # ------------------------------------------------------------------ #
    if not args.no_display:
        display = ScoreDisplay(page_images, fullscreen=False)

        def on_page_change(page_idx):
            display.request_page(page_idx)
    else:
        display = None

        def on_page_change(page_idx):
            print(f"[page turner] → page {page_idx + 1}")

    turner = PageTurner(bar_to_page, bar_times, on_page_change, lead_time=args.lead_time)
    turner.start()

    # ------------------------------------------------------------------ #
    # 5. Run inference in a background thread
    # ------------------------------------------------------------------ #
    def inference_callback(bar_number, confidence, time_sec):
        turner.push_prediction(bar_number, confidence)

    inference_thread = threading.Thread(
        target=run_offline,
        args=(wav_path, args.checkpoint),
        kwargs={"callback": inference_callback},
        daemon=True,
    )
    inference_thread.start()
    print("Inference running in background…")

    # ------------------------------------------------------------------ #
    # 6. Block on the display (or wait for inference to finish)
    # ------------------------------------------------------------------ #
    if display is not None:
        print("Score display open. Press Escape to quit.")
        display.run()
    else:
        inference_thread.join()

    turner.stop()

    # Clean up temp WAV
    if _tmp_wav is not None and os.path.exists(wav_path):
        os.remove(wav_path)

    print("Demo complete.")


if __name__ == "__main__":
    main()
