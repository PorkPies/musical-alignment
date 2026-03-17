# Musical Alignment

A pipeline for training a CNN to classify which bar of a musical score a given audio snippet corresponds to. Built around Bach chorales, it takes MIDI and MusicXML scores, synthesizes audio, extracts CQT features, and trains a classifier.

## Pipeline Overview

```
MusicXML / MIDI  →  CQT features  →  bar-labelled snippets  →  CNN training
```

### 1. Data acquisition

**`data/scripts/download_bach_chorales.py`**
Downloads Bach chorales from the music21 corpus and saves them as both MIDI (`data/raw/`) and MusicXML (`data/scores/`).

```bash
python data/scripts/download_bach_chorales.py
```

**`data/scripts/convert_musicxml_to_midi.py`**
Converts an existing MusicXML file to MIDI using music21.

```bash
python data/scripts/convert_musicxml_to_midi.py
```

### 2. Synthetic audio generation

**`data/scripts/generate_synthetic_dataset.py`**
Synthesizes each MIDI file to WAV using FluidSynth + the FluidR3_GM soundfont (auto-downloaded if missing), then extracts CQT features and saves them as `.npy` files in `data/processed/`.

```bash
python data/scripts/generate_synthetic_dataset.py
```

### 3. Feature extraction

**`data/scripts/extract_audio_features.py`**
Extracts a dB-scaled Constant-Q Transform (CQT) from any WAV file and saves it as a NumPy array.

```python
from data.scripts.extract_audio_features import extract_cqt
extract_cqt("audio.wav", "output.npy")
```

### 4. Score alignment

**`data/scripts/extract_bar_times.py`**
Parses a MusicXML score to produce a list of bar onset times in seconds, using the score's tempo and time signature.

**`data/scripts/match_features_to_scores.py`**
Pairs each processed CQT `.npy` file with its corresponding MusicXML score file based on filename conventions.

**`data/scripts/split_snippets.py`**
Slices each CQT array into overlapping 128-frame snippets (64-frame hop), labels each snippet with its nearest bar number, and saves them to `data/snippets/`.

```bash
python data/scripts/split_snippets.py
```

### 5. Training

**`models/train.py`**
Trains the `BaselineCNN` on the labelled snippets. Uses an 80/20 train/validation split and logs train and validation loss each epoch.

```bash
python models/train.py
```

Config (edit directly in `models/train.py`):

| Parameter | Default |
|-----------|---------|
| `BATCH_SIZE` | 8 |
| `EPOCHS` | 10 |
| `LEARNING_RATE` | 1e-3 |
| `VAL_SPLIT` | 0.2 |

## Model

**`models/baseline_model.py`** — `BaselineCNN`

A two-block CNN followed by a linear classifier:
- Conv2d(1→32) + ReLU + MaxPool
- Conv2d(32→64) + ReLU + AdaptiveAvgPool → (64,)
- Linear(64 → num_classes)

`num_classes` is determined at runtime from the number of unique bar positions in the snippet dataset.

### 6. Audio augmentation

**`data/scripts/augment_audio.py`**
Applies six augmentations to a synthesized WAV to improve generalisation to real instrument audio. Each produces an independently saved WAV variant.

| Augmentation | Description |
|---|---|
| `pitch_up` | Pitch shift +2 semitones |
| `pitch_down` | Pitch shift −2 semitones |
| `stretch_slow` | Time stretch × 0.9 |
| `stretch_fast` | Time stretch × 1.1 |
| `noise` | Additive Gaussian noise |
| `reverb` | Exponential-decay convolution reverb |

Call `augment_audio(wav_path, output_dir, base_name)` to apply all six, or pass an `augmentations` list to select a subset.

### 7. Inference

**`models/inference.py`**
Runs the trained model on audio in three modes:

| Flag | Mode |
|---|---|
| `--wav path` | Offline: slide over a WAV file and print bar predictions |
| `--mic` | Live: stream from the default microphone |
| `--pipe path` | Pipe: read raw 16-bit mono PCM from a named pipe (e.g. FluidSynth stdout) |

```bash
python models/inference.py --wav data/test/temp_0.wav
python models/inference.py --mic
```

A rolling majority-vote smoother (`SMOOTH_WINDOW = 5` predictions) is applied before each result is emitted. An optional `callback(bar_number, confidence, time_sec)` can be passed for integration with the display layer.

Key parameters (edit in `models/inference.py`):

| Parameter | Default | Description |
|---|---|---|
| `BUFFER_SECONDS` | 3 | Audio buffer length for live/pipe modes |
| `STRIDE_FRAMES` | 64 | CQT frames between inference calls |
| `SMOOTH_WINDOW` | 5 | Predictions to majority-vote over |

### 8. Score display

**`display/score_renderer.py`**
Calls the MuseScore CLI to render a MusicXML file to per-page PNG images, then builds a `{bar_number: page_index}` mapping via music21. Falls back to a configurable `DEFAULT_MEASURES_PER_PAGE = 8` if page count cannot be determined.

**`display/page_turner.py`** — `PageTurner`
Receives predicted bar numbers from the inference pipeline (via `push_prediction`) and decides when to change pages. Two strategies:
- **Reactive:** turns immediately when the model's prediction crosses a page boundary.
- **Predictive:** fires a turn `LEAD_TIME` seconds (default 2.5 s) before the first bar of the next page, using bar onset times from `extract_bar_times`.

**`display/app.py`** — `ScoreDisplay`
A `tkinter` window that pre-loads all page PNGs and polls a thread-safe queue for page-change events. Arrow keys allow manual flipping; Escape closes the window.

### 9. Demo

**`demo/run_demo.py`**
Single entry point for the end-to-end PoC:

```bash
python demo/run_demo.py [path/to/score.musicxml] [--midi ...] [--checkpoint ...]
```

1. Renders score pages via `score_renderer.py`.
2. Synthesizes the MIDI to WAV via FluidSynth.
3. Opens the `ScoreDisplay` window.
4. Runs inference on the WAV in a background thread.
5. Inference predictions feed `PageTurner`, which drives the display.

Pass `--no-display` for headless inference output only.

---

## Module reference

| Module | Key export | Responsibility |
|---|---|---|
| `data/scripts/extract_audio_features.py` | `extract_cqt(wav, out, sr)` | Compute dB CQT from WAV, save as `.npy` |
| `data/scripts/extract_bar_times.py` | `extract_bar_times(xml)` | Parse MusicXML → list of bar onset times (seconds) |
| `data/scripts/augment_audio.py` | `augment_audio(wav, out_dir, base_name)` | Write augmented WAV variants |
| `data/scripts/generate_synthetic_dataset.py` | `generate_synthetic_data(...)` | Synthesize all MIDIs → WAV → CQT |
| `data/scripts/split_snippets.py` | `split_and_label(...)` | Slice CQT into bar-labelled 128-frame snippets |
| `data/scripts/match_features_to_scores.py` | `get_score_match_map()` | Map `piece_id → MusicXML path` by filename |
| `models/baseline_model.py` | `BaselineCNN` | 2-block CNN outputting bar-class logits |
| `models/utils.py` | `CQTBarWithScoreDataset` | PyTorch Dataset: loads snippets, maps bars to class indices |
| `models/train.py` | `train()` | Training loop; logs val accuracy; saves `models/checkpoint.pt` |
| `models/inference.py` | `run_offline / run_live / run_from_pipe` | Sliding-window inference with majority-vote smoothing |
| `display/score_renderer.py` | `render_score_pages / build_bar_to_page` | MuseScore rendering + bar→page mapping |
| `display/page_turner.py` | `PageTurner` | Reactive + predictive page turn logic |
| `display/app.py` | `ScoreDisplay` | tkinter score display with thread-safe page changes |
| `demo/run_demo.py` | `main()` | End-to-end demo entry point |

---

## Tests

Run the suite with:

```bash
/path/to/venv/bin/python -m pytest tests/ -v
```

Tests live in `tests/` and use `pytest`. No network access, FluidSynth, or MuseScore is required — all external calls (MuseScore subprocess, file I/O) are either faked with synthetic data or mocked.

| Test file | Module under test | What is verified |
|---|---|---|
| `test_extract_audio_features.py` | `extract_audio_features` | Output file created; array is 2-D with 84 freq bins; values ≤ 0 dB; missing parent directories created automatically |
| `test_extract_bar_times.py` | `extract_bar_times` | Real MusicXML parsed correctly: non-empty, first bar at t=0, strictly increasing, all floats, duration in plausible range |
| `test_augment_audio.py` | `augment_audio` | Noise and reverb preserve length and clip to ±1; `augment_audio` writes all six variants, each is a valid WAV; subset and unknown-name handling |
| `test_baseline_model.py` | `BaselineCNN` | Forward-pass output shape `(batch, num_classes)`; gradients flow to all parameters; works for arbitrary class counts |
| `test_inference.py` | `inference` | `_majority` correctness; CQT shape (84 bins, ≥128 frames, ≤0 dB); `load_model` returns eval-mode model; `predict_snippet` returns valid bar + confidence in [0, 1]; `run_offline` fires callback |
| `test_page_turner.py` | `page_turner` | `_build_page_first_and_last` maps correctly; reactive turns fire on page crossing; no spurious turns within a page; page-skip handled; `current_page` tracks state; predictive turn fires within lead time, does not fire outside it |
| `test_score_renderer.py` | `score_renderer` | `build_bar_to_page` covers all unique bar numbers, indices within page count, empty-pages fallback; `render_score_pages` collects numbered PNGs, handles missing MuseScore, falls back to unsuffixed PNG |

---

## Data layout

```
data/
  raw/          # MIDI files
  scores/       # MusicXML files
  processed/    # Per-piece CQT .npy files
  snippets/     # Bar-labelled 128-frame CQT snippets
  test/         # Scratch files for testing scripts
models/
  baseline_model.py
  utils.py      # CQTBarWithScoreDataset (PyTorch Dataset)
  train.py
  inference.py
  checkpoint.pt # Written by train.py (not committed)
display/
  score_renderer.py
  page_turner.py
  app.py
  rendered/     # PNG pages written at runtime (not committed)
demo/
  run_demo.py
tests/
  conftest.py   # Shared fixtures
  test_*.py     # One file per module
```

## Dependencies

Key libraries: `torch`, `librosa`, `music21`, `pyfluidsynth`, `soundfile`, `numpy`, `scipy`, `Pillow`, `scikit-learn`.

Install all dependencies:

```bash
pip install -r requirements.txt
```

System packages also required:
- `fluidsynth` — MIDI synthesis (`apt install fluidsynth`)
- `musescore` — score rendering for the display (`apt install musescore3`)
