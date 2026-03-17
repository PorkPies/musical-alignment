"""
Shared pytest fixtures.

All paths and synthetic data are created in temporary directories so tests
are hermetic and leave no artefacts in the source tree.
"""
import os
import sys
import tempfile

import numpy as np
import pytest
import soundfile as sf

# Make project root importable regardless of working directory
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Real assets bundled with the project
REAL_XML = os.path.join(PROJECT_ROOT, "data", "scores", "bach_chorale_0.musicxml")


# ---------------------------------------------------------------------------
# Audio / CQT helpers
# ---------------------------------------------------------------------------

SR = 22050
SNIPPET_LEN = 128


@pytest.fixture
def sine_wav(tmp_path):
    """Write a 3-second 440 Hz sine wave WAV and return its path."""
    duration = 3.0
    t = np.linspace(0, duration, int(SR * duration), endpoint=False)
    y = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    wav_path = str(tmp_path / "test_sine.wav")
    sf.write(wav_path, y, SR)
    return wav_path


@pytest.fixture
def short_audio():
    """Return a short numpy audio array (≥ SNIPPET_LEN CQT frames worth)."""
    # SNIPPET_LEN frames × hop_size samples, plus a little extra
    n_samples = SNIPPET_LEN * 512 + 4096
    t = np.linspace(0, n_samples / SR, n_samples, endpoint=False)
    return (0.4 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)


@pytest.fixture
def dummy_cqt():
    """Return a plausible CQT array (84 freq bins x 300 frames)."""
    rng = np.random.default_rng(42)
    return rng.uniform(-80, 0, size=(84, 300)).astype(np.float32)


@pytest.fixture
def snippet(dummy_cqt):
    """A single 128-frame CQT snippet."""
    return dummy_cqt[:, :SNIPPET_LEN]


# ---------------------------------------------------------------------------
# Score / bar-time helpers
# ---------------------------------------------------------------------------

@pytest.fixture
def bar_times():
    """Synthetic bar times: 30 bars, 1 second each."""
    return [float(i) for i in range(30)]


@pytest.fixture
def bar_to_page():
    """
    10 bars per page, 3 pages.
    Bars 1-10 → page 0, 11-20 → page 1, 21-30 → page 2.
    """
    return {bar: (bar - 1) // 10 for bar in range(1, 31)}


# ---------------------------------------------------------------------------
# Model helpers
# ---------------------------------------------------------------------------

NUM_CLASSES = 30


@pytest.fixture
def tiny_model():
    """An untrained BaselineCNN with NUM_CLASSES output classes."""
    import torch
    from models.baseline_model import BaselineCNN

    input_shape = (1, 84, SNIPPET_LEN)
    return BaselineCNN(input_shape, NUM_CLASSES)


@pytest.fixture
def checkpoint(tmp_path, tiny_model):
    """Save a minimal checkpoint and return its path."""
    import torch
    from models.baseline_model import BaselineCNN

    bar_to_class = {bar: bar - 1 for bar in range(1, NUM_CLASSES + 1)}
    ckpt_path = str(tmp_path / "checkpoint.pt")
    torch.save(
        {
            "model_state_dict": tiny_model.state_dict(),
            "num_classes": NUM_CLASSES,
            "input_shape": (1, 84, SNIPPET_LEN),
            "bar_to_class": bar_to_class,
        },
        ckpt_path,
    )
    return ckpt_path
