"""Tests for models/inference.py."""
import os
import torch
import numpy as np
import pytest

from models.inference import (
    _majority,
    extract_cqt_from_audio,
    load_model,
    predict_snippet,
)

SNIPPET_LEN = 128


# ---------------------------------------------------------------------------
# _majority
# ---------------------------------------------------------------------------

def test_majority_single():
    assert _majority([5]) == 5


def test_majority_clear_winner():
    assert _majority([1, 2, 1, 1, 3]) == 1


def test_majority_two_elements():
    # With only two distinct values, the one appearing more often wins
    result = _majority([7, 7, 8])
    assert result == 7


# ---------------------------------------------------------------------------
# extract_cqt_from_audio
# ---------------------------------------------------------------------------

def test_cqt_from_audio_is_2d(short_audio):
    cqt = extract_cqt_from_audio(short_audio)
    assert cqt.ndim == 2


def test_cqt_from_audio_has_84_bins(short_audio):
    cqt = extract_cqt_from_audio(short_audio)
    assert cqt.shape[0] == 84


def test_cqt_from_audio_enough_frames(short_audio):
    """Audio is long enough to yield at least SNIPPET_LEN frames."""
    cqt = extract_cqt_from_audio(short_audio)
    assert cqt.shape[1] >= SNIPPET_LEN


def test_cqt_values_nonpositive(short_audio):
    """dB CQT with ref=np.max should have max value ≤ 0."""
    cqt = extract_cqt_from_audio(short_audio)
    assert cqt.max() <= 0.0


# ---------------------------------------------------------------------------
# load_model
# ---------------------------------------------------------------------------

def test_load_model_returns_model_and_map(checkpoint):
    device = torch.device("cpu")
    model, bar_to_class = load_model(checkpoint, device)
    from models.baseline_model import BaselineCNN
    assert isinstance(model, BaselineCNN)
    assert isinstance(bar_to_class, dict)
    assert len(bar_to_class) > 0


def test_load_model_is_eval_mode(checkpoint):
    device = torch.device("cpu")
    model, _ = load_model(checkpoint, device)
    assert not model.training, "Model should be in eval mode after loading"


# ---------------------------------------------------------------------------
# predict_snippet
# ---------------------------------------------------------------------------

def test_predict_snippet_returns_bar_and_confidence(checkpoint, snippet):
    device = torch.device("cpu")
    model, bar_to_class = load_model(checkpoint, device)
    bar, conf = predict_snippet(model, snippet, bar_to_class, device)
    assert isinstance(bar, int)
    assert isinstance(conf, float)


def test_predict_snippet_bar_in_known_bars(checkpoint, snippet):
    device = torch.device("cpu")
    model, bar_to_class = load_model(checkpoint, device)
    bar, _ = predict_snippet(model, snippet, bar_to_class, device)
    assert bar in bar_to_class, f"Predicted bar {bar} not in bar_to_class"


def test_predict_snippet_confidence_in_unit_interval(checkpoint, snippet):
    device = torch.device("cpu")
    model, bar_to_class = load_model(checkpoint, device)
    _, conf = predict_snippet(model, snippet, bar_to_class, device)
    assert 0.0 <= conf <= 1.0, f"Confidence {conf} outside [0, 1]"


def test_run_offline_calls_callback(checkpoint, sine_wav):
    """run_offline fires the callback at least once for a valid WAV."""
    from models.inference import run_offline

    results = []

    def cb(bar, conf, t):
        results.append((bar, conf, t))

    run_offline(sine_wav, checkpoint, callback=cb)
    assert len(results) > 0, "Callback was never called"
