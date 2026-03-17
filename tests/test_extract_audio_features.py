"""Tests for data/scripts/extract_audio_features.py — extract_cqt()."""
import os
import numpy as np
import pytest

from data.scripts.extract_audio_features import extract_cqt


def test_extract_cqt_saves_npy(sine_wav, tmp_path):
    """extract_cqt writes a .npy file at the specified output path."""
    out_path = str(tmp_path / "out.npy")
    extract_cqt(sine_wav, out_path)
    assert os.path.exists(out_path), "Output .npy file was not created"


def test_extract_cqt_output_is_2d(sine_wav, tmp_path):
    """CQT output has 2 dimensions (freq_bins × time_frames)."""
    out_path = str(tmp_path / "out.npy")
    extract_cqt(sine_wav, out_path)
    arr = np.load(out_path)
    assert arr.ndim == 2, f"Expected 2-D array, got shape {arr.shape}"


def test_extract_cqt_freq_bins(sine_wav, tmp_path):
    """Default librosa CQT produces 84 frequency bins."""
    out_path = str(tmp_path / "out.npy")
    extract_cqt(sine_wav, out_path)
    arr = np.load(out_path)
    assert arr.shape[0] == 84, f"Expected 84 freq bins, got {arr.shape[0]}"


def test_extract_cqt_values_are_db(sine_wav, tmp_path):
    """CQT values are in dB (≤ 0, since ref=np.max)."""
    out_path = str(tmp_path / "out.npy")
    extract_cqt(sine_wav, out_path)
    arr = np.load(out_path)
    assert arr.max() <= 0.0, "Max value should be ≤ 0 dB when ref=np.max"


def test_extract_cqt_creates_output_dir(sine_wav, tmp_path):
    """extract_cqt creates missing parent directories automatically."""
    nested = str(tmp_path / "a" / "b" / "c" / "out.npy")
    extract_cqt(sine_wav, nested)
    assert os.path.exists(nested)
