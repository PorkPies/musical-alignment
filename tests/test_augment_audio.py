"""Tests for data/scripts/augment_audio.py."""
import os
import numpy as np
import pytest
import soundfile as sf

from data.scripts.augment_audio import (
    AUGMENTATIONS,
    add_gaussian_noise,
    add_reverb,
    augment_audio,
)

SR = 22050
_rng = np.random.default_rng(0)
_AUDIO = (_rng.uniform(-0.5, 0.5, SR * 2)).astype(np.float32)  # 2-second dummy audio


# ---------------------------------------------------------------------------
# add_gaussian_noise
# ---------------------------------------------------------------------------

def test_noise_preserves_length():
    out = add_gaussian_noise(_AUDIO)
    assert len(out) == len(_AUDIO)


def test_noise_clips_to_unit():
    out = add_gaussian_noise(_AUDIO, noise_std=1.0)
    assert out.max() <= 1.0 and out.min() >= -1.0


def test_noise_changes_signal():
    out = add_gaussian_noise(_AUDIO, noise_std=0.1)
    assert not np.allclose(out, _AUDIO), "Noise should change the signal"


# ---------------------------------------------------------------------------
# add_reverb
# ---------------------------------------------------------------------------

def test_reverb_preserves_length():
    out = add_reverb(_AUDIO, SR)
    assert len(out) == len(_AUDIO)


def test_reverb_clips_to_unit():
    out = add_reverb(_AUDIO, SR)
    assert out.max() <= 1.0 and out.min() >= -1.0


def test_reverb_changes_signal():
    out = add_reverb(_AUDIO, SR)
    assert not np.allclose(out, _AUDIO), "Reverb should change the signal"


# ---------------------------------------------------------------------------
# augment_audio (end-to-end)
# ---------------------------------------------------------------------------

def test_augment_audio_default_creates_all(sine_wav, tmp_path):
    """With default settings, one file per augmentation is written."""
    out_dir = str(tmp_path / "aug")
    paths = augment_audio(sine_wav, out_dir, "test")
    assert len(paths) == len(AUGMENTATIONS)


def test_augment_audio_files_exist(sine_wav, tmp_path):
    out_dir = str(tmp_path / "aug")
    paths = augment_audio(sine_wav, out_dir, "test")
    for p in paths:
        assert os.path.exists(p), f"Expected file {p} to exist"


def test_augment_audio_outputs_are_readable(sine_wav, tmp_path):
    """Each augmented WAV can be loaded back via soundfile."""
    out_dir = str(tmp_path / "aug")
    paths = augment_audio(sine_wav, out_dir, "test")
    for p in paths:
        data, sr = sf.read(p)
        assert sr == SR
        assert len(data) > 0


def test_augment_audio_subset(sine_wav, tmp_path):
    """Requesting only two augmentations produces exactly two files."""
    out_dir = str(tmp_path / "aug_sub")
    paths = augment_audio(sine_wav, out_dir, "test", augmentations=["noise", "reverb"])
    assert len(paths) == 2


def test_augment_audio_unknown_name_is_skipped(sine_wav, tmp_path):
    """An unknown augmentation name is skipped without raising."""
    out_dir = str(tmp_path / "aug_skip")
    paths = augment_audio(sine_wav, out_dir, "test", augmentations=["noise", "nonexistent"])
    assert len(paths) == 1  # only 'noise' succeeds
