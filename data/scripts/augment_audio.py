import os
import sys
import numpy as np
import librosa
import soundfile as sf
from scipy.signal import fftconvolve


def add_gaussian_noise(y, noise_std=0.005):
    noise = np.random.normal(0, noise_std, y.shape).astype(y.dtype)
    return np.clip(y + noise, -1.0, 1.0)


def add_reverb(y, sr, decay=0.3, delay_ms=30):
    """Simple reverb via exponential decay impulse response."""
    delay_samples = int(sr * delay_ms / 1000)
    ir_length = int(sr * 0.5)  # 500 ms IR
    ir = np.zeros(ir_length)
    ir[0] = 1.0
    for k in range(1, 6):
        idx = delay_samples * k
        if idx < ir_length:
            ir[idx] = decay ** k
    wet = fftconvolve(y, ir)[:len(y)]
    return np.clip(wet / np.abs(wet).max() if np.abs(wet).max() > 0 else wet, -1.0, 1.0)


AUGMENTATIONS = {
    "pitch_up":     lambda y, sr: librosa.effects.pitch_shift(y, sr=sr, n_steps=2),
    "pitch_down":   lambda y, sr: librosa.effects.pitch_shift(y, sr=sr, n_steps=-2),
    "stretch_slow": lambda y, sr: librosa.effects.time_stretch(y, rate=0.9),
    "stretch_fast": lambda y, sr: librosa.effects.time_stretch(y, rate=1.1),
    "noise":        lambda y, sr: add_gaussian_noise(y),
    "reverb":       lambda y, sr: add_reverb(y, sr),
}


def augment_audio(wav_path, output_dir, base_name, augmentations=None, sr=22050):
    """
    Apply augmentations to a WAV file and save each variant.

    Parameters:
        wav_path (str): Path to the source WAV file.
        output_dir (str): Directory where augmented WAVs are saved.
        base_name (str): Filename prefix for outputs (no extension).
        augmentations (list[str] | None): Names of augmentations to apply.
            Defaults to all six defined in AUGMENTATIONS.
        sr (int): Sampling rate.

    Returns:
        list[str]: Paths of the written WAV files.
    """
    y, _ = librosa.load(wav_path, sr=sr)
    os.makedirs(output_dir, exist_ok=True)

    if augmentations is None:
        augmentations = list(AUGMENTATIONS.keys())

    output_paths = []
    for aug_name in augmentations:
        if aug_name not in AUGMENTATIONS:
            print(f"Warning: unknown augmentation '{aug_name}', skipping.")
            continue
        aug_fn = AUGMENTATIONS[aug_name]
        try:
            y_aug = aug_fn(y, sr)
            # Ensure float32 and trim/pad to original length
            y_aug = y_aug.astype(np.float32)
            out_path = os.path.join(output_dir, f"{base_name}_{aug_name}.wav")
            sf.write(out_path, y_aug, sr)
            output_paths.append(out_path)
        except Exception as e:
            print(f"Warning: augmentation '{aug_name}' failed for {wav_path}: {e}")

    return output_paths


if __name__ == "__main__":
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    wav = os.path.join(base_dir, "data", "test", "temp_0.wav")
    out_dir = os.path.join(base_dir, "data", "test", "augmented")
    if not os.path.exists(wav):
        print(f"Test WAV not found: {wav}")
        sys.exit(1)
    paths = augment_audio(wav, out_dir, "temp_0")
    print("Augmented files:")
    for p in paths:
        print(" ", p)
