import librosa
import numpy as np
import os
import sys

def extract_cqt(audio_path, output_path, sr=22050):
    """
    Extracts Constant-Q Transform (CQT) features from an audio file
    and saves them as a NumPy .npy file.

    Parameters:
        audio_path (str): Path to the input audio file.
        output_path (str): Path to save the extracted CQT features.
        sr (int): Sampling rate used when loading the audio.
    """
    # Load the audio file at the given sampling rate
    y, _ = librosa.load(audio_path, sr=sr)

    # Compute CQT and convert to decibel scale
    cqt = librosa.amplitude_to_db(librosa.cqt(y, sr=sr), ref=np.max)

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Save the CQT array as a .npy file
    np.save(output_path, cqt)

if __name__ == "__main__":
    # Example file names (replace with real paths for actual use)
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    input_audio = os.path.join(base_dir, "data", "test", "temp_0.wav")
    output_features = os.path.join(base_dir, "data", "test", "example_cqt.npy")

    if os.path.exists(input_audio):
        print(f"Extracting CQT from {input_audio}...")
        extract_cqt(input_audio, output_features)
        print(f"CQT features saved to {output_features}")
    else:
        print(f"Input file not found: {input_audio}")
