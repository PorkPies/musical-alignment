import os
import shutil
import sys
import subprocess
import urllib.request
import numpy as np

import soundfile as sf  # required by some fluidSynth setups
import fluidsynth

# Add parent directory to import path for extract_cqt
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from data.scripts.extract_audio_features import extract_cqt

def download_soundfont(soundfont_path):
    """
    Download the FluidR3_GM.sf2 soundfont if it's not already available.

    Parameters:
        soundfont_path (str): Local path where the soundfont will be saved.
    """
    url = "https://github.com/Jacalz/fluid-soundfont/raw/refs/heads/master/original-files/FluidR3_GM.sf2"
    print("Downloading FluidR3_GM.sf2 soundfont...")
    urllib.request.urlretrieve(url, soundfont_path)
    print("Downloaded soundfont to:", soundfont_path)

def synthesize_midi_to_wav(midi_path, wav_path, soundfont_path):
    """
    Use FluidSynth to convert a MIDI file to WAV using the provided soundfont.

    Parameters:
        midi_path (str): Path to the MIDI file.
        wav_path (str): Path where the WAV file will be saved.
        soundfont_path (str): Path to the .sf2 soundfont file.
    """
    print(f"Synthesizing {midi_path} -> {wav_path}")
    cmd = [
        "fluidsynth",
        "-ni",
        soundfont_path,
        midi_path,
        "-F", wav_path,
        "-r", "44100"
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"FluidSynth error: {result.stderr}")

def generate_synthetic_data(midi_dir, output_dir, soundfont_path="FluidR3_GM.sf2"):
    """
    Process all MIDI files in a directory:
    - Synthesize each to WAV using FluidSynth
    - Extract Constant-Q Transform (CQT) features
    - Save the CQT as a NumPy array

    Parameters:
        midi_dir (str): Directory containing input MIDI files.
        output_dir (str): Directory to save processed CQT .npy files.
        soundfont_path (str): Path to the .sf2 soundfont file.
    """
    if not os.path.exists(soundfont_path):
        download_soundfont(soundfont_path)

    if not os.path.exists(midi_dir):
        print(f"Input MIDI directory '{midi_dir}' does not exist. Please add MIDI files and rerun.")
        return

    os.makedirs(output_dir, exist_ok=True)

    for i, midi_file in enumerate(os.listdir(midi_dir)):
        print(f"Found file: {midi_file}")
        if not midi_file.endswith(".mid"):
            print("Skipped (not a .mid file).")
            continue

        midi_path = os.path.join(midi_dir, midi_file)
        wav_path = os.path.join(output_dir, f"temp_{i}.wav")
        cqt_output = os.path.join(output_dir, f"{i}_{midi_file[:-4]}.npy")

        try:
            print(f"Processing: {midi_file}")
            synthesize_midi_to_wav(midi_path, wav_path, soundfont_path)
            extract_cqt(wav_path, cqt_output)
            print(f"Saved: {cqt_output}")
        except Exception as e:
            print(f"Error processing {midi_file}: {e}")
        finally:
            if os.path.exists(wav_path):
                os.remove(wav_path)

if __name__ == "__main__":
    # Determine base project directory
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    
    # Set default input and output paths
    midi_dir = os.path.join(base_dir, "data", "raw")
    output_dir = os.path.join(base_dir, "data", "processed")

    print(f"Generating synthetic data from:\n  MIDI: {midi_dir}\n  Output: {output_dir}")
    generate_synthetic_data(midi_dir, output_dir)
    print("Dataset generation complete.")
