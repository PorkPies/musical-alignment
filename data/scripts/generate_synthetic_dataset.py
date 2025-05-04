import os
import subprocess
from data.scripts.extract_audio_features import extract_cqt

# Requires: fluidsynth, soundfont (e.g. FluidR3_GM.sf2)

def synthesize_midi_to_wav(midi_path, wav_path, soundfont_path):
    cmd = [
        "fluidsynth", "-ni", soundfont_path, midi_path, "-F", wav_path, "-r", "44100"
    ]
    subprocess.run(cmd, check=True)

def generate_synthetic_data(midi_dir, output_dir, soundfont_path):
    os.makedirs(output_dir, exist_ok=True)
    for i, midi_file in enumerate(os.listdir(midi_dir)):
        if not midi_file.endswith(".mid"):
            continue
        midi_path = os.path.join(midi_dir, midi_file)
        wav_path = os.path.join(output_dir, f"temp_{i}.wav")
        cqt_output = os.path.join(output_dir, f"{i}_{midi_file[:-4]}.npy")

        synthesize_midi_to_wav(midi_path, wav_path, soundfont_path)
        extract_cqt(wav_path, cqt_output)
        os.remove(wav_path)

if __name__ == "__main__":
    generate_synthetic_data("data/raw", "data/processed", "FluidR3_GM.sf2")