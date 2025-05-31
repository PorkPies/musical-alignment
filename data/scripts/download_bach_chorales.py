import os
from music21 import corpus

def download_bach_chorales(n=10, raw_dir="data/raw", scores_dir="data/scores"):
    """
    Downloads `n` Bach chorales from the music21 corpus and saves them as both MIDI and MusicXML files.

    Parameters:
        n (int): Number of chorales to download.
        raw_dir (str): Directory to save MIDI files.
        scores_dir (str): Directory to save MusicXML files.
    """
    # Ensure the output directories exist
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(scores_dir, exist_ok=True)

    # Access chorales from the music21 corpus
    chorales = corpus.chorales.Iterator()
    for i, score in enumerate(chorales):
        if i >= n:
            break
        # Define output paths for MIDI and MusicXML
        midi_path = os.path.join(raw_dir, f"bach_chorale_{i}.mid")
        xml_path = os.path.join(scores_dir, f"bach_chorale_{i}.musicxml")
        print(f"Saving: {midi_path}, {xml_path}")
        
        # Write the chorale to both formats
        score.write('midi', fp=midi_path)
        score.write('musicxml', fp=xml_path)

if __name__ == "__main__":
    # Define the base directory two levels up from this script
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))

    # Set up full paths for raw and scores directories
    raw_dir = os.path.join(base_dir, "data", "raw")
    scores_dir = os.path.join(base_dir, "data", "scores")

    # Download and save 10 chorales
    print(f"Downloading Bach chorales to:\n  MIDI: {raw_dir}\n  MusicXML: {scores_dir}")
    download_bach_chorales(n=10, raw_dir=raw_dir, scores_dir=scores_dir)
    print("Download complete.")
