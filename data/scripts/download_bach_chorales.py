import os
from music21 import corpus

def download_bach_chorales(n=10, raw_dir="data/raw", scores_dir="data/scores"):
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(scores_dir, exist_ok=True)

    chorales = corpus.chorales.Iterator()
    for i, score in enumerate(chorales):
        if i >= n:
            break
        midi_path = os.path.join(raw_dir, f"bach_chorale_{i}.mid")
        xml_path = os.path.join(scores_dir, f"bach_chorale_{i}.musicxml")
        print(f"Saving: {midi_path}, {xml_path}")
        score.write('midi', fp=midi_path)
        score.write('musicxml', fp=xml_path)

if __name__ == "__main__":
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    raw_dir = os.path.join(base_dir, "data", "raw")
    scores_dir = os.path.join(base_dir, "data", "scores")
    download_bach_chorales(n=10, raw_dir=raw_dir, scores_dir=scores_dir)