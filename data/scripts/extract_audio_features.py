import librosa
import numpy as np
import os

def extract_cqt(audio_path, output_path, sr=22050):
    y, _ = librosa.load(audio_path, sr=sr)
    cqt = librosa.amplitude_to_db(librosa.cqt(y, sr=sr), ref=np.max)
    np.save(output_path, cqt)
