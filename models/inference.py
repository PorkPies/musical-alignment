"""
Musical alignment inference pipeline.

Offline mode:  python models/inference.py --wav data/test/temp_0.wav
Live mic mode: python models/inference.py --mic
Pipe mode:     python models/inference.py --pipe /tmp/audio_pipe
"""
import os
import sys
import argparse
import numpy as np
import torch
import librosa
from collections import deque

base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, base_dir)

from models.baseline_model import BaselineCNN

# CQT parameters — must match training (split_snippets.py)
SR = 22050
HOP_SIZE = 512
SNIPPET_LEN = 128

# Inference parameters
BUFFER_SECONDS = 3
STRIDE_FRAMES = 64
SMOOTH_WINDOW = 5


def load_model(checkpoint_path, device):
    """Load a saved checkpoint and return (model, bar_to_class)."""
    ckpt = torch.load(checkpoint_path, map_location=device)
    model = BaselineCNN(ckpt["input_shape"], ckpt["num_classes"])
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    return model, ckpt["bar_to_class"]


def extract_cqt_from_audio(y, sr=SR):
    """Return a CQT array (freq_bins × time_frames) in dB."""
    return librosa.amplitude_to_db(
        librosa.cqt(y, sr=sr, hop_length=HOP_SIZE), ref=np.max
    )


def predict_snippet(model, cqt_snippet, bar_to_class, device):
    """
    Run a forward pass on a single CQT snippet.

    Parameters:
        cqt_snippet: np.ndarray of shape (freq_bins, SNIPPET_LEN)

    Returns:
        (bar_number, confidence)
    """
    class_to_bar = {v: k for k, v in bar_to_class.items()}
    x = torch.tensor(cqt_snippet, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)
        conf, pred_class = probs.max(dim=1)
    bar = class_to_bar[pred_class.item()]
    return bar, conf.item()


def _majority(predictions):
    return max(set(predictions), key=list(predictions).count)


def run_offline(wav_path, checkpoint_path, callback=None):
    """
    Slide over a WAV file and print predicted bar numbers.

    Parameters:
        callback: optional callable(bar_number, confidence, time_sec)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, bar_to_class = load_model(checkpoint_path, device)

    y, _ = librosa.load(wav_path, sr=SR)
    cqt = extract_cqt_from_audio(y)
    n_frames = cqt.shape[1]

    predictions = deque(maxlen=SMOOTH_WINDOW)
    for start in range(0, n_frames - SNIPPET_LEN + 1, STRIDE_FRAMES):
        snippet = cqt[:, start:start + SNIPPET_LEN]
        bar, conf = predict_snippet(model, snippet, bar_to_class, device)
        predictions.append(bar)
        smoothed = _majority(predictions)
        time_sec = (start * HOP_SIZE) / SR
        print(f"t={time_sec:.2f}s  bar={smoothed}  conf={conf:.3f}")
        if callback:
            callback(smoothed, conf, time_sec)


def run_live(checkpoint_path, callback=None):
    """
    Stream audio from the default microphone and predict bar numbers in real time.

    Parameters:
        callback: optional callable(bar_number, confidence, time_sec=None)
    """
    import sounddevice as sd

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, bar_to_class = load_model(checkpoint_path, device)

    buffer_samples = int(BUFFER_SECONDS * SR)
    audio_buffer = np.zeros(buffer_samples, dtype=np.float32)
    stride_samples = STRIDE_FRAMES * HOP_SIZE
    snippet_samples = SNIPPET_LEN * HOP_SIZE
    frames_since_last = [0]
    predictions = deque(maxlen=SMOOTH_WINDOW)

    def audio_callback(indata, frames, time_info, status):
        chunk = indata[:, 0].astype(np.float32)
        audio_buffer[:-len(chunk)] = audio_buffer[len(chunk):]
        audio_buffer[-len(chunk):] = chunk
        frames_since_last[0] += len(chunk)
        if frames_since_last[0] >= stride_samples:
            frames_since_last[0] = 0
            segment = audio_buffer[-snippet_samples:]
            cqt = extract_cqt_from_audio(segment)
            if cqt.shape[1] >= SNIPPET_LEN:
                snippet = cqt[:, :SNIPPET_LEN]
                bar, conf = predict_snippet(model, snippet, bar_to_class, device)
                predictions.append(bar)
                smoothed = _majority(predictions)
                print(f"bar={smoothed}  conf={conf:.3f}")
                if callback:
                    callback(smoothed, conf, None)

    print("Listening on microphone... Press Ctrl+C to stop.")
    with sd.InputStream(samplerate=SR, channels=1, callback=audio_callback, blocksize=1024):
        try:
            while True:
                sd.sleep(100)
        except KeyboardInterrupt:
            pass


def run_from_pipe(pipe_path, checkpoint_path, callback=None):
    """
    Read raw 16-bit mono PCM at SR from a named pipe (e.g. FluidSynth stdout).

    Parameters:
        callback: optional callable(bar_number, confidence, time_sec=None)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, bar_to_class = load_model(checkpoint_path, device)

    buffer_samples = int(BUFFER_SECONDS * SR)
    audio_buffer = np.zeros(buffer_samples, dtype=np.float32)
    stride_samples = STRIDE_FRAMES * HOP_SIZE
    snippet_samples = SNIPPET_LEN * HOP_SIZE
    frames_since_last = 0
    predictions = deque(maxlen=SMOOTH_WINDOW)

    print(f"Reading audio from pipe: {pipe_path}")
    with open(pipe_path, "rb") as f:
        while True:
            raw = f.read(2048)  # 1024 16-bit samples
            if not raw:
                break
            chunk = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
            audio_buffer[:-len(chunk)] = audio_buffer[len(chunk):]
            audio_buffer[-len(chunk):] = chunk
            frames_since_last += len(chunk)

            if frames_since_last >= stride_samples:
                frames_since_last = 0
                segment = audio_buffer[-snippet_samples:]
                cqt = extract_cqt_from_audio(segment)
                if cqt.shape[1] >= SNIPPET_LEN:
                    snippet = cqt[:, :SNIPPET_LEN]
                    bar, conf = predict_snippet(model, snippet, bar_to_class, device)
                    predictions.append(bar)
                    smoothed = _majority(predictions)
                    print(f"bar={smoothed}  conf={conf:.3f}")
                    if callback:
                        callback(smoothed, conf, None)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Musical alignment inference")
    parser.add_argument(
        "--checkpoint",
        default=os.path.join(base_dir, "models", "checkpoint.pt"),
        help="Path to saved model checkpoint (.pt)",
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--wav", help="Path to WAV file for offline inference")
    group.add_argument("--mic", action="store_true", help="Use microphone for live inference")
    group.add_argument("--pipe", help="Path to named pipe with raw 16-bit mono PCM audio")
    args = parser.parse_args()

    if args.wav:
        run_offline(args.wav, args.checkpoint)
    elif args.mic:
        run_live(args.checkpoint)
    elif args.pipe:
        run_from_pipe(args.pipe, args.checkpoint)
