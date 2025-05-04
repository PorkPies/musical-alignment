# Music Alignment Project

This project aligns live or recorded audio to symbolic sheet music for real-time score following and page turning.

## Features
- Audio feature extraction using Constant-Q Transform
- Sheet music parsing and MIDI conversion
- CNN-based bar classification model
- Ready for Azure training and deployment

## Usage
- Place raw audio and MusicXML files in `data/raw`
- Use scripts in `data/scripts/` to preprocess them
- Train model with `models/train.py`