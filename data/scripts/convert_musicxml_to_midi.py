from music21 import converter
import os

def convert_musicxml_to_midi(xml_path, midi_output_path):
    score = converter.parse(xml_path)
    score.write('midi', fp=midi_output_path)