from music21 import converter
import os

def convert_musicxml_to_midi(xml_path, midi_output_path):
    """
    Converts a MusicXML file to a MIDI file.

    Parameters:
        xml_path (str): Path to the input MusicXML file.
        midi_output_path (str): Path to save the output MIDI file.
    """
    # Parse the MusicXML file
    score = converter.parse(xml_path)
    
    # Write the parsed score as a MIDI file to the specified path
    score.write('midi', fp=midi_output_path)

if __name__ == '__main__':
    # Example usage
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    scores_dir = os.path.join(base_dir, "data", "scores")
    example_xml = os.path.join(scores_dir,"bach_chorale_0.musicxml")
    example_midi = "output_example.mid"

    # Check if the example MusicXML file exists
    if os.path.exists(example_xml):
        print(f"Converting {example_xml} to {example_midi}...")
        convert_musicxml_to_midi(example_xml, example_midi)
        print("Conversion complete.")
    else:
        print(f"Example file '{example_xml}' not found. Please provide a valid MusicXML file.")
