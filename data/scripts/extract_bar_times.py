from music21 import converter, tempo, meter
import os

def extract_bar_times(xml_path):
    score = converter.parse(xml_path)
    bars = []
    current_time = 0.0

    metronome = score.metronomeMarkBoundaries()
    ts = score.recurse().getElementsByClass(meter.TimeSignature).first()
    qpm = 120  # default tempo
    if metronome:
        _, _, mark = metronome[0]
        qpm = mark.number

    seconds_per_quarter = 60.0 / qpm
    measure_offset_quarters = 0

    for measure in score.parts[0].getElementsByClass('Measure'):
        bars.append(measure_offset_quarters * seconds_per_quarter)
        measure_duration_quarters = measure.barDuration.quarterLength
        measure_offset_quarters += measure_duration_quarters

    return bars


if __name__ == "__main__":
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    xml_file = os.path.join(base_dir, "data", "scores", "bach_chorale_0.musicxml")
    bar_times = extract_bar_times(xml_file)
    print(bar_times)
