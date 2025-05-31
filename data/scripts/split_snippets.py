import os
import numpy as np
from extract_bar_times import extract_bar_times
from match_features_to_scores import get_score_match_map

SNIPPET_LEN = 128
HOP_LEN = 64
SR = 22050
HOP_SIZE = 512
FPS = SR / HOP_SIZE  # e.g., ~43.07 frames/sec for typical CQT

def find_closest_bar(time, bar_times):
    return min(range(len(bar_times)), key=lambda i: abs(bar_times[i] - time))

def split_and_label(base_name, npy_path, xml_path, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    cqt = np.load(npy_path)
    bar_times = extract_bar_times(xml_path)

    total_frames = cqt.shape[1]
    snippet_count = 0

    for start in range(0, total_frames - SNIPPET_LEN + 1, HOP_LEN):
        end = start + SNIPPET_LEN
        snippet = cqt[:, start:end]
        center_frame = start + SNIPPET_LEN // 2
        time_sec = center_frame / FPS
        bar = find_closest_bar(time_sec, bar_times)

        filename = f"{base_name}_bar_{bar:03d}_snip_{snippet_count:03d}.npy"
        np.save(os.path.join(out_dir, filename), snippet)
        snippet_count += 1


if __name__ == "__main__":
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    data_dir = os.path.join(base_dir, "data", "processed")
    out_dir = os.path.join(base_dir, "data", "snippets")

    score_map = get_score_match_map()  # Maps {piece_id: path_to_xml}

    for fname in os.listdir(data_dir):
        if not fname.endswith(".npy"):
            continue
        piece_id = os.path.splitext(fname)[0]           # e.g., "12_bach_chorale_3"
        base_name = "_".join(fname.split("_")[1:]).replace(".npy", "")  # "bach_chorale_3"
        npy_path = os.path.join(data_dir, fname)
        if piece_id not in score_map:
            print(f"Missing score mapping for {piece_id}, skipping.")
            continue
        xml_path = score_map[piece_id]
        split_and_label(base_name, npy_path, xml_path, out_dir)
