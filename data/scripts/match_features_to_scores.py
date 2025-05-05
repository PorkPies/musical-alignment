import os

def match_cqt_to_score(cqt_dir, scores_dir):
    matches = []
    cqt_files = [f for f in os.listdir(cqt_dir) if f.endswith(".npy")]
    score_files = [f for f in os.listdir(scores_dir) if f.endswith(".musicxml")]

    for cqt_file in cqt_files:
        base = "_".join(cqt_file.split("_")[1:]).replace(".npy", "")
        possible_score = f"{base}.musicxml"
        if possible_score in score_files:
            match = (os.path.join(cqt_dir, cqt_file), os.path.join(scores_dir, possible_score))
            matches.append(match)

    print(f"Found {len(matches)} matching pairs.")
    return matches

if __name__ == "__main__":
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    cqt_dir = os.path.join(base_dir, "data", "processed")
    scores_dir = os.path.join(base_dir, "data", "scores")
    match_cqt_to_score(cqt_dir, scores_dir)