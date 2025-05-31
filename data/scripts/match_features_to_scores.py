import os

def match_cqt_to_score(cqt_dir, scores_dir):
    """
    Matches CQT feature files to corresponding MusicXML score files
    based on naming convention: <index>_<basename>.npy <-> <basename>.musicxml

    Parameters:
        cqt_dir (str): Directory containing CQT .npy files.
        scores_dir (str): Directory containing MusicXML files.

    Returns:
        List of tuples: (path_to_cqt_file, path_to_score_file)
    """
    if not os.path.isdir(cqt_dir):
        raise FileNotFoundError(f"CQT directory not found: {cqt_dir}")
    if not os.path.isdir(scores_dir):
        raise FileNotFoundError(f"Scores directory not found: {scores_dir}")

    matches = []
    cqt_files = [f for f in os.listdir(cqt_dir) if f.endswith(".npy")]
    score_files = set(f for f in os.listdir(scores_dir) if f.endswith(".musicxml"))

    for cqt_file in cqt_files:
        # Strip the index and extension to get the base filename
        base = "_".join(cqt_file.split("_")[1:]).replace(".npy", "")
        possible_score = f"{base}.musicxml"

        if possible_score in score_files:
            match = (
                os.path.join(cqt_dir, cqt_file),
                os.path.join(scores_dir, possible_score)
            )
            matches.append(match)

    print(f"Found {len(matches)} matching pairs.")
    return matches

def get_score_match_map():
    """
    Builds a dictionary mapping piece ID (from processed CQT files) to matching MusicXML file path.
    
    Returns:
        Dict[str, str]: { 'piece_id': '/path/to/score.musicxml' }
    """
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    cqt_dir = os.path.join(base_dir, "data", "processed")
    scores_dir = os.path.join(base_dir, "data", "scores")
    
    matches = match_cqt_to_score(cqt_dir, scores_dir)
    
    match_map = {}
    for cqt_path, xml_path in matches:
        cqt_filename = os.path.basename(cqt_path)
        piece_id = os.path.splitext(cqt_filename)[0]  # e.g., "12_chorale03"
        match_map[piece_id] = xml_path
    
    return match_map

if __name__ == "__main__":
    # Set up project-relative paths
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    cqt_dir = os.path.join(base_dir, "data", "processed")
    scores_dir = os.path.join(base_dir, "data", "scores")

    # Run matching
    print(f"Matching CQT files in '{cqt_dir}' with scores in '{scores_dir}'...")
    matches = match_cqt_to_score(cqt_dir, scores_dir)

    # Optionally print a preview of matches
    for cqt, score in matches[:5]:
        print(f"MATCH: {os.path.basename(cqt)} ↔ {os.path.basename(score)}")
