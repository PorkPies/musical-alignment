import torch
from torch.utils.data import Dataset
import numpy as np
import os

import sys
from torch.utils.data import Dataset

from data.scripts.match_features_to_scores import get_score_match_map

class CQTBarWithScoreDataset(Dataset):
    def __init__(self, snippets_dir):
        self.snippets_dir = snippets_dir
        self.files = [f for f in os.listdir(snippets_dir) if f.endswith(".npy")]
        self.score_map = get_score_match_map()  # piece_id → xml path
        self.labels = []
        for f in self.files:
            bar = int(f.split("_bar_")[1].split("_")[0])
            self.labels.append(bar)

        # Map raw bar numbers to 0..N-1 class indices
        unique_bars = sorted(set(self.labels))
        self.bar_to_class = {bar: i for i, bar in enumerate(unique_bars)}


    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        filename = self.files[idx]
        file_path = os.path.join(self.snippets_dir, filename)
        cqt = np.load(file_path)
        cqt_tensor = torch.tensor(cqt).unsqueeze(0).float()

        # Parse metadata
        parts = filename.replace(".npy", "").split("_")
        base_name = "_".join(parts[:-4]) if len(parts) > 4 else parts[0]  # robust fallback
        bar_number = int(parts[-3])
        bar_label = self.bar_to_class[bar_number]

        # Match XML file
        # Find full matching piece_id from score_map (might be '12_bach_chorale_9' → 'bach_chorale_9')
        matched = [v for k, v in self.score_map.items() if base_name in k or base_name in os.path.basename(v)]
        if matched:
            xml_path = matched[0]
        else:
            raise FileNotFoundError(f"No matching XML for snippet {filename}")

        return cqt_tensor, bar_label, xml_path

if __name__ == "__main__":
    import os

    # Set base_dir relative to this script
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    snippets_dir = os.path.join(base_dir, "data", "snippets")

    dataset = CQTBarWithScoreDataset(snippets_dir)

    print(f"Loaded dataset with {len(dataset)} snippets.")
    print("Previewing the first 3 samples:")

    for i in range(min(3, len(dataset))):
        snippet, bar, xml_path = dataset[i]
        print(f"Sample {i}: snippet shape = {snippet.shape}, bar = {bar}, xml = {os.path.basename(xml_path)}")


    
