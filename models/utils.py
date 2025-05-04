import torch
from torch.utils.data import Dataset
import numpy as np
import os

class CQTBarDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.files = [f for f in os.listdir(data_dir) if f.endswith(".npy")]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        cqt = np.load(os.path.join(self.data_dir, self.files[idx]))
        label = int(self.files[idx].split('_')[0])  # assumes filename like '12_piece.npy'
        return torch.tensor(cqt).unsqueeze(0).float(), label