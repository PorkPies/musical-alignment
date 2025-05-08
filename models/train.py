import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.baseline_model import BaselineCNN
from models.utils import CQTBarDataset

# Config
DATA_DIR = "musical-alignment/data/processed/"
BATCH_SIZE = 8
EPOCHS = 10
LEARNING_RATE = 1e-3
NUM_CLASSES = 100  # adjust to your dataset


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = CQTBarDataset(DATA_DIR)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    sample, _ = dataset[0]
    model = BaselineCNN(sample.shape, NUM_CLASSES).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {total_loss:.4f}")

if __name__ == "__main__":
    train()