import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

# Set up project base
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Import model and dataset
from models.baseline_model import BaselineCNN
from models.utils import CQTBarWithScoreDataset

# Config
SNIPPETS_DIR = os.path.join(base_dir, "data", "snippets")
BATCH_SIZE = 8
EPOCHS = 10
LEARNING_RATE = 1e-3
VAL_SPLIT = 0.2

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    full_dataset = CQTBarWithScoreDataset(SNIPPETS_DIR)

    # Train/Val split
    val_size = int(len(full_dataset) * VAL_SPLIT)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    sample, _, _ = full_dataset[0]
    num_classes = len(full_dataset.bar_to_class)

    model = BaselineCNN(sample.shape, num_classes).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0
        for inputs, labels, _ in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for inputs, labels, _ in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
        model.train()

        print(f"Epoch {epoch+1}/{EPOCHS}, Train Loss: {total_loss:.4f}, Val Loss: {val_loss:.4f}")

if __name__ == "__main__":
    train()
