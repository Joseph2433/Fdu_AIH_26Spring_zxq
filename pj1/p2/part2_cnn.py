import glob
import os
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


class BMPDataset(Dataset):
    def __init__(self, train_dir: str ):
        if not os.path.isdir(train_dir):
            raise FileNotFoundError("train 目录不存在。请先执行: tar -xf train_data.rar")

        class_dirs = [d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))]
        class_dirs = sorted(class_dirs, key=lambda s: int(s))
        self.label_map: Dict[int, str] = {i: cls for i, cls in enumerate(class_dirs)}

        samples: List[Tuple[str, int]] = []
        for idx, cls in enumerate(class_dirs):
            paths = sorted(glob.glob(os.path.join(train_dir, cls, "*.bmp")))
            samples.extend((p, idx) for p in paths)

        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        path, label = self.samples[index]
        arr = np.array(Image.open(path).convert("L"), dtype=np.float32) / 255.0
        x = torch.from_numpy(arr).unsqueeze(0)  
        y = torch.tensor(label, dtype=torch.long)
        return x, y


class SimpleCNN(nn.Module):
    def __init__(self, num_classes: int, dropout_features: float = 0.2, dropout_classifier: float = 0.5):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(p=dropout_features),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(p=dropout_features),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_classifier),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


def stratified_indices(dataset: BMPDataset, val_ratio: float, seed: int ):
    rng = np.random.default_rng(seed)
    labels = np.array([label for _, label in dataset.samples], dtype=np.int64)
    train_idx = []
    val_idx = []
    for c in np.unique(labels):
        idx = np.where(labels == c)[0]
        rng.shuffle(idx)
        n_val = max(1, int(len(idx) * val_ratio))
        val_idx.extend(idx[:n_val].tolist())
        train_idx.extend(idx[n_val:].tolist())
    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    return train_idx, val_idx


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device):
    model.eval()
    total_loss = 0.0
    total = 0
    correct = 0
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        logits = model(x)
        loss = criterion(logits, y)
        total_loss += float(loss.item()) * y.size(0)
        pred = torch.argmax(logits, dim=1)
        correct += int((pred == y).sum().item())
        total += y.size(0)
    return total_loss / max(total, 1), correct / max(total, 1)


def main():
    print("Part2: CNN for 12-class Chinese character classification", flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)

    dataset = BMPDataset("../train")
    val_ratio=0.2
    seed=42
    train_idx, val_idx = stratified_indices(dataset, val_ratio, seed)
    train_set = Subset(dataset, train_idx)
    val_set = Subset(dataset, val_idx)

    train_loader = DataLoader(train_set, batch_size=64, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=128, shuffle=False, num_workers=0)

    num_classes = len(dataset.label_map)
    model = SimpleCNN(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    lr=1e-3
    optimizer = torch.optim.Adam(model.parameters(), lr)

    epochs = 10
    train_losses = []
    val_accs = []

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        total = 0

        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            running_loss += float(loss.item()) * y.size(0)
            total += y.size(0)

        train_loss = running_loss / max(total, 1)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        train_losses.append(train_loss)
        val_accs.append(val_acc)

        print(
            f"Epoch {epoch:2d}/{epochs} | Train Loss: {train_loss:.6f} "
            f"| Val Loss: {val_loss:.6f} | Val Acc: {val_acc:.4f}"
            ,
            flush=True,
        )

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "label_map": dataset.label_map,
            "num_classes": num_classes,
        },
        "part2_cnn.pt",
    )
    print("Saved model: part2_cnn.pt", flush=True)

    if HAS_MATPLOTLIB:
        plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1)
        plt.plot(train_losses)
        plt.title("Train Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.grid(alpha=0.3)

        plt.subplot(1, 2, 2)
        plt.plot(val_accs)
        plt.title("Validation Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.ylim(0, 1)
        plt.grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig("part2_cnn_result.png", dpi=120)
        print("Saved figure: part2_cnn_result.png", flush=True)

    return model


if __name__ == "__main__":
    main()
