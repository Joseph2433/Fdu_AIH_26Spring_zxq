import glob
import os
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image, ImageChops

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


class BMPDataset(Dataset):
    def __init__(self, train_dir: str, augment: bool = False):
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
        self.augment = augment

    def _random_augment(self, img: Image.Image) -> Image.Image:
        width, height = img.size

        # 1) Small random rotation.
        angle = float(np.random.uniform(-10.0, 10.0))
        img = img.rotate(angle, resample=Image.BILINEAR, fillcolor=0)

        # 2) Small random translation (up to 10% in x/y).
        shift_x = int(round(np.random.uniform(-0.1, 0.1) * width))
        shift_y = int(round(np.random.uniform(-0.1, 0.1) * height))
        if shift_x != 0 or shift_y != 0:
            img = ImageChops.offset(img, shift_x, shift_y)
            if shift_x > 0:
                img.paste(0, (0, 0, shift_x, height))
            elif shift_x < 0:
                img.paste(0, (width + shift_x, 0, width, height))
            if shift_y > 0:
                img.paste(0, (0, 0, width, shift_y))
            elif shift_y < 0:
                img.paste(0, (0, height + shift_y, width, height))

        # 3) Mild random scaling (0.9x to 1.1x).
        scale = float(np.random.uniform(0.9, 1.1))
        new_w = max(1, int(round(width * scale)))
        new_h = max(1, int(round(height * scale)))
        resized = img.resize((new_w, new_h), resample=Image.BILINEAR)
        if scale <= 1.0:
            canvas = Image.new("L", (width, height), color=0)
            left = (width - new_w) // 2
            top = (height - new_h) // 2
            canvas.paste(resized, (left, top))
            img = canvas
        else:
            left = (new_w - width) // 2
            top = (new_h - height) // 2
            img = resized.crop((left, top, left + width, top + height))

        return img

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        path, label = self.samples[index]
        img = Image.open(path).convert("L")
        if self.augment:
            img = self._random_augment(img)

        arr = np.array(img, dtype=np.float32) / 255.0
        x = torch.from_numpy(arr).unsqueeze(0)  
        y = torch.tensor(label, dtype=torch.long)
        return x, y


class SimpleCNN(nn.Module):
    def __init__(self, num_classes: int, dropout_features: float = 0.2, dropout_classifier: float = 0.5):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(p=dropout_features),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(p=dropout_features),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
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
def tta_affine(x: torch.Tensor) -> torch.Tensor:
    n = x.size(0)
    device = x.device
    dtype = x.dtype

    # TTA perturbation range: mild rotation/translation/scaling.
    angles = torch.empty(n, device=device, dtype=dtype).uniform_(-8.0, 8.0) * (np.pi / 180.0)
    scales = torch.empty(n, device=device, dtype=dtype).uniform_(0.95, 1.05)
    tx = torch.empty(n, device=device, dtype=dtype).uniform_(-0.16, 0.16)
    ty = torch.empty(n, device=device, dtype=dtype).uniform_(-0.16, 0.16)

    cos_a = torch.cos(angles) * scales
    sin_a = torch.sin(angles) * scales

    theta = torch.zeros((n, 2, 3), device=device, dtype=dtype)
    theta[:, 0, 0] = cos_a
    theta[:, 0, 1] = -sin_a
    theta[:, 1, 0] = sin_a
    theta[:, 1, 1] = cos_a
    theta[:, 0, 2] = tx
    theta[:, 1, 2] = ty

    grid = F.affine_grid(theta, x.size(), align_corners=False)
    return F.grid_sample(x, grid, mode="bilinear", padding_mode="zeros", align_corners=False)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    tta_times: int = 1,
):
    model.eval()
    tta_times = max(1, int(tta_times))
    total_loss = 0.0
    total = 0
    correct = 0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        probs_sum = None
        loss_sum = 0.0

        for t in range(tta_times):
            x_in = x if t == 0 else tta_affine(x)
            logits = model(x_in)
            loss = criterion(logits, y)
            loss_sum += float(loss.item()) * y.size(0)

            probs = torch.softmax(logits, dim=1)
            probs_sum = probs if probs_sum is None else (probs_sum + probs)

        avg_probs = probs_sum / tta_times
        total_loss += loss_sum / tta_times
        pred = torch.argmax(avg_probs, dim=1)
        correct += int((pred == y).sum().item())
        total += y.size(0)

    return total_loss / max(total, 1), correct / max(total, 1)


def main():
    print("Part2: CNN for 12-class Chinese character classification", flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)

    dataset = BMPDataset("../train", augment=False)
    val_ratio=0.2
    seed=42
    train_idx, val_idx = stratified_indices(dataset, val_ratio, seed)

    train_dataset = BMPDataset("../train", augment=True)
    val_dataset = BMPDataset("../train", augment=False)
    train_set = Subset(train_dataset, train_idx)
    val_set = Subset(val_dataset, val_idx)

    train_loader = DataLoader(train_set, batch_size=64, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=128, shuffle=False, num_workers=0)

    num_classes = len(dataset.label_map)
    model = SimpleCNN(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    lr=1e-3
    optimizer = torch.optim.Adam(model.parameters(), lr)

    use_tta_val = True
    tta_times = 5

    epochs = 50
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
        val_loss, val_acc = evaluate(
            model,
            val_loader,
            criterion,
            device,
            tta_times=tta_times if use_tta_val else 1,
        )
        train_losses.append(train_loss)
        val_accs.append(val_acc)

        tta_tag = f" (TTAx{tta_times})" if use_tta_val else ""
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
