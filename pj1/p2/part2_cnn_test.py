import glob
import os
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from part2_cnn import SimpleCNN


class LabeledBMPDataset(Dataset):
    def __init__(self, test_dir: str, label_map: Dict[int, str]):
        if not os.path.isdir(test_dir):
            raise FileNotFoundError(f"测试目录不存在: {test_dir}")

        self.label_map = label_map
        self.name_to_idx = {name: idx for idx, name in label_map.items()}

        samples: List[Tuple[str, int]] = []
        for class_name, class_idx in self.name_to_idx.items():
            class_dir = os.path.join(test_dir, class_name)
            if not os.path.isdir(class_dir):
                continue
            paths = sorted(glob.glob(os.path.join(class_dir, "*.bmp")))
            samples.extend((p, class_idx) for p in paths)

        if not samples:
            raise ValueError(
                "未读取到任何带标签样本。请确认测试集结构为 test_dir/类别名/*.bmp，且类别名与训练时一致。"
            )

        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        path, label = self.samples[index]
        arr = np.array(Image.open(path).convert("L"), dtype=np.float32) / 255.0
        x = torch.from_numpy(arr).unsqueeze(0)
        y = torch.tensor(label, dtype=torch.long)
        return x, y


@torch.no_grad()
def evaluate_labeled(model: nn.Module, loader: DataLoader, device: torch.device):
    model.eval()
    criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    total = 0
    correct = 0

    all_y: List[int] = []
    all_pred: List[int] = []

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        logits = model(x)
        loss = criterion(logits, y)

        pred = torch.argmax(logits, dim=1)

        total_loss += float(loss.item()) * y.size(0)
        correct += int((pred == y).sum().item())
        total += y.size(0)

        all_y.extend(y.cpu().numpy().tolist())
        all_pred.extend(pred.cpu().numpy().tolist())

    avg_loss = total_loss / max(total, 1)
    acc = correct / max(total, 1)
    return avg_loss, acc, np.array(all_y), np.array(all_pred)

def main():
    # 直接在这里修改测试配置
    model_path = "part2_cnn.pt"
    test_dir = "../test_data"
    batch_size = 128
    device_name = "auto"  # 可选: "auto" / "cpu" / "cuda"

    if device_name == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_name)

    ckpt = torch.load(model_path, map_location=device)
    label_map = ckpt["label_map"]
    num_classes = ckpt["num_classes"]

    model = SimpleCNN(num_classes=num_classes).to(device)
    model.load_state_dict(ckpt["model_state_dict"])

    print(f"Device: {device}")
    print(f"Model : {model_path}")
    print(f"Test  : {test_dir}")
    dataset = LabeledBMPDataset(test_dir, label_map)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    test_loss, test_acc, y_true, y_pred = evaluate_labeled(model, loader, device)

    print(f"Test Loss: {test_loss:.6f}")
    print(f"Test Acc : {test_acc:.4f}")

    print("Per-class accuracy:")
    for c in range(num_classes):
        mask = y_true == c
        if np.any(mask):
            cls_acc = float(np.mean(y_pred[mask] == y_true[mask]))
            print(f"  Class {label_map[c]:>2s}: {cls_acc:.4f} ({int(mask.sum())} samples)")


if __name__ == "__main__":
    main()
