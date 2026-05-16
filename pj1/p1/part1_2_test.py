import glob
import os
import pickle
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image

from part1_2_classifier import NeuralNetwork, evaluate


class LabeledNumpyDataset:
    def __init__(self, test_dir: str, label_map: Dict[int, str]):
        if not os.path.isdir(test_dir):
            raise FileNotFoundError(f"测试目录不存在: {test_dir}")

        self.label_map = label_map
        self.name_to_idx = {name: idx for idx, name in label_map.items()}

        self.paths: List[str] = []
        self.labels: List[int] = []

        for class_name, class_idx in self.name_to_idx.items():
            class_dir = os.path.join(test_dir, class_name)
            if not os.path.isdir(class_dir):
                continue

            bmp_files = sorted(glob.glob(os.path.join(class_dir, "*.bmp")))
            self.paths.extend(bmp_files)
            self.labels.extend([class_idx] * len(bmp_files))

        if not self.paths:
            raise ValueError(
                "未读取到任何带标签样本。请确认测试集结构为 test_dir/类别名/*.bmp，且类别名与训练时一致。"
            )

    def to_numpy(self) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        images = []
        for p in self.paths:
            arr = np.array(Image.open(p).convert("L"), dtype=np.float32)
            images.append(arr.reshape(-1) / 255.0)

        X = np.stack(images).astype(np.float32)
        y = np.array(self.labels, dtype=np.int64)
        return X, y, self.paths


def load_model(model_path: str) -> Tuple[NeuralNetwork, Dict[int, str], int]:
    with open(model_path, "rb") as f:
        ckpt = pickle.load(f)

    model = NeuralNetwork(
        layer_sizes=ckpt["layer_sizes"],
        learning_rate=ckpt["learning_rate"],
        dropout_rate=ckpt["dropout_rate"],
    )
    model.weights = ckpt["weights"]
    model.biases = ckpt["biases"]

    label_map = ckpt["label_map"]
    num_classes = len(label_map)
    return model, label_map, num_classes


def main():
    # 直接在这里修改测试配置
    model_path = "part1_2_model.pkl"
    test_dir = "../test_data"

    model, label_map, num_classes = load_model(model_path)
    print(f"Model: {model_path}")
    print(f"Test : {test_dir}")

    dataset = LabeledNumpyDataset(test_dir, label_map)
    X_test, y_test, _ = dataset.to_numpy()

    test_acc, test_loss = evaluate(model, X_test, y_test, num_classes)
    print(f"Test Acc : {test_acc:.4f}")
    print(f"Test Loss: {test_loss:.6f}")

    y_pred = model.predict(X_test)
    print("Per-class accuracy:")
    for c in range(num_classes):
        mask = y_test == c
        if np.any(mask):
            cls_acc = float(np.mean(y_pred[mask] == y_test[mask]))
            print(f"  Class {label_map[c]:>2s}: {cls_acc:.4f} ({int(mask.sum())} samples)")


if __name__ == "__main__":
    main()
