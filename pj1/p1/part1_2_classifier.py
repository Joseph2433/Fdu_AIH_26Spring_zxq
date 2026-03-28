import os
import glob
import pickle
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


class NeuralNetwork:
    def __init__(self, layer_sizes: List[int], learning_rate: float, dropout_rate: float ):
        self.layer_sizes = layer_sizes
        self.learning_rate = learning_rate
        self.dropout_rate = dropout_rate
        self.num_layers = len(layer_sizes)
        self.weights = []
        self.biases = []

        for i in range(len(layer_sizes) - 1):
            fan_in = layer_sizes[i]
            fan_out = layer_sizes[i + 1]
            limit = np.sqrt(6.0 / (fan_in + fan_out))
            self.weights.append(np.random.uniform(-limit, limit, (fan_in, fan_out)).astype(np.float32))
            self.biases.append(np.zeros((1, fan_out), dtype=np.float32))

    def relu(self, x: np.ndarray) -> np.ndarray:
        return np.maximum(0.0, x)

    def relu_derivative(self, x: np.ndarray) -> np.ndarray:
        return (x > 0.0).astype(np.float32)

    def softmax(self, x: np.ndarray) -> np.ndarray:
        x = x - np.max(x, axis=1, keepdims=True)
        exp_x = np.exp(x)
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    def forward(
        self,
        X: np.ndarray,
        training: bool = False,
    ) -> Tuple[np.ndarray, List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
        activations = [X]
        zs = []
        dropout_masks = []

        for i in range(self.num_layers - 1):
            z = np.dot(activations[-1], self.weights[i]) + self.biases[i]
            zs.append(z)
            if i < self.num_layers - 2:
                a = self.relu(z)
                if training and self.dropout_rate > 0.0:
                    keep_prob = 1.0 - self.dropout_rate
                    mask = (np.random.rand(*a.shape) < keep_prob).astype(np.float32) / keep_prob
                    a = a * mask
                else:
                    mask = np.ones_like(a, dtype=np.float32)
                dropout_masks.append(mask)
            else:
                a = self.softmax(z)
            activations.append(a)

        return activations[-1], zs, activations, dropout_masks

    def backward(
        self,
        y: np.ndarray,
        output: np.ndarray,
        zs: List[np.ndarray],
        activations: List[np.ndarray],
        dropout_masks: List[np.ndarray],
    ) -> None:
        m = y.shape[0]
        delta = (output - y) / m

        for i in range(self.num_layers - 2, -1, -1):
            w_curr = self.weights[i].copy()
            dW = np.dot(activations[i].T, delta)
            db = np.sum(delta, axis=0, keepdims=True)
            self.weights[i] -= self.learning_rate * dW
            self.biases[i] -= self.learning_rate * db
            if i > 0:
                # Use pre-update weights for stable backpropagation.
                delta = np.dot(delta, w_curr.T)
                delta *= self.relu_derivative(zs[i - 1])
                delta *= dropout_masks[i - 1]

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int ,
        batch_size: int ,
        verbose: bool = True,
    ) -> List[float]:
        losses = []

        for epoch in range(epochs):
            indices = np.random.permutation(X.shape[0])
            X_shuffled = X[indices]
            y_shuffled = y[indices]

            epoch_loss = 0.0
            batch_count = 0
            for i in range(0, X.shape[0], batch_size):
                X_batch = X_shuffled[i : i + batch_size]
                y_batch = y_shuffled[i : i + batch_size]
                output, zs, activations, dropout_masks = self.forward(X_batch, training=True)
                loss = -np.sum(y_batch * np.log(output + 1e-8)) / X_batch.shape[0]
                epoch_loss += float(loss)
                batch_count += 1
                self.backward(y_batch, output, zs, activations, dropout_masks)

            avg_loss = epoch_loss / max(batch_count, 1)
            losses.append(avg_loss)
            if verbose and ((epoch + 1) % 10 == 0 or epoch == 0):
                print(f"Epoch {epoch + 1:3d}/{epochs}, Loss: {avg_loss:.6f}")

        return losses

    def predict(self, X: np.ndarray) -> np.ndarray:
        output, _, _, _ = self.forward(X, training=False)
        return np.argmax(output, axis=1)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        output, _, _, _ = self.forward(X, training=False)
        return output


def one_hot_encode(y: np.ndarray, num_classes: int) -> np.ndarray:
    out = np.zeros((y.shape[0], num_classes), dtype=np.float32)
    out[np.arange(y.shape[0]), y] = 1.0
    return out


def load_images_from_train_dir(train_dir: str = "train") -> Tuple[np.ndarray, np.ndarray, Dict[int, str]]:
    if not os.path.isdir(train_dir):
        raise FileNotFoundError(
            "train 目录不存在。请先执行: tar -xf train_data.rar"
        )

    class_dirs = [d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))]
    class_dirs = sorted(class_dirs, key=lambda s: int(s))
    label_map = {idx: name for idx, name in enumerate(class_dirs)}

    images = []
    labels = []
    for idx, cls_name in enumerate(class_dirs):
        bmp_files = sorted(glob.glob(os.path.join(train_dir, cls_name, "*.bmp")))
        for path in bmp_files:
            arr = np.array(Image.open(path).convert("L"), dtype=np.float32)
            images.append(arr.reshape(-1) / 255.0)
            labels.append(idx)

    X = np.stack(images).astype(np.float32)
    y = np.array(labels, dtype=np.int64)
    return X, y, label_map


def stratified_split(
    X: np.ndarray,
    y: np.ndarray,
    val_ratio: float,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    train_idx = []
    val_idx = []

    for c in np.unique(y):
        idx = np.where(y == c)[0]
        rng.shuffle(idx)
        n_val = max(1, int(len(idx) * val_ratio))
        val_idx.extend(idx[:n_val].tolist())
        train_idx.extend(idx[n_val:].tolist())

    train_idx = np.array(train_idx, dtype=np.int64)
    val_idx = np.array(val_idx, dtype=np.int64)
    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    return X[train_idx], y[train_idx], X[val_idx], y[val_idx]


def evaluate(model: NeuralNetwork, X: np.ndarray, y: np.ndarray, num_classes: int) -> Tuple[float, float]:
    pred = model.predict(X)
    acc = float(np.mean(pred == y))
    prob = model.predict_proba(X)
    y_one_hot = one_hot_encode(y, num_classes)
    loss = float(-np.sum(y_one_hot * np.log(prob + 1e-8)) / X.shape[0])
    return acc, loss


def main():
    print("Part1-2: BP MLP for 12-class Chinese character classification")

    X_all, y_all, label_map = load_images_from_train_dir("../train")
    num_classes = len(label_map)
    print(f"Loaded samples: {X_all.shape[0]}, feature dim: {X_all.shape[1]}, classes: {num_classes}")

    val_ratio=0.2
    seed=42
    X_train, y_train, X_val, y_val = stratified_split(X_all, y_all, val_ratio, seed)
    y_train_one_hot = one_hot_encode(y_train, num_classes)
    print(f"Train/Val split: {X_train.shape[0]} / {X_val.shape[0]}")

    layer_sizes=[784, 256, 128, num_classes]
    learning_rate=0.01
    dropout_rate=0.3
    model = NeuralNetwork(layer_sizes, learning_rate, dropout_rate)
    epochs=100
    batch_size=64
    losses = model.train(X_train, y_train_one_hot, epochs, batch_size, verbose=True)

    train_acc, train_loss = evaluate(model, X_train, y_train, num_classes)
    val_acc, val_loss = evaluate(model, X_val, y_val, num_classes)

    print("\nMetrics")
    print(f"Train Acc: {train_acc:.4f}, Train Loss: {train_loss:.6f}")
    print(f"Val   Acc: {val_acc:.4f}, Val   Loss: {val_loss:.6f}")

    val_pred = model.predict(X_val)
    print("\nPer-class validation accuracy")
    for c in range(num_classes):
        mask = y_val == c
        if np.any(mask):
            cls_acc = np.mean(val_pred[mask] == y_val[mask])
            print(f"Class {label_map[c]:>2s}: {cls_acc:.4f}")

    with open("part1_2_model.pkl", "wb") as f:
        pickle.dump(
            {
                "weights": model.weights,
                "biases": model.biases,
                "layer_sizes": model.layer_sizes,
                "learning_rate": model.learning_rate,
                "dropout_rate": model.dropout_rate,
                "label_map": label_map,
            },
            f,
        )
    print("\nSaved model: part1_2_model.pkl")

    if HAS_MATPLOTLIB:
        plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1)
        plt.plot(losses)
        plt.title("Part1-2 Loss")
        plt.xlabel("Epoch")
        plt.ylabel("CrossEntropy")
        plt.grid(alpha=0.3)

        plt.subplot(1, 2, 2)
        plt.bar(["Train", "Val"], [train_acc, val_acc])
        plt.ylim(0, 1)
        plt.title("Accuracy")
        plt.grid(alpha=0.3, axis="y")

        plt.tight_layout()
        plt.savefig("part1_2_result.png", dpi=120)
        print("Saved figure: part1_2_result.png")

    return model, losses, val_acc


if __name__ == "__main__":
    main()
