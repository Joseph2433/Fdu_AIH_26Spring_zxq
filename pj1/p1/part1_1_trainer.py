import numpy as np
from typing import Tuple, List

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


class NeuralNetwork:
    def __init__(self, layer_sizes: List[int], learning_rate: float):
        
        self.layer_sizes = layer_sizes
        self.learning_rate = learning_rate
        self.num_layers = len(layer_sizes)
        
        # Initialize weights and biases
        self.weights = []  # weights[i] is the weight matrix from layer i to layer i+1
        self.biases = []   # biases[i] is the bias vector for layer i+1
        
        # Xavier initialization helps stabilize convergence with moderate initial values
        for i in range(len(layer_sizes) - 1):
            fan_in = layer_sizes[i]
            fan_out = layer_sizes[i + 1]
            
            # Xavier initialization
            limit = np.sqrt(6.0 / (fan_in + fan_out))
            w = np.random.uniform(-limit, limit, (fan_in, fan_out))
            self.weights.append(w)
            
            # Initialize biases in a small range to support stable training
            b = np.random.uniform(-0.1, 0.1, (1, fan_out))
            self.biases.append(b)
    
    def sigmoid(self, x: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))  # Prevent overflow
    
    def sigmoid_derivative(self, x: np.ndarray) -> np.ndarray:
        s = self.sigmoid(x)
        return s * (1 - s)
    
    def relu(self, x: np.ndarray) -> np.ndarray:
        return np.maximum(0, x)
    
    def relu_derivative(self, x: np.ndarray) -> np.ndarray:
        return (x > 0).astype(float)
    
    def forward(
        self,
        X: np.ndarray,
    ) -> Tuple[np.ndarray, List[np.ndarray], List[np.ndarray]]:
        activations = [X]
        zs = []
        
        for i in range(self.num_layers - 1):
            # Linear transformation
            z = np.dot(activations[-1], self.weights[i]) + self.biases[i]
            zs.append(z)
            
            # Activation: ReLU for hidden layers, no activation for output (regression)
            if i < self.num_layers - 2:
                a = self.relu(z)
            else:
                # Output layer: no activation, use raw output
                a = z
            
            activations.append(a)
        
        return activations[-1], zs, activations
    
    def backward(self, X: np.ndarray, y: np.ndarray, output: np.ndarray, 
                 zs: List[np.ndarray], activations: List[np.ndarray]) -> None:
        m = X.shape[0]  # Number of samples
        
        # Compute output-layer gradient
        delta = (output - y) / m  # Gradient of MSE loss
        
        # Backpropagation: from last layer to first layer
        for i in range(self.num_layers - 2, -1, -1):
            # Backup current-layer weights
            w_curr = self.weights[i].copy()
            # Compute gradients for weights and biases
            dW = np.dot(activations[i].T, delta)
            db = np.sum(delta, axis=0, keepdims=True)
            
            # Update weights and biases
            self.weights[i] -= self.learning_rate * dW
            self.biases[i] -= self.learning_rate * db
            
            # Compute gradient for previous layer (if not input layer)
            if i > 0:
                delta = np.dot(delta, w_curr.T)
                # Apply activation derivative
                delta *= self.relu_derivative(zs[i - 1])
    
    def train(self, X: np.ndarray, y: np.ndarray, epochs: int = 1000, 
              batch_size: int = 32, verbose: bool = True) -> List[float]:
        losses = []
        
        for epoch in range(epochs):
            # Shuffle data
            indices = np.random.permutation(X.shape[0])
            X_shuffled = X[indices]
            y_shuffled = y[indices]
            
            epoch_loss = 0.0
            batch_count = 0
            
            # Train
            for i in range(0, X.shape[0], batch_size):
                X_batch = X_shuffled[i:i + batch_size]
                y_batch = y_shuffled[i:i + batch_size]
                
                # Forward pass
                output, zs, activations = self.forward(X_batch)
                
                # Compute loss
                loss = np.mean((output - y_batch) ** 2)
                epoch_loss += float(loss)
                batch_count += 1
                
                # Backward pass
                self.backward(X_batch, y_batch, output, zs, activations)
            
            avg_loss = epoch_loss / max(batch_count, 1)
            losses.append(avg_loss)
            
            if verbose and (epoch + 1) % 100 == 0:
                print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.6f}")
        
        return losses
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        output, _, _ = self.forward(X)
        return output


def generate_sin_data(num_samples: int , x_range: Tuple[float, float] = (-np.pi, np.pi)) -> Tuple[np.ndarray, np.ndarray]:
    X = np.random.uniform(x_range[0], x_range[1], (num_samples, 1))
    y = np.sin(X)
    return X, y


def evaluate_model(model: NeuralNetwork, X_test: np.ndarray, y_test: np.ndarray) -> float:
    y_pred = model.predict(X_test)
    mae = np.mean(np.abs(y_pred - y_test))
    return mae


def main():
    print("Part1-1: Neural network fitting for sin(x)")
    
    # Generate training data
    print("\n1. Generating training data...")
    X_train, y_train = generate_sin_data(num_samples=200, x_range=(-np.pi, np.pi))
    print(f"   Training set size: {X_train.shape[0]} samples")
    
    # Generate test data (uniform distribution for evaluation)
    print("\n2. Generating test data...")
    x_test = np.linspace(-np.pi, np.pi, 100).reshape(-1, 1)
    y_test = np.sin(x_test)
    print(f"   Test set size: {x_test.shape[0]} samples")
    
    # Create and train the model
    print("\n3. Creating and training the neural network...")

    # Network architecture
    layer_sizes=[1, 16, 16, 1]
    # Training hyperparameters
    learning_rate=0.05
    epochs=1000
    batch_size=32
    verbose=True

    model = NeuralNetwork(
        layer_sizes,
        learning_rate
    )
    
    losses = model.train(
        X_train, y_train,
        epochs,
        batch_size,
        verbose
    )
    
    # Evaluate the model
    print("\n4. Evaluating the model...")
    mae = evaluate_model(model, x_test, y_test)
    print(f"   Test set mean absolute error (MAE): {mae:.6f}")
    
    if mae < 0.01:
        print(f"   SUCCESS! Error {mae:.6f} < 0.01")
    else:
        print(f"   TARGET NOT MET. Error {mae:.6f} >= 0.01")
    
    # Plot results
    print("\n5. Plotting results...")
    y_pred = model.predict(x_test)
    
    if HAS_MATPLOTLIB:
        plt.figure(figsize=(12, 4))
        
        # Left panel: fitting result
        plt.subplot(1, 2, 1)
        plt.scatter(X_train, y_train, alpha=0.3, label='Training data', s=20)
        plt.plot(x_test, y_test, 'r-', linewidth=2, label='True sin(x)')
        plt.plot(x_test, y_pred, 'b--', linewidth=2, label='Prediction')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('Neural network fit for sin(x)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Right panel: loss curve
        plt.subplot(1, 2, 2)
        plt.plot(losses, linewidth=1)
        plt.xlabel('Epoch')
        plt.ylabel('MSE Loss')
        plt.title('Training loss curve')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('part1_1_result.png', dpi=100, bbox_inches='tight')
        print("   Figure saved to part1_1_result.png")
        
        plt.show()
    else:
        print("   (matplotlib not installed, skipping plotting)")
    
    return model, losses, mae


if __name__ == "__main__":
    model, losses, mae = main()
