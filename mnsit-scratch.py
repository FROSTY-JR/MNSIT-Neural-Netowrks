import numpy as np
import matplotlib.pyplot as plt

class MNISTNeuralNetwork:
    def __init__(self, input_size=784, hidden_sizes=[128, 64], output_size=10, learning_rate=0.01):
        
        self.learning_rate = learning_rate
        self.layers = [input_size] + hidden_sizes + [output_size]
        self.num_layers = len(self.layers)
        
        # Initialize weights and biases
        self.weights = []
        self.biases = []
        
        for i in range(self.num_layers - 1):
            # Xavier initialization
            w = np.random.randn(self.layers[i], self.layers[i + 1]) * np.sqrt(2.0 / self.layers[i])
            b = np.zeros((1, self.layers[i + 1]))
            self.weights.append(w)
            self.biases.append(b)
    
    def relu(self, x):
       
        return np.maximum(0, x)
    
    def relu_derivative(self, x):
       
        return (x > 0).astype(float)
    
    def softmax(self, x):
        
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
    
    def forward(self, X):
        """Forward propagation"""
        self.activations = [X]
        self.z_values = []
        
        current_input = X
        
        # Forward through hidden layers (with ReLU)
        for i in range(len(self.weights) - 1):
            z = np.dot(current_input, self.weights[i]) + self.biases[i]
            self.z_values.append(z)
            a = self.relu(z)
            self.activations.append(a)
            current_input = a
        
        # Output layer (with softmax)
        z_output = np.dot(current_input, self.weights[-1]) + self.biases[-1]
        self.z_values.append(z_output)
        output = self.softmax(z_output)
        self.activations.append(output)
        
        return output
    
    def backward(self, X, y):
        """Backward propagation"""
        m = X.shape[0]
        
        # Convert labels to one-hot encoding
        y_onehot = np.zeros((m, 10))
        y_onehot[np.arange(m), y] = 1
        
        # Calculate gradients
        gradients_w = []
        gradients_b = []
        
        # Output layer gradient (softmax + cross-entropy)
        delta = self.activations[-1] - y_onehot
        
        # Backpropagate through all layers
        for i in range(len(self.weights) - 1, -1, -1):
            # Calculate gradients for current layer
            grad_w = (1/m) * np.dot(self.activations[i].T, delta)
            grad_b = (1/m) * np.sum(delta, axis=0, keepdims=True)
            
            gradients_w.insert(0, grad_w)
            gradients_b.insert(0, grad_b)
            
            # Calculate delta for previous layer (if not input layer)
            if i > 0:
                delta = np.dot(delta, self.weights[i].T) * self.relu_derivative(self.z_values[i-1])
        
        # Update weights and biases
        for i in range(len(self.weights)):
            self.weights[i] -= self.learning_rate * gradients_w[i]
            self.biases[i] -= self.learning_rate * gradients_b[i]
    
    def cross_entropy_loss(self, predictions, labels):
        """Calculate cross-entropy loss"""
        m = labels.shape[0]
        # Add small epsilon to prevent log(0)
        epsilon = 1e-15
        predictions = np.clip(predictions, epsilon, 1 - epsilon)
        
        # One-hot encode labels
        y_onehot = np.zeros((m, 10))
        y_onehot[np.arange(m), labels] = 1
        
        loss = -np.sum(y_onehot * np.log(predictions)) / m
        return loss
    
    def train(self, X_train, y_train, X_val, y_val, epochs=100, batch_size=128):
        """Train the neural network"""
        train_losses = []
        val_losses = []
        train_accuracies = []
        val_accuracies = []
        
        n_batches = len(X_train) // batch_size
        
        for epoch in range(epochs):
            epoch_loss = 0
            
            # Shuffle training data
            indices = np.random.permutation(len(X_train))
            X_train_shuffled = X_train[indices]
            y_train_shuffled = y_train[indices]
            
            # Mini-batch training
            for i in range(n_batches):
                start_idx = i * batch_size
                end_idx = start_idx + batch_size
                
                X_batch = X_train_shuffled[start_idx:end_idx]
                y_batch = y_train_shuffled[start_idx:end_idx]
                
                # Forward and backward pass
                predictions = self.forward(X_batch)
                self.backward(X_batch, y_batch)
                
                batch_loss = self.cross_entropy_loss(predictions, y_batch)
                epoch_loss += batch_loss
            
            # Calculate average epoch loss
            avg_train_loss = epoch_loss / n_batches
            
            # Evaluate on validation set
            val_predictions = self.predict(X_val)
            val_loss = self.cross_entropy_loss(self.forward(X_val), y_val)
            
            # Calculate accuracies
            train_pred = self.predict(X_train[:1000])  # Use subset for speed
            train_acc = np.mean(train_pred == y_train[:1000]) * 100
            val_acc = np.mean(val_predictions == y_val) * 100
            
            # Store metrics
            train_losses.append(avg_train_loss)
            val_losses.append(val_loss)
            train_accuracies.append(train_acc)
            val_accuracies.append(val_acc)
            
            # Print progress
            if epoch % 10 == 0:
                print(f"Epoch {epoch:3d} | Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f} | "
                      f"Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%")
        
        return {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'train_accuracies': train_accuracies,
            'val_accuracies': val_accuracies
        }
    
    def predict(self, X):
        """Make predictions"""
        output = self.forward(X)
        return np.argmax(output, axis=1)

def load_mnist_data():
    """Load MNIST dataset using Keras"""
    print("Loading MNIST dataset...")
    
    # Try Keras first (best option)
    try:
        from tensorflow.keras.datasets import mnist
        print("Loading MNIST from Keras...")
        
        (X_train, y_train), (X_test, y_test) = mnist.load_data()
        
        # Reshape to flatten images
        X_train = X_train.reshape(X_train.shape[0], -1)
        X_test = X_test.reshape(X_test.shape[0], -1)
        
        print(f"Loaded from Keras: {X_train.shape[0]} training, {X_test.shape[0]} test samples")
        return X_train, y_train, X_test, y_test
        
    except Exception as e:
        print(f"Keras loading failed: {e}")
    
    # Fallback to sklearn
    try:
        from sklearn.datasets import fetch_openml
        print("Attempting to load from sklearn...")
        
        # Load MNIST from OpenML
        mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')
        X, y = mnist.data, mnist.target.astype(int)
        
        # Split into train and test
        X_train, X_test = X[:60000], X[60000:]
        y_train, y_test = y[:60000], y[60000:]
        
        print(f"Loaded from sklearn: {X_train.shape[0]} training, {X_test.shape[0]} test samples")
        return X_train, y_train, X_test, y_test
        
    except Exception as e:
        print(f"sklearn loading failed: {e}")
    
    # Create simple synthetic data as last resort
    print("Creating simple synthetic dataset for demonstration...")
    
    np.random.seed(42)
    
    # Create very simple synthetic data
    n_samples_per_class = 1000
    n_classes = 10
    
    X_train = []
    y_train = []
    X_test = []
    y_test = []
    
    for digit in range(n_classes):
        # Create simple patterns for each digit
        base_pattern = np.random.rand(784) * 0.3  # Base noise
        
        # Add digit-specific pattern
        if digit == 0:
            base_pattern[300:400] += 0.7  # Middle region
        elif digit == 1:
            base_pattern[350:450] += 0.8  # Vertical line region
        elif digit == 2:
            base_pattern[200:300] += 0.6
            base_pattern[500:600] += 0.6
        else:
            # Simple random pattern for other digits
            start_idx = digit * 70
            base_pattern[start_idx:start_idx+100] += 0.7
        
        # Generate training samples
        for _ in range(n_samples_per_class):
            sample = base_pattern + np.random.normal(0, 0.1, 784)
            sample = np.clip(sample, 0, 1)
            X_train.append(sample)
            y_train.append(digit)
        
        # Generate test samples
        for _ in range(n_samples_per_class // 5):
            sample = base_pattern + np.random.normal(0, 0.1, 784)
            sample = np.clip(sample, 0, 1)
            X_test.append(sample)
            y_test.append(digit)
    
    # Convert to numpy arrays and shuffle
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_test = np.array(X_test)
    y_test = np.array(y_test)
    
    # Shuffle
    train_indices = np.random.permutation(len(X_train))
    test_indices = np.random.permutation(len(X_test))
    
    X_train = X_train[train_indices]
    y_train = y_train[train_indices]
    X_test = X_test[test_indices]
    y_test = y_test[test_indices]
    
    print(f"Created synthetic dataset: {X_train.shape[0]} training, {X_test.shape[0]} test samples")
    return X_train, y_train, X_test, y_test

def preprocess_data(X_train, X_test):
    """Preprocess the data"""
    # Normalize pixel values to [0, 1]
    X_train = X_train.astype(np.float32) / 255.0
    X_test = X_test.astype(np.float32) / 255.0
    
    return X_train, X_test

def plot_training_history(history):
    """Plot training history"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot losses
    ax1.plot(history['train_losses'], label='Training Loss')
    ax1.plot(history['val_losses'], label='Validation Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Plot accuracies
    ax2.plot(history['train_accuracies'], label='Training Accuracy')
    ax2.plot(history['val_accuracies'], label='Validation Accuracy')
    ax2.set_title('Training and Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    ax2.grid(True)
    
    # Plot loss comparison
    ax3.semilogy(history['train_losses'], label='Training Loss')
    ax3.semilogy(history['val_losses'], label='Validation Loss')
    ax3.set_title('Loss (Log Scale)')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Loss (log)')
    ax3.legend()
    ax3.grid(True)
    
    # Plot final metrics
    final_train_acc = history['train_accuracies'][-1]
    final_val_acc = history['val_accuracies'][-1]
    ax4.bar(['Training', 'Validation'], [final_train_acc, final_val_acc])
    ax4.set_title('Final Accuracy')
    ax4.set_ylabel('Accuracy (%)')
    ax4.set_ylim(0, 100)
    
    for i, v in enumerate([final_train_acc, final_val_acc]):
        ax4.text(i, v + 1, f'{v:.2f}%', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.show()

def visualize_predictions(X_test, y_test, model, num_samples=10):
    """Visualize some predictions"""
    # Get random samples
    indices = np.random.choice(len(X_test), num_samples, replace=False)
    
    fig, axes = plt.subplots(2, 5, figsize=(12, 6))
    axes = axes.ravel()
    
    for i, idx in enumerate(indices):
        # Reshape image for display
        image = X_test[idx].reshape(28, 28)
        
        # Make prediction
        pred = model.predict(X_test[idx:idx+1])[0]
        actual = y_test[idx]
        
        # Plot image
        axes[i].imshow(image, cmap='gray')
        axes[i].set_title(f'Pred: {pred}, Actual: {actual}')
        axes[i].axis('off')
        
        # Color title based on correctness
        if pred == actual:
            axes[i].title.set_color('green')
        else:
            axes[i].title.set_color('red')
    
    plt.tight_layout()
    plt.show()

def main():
    print("MNIST Digit Classification with Neural Network from Scratch\n")
    
    # Load and preprocess data
    X_train, y_train, X_test, y_test = load_mnist_data()
    X_train, X_test = preprocess_data(X_train, X_test)
    
    # Create validation set from training set
    val_size = 10000
    X_val = X_train[-val_size:]
    y_val = y_train[-val_size:]
    X_train = X_train[:-val_size]
    y_train = y_train[:-val_size]
    
    print(f"Training set: {X_train.shape}")
    print(f"Validation set: {X_val.shape}")
    print(f"Test set: {X_test.shape}\n")
    
    # Create and train model
    model = MNISTNeuralNetwork(
        input_size=784,
        hidden_sizes=[128, 64],
        output_size=10,
        learning_rate=0.01
    )
    
    print("Network Architecture:")
    print(f"Input Layer: 784 neurons (28x28 pixels)")
    print(f"Hidden Layer 1: 128 neurons (ReLU)")
    print(f"Hidden Layer 2: 64 neurons (ReLU)")
    print(f"Output Layer: 10 neurons (Softmax)")
    
    # Count parameters
    total_params = sum(w.size + b.size for w, b in zip(model.weights, model.biases))
    print(f"Total Parameters: {total_params:,}\n")
    
    # Train the model
    print("Starting training...")
    history = model.train(X_train, y_train, X_val, y_val, epochs=50, batch_size=128)
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    test_predictions = model.predict(X_test)
    test_accuracy = np.mean(test_predictions == y_test) * 100
    
    print(f"Test Accuracy: {test_accuracy:.2f}%")
    
    # Calculate per-class accuracy
    print("\nPer-class accuracy:")
    for digit in range(10):
        digit_mask = y_test == digit
        digit_acc = np.mean(test_predictions[digit_mask] == y_test[digit_mask]) * 100
        print(f"Digit {digit}: {digit_acc:.2f}%")
    
    # Plot training history
    plot_training_history(history)
    
    # Visualize some predictions
    visualize_predictions(X_test, y_test, model)
    
    print(f"\nTraining completed! Final test accuracy: {test_accuracy:.2f}%")

if __name__ == "__main__":
    main()