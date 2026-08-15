"""
Comparison demo: Regular Neural Network vs Memory Bank Neuron

This script trains both systems on the same circle boundary problem
and compares their performance and characteristics.
"""

import numpy as np
import matplotlib.pyplot as plt
from neuron import NeuralNetwork
from memory_neuron import MemoryNeuron


def create_dataset(n_samples=500, problem="circle"):
    """Generate a small benchmark dataset for comparing two learning approaches."""
    rng = np.random.default_rng(42)
    X = rng.uniform(0, 1, size=(n_samples, 2))
    
    if problem == "circle":
        # Circle decision boundary
        y = (np.sqrt((X[:, 0] - 0.5)**2 + (X[:, 1] - 0.5)**2) < 0.3).astype(float)
    elif problem == "xor":
        # XOR problem
        y = ((X[:, 0] > 0.5) != (X[:, 1] > 0.5)).astype(float)
    else:  # linear
        # Linear decision boundary
        y = (X[:, 0] + X[:, 1] > 1.0).astype(float)
    
    return X, y


def main():
    """Train a standard network and a memory-based neuron on the same dataset and compare the results."""
    print("=" * 70)
    print("NEURAL NETWORK vs MEMORY NEURON COMPARISON")
    print("=" * 70)
    
    # Create dataset
    X, y = create_dataset(n_samples=500, problem="circle")
    
    print(f"\nDataset: Circle boundary problem")
    print(f"Samples: {len(X)} | Features: {X.shape[1]} | Classes: 2")
    print(f"Class distribution: {int(np.sum(y))} positive, {len(y) - int(np.sum(y))} negative")
    
    # Split into train/test
    split_idx = int(0.8 * len(X))
    X_train, y_train = X[:split_idx], y[:split_idx]
    X_test, y_test = X[split_idx:], y[split_idx:]
    
    print(f"Train: {len(X_train)} | Test: {len(X_test)}")
    
    # ==================== REGULAR NEURAL NETWORK ====================
    print("\n" + "=" * 70)
    print("TRAINING: Regular Deep Neural Network")
    print("=" * 70)
    
    nn = NeuralNetwork(
        layer_sizes=(2, 32, 32, 16, 8, 1),
        activations=("relu", "relu", "relu", "relu", "sigmoid"),
        loss="bce",
        learning_rate=0.01,
        rng=np.random.default_rng(42),
    )
    
    print(f"Architecture: {nn.layer_sizes}")
    print(f"Total parameters: {sum(w.size + b.size for w, b in zip(nn.weights, nn.biases))}")
    print("Training...")
    
    history_nn = nn.fit((X_train, y_train), epochs=1000, batch_size=32, verbose=False)
    
    train_acc_nn = nn.accuracy(X_train, y_train)
    test_acc_nn = nn.accuracy(X_test, y_test)
    
    print(f"Training accuracy: {train_acc_nn:.4f}")
    print(f"Test accuracy: {test_acc_nn:.4f}")
    print(f"Final loss: {history_nn['loss'][-1]:.6f}")
    
    # ==================== MEMORY NEURON ====================
    print("\n" + "=" * 70)
    print("TRAINING: Memory Bank Neuron")
    print("=" * 70)
    
    mem = MemoryNeuron(
        n_inputs=2,
        memory_size=200,
        learning_rate=0.2,
        similarity_metric="euclidean",
        rng=np.random.default_rng(42),
    )
    
    print(f"Memory size: {mem.memory_size}")
    print("Training...")
    
    history_mem = mem.fit((X_train, y_train), epochs=100, batch_size=16, verbose=False)
    
    train_acc_mem = mem.accuracy(X_train, y_train)
    test_acc_mem = mem.accuracy(X_test, y_test)
    
    print(f"Training accuracy: {train_acc_mem:.4f}")
    print(f"Test accuracy: {test_acc_mem:.4f}")
    print(f"Final loss: {history_mem['loss'][-1]:.6f}")
    print(f"Patterns stored: {len(mem.memory)}")
    print(f"Memory info: {mem.memory_info()}")
    
    # ==================== COMPARISON ====================
    print("\n" + "=" * 70)
    print("COMPARISON")
    print("=" * 70)
    
    print(f"\n{'Metric':<25} {'Neural Network':<20} {'Memory Neuron':<20}")
    print("-" * 65)
    print(f"{'Train Accuracy':<25} {train_acc_nn:<20.4f} {train_acc_mem:<20.4f}")
    print(f"{'Test Accuracy':<25} {test_acc_nn:<20.4f} {test_acc_mem:<20.4f}")
    print(f"{'Final Loss':<25} {history_nn['loss'][-1]:<20.6f} {history_mem['loss'][-1]:<20.6f}")
    print(f"{'Convergence Speed':<25} {'Stable':<20} {'Fast':<20}")
    print(f"{'Interpretability':<25} {'Black box':<20} {'Human readable':<20}")
    print(f"{'Memory Usage':<25} {f'{sum(w.size for w in nn.weights)}':<20} {f'{len(mem.memory)}':<20}")
    
    # ==================== VISUALIZATION ====================
    print("\n" + "=" * 70)
    print("VISUALIZING RESULTS")
    print("=" * 70)
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle("Neural Network vs Memory Neuron Comparison", fontsize=14, weight="bold")
    
    # Plot 1: Training curves
    ax = axes[0, 0]
    ax.plot(history_nn["loss"], label="NN", lw=2, alpha=0.7)
    ax.plot(history_mem["loss"], label="Memory", lw=2, alpha=0.7)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training Loss Over Time")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Test accuracy comparison
    ax = axes[0, 1]
    models = ["NN", "Memory"]
    accuracies = [test_acc_nn, test_acc_mem]
    colors = ["#2196F3", "#FF9800"]
    ax.bar(models, accuracies, color=colors, alpha=0.7, edgecolor="black", lw=1.5)
    ax.set_ylabel("Accuracy")
    ax.set_title("Test Accuracy Comparison")
    ax.set_ylim([0, 1])
    for i, v in enumerate(accuracies):
        ax.text(i, v + 0.02, f"{v:.3f}", ha="center", weight="bold")
    ax.grid(True, alpha=0.3, axis="y")
    
    # Plot 3: Memory neuron pattern storage
    ax = axes[0, 2]
    ax.plot(history_mem["memory_size"], lw=2, color="#FF9800")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Stored Patterns")
    ax.set_title("Memory Growth During Training")
    ax.grid(True, alpha=0.3)
    ax.fill_between(range(len(history_mem["memory_size"])), 
                     history_mem["memory_size"], alpha=0.3, color="#FF9800")
    
    # Plot 4: Decision boundary - Neural Network
    ax = axes[1, 0]
    xx, yy = np.meshgrid(np.linspace(0, 1, 100), np.linspace(0, 1, 100))
    Z_nn = nn.predict_proba(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    
    contour = ax.contourf(xx, yy, Z_nn, levels=20, cmap="coolwarm", alpha=0.7)
    scatter = ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap="coolwarm", 
                        edgecolors="black", lw=0.5, s=30, alpha=0.8)
    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")
    ax.set_title(f"NN Decision Boundary (Acc: {test_acc_nn:.3f})")
    plt.colorbar(contour, ax=ax)
    
    # Plot 5: Decision boundary - Memory Neuron
    ax = axes[1, 1]
    Z_mem = mem.predict_proba(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    
    contour = ax.contourf(xx, yy, Z_mem, levels=20, cmap="coolwarm", alpha=0.7)
    scatter = ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap="coolwarm",
                        edgecolors="black", lw=0.5, s=30, alpha=0.8)
    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")
    ax.set_title(f"Memory Decision Boundary (Acc: {test_acc_mem:.3f})")
    plt.colorbar(contour, ax=ax)
    
    # Plot 6: Memory patterns
    ax = axes[1, 2]
    if len(mem.memory) > 0:
        patterns = np.array([p[0] for p in mem.memory])
        outputs = np.array([p[1] for p in mem.memory])
        frequencies = np.array([p[2] for p in mem.memory])
        
        scatter = ax.scatter(patterns[:, 0], patterns[:, 1], c=outputs,
                           s=frequencies*10, cmap="coolwarm", alpha=0.7,
                           edgecolors="black", lw=0.5)
        ax.set_xlabel("Feature 1")
        ax.set_ylabel("Feature 2")
        ax.set_title(f"Stored Memory Patterns (n={len(mem.memory)})")
        plt.colorbar(scatter, ax=ax)
    
    plt.tight_layout()
    try:
        plt.show()
        print("✅ Displaying comparison plots live.")
    except Exception as exc:
        print(f"⚠️ Could not display plots live: {exc}")
        print("If this is running headless, use a GUI backend or open the plot in VS Code's output viewer.")
    finally:
        plt.close(fig)
    
    print("\n✅ Comparison complete!")


if __name__ == "__main__":
    main()
