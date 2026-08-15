"""
Memory Bank Neuron - A pattern-matching neuron with memory storage.

This neuron stores input-output patterns and learns by retrieving similar patterns
from memory when making predictions. It's inspired by memory networks and 
nearest-neighbor approaches, but with learnable similarity metrics.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt


class MemoryNeuron:
    """A neuron that learns and predicts using a pattern memory bank.
    
    Instead of just having weights, this neuron stores actual input-output patterns
    and uses similarity to make predictions. It's like a neural network that learns
    by remembering examples.
    """

    def __init__(
        self,
        n_inputs: int,
        memory_size: int = 100,
        learning_rate: float = 0.1,
        similarity_metric: str = "euclidean",
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        """
        Initialize a Memory Neuron.
        
        Args:
            n_inputs: Number of input features
            memory_size: Maximum number of patterns to store
            learning_rate: How much to update pattern outputs during training
            similarity_metric: How to measure pattern similarity ("euclidean" or "cosine")
            rng: Random number generator for reproducibility
        """
        self.n_inputs = n_inputs
        self.memory_size = memory_size
        self.learning_rate = float(learning_rate)
        self.similarity_metric = similarity_metric
        self.rng = rng or np.random.default_rng()
        
        # Memory storage: list of (input_pattern, output_value, frequency)
        self.memory: List[Tuple[np.ndarray, float, int]] = []
        self.history: Dict[str, list] = {"loss": [], "memory_size": []}
        
        # Learn a scaling factor for similarity (helps with generalization)
        self.similarity_scale = 1.0

    def _compute_similarity(self, x1: np.ndarray, x2: np.ndarray) -> float:
        """Compute similarity between two patterns (0 to 1, where 1 = identical)."""
        if self.similarity_metric == "euclidean":
            dist = float(np.linalg.norm(x1 - x2))
            # Convert distance to similarity (Gaussian kernel)
            similarity = float(np.exp(-self.similarity_scale * dist ** 2))
        else:  # cosine
            norm1 = float(np.linalg.norm(x1))
            norm2 = float(np.linalg.norm(x2))
            if norm1 == 0 or norm2 == 0:
                similarity = 0.0
            else:
                similarity = float(np.dot(x1, x2) / (norm1 * norm2))
                similarity = (similarity + 1) / 2  # Normalize to [0, 1]
        return np.clip(similarity, 0.0, 1.0)

    def _find_similar_patterns(
        self, 
        x: np.ndarray, 
        k: int = 5
    ) -> List[Tuple[float, float, int]]:
        """Find k most similar patterns in memory.
        
        Returns list of (similarity, output, frequency) tuples.
        """
        if len(self.memory) == 0:
            return []
        
        similarities = []
        for pattern, output, freq in self.memory:
            sim = self._compute_similarity(x, pattern)
            similarities.append((sim, output, freq))
        
        # Sort by similarity (descending) and return top k
        similarities.sort(reverse=True)
        return similarities[:k]

    def forward(self, X: np.ndarray) -> np.ndarray:
        """Predict outputs using weighted sum of similar patterns in memory."""
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        if X.shape[1] != self.n_inputs:
            raise ValueError(f"Input must have shape (n_samples, {self.n_inputs})")
        
        predictions = []
        for x in X:
            similar = self._find_similar_patterns(x, k=min(5, len(self.memory)))
            
            if len(similar) == 0:
                # No memory yet, return random activation
                pred = float(self.rng.uniform(0, 1))
            else:
                # Weighted average of similar patterns (similarity = weight)
                total_sim = sum(sim for sim, _, _ in similar)
                if total_sim > 0:
                    weighted_output = sum(
                        sim * output for sim, output, _ in similar
                    ) / total_sim
                    pred = float(weighted_output)
                else:
                    pred = float(similar[0][1])  # Fallback to most similar
            
            predictions.append(pred)
        
        return np.array(predictions).squeeze()

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return continuous predictions (already in [0, 1] for sigmoid-like)."""
        return self.forward(X)

    def predict_class(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Convert continuous predictions to binary class."""
        proba = self.predict_proba(X)
        return (np.asarray(proba) >= float(threshold)).astype(int)

    def _add_to_memory(self, x: np.ndarray, y: float) -> None:
        """Add pattern to memory, checking for similar patterns first."""
        x = np.asarray(x, dtype=float).flatten()
        
        # Check if we already have a very similar pattern
        for i, (stored_x, stored_y, freq) in enumerate(self.memory):
            similarity = self._compute_similarity(x, stored_x)
            if similarity > 0.95:  # Found very similar pattern
                # Update the output (learning) and increase frequency
                new_y = stored_y + self.learning_rate * (y - stored_y)
                self.memory[i] = (stored_x, float(new_y), freq + 1)
                return
        
        # If memory is full, remove least used pattern
        if len(self.memory) >= self.memory_size:
            # Find pattern with lowest frequency * similarity to inputs
            min_score = float('inf')
            min_idx = 0
            for i, (pat, _, freq) in enumerate(self.memory):
                # Low frequency patterns are more likely to be removed
                if freq < min_score:
                    min_score = freq
                    min_idx = i
            self.memory.pop(min_idx)
        
        # Add new pattern
        self.memory.append((x.copy(), float(y), 1))

    def train_step(self, X: np.ndarray, y: np.ndarray) -> float:
        """Train on a batch of samples."""
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        
        if X.ndim == 1:
            X = X.reshape(1, -1)
        if y.ndim == 0:
            y = y.reshape(1)
        if y.ndim == 1:
            y = y.reshape(-1)
        
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have same number of samples")
        
        # Store each pattern in memory
        for x_sample, y_sample in zip(X, y):
            self._add_to_memory(x_sample, y_sample)
        
        # Compute loss on this batch
        preds = self.forward(X)
        loss = float(np.mean((preds - y) ** 2))
        
        return loss

    def fit(
        self,
        training_data: Tuple[np.ndarray, np.ndarray],
        epochs: int = 100,
        batch_size: int = 4,
        shuffle: bool = True,
        verbose: bool = False,
    ) -> Dict[str, list]:
        """Train the memory neuron."""
        X_all, y_all = training_data
        X_all = np.asarray(X_all, dtype=float)
        y_all = np.asarray(y_all, dtype=float).reshape(-1)
        
        if X_all.ndim == 1:
            X_all = X_all.reshape(1, -1)
        if X_all.shape[0] != y_all.shape[0]:
            raise ValueError("X and y must have same number of samples")
        
        n_samples = X_all.shape[0]
        if batch_size <= 0 or batch_size > n_samples:
            batch_size = n_samples
        
        self.history = {"loss": [], "memory_size": []}
        
        for epoch in range(1, epochs + 1):
            if shuffle:
                idx = self.rng.permutation(n_samples)
                X_all = X_all[idx]
                y_all = y_all[idx]
            
            epoch_losses = []
            for start in range(0, n_samples, batch_size):
                Xb = X_all[start : start + batch_size]
                yb = y_all[start : start + batch_size]
                loss = self.train_step(Xb, yb)
                epoch_losses.append(loss)
            
            avg_loss = float(np.mean(epoch_losses))
            self.history["loss"].append(avg_loss)
            self.history["memory_size"].append(len(self.memory))
            
            if verbose and epoch % max(1, epochs // 10) == 0:
                print(f"Epoch {epoch}: loss={avg_loss:.6f}, memory={len(self.memory)}")
        
        return self.history

    def accuracy(self, X: np.ndarray, y_true: np.ndarray, threshold: float = 0.5) -> float:
        """Compute accuracy for binary classification."""
        y_true = np.asarray(y_true, dtype=float).reshape(-1)
        y_pred = self.predict_class(X, threshold)
        return float(np.mean(y_pred == y_true))

    def memory_info(self) -> Dict:
        """Get information about stored patterns."""
        if len(self.memory) == 0:
            return {"total_patterns": 0, "avg_frequency": 0}
        
        frequencies = [freq for _, _, freq in self.memory]
        return {
            "total_patterns": len(self.memory),
            "avg_frequency": float(np.mean(frequencies)),
            "max_frequency": int(np.max(frequencies)),
            "min_frequency": int(np.min(frequencies)),
        }


def visualize_memory(neuron: MemoryNeuron, save_path: Optional[str] = None):
    """Visualize the memory bank patterns."""
    if len(neuron.memory) == 0:
        print("Memory is empty!")
        return
    
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib required for visualization")
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Plot 1: Memory patterns in 2D (if n_inputs == 2)
    if neuron.n_inputs == 2:
        ax = axes[0]
        patterns = np.array([p[0] for p in neuron.memory])
        outputs = np.array([p[1] for p in neuron.memory])
        frequencies = np.array([p[2] for p in neuron.memory])
        
        scatter = ax.scatter(
            patterns[:, 0], patterns[:, 1], 
            c=outputs, s=frequencies*20, 
            cmap="coolwarm", alpha=0.6, edgecolors="black", lw=1
        )
        ax.set_xlabel("Input 1")
        ax.set_ylabel("Input 2")
        ax.set_title(f"Memory Patterns (n={len(neuron.memory)})")
        ax.grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=ax, label="Output")
    else:
        # For higher dimensions, show first two dimensions
        ax = axes[0]
        patterns = np.array([p[0][:2] for p in neuron.memory])
        outputs = np.array([p[1] for p in neuron.memory])
        frequencies = np.array([p[2] for p in neuron.memory])
        
        scatter = ax.scatter(
            patterns[:, 0], patterns[:, 1],
            c=outputs, s=frequencies*20,
            cmap="coolwarm", alpha=0.6, edgecolors="black", lw=1
        )
        ax.set_xlabel("Dimension 1")
        ax.set_ylabel("Dimension 2")
        ax.set_title(f"Memory Patterns (first 2 dims, n={len(neuron.memory)})")
        ax.grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=ax, label="Output")
    
    # Plot 2: Memory statistics
    ax = axes[1]
    frequencies = [freq for _, _, freq in neuron.memory]
    ax.hist(frequencies, bins=20, color="#2196F3", alpha=0.7, edgecolor="black")
    ax.set_xlabel("Pattern Frequency")
    ax.set_ylabel("Count")
    ax.set_title("Pattern Frequency Distribution")
    ax.grid(True, alpha=0.3, axis="y")
    
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    # Test the memory neuron
    print("Testing Memory Neuron...")
    
    # Simple XOR-like problem
    X_train = np.array([
        [0.0, 0.0],
        [0.0, 1.0],
        [1.0, 0.0],
        [1.0, 1.0],
    ])
    y_train = np.array([0.0, 1.0, 1.0, 0.0])
    
    neuron = MemoryNeuron(n_inputs=2, memory_size=50, learning_rate=0.1)
    
    print("\nTraining...")
    history = neuron.fit((X_train, y_train), epochs=50, verbose=True)
    
    print("\nMemory Info:", neuron.memory_info())
    print("Predictions:", neuron.predict_proba(X_train))
    print("Predicted Classes:", neuron.predict_class(X_train))
    print("Accuracy:", neuron.accuracy(X_train, y_train))
    
    # Visualize
    print("\nVisualizing memory...")
    visualize_memory(neuron)
