import numpy as np
from typing import Dict, Optional, Tuple, Union


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def sigmoid_derivative(logits: np.ndarray, output: np.ndarray) -> np.ndarray:
    return output * (1.0 - output)


def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, x)


def relu_derivative(logits: np.ndarray, output: np.ndarray) -> np.ndarray:
    return (logits > 0).astype(float)


def tanh(x: np.ndarray) -> np.ndarray:
    return np.tanh(x)


def tanh_derivative(logits: np.ndarray, output: np.ndarray) -> np.ndarray:
    return 1.0 - output ** 2


def linear(x: np.ndarray) -> np.ndarray:
    return x


def linear_derivative(logits: np.ndarray, output: np.ndarray) -> np.ndarray:
    return np.ones_like(output)


class Neuron:
    """A simple, vectorized single neuron.

    Supports forward inference, binary classification helpers, and basic gradient descent.
    """

    SUPPORTED_ACTIVATIONS = {"sigmoid", "relu", "tanh", "linear"}
    SUPPORTED_LOSSES = {"mse", "bce"}

    def __init__(
        self,
        n_inputs: int,
        weights: Optional[np.ndarray] = None,
        bias: float = 0.0,
        learning_rate: float = 0.1,
        activation: str = "sigmoid",
        loss: str = "mse",
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        if n_inputs < 1:
            raise ValueError("n_inputs must be at least 1")
        if activation not in self.SUPPORTED_ACTIVATIONS:
            raise ValueError(f"Unsupported activation: {activation}")
        if loss not in self.SUPPORTED_LOSSES:
            raise ValueError(f"Unsupported loss: {loss}")
        if loss == "bce" and activation != "sigmoid":
            raise ValueError("Binary cross-entropy loss requires sigmoid activation")

        self.n_inputs = n_inputs
        self.rng = rng or np.random.default_rng()
        self.activation_name = activation
        self.loss = loss
        self.learning_rate = float(learning_rate)
        self.bias = float(bias)
        self.history: Dict[str, list] = {"loss": []}

        if weights is None:
            limit = np.sqrt(6 / (n_inputs + 1))
            self.weights = self.rng.uniform(-limit, limit, size=(n_inputs,))
        else:
            self.weights = np.array(weights, dtype=float)
            if self.weights.shape != (n_inputs,):
                raise ValueError("weights must have shape (n_inputs,) if provided")

        if activation == "sigmoid":
            self._act = sigmoid
            self._act_deriv = sigmoid_derivative
        elif activation == "relu":
            self._act = relu
            self._act_deriv = relu_derivative
        elif activation == "tanh":
            self._act = tanh
            self._act_deriv = tanh_derivative
        else:
            self._act = linear
            self._act_deriv = linear_derivative

    def _validate_features(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        if X.ndim != 2 or X.shape[1] != self.n_inputs:
            raise ValueError(f"Input must have shape (n_samples, {self.n_inputs})")
        return X

    def forward(self, X: np.ndarray) -> np.ndarray:
        """Forward pass returning activation outputs.

        Accepts a single sample shape (n_features,) or batched shape (batch, n_features).
        """
        X = self._validate_features(X)
        logits = X @ self.weights + self.bias
        return self._act(logits).squeeze()

    def predict(self, x: np.ndarray) -> float:
        prob = self.forward(x)
        if np.ndim(prob) != 0:
            raise ValueError("predict expects a single feature vector, not a batch")
        return float(prob)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.forward(X)

    def predict_class(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        proba = self.predict_proba(X)
        preds = np.asarray(proba) >= float(threshold)
        return preds.astype(int)

    def accuracy(self, X: np.ndarray, y_true: np.ndarray, threshold: float = 0.5) -> float:
        y_true = np.asarray(y_true, dtype=float)
        y_pred = self.predict_class(X, threshold)
        if y_pred.ndim == 0:
            return float(y_pred == y_true)
        return float(np.mean(y_pred == y_true))

    def compute_loss(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        y_pred = np.asarray(y_pred, dtype=float)
        y_true = np.asarray(y_true, dtype=float)
        if y_pred.shape != y_true.shape:
            y_pred = y_pred.reshape(y_true.shape)
        if self.loss == "mse":
            return float(np.mean((y_true - y_pred) ** 2))
        if self.loss == "bce":
            eps = 1e-12
            p = np.clip(y_pred, eps, 1 - eps)
            return float(-np.mean(y_true * np.log(p) + (1 - y_true) * np.log(1 - p)))
        raise ValueError(f"Unsupported loss: {self.loss}")

    def train_step(self, X: np.ndarray, y: np.ndarray) -> float:
        X = self._validate_features(X)
        y = np.asarray(y, dtype=float).reshape(-1)
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have the same number of samples")

        logits = X @ self.weights + self.bias
        preds = self._act(logits)

        if self.loss == "mse":
            dL_dp = 2 * (preds - y)
        else:
            eps = 1e-12
            p = np.clip(preds, eps, 1 - eps)
            dL_dp = -(y / p) + ((1 - y) / (1 - p))

        dp_dz = self._act_deriv(logits, preds)
        dL_dz = dL_dp * dp_dz

        grad_w = np.mean(X * dL_dz.reshape(-1, 1), axis=0)
        grad_b = float(np.mean(dL_dz))

        self.weights -= self.learning_rate * grad_w
        self.bias -= self.learning_rate * grad_b

        return self.compute_loss(preds, y)

    def fit(
        self,
        training_data: Tuple[np.ndarray, np.ndarray],
        epochs: int = 1000,
        batch_size: int = 0,
        shuffle: bool = True,
        verbose: bool = False,
        early_stopping: Optional[int] = None,
    ) -> Dict[str, list]:
        X_all, y_all = training_data
        X_all = self._validate_features(X_all)
        y_all = np.asarray(y_all, dtype=float).reshape(-1)
        if X_all.shape[0] != y_all.shape[0]:
            raise ValueError("X and y must have the same number of samples")
        if epochs < 0:
            raise ValueError("epochs must be non-negative")

        n_samples = X_all.shape[0]
        if batch_size <= 0 or batch_size > n_samples:
            batch_size = n_samples

        self.history = {"loss": []}
        best_loss = float("inf")
        best_weights = self.weights.copy()
        best_bias = self.bias
        no_improve = 0

        for epoch in range(1, epochs + 1):
            if shuffle:
                idx = self.rng.permutation(n_samples)
                X_all = X_all[idx]
                y_all = y_all[idx]

            epoch_losses = []
            for start in range(0, n_samples, batch_size):
                Xb = X_all[start : start + batch_size]
                yb = y_all[start : start + batch_size]
                epoch_losses.append(self.train_step(Xb, yb))

            avg_loss = float(np.mean(epoch_losses))
            self.history["loss"].append(avg_loss)

            if verbose and epoch % max(1, epochs // 10) == 0:
                print(f"Epoch {epoch}: loss={avg_loss:.6f}")

            if early_stopping is not None:
                if avg_loss + 1e-12 < best_loss:
                    best_loss = avg_loss
                    best_weights = self.weights.copy()
                    best_bias = self.bias
                    no_improve = 0
                else:
                    no_improve += 1
                    if no_improve >= early_stopping:
                        if verbose:
                            print(f"Stopping early at epoch {epoch}")
                        self.weights = best_weights
                        self.bias = best_bias
                        break

        return self.history

    def save(self, path: str) -> None:
        np.savez(
            path,
            weights=self.weights,
            bias=self.bias,
            activation=self.activation_name,
            loss=self.loss,
        )

    @classmethod
    def load(cls, path: str, learning_rate: float = 0.1) -> "Neuron":
        data = np.load(path)
        w = data["weights"]
        b = float(data["bias"])
        activation = str(data["activation"].tolist()) if "activation" in data else "sigmoid"
        loss = str(data["loss"].tolist()) if "loss" in data else "mse"
        return cls(
            n_inputs=w.shape[0],
            weights=w,
            bias=b,
            learning_rate=learning_rate,
            activation=activation,
            loss=loss,
        )


def visualize_network(model, save_path: Optional[str] = None):
    try:
        import matplotlib
    except ImportError:
        print("matplotlib is required for graphical visualization. Showing ASCII diagram instead.")
        return visualize_network_ascii(model)

    backend = matplotlib.get_backend().lower()
    if backend == "agg":
        for candidate in ["tkagg", "qt5agg", "qt6agg", "wxagg", "gtk3agg", "gtk4agg"]:
            try:
                matplotlib.use(candidate, force=True)
                backend = matplotlib.get_backend().lower()
                break
            except Exception:
                continue

    noninteractive = {"agg", "pdf", "svg", "ps", "cairo"}
    try:
        import matplotlib.pyplot as plt
        from matplotlib.patches import Circle, FancyArrowPatch
    except ImportError:
        print("matplotlib.pyplot is not available. Showing ASCII diagram instead.")
        return visualize_network_ascii(model)

    if isinstance(model, Neuron):
        fig, ax = plt.subplots(figsize=(4, 3))
        ax.add_patch(Circle((0.5, 0.5), 0.15, fill=True, color="skyblue", ec="black"))
        ax.text(0.5, 0.5, "Neuron", ha="center", va="center")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")
        if backend in noninteractive:
            output_path = save_path or "neuron_visualization.png"
            fig.savefig(output_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"Non-interactive backend '{backend}' detected, saved diagram to {output_path}")
            return
        plt.show()
        return

    layer_x = [0.15 + 0.7 * i / (len(model.layer_sizes) - 1) for i in range(len(model.layer_sizes))]
    fig, ax = plt.subplots(figsize=(8, 4))
    for layer_idx, size in enumerate(model.layer_sizes):
        x = layer_x[layer_idx]
        for node_idx in range(size):
            y = 0.1 + 0.8 * node_idx / (size - 1) if size > 1 else 0.5
            circle = Circle((x, y), 0.05, fill=True, color="skyblue", ec="black")
            ax.add_patch(circle)
            ax.text(x, y, f"{layer_idx},{node_idx}", ha="center", va="center", fontsize=8)
            if layer_idx > 0:
                prev_x = layer_x[layer_idx - 1]
                for prev_idx in range(model.layer_sizes[layer_idx - 1]):
                    prev_y = 0.1 + 0.8 * prev_idx / (model.layer_sizes[layer_idx - 1] - 1) if model.layer_sizes[layer_idx - 1] > 1 else 0.5
                    arrow = FancyArrowPatch((prev_x + 0.05, prev_y), (x - 0.05, y), arrowstyle="-|>", mutation_scale=10, color="gray", lw=0.8)
                    ax.add_patch(arrow)
    ax.axis("off")
    plt.title("Neural Network Structure")
    if backend in noninteractive:
        output_path = save_path or "network_visualization.png"
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Non-interactive backend '{backend}' detected, saved diagram to {output_path}")
        return
    plt.show()


def visualize_network_ascii(model):
    if isinstance(model, Neuron):
        print("[Neuron] -> single unit with sigmoid activation")
        return

    print("Neural Network structure:")
    for i, size in enumerate(model.layer_sizes):
        if i == 0:
            print(f" Input layer ({size} nodes)")
        elif i == len(model.layer_sizes) - 1:
            print(f" Output layer ({size} nodes) [{model.activation_names[-1]}]")
        else:
            print(f" Hidden layer {i} ({size} nodes) [{model.activation_names[i-1]}]")


class NeuralNetwork:
    """A simple feed-forward neural network with dense layers."""

    def __init__(
        self,
        layer_sizes: Tuple[int, ...],
        activations: Optional[Union[str, Tuple[str, ...]]] = None,
        loss: str = "bce",
        learning_rate: float = 0.1,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        if len(layer_sizes) < 2:
            raise ValueError("layer_sizes must contain at least input and output sizes")
        self.layer_sizes = tuple(layer_sizes)
        self.n_layers = len(layer_sizes) - 1
        self.rng = rng or np.random.default_rng()
        self.loss = loss
        self.learning_rate = float(learning_rate)
        self.history: Dict[str, list] = {"loss": []}

        if loss not in Neuron.SUPPORTED_LOSSES:
            raise ValueError(f"Unsupported loss: {loss}")

        if activations is None:
            activations = tuple(
                "tanh" if i < self.n_layers - 1 else "sigmoid"
                for i in range(self.n_layers)
            )
        elif isinstance(activations, str):
            activations = tuple([activations] * self.n_layers)
        else:
            activations = tuple(activations)

        if len(activations) != self.n_layers:
            raise ValueError("activations must have one entry per layer")

        self.weights = []
        self.biases = []
        self.activations = []
        self.derivatives = []
        self.activation_names = []

        for in_dim, out_dim, activation in zip(
            self.layer_sizes[:-1], self.layer_sizes[1:], activations
        ):
            if activation not in Neuron.SUPPORTED_ACTIVATIONS:
                raise ValueError(f"Unsupported activation: {activation}")
            self.activation_names.append(activation)
            if activation == "sigmoid":
                self.activations.append(sigmoid)
                self.derivatives.append(sigmoid_derivative)
            elif activation == "relu":
                self.activations.append(relu)
                self.derivatives.append(relu_derivative)
            elif activation == "tanh":
                self.activations.append(tanh)
                self.derivatives.append(tanh_derivative)
            else:
                self.activations.append(linear)
                self.derivatives.append(linear_derivative)

            limit = np.sqrt(6 / (in_dim + out_dim))
            self.weights.append(
                self.rng.uniform(-limit, limit, size=(in_dim, out_dim))
            )
            self.biases.append(np.zeros(out_dim, dtype=float))

    def _validate_features(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        if X.ndim != 2 or X.shape[1] != self.layer_sizes[0]:
            raise ValueError(f"Input must have shape (n_samples, {self.layer_sizes[0]})")
        return X

    def forward(self, X: np.ndarray) -> np.ndarray:
        out = self._validate_features(X)
        for W, b, activation in zip(self.weights, self.biases, self.activations):
            out = activation(out @ W + b)
        return out.squeeze()

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.forward(X)

    def predict_class(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        proba = self.predict_proba(X)
        return (np.asarray(proba) >= float(threshold)).astype(int)

    def accuracy(self, X: np.ndarray, y_true: np.ndarray, threshold: float = 0.5) -> float:
        y_true = np.asarray(y_true, dtype=float).reshape(-1)
        y_pred = self.predict_class(X, threshold)
        return float(np.mean(y_pred == y_true))

    def compute_loss(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        y_pred = np.asarray(y_pred, dtype=float)
        y_true = np.asarray(y_true, dtype=float).reshape(-1)
        if y_pred.shape != y_true.shape:
            y_pred = y_pred.reshape(y_true.shape)
        if self.loss == "mse":
            return float(np.mean((y_true - y_pred) ** 2))
        eps = 1e-12
        p = np.clip(y_pred, eps, 1 - eps)
        return float(-np.mean(y_true * np.log(p) + (1 - y_true) * np.log(1 - p)))

    def _forward_pass(self, X: np.ndarray):
        activations = [X]
        logits = []
        out = X
        for W, b, activation in zip(self.weights, self.biases, self.activations):
            z = out @ W + b
            logits.append(z)
            out = activation(z)
            activations.append(out)
        return activations, logits

    def train_step(self, X: np.ndarray, y: np.ndarray) -> float:
        X = self._validate_features(X)
        y = np.asarray(y, dtype=float).reshape(-1)
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have the same number of samples")

        activations, logits = self._forward_pass(X)
        preds = activations[-1]
        y = y.reshape(preds.shape)

        if self.loss == "mse":
            dL_dout = 2 * (preds - y)
        else:
            if self.activation_names[-1] == "sigmoid":
                dL_dout = preds - y
            else:
                eps = 1e-12
                p = np.clip(preds, eps, 1 - eps)
                dL_dout = -(y / p) + ((1 - y) / (1 - p))

        for layer in reversed(range(self.n_layers)):
            z = logits[layer]
            a = activations[layer]
            deriv = self.derivatives[layer](z, activations[layer + 1])
            delta = dL_dout * deriv
            grad_w = a.T @ delta / X.shape[0]
            grad_b = np.mean(delta, axis=0)
            self.weights[layer] -= self.learning_rate * grad_w
            self.biases[layer] -= self.learning_rate * grad_b
            if layer > 0:
                dL_dout = delta @ self.weights[layer].T

        return self.compute_loss(preds, y)

    def fit(
        self,
        training_data: Tuple[np.ndarray, np.ndarray],
        epochs: int = 1000,
        batch_size: int = 0,
        shuffle: bool = True,
        verbose: bool = False,
        early_stopping: Optional[int] = None,
    ) -> Dict[str, list]:
        X_all, y_all = training_data
        X_all = self._validate_features(X_all)
        y_all = np.asarray(y_all, dtype=float).reshape(-1)
        if X_all.shape[0] != y_all.shape[0]:
            raise ValueError("X and y must have the same number of samples")
        if epochs < 0:
            raise ValueError("epochs must be non-negative")

        n_samples = X_all.shape[0]
        if batch_size <= 0 or batch_size > n_samples:
            batch_size = n_samples

        self.history = {"loss": []}
        best_loss = float("inf")
        best_weights = [w.copy() for w in self.weights]
        best_biases = [b.copy() for b in self.biases]
        no_improve = 0

        for epoch in range(1, epochs + 1):
            if shuffle:
                idx = self.rng.permutation(n_samples)
                X_all = X_all[idx]
                y_all = y_all[idx]

            epoch_losses = []
            for start in range(0, n_samples, batch_size):
                Xb = X_all[start : start + batch_size]
                yb = y_all[start : start + batch_size]
                epoch_losses.append(self.train_step(Xb, yb))

            avg_loss = float(np.mean(epoch_losses))
            self.history["loss"].append(avg_loss)

            if verbose and epoch % max(1, epochs // 10) == 0:
                print(f"Epoch {epoch}: loss={avg_loss:.6f}")

            if early_stopping is not None:
                if avg_loss + 1e-12 < best_loss:
                    best_loss = avg_loss
                    best_weights = [w.copy() for w in self.weights]
                    best_biases = [b.copy() for b in self.biases]
                    no_improve = 0
                else:
                    no_improve += 1
                    if no_improve >= early_stopping:
                        if verbose:
                            print(f"Stopping early at epoch {epoch}")
                        self.weights = [w.copy() for w in best_weights]
                        self.biases = [b.copy() for b in best_biases]
                        break

        return self.history
