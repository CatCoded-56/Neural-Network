import argparse
import time
import numpy as np
import matplotlib.pyplot as plt
from neuron import NeuralNetwork


def draw_network(ax, model, node_vals=None, node_radius=0.022):
    """Render a live architecture diagram showing node activations and weight strength."""
    layer_sizes = model.layer_sizes
    activation_names = getattr(model, "activation_names", [])
    
    # Layer positioning
    layer_x = [0.08 + 0.84 * i / (len(layer_sizes) - 1) for i in range(len(layer_sizes))]
    
    # Set background
    ax.set_facecolor("#f8f9fa")

    # Prepare richer colormap for node activation values
    cmap = plt.cm.plasma
    vmin, vmax = 0.0, 1.0
    if node_vals is not None:
        flattened = [v for layer in node_vals for v in np.ravel(layer).tolist()]
        if len(flattened) > 0:
            all_vals = np.array(flattened, dtype=float)
            vmin = float(np.min(all_vals))
            vmax = float(np.max(all_vals))
            if vmin == vmax:
                vmin -= 1.0
                vmax += 1.0

    # Draw connections (edges) with richer weight-based styling
    for layer_idx in range(1, len(layer_sizes)):
        prev_x = layer_x[layer_idx - 1]
        x = layer_x[layer_idx]
        W = None
        if hasattr(model, "weights"):
            try:
                W = model.weights[layer_idx - 1]
            except Exception:
                pass
        
        max_w = 1.0
        if W is not None and W.size > 0:
            max_w = float(np.max(np.abs(W))) or 1.0

        for i in range(layer_sizes[layer_idx - 1]):
            prev_y = 0.1 + 0.8 * i / (layer_sizes[layer_idx - 1] - 1) if layer_sizes[layer_idx - 1] > 1 else 0.5
            for j in range(layer_sizes[layer_idx]):
                y = 0.1 + 0.8 * j / (layer_sizes[layer_idx] - 1) if layer_sizes[layer_idx] > 1 else 0.5
                
                lw = 0.5
                color = "#d0d0d0"
                alpha = 0.4
                
                if W is not None:
                    w = float(W[i, j])
                    magnitude = abs(w) / max_w if max_w > 0 else 0.0
                    lw = 0.35 + 2.8 * magnitude
                    if w < 0:
                        red = int(255 * (0.35 + 0.65 * magnitude))
                        green = int(120 * (1.0 - magnitude))
                        blue = int(180 * (0.45 + 0.55 * magnitude))
                        color = (red / 255, green / 255, blue / 255)
                        alpha = 0.8
                    else:
                        red = int(140 * (0.4 + 0.6 * magnitude))
                        green = int(190 * (0.5 + 0.5 * magnitude))
                        blue = int(255 * (0.65 + 0.35 * magnitude))
                        color = (red / 255, green / 255, blue / 255)
                        alpha = 0.8
                
                ax.plot([prev_x + node_radius, x - node_radius], [prev_y, y], 
                       color=color, lw=lw, alpha=alpha, zorder=1)

    # Draw nodes
    for layer_idx, size in enumerate(layer_sizes):
        x = layer_x[layer_idx]
        vals = None
        if node_vals is not None and layer_idx < len(node_vals):
            vals = np.ravel(node_vals[layer_idx])
        
        # Layer label
        if layer_idx == 0:
            layer_label = "In"
        elif layer_idx == len(layer_sizes) - 1:
            layer_label = "Out"
        else:
            layer_label = f"H{layer_idx}"
        
        ax.text(x, -0.12, layer_label, ha="center", va="top", fontsize=8,
               weight="bold", color="#333333")
        
        for node_idx in range(size):
            y = 0.1 + 0.8 * node_idx / (size - 1) if size > 1 else 0.5
            
            # Determine node color with higher contrast and richer shading
            if vals is None:
                if layer_idx == 0:
                    face = "#34d399"  # bright green for input
                    edge_color = "#065f46"
                elif layer_idx == len(layer_sizes) - 1:
                    face = "#fbbf24"  # golden output
                    edge_color = "#78350f"
                else:
                    face = "#60a5fa"  # vivid blue hidden
                    edge_color = "#1d4ed8"
            else:
                v = float(vals[node_idx])
                norm = (v - vmin) / (vmax - vmin) if vmax > vmin else 0.5
                norm = np.clip(norm, 0.0, 1.0)
                face = cmap(norm)
                edge_color = "#111827"

            # Draw node circle
            circle = plt.Circle((x, y), node_radius, color=face, ec=edge_color, 
                              lw=1.2, zorder=3, alpha=0.9)
            ax.add_patch(circle)
            
            # Show activation function name for hidden/output layers (only once per layer)
            if layer_idx > 0 and node_idx == 0:
                act = activation_names[layer_idx - 1] if layer_idx - 1 < len(activation_names) else ""
                ax.text(x + 0.05, -0.18, f"[{act}]", ha="center", va="top",
                       fontsize=8, style="italic", color="#555555")

            # Show numeric activation value
            if vals is not None:
                v = float(vals[node_idx])
                ax.text(x, y + 0.005, f"{v:.2f}", ha="center", va="bottom", 
                       fontsize=7, color="#000000", weight="bold")

    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.28, 1.02)
    ax.axis("off")
    ax.set_title("Network Architecture", fontsize=12, weight="bold", pad=15)


def main():
    """Train a deeper feedforward network on a circular decision boundary with optional live plotting."""
    parser = argparse.ArgumentParser(description="Train the deeper neural network demo.")
    parser.add_argument("--live", action="store_true", help="Open the live training visualization and keep it open during training.")
    parser.add_argument("--headless", action="store_true", help="Skip the live plot and run quietly in non-GUI mode.")
    parser.add_argument("--epochs", type=int, default=12000, help="Maximum number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=64, help="Mini-batch size for SGD updates.")
    parser.add_argument("--update-every", type=int, default=25, help="How often to refresh the live training plot.")
    parser.add_argument("--patience", type=int, default=80, help="Stop early if validation-like loss is not improving.")
    args = parser.parse_args()

    # More complex dataset: random points with non-linear boundary
    rng = np.random.default_rng(42)
    n_samples = 7000
    X = rng.uniform(0, 1, size=(n_samples, 2))
    # Complex decision boundary: circle in 2D space
    y = (np.sqrt((X[:, 0] - 0.5)**2 + (X[:, 1] - 0.5)**2) < 0.3).astype(float)

    model = NeuralNetwork(
        layer_sizes=(2, 4, 6, 8, 12, 16, 12, 8, 4, 1),
        activations=("relu", "relu", "relu", "relu", "relu", "relu", "relu", "relu", "sigmoid"),
        loss="bce",
        learning_rate=0.02,
        rng=rng,
    )

    epochs = max(1, args.epochs)
    batch_size = max(1, args.batch_size)
    update_every = max(1, args.update_every)
    patience = max(1, args.patience)
    display_X = X[:64]

    backend_name = plt.get_backend().lower()
    live_mode = args.live or ("agg" not in backend_name and not args.headless)
    if args.headless:
        live_mode = False

    if live_mode:
        plt.ion()
        fig, (ax_net, ax_loss) = plt.subplots(1, 2, figsize=(12, 5))
        fig.patch.set_facecolor("#ffffff")
        losses = []
        line_loss, = ax_loss.plot([], [], color="#1f77b4", lw=2.5, label="Loss")
        ax_loss.set_xlabel("Epoch", fontsize=10, weight="bold")
        ax_loss.set_ylabel("Loss", fontsize=10, weight="bold")
        ax_loss.set_title("Training Loss", fontsize=12, weight="bold", pad=15)
        ax_loss.set_facecolor("#f8f9fa")
        ax_loss.grid(True, lw=0.4, alpha=0.3, color="#cccccc")
        ax_loss.legend(loc="upper right", fontsize=9)
        ax_loss.spines["top"].set_visible(False)
        ax_loss.spines["right"].set_visible(False)
    else:
        fig = None
        ax_net = None
        ax_loss = None
        losses = []
        line_loss = None

    n_samples = X.shape[0]
    best_loss = np.inf
    stale_epochs = 0

    print(f"Training network on {n_samples} samples with circle boundary problem...")
    print(f"Network: {model.layer_sizes} | Learning rate: {model.learning_rate} | Batch size: {batch_size}")
    print(f"Live plotting: {'enabled' if live_mode else 'disabled'}")
    for epoch in range(1, epochs + 1):
        idx = model.rng.permutation(n_samples)
        Xs = X[idx]
        ys = y[idx]
        epoch_losses = []
        for start in range(0, n_samples, batch_size):
            Xb = Xs[start : start + batch_size]
            yb = ys[start : start + batch_size]
            loss = model.train_step(Xb, yb)
            epoch_losses.append(loss)

        avg = float(np.mean(epoch_losses))
        losses.append(avg)

        if avg < best_loss - 1e-6:
            best_loss = avg
            stale_epochs = 0
        else:
            stale_epochs += 1

        if live_mode and (epoch % update_every == 0 or epoch == 1 or epoch == epochs or stale_epochs >= patience):
            ax_net.clear()
            try:
                activations, _ = model._forward_pass(display_X)
                node_vals = [np.mean(a, axis=0) for a in activations]
            except Exception:
                node_vals = None

            draw_network(ax_net, model, node_vals=node_vals)
            line_loss.set_data(range(len(losses)), losses)
            line_loss.set_label(f"Loss: {avg:.2f}")
            ax_loss.legend(loc="upper right", fontsize=9)
            ax_loss.relim()
            ax_loss.autoscale_view()
            fig.canvas.draw_idle()
            fig.canvas.flush_events()
            time.sleep(0.01)

        if stale_epochs >= patience:
            print(f"Early stopping at epoch {epoch} with loss {avg:.6f}")
            break

    if live_mode:
        plt.ioff()
        print("\nTraining complete. Keeping the plot open for live inspection.")
        try:
            plt.show(block=True)
        except Exception:
            print("The plot window is available, but the backend did not block the UI as expected.")
    else:
        print("Headless mode enabled; skipping live window display to reduce compute cost.")

    print("Probabilities:", model.predict_proba(X))
    print("Predicted classes:", model.predict_class(X))
    print("Accuracy:", model.accuracy(X, y))


if __name__ == "__main__":
    main()
