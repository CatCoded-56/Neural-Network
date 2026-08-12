import time
import numpy as np
import matplotlib.pyplot as plt
from neuron import NeuralNetwork


def draw_network(ax, model, node_vals=None, node_radius=0.035):
    """Draw network. If `node_vals` is provided it should be a list of arrays
    (one per layer) containing a scalar activation per node (e.g. mean across samples).
    """
    layer_sizes = model.layer_sizes
    activation_names = getattr(model, "activation_names", [])
    layer_x = [0.1 + 0.8 * i / (len(layer_sizes) - 1) for i in range(len(layer_sizes))]

    # prepare colormap if node values are provided
    cmap = plt.cm.seismic
    all_vals = None
    if node_vals is not None:
        flattened = []
        for v in node_vals:
            flattened.extend(np.ravel(v).tolist())
        if len(flattened) > 0:
            all_vals = np.array(flattened, dtype=float)
            vmin = float(np.min(all_vals))
            vmax = float(np.max(all_vals))
            if vmin == vmax:
                vmin -= 1.0
                vmax += 1.0
        else:
            all_vals = None

    # draw connections (optionally styled by weights if present)
    for layer_idx in range(1, len(layer_sizes)):
        prev_x = layer_x[layer_idx - 1]
        x = layer_x[layer_idx]
        W = None
        if hasattr(model, "weights"):
            try:
                W = model.weights[layer_idx - 1]
            except Exception:
                W = None
        max_w = 1.0
        if W is not None and W.size > 0:
            max_w = float(np.max(np.abs(W))) or 1.0
        for i in range(layer_sizes[layer_idx - 1]):
            prev_y = 0.1 + 0.8 * i / (layer_sizes[layer_idx - 1] - 1) if layer_sizes[layer_idx - 1] > 1 else 0.5
            for j in range(layer_sizes[layer_idx]):
                y = 0.1 + 0.8 * j / (layer_sizes[layer_idx] - 1) if layer_sizes[layer_idx] > 1 else 0.5
                lw = 0.6
                color = "#999999"
                if W is not None:
                    w = float(W[i, j])
                    lw = 0.4 + 2.0 * (abs(w) / max_w)
                    color = "#c0392b" if w < 0 else "#2980b9"
                ax.plot([prev_x + node_radius, x - node_radius], [prev_y, y], color=color, lw=lw, alpha=0.8)

    # draw nodes with styling per layer; if node_vals provided, color by value
    for layer_idx, size in enumerate(layer_sizes):
        x = layer_x[layer_idx]
        vals = None
        if node_vals is not None and layer_idx < len(node_vals):
            vals = np.ravel(node_vals[layer_idx])
        for node_idx in range(size):
            y = 0.1 + 0.8 * node_idx / (size - 1) if size > 1 else 0.5
            if vals is None:
                if layer_idx == 0:
                    face = "#bdc3c7"  # input
                elif layer_idx == len(layer_sizes) - 1:
                    face = "#2ecc71"  # output
                else:
                    face = "#87CEEB"  # hidden
            else:
                # normalize per global vmin/vmax
                v = float(vals[node_idx])
                norm = (v - vmin) / (vmax - vmin)
                norm = np.clip(norm, 0.0, 1.0)
                face = cmap(norm)

            circle = plt.Circle((x, y), node_radius, color=face, ec="black", lw=0.6)
            ax.add_patch(circle)
            # show small label with activation for hidden/output layers
            if layer_idx > 0:
                act = activation_names[layer_idx - 1] if layer_idx - 1 < len(activation_names) else ""
                ax.text(x, y - node_radius - 0.02, f"{act}", ha="center", va="top", fontsize=6)
            # show numeric activation if available
            if vals is not None:
                v = float(vals[node_idx])
                ax.text(x + node_radius + 0.01, y, f"{v:.2f}", ha="left", va="center", fontsize=6)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_title("Network Structure")


def main():
    X = np.array([
        [0.0, 0.0],
        [0.0, 1.0],
        [1.0, 0.0],
        [1.0, 1.0],
    ])
    y = np.array([0.0, 1.0, 1.0, 0.0])

    model = NeuralNetwork(
        layer_sizes=(2, 8, 8, 1),
        activations=("tanh", "tanh", "sigmoid"),
        loss="bce",
        learning_rate=0.5,
        rng=np.random.default_rng(42),
    )

    epochs = 3000
    batch_size = 4
    update_every = 10

    plt.ion()
    fig, (ax_net, ax_loss) = plt.subplots(1, 2, figsize=(10, 4))
    losses = []
    line_loss, = ax_loss.plot([], [], color="tab:blue")
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Loss")
    ax_loss.set_title("Training Loss")
    ax_loss.grid(True, lw=0.3)

    n_samples = X.shape[0]
    print("Training deeper network for XOR with live updates...")
    for epoch in range(1, epochs + 1):
        # simple full-batch shuffle per epoch
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

        if epoch % update_every == 0 or epoch == 1 or epoch == epochs:
            ax_net.clear()
            # compute mean activations per node across the full dataset for coloring
            try:
                activations, logits = model._forward_pass(X)
                # activations[0] is input; include it so layers align with layer_sizes
                node_vals = [np.mean(a, axis=0) for a in activations]
            except Exception:
                node_vals = None

            draw_network(ax_net, model, node_vals=node_vals)
            line_loss.set_data(range(len(losses)), losses)
            ax_loss.relim()
            ax_loss.autoscale_view()
            fig.canvas.draw()
            fig.canvas.flush_events()
            time.sleep(0.001)

    plt.ioff()
    plt.show()

    print("Probabilities:", model.predict_proba(X))
    print("Predicted classes:", model.predict_class(X))
    print("Accuracy:", model.accuracy(X, y))


if __name__ == "__main__":
    main()
