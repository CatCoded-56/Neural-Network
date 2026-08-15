import argparse
import numpy as np
from neuron import Neuron, NeuralNetwork, visualize_network


def demo_neuron():
    """Train a single neuron on a simple binary classification problem and print the result."""
    X = np.array([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1],
    ])
    y = np.array([0.0, 0.0, 0.0, 1.0])

    neuron = Neuron(n_inputs=2, learning_rate=0.5)
    neuron.fit((X, y), epochs=2000, batch_size=4, verbose=True, early_stopping=200)

    print("\nFinal weights:", neuron.weights)
    print("Final bias:", neuron.bias)
    print("Prediction [1,1]:", neuron.predict([1, 1]))
    return neuron


def demo_network():
    """Train a tiny neural network on the XOR task and return the learned model."""
    X = np.array([
        [0.0, 0.0],
        [0.0, 1.0],
        [1.0, 0.0],
        [1.0, 1.0],
    ])
    y = np.array([0.0, 1.0, 1.0, 0.0])

    model = NeuralNetwork(
        layer_sizes=(2, 2, 1),
        activations=("tanh", "sigmoid"),
        loss="bce",
        learning_rate=0.5,
    )
    model.fit((X, y), epochs=10000, batch_size=4, shuffle=True, early_stopping=100)

    print("\nTrained network weights:")
    for idx, w in enumerate(model.weights, start=1):
        print(f" Layer {idx}:\n", w)
    print("Trained network biases:")
    for idx, b in enumerate(model.biases, start=1):
        print(f" Layer {idx}:", b)
    print("XOR predictions:", model.predict_class(X))
    return model


def interactive_session(model):
    """Prompt for feature inputs and print the model's probability and class prediction."""
    print("\nInteractive mode: enter comma-separated feature values, or 'q' to quit.")
    feature_count = model.n_inputs if isinstance(model, Neuron) else model.layer_sizes[0]
    while True:
        raw = input(f"Input {feature_count} values> ").strip()
        if raw.lower() in {"q", "quit", "exit"}:
            break
        try:
            values = [float(x) for x in raw.split(",")]
            if len(values) != feature_count:
                raise ValueError
            if isinstance(model, Neuron):
                prob = model.predict_proba(values)
            else:
                prob = model.predict_proba(np.array([values]))
            print("Probability:", prob)
            print("Class:", model.predict_class(np.array([values])) if not isinstance(model, Neuron) else model.predict_class(values))
        except ValueError:
            print(f"Please provide exactly {feature_count} numeric values separated by commas.")


def main():
    """CLI entry point for training the single-neuron and tiny-network demos."""
    parser = argparse.ArgumentParser(description="Train and inspect the neuron or small network.")
    parser.add_argument("--demo", choices=["neuron", "network"], default="neuron", help="Choose the demo to run")
    parser.add_argument("--visualize", action="store_true", help="Render a network diagram if possible")
    parser.add_argument("--interactive", action="store_true", help="Enter interactive input mode after training")
    args = parser.parse_args()

    if args.demo == "network":
        model = demo_network()
    else:
        model = demo_neuron()

    if args.visualize:
        visualize_network(model)

    if args.interactive:
        interactive_session(model)


if __name__ == "__main__":
    main()
