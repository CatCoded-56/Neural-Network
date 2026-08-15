# Neural Network Playground

This project is a compact NumPy-based playground for learning how neural networks behave in practice. It includes:

- a single-unit perceptron-style neuron in [neuron.py](neuron.py)
- a deeper feedforward model in [deeper_network.py](deeper_network.py)
- a memory-based classifier in [memory_neuron.py](memory_neuron.py)
- a live comparison script in [compare_networks.py](compare_networks.py)
- a small training/demo CLI in [train_and_eval.py](train_and_eval.py)
- tests in [tests/test_neuron.py](tests/test_neuron.py)

The code is designed for learning, experimentation, and visual understanding of training behavior rather than production ML pipelines.

## Quick Start

Create and activate a virtual environment, then install dependencies:

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Run the demos

| Command | What it does |
|---|---|
| `python train_and_eval.py --demo neuron` | Trains a single neuron on a simple binary classification task |
| `python train_and_eval.py --demo neuron --interactive` | Lets you input feature values manually after training |
| `python train_and_eval.py --demo network` | Trains a tiny XOR network |
| `python train_and_eval.py --demo network --visualize` | Draws the network structure after training |
| `python deeper_network.py --live` | Runs the deeper network with live plotting and keeps the figure open while training |
| `python deeper_network.py --headless` | Runs the deeper network quietly without opening the GUI window |
| `python deeper_network.py --epochs 1200 --update-every 50` | Customizes depth training behavior for slower/finer monitoring |
| `python memory_neuron.py` | Trains and visualizes the memory-based neuron |
| `python compare_networks.py` | Compares a standard neural network against the memory neuron on the same task |
| `python -m pytest -q` | Runs the project test suite |

## Headless / non-GUI environments

These scripts use Matplotlib for visualization. For real-time training, use the live mode in a GUI-capable environment:

```powershell
python deeper_network.py --live
```

For a non-interactive terminal run, skip the visualizer:

```powershell
python deeper_network.py --headless
```

If you need a headless backend explicitly, you can also force Matplotlib to offload rendering:

```powershell
$env:MPLBACKEND="Agg"; python deeper_network.py --headless
```

## Project structure

- [neuron.py](neuron.py) — the core neuron and feedforward network implementation
- [deeper_network.py](deeper_network.py) — deeper network demo with live training visualization
- [memory_neuron.py](memory_neuron.py) — memory-based pattern classifier
- [compare_networks.py](compare_networks.py) — benchmark and comparison script
- [train_and_eval.py](train_and_eval.py) — lightweight CLI for running demos
- [tests/test_neuron.py](tests/test_neuron.py) — verification tests for learning behavior

## Included features

- single-neuron and deep-network training using NumPy
- ReLU, tanh, sigmoid, and linear activations
- BCE and MSE loss options
- mini-batch training, shuffling, and early stopping
- live architecture plotting for training intuition
- memory-based pattern recognition for comparison

## Notes

- The circular boundary task is intentionally more difficult than XOR and is a good fit for deeper models.
- The memory neuron is highly interpretable and useful for comparing learned examples to a standard neural network.
- The project is intended for teaching and experimentation, not as a full production ML framework.

## Verification

```bash
python -m pytest -q
```
