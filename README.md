# Neuron (single-unit)

This folder contains a minimal NumPy-backed single neuron implementation (`neuron.py`),
a small training demo (`train_and_eval.py`), and tests under `tests/`.

Quick start:

1. (Optional) create a virtual environment and install requirements:

```bash
python -m venv .venv
.
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

2. Run the demo:

```bash
python train_and_eval.py
```

3. Run an interactive neuron session:

```bash
python train_and_eval.py --demo neuron --interactive
```

4. Run the small XOR network demo:

```bash
python train_and_eval.py --demo network
```

5. Visualize the model structure:

```bash
python train_and_eval.py --demo network --visualize
```

6. If visualization does not open, the script will save a PNG file instead.

7. Run tests:

```bash
python -m pytest -q
```
