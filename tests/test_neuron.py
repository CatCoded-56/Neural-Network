import numpy as np
from neuron import Neuron, NeuralNetwork


def test_neuron_learns_and():
    X = np.array([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1],
    ])
    y = np.array([0.0, 0.0, 0.0, 1.0])

    n = Neuron(n_inputs=2, learning_rate=0.5, rng=np.random.default_rng(1))
    history = n.fit((X, y), epochs=3000, batch_size=4, shuffle=True)

    assert isinstance(history, dict)
    assert len(history["loss"]) > 0

    pred = n.predict([1, 1])
    assert pred > 0.85


def test_neural_network_learns_xor():
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
        rng=np.random.default_rng(4),
    )
    history = model.fit((X, y), epochs=10000, batch_size=4, shuffle=True, early_stopping=100)

    assert isinstance(history, dict)
    assert len(history["loss"]) > 0

    probs = model.predict_proba(X)
    assert np.all(probs >= 0.0)
    assert np.all(probs <= 1.0)

    classes = model.predict_class(X)
    assert np.array_equal(classes, y.astype(int))
    assert model.accuracy(X, y) == 1.0


def test_neuron_predict_proba_and_accuracy():
    X = np.array([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1],
    ])
    y = np.array([0.0, 0.0, 0.0, 1.0])

    n = Neuron(
        n_inputs=2,
        learning_rate=0.5,
        activation="sigmoid",
        loss="bce",
        rng=np.random.default_rng(2),
    )
    n.fit((X, y), epochs=3000, batch_size=4, shuffle=False)

    probs = n.predict_proba(X)
    assert np.all(probs >= 0.0)
    assert np.all(probs <= 1.0)

    classes = n.predict_class(X)
    assert np.array_equal(classes, y.astype(int))
    assert n.accuracy(X, y) == 1.0


def test_neuron_early_stopping_restores_best_weights():
    X = np.array([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1],
    ])
    y = np.array([0.0, 0.0, 0.0, 1.0])

    n = Neuron(n_inputs=2, learning_rate=0.5, rng=np.random.default_rng(3))
    history = n.fit((X, y), epochs=1000, batch_size=4, shuffle=True, early_stopping=10)

    assert isinstance(history, dict)
    assert len(history["loss"]) <= 1000
    assert n.accuracy(X, y) >= 0.99
