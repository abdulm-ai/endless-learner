"""A multilayer perceptron with backpropagation and Adam, written out by hand.

Parameters are a flat list [W0, b0, W1, b1, ...]. Hidden layers use ReLU, the
output layer is linear. Everything the trainer needs is here: the forward pass,
the backward pass, the optimiser, the Huber loss and a numerical gradient check
that proves the backward pass is right.
"""

from __future__ import annotations

import numpy as np


def init_mlp(sizes: list[int], rng: np.random.Generator, dtype=np.float32) -> list[np.ndarray]:
    """He initialisation for ReLU layers; the output layer starts small so early Q-values sit near 0."""
    params = []
    n = len(sizes) - 1
    for i in range(n):
        fan_in, fan_out = sizes[i], sizes[i + 1]
        scale = np.sqrt(2.0 / fan_in) if i < n - 1 else np.sqrt(1.0 / fan_in) * 0.1
        params.append((rng.standard_normal((fan_in, fan_out)) * scale).astype(dtype))
        params.append(np.zeros(fan_out, dtype=dtype))
    return params


def init_layer(params: list[np.ndarray], layer: int, units: np.ndarray, rng: np.random.Generator) -> None:
    """Re-initialise hidden `units` of `layer`: fresh incoming weights, zero bias, zero outgoing weights.

    Zero outgoing weights mean the reset unit changes nothing until it has
    learned something, which is how continual backprop keeps resets harmless.
    """
    W, b = params[2 * layer], params[2 * layer + 1]
    W[:, units] = (rng.standard_normal((W.shape[0], len(units))) * np.sqrt(2.0 / W.shape[0])).astype(W.dtype)
    b[units] = 0.0
    params[2 * layer + 2][units, :] = 0.0


def forward(params: list[np.ndarray], x: np.ndarray) -> tuple[np.ndarray, list[np.ndarray]]:
    """Returns the output and the activations each layer received: acts[0] is x, acts[i] the ReLU output of layer i-1."""
    acts = [x]
    h = x
    n = len(params) // 2
    for i in range(n):
        z = h @ params[2 * i] + params[2 * i + 1]
        h = np.maximum(z, 0.0) if i < n - 1 else z
        acts.append(h)
    return h, acts


def backward(params: list[np.ndarray], acts: list[np.ndarray], dout: np.ndarray) -> list[np.ndarray]:
    """Gradients of the loss with respect to every parameter, given dLoss/dOutput."""
    n = len(params) // 2
    grads: list[np.ndarray] = [None] * len(params)  # type: ignore[list-item]
    g = dout
    for i in reversed(range(n)):
        grads[2 * i] = acts[i].T @ g
        grads[2 * i + 1] = g.sum(axis=0)
        if i > 0:
            g = (g @ params[2 * i].T) * (acts[i] > 0.0)
    return grads


def global_norm(grads: list[np.ndarray]) -> float:
    return float(np.sqrt(sum(float(np.dot(g.ravel(), g.ravel())) for g in grads)))


def clip_by_norm(grads: list[np.ndarray], max_norm: float) -> float:
    norm = global_norm(grads)
    if max_norm and norm > max_norm:
        s = max_norm / (norm + 1e-12)
        for g in grads:
            g *= s
    return norm


class Adam:
    """Kingma and Ba (2015), with bias correction. Updates parameters in place."""

    def __init__(self, params: list[np.ndarray], lr: float = 3e-4, b1: float = 0.9, b2: float = 0.999, eps: float = 1e-5):
        self.lr, self.b1, self.b2, self.eps = lr, b1, b2, eps
        self.m = [np.zeros_like(p) for p in params]
        self.v = [np.zeros_like(p) for p in params]
        self.t = 0

    def step(self, params: list[np.ndarray], grads: list[np.ndarray]) -> None:
        self.t += 1
        c1 = 1.0 - self.b1**self.t
        c2 = 1.0 - self.b2**self.t
        for p, g, m, v in zip(params, grads, self.m, self.v):
            m *= self.b1
            m += (1.0 - self.b1) * g
            v *= self.b2
            v += (1.0 - self.b2) * (g * g)
            p -= (self.lr / c1) * m / (np.sqrt(v / c2) + self.eps)

    def reset_units(self, layer: int, units: np.ndarray) -> None:
        """Forget the moment estimates of re-initialised units so they start like new ones."""
        for idx, sl in ((2 * layer, (slice(None), units)), (2 * layer + 1, (units,)), (2 * layer + 2, (units, slice(None)))):
            self.m[idx][sl] = 0.0
            self.v[idx][sl] = 0.0

    def state(self) -> dict:
        return {"m": self.m, "v": self.v, "t": self.t, "lr": self.lr}

    def load(self, s: dict) -> None:
        self.m, self.v, self.t, self.lr = s["m"], s["v"], s["t"], s["lr"]


def huber(delta: np.ndarray, k: float = 1.0) -> np.ndarray:
    a = np.abs(delta)
    return np.where(a <= k, 0.5 * delta * delta, k * (a - 0.5 * k))


def huber_grad(delta: np.ndarray, k: float = 1.0) -> np.ndarray:
    return np.clip(delta, -k, k)


def gradcheck(sizes: list[int], batch: int = 5, seed: int = 0, h: float = 1e-6) -> list[float]:
    """Relative error between backprop and central differences, per parameter array, in float64.

    The loss is a weighted Huber loss on random targets, the same shape of loss
    the agent trains on. Biases start at small random values so ReLU kinks are
    unlikely to sit exactly on a sample.
    """
    rng = np.random.default_rng(seed)
    params = init_mlp(sizes, rng, dtype=np.float64)
    for i in range(1, len(params), 2):
        params[i] += rng.standard_normal(params[i].shape) * 0.1
    params[-2] *= 10.0
    x = rng.standard_normal((batch, sizes[0]))
    target = rng.standard_normal((batch, sizes[-1]))
    w = rng.random((batch, 1)) + 0.5

    def loss() -> float:
        out, _ = forward(params, x)
        return float((w * huber(out - target)).sum())

    out, acts = forward(params, x)
    grads = backward(params, acts, w * huber_grad(out - target))
    errors = []
    for p, g in zip(params, grads):
        num = np.zeros_like(p)
        it = np.nditer(p, flags=["multi_index"])
        for _ in it:
            i = it.multi_index
            old = p[i]
            p[i] = old + h
            up = loss()
            p[i] = old - h
            down = loss()
            p[i] = old
            num[i] = (up - down) / (2 * h)
        denom = np.maximum(np.abs(num) + np.abs(g), 1e-7)
        errors.append(float((np.abs(num - g) / denom).max()))
    return errors
