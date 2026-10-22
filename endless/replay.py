"""Replay memory: prioritised replay over recent experience, a reservoir over all of it, and n-step returns.

Prioritised replay (Schaul et al., 2016) samples transition i with probability
p_i^alpha / sum_j p_j^alpha, where p_i is its last TD error, and corrects the
bias with importance weights (N * P(i))^-beta. The sum tree below does both
sampling and updates for a whole batch at once, one tree level per NumPy call.

The reservoir keeps a uniform sample of every transition ever seen (Vitter's
algorithm R), so old tiers stay in the training data after the ring buffer
has overwritten them.
"""

from __future__ import annotations

from collections import deque

import numpy as np


class SumTree:
    def __init__(self, capacity: int):
        self.cap = 1 << int(np.ceil(np.log2(max(capacity, 2))))
        self.tree = np.zeros(2 * self.cap, dtype=np.float64)

    @property
    def total(self) -> float:
        return float(self.tree[1])

    def update(self, idx: np.ndarray, p: np.ndarray) -> None:
        node = np.asarray(idx, dtype=np.int64) + self.cap
        self.tree[node] = p
        node //= 2
        while node[0] >= 1:
            # Parents are recomputed from their children, so repeated indices in a batch are harmless.
            self.tree[node] = self.tree[2 * node] + self.tree[2 * node + 1]
            node //= 2

    def set(self, i: int, p: float) -> None:
        """One leaf, with a scalar walk up the tree: much cheaper than the batch path for a single add."""
        tree = self.tree
        node = i + self.cap
        tree[node] = p
        node //= 2
        while node >= 1:
            tree[node] = tree[2 * node] + tree[2 * node + 1]
            node //= 2

    def sample(self, n: int, rng: np.random.Generator, size: int) -> np.ndarray:
        """Stratified: one draw from each of n equal slices of the total mass."""
        seg = self.total / n
        u = (np.arange(n) + rng.random(n)) * seg
        node = np.ones(n, dtype=np.int64)
        while node[0] < self.cap:
            left = 2 * node
            lv = self.tree[left]
            right = u > lv
            u = np.where(right, u - lv, u)
            node = left + right
        return np.minimum(node - self.cap, size - 1)


class Replay:
    def __init__(self, capacity: int, obs_dim: int, alpha: float = 0.6, prioritised: bool = True, seed: int = 0):
        self.capacity = capacity
        self.alpha = alpha
        self.prioritised = prioritised
        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.act = np.zeros(capacity, dtype=np.int64)
        self.ret = np.zeros(capacity, dtype=np.float32)
        self.disc = np.zeros(capacity, dtype=np.float32)  # gamma^n, or 0 at a death or finish
        self.near_death = np.zeros(capacity, dtype=bool)
        self.tree = SumTree(capacity)
        self.max_p = 1.0
        self.pos = 0
        self.size = 0

    def add(self, s, a, r, s2, disc) -> int:
        i = self.pos
        self.obs[i], self.act[i], self.ret[i], self.next_obs[i], self.disc[i] = s, a, r, s2, disc
        self.near_death[i] = False
        self.tree.set(i, self.max_p**self.alpha)
        self.pos = (i + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
        return i

    def sample(self, n: int, beta: float, rng: np.random.Generator):
        if self.prioritised:
            idx = self.tree.sample(n, rng, self.size)
            p = self.tree.tree[idx + self.tree.cap] / self.tree.total
            w = (self.size * np.maximum(p, 1e-12)) ** (-beta)
            w = (w / w.max()).astype(np.float32)
        else:
            idx = rng.integers(0, self.size, n)
            w = np.ones(n, dtype=np.float32)
        return idx, w

    def update(self, idx: np.ndarray, td: np.ndarray) -> None:
        if not self.prioritised:
            return
        p = np.abs(td).astype(np.float64) + 1e-3
        self.max_p = max(self.max_p, float(p.max()))
        self.tree.update(idx, p**self.alpha)

    def batch(self, idx: np.ndarray):
        return self.obs[idx], self.act[idx], self.ret[idx], self.next_obs[idx], self.disc[idx]

    def state(self) -> dict:
        return {k: getattr(self, k) for k in ("obs", "next_obs", "act", "ret", "disc", "near_death", "max_p", "pos", "size")} | {
            "tree": self.tree.tree
        }

    def load(self, s: dict) -> None:
        for k, v in s.items():
            if k == "tree":
                self.tree.tree = v
            else:
                setattr(self, k, v)


class Reservoir:
    """A uniform sample over the whole history of transitions, at fixed size."""

    def __init__(self, capacity: int, obs_dim: int):
        self.capacity = capacity
        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.act = np.zeros(capacity, dtype=np.int64)
        self.ret = np.zeros(capacity, dtype=np.float32)
        self.disc = np.zeros(capacity, dtype=np.float32)
        self.seen = 0
        self.size = 0

    def offer(self, s, a, r, s2, disc, rng: np.random.Generator) -> None:
        self.seen += 1
        if self.size < self.capacity:
            i = self.size
            self.size += 1
        else:
            i = int(rng.integers(0, self.seen))
            if i >= self.capacity:
                return
        self.obs[i], self.act[i], self.ret[i], self.next_obs[i], self.disc[i] = s, a, r, s2, disc

    def sample(self, n: int, rng: np.random.Generator):
        idx = rng.integers(0, self.size, n)
        return self.obs[idx], self.act[idx], self.ret[idx], self.next_obs[idx], self.disc[idx]

    def state(self) -> dict:
        return {k: getattr(self, k) for k in ("obs", "next_obs", "act", "ret", "disc", "seen", "size")}

    def load(self, s: dict) -> None:
        for k, v in s.items():
            setattr(self, k, v)


class NStep:
    """Turns single steps into n-step transitions: R = r_0 + g r_1 + ... + g^(m-1) r_(m-1), bootstrapped with g^m."""

    def __init__(self, n: int, gamma: float):
        self.n, self.gamma = n, gamma
        self.buf: deque = deque()

    def push(self, s, a, r, s2, terminal: bool, truncated: bool = False) -> list[tuple]:
        self.buf.append((s, a, r))
        out = []
        if terminal or truncated:
            while self.buf:
                out.append(self._emit(s2, 0.0 if terminal else None))
                self.buf.popleft()
        elif len(self.buf) == self.n:
            out.append(self._emit(s2, None))
            self.buf.popleft()
        return out

    def _emit(self, s_last, disc_override):
        R, g = 0.0, 1.0
        for _, _, r in self.buf:
            R += g * r
            g *= self.gamma
        s0, a0, _ = self.buf[0]
        disc = g if disc_override is None else disc_override
        return s0, a0, R, s_last, disc

    def clear(self) -> None:
        self.buf.clear()
