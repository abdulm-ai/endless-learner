"""The curriculum that never ends, and which levels to replay inside a tier.

Tiers: the agent moves up when its exam success on the current tier passes
`promote` over a full window, and moves down only after a collapse: the tier
once went well (`collapse_from` or better over a window) and then fell under
`demote`. A new tier is supposed to be hard at first, and demoting on first
contact made the agent bounce between two tiers instead of learning the new
one. The window starts empty after every change.

Levels: prioritised level replay (Jiang, Grefenstette and Rocktaschel, 2021).
Each seed the agent has played keeps a score, the mean absolute one-step TD
error of its last episode: a level the agent mispredicts still has something to
teach. Seeds are replayed with probability rank^(-1/temperature), mixed with a
staleness term so no seed's score gets too old.
"""

from __future__ import annotations

from collections import deque

import numpy as np

from .levels import train_seed


class Curriculum:
    def __init__(
        self, window: int = 100, promote: float = 0.8, demote: float = 0.2, fixed_tier: int | None = None, collapse_from: float = 0.5
    ):
        self.window, self.promote, self.demote, self.collapse_from = window, promote, demote, collapse_from
        self.fixed_tier = fixed_tier
        self.tier = fixed_tier or 0
        self.results: deque = deque(maxlen=window)
        self.best_tier = self.tier
        self.best_rate = 0.0  # best full-window rate at the current tier

    def record(self, success: bool) -> str | None:
        if self.fixed_tier is not None:
            return None
        self.results.append(bool(success))
        if len(self.results) < self.window:
            return None
        rate = sum(self.results) / len(self.results)
        self.best_rate = max(self.best_rate, rate)
        if rate >= self.promote:
            self.tier += 1
            self.best_tier = max(self.best_tier, self.tier)
            self.results.clear()
            self.best_rate = 0.0
            return "promote"
        if rate < self.demote and self.tier > 0 and self.best_rate >= self.collapse_from:
            self.tier -= 1
            self.results.clear()
            self.best_rate = 0.0
            return "demote"
        return None

    @property
    def rate(self) -> float:
        return sum(self.results) / len(self.results) if self.results else 0.0

    def state(self) -> dict:
        return {"tier": self.tier, "results": list(self.results), "best_tier": self.best_tier, "best_rate": self.best_rate}

    def load(self, s: dict) -> None:
        self.tier, self.best_tier, self.best_rate = s["tier"], s["best_tier"], s.get("best_rate", 0.0)
        self.results = deque(s["results"], maxlen=self.window)


class LevelSampler:
    def __init__(
        self,
        rng: np.random.Generator,
        enabled: bool = True,
        replay_p: float = 0.5,
        pool: int = 500,
        temperature: float = 0.3,
        staleness: float = 0.1,
        min_pool: int = 20,
    ):
        self.rng = rng
        self.enabled = enabled
        self.replay_p, self.pool, self.temperature = replay_p, pool, temperature
        self.staleness, self.min_pool = staleness, min_pool
        self.pools: dict[int, dict[int, list]] = {}

    def sample(self, tier: int, episode: int) -> tuple[int, bool]:
        pool = self.pools.get(tier, {})
        if self.enabled and len(pool) >= self.min_pool and self.rng.random() < self.replay_p:
            seeds = list(pool)
            scores = np.array([pool[s][0] for s in seeds])
            last = np.array([pool[s][1] for s in seeds], dtype=np.float64)
            rank = np.empty(len(seeds))
            rank[np.argsort(-scores)] = np.arange(1, len(seeds) + 1)
            h = rank ** (-1.0 / self.temperature)
            p_score = h / h.sum()
            stale = episode - last
            p_stale = stale / stale.sum() if stale.sum() > 0 else np.full(len(seeds), 1 / len(seeds))
            p = (1 - self.staleness) * p_score + self.staleness * p_stale
            return seeds[int(self.rng.choice(len(seeds), p=p / p.sum()))], True
        return train_seed(self.rng), False

    def update(self, tier: int, seed: int, score: float, episode: int) -> None:
        pool = self.pools.setdefault(tier, {})
        pool[seed] = [float(score), episode]
        if len(pool) > self.pool:
            del pool[min(pool, key=lambda s: pool[s][0])]

    def seen(self, tier: int, n: int) -> list[int]:
        pool = list(self.pools.get(tier, {}))
        if len(pool) <= n:
            return pool
        return [pool[i] for i in self.rng.choice(len(pool), n, replace=False)]

    def state(self) -> dict:
        return {"pools": self.pools}

    def load(self, s: dict) -> None:
        self.pools = s["pools"]
