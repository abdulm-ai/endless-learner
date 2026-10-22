"""The training environment: the game plus what the agent sees and is rewarded for.

The agent sees a fan of ray casts from the front of its body. Each ray reports
how far away the first thing it hits is, what that thing is (ground, a pit,
a block or an overhang), and how fast that distance is closing, which is how
a moving block shows up. Five more numbers describe the agent itself.
"""

from __future__ import annotations

from collections import deque

import numpy as np

from .game import (
    AGENT_W,
    DEAD,
    DONE,
    DUCK_H,
    OVERHANG,
    RUNNING,
    STAND_H,
    START,
    TICKS_PER_DECISION,
    Level,
    State,
    advance,
)

RAY_ANGLES_DEG = (-50, -32, -18, -7, 3, 14, 27, 42, 60)
RAY_RANGE = 18.0
N_RAYS = len(RAY_ANGLES_DEG)
HIT_TYPES = ("ground", "pit", "block", "overhang")
OBS_DIM = N_RAYS * (2 + len(HIT_TYPES)) + 5

REWARD_PASS = 0.2
REWARD_DEATH = -1.0
REWARD_FINISH = 1.0

_ANG = np.deg2rad(np.array(RAY_ANGLES_DEG, dtype=np.float64))
_DX = np.cos(_ANG)
_DY = np.sin(_ANG)  # no ray is horizontal, so dy is never 0
_DOWN = _DY < 0


def cast_rays(level: Level, state: State, ducking: bool) -> tuple[np.ndarray, np.ndarray]:
    """Distance to the first hit along each ray (RAY_RANGE if none) and its type (-1 if none)."""
    t, y, _ = state
    px = t * level.speed
    ox = px + AGENT_W
    oy = y + (DUCK_H if ducking else STAND_H) * 0.5
    dist = np.full(N_RAYS, RAY_RANGE)
    kind = np.full(N_RAYS, -1, dtype=np.int64)

    # The ground plane, or the pit where a gap cuts through it.
    tg = np.where(_DOWN, -oy / _DY, np.inf)
    xg = ox + tg * _DX
    if len(level.gx0):
        pit = ((xg[:, None] > level.gx0) & (xg[:, None] < level.gx1)).any(axis=1)
    else:
        pit = np.zeros(N_RAYS, dtype=bool)
    g = tg <= RAY_RANGE
    dist[g] = tg[g]
    kind[g] = np.where(pit[g], 1, 0)

    # Boxes in range, by the slab method.
    if len(level.kind):
        shift = level.u * t
        X0 = level.x0 - shift
        X1 = level.x1 - shift
        near = (X1 > ox) & (X0 < ox + RAY_RANGE)
        if near.any():
            X0, X1 = X0[near], X1[near]
            Y0, Y1 = level.y0[near], level.y1[near]
            tx0 = (X0[None, :] - ox) / _DX[:, None]
            tx1 = (X1[None, :] - ox) / _DX[:, None]
            ty0 = (Y0[None, :] - oy) / _DY[:, None]
            ty1 = (Y1[None, :] - oy) / _DY[:, None]
            t_in = np.maximum(np.minimum(tx0, tx1), np.minimum(ty0, ty1))
            t_out = np.minimum(np.maximum(tx0, tx1), np.maximum(ty0, ty1))
            t_in = np.maximum(t_in, 0.0)
            hit = t_out >= t_in
            d = np.where(hit, t_in, np.inf)
            j = d.argmin(axis=1)
            best = d[np.arange(N_RAYS), j]
            closer = best < dist
            dist[closer] = best[closer]
            box_type = np.where(level.kind[near][j] == OVERHANG, 3, 2)
            kind[closer] = box_type[closer]
    return dist, kind


class RunnerEnv:
    """One level at a time. `reset` can start from any saved state, which is how the death bank restarts work."""

    def __init__(self, record: bool = False, history: int = 64):
        self.record = record
        self.history: deque[State] = deque(maxlen=history)
        self.level: Level | None = None
        self.frames: list = []
        self.ray_log: list = []

    def reset(self, level: Level, state: State = START) -> np.ndarray:
        self.level = level
        self.state = state
        self.posture = 0
        self.history.clear()
        self.passed = level.passed(state[0])
        self.frames = []
        self.ray_log = []
        self._prev = None
        return self._observe()

    def _observe(self) -> np.ndarray:
        level = self.level
        t, y, vy = self.state
        dist, kind = cast_rays(level, self.state, self.posture == 2)
        if self.record:
            self.ray_log.append((t, dist.round(2).tolist(), kind.tolist()))
        d = dist / RAY_RANGE
        onehot = np.zeros((N_RAYS, len(HIT_TYPES)))
        hit = kind >= 0
        onehot[np.nonzero(hit)[0], kind[hit]] = 1.0
        if self._prev is None:
            closing = np.zeros(N_RAYS)
        else:
            # In units of one decision's scroll: about -1 for anything standing still.
            closing = np.clip((d - self._prev) * RAY_RANGE / (level.speed * TICKS_PER_DECISION), -3.0, 3.0)
        self._prev = d
        grounded = 1.0 if (y <= 0.0 and vy <= 0.0) else 0.0
        me = np.array(
            [y / 4.5, vy / 0.6, grounded, 1.0 if self.posture == 2 else 0.0, (level.speed - 0.25) / 0.3]
        )
        return np.concatenate([d, onehot.ravel(), closing, me]).astype(np.float32)

    def step(self, action: int):
        """Returns (obs, reward, outcome, info). outcome is RUNNING, DEAD or DONE."""
        self.history.append(self.state)
        self.state, outcome, cause, self.posture = advance(
            self.level, self.state, int(action), self.frames if self.record else None
        )
        passed = self.level.passed(self.state[0])
        reward = REWARD_PASS * (passed - self.passed)
        self.passed = passed
        if outcome == DEAD:
            reward += REWARD_DEATH
        elif outcome == DONE:
            reward += REWARD_FINISH
        obs = self._observe() if outcome == RUNNING else np.zeros(OBS_DIM, dtype=np.float32)
        return obs, reward, outcome, {"cause": cause, "posture": self.posture}

    @property
    def progress(self) -> float:
        """Share of the course covered, 0 to 1."""
        return min(self.state[0] * self.level.speed / self.level.length, 1.0)

    @property
    def distance(self) -> float:
        return self.state[0] * self.level.speed
