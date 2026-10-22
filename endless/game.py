"""The game: a side-scrolling runner on a generated course.

The physics here is the single source of truth. The training environment, the
solvability check and the live view all move the world forward through
`advance`, so a level the solver can finish is a level the agent can finish.

Units: x and y in world units, time in ticks. The course scrolls at `speed`
units per tick, so the agent's left edge is at `t * speed`. The agent makes a
decision every `TICKS_PER_DECISION` ticks.
"""

from __future__ import annotations

import bisect
from dataclasses import dataclass, field

import numpy as np

TICKS_PER_DECISION = 2
GRAVITY = 0.04
AGENT_W = 1.0
STAND_H = 2.0
DUCK_H = 1.0

NOOP, SHORT, LONG, DUCK = range(4)
ACTIONS = ("noop", "short", "long", "duck")
N_ACTIONS = len(ACTIONS)
JUMP_SPEED = (0.0, 0.40, 0.58, 0.0)  # indexed by action: peaks near 2.2 and 4.5

LOW, TALL, OVERHANG, MOVING, GAP = range(5)
CAUSES = ("low", "tall", "overhang", "moving", "gap")
POSTURES = ("ground", "air", "duck")
OVERHANG_BOTTOM = 1.5  # above a ducking agent (1.0), below a standing one (2.0)
OVERHANG_TOP = 50.0

RUNNING, DEAD, DONE = range(3)

State = tuple[int, float, float]  # (tick, height of the agent's feet, vertical speed)
START: State = (0, 0.0, 0.0)


@dataclass
class Level:
    """One course. Obstacles are boxes; moving ones slide left at `u` units per tick."""

    tier: int
    seed: int
    speed: float
    length: float
    kind: np.ndarray  # (K,) LOW, TALL, OVERHANG or MOVING
    x0: np.ndarray  # box edges at tick 0
    x1: np.ndarray
    y0: np.ndarray
    y1: np.ndarray
    u: np.ndarray  # leftward speed, 0 for static boxes
    gx0: np.ndarray  # gaps in the ground, sorted
    gx1: np.ndarray
    _static: list = field(init=False, repr=False)
    _sx0: list = field(init=False, repr=False)
    _max_w: float = field(init=False, repr=False)
    _moving: list = field(init=False, repr=False)
    _gaps0: list = field(init=False, repr=False)
    _gaps1: list = field(init=False, repr=False)
    _passed_x1: list = field(init=False, repr=False)

    def __post_init__(self) -> None:
        # Plain Python lists for the per-tick collision test: with a handful of
        # nearby boxes, list walking beats NumPy's per-call overhead.
        boxes = [
            (float(a), float(b), float(c), float(d), int(k), float(v))
            for a, b, c, d, k, v in zip(self.x0, self.x1, self.y0, self.y1, self.kind, self.u)
        ]
        self._static = sorted((b[:5] for b in boxes if b[5] == 0.0), key=lambda b: b[0])
        self._sx0 = [b[0] for b in self._static]
        self._max_w = max((b[1] - b[0] for b in self._static), default=0.0)
        self._moving = [b for b in boxes if b[5] != 0.0]
        self._gaps0 = [float(g) for g in self.gx0]
        self._gaps1 = [float(g) for g in self.gx1]
        self._passed_x1 = sorted([b[1] for b in self._static] + self._gaps1)

    @property
    def n_features(self) -> int:
        return len(self.kind) + len(self.gx0)

    def hit(self, t: int, px: float, y: float, h: float) -> int:
        """Kind of the box the agent overlaps at tick t, or -1."""
        right, top = px + AGENT_W, y + h
        j = bisect.bisect_left(self._sx0, right) - 1
        lim = px - self._max_w
        static = self._static
        while j >= 0 and self._sx0[j] > lim:
            x0, x1, y0, y1, k = static[j]
            if x1 > px and y0 < top and y1 > y:
                return k
            j -= 1
        for x0, x1, y0, y1, k, u in self._moving:
            s = u * t
            if x0 - s < right and x1 - s > px and y0 < top and y1 > y:
                return k
        return -1

    def over_gap(self, x: float) -> bool:
        i = bisect.bisect_right(self._gaps0, x) - 1
        return i >= 0 and x < self._gaps1[i]

    def passed(self, t: int) -> int:
        """Obstacles and gaps fully behind the agent at tick t. Never decreases."""
        px = t * self.speed
        n = bisect.bisect_right(self._passed_x1, px)
        for x0, x1, y0, y1, k, u in self._moving:
            if x1 - u * t <= px:
                n += 1
        return n

    def to_json(self) -> dict:
        return {
            "tier": self.tier,
            "seed": self.seed,
            "speed": self.speed,
            "length": self.length,
            "boxes": [
                [int(k), round(float(a), 3), round(float(b), 3), round(float(c), 3), round(float(d), 3), round(float(v), 4)]
                for k, a, b, c, d, v in zip(self.kind, self.x0, self.x1, self.y0, self.y1, self.u)
            ],
            "gaps": [[round(float(a), 3), round(float(b), 3)] for a, b in zip(self.gx0, self.gx1)],
        }


def airborne(state: State) -> bool:
    return state[1] > 0.0 or state[2] > 0.0


def advance(level: Level, state: State, action: int, frames: list | None = None) -> tuple[State, int, int, int]:
    """Run one decision (TICKS_PER_DECISION ticks).

    Returns (state, outcome, cause, posture). Jumps start only from the ground
    and only on the first tick of a decision; ducking holds while the action is
    DUCK and the agent is on the ground. `frames`, if given, gets one
    (tick, y, ducking) entry per tick for drawing.
    """
    t, y, vy = state
    speed = level.speed
    posture = 0
    for i in range(TICKS_PER_DECISION):
        up = y > 0.0 or vy > 0.0
        if not up and i == 0 and (action == SHORT or action == LONG):
            vy = JUMP_SPEED[action]
            up = True
        t += 1
        landed = False
        if up:
            y += vy
            vy -= GRAVITY
            if y <= 0.0:
                y, vy, landed = 0.0, 0.0, True
        grounded = y <= 0.0
        duck = grounded and action == DUCK
        posture = 2 if duck else (1 if (landed or not grounded) else 0)
        px = t * speed
        if frames is not None:
            frames.append((t, y, duck))
        if grounded and level.over_gap(px + AGENT_W * 0.5):
            return (t, y, vy), DEAD, GAP, posture
        cause = level.hit(t, px, y, DUCK_H if duck else STAND_H)
        if cause >= 0:
            return (t, y, vy), DEAD, cause, posture
        if px >= level.length:
            return (t, y, vy), DONE, -1, posture
    return (t, y, vy), RUNNING, -1, posture
