"""Level generation by tier, and the solvability check.

A level is a pure function of (tier, seed): the same pair always gives the same
course. Every level handed out has been solved by `solve`, a search over jump
timings that runs on the same physics as the game, so no failure the agent sees
is the course's fault.

Held-out evaluation seeds live below EVAL_SEED_LIMIT and training seeds above
it, so training can never draw an evaluation level.
"""

from __future__ import annotations

from collections import OrderedDict

import numpy as np

from .game import (
    DONE,
    DUCK,
    LONG,
    LOW,
    MOVING,
    NOOP,
    OVERHANG,
    OVERHANG_BOTTOM,
    OVERHANG_TOP,
    RUNNING,
    SHORT,
    START,
    TALL,
    GAP,
    Level,
    State,
    advance,
    airborne,
)

EVAL_SEED_LIMIT = 1 << 20
MAX_ATTEMPTS = 40


def difficulty(tier: int) -> dict:
    """The knobs for a tier. There is no top tier: the knobs keep turning until they hit their caps."""
    t = max(int(tier), 0)
    return {
        "speed": min(0.25 + 0.012 * t, 0.55),
        "count": min(10 + t, 30),
        "weights": {
            LOW: 1.0,
            GAP: 0.0 if t < 1 else 1.0,
            TALL: 0.0 if t < 2 else 0.8,
            OVERHANG: 0.0 if t < 3 else 0.8,
            MOVING: 0.0 if t < 4 else 0.6,
        },
        "spacing": (max(5.0, 12.0 - 0.5 * t), max(5.0, 12.0 - 0.5 * t) + 8.0),
        "combo_p": 0.0 if t < 5 else min(0.06 * (t - 4), 0.5),
        "combo_spacing": (2.5, 4.5),
        "gap_w": (1.5, min(2.0 + 0.15 * t, 4.5)),
        "low_w": (0.8, min(1.6 + 0.05 * t, 2.5)),
        "low_h": (0.8, 1.2),
        "tall_w": (0.8, 1.4),
        "tall_h": (2.4, min(2.6 + 0.05 * t, 3.4)),
        "overhang_w": (2.0, min(3.0 + 0.3 * t, 9.0)),
        "moving_u": (0.03, min(0.05 + 0.01 * t, 0.15)),
    }


def _build(tier: int, seed: int, attempt: int) -> Level:
    d = difficulty(tier)
    rng = np.random.default_rng([tier, seed, attempt])
    speed = d["speed"]
    kinds = [k for k, w in d["weights"].items() if w > 0]
    probs = np.array([d["weights"][k] for k in kinds])
    probs /= probs.sum()
    boxes, gaps = [], []
    pos = 25.0
    last_attempt = attempt == MAX_ATTEMPTS - 1
    for _ in range(d["count"]):
        k = kinds[rng.choice(len(kinds), p=probs)]
        if rng.random() < d["combo_p"] and not last_attempt:
            pos += rng.uniform(*d["combo_spacing"])
        else:
            lo, hi = d["spacing"]
            pos += rng.uniform(lo, hi) if not last_attempt else hi + 4.0
        if k == GAP:
            w = rng.uniform(*d["gap_w"])
            gaps.append((pos, pos + w))
        elif k == LOW:
            w = rng.uniform(*d["low_w"])
            boxes.append((LOW, pos, pos + w, 0.0, rng.uniform(*d["low_h"]), 0.0))
        elif k == TALL:
            w = rng.uniform(*d["tall_w"])
            boxes.append((TALL, pos, pos + w, 0.0, rng.uniform(*d["tall_h"]), 0.0))
        elif k == OVERHANG:
            w = rng.uniform(*d["overhang_w"])
            boxes.append((OVERHANG, pos, pos + w, OVERHANG_BOTTOM, OVERHANG_TOP, 0.0))
        else:
            # Placed so that it reaches `pos` at the tick the agent does.
            w = rng.uniform(*d["low_w"])
            u = rng.uniform(*d["moving_u"])
            x0 = pos * (speed + u) / speed
            boxes.append((MOVING, x0, x0 + w, 0.0, rng.uniform(*d["low_h"]), u))
        pos += w
    arr = np.array(boxes, dtype=np.float64).reshape(-1, 6)
    g = np.array(gaps, dtype=np.float64).reshape(-1, 2)
    return Level(
        tier=tier,
        seed=seed,
        speed=speed,
        length=pos + 20.0,
        kind=arr[:, 0].astype(np.int64),
        x0=arr[:, 1],
        x1=arr[:, 2],
        y0=arr[:, 3],
        y1=arr[:, 4],
        u=arr[:, 5],
        gx0=g[:, 0],
        gx1=g[:, 1],
    )


def _edge(level: Level, state: State, action: int):
    """Take `action`, then coast through any jump. None if the agent dies on the way."""
    actions = [action]
    st, out, _, _ = advance(level, state, action)
    while out == RUNNING and airborne(st):
        st, out, _, _ = advance(level, st, NOOP)
        actions.append(NOOP)
    if out == RUNNING or out == DONE:
        return st, actions, out == DONE
    return None


def solve(level: Level, max_expansions: int = 60_000) -> list[int] | None:
    """Depth-first search over decisions taken on the ground.

    On the ground the state is fully set by the tick (height and vertical speed
    are both 0, and moving boxes are a function of time), so each grounded tick
    is explored at most once. Trying NOOP first finds a plan in roughly one pass
    along the course; jumps are only tried where running straight on dies.
    """
    order = (NOOP, DUCK, SHORT, LONG)
    visited: set[int] = set()
    stack = [[START, 0, []]]
    expansions = 0
    while stack:
        frame = stack[-1]
        state, k, _ = frame
        if k >= len(order):
            stack.pop()
            continue
        frame[1] += 1
        expansions += 1
        if expansions > max_expansions:
            return None
        res = _edge(level, state, order[k])
        if res is None:
            continue
        new_state, actions, finished = res
        if finished:
            return [a for f in stack for a in f[2]] + actions
        if new_state[0] in visited:
            continue
        visited.add(new_state[0])
        stack.append([new_state, 0, actions])
    return None


def generate(tier: int, seed: int) -> Level:
    """The first solvable course for (tier, seed). The last attempt spreads obstacles out wide."""
    for attempt in range(MAX_ATTEMPTS):
        level = _build(tier, seed, attempt)
        if solve(level) is not None:
            return level
    raise RuntimeError(f"no solvable level for tier {tier}, seed {seed}")


def eval_seed(tier: int, i: int) -> int:
    return (tier * 9973 + i * 131 + 7) % EVAL_SEED_LIMIT


def train_seed(rng: np.random.Generator) -> int:
    return int(rng.integers(EVAL_SEED_LIMIT, 1 << 31))


class LevelCache:
    """Generating and solving a level takes milliseconds; replayed seeds come back often."""

    def __init__(self, size: int = 4096):
        self.size = size
        self._d: OrderedDict = OrderedDict()

    def get(self, tier: int, seed: int) -> Level:
        key = (tier, seed)
        lvl = self._d.get(key)
        if lvl is None:
            lvl = generate(tier, seed)
            self._d[key] = lvl
            if len(self._d) > self.size:
                self._d.popitem(last=False)
        else:
            self._d.move_to_end(key)
        return lvl
