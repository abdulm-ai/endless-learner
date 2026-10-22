"""Running policies on levels: batched greedy evaluation, the two baselines, and recorded rollouts for the viewer.

Evaluation plays many levels in lockstep so the network sees one batch per
decision instead of one state at a time.
"""

from __future__ import annotations

import numpy as np

from . import nn
from .env import RunnerEnv
from .failure import signature
from .game import AGENT_W, DEAD, DONE, DUCK, GAP, GRAVITY, JUMP_SPEED, LONG, NOOP, OVERHANG, RUNNING, SHORT, TALL, Level


def run_levels(levels: list[Level], policy, max_decisions: int = 20_000, record: bool = False) -> list[dict]:
    """Play every level to the end with `policy(obs_batch, envs) -> actions`."""
    envs = [RunnerEnv(record=record) for _ in levels]
    obs = [env.reset(lv) for env, lv in zip(envs, levels)]
    results: list[dict | None] = [None] * len(levels)
    alive = list(range(len(levels)))
    for _ in range(max_decisions):
        if not alive:
            break
        acts = policy(np.stack([obs[i] for i in alive]), [envs[i] for i in alive])
        still = []
        for i, a in zip(alive, acts):
            obs[i], _, out, info = envs[i].step(int(a))
            if out == RUNNING:
                still.append(i)
                continue
            results[i] = {
                "success": out == DONE,
                "progress": envs[i].progress,
                "distance": envs[i].distance,
                "death_sig": signature(info["cause"], info["posture"]) if out == DEAD else None,
                "env": envs[i] if record else None,
            }
        alive = still
    for i in alive:
        results[i] = {"success": False, "progress": envs[i].progress, "distance": envs[i].distance, "death_sig": "timeout", "env": None}
    return results  # type: ignore[return-value]


def greedy(params: list[np.ndarray]):
    def policy(obs, envs):
        return nn.forward(params, obs)[0].argmax(axis=1)

    return policy


def random_policy(seed: int = 0):
    rng = np.random.default_rng(seed)

    def policy(obs, envs):
        return rng.integers(0, 4, len(envs))

    return policy


def _airtime(action: int) -> int:
    y, vy, n = 0.0, JUMP_SPEED[action], 0
    while True:
        y += vy
        vy -= GRAVITY
        n += 1
        if y <= 0.0:
            return n


AIRTIME = {SHORT: _airtime(SHORT), LONG: _airtime(LONG)}


def rule_action(env: RunnerEnv) -> int:
    """A hand-written player with full knowledge of the level: duck under overhangs, and
    time each jump so its arc is centred over the next obstacle. A scale for the learned agent."""
    level = env.level
    t, y, vy = env.state
    if y > 0.0 or vy > 0.0:
        return NOOP
    v = level.speed
    px = t * v
    best = None
    for k, x0, x1, u in zip(level.kind, level.x0, level.x1, level.u):
        a, b = x0 - u * t, x1 - u * t
        if b > px:
            eta = (a - px - AGENT_W) / (v + u)
            if best is None or eta < best[0]:
                best = (eta, int(k), a, b, u)
    for a, b in zip(level.gx0, level.gx1):
        if b > px:
            eta = (a - px - AGENT_W * 0.5) / v
            if best is None or eta < best[0]:
                best = (eta, GAP, a, b, 0.0)
    if best is None:
        return NOOP
    _, kind, a, b, u = best
    if kind == OVERHANG:
        return DUCK if a - (px + AGENT_W) < 0.6 else NOOP
    if kind == GAP:
        ref, span = px + AGENT_W * 0.5, (b - a) + 0.6
    else:
        ref, span = px + AGENT_W, (b - a) + AGENT_W
    closing = v + u
    jump = SHORT if kind != TALL and AIRTIME[SHORT] * closing > span + 1.0 else LONG
    lead = (AIRTIME[jump] * closing - span) / 2
    return jump if a - ref <= lead else NOOP


def rule_policy():
    def policy(obs, envs):
        return [rule_action(e) for e in envs]

    return policy


def rollout(params: list[np.ndarray], level: Level, max_decisions: int = 20_000) -> dict:
    """One greedy run with every tick recorded, for the live view and the ghost races."""
    res = run_levels([level], greedy(params), max_decisions=max_decisions, record=True)[0]
    env = res["env"]
    frames = [[t, round(y, 3), int(d)] for t, y, d in env.frames] if env else []
    rays = env.ray_log if env else []
    return {
        "success": res["success"],
        "progress": round(res["progress"], 4),
        "death_sig": res["death_sig"],
        "frames": frames,
        "rays": rays,
    }


def summarize(results: list[dict]) -> dict:
    prog = np.array([r["progress"] for r in results])
    return {
        "levels": len(results),
        "success": float(np.mean([r["success"] for r in results])) if results else 0.0,
        "progress_median": float(np.median(prog)) if len(prog) else 0.0,
        "progress_p90": float(np.percentile(prog, 90)) if len(prog) else 0.0,
    }
