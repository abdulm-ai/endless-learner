"""Double DQN on the hand-written network, plus continual backprop and plasticity measurements.

The target for a sampled n-step transition (s, a, R, s', d) is

    y = R + d * Q_target(s', argmax_a' Q_online(s', a'))

which is Double Q-learning (van Hasselt et al., 2016): the online network picks
the action and the target network values it, which removes most of the
overestimation of plain DQN (Mnih et al., 2015). d is gamma^n, or 0 when the
transition ends in a death or a finish.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import nn


@dataclass
class AgentConfig:
    hidden: tuple[int, ...] = (128, 128)
    lr: float = 1e-4
    actor_tau: float = 0.005  # EMA rate of the acting weights per update; 1.0 acts with the online net
    gamma: float = 0.99
    n_step: int = 3
    batch: int = 64
    target_every: int = 2000  # updates between hard target syncs
    grad_clip: float = 10.0
    cbp: bool = False  # continual backprop (Dohare et al., 2024)
    cbp_rate: float = 1e-4  # share of each hidden layer replaced per update
    cbp_maturity: int = 1000  # updates a unit is protected after a reset
    cbp_decay: float = 0.99  # running average for unit utility


class DQN:
    def __init__(self, obs_dim: int, n_actions: int, cfg: AgentConfig, rng: np.random.Generator):
        self.cfg = cfg
        self.sizes = [obs_dim, *cfg.hidden, n_actions]
        self.online = nn.init_mlp(self.sizes, rng)
        self.target = [p.copy() for p in self.online]
        self.actor = [p.copy() for p in self.online]
        self.opt = nn.Adam(self.online, lr=cfg.lr)
        self.updates = 0
        self.rng = rng
        n_hidden = len(cfg.hidden)
        self.utility = [np.zeros(h) for h in cfg.hidden]
        self.age = [np.zeros(h, dtype=np.int64) for h in cfg.hidden]
        self._replace_acc = [0.0] * n_hidden
        self.units_reset = 0

    def q(self, obs: np.ndarray) -> np.ndarray:
        return nn.forward(self.online, obs)[0]

    def act(self, obs: np.ndarray, eps: float) -> tuple[int, np.ndarray]:
        """Acts with the averaged weights, not the online ones.

        Every update flips the greedy action in some states (policy churn), and
        in a game where a jump one tick late is a death, an episode played by a
        network that changes a hundred times along the way does far worse than
        any frozen snapshot of it. An exponential moving average of the weights
        (Polyak averaging) moves slowly enough to play consistently.
        """
        q = nn.forward(self.actor, obs[None])[0][0]
        if self.rng.random() < eps:
            return int(self.rng.integers(len(q))), q
        return int(q.argmax()), q

    def learn(self, s, a, R, s2, disc, w) -> dict:
        B = len(a)
        rows = np.arange(B)
        # One online pass over s and s' together: half the calls, same arithmetic.
        q_both, acts = nn.forward(self.online, np.concatenate([s, s2]))
        q, a2 = q_both[:B], q_both[B:].argmax(axis=1)
        acts = [x[:B] for x in acts]
        q2 = nn.forward(self.target, s2)[0][rows, a2]
        y = R + disc * q2
        td = q[rows, a] - y
        dq = np.zeros_like(q)
        dq[rows, a] = nn.huber_grad(td) * w / B
        grads = nn.backward(self.online, acts, dq)
        norm = nn.clip_by_norm(grads, self.cfg.grad_clip)
        self.opt.step(self.online, grads)
        tau = self.cfg.actor_tau
        for pa, po in zip(self.actor, self.online):
            pa += tau * (po - pa)
        self.updates += 1
        if self.updates % self.cfg.target_every == 0:
            self.target = [p.copy() for p in self.online]
        if self.cfg.cbp:
            self._continual_backprop(acts)
        return {
            "td": td,
            "loss": float((w * nn.huber(td)).mean()),
            "grad_norm": norm,
            "q_mean": float(q.mean()),
            "q_max": float(q.max()),
        }

    def _continual_backprop(self, acts: list[np.ndarray]) -> None:
        """Replace a trickle of the least useful mature hidden units with fresh ones.

        Utility of unit i in layer l is a running average of
        |h_i| * sum_j |W_(l+1)[i, j]|: how much it contributes downstream.
        """
        c = self.cfg
        for layer in range(len(c.hidden)):
            h = np.abs(acts[layer + 1]).mean(axis=0)
            out_w = np.abs(self.online[2 * layer + 2]).sum(axis=1)
            self.utility[layer] = c.cbp_decay * self.utility[layer] + (1 - c.cbp_decay) * h * out_w
            self.age[layer] += 1
            self._replace_acc[layer] += c.cbp_rate * c.hidden[layer]
            n = int(self._replace_acc[layer])
            if n < 1:
                continue
            self._replace_acc[layer] -= n
            mature = np.nonzero(self.age[layer] > c.cbp_maturity)[0]
            if len(mature) == 0:
                continue
            units = mature[np.argsort(self.utility[layer][mature])[:n]]
            nn.init_layer(self.online, layer, units, self.rng)
            self.opt.reset_units(layer, units)
            self.utility[layer][units] = 0.0
            self.age[layer][units] = 0
            self.units_reset += len(units)

    def plasticity(self, s: np.ndarray) -> dict:
        """Dead units, feature rank and weight norm on a batch of states.

        A unit is dead if it outputs 0 for every state in the batch. srank is
        the number of singular values of the last hidden layer's features
        needed to cover 99% of their sum (Kumar et al., 2021).
        """
        _, acts = nn.forward(self.online, s)
        dead = [float((a.max(axis=0) <= 0.0).mean()) for a in acts[1:-1]]
        sv = np.linalg.svd(acts[-2].astype(np.float64), compute_uv=False)
        c = np.cumsum(sv) / max(sv.sum(), 1e-12)
        srank = int(np.searchsorted(c, 0.99) + 1)
        wnorm = float(np.sqrt(sum(float((p.astype(np.float64) ** 2).sum()) for p in self.online[0::2])))
        return {"dead_frac": float(np.mean(dead)), "dead_by_layer": dead, "srank": srank, "wnorm": wnorm}

    def state(self) -> dict:
        return {
            "online": self.online,
            "target": self.target,
            "actor": self.actor,
            "opt": self.opt.state(),
            "updates": self.updates,
            "utility": self.utility,
            "age": self.age,
            "replace_acc": self._replace_acc,
            "units_reset": self.units_reset,
        }

    def load(self, s: dict) -> None:
        self.online, self.target, self.actor = s["online"], s["target"], s["actor"]
        self.opt.load(s["opt"])
        self.updates = s["updates"]
        self.utility, self.age = s["utility"], s["age"]
        self._replace_acc = s["replace_acc"]
        self.units_reset = s["units_reset"]


def save_net(path: str, params: list[np.ndarray], meta: dict) -> None:
    """Small file with just the online weights, for the viewer and the hall of fame."""
    import io
    import json
    import os
    import time

    buf = io.BytesIO()
    np.savez(buf, *params, meta=np.array(json.dumps(meta)))
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        f.write(buf.getvalue())
    for _ in range(20):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:  # Windows: a reader has the file open for a moment
            time.sleep(0.05)
    os.replace(tmp, path)


def load_net(path: str) -> tuple[list[np.ndarray], dict]:
    import io
    import json

    with open(path, "rb") as f:
        data = np.load(io.BytesIO(f.read()))
    keys = sorted((k for k in data.files if k.startswith("arr_")), key=lambda k: int(k[4:]))
    return [data[k] for k in keys], json.loads(str(data["meta"]))
