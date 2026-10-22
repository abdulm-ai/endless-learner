"""The training loop that never stops.

One process plays, learns, evaluates and checkpoints. A run lives in one
folder: the checkpoint, the metrics database, the latest weights for the
viewer and the hall of fame. Starting `train` on a folder that has a checkpoint
carries on exactly where it stopped.

Time is counted in active seconds: wall-clock time while the loop runs, with
any single gap capped, so a laptop that sleeps mid-run doesn't count its sleep.
"""

from __future__ import annotations

import json
import os
import pickle
import time
from collections import deque
from dataclasses import asdict, dataclass, field

import numpy as np

from .agent import DQN, AgentConfig, save_net
from .curriculum import Curriculum, LevelSampler
from .env import OBS_DIM, RunnerEnv
from .evaluate import greedy, run_levels, summarize
from .failure import PASS_MARGIN, DeathBank, signature
from .game import DEAD, DONE, N_ACTIONS, RUNNING, START
from .levels import LevelCache, eval_seed
from .metrics import MetricsDB, ProcessStats

CHECKPOINT_VERSION = 1
MAX_GAP_S = 5.0


@dataclass
class TrainConfig:
    seed: int = 0
    agent: AgentConfig = field(default_factory=AgentConfig)
    replay_size: int = 100_000
    per: bool = True  # prioritised replay
    per_alpha: float = 0.6
    beta0: float = 0.4
    beta_steps: int = 1_000_000
    warmup: int = 5_000
    train_every: int = 4
    eps_start: float = 1.0
    eps_floor: float = 0.03
    eps_steps: int = 100_000
    restarts: bool = True  # death bank restarts
    restart_p: float = 0.25
    old_tiers: bool = True
    old_tier_p: float = 0.15
    reservoir: bool = True
    reservoir_size: int = 30_000
    reservoir_share: float = 0.25
    curriculum: bool = True
    fixed_tier: int = 6  # the tier used when the curriculum is off
    plr: bool = True  # prioritised level replay
    exam_p: float = 0.3  # share of fresh levels played greedily; only these count for promotion
    window: int = 50
    promote: float = 0.8
    demote: float = 0.2
    eval_every: int = 200_000
    eval_levels: int = 20
    frontier_success: float = 0.8
    log_every: int = 5_000
    plasticity_every: int = 50_000
    checkpoint_every_s: float = 300.0
    latest_every_s: float = 30.0
    hof_every_s: float = 3600.0

    @staticmethod
    def from_dict(d: dict) -> "TrainConfig":
        d = dict(d)
        a = d.pop("agent", {})
        a["hidden"] = tuple(a.get("hidden", AgentConfig.hidden))
        return TrainConfig(agent=AgentConfig(**a), **d)


class Trainer:
    def __init__(self, run_dir: str, cfg: TrainConfig | None = None, log=print):
        self.dir = run_dir
        self.log = log
        os.makedirs(os.path.join(run_dir, "hof"), exist_ok=True)
        self.ckpt_path = os.path.join(run_dir, "checkpoint.pkl")
        resuming = os.path.exists(self.ckpt_path)
        if resuming:
            with open(os.path.join(run_dir, "config.json")) as f:
                cfg = TrainConfig.from_dict(json.load(f))
        cfg = cfg or TrainConfig()
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)
        self.agent = DQN(OBS_DIM, N_ACTIONS, cfg.agent, self.rng)
        from .replay import NStep, Replay, Reservoir

        self.replay = Replay(cfg.replay_size, OBS_DIM, cfg.per_alpha, cfg.per)
        self.reservoir = Reservoir(cfg.reservoir_size, OBS_DIM)
        self.nstep = NStep(cfg.agent.n_step, cfg.agent.gamma)
        self.bank = DeathBank()
        self.curr = Curriculum(cfg.window, cfg.promote, cfg.demote, None if cfg.curriculum else cfg.fixed_tier)
        self.sampler = LevelSampler(self.rng, enabled=cfg.plr)
        self.cache = LevelCache()
        self.env = RunnerEnv(history=64)
        self.steps = 0
        self.episodes = 0
        self.active_s = 0.0
        self.eval_round = 0
        self.frontier = -1
        self.nan_events = 0
        self.next_hof = 0
        self.tier_since = 0  # steps when the current tier started
        self.db = MetricsDB(os.path.join(run_dir, "metrics.sqlite"))
        if resuming:
            self._load()
            lost = self.db.truncate_after(self.steps)
            self._event("resume", {"lost_steps": lost})
            self.log(f"resumed at {self.steps:,} steps, {self.active_s / 3600:.2f} active hours; {lost:,} steps lost since the last checkpoint")
        else:
            with open(os.path.join(run_dir, "config.json"), "w") as f:
                json.dump(asdict(cfg), f, indent=2)
            self._event("start", asdict(cfg))
            self._save_hof()
            # A checkpoint at step 0, so a crash before the first scheduled one still resumes
            # and the lost work gets counted, instead of a silent fresh start.
            self.checkpoint()
            self.db.flush(force=True)
        self.last_good = [p.copy() for p in self.agent.online]
        self.stats = ProcessStats()
        self._acc = self._fresh_acc()

    # ---- time and schedules -------------------------------------------------

    def epsilon(self) -> float:
        c = self.cfg
        f = min(self.steps / c.eps_steps, 1.0)
        return c.eps_start + f * (c.eps_floor - c.eps_start)

    def beta(self) -> float:
        c = self.cfg
        return c.beta0 + min(self.steps / c.beta_steps, 1.0) * (1.0 - c.beta0)

    def _clock(self) -> None:
        now = time.monotonic()
        self.active_s += min(now - self._last, MAX_GAP_S)
        self._last = now

    @staticmethod
    def _fresh_acc() -> dict:
        return {"loss": 0.0, "td": 0.0, "n": 0, "nd": 0.0, "nd_n": 0, "rt": 0.0, "rt_n": 0, "q_mean": 0.0, "q_max": -1e9, "grad": 0.0, "updates": 0}

    # ---- the loop -----------------------------------------------------------

    def train(self, max_steps: int | None = None, max_hours: float | None = None) -> None:
        self._last = time.monotonic()
        self._interval_t = self._last
        self._interval_steps = self.steps
        self._interval_updates = self.agent.updates
        self._next_latest = self.active_s
        self._next_ckpt = self.active_s + self.cfg.checkpoint_every_s
        self._next_status = time.monotonic()
        stop_file = os.path.join(self.dir, "STOP")
        self.stop_reason = None
        try:
            while self.stop_reason is None:
                self._episode()
                if max_steps is not None and self.steps >= max_steps:
                    self.stop_reason = "max_steps"
                elif max_hours is not None and self.active_s >= max_hours * 3600:
                    self.stop_reason = "max_hours"
                elif os.path.exists(stop_file):
                    os.remove(stop_file)
                    self.stop_reason = "stop_file"
        except KeyboardInterrupt:
            self.stop_reason = "interrupted"
        finally:
            self._event("stop", {"reason": self.stop_reason})
            self.checkpoint()
            self._save_latest()
            self.db.close()
            self.log(f"stopped ({self.stop_reason}) at {self.steps:,} steps; checkpoint saved")

    def _choose(self):
        c, r = self.cfg, self.rng.random()
        tier = self.curr.tier
        if c.restarts and len(self.bank) and r < c.restart_p:
            e = self.bank.sample(self.rng)
            return "restart", e.tier, e.seed, False, e
        if c.old_tiers and tier > 0 and r < c.restart_p + c.old_tier_p:
            t = int(self.rng.integers(0, tier))
            seed, replayed = self.sampler.sample(t, self.episodes)
            return "old", t, seed, replayed, None
        seed, replayed = self.sampler.sample(tier, self.episodes)
        # Exams: unseen levels played with no exploration. Random actions kill often enough
        # that exploring episodes understate what the policy can do, so promotion uses exams.
        if not replayed and self.rng.random() < c.exam_p:
            return "exam", tier, seed, False, None
        return "fresh", tier, seed, replayed, None

    def _episode(self) -> None:
        c, agent, env = self.cfg, self.agent, self.env
        kind, tier, seed, replayed, entry = self._choose()
        level = self.cache.get(tier, seed)
        obs = env.reset(level, entry.state if entry else START)
        self.nstep.clear()
        recent = deque(maxlen=self.bank.rewind)
        cut = entry.death_tick + PASS_MARGIN if entry else None
        gamma = c.agent.gamma
        td_sum, td_n, prev, ret, decisions = 0.0, 0, None, 0.0, 0
        eps = 0.0 if kind == "exam" else self.epsilon()
        while True:
            a, q = agent.act(obs, eps)
            if prev is not None:
                td_sum += abs(prev[1] + gamma * float(q.max()) - prev[0])
                td_n += 1
            obs2, r, out, info = env.step(a)
            ret += r
            decisions += 1
            self.steps += 1
            terminal = out != RUNNING
            truncated = not terminal and cut is not None and env.state[0] >= cut
            if terminal:
                td_sum += abs(r - float(q[a]))
                td_n += 1
                prev = None
            else:
                prev = (float(q[a]), r)
            for tr in self.nstep.push(obs, a, r, obs2, terminal, truncated):
                recent.append(self.replay.add(*tr))
                if c.reservoir:
                    self.reservoir.offer(*tr, self.rng)
            obs = obs2
            self._clock()
            if self.steps >= c.warmup and self.steps % c.train_every == 0:
                self._learn()
            if self.steps % c.log_every == 0:
                self._periodic()
                if kind != "exam":
                    eps = self.epsilon()
            if terminal or truncated:
                break

        sig = signature(info["cause"], info["posture"]) if out == DEAD else None
        success = out == DONE
        mean_td = td_sum / max(td_n, 1)
        if out == DEAD:
            for i in recent:
                self.replay.near_death[i] = True
        if entry is not None:
            passed = success or truncated
            outcome = self.bank.report(entry, passed)
            self.db.log(
                "restarts", active_s=self.active_s, steps=self.steps, tier=entry.tier, signature=entry.signature,
                passed=int(passed), attempt=entry.attempts, outcome=outcome,
            )
        else:
            if out == DEAD and c.restarts:
                self.bank.add(tier, seed, list(env.history), env.state[0], sig, self.episodes)
            self.sampler.update(tier, seed, mean_td, self.episodes)
            if kind == "exam":
                change = self.curr.record(success)
                if change:
                    self._event(change, {"tier": self.curr.tier, "steps_at_previous_tier": self.steps - self.tier_since})
                    self.tier_since = self.steps
                    self.log(f"{change} to tier {self.curr.tier} at {self.steps:,} steps")
        self.db.log(
            "episodes", wall=time.time(), active_s=self.active_s, steps=self.steps, episode=self.episodes,
            tier=tier, seed=seed, kind=kind, replayed=int(replayed), success=int(success),
            progress=env.progress, distance=env.distance, decisions=decisions, ret=ret,
            death_sig=sig, mean_td=mean_td, eps=eps,
        )
        self.episodes += 1

    def _learn(self) -> None:
        c = self.cfg
        B = c.agent.batch
        n_res = int(B * c.reservoir_share) if c.reservoir and self.reservoir.size >= B else 0
        idx, w = self.replay.sample(B - n_res, self.beta(), self.rng)
        s, a, R, s2, d = self.replay.batch(idx)
        if n_res:
            rs, ra, rR, rs2, rd = self.reservoir.sample(n_res, self.rng)
            s, a, R, s2, d = (np.concatenate(x) for x in ((s, rs), (a, ra), (R, rR), (s2, rs2), (d, rd)))
            w = np.concatenate([w, np.ones(n_res, dtype=np.float32)])
        out = self.agent.learn(s, a, R, s2, d, w)
        td = out["td"]
        if not np.isfinite(out["loss"]) or not all(np.isfinite(p).all() for p in self.agent.online):
            self._recover()
            return
        k = B - n_res
        self.replay.update(idx, td[:k])
        acc = self._acc
        acc["loss"] += out["loss"]
        acc["td"] += float(np.abs(td).mean())
        nd = self.replay.near_death[idx]
        acc["nd"] += float(np.abs(td[:k][nd]).sum())
        acc["nd_n"] += int(nd.sum())
        acc["rt"] += float(np.abs(td[:k][~nd]).sum())
        acc["rt_n"] += int((~nd).sum())
        acc["q_mean"] += out["q_mean"]
        acc["q_max"] = max(acc["q_max"], out["q_max"])
        acc["grad"] += out["grad_norm"]
        acc["updates"] += 1

    def _recover(self) -> None:
        """A NaN or inf in the loss or the weights: go back to the last good weights and restart the optimiser state."""
        self.nan_events += 1
        self.agent.online = [p.copy() for p in self.last_good]
        self.agent.target = [p.copy() for p in self.last_good]
        self.agent.actor = [p.copy() for p in self.last_good]
        for m, v in zip(self.agent.opt.m, self.agent.opt.v):
            m[...] = 0.0
            v[...] = 0.0
        self._event("nan_recovered", {"count": self.nan_events})

    # ---- periodic work ------------------------------------------------------

    def _periodic(self) -> None:
        c = self.cfg
        now = time.monotonic()
        dt = max(now - self._interval_t, 1e-9)
        acc = self._acc
        n = max(acc["updates"], 1)
        sps = (self.steps - self._interval_steps) / dt
        ups = (self.agent.updates - self._interval_updates) / dt
        self.db.log(
            "train", wall=time.time(), active_s=self.active_s, steps=self.steps, updates=self.agent.updates,
            tier=self.curr.tier, loss=acc["loss"] / n, td_abs=acc["td"] / n,
            td_near_death=acc["nd"] / acc["nd_n"] if acc["nd_n"] else None,
            td_routine=acc["rt"] / acc["rt_n"] if acc["rt_n"] else None,
            q_mean=acc["q_mean"] / n, q_max=acc["q_max"] if acc["updates"] else None, grad_norm=acc["grad"] / n,
            eps=self.epsilon(), beta=self.beta(), sps=sps, ups=ups, cpu=self.stats.cpu_percent(),
            rss_mb=self.stats.rss_mb(), nan_events=self.nan_events, bank_size=len(self.bank),
            bank_fixed=self.bank.fixed, bank_gave_up=self.bank.gave_up, units_reset=self.agent.units_reset,
        )
        self._interval_t, self._interval_steps, self._interval_updates = now, self.steps, self.agent.updates
        self.last_good = [p.copy() for p in self.agent.online]
        if self.steps % c.plasticity_every == 0 and self.replay.size >= 1024:
            s = self.replay.obs[self.rng.integers(0, self.replay.size, 1024)]
            p = self.agent.plasticity(s)
            dl = p["dead_by_layer"] + [None, None]
            self.db.log(
                "plasticity", active_s=self.active_s, steps=self.steps, dead_frac=p["dead_frac"],
                dead_l0=dl[0], dead_l1=dl[1], srank=p["srank"], wnorm=p["wnorm"],
            )
        if self.steps % c.eval_every == 0:
            self.evaluate()
        if self.active_s >= self._next_latest:
            self._save_latest()
            self._next_latest = self.active_s + c.latest_every_s
        if self.active_s >= self.next_hof * c.hof_every_s:
            self._save_hof()
        if self.active_s >= self._next_ckpt:
            self.checkpoint()
            self._next_ckpt = self.active_s + c.checkpoint_every_s
        self.db.flush()
        if now >= self._next_status:
            self._next_status = now + 30.0
            self.log(
                f"[{self.active_s / 3600:6.2f} h] steps {self.steps:>10,}  tier {self.curr.tier} "
                f"(exam success {self.curr.rate:.2f})  frontier {self.frontier}  eps {self.epsilon():.3f}  "
                f"{sps:,.0f} steps/s  bank {len(self.bank)} (fixed {self.bank.fixed})  loss {acc['loss'] / n:.4f}"
            )
        self._acc = self._fresh_acc()

    def evaluate(self) -> int:
        """Greedy play on fixed held-out seeds for every tier up to one past the best, plus seen training seeds."""
        c = self.cfg
        t0 = time.monotonic()
        self.eval_round += 1
        top = max(self.curr.tier, self.curr.best_tier) + 1
        levels, tags = [], []
        for tier in range(top + 1):
            for i in range(c.eval_levels):
                levels.append(self.cache.get(tier, eval_seed(tier, i)))
                tags.append((tier, "heldout"))
            for seed in self.sampler.seen(tier, c.eval_levels):
                levels.append(self.cache.get(tier, seed))
                tags.append((tier, "train"))
        results = run_levels(levels, greedy(self.agent.actor))
        groups: dict = {}
        for tag, r in zip(tags, results):
            groups.setdefault(tag, []).append(r)
        frontier = -1
        for (tier, split), rs in sorted(groups.items()):
            s = summarize(rs)
            self.db.log("evals", round=self.eval_round, active_s=self.active_s, steps=self.steps, tier=tier, split=split, **s)
            if split == "heldout" and s["success"] >= c.frontier_success:
                frontier = max(frontier, tier)
        self.frontier = frontier
        # Evaluation can run longer than the clock's gap cap, so count its time directly.
        self.active_s += time.monotonic() - t0
        self._last = time.monotonic()
        return frontier

    # ---- files --------------------------------------------------------------

    def _event(self, kind: str, detail: dict) -> None:
        self.db.log("events", wall=time.time(), active_s=self.active_s, steps=self.steps, kind=kind, detail=json.dumps(detail))

    def _meta(self) -> dict:
        return {
            "active_hours": round(self.active_s / 3600, 4), "steps": self.steps, "tier": self.curr.tier,
            "frontier": self.frontier, "episodes": self.episodes, "wall": time.time(),
        }

    def _save_latest(self) -> None:
        save_net(os.path.join(self.dir, "latest_net.npz"), self.agent.actor, self._meta())

    def _save_hof(self) -> None:
        save_net(os.path.join(self.dir, "hof", f"h{self.next_hof:04d}.npz"), self.agent.actor, self._meta())
        self.next_hof += 1

    def checkpoint(self) -> None:
        t0 = time.monotonic()
        state = {
            "version": CHECKPOINT_VERSION,
            "agent": self.agent.state(),
            "replay": self.replay.state(),
            "reservoir": self.reservoir.state(),
            "bank": self.bank.state(),
            "curriculum": self.curr.state(),
            "sampler": self.sampler.state(),
            "rng": self.rng.bit_generator.state,
            "counters": {
                "steps": self.steps, "episodes": self.episodes, "active_s": self.active_s,
                "eval_round": self.eval_round, "frontier": self.frontier, "nan_events": self.nan_events,
                "next_hof": self.next_hof, "tier_since": self.tier_since,
            },
        }
        tmp = self.ckpt_path + ".tmp"
        with open(tmp, "wb") as f:
            pickle.dump(state, f, protocol=5)
        os.replace(tmp, self.ckpt_path)
        self._event("checkpoint", {"seconds": round(time.monotonic() - t0, 3), "bytes": os.path.getsize(self.ckpt_path)})

    def _load(self) -> None:
        with open(self.ckpt_path, "rb") as f:
            s = pickle.load(f)
        self.agent.load(s["agent"])
        self.replay.load(s["replay"])
        self.reservoir.load(s["reservoir"])
        self.bank.load(s["bank"])
        self.curr.load(s["curriculum"])
        self.sampler.load(s["sampler"])
        self.rng.bit_generator.state = s["rng"]
        for k, v in s["counters"].items():
            setattr(self, k, v)
