"""Command line: python -m endless <command> ..."""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

from .agent import AgentConfig, load_net
from .evaluate import greedy, random_policy, rule_policy, run_levels, summarize
from .levels import LevelCache, eval_seed


def _tiers(s: str) -> list[int]:
    if "-" in s:
        a, b = s.split("-")
        return list(range(int(a), int(b) + 1))
    return [int(x) for x in s.split(",")]


def cmd_train(a) -> None:
    from .server import serve
    from .trainer import Trainer, TrainConfig

    cfg = TrainConfig(
        seed=a.seed,
        agent=AgentConfig(cbp=a.cbp),
        per=not a.no_per,
        restarts=not a.no_restarts,
        curriculum=not a.no_curriculum,
        fixed_tier=a.fixed_tier,
        reservoir=not a.no_reservoir,
        old_tiers=not a.no_old_tiers,
        plr=not a.no_plr,
    )
    if a.hof_every:
        cfg.hof_every_s = a.hof_every
    if os.path.exists(os.path.join(a.run, "checkpoint.pkl")):
        print(f"{a.run} has a checkpoint: resuming with its saved config (flags other than limits and port are ignored)")
    trainer = Trainer(a.run, cfg, log=lambda m: print(m, flush=True))
    if not a.no_view:
        try:
            serve(a.run, a.port, background=True)
            print(f"live view: http://127.0.0.1:{a.port}", flush=True)
        except OSError as e:
            print(f"live view not started ({e}); run `python -m endless view --run {a.run}` separately")
    trainer.train(max_steps=a.max_steps, max_hours=a.max_hours, max_rss_mb=a.max_rss_mb)


def cmd_view(a) -> None:
    from .server import serve

    print(f"live view of {a.run}: http://127.0.0.1:{a.port}  (Ctrl+C to stop)")
    try:
        serve(a.run, a.port)
    except KeyboardInterrupt:
        pass


def cmd_stop(a) -> None:
    open(os.path.join(a.run, "STOP"), "w").close()
    print("asked the trainer to stop; it checkpoints and exits at the end of the current episode")


def _eval_levels(tiers: list[int], n: int):
    cache = LevelCache()
    levels, tags = [], []
    for t in tiers:
        for i in range(n):
            levels.append(cache.get(t, eval_seed(t, i)))
            tags.append(t)
    return levels, tags


def _print_table(name: str, tiers, tags, results) -> dict:
    out = {}
    print(f"\n{name}")
    print("tier  success  median progress")
    for t in tiers:
        s = summarize([r for tag, r in zip(tags, results) if tag == t])
        out[t] = s
        print(f"{t:>4}  {s['success']:>7.2f}  {s['progress_median']:>15.2f}")
    return out


def cmd_baseline(a) -> None:
    tiers = _tiers(a.tiers)
    levels, tags = _eval_levels(tiers, a.levels)
    res = {
        "random": _print_table("random agent", tiers, tags, run_levels(levels, random_policy(0))),
        "rule": _print_table("rule-based agent (sees the level directly)", tiers, tags, run_levels(levels, rule_policy())),
    }
    if a.out:
        with open(a.out, "w") as f:
            json.dump(res, f, indent=2)


def cmd_eval(a) -> None:
    path = os.path.join(a.run, "latest_net.npz") if a.net == "latest" else os.path.join(a.run, "hof", f"{a.net}.npz")
    params, meta = load_net(path)
    tiers = _tiers(a.tiers)
    levels, tags = _eval_levels(tiers, a.levels)
    print(f"{a.net}: {meta}")
    _print_table("learned agent, held-out levels", tiers, tags, run_levels(levels, greedy(params)))


def cmd_gradcheck(a) -> None:
    from .nn import gradcheck

    errs = gradcheck([7, 6, 5, 4])
    for i, e in enumerate(errs):
        print(f"{'W' if i % 2 == 0 else 'b'}{i // 2}: max relative error {e:.2e}")
    ok = max(errs) < 1e-5
    print("PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


def cmd_report(a) -> None:
    from .report import report

    report(a.run, hof=a.hof)


def main(argv=None) -> None:
    p = argparse.ArgumentParser(prog="python -m endless", description="A game agent that learns from its failures, forever, on a CPU.")
    p.add_argument("--threads", default="1", help="NumPy threads (set before import)")
    sub = p.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("train", help="train, or resume if the run folder has a checkpoint")
    t.add_argument("--run", default="runs/main")
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--port", type=int, default=8765)
    t.add_argument("--no-view", action="store_true")
    t.add_argument("--max-steps", type=int)
    t.add_argument("--max-hours", type=float, help="active hours")
    t.add_argument("--max-rss-mb", type=float, default=1500, help="stop cleanly if this process passes this much RAM (default 1500)")
    t.add_argument("--hof-every", type=float, help="seconds between hall-of-fame snapshots (default 3600)")
    t.add_argument("--no-per", action="store_true", help="ablation: uniform replay")
    t.add_argument("--no-restarts", action="store_true", help="ablation: no death bank restarts")
    t.add_argument("--no-curriculum", action="store_true", help="ablation: always play --fixed-tier")
    t.add_argument("--fixed-tier", type=int, default=6)
    t.add_argument("--no-reservoir", action="store_true", help="ablation: no replay of old transitions")
    t.add_argument("--no-old-tiers", action="store_true", help="ablation: never replay passed tiers")
    t.add_argument("--no-plr", action="store_true", help="ablation: fresh seeds only")
    t.add_argument("--cbp", action="store_true", help="continual backprop against plasticity loss")
    t.set_defaults(fn=cmd_train)

    v = sub.add_parser("view", help="serve the live view for a run folder")
    v.add_argument("--run", default="runs/main")
    v.add_argument("--port", type=int, default=8765)
    v.set_defaults(fn=cmd_view)

    s = sub.add_parser("stop", help="stop a running trainer cleanly")
    s.add_argument("--run", default="runs/main")
    s.set_defaults(fn=cmd_stop)

    b = sub.add_parser("baseline", help="random and rule-based agents on the held-out levels")
    b.add_argument("--tiers", default="0-10")
    b.add_argument("--levels", type=int, default=10)
    b.add_argument("--out")
    b.set_defaults(fn=cmd_baseline)

    e = sub.add_parser("eval", help="evaluate saved weights on the held-out levels")
    e.add_argument("--run", default="runs/main")
    e.add_argument("--net", default="latest")
    e.add_argument("--tiers", default="0-10")
    e.add_argument("--levels", type=int, default=10)
    e.set_defaults(fn=cmd_eval)

    r = sub.add_parser("report", help="print the KPIs of a run")
    r.add_argument("--run", default="runs/main")
    r.add_argument("--hof", action="store_true", help="also race the newest snapshot against every older one")
    r.set_defaults(fn=cmd_report)

    g = sub.add_parser("gradcheck", help="check backprop against numerical gradients")
    g.set_defaults(fn=cmd_gradcheck)

    args = p.parse_args(argv)
    np.seterr(over="ignore")
    args.fn(args)
