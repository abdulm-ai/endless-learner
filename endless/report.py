"""Print every KPI of a run from its metrics database. The numbers in the README come from here."""

from __future__ import annotations

import glob
import json
import os

import numpy as np

from .agent import load_net
from .evaluate import greedy, run_levels
from .levels import LevelCache, eval_seed
from .metrics import read_only


def _q(con, sql, args=()):
    return con.execute(sql, args).fetchall()


def report(run: str, hof: bool = False) -> dict:
    con = read_only(os.path.join(run, "metrics.sqlite"))
    out: dict = {}
    last = dict(_q(con, "SELECT * FROM train ORDER BY id DESC LIMIT 1")[0])
    first_wall = _q(con, "SELECT wall FROM events ORDER BY id LIMIT 1")[0][0]
    n_ep = _q(con, "SELECT COUNT(*) FROM episodes")[0][0]
    print(f"run {run}")
    print(f"  active hours {last['active_s'] / 3600:.2f}, calendar hours {(last['wall'] - first_wall) / 3600:.2f}, "
          f"steps {last['steps']:,}, episodes {n_ep:,}, current tier {last['tier']}")

    # ---- learning
    rounds = [r[0] for r in _q(con, "SELECT DISTINCT round FROM evals ORDER BY round")]
    if rounds:
        r = rounds[-1]
        rows = {(t, s): (succ, prog) for t, s, succ, prog in _q(con, "SELECT tier, split, success, progress_median FROM evals WHERE round=?", (r,))}
        tiers = sorted({t for t, _ in rows})
        frontier = max([t for t in tiers if rows.get((t, "heldout"), (0,))[0] >= 0.8], default=-1)
        out["frontier"] = frontier
        print(f"\nLearning (eval round {r})")
        print(f"  frontier tier: {frontier}")
        print("  tier  held-out success  median progress  train-seed success  gap  forgetting  backward transfer")
        promos = {json.loads(d)["tier"]: s for d, s in _q(con, "SELECT detail, steps FROM events WHERE kind='promote'")}
        for t in tiers:
            h = rows.get((t, "heldout"), (None, None))
            tr = rows.get((t, "train"), (None, None))
            best = _q(con, "SELECT MAX(success) FROM evals WHERE tier=? AND split='heldout'", (t,))[0][0]
            forget = best - h[0] if h[0] is not None and best is not None else None
            bwt = None
            if (t + 1) in promos:
                after = _q(con, "SELECT success FROM evals WHERE tier=? AND split='heldout' AND steps>=? ORDER BY round LIMIT 1", (t, promos[t + 1]))
                if after:
                    bwt = h[0] - after[0][0]
            gap = tr[0] - h[0] if tr[0] is not None and h[0] is not None else None
            print(f"  {t:>4}  {_f(h[0]):>16}  {_f(h[1]):>15}  {_f(tr[0]):>18}  {_f(gap):>4}  {_f(forget):>10}  {_f(bwt):>17}")
    print("\n  steps to reach each tier")
    for d, steps, act in _q(con, "SELECT detail, steps, active_s FROM events WHERE kind='promote' ORDER BY id"):
        d = json.loads(d)
        print(f"    tier {d['tier']:>3}: {steps:>12,} steps, {act / 3600:6.2f} active h  ({d['steps_at_previous_tier']:,} steps at the tier before)")
    ex = _q(con, "SELECT success FROM episodes WHERE kind='exam' ORDER BY id DESC LIMIT 200")
    if ex:
        print(f"  exam success, last {len(ex)} exams: {np.mean([e[0] for e in ex]):.2f}")

    # ---- learning from failure
    print("\nLearning from failure")
    rs = _q(con, "SELECT passed, outcome, attempt FROM restarts ORDER BY id")
    if rs:
        passed = np.array([r[0] for r in rs])
        fixed = [r[2] for r in rs if r[1] == "fixed"]
        gave = sum(1 for r in rs if r[1] == "gave_up")
        tail = passed[-max(len(passed) // 10, 1):]
        print(f"  death-bank restarts {len(rs):,}: pass rate {passed.mean():.2f} overall, {tail.mean():.2f} in the last 10%")
        if fixed:
            print(f"  failures fixed {len(fixed):,}: attempts to fix mean {np.mean(fixed):.1f}, median {np.median(fixed):.0f}; gave up on {gave:,}")
    n_tr = _q(con, "SELECT COUNT(*) FROM train")[0][0]
    td = _q(con, "SELECT AVG(td_near_death), AVG(td_routine) FROM train WHERE id > ?", (int(n_tr * 0.8),))[0]
    if td[0] is not None:
        print(f"  |TD| near deaths {td[0]:.3f} vs routine {td[1]:.3f} (last 20% of training), ratio {td[0] / td[1]:.1f}x")
    exams = _q(con, "SELECT death_sig FROM episodes WHERE kind='exam' ORDER BY id")
    if len(exams) >= 40:
        q = len(exams) // 4
        early, late = exams[:q], exams[-q:]
        sigs = sorted({e[0] for e in exams if e[0]}, key=lambda s: -sum(1 for e in exams if e[0] == s))
        print("  deaths per exam by cause, first quarter of exams vs last quarter")
        for s in sigs[:8]:
            a = sum(1 for e in early if e[0] == s) / len(early)
            b = sum(1 for e in late if e[0] == s) / len(late)
            print(f"    {s:<16} {a:.3f} -> {b:.3f}")

    # ---- never stops
    print("\nNever stops")
    pl = _q(con, "SELECT steps, dead_frac, srank, wnorm FROM plasticity ORDER BY id")
    if pl:
        print(f"  dead units {pl[0][1]:.3f} -> {pl[-1][1]:.3f}, srank {pl[0][2]} -> {pl[-1][2]}, weight norm {pl[0][3]:.1f} -> {pl[-1][3]:.1f}")
    st = _q(con, "SELECT MAX(q_max), MAX(nan_events), MAX(units_reset) FROM train")[0]
    print(f"  max Q {_f(st[0])}, NaN recoveries {st[1]}, units reset by continual backprop {st[2]}")

    # ---- systems
    print("\nSystems")
    sys_rows = np.array(_q(con, "SELECT sps, cpu, rss_mb FROM train ORDER BY id"), dtype=np.float64)
    if len(sys_rows):
        warm = sys_rows[len(sys_rows) // 10:] if len(sys_rows) > 10 else sys_rows
        print(f"  steps/s median {np.median(warm[:, 0]):,.0f}, CPU median {np.median(warm[:, 1]):.0f}% of one core")
        print(f"  RAM after warm-up {warm[0, 2]:.0f} MB -> {warm[-1, 2]:.0f} MB, max {warm[:, 2].max():.0f} MB")
    res = _q(con, "SELECT detail FROM events WHERE kind='resume'")
    lost = [json.loads(r[0]).get("lost_steps", 0) for r in res]
    print(f"  resumes {len(res)}, steps lost per resume {lost if lost else '-'}")
    ck = _q(con, "SELECT detail FROM events WHERE kind='checkpoint' ORDER BY id DESC LIMIT 1")
    if ck:
        c = json.loads(ck[0][0])
        print(f"  last checkpoint {c['bytes'] / 2**20:.0f} MB in {c['seconds']:.2f} s")
    con.close()

    if hof:
        out["hof"] = hof_wins(run)
    return out


def hof_wins(run: str, levels: int = 20) -> list:
    """Newest snapshot against each older one, same held-out levels at the newest snapshot's tier, by distance."""
    files = sorted(glob.glob(os.path.join(run, "hof", "h*.npz")))
    if len(files) < 2:
        print("\nHall of fame: fewer than two snapshots")
        return []
    new_p, new_meta = load_net(files[-1])
    tier = int(new_meta["tier"])
    cache = LevelCache()
    lv = [cache.get(tier, eval_seed(tier, i)) for i in range(levels)]
    new_res = run_levels(lv, greedy(new_p))
    print(f"\nHall of fame: {os.path.basename(files[-1])} against older snapshots on {levels} held-out tier-{tier} levels")
    out = []
    for f in files[:-1]:
        p, meta = load_net(f)
        old_res = run_levels(lv, greedy(p))
        wins = sum(1.0 if a["distance"] > b["distance"] + 1e-6 else 0.5 if abs(a["distance"] - b["distance"]) <= 1e-6 else 0.0 for a, b in zip(new_res, old_res))
        rate = wins / levels
        out.append((os.path.basename(f), meta["active_hours"], rate))
        print(f"  vs {os.path.basename(f)} ({meta['active_hours']:.2f} h): newest wins {rate:.2f}")
    return out


def _f(x) -> str:
    return "-" if x is None else f"{x:.2f}"
