"""Freeze the results: every number and chart in the README, from the runs' metrics databases.

    python scripts/results.py

Writes results/results.json and results/*.png. Runs keep training after the
cut-off, so every query is limited to the step counts below, and running this
again later gives the same numbers. Charts need matplotlib; nothing else in the
repo does.
"""

from __future__ import annotations

import glob
import json
import os
import sqlite3
import statistics as st
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
OUT = os.path.join(ROOT, "results")

CUT_SEED0 = 226_000_000  # runs/main and runs/cbp, cut-off on 2026-10-25
CUT_SEED1 = 100_000_000  # runs/plain-s1 and runs/cbp-s1, finished
ABL_WINDOW = (4_000_000, 5_000_000)  # ablations: held-out averaged over this window
WINDOW = 10_000_000  # long runs: tier-7 held-out averaged over windows this wide
VARIANTS = ["full", "no-per", "no-restarts", "no-curriculum", "no-old-replay"]
LABELS = {
    "full": "full config",
    "no-per": "no prioritised replay",
    "no-restarts": "no death-bank restarts",
    "no-curriculum": "no curriculum (always tier 6)",
    "no-old-replay": "no old-tier replay",
}


def db(run: str) -> sqlite3.Connection:
    return sqlite3.connect("file:" + os.path.join(ROOT, run, "metrics.sqlite").replace("\\", "/") + "?mode=ro", uri=True)


def heldout(c, tier: int, lo: float, hi: float) -> float | None:
    v = [r[0] for r in c.execute("SELECT success FROM evals WHERE split='heldout' AND tier=? AND steps BETWEEN ? AND ?", (tier, lo, hi))]
    return st.mean(v) if v else None


def promotions(c, cut: float) -> list:
    return [
        {"tier": json.loads(d)["tier"], "kind": k, "steps": s, "active_h": round(a / 3600, 2)}
        for d, s, a, k in c.execute(
            "SELECT detail, steps, active_s, kind FROM events WHERE kind IN ('promote','demote') AND steps <= ? ORDER BY id", (cut,)
        )
    ]


def ablations() -> dict:
    lo, hi = ABL_WINDOW
    out = {}
    for var in VARIANTS:
        seeds = []
        for run in sorted(glob.glob(os.path.join(ROOT, "runs", "abl", f"{var}-s*", ""))):
            name = os.path.relpath(run, ROOT).rstrip("\\/")
            c = db(name)
            by_tier = [heldout(c, t, lo, hi) for t in range(8)]
            score = st.mean([x if x is not None else 0.0 for x in by_tier[:6]])
            promo = promotions(c, 5_000_000)
            first = {p["tier"]: p["steps"] for p in reversed(promo) if p["kind"] == "promote"}
            seeds.append({"run": name, "score_t0_5": round(score, 3), "by_tier": [None if x is None else round(x, 3) for x in by_tier], "first_reached": first})
            c.close()
        scores = [s["score_t0_5"] for s in seeds]
        out[var] = {
            "label": LABELS[var],
            "mean": round(st.mean(scores), 3),
            "min": min(scores),
            "max": max(scores),
            "seeds_reaching": {t: sum(1 for s in seeds if t in s["first_reached"]) for t in (5, 6, 7)},
            "seeds": seeds,
        }
    return out


def long_pair(plain: str, cbp: str, cut: float) -> dict:
    out = {"cut_steps": cut, "windows": []}
    cs = {r: db(r) for r in (plain, cbp)}
    for lo in range(0, int(cut), WINDOW):
        hi = min(lo + WINDOW, cut)
        row = {"from": lo, "to": hi}
        for r, c in cs.items():
            dead, srank, wnorm = c.execute(
                "SELECT AVG(dead_frac), AVG(srank), AVG(wnorm) FROM plasticity WHERE steps BETWEEN ? AND ?", (lo, hi)
            ).fetchone()
            t06 = [heldout(c, t, lo, hi) for t in range(7)]
            row[r] = {
                "tier7": heldout(c, 7, lo, hi),
                "tiers0_6": st.mean([x or 0.0 for x in t06]),
                "dead": dead,
                "srank": srank,
                "wnorm": wnorm,
                "avg_tier": c.execute("SELECT AVG(tier) FROM train WHERE steps BETWEEN ? AND ?", (lo, hi)).fetchone()[0],
            }
        out["windows"].append(row)
    for r, c in cs.items():
        last = c.execute("SELECT dead_frac, srank, wnorm, steps FROM plasticity WHERE steps <= ? ORDER BY id DESC LIMIT 1", (cut,)).fetchone()
        act = c.execute("SELECT MAX(active_s) FROM train WHERE steps <= ?", (cut,)).fetchone()[0]
        out[r] = {
            "promotions": promotions(c, cut),
            "final_plasticity": {"dead": last[0], "srank": last[1], "wnorm": last[2], "steps": last[3]},
            "active_h": round(act / 3600, 1),
            "units_reset": c.execute("SELECT MAX(units_reset) FROM train WHERE steps <= ?", (cut,)).fetchone()[0],
        }
        c.close()
    return out


def failure_kpis(run: str, cut: float) -> dict:
    c = db(run)
    rs = c.execute("SELECT passed, outcome, attempt FROM restarts WHERE steps <= ?", (cut,)).fetchall()
    fixed = [a for _, o, a in rs if o == "fixed"]
    n = c.execute("SELECT COUNT(*) FROM train WHERE steps <= ?", (cut,)).fetchone()[0]
    nd, rt = c.execute(
        "SELECT AVG(td_near_death), AVG(td_routine) FROM train WHERE steps <= ? AND id > ?", (cut, int(n * 0.8))
    ).fetchone()
    out = {
        "restarts": len(rs),
        "restart_pass_rate": round(st.mean(p for p, _, _ in rs), 3),
        "fixed": len(fixed),
        "gave_up": sum(1 for _, o, _ in rs if o == "gave_up"),
        "attempts_to_fix_median": st.median(fixed),
        "attempts_to_fix_mean": round(st.mean(fixed), 2),
        "td_near_death": round(nd, 3),
        "td_routine": round(rt, 3),
    }
    c.close()
    return out


def systems(run: str, cut: float) -> dict:
    c = db(run)
    rows = c.execute("SELECT sps, rss_mb, active_s, cpu FROM train WHERE steps <= ? ORDER BY id", (cut,)).fetchall()
    warm = rows[len(rows) // 20:]
    ck = c.execute("SELECT detail FROM events WHERE kind='checkpoint' AND steps <= ? ORDER BY id DESC LIMIT 1", (cut,)).fetchone()
    ck = json.loads(ck[0])
    out = {
        "sps_median": round(st.median(r[0] for r in warm)),
        "cpu_median": round(st.median(r[3] for r in warm)),
        "rss_start_mb": round(warm[0][1]),
        "rss_end_mb": round(warm[-1][1]),
        "rss_growth_mb_per_active_h": round((warm[-1][1] - warm[0][1]) / ((warm[-1][2] - warm[0][2]) / 3600), 1),
        "checkpoint_mb": round(ck["bytes"] / 2**20),
        "checkpoint_s": ck["seconds"],
        "resumes": c.execute("SELECT COUNT(*) FROM events WHERE kind='resume' AND steps <= ?", (cut,)).fetchone()[0],
        "nan_recoveries": c.execute("SELECT MAX(nan_events) FROM train WHERE steps <= ?", (cut,)).fetchone()[0],
        "episodes": c.execute("SELECT COUNT(*) FROM episodes WHERE steps <= ?", (cut,)).fetchone()[0],
    }
    c.close()
    return out


def baselines(levels: int = 20) -> dict:
    from endless.evaluate import random_policy, rule_policy, run_levels, summarize
    from endless.levels import LevelCache, eval_seed

    cache = LevelCache()
    out = {"random": {}, "rule": {}}
    for t in range(9):
        lv = [cache.get(t, eval_seed(t, i)) for i in range(levels)]
        out["random"][t] = summarize(run_levels(lv, random_policy(t)))["success"]
        out["rule"][t] = summarize(run_levels(lv, rule_policy()))["success"]
    return out


# ---- charts ------------------------------------------------------------------

PLAIN, CBP = "#d1495b", "#2e86ab"
INK, MUTED, GRID = "#1f2328", "#656d76", "#d0d7de"


def _style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=10)
    ax.yaxis.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def _smooth(v: list, k: int = 25) -> list:
    """Centred rolling mean, so a noisy per-50k-step metric reads as a trend."""
    out = []
    for i in range(len(v)):
        w = v[max(0, i - k): i + k + 1]
        out.append(sum(w) / len(w))
    return out


def charts(res: dict) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.family": "DejaVu Sans", "text.color": INK, "axes.labelcolor": INK, "axes.titlesize": 13, "axes.titleweight": "bold"})

    # 1. Ablations
    ab = res["ablations"]
    order = sorted(VARIANTS, key=lambda v: ab[v]["mean"])
    fig, ax = plt.subplots(figsize=(9, 4.2), dpi=150)
    for i, v in enumerate(order):
        color = "#8c959f" if v == "full" else ("#d1495b" if v == "no-old-replay" else "#2e86ab")
        ax.barh(i, ab[v]["mean"], color=color, height=0.55)
        for s in ab[v]["seeds"]:
            ax.plot(s["score_t0_5"], i, "o", color=INK, markersize=4)
        ax.text(max(ab[v]["max"], ab[v]["mean"]) + 0.02, i, f"{ab[v]['mean']:.2f}", va="center", fontsize=10, color=INK)
    ax.set_yticks(range(len(order)), [ab[v]["label"] for v in order], fontsize=10)
    ax.set_xlim(0, 1.08)
    ax.set_xlabel("held-out success, tiers 0 to 5 (bars: mean of 3 seeds, dots: each seed)", color=MUTED, fontsize=10)
    ax.set_title("Removing one piece at a time, 5M steps each", loc="left")
    _style(ax)
    ax.yaxis.grid(False)
    ax.xaxis.grid(True, color=GRID, linewidth=0.6)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "ablations.png"))
    plt.close(fig)

    # 2. Forgetting: full against no old-tier replay, by tier
    fig, ax = plt.subplots(figsize=(8, 4), dpi=150)
    for v, color in (("full", "#2e86ab"), ("no-old-replay", "#d1495b")):
        means = []
        for t in range(8):
            vals = [s["by_tier"][t] for s in ab[v]["seeds"] if s["by_tier"][t] is not None]
            means.append(st.mean(vals) if vals else None)
        ax.plot(range(8), means, "o-", color=color, label=ab[v]["label"], linewidth=2)
    ax.set_xticks(range(8))
    ax.set_xlabel("tier (0 = easiest)", color=MUTED, fontsize=10)
    ax.set_ylabel("held-out success", color=MUTED, fontsize=10)
    ax.set_ylim(-0.03, 1.05)
    ax.set_title("Without replaying old tiers, it forgets how to play the easy ones", loc="left")
    ax.legend(frameon=False, fontsize=10)
    _style(ax)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "forgetting.png"))
    plt.close(fig)

    # 3. Plasticity over training
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), dpi=150)
    series = [("runs/main", PLAIN, "-", "plain, seed 0"), ("runs/cbp", CBP, "-", "continual backprop, seed 0"),
              ("runs/plain-s1", PLAIN, "--", "plain, seed 1"), ("runs/cbp-s1", CBP, "--", "continual backprop, seed 1")]
    for run, color, ls, label in series:
        cut = CUT_SEED0 if "s1" not in run else CUT_SEED1
        c = db(run)
        rows = c.execute("SELECT steps, dead_frac, srank FROM plasticity WHERE steps <= ? ORDER BY id", (cut,)).fetchall()
        c.close()
        x = _smooth([r[0] / 1e6 for r in rows])
        axes[0].plot(x, _smooth([r[1] for r in rows]), ls, color=color, label=label, linewidth=1.8)
        axes[1].plot(x, _smooth([r[2] for r in rows]), ls, color=color, label=label, linewidth=1.8)
    axes[0].set_title("Dead hidden units", loc="left")
    axes[0].set_ylabel("share of units that never fire", color=MUTED, fontsize=10)
    axes[1].set_title("Feature rank (srank of the last hidden layer)", loc="left")
    axes[1].set_ylabel("out of 128", color=MUTED, fontsize=10)
    for ax in axes:
        ax.set_xlabel("environment steps (millions)", color=MUTED, fontsize=10)
        _style(ax)
    axes[0].legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "plasticity.png"))
    plt.close(fig)

    # 4. Tier-7 held-out success over training
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), dpi=150, gridspec_kw={"width_ratios": [2.26, 1]})
    for ax, key, plain, cbp, title in (
        (axes[0], "seed0", "runs/main", "runs/cbp", "Seed 0"),
        (axes[1], "seed1", "runs/plain-s1", "runs/cbp-s1", "Seed 1"),
    ):
        w = res[key]["windows"]
        for run, color, label in ((plain, PLAIN, "plain"), (cbp, CBP, "continual backprop")):
            x = [(r["from"] + r["to"]) / 2e6 for r in w if r[run]["tier7"] is not None]
            y = [r[run]["tier7"] for r in w if r[run]["tier7"] is not None]
            ax.plot(x, y, "o-", color=color, label=label, linewidth=1.8, markersize=3.5)
        ax.set_ylim(0, 1)
        ax.set_title(title, loc="left", fontsize=11)
        ax.set_xlabel("environment steps (millions)", color=MUTED, fontsize=10)
        _style(ax)
    axes[0].legend(frameon=False, fontsize=10, loc="lower left")
    axes[0].set_ylabel("held-out success on tier 7", color=MUTED, fontsize=10)
    fig.suptitle("Tier-7 held-out success, averaged over 10M-step windows", x=0.01, ha="left", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "tier7.png"))
    plt.close(fig)

    # 5. The climb and the plateau: training tier over the seed-0 runs
    fig, ax = plt.subplots(figsize=(11, 3.6), dpi=150)
    for run, color, label in (("runs/main", PLAIN, "plain"), ("runs/cbp", CBP, "continual backprop")):
        c = db(run)
        rows = c.execute("SELECT steps, tier FROM train WHERE steps <= ? ORDER BY id", (CUT_SEED0,)).fetchall()
        c.close()
        ax.step([r[0] / 1e6 for r in rows], [r[1] for r in rows], where="post", color=color, label=label, linewidth=1.6)
    ax.set_xscale("symlog", linthresh=1)
    ax.set_xticks([0, 1, 2, 5, 10, 20, 50, 100, 200], ["0", "1M", "2M", "5M", "10M", "20M", "50M", "100M", "200M"])
    ax.set_yticks(range(9))
    ax.set_ylabel("training tier", color=MUTED, fontsize=10)
    ax.set_xlabel("environment steps (log scale past 1M)", color=MUTED, fontsize=10)
    ax.set_title("Tiers 0 to 6 within 3.3M steps, then a long plateau at tier 7", loc="left")
    ax.legend(frameon=False, fontsize=10, loc="lower right")
    _style(ax)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "climb.png"))
    plt.close(fig)


def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    res = {
        "cut_off": {"seed0_steps": CUT_SEED0, "seed1_steps": CUT_SEED1, "date": "2026-10-25"},
        "ablations": ablations(),
        "seed0": long_pair("runs/main", "runs/cbp", CUT_SEED0),
        "seed1": long_pair("runs/plain-s1", "runs/cbp-s1", CUT_SEED1),
        "failure_kpis_main": failure_kpis("runs/main", CUT_SEED0),
        "systems_main": systems("runs/main", CUT_SEED0),
        "systems_cbp": systems("runs/cbp", CUT_SEED0),
        "baselines": baselines(),
    }
    with open(os.path.join(OUT, "results.json"), "w") as f:
        json.dump(res, f, indent=1, default=float)
    charts(res)
    print(json.dumps({k: res[k] for k in ("failure_kpis_main", "systems_main", "systems_cbp", "baselines")}, indent=1))
    for v in VARIANTS:
        a = res["ablations"][v]
        print(f"{v:16} {a['mean']:.2f} ({a['min']:.2f} to {a['max']:.2f}) reaching 5/6/7: {a['seeds_reaching']}")
    for key in ("seed0", "seed1"):
        p = res[key]
        runs = [k for k in p if k.startswith("runs/")]
        for r in runs:
            print(r, p[r]["active_h"], "h", p[r]["final_plasticity"], [(x["tier"], x["kind"][0], round(x["steps"] / 1e6, 1)) for x in p[r]["promotions"]])


if __name__ == "__main__":
    main()
