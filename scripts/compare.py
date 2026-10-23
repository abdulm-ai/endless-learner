"""Compare runs with steadier numbers than one eval round: the ablations at 5M steps, and runs/main against runs/cbp at equal steps.

    python scripts/compare.py

Held-out success is averaged over every eval round in a step window, so one lucky or unlucky round of 20 levels does not decide anything.
"""
import glob
import json
import os
import sqlite3
import statistics as st

ROOT = r"C:\Users\abdul\showcase\endless-learner"


def con(run):
    return sqlite3.connect("file:" + os.path.join(ROOT, run, "metrics.sqlite").replace("\\", "/") + "?mode=ro", uri=True)


def late_heldout(c, lo, hi, tiers):
    """Mean held-out success per tier over eval rounds with lo <= steps <= hi."""
    out = {}
    for t in tiers:
        v = [r[0] for r in c.execute("SELECT success FROM evals WHERE split='heldout' AND tier=? AND steps BETWEEN ? AND ?", (t, lo, hi))]
        out[t] = st.mean(v) if v else None
    return out


def first_promo(c, tier):
    for (d, s) in c.execute("SELECT detail, steps FROM events WHERE kind='promote' ORDER BY id"):
        if json.loads(d)["tier"] == tier:
            return s
    return None


def fmt(x, n=2):
    return "  -  " if x is None else f"{x:.{n}f}"


print("=== Ablations at 5M steps, held-out success averaged over eval rounds from 4M to 5M ===")
groups = {}
for d in sorted(glob.glob(os.path.join(ROOT, "runs", "abl", "*", ""))):
    name = os.path.basename(d.rstrip("\\/"))
    var, seed = name.rsplit("-s", 1)
    groups.setdefault(var, []).append(name)
tiers = list(range(8))
print(f"{'variant':16} {'seed':>4} " + " ".join(f"t{t:<4}" for t in tiers) + "  mean0-5  mean0-6  tier5@   tier6@   tier7@")
summary = {}
for var, runs in groups.items():
    rows = []
    for run in runs:
        c = con(os.path.join("runs", "abl", run))
        h = late_heldout(c, 4_000_000, 5_000_000, tiers)
        m5 = st.mean([h[t] if h[t] is not None else 0.0 for t in range(6)])
        m6 = st.mean([h[t] if h[t] is not None else 0.0 for t in range(7)])
        p = [first_promo(c, k) for k in (5, 6, 7)]
        rows.append((m5, m6, p))
        print(f"{var:16} {run[-1]:>4} " + " ".join(fmt(h[t]) + " " for t in tiers) + f"  {m5:.2f}     {m6:.2f}   " + "  ".join(f"{(x / 1e6):5.2f}M" if x else "   -  " for x in p))
        c.close()
    summary[var] = (st.mean(r[0] for r in rows), st.mean(r[1] for r in rows), [sum(1 for r in rows if r[2][i]) for i in range(3)])
print()
for var, (m5, m6, reached) in summary.items():
    print(f"{var:16} mean held-out tiers 0-5: {m5:.2f}  tiers 0-6: {m6:.2f}  seeds reaching tier 5/6/7: {reached}")

print("\n=== main vs cbp, same seed and config, at equal steps ===")
for lo, hi in ((4e6, 6e6), (9e6, 11e6), (19e6, 21e6), (30e6, 36e6)):
    line = []
    for run in ("runs/main", "runs/cbp"):
        c = con(run)
        h = late_heldout(c, lo, hi, tiers)
        m = st.mean([h[t] if h[t] is not None else 0.0 for t in range(7)])
        pl = c.execute("SELECT AVG(dead_frac), AVG(srank), AVG(wnorm) FROM plasticity WHERE steps BETWEEN ? AND ?", (lo, hi)).fetchone()
        ex = c.execute("SELECT AVG(success), tier FROM episodes WHERE kind='exam' AND steps BETWEEN ? AND ? GROUP BY tier ORDER BY COUNT(*) DESC LIMIT 1", (lo, hi)).fetchone()
        line.append(f"{run[5:]:4} held-out 0-6 {m:.2f} | dead {pl[0]:.3f} srank {pl[1]:.0f} wnorm {pl[2]:.0f} | exams {ex[0]:.2f} on tier {ex[1]}")
        c.close()
    print(f"{lo / 1e6:.0f}-{hi / 1e6:.0f}M: " + "  ||  ".join(line))
for run in ("runs/main", "runs/cbp"):
    c = con(run)
    print(run, "tier 7 at", fmt(first_promo(c, 7) / 1e6 if first_promo(c, 7) else None), "M;", "units reset", c.execute("SELECT MAX(units_reset) FROM train").fetchone()[0])
    c.close()
