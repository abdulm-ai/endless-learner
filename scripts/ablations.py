"""Run the ablations: each variant at the same step budget and the same three seeds, a few at a time.

    python scripts/ablations.py [--after runs/cbp] [--parallel 3] [--steps 5000000]

Before starting each run it waits until the machine has enough free memory.
Runs that already finished are skipped and unfinished ones resume, so after a
reboot the same command carries on. With --after it first waits for that run
to finish, so the ablations don't compete with it for CPU.
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

VARIANTS = {
    "full": [],
    "no-per": ["--no-per"],
    "no-restarts": ["--no-restarts"],
    "no-curriculum": ["--no-curriculum", "--fixed-tier", "6"],
    "no-old-replay": ["--no-reservoir", "--no-old-tiers"],
}
SEEDS = (1, 2, 3)


def free_mb() -> float:
    if sys.platform == "win32":
        import ctypes

        class MS(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong), ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong), ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong), ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong), ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        m = MS()
        m.dwLength = ctypes.sizeof(MS)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
        return m.ullAvailPhys / 2**20
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) / 1024
    return 0.0


def last_stop(run: str) -> str | None:
    """The reason the run last stopped, or None if it never did."""
    path = os.path.join(ROOT, run, "metrics.sqlite")
    if not os.path.exists(path):
        return None
    try:
        con = sqlite3.connect("file:" + path.replace("\\", "/") + "?mode=ro", uri=True)
        row = con.execute("SELECT detail FROM events WHERE kind IN ('stop','resume','start') ORDER BY id DESC LIMIT 1").fetchone()
        con.close()
    except sqlite3.Error:
        return None
    if not row:
        return None
    return json.loads(row[0]).get("reason")


def log(msg: str) -> None:
    print(time.strftime("%Y-%m-%d %H:%M:%S"), msg, flush=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--after", help="wait for this run folder to finish first")
    p.add_argument("--parallel", type=int, default=3)
    p.add_argument("--steps", type=int, default=5_000_000)
    p.add_argument("--min-free-mb", type=float, default=1500, help="free RAM needed before starting another run")
    p.add_argument("--out", default="runs/abl", help="folder for the ablation runs")
    a = p.parse_args()

    if a.after:
        log(f"waiting for {a.after} to finish")
        while last_stop(a.after) != "max_steps":
            time.sleep(60)
        log(f"{a.after} finished")

    queue = [(v, s) for s in SEEDS for v in VARIANTS]
    running: dict = {}
    tries: dict = {}
    while queue or running:
        for key, (proc, f) in list(running.items()):
            if proc.poll() is None:
                continue
            f.close()
            del running[key]
            run = f"{a.out}/{key[0]}-s{key[1]}"
            reason = last_stop(run)
            if reason == "max_steps":
                log(f"done {run}")
            elif tries.get(key, 0) < 3:
                log(f"{run} stopped ({reason}); will resume it")
                queue.insert(0, key)
            else:
                log(f"{run} stopped ({reason}) three times; skipping it")
        while queue and len(running) < a.parallel:
            key = queue[0]
            run = f"{a.out}/{key[0]}-s{key[1]}"
            if last_stop(run) == "max_steps":
                queue.pop(0)
                log(f"already finished {run}")
                continue
            if free_mb() < a.min_free_mb:
                break  # wait for memory; try again next round
            queue.pop(0)
            tries[key] = tries.get(key, 0) + 1
            os.makedirs(os.path.join(ROOT, a.out), exist_ok=True)
            f = open(os.path.join(ROOT, run + ".log"), "a")
            cmd = [sys.executable, "-m", "endless", "train", "--run", run, "--seed", str(key[1]), "--max-steps", str(a.steps),
                   "--no-view", "--hof-every", "1800", *VARIANTS[key[0]]]
            running[key] = (subprocess.Popen(cmd, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT), f)
            log(f"started {run} ({len(queue)} queued, {free_mb():,.0f} MB free)")
        time.sleep(30 if a.steps > 100_000 else 2)
    log("all ablations finished")


if __name__ == "__main__":
    main()
