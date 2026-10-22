"""The metrics log: one SQLite file per run, written by the trainer and read by the viewer.

Every KPI in the README comes from these tables. WAL mode lets the viewer read
while the trainer writes.
"""

from __future__ import annotations

import os
import sqlite3
import sys
import time

SCHEMA = """
CREATE TABLE IF NOT EXISTS episodes (
    id INTEGER PRIMARY KEY, wall REAL, active_s REAL, steps INTEGER, episode INTEGER,
    tier INTEGER, seed INTEGER, kind TEXT, replayed INTEGER, success INTEGER, progress REAL,
    distance REAL, decisions INTEGER, ret REAL, death_sig TEXT, mean_td REAL, eps REAL
);
CREATE TABLE IF NOT EXISTS train (
    id INTEGER PRIMARY KEY, wall REAL, active_s REAL, steps INTEGER, updates INTEGER, tier INTEGER,
    loss REAL, td_abs REAL, td_near_death REAL, td_routine REAL, q_mean REAL, q_max REAL, grad_norm REAL,
    eps REAL, beta REAL, sps REAL, ups REAL, cpu REAL, rss_mb REAL, nan_events INTEGER,
    bank_size INTEGER, bank_fixed INTEGER, bank_gave_up INTEGER, units_reset INTEGER
);
CREATE TABLE IF NOT EXISTS plasticity (
    id INTEGER PRIMARY KEY, active_s REAL, steps INTEGER, dead_frac REAL, dead_l0 REAL, dead_l1 REAL,
    srank INTEGER, wnorm REAL
);
CREATE TABLE IF NOT EXISTS evals (
    id INTEGER PRIMARY KEY, round INTEGER, active_s REAL, steps INTEGER, tier INTEGER, split TEXT,
    levels INTEGER, success REAL, progress_median REAL, progress_p90 REAL
);
CREATE TABLE IF NOT EXISTS restarts (
    id INTEGER PRIMARY KEY, active_s REAL, steps INTEGER, tier INTEGER, signature TEXT,
    passed INTEGER, attempt INTEGER, outcome TEXT
);
CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY, wall REAL, active_s REAL, steps INTEGER, kind TEXT, detail TEXT
);
CREATE INDEX IF NOT EXISTS ep_steps ON episodes(steps);
CREATE INDEX IF NOT EXISTS ev_round ON evals(round);
"""

STEP_TABLES = ("episodes", "train", "plasticity", "evals", "restarts", "events")


class MetricsDB:
    def __init__(self, path: str):
        self.path = path
        self.con = sqlite3.connect(path, check_same_thread=False)
        self.con.execute("PRAGMA journal_mode=WAL")
        self.con.execute("PRAGMA synchronous=NORMAL")
        self.con.executescript(SCHEMA)
        self.pending: dict[str, list[dict]] = {}
        self.last_commit = time.monotonic()

    def log(self, table: str, **row) -> None:
        self.pending.setdefault(table, []).append(row)

    def flush(self, force: bool = False) -> None:
        if not force and time.monotonic() - self.last_commit < 5.0:
            return
        for table, rows in self.pending.items():
            if not rows:
                continue
            cols = list(rows[0])
            sql = f"INSERT INTO {table} ({','.join(cols)}) VALUES ({','.join('?' * len(cols))})"
            self.con.executemany(sql, [tuple(r[c] for c in cols) for r in rows])
        self.pending = {}
        self.con.commit()
        self.last_commit = time.monotonic()

    def truncate_after(self, steps: int) -> int:
        """Drop rows logged after the checkpoint being resumed: that training was lost. Returns the steps lost."""
        lost = 0
        for table in STEP_TABLES:
            row = self.con.execute(f"SELECT MAX(steps) FROM {table}").fetchone()
            if row and row[0] is not None:
                lost = max(lost, row[0] - steps)
            self.con.execute(f"DELETE FROM {table} WHERE steps > ?", (steps,))
        self.con.commit()
        return max(lost, 0)

    def close(self) -> None:
        self.flush(force=True)
        self.con.close()


def read_only(path: str) -> sqlite3.Connection:
    uri = "file:" + os.path.abspath(path).replace("\\", "/") + "?mode=ro"
    con = sqlite3.connect(uri, uri=True, check_same_thread=False)
    con.row_factory = sqlite3.Row
    return con


class ProcessStats:
    """CPU share of one core and resident memory, with the standard library only."""

    def __init__(self):
        self._t = time.monotonic()
        self._cpu = self._cpu_time()

    @staticmethod
    def _cpu_time() -> float:
        t = os.times()
        return t.user + t.system

    def cpu_percent(self) -> float:
        now, cpu = time.monotonic(), self._cpu_time()
        dt = now - self._t
        pct = 100.0 * (cpu - self._cpu) / dt if dt > 0 else 0.0
        self._t, self._cpu = now, cpu
        return pct

    @staticmethod
    def rss_mb() -> float:
        if sys.platform == "win32":
            import ctypes
            from ctypes import wintypes

            class PMC(ctypes.Structure):
                _fields_ = [
                    ("cb", wintypes.DWORD),
                    ("PageFaultCount", wintypes.DWORD),
                    ("PeakWorkingSetSize", ctypes.c_size_t),
                    ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t),
                    ("PeakPagefileUsage", ctypes.c_size_t),
                ]

            k32, psapi = ctypes.windll.kernel32, ctypes.windll.psapi
            k32.GetCurrentProcess.restype = wintypes.HANDLE
            psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(PMC), wintypes.DWORD]
            pmc = PMC()
            pmc.cb = ctypes.sizeof(PMC)
            psapi.GetProcessMemoryInfo(k32.GetCurrentProcess(), ctypes.byref(pmc), pmc.cb)
            return pmc.WorkingSetSize / 2**20
        try:
            with open("/proc/self/statm") as f:
                return int(f.read().split()[1]) * os.sysconf("SC_PAGE_SIZE") / 2**20
        except OSError:
            return 0.0
