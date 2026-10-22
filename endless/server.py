"""The live view: a small HTTP server over a run folder.

It only reads files: the metrics database, the latest weights and the hall of
fame. So it works the same inside the training process, in a second process
while training runs, or after training has stopped. Standard library only.
"""

from __future__ import annotations

import glob
import json
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import numpy as np

from .agent import load_net
from .evaluate import rollout
from .levels import LevelCache, eval_seed
from .metrics import read_only

WEB = os.path.join(os.path.dirname(__file__), "web")


class RunView:
    def __init__(self, run_dir: str):
        self.dir = run_dir
        self.db_path = os.path.join(run_dir, "metrics.sqlite")
        self.cache = LevelCache(256)
        self._nets: dict = {}
        self._lock = threading.Lock()
        self.rng = np.random.default_rng()

    def _con(self):
        return read_only(self.db_path)

    def net(self, name: str):
        path = os.path.join(self.dir, "latest_net.npz") if name == "latest" else os.path.join(self.dir, "hof", f"{name}.npz")
        if not os.path.exists(path):
            raise FileNotFoundError(name)
        mtime = os.path.getmtime(path)
        with self._lock:
            hit = self._nets.get(path)
            if hit and hit[0] == mtime:
                return hit[1], hit[2]
            params, meta = load_net(path)
            self._nets[path] = (mtime, params, meta)
            return params, meta

    def hof(self) -> list[dict]:
        out = []
        for p in sorted(glob.glob(os.path.join(self.dir, "hof", "h*.npz"))):
            name = os.path.basename(p)[:-4]
            _, meta = self.net(name)
            out.append({"name": name, **meta})
        return out

    def summary(self) -> dict:
        con = self._con()
        try:
            last = con.execute("SELECT * FROM train ORDER BY id DESC LIMIT 1").fetchone()
            first = con.execute("SELECT wall FROM events ORDER BY id LIMIT 1").fetchone()
            r = con.execute("SELECT MAX(round) FROM evals").fetchone()[0]
            frontier = -1
            if r is not None:
                row = con.execute(
                    "SELECT MAX(tier) FROM evals WHERE round=? AND split='heldout' AND success>=0.8", (r,)
                ).fetchone()
                frontier = row[0] if row[0] is not None else -1
            eps = con.execute("SELECT COUNT(*), SUM(success) FROM episodes").fetchone()
            resumes = con.execute("SELECT COUNT(*) FROM events WHERE kind='resume'").fetchone()[0]
            lost = con.execute("SELECT detail FROM events WHERE kind='resume'").fetchall()
            changes = [
                dict(r) for r in con.execute(
                    "SELECT active_s, steps, kind, detail FROM events WHERE kind IN ('promote','demote') ORDER BY id DESC LIMIT 12"
                )
            ]
            restarts = con.execute("SELECT COUNT(*), SUM(passed) FROM restarts").fetchone()
            fix = con.execute("SELECT AVG(attempt), COUNT(*) FROM restarts WHERE outcome='fixed'").fetchone()
        finally:
            con.close()
        last = dict(last) if last else {}
        return {
            "training": bool(last) and time.time() - last.get("wall", 0) < 120,
            "last": last,
            "frontier": frontier,
            "eval_round": r,
            "episodes": eps[0] or 0,
            "successes": eps[1] or 0,
            "calendar_hours": (time.time() - first[0]) / 3600 if first else 0.0,
            "resumes": resumes,
            "lost_steps": sum(json.loads(x[0]).get("lost_steps", 0) for x in lost),
            "changes": changes,
            "restarts": restarts[0] or 0,
            "restart_passes": restarts[1] or 0,
            "attempts_to_fix": fix[0],
            "fixed": fix[1] or 0,
        }

    def series(self) -> dict:
        """Everything the charts need, aggregated in SQL so a week-long run stays fast to read."""
        con = self._con()
        try:
            n = con.execute("SELECT COUNT(*) FROM train").fetchone()[0]
            k = max(1, n // 400)
            train = [dict(r) for r in con.execute(f"SELECT * FROM train WHERE id % {k} = 0 ORDER BY id")]
            plast = [dict(r) for r in con.execute("SELECT * FROM plasticity ORDER BY id")]
            evals = [
                dict(r)
                for r in con.execute("SELECT round, active_s, steps, tier, split, success, progress_median FROM evals ORDER BY id")
            ]
            top = con.execute("SELECT MAX(active_s) FROM episodes").fetchone()[0] or 1.0
            n_exam = con.execute("SELECT COUNT(*) FROM episodes WHERE kind='exam'").fetchone()[0]
            # Up to 200 bins, but with about 20 exams in each so the curve isn't noise.
            w = max(top / min(200, max(n_exam // 20, 8)), 1.0)
            episodes = [
                {"hours": r[1] / 3600, "progress": r[2], "success": r[3], "n": r[4]}
                for r in con.execute(
                    "SELECT CAST(active_s / ? AS INT) b, AVG(active_s), AVG(progress), AVG(success), COUNT(*) "
                    "FROM episodes WHERE kind='exam' GROUP BY b ORDER BY b",
                    (w,),
                )
            ]
            restarts = [
                {"hours": r[1] / 3600, "pass_rate": r[2], "n": r[3]}
                for r in con.execute(
                    "SELECT CAST(active_s / ? AS INT) b, AVG(active_s), AVG(passed), COUNT(*) FROM restarts GROUP BY b ORDER BY b",
                    (w * 2,),
                )
            ]
            wd = w * 3
            exams = {r[0]: r[1] for r in con.execute(
                "SELECT CAST(active_s / ? AS INT) b, COUNT(*) FROM episodes WHERE kind='exam' GROUP BY b", (wd,)
            )}
            rates: dict[str, dict[int, float]] = {}
            counts: dict[str, int] = {}
            for b, sig, c in con.execute(
                "SELECT CAST(active_s / ? AS INT) b, death_sig, COUNT(*) FROM episodes "
                "WHERE kind='exam' AND death_sig IS NOT NULL GROUP BY b, death_sig",
                (wd,),
            ):
                rates.setdefault(sig, {})[b] = c / exams.get(b, 1)
                counts[sig] = counts.get(sig, 0) + c
        finally:
            con.close()
        bins = sorted(exams)
        return {
            "train": train,
            "plasticity": plast,
            "evals": evals,
            "episodes": episodes,
            "restarts": restarts,
            "deaths": {
                "hours": [(b + 0.5) * wd / 3600 for b in bins],
                "rates": {s: [round(r.get(b, 0.0), 4) for b in bins] for s, r in sorted(rates.items(), key=lambda kv: -counts[kv[0]])},
                "counts": counts,
            },
        }

    def play(self, net: str, tier: int | None, seed: int | None) -> dict:
        params, meta = self.net(net)
        tier = int(meta.get("tier", 0)) if tier is None else tier
        seed = eval_seed(tier, int(self.rng.integers(0, 10))) if seed is None else seed
        level = self.cache.get(tier, seed)
        return {"level": level.to_json(), "net": net, "meta": meta, "run": rollout(params, level)}

    def race(self, a: str, b: str, tier: int | None, seed: int | None) -> dict:
        pa, ma = self.net(a)
        pb, mb = self.net(b)
        tier = int(max(ma.get("tier", 0), mb.get("tier", 0))) if tier is None else tier
        seed = eval_seed(tier, int(self.rng.integers(0, 10))) if seed is None else seed
        level = self.cache.get(tier, seed)
        return {
            "level": level.to_json(),
            "a": {"net": a, "meta": ma, "run": rollout(pa, level)},
            "b": {"net": b, "meta": mb, "run": rollout(pb, level)},
        }


def make_handler(view: RunView):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):  # keep the training console clean
            pass

        def _send(self, code: int, body: bytes, ctype: str) -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _json(self, obj, code: int = 200) -> None:
            self._send(code, json.dumps(obj, default=_jsonable).encode(), "application/json")

        def do_GET(self):
            u = urlparse(self.path)
            q = {k: v[0] for k, v in parse_qs(u.query).items()}

            def opt_int(k):
                return int(q[k]) if q.get(k, "") != "" else None

            try:
                if u.path in ("/", "/index.html"):
                    with open(os.path.join(WEB, "index.html"), "rb") as f:
                        self._send(200, f.read(), "text/html; charset=utf-8")
                elif u.path == "/api/summary":
                    self._json(view.summary())
                elif u.path == "/api/series":
                    self._json(view.series())
                elif u.path == "/api/hof":
                    self._json(view.hof())
                elif u.path == "/api/play":
                    self._json(view.play(q.get("net", "latest"), opt_int("tier"), opt_int("seed")))
                elif u.path == "/api/race":
                    self._json(view.race(q["a"], q["b"], opt_int("tier"), opt_int("seed")))
                else:
                    self._json({"error": "not found"}, 404)
            except FileNotFoundError as e:
                self._json({"error": f"not found: {e}"}, 404)
            except Exception as e:  # report, don't kill the server thread
                self._json({"error": f"{type(e).__name__}: {e}"}, 500)

    return Handler


def _jsonable(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))


def serve(run_dir: str, port: int = 8765, background: bool = False) -> ThreadingHTTPServer:
    httpd = ThreadingHTTPServer(("127.0.0.1", port), make_handler(RunView(run_dir)))
    if background:
        threading.Thread(target=httpd.serve_forever, daemon=True).start()
    else:
        httpd.serve_forever()
    return httpd
