"""Render gameplay videos from hall-of-fame snapshots, for the README and for posts.

    python scripts/media.py

Writes results/learning.mp4 (the same unseen level played by the agent at
several ages), results/race.mp4 and results/race.gif (an early and a late
snapshot racing on one level). Every frame comes from a deterministic replay
of a saved snapshot on the game's own physics, so nothing here is staged.
Needs Pillow and ffmpeg on the PATH.
"""

from __future__ import annotations

import math
import os
import shutil
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from PIL import Image, ImageDraw, ImageFont  # noqa: E402

from endless.agent import load_net  # noqa: E402
from endless.env import RAY_ANGLES_DEG  # noqa: E402
from endless.evaluate import rollout  # noqa: E402
from endless.levels import LevelCache, eval_seed  # noqa: E402

OUT = os.path.join(ROOT, "results")
RUN = "runs/cbp-s1"  # snapshots every 15 active minutes
LEARNING = [("h0000", "untrained"), ("h0001", "15 minutes"), ("h0002", "30 minutes"), ("h0004", "1 hour"),
            ("h0006", "1.5 hours"), ("h0008", "2 hours"), ("h0118", "29.5 hours")]
TIER = 6

BG, GROUND, BLOCK, OVERHANG, MOVING = (18, 21, 27), (58, 66, 82), (208, 214, 226), (227, 179, 65), (255, 123, 114)
AGENT, GHOST, TEXT, MUTED, GOOD = (108, 182, 255), (195, 155, 255), (230, 232, 238), (139, 147, 167), (95, 211, 141)


def font(name: str, size: int):
    for path in (os.path.join(os.environ.get("WINDIR", "C:/Windows"), "Fonts", name), name):
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default()


def draw_frame(img, level, runs, t: int, colors, labels, box, caption: str | None, sub: str | None):
    """One tick. `box` is the stage rectangle (x0, y0, x1, y1) inside the image."""
    d = ImageDraw.Draw(img, "RGBA")
    x0, y0, x1, y1 = box
    W, H = x1 - x0, y1 - y0
    scale = W / 34.0
    ground_y = y1 - 0.16 * H
    L = level
    px = t * L["speed"]
    cam = px - 6.0

    def SX(x):
        return x0 + (x - cam) * scale

    def SY(y):
        return ground_y - y * scale

    def rect(a, b, c, e, fill):
        if c > a and e > b:
            d.rectangle((a, b, c, e), fill=fill)

    d.rectangle(box, fill=BG)
    # ground, with gaps cut out
    x = cam
    for a, b in sorted(L["gaps"]):
        if b < cam or a > cam + 36:
            continue
        if a > x:
            rect(SX(x), ground_y, SX(a), y1, GROUND)
        x = max(x, b)
    rect(SX(x), ground_y, SX(cam + 36), y1, GROUND)
    if L["length"] < cam + 36:
        rect(SX(L["length"]), ground_y - 6 * scale, SX(L["length"]) + 4, ground_y, GOOD)
    for k, a, b, c, top, u in L["boxes"]:
        s = u * t
        a, b = a - s, b - s
        if b < cam or a > cam + 36:
            continue
        col = OVERHANG if k == 2 else MOVING if k == 3 else BLOCK
        rect(max(SX(a), x0), max(SY(min(top, 30)), y0), min(SX(b), x1), SY(c), col)
    # agents, the lead one with its rays
    for i, r in enumerate(runs):
        fr = r["frames"]
        if not fr:
            continue
        j = min(max(0, t - fr[0][0]), len(fr) - 1)
        ft, fy, fduck = fr[j]
        dead = t >= fr[-1][0] and not r["success"]
        h = 1.0 if fduck else 2.0
        ax = ft * L["speed"]
        if i == 0 and not dead and r["rays"]:
            k = max((n for n, q in enumerate(r["rays"]) if q[0] <= ft), default=None)
            if k is not None:
                _, dist, kind = r["rays"][k]
                ox, oy = ax + 1.0, fy + h / 2
                for deg, dd, kk in zip(RAY_ANGLES_DEG, dist, kind):
                    rad = math.radians(deg)
                    col = (255, 123, 114, 150) if kk >= 2 else (227, 179, 65, 160) if kk == 1 else (108, 182, 255, 50)
                    d.line((SX(ox), SY(oy), SX(ox + math.cos(rad) * dd), SY(oy + math.sin(rad) * dd)), fill=col, width=2)
        col = colors[i]
        alpha = 90 if dead else (255 if i == 0 else 200)
        rect(SX(ax), SY(fy + h), SX(ax + 1.0), SY(fy), col + (alpha,))
    # text
    big, mid, small = font("segoeuib.ttf", 34), font("segoeui.ttf", 24), font("segoeui.ttf", 20)
    if caption:
        d.text((40, 30), caption, font=big, fill=TEXT)
    if sub:
        d.text((40, 78), sub, font=mid, fill=MUTED)
    y = y0 + 14
    for i, r in enumerate(runs):
        end = r["frames"][-1][0] if r["frames"] else 0
        if t >= end:
            status = "finished the level" if r["success"] else "died: " + r["death_sig"].replace("/", ", ")
        else:
            status = f"{min(100, round(100 * t * L['speed'] / L['length']))}%"
        d.rectangle((x0 + 18, y + 8, x0 + 32, y + 22), fill=colors[i])
        d.text((x0 + 42, y), f"{labels[i]}: {status}", font=small, fill=TEXT)
        y += 30
    d.text((x1 - 330, y0 + 14), f"unseen level, tier {L['tier']}", font=small, fill=MUTED)


class Encoder:
    def __init__(self, path: str, size: tuple[int, int], fps: int = 60):
        ff = shutil.which("ffmpeg")
        if not ff:
            raise SystemExit("ffmpeg not found on PATH")
        self.size = size
        self.p = subprocess.Popen(
            [ff, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{size[0]}x{size[1]}", "-r", str(fps),
             "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", "-movflags", "+faststart", path],
            stdin=subprocess.PIPE,
        )

    def write(self, img) -> None:
        self.p.stdin.write(img.tobytes())

    def close(self) -> None:
        self.p.stdin.close()
        self.p.wait()


def end_tick(r) -> int:
    return r["frames"][-1][0] if r["frames"] else 0


def pick_level(nets, cache):
    """A held-out tier-6 level the trained agent finishes, where the early agents die early on."""
    final = nets[-1]
    for i in range(20):
        lv = cache.get(TIER, eval_seed(TIER, i))
        r_final = rollout(final, lv)
        if not r_final["success"]:
            continue
        r_early = rollout(nets[1], lv)
        if r_early["progress"] < 0.3:
            return lv
    return cache.get(TIER, eval_seed(TIER, 0))


def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    cache = LevelCache()
    nets, metas = [], []
    for name, _ in LEARNING:
        p, m = load_net(os.path.join(ROOT, RUN, "hof", f"{name}.npz"))
        nets.append(p)
        metas.append(m)
    level = pick_level(nets, cache)
    L = level.to_json()
    print(f"level: tier {level.tier}, seed {level.seed}")

    # 1. The same unseen level, played by the agent at each age.
    size = (1280, 720)
    enc = Encoder(os.path.join(OUT, "learning.mp4"), size)
    for (name, age), p, m in zip(LEARNING, nets, metas):
        r = rollout(p, level)
        last = min(end_tick(r) + 60, 18 * 60)
        steps = f"{m['steps'] / 1e6:.1f}M steps" if m["steps"] else "0 steps"
        for t in range(1, last + 1):
            img = Image.new("RGB", size, (12, 14, 18))
            draw_frame(img, L, [r], t, [AGENT], [f"after {age}"], (0, 130, 1280, 720),
                       f"After {age} of training", f"{steps} on one laptop CPU core, same unseen level every time")
            enc.write(img)
        print(f"  {name} ({age}): {'finished' if r['success'] else r['death_sig']} at {r['progress']:.0%}")
    enc.close()

    # 2. A ghost race: 30 minutes against 29.5 hours, on the same level at the same time.
    early, late = rollout(nets[2], level), rollout(nets[-1], level)
    last = max(end_tick(early), end_tick(late)) + 60
    for path, size, step, cap in ((os.path.join(OUT, "race.mp4"), (1280, 720), 1, True), (os.path.join(OUT, "race_small.mp4"), (640, 240), 1, False)):
        enc = Encoder(path, size)
        for t in range(1, last + 1, step):
            img = Image.new("RGB", size, (12, 14, 18))
            box = (0, 130, size[0], size[1]) if cap else (0, 0, size[0], size[1])
            draw_frame(img, L, [late, early], t, [AGENT, GHOST], ["29.5 hours", "30 minutes (ghost)"], box,
                       "Ghost race: 30 minutes against 29.5 hours of training" if cap else None,
                       "same unseen level, both agents start together" if cap else None)
            enc.write(img)
        enc.close()
    ff = shutil.which("ffmpeg")
    subprocess.run([ff, "-y", "-loglevel", "error", "-i", os.path.join(OUT, "race_small.mp4"), "-vf",
                    "fps=25,split[a][b];[a]palettegen=max_colors=48[p];[b][p]paletteuse=dither=none",
                    os.path.join(OUT, "race.gif")], check=True)
    os.remove(os.path.join(OUT, "race_small.mp4"))
    for f in ("learning.mp4", "race.mp4", "race.gif"):
        print(f, f"{os.path.getsize(os.path.join(OUT, f)) / 2**20:.1f} MB")


if __name__ == "__main__":
    main()
