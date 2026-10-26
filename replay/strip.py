"""Render a recording from record-episodes.js as a strip of frames (PNG), for a quick look.
    python3 replay/strip.py recording.json out.png [episode] [count]
"""
import json
import math
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Polygon

rec = json.load(open(sys.argv[1]))
out = sys.argv[2]
ep = rec["episodes"][int(sys.argv[3]) if len(sys.argv) > 3 else 0]
count = int(sys.argv[4]) if len(sys.argv) > 4 else 12
shapes = rec["shapes"]
frames = ep["frames"]
picks = [round(i * (len(frames) - 1) / (count - 1)) for i in range(count)]
LEFT = {3, 4, 5, 6, 7, 8}
order = [3, 4, 5, 6, 7, 8, 15, 0, 1, 2, 9, 10, 11, 12, 13, 14]


def color(i):
    if i == 2:
        return "#e9b35a"
    if i in (8, 14):
        return "#222"
    if i in (6, 7):
        return "#2c4f72"
    if i in (12, 13):
        return "#3f6f9f"
    if i in (3, 4, 5):
        return "#9c3a30"
    return "#c94a3d"


def rect(cx, cy, a, hx, hy):
    c, s = math.cos(a), math.sin(a)
    pts = [(-hx, -hy), (hx, -hy), (hx, hy), (-hx, hy)]
    return [(cx + c * x - s * y, cy + s * x + c * y) for x, y in pts]


fig, axes = plt.subplots(1, count, figsize=(count * 1.6, 3.2))
for ax, fi in zip(axes, picks):
    f = frames[fi]
    p = f["p"]
    cx = p[0]
    for i in order:
        x, y, a = p[i * 3], p[i * 3 + 1], p[i * 3 + 2]
        sh = shapes[i]
        col = color(i)
        if sh["circle"]:
            ax.add_patch(Circle((x, y), sh["r"], color=col))
        elif sh["capsule"]:
            c, s = math.cos(a), math.sin(a)
            ax_, ay_ = sh["ax"], sh["ay"]
            x1, y1 = x - (c * ax_ - s * ay_), y - (s * ax_ + c * ay_)
            x2, y2 = x + (c * ax_ - s * ay_), y + (s * ax_ + c * ay_)
            ax.plot([x1, x2], [y1, y2], color=col, linewidth=sh["r"] * 2 * 72 / 2.2 * 1.6, solid_capstyle="round")
        else:
            ax.add_patch(Polygon(rect(x, y, a, sh["hx"], sh["hy"]), color=col))
    if ep.get("box"):
        b = ep["box"]
        ax.add_patch(Polygon([(b["x0"], 0), (b["x1"], 0), (b["x1"], b["top"]), (b["x0"], b["top"])], color="#b7a17a"))
    if ep.get("board"):
        b = ep["board"]
        ang = f.get("ba") or 0
        c, sn = math.cos(ang), math.sin(ang)
        cy = b["pivot"] + b["thick"]
        ax.add_patch(Polygon([(b["x"] - 0.05, 0), (b["x"] + 0.05, 0), (b["x"], b["pivot"])], color="#8a7552"))
        pts = [(-b["half"], -b["thick"]), (b["half"], -b["thick"]), (b["half"], b["thick"]), (-b["half"], b["thick"])]
        ax.add_patch(Polygon([(b["x"] + c * x - sn * (y + b["thick"]) , b["pivot"] + sn * x + c * (y + b["thick"])) for x, y in pts], color="#b7a17a"))
    if f.get("mx") is not None:
        ax.add_patch(Circle((f["mx"], f["my"]), 0.12, fill=False, ls="--", color="#2a7fb8"))
    ax.axhline(0, color="#888", lw=1)
    ax.set_xlim(cx - 0.6, cx + 0.6)
    ax.set_ylim(-0.05, 2.0)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(f"{fi / rec['fps']:.1f}s {'UP' if f['up'] else ''}", fontsize=8)
plt.tight_layout()
plt.savefig(out, dpi=70)
print("wrote", out)
