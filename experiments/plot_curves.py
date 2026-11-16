"""Plots the ablation learning curves as SVG (standard library + numpy).

    uv run python experiments/plot_curves.py runs/ablation/results.json docs/ablation_curves.svg

For each variant: the champion's mean apples (over the K training episodes of each generation),
averaged across seeds and smoothed over a sliding window of 9 generations, with the inter-seed
min–max envelope shown translucent.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

COLORS = ["#2563eb", "#16a34a", "#dc2626", "#9333ea", "#d97706", "#0891b2"]


def smooth(y: np.ndarray, k: int = 9) -> np.ndarray:
    pad = np.pad(y, (k // 2, k // 2), mode="edge")
    return np.convolve(pad, np.ones(k) / k, mode="valid")


def main(src: str, dst: str) -> None:
    results = json.loads(Path(src).read_text())
    W, H, L, R, T, B = 760, 420, 56, 230, 30, 44
    curves = {}
    for name, runs in results.items():
        a = np.array([[c[0] for c in r["curve"]] for r in runs])  # seeds × generations
        a = np.array([smooth(row) for row in a])
        curves[name] = (a.mean(0), a.min(0), a.max(0))
    n_gen = max(len(c[0]) for c in curves.values())
    ymax = max(c[2].max() for c in curves.values()) * 1.05
    X = lambda g: L + g / (n_gen - 1) * (W - L - R)          # noqa: E731
    Y = lambda v: H - B - v / ymax * (H - T - B)              # noqa: E731

    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" font-family="sans-serif" font-size="12">',
           f'<rect width="{W}" height="{H}" fill="white"/>']
    step = 10 if ymax < 80 else 20
    for v in range(0, int(ymax) + 1, step):
        out.append(f'<line x1="{L}" x2="{W - R}" y1="{Y(v):.1f}" y2="{Y(v):.1f}" stroke="#e5e7eb"/>')
        out.append(f'<text x="{L - 6}" y="{Y(v) + 4:.1f}" text-anchor="end" fill="#6b7280">{v}</text>')
    for g in range(0, n_gen, 50):
        out.append(f'<text x="{X(g):.1f}" y="{H - B + 16}" text-anchor="middle" fill="#6b7280">{g}</text>')
    out.append(f'<text x="{(L + W - R) / 2}" y="{H - 8}" text-anchor="middle" fill="#374151">generation</text>')
    out.append(f'<text x="{L}" y="18" fill="#111827" font-weight="bold">Champion apples per generation '
               f'(mean of {len(next(iter(results.values())))} seeds, min–max envelope)</text>')
    for i, (name, (m, lo, hi)) in enumerate(curves.items()):
        col = COLORS[i % len(COLORS)]
        g = np.arange(len(m))
        band = [f"{X(x):.1f},{Y(v):.1f}" for x, v in zip(g, hi)] + \
               [f"{X(x):.1f},{Y(v):.1f}" for x, v in zip(g[::-1], lo[::-1])]
        out.append(f'<polygon points="{" ".join(band)}" fill="{col}" fill-opacity="0.12"/>')
        line = " ".join(f"{X(x):.1f},{Y(v):.1f}" for x, v in zip(g, m))
        out.append(f'<polyline points="{line}" fill="none" stroke="{col}" stroke-width="2"/>')
        ly = T + 20 + i * 22
        out.append(f'<line x1="{W - R + 14}" x2="{W - R + 34}" y1="{ly}" y2="{ly}" stroke="{col}" stroke-width="3"/>')
        out.append(f'<text x="{W - R + 40}" y="{ly + 4}" fill="#111827">{name}</text>')
    out.append("</svg>")
    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    Path(dst).write_text("\n".join(out))
    print(f"wrote: {dst}")


if __name__ == "__main__":
    main(*(sys.argv[1:3] if len(sys.argv) >= 3 else ("runs/ablation/results.json", "docs/ablation_curves.svg")))
