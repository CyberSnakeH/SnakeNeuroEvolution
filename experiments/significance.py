"""Welch tests of every ablation variant against the baseline, with Holm–Bonferroni correction.

    uv run python experiments/significance.py [runs/ablation/results.json]

For each variant, the unit of observation is the held-out test score (mean apples over 200 games) of
the final champion of one training seed. Welch's t statistic, the Welch–Satterthwaite degrees of
freedom and the two-sided p-value are computed exactly; the Student tail probability is obtained from
the regularized incomplete beta function, p = I_{ν/(ν+t²)}(ν/2, 1/2), evaluated with Lentz's continued
fraction (standard library only).
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

BASELINE = "NEAT (baseline)"


def _betacf(a: float, b: float, x: float, eps: float = 1e-15, max_iter: int = 10_000) -> float:
    """Continued fraction of the incomplete beta function (modified Lentz method)."""
    tiny = 1e-300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    d = 1.0 / (d if abs(d) > tiny else tiny)
    h = d
    for m in range(1, max_iter + 1):
        m2 = 2 * m
        for aa in (m * (b - m) * x / ((qam + m2) * (a + m2)),
                   -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))):
            d = 1.0 + aa * d
            d = 1.0 / (d if abs(d) > tiny else tiny)
            c = 1.0 + aa / c
            c = c if abs(c) > tiny else tiny
            h *= d * c
        if abs(d * c - 1.0) < eps:
            return h
    raise RuntimeError("continued fraction did not converge")


def betainc(a: float, b: float, x: float) -> float:
    """Regularized incomplete beta function I_x(a, b)."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    log_front = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log1p(-x)
    if x < (a + 1.0) / (a + b + 2.0):
        return math.exp(log_front) * _betacf(a, b, x) / a
    return 1.0 - math.exp(log_front) * _betacf(b, a, 1.0 - x) / b


def welch(x: list[float], y: list[float]) -> tuple[float, float, float]:
    """Welch's t, Welch–Satterthwaite degrees of freedom, two-sided p-value."""
    n1, n2 = len(x), len(y)
    m1, m2 = sum(x) / n1, sum(y) / n2
    v1 = sum((a - m1) ** 2 for a in x) / (n1 - 1) / n1
    v2 = sum((a - m2) ** 2 for a in y) / (n2 - 1) / n2
    t = (m1 - m2) / math.sqrt(v1 + v2)
    nu = (v1 + v2) ** 2 / (v1 ** 2 / (n1 - 1) + v2 ** 2 / (n2 - 1))
    return t, nu, betainc(nu / 2.0, 0.5, nu / (nu + t * t))


def holm(pvalues: list[float]) -> list[float]:
    """Holm–Bonferroni adjusted p-values, in the original order."""
    m = len(pvalues)
    order = sorted(range(m), key=lambda i: pvalues[i])
    adjusted, running = [0.0] * m, 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * pvalues[i]))
        adjusted[i] = running
    return adjusted


def main(path: str = "runs/ablation/results.json") -> None:
    results = json.loads(Path(path).read_text())
    scores = {name: [r["test"]["apples_mean"] for r in runs] for name, runs in results.items()}
    base = scores[BASELINE]
    others = [name for name in scores if name != BASELINE]
    tests = [welch(base, scores[name]) for name in others]
    adjusted = holm([p for _, _, p in tests])

    def mean_sd(v: list[float]) -> tuple[float, float]:
        m = sum(v) / len(v)
        return m, math.sqrt(sum((a - m) ** 2 for a in v) / (len(v) - 1))

    print("| Variant | mean | std. | t | ν | p | Holm p |")
    print("|---|---|---|---|---|---|---|")
    m, sd = mean_sd(base)
    print(f"| {BASELINE} | {m:.1f} | {sd:.1f} | | | | |")
    for name, (t, nu, p), q in zip(others, tests, adjusted):
        m, sd = mean_sd(scores[name])
        print(f"| {name} | {m:.1f} | {sd:.1f} | {t:.2f} | {nu:.2f} | {p:.3f} | {q:.3f} |")


if __name__ == "__main__":
    main(*sys.argv[1:2])
