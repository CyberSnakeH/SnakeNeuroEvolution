"""Ablation study: each variant × several seeds, final held-out evaluation.

    uv run python experiments/ablation.py --generations 200 --seeds 0 1 2

Writes runs/ablation/results.json and a Markdown table to standard output.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

from snake_neat.config import GameConfig, NEATConfig
from snake_neat.evaluation import Evaluator, benchmark
from snake_neat.population import Population

BASE_GAME = GameConfig()
BASE_NEAT = NEATConfig()

VARIANTS: dict[str, tuple[GameConfig, NEATConfig]] = {
    "NEAT (baseline)": (BASE_GAME, BASE_NEAT),
    "δ unnormalized (N=1)": (BASE_GAME, replace(BASE_NEAT, distance_norm="none")),
    "no speciation": (BASE_GAME, replace(BASE_NEAT, compat_threshold=1e9, target_species=1)),
    "fixed topology (weights only)": (BASE_GAME, replace(BASE_NEAT, p_add_node=0.0, p_add_conn=0.0, p_toggle=0.0)),
    "+ food compass": (replace(BASE_GAME, food_compass=True), BASE_NEAT),
}


def run(game: GameConfig, neat: NEATConfig, seed: int, generations: int, episodes: int) -> dict:
    pop = Population(neat, game.n_inputs, game.n_outputs, seed=seed)
    curve = []
    t0 = time.perf_counter()
    with Evaluator(game, episodes, seed=seed) as ev:
        for _ in range(generations):
            st = pop.run_generation(lambda gs: ev(gs, pop.generation))
            curve.append((st.best_apples, st.mean_apples, st.n_species, st.champion_hidden))
    # Champion of the last generation, measured on 200 never-seen test seeds.
    final = st.champion
    res = benchmark(final, game, 200)
    return {"seed": seed, "time_s": time.perf_counter() - t0, "test": res,
            "curve": curve, "hidden": final.n_hidden, "conns": final.n_enabled}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--generations", type=int, default=200)
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--only", nargs="*", help="subset of variants (prefixes)")
    ap.add_argument("--out", default="runs/ablation")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    results: dict[str, list[dict]] = {}
    for name, (game, neat) in VARIANTS.items():
        if args.only and not any(name.startswith(p) for p in args.only):
            continue
        results[name] = []
        for seed in args.seeds:
            r = run(game, neat, seed, args.generations, args.episodes)
            results[name].append(r)
            print(f"[{name}] seed {seed}: {r['test']['apples_mean']:.1f} ± {r['test']['apples_ci95']:.1f} apples "
                  f"({r['hidden']} hidden, {r['time_s']:.0f}s)", flush=True)
        (out / "results.json").write_text(json.dumps(results, indent=1, ensure_ascii=False))

    print("\n| Variant | apples (test, mean over seeds) | inter-seed std. dev. | gen. to reach 20 apples | hidden nodes |")
    print("|---|---|---|---|---|")
    for name, rs in results.items():
        means = np.array([r["test"]["apples_mean"] for r in rs])
        reach = []
        for r in rs:
            hits = [i for i, c in enumerate(r["curve"]) if c[0] >= 20]
            reach.append(hits[0] if hits else np.nan)
        hidden = np.mean([r["hidden"] for r in rs])
        print(f"| {name} | {means.mean():.1f} | {means.std(ddof=1) if len(means) > 1 else 0:.1f} "
              f"| {np.nanmedian(reach) if not np.all(np.isnan(reach)) else '—'} | {hidden:.1f} |")


if __name__ == "__main__":
    main()
