"""Command-line entry point.

    snake-neat                     graphical interface (live evolution)
    snake-neat gui [options]       same, with options
    snake-neat play FILE.json      graphical interface, replays a saved genome
    snake-neat train [options]     headless training (CSV log + genomes)
    snake-neat eval FILE.json      evaluation on held-out seeds (95% CI)
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

from .config import GameConfig, NEATConfig
from .evaluation import Evaluator, benchmark
from .persistence import load_genome, save_genome
from .population import Population


def _add_common(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group("environment")
    g.add_argument("--width", type=int, default=15)
    g.add_argument("--height", type=int, default=15)
    g.add_argument("--hunger", type=int, default=0, help="steps without eating before starving (0 = width×height)")
    g.add_argument("--compass", action="store_true",
                   help="adds 2 inputs: egocentric head→apple vector (on top of the 24 rays)")
    n = p.add_argument_group("NEAT")
    n.add_argument("--pop", type=int, default=150, help="population size")
    n.add_argument("--species", type=int, default=10, help="target number of species")
    n.add_argument("--episodes", type=int, default=3, help="episodes per evaluation")
    n.add_argument("--workers", type=int, default=0, help="evaluation processes (0 = number of cores)")
    n.add_argument("--seed", type=int, default=0)


def _configs(args: argparse.Namespace) -> tuple[GameConfig, NEATConfig]:
    game = GameConfig(width=args.width, height=args.height, hunger=args.hunger, food_compass=args.compass)
    neat = NEATConfig(population_size=args.pop, target_species=args.species)
    return game, neat


def cmd_gui(args: argparse.Namespace) -> None:
    from .gui import run_gui

    game, neat = _configs(args)
    run_gui(game, neat, seed=args.seed, episodes=args.episodes, workers=args.workers,
            genome_path=getattr(args, "genome", None), view=getattr(args, "view", "champion"))


def cmd_train(args: argparse.Namespace) -> None:
    game, neat = _configs(args)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(
        {"game": game.to_dict(), "neat": neat.to_dict(), "seed": args.seed, "episodes": args.episodes}, indent=1))

    pop = Population(neat, game.n_inputs, game.n_outputs, seed=args.seed)
    t_start = time.perf_counter()
    with Evaluator(game, args.episodes, args.workers, seed=args.seed) as ev, \
            open(out / "history.csv", "w", newline="") as fh:
        writer = None
        for _ in range(args.generations):
            t0 = time.perf_counter()
            stats = pop.run_generation(lambda gs: ev(gs, pop.generation))
            stats.elapsed = time.perf_counter() - t0
            row = stats.row()
            if writer is None:
                writer = csv.DictWriter(fh, fieldnames=list(row))
                writer.writeheader()
            writer.writerow(row)
            fh.flush()
            meta = {"generation": stats.generation, "fitness": stats.best_fitness}
            save_genome(out / "champion.json", stats.champion, game, meta)
            if stats.generation % args.checkpoint == 0:
                save_genome(out / f"champion_g{stats.generation:04d}.json", stats.champion, game, meta)
            print(f"gen {stats.generation:4d} | best F {stats.best_fitness:9.1f} | mean F {stats.mean_fitness:8.1f} "
                  f"| champ. apples {stats.best_apples:5.1f} (max {stats.max_apples:3d}) mean {stats.mean_apples:5.2f} "
                  f"| species {stats.n_species:2d} δt={stats.threshold:.2f} "
                  f"| champ. {stats.champion_hidden} hidden/{stats.champion_conns} conn. | {stats.elapsed:5.2f}s",
                  flush=True)
    print(f"Done in {time.perf_counter() - t_start:.1f}s — genomes in {out}/")
    if args.final_eval:
        res = benchmark(stats.champion, game, args.final_eval)
        (out / "final_eval.json").write_text(json.dumps(res, indent=1))
        print("Final evaluation (held-out seeds):", json.dumps(res, ensure_ascii=False))


def cmd_eval(args: argparse.Namespace) -> None:
    genome, game, meta = load_genome(args.genome)
    res = benchmark(genome, game, args.episodes)
    print(json.dumps({"genome": str(args.genome), "meta": meta, **res}, indent=1, ensure_ascii=False))


def main(argv: list[str] | None = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0].startswith("-"):
        argv.insert(0, "gui")

    parser = argparse.ArgumentParser(prog="snake-neat", description="Snake learned by NEAT (numpy only).")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("gui", help="graphical interface")
    _add_common(p)
    p.add_argument("--genome", help="genome JSON to replay at startup")
    p.add_argument("--view", choices=["champion", "species"], default="champion",
                   help="initial view: the champion alone, or one snake per species")
    p.set_defaults(func=cmd_gui)

    p = sub.add_parser("play", help="replay a genome in the graphical interface")
    p.add_argument("genome")
    _add_common(p)
    p.set_defaults(func=cmd_gui)

    p = sub.add_parser("train", help="headless training")
    _add_common(p)
    p.add_argument("--generations", type=int, default=200)
    p.add_argument("--out", default="runs/default")
    p.add_argument("--checkpoint", type=int, default=25, help="champion checkpoint period (generations)")
    p.add_argument("--final-eval", type=int, default=200, help="final evaluation episodes (0 = none)")
    p.set_defaults(func=cmd_train)

    p = sub.add_parser("eval", help="evaluate a genome on held-out seeds")
    p.add_argument("genome")
    p.add_argument("--episodes", type=int, default=500)
    p.set_defaults(func=cmd_eval)

    args = parser.parse_args(argv)
    args.func(args)
