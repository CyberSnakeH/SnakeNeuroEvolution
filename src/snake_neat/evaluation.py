"""Genome evaluation: fitness, episodes, multi-process parallelism.

Episode fitness: F = 1000 · apples + survival steps.
Since the number of steps between two apples is bounded by the hunger limit h
(h ≤ W·H < 1000 by default), F induces a quasi-lexicographic order: eat first,
survive second. Survival alone (circling) is capped at h.

Variance reduction: each genome is evaluated over K episodes; within a given
generation, *all* genomes see the same K seeds (common random numbers).
Comparisons between individuals then reflect the policy rather than the luck
of food placement. Seeds change every generation to avoid overfitting to one
layout, and elites are re-evaluated, which corrects the maximum bias
("winner's curse") of a noisy estimate carried over from one generation to
the next.
"""

from __future__ import annotations

import multiprocessing as mp
import os
from dataclasses import dataclass

import numpy as np

from .config import GameConfig
from .game import SnakeGame
from .genome import Genome
from .network import FeedForwardNetwork

APPLE_REWARD = 1000.0


def fitness(apples: int, steps: int) -> float:
    return APPLE_REWARD * apples + steps


@dataclass(frozen=True)
class EpisodeResult:
    apples: int
    steps: int
    death: str | None

    @property
    def fitness(self) -> float:
        return fitness(self.apples, self.steps)


def play_episode(net: FeedForwardNetwork, cfg: GameConfig, seed: int) -> EpisodeResult:
    game = SnakeGame(cfg, seed)
    while game.alive:
        game.step(net.decide(game.observe()))
    return EpisodeResult(game.apples, game.steps, game.death)


def evaluate_network(net: FeedForwardNetwork, cfg: GameConfig, seeds: list[int]) -> dict[str, float]:
    res = [play_episode(net, cfg, s) for s in seeds]
    return {
        "fitness": float(np.mean([r.fitness for r in res])),
        "apples": float(np.mean([r.apples for r in res])),
        "steps": float(np.mean([r.steps for r in res])),
        "max_apples": int(max(r.apples for r in res)),
    }


def _worker(task: tuple[FeedForwardNetwork, GameConfig, list[int]]) -> dict[str, float]:
    return evaluate_network(*task)


class Evaluator:
    """Evaluates a list of genomes, in parallel if `workers > 1`."""

    def __init__(self, cfg: GameConfig, episodes: int = 3, workers: int | None = None, seed: int = 0) -> None:
        self.cfg = cfg
        self.episodes = episodes
        self.workers = workers if workers else (os.cpu_count() or 1)
        self.seed = seed
        self._pool = None

    def episode_seeds(self, generation: int) -> list[int]:
        ss = np.random.SeedSequence([self.seed, generation])
        return [int(x) for x in ss.generate_state(self.episodes)]

    def __call__(self, genomes: list[Genome], generation: int) -> None:
        seeds = self.episode_seeds(generation)
        tasks = [(FeedForwardNetwork.from_genome(g), self.cfg, seeds) for g in genomes]
        if self.workers > 1:
            if self._pool is None:
                # "spawn": safe even if the parent process has threads (GUI).
                self._pool = mp.get_context("spawn").Pool(self.workers)
            chunk = max(1, len(tasks) // (4 * self.workers))
            results = self._pool.map(_worker, tasks, chunksize=chunk)
        else:
            results = [_worker(t) for t in tasks]
        for g, r in zip(genomes, results):
            g.fitness = r["fitness"]
            g.stats = r

    def close(self) -> None:
        if self._pool is not None:
            self._pool.terminate()
            self._pool.join()
            self._pool = None

    def __enter__(self) -> "Evaluator":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def benchmark(genome: Genome, cfg: GameConfig, n_episodes: int, first_seed: int = 10**6) -> dict[str, float]:
    """Held-out evaluation on reserved seeds; 95% CI (normal approximation)."""
    net = FeedForwardNetwork.from_genome(genome)
    res = [play_episode(net, cfg, first_seed + k) for k in range(n_episodes)]
    apples = np.array([r.apples for r in res], dtype=float)
    deaths: dict[str, int] = {}
    for r in res:
        deaths[r.death or "?"] = deaths.get(r.death or "?", 0) + 1
    half = 1.96 * apples.std(ddof=1) / np.sqrt(n_episodes) if n_episodes > 1 else float("nan")
    return {
        "episodes": n_episodes,
        "apples_mean": float(apples.mean()),
        "apples_ci95": float(half),
        "apples_median": float(np.median(apples)),
        "apples_max": int(apples.max()),
        "steps_mean": float(np.mean([r.steps for r in res])),
        "deaths": deaths,
    }
