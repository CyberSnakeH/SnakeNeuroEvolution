"""NEAT evolutionary loop: speciation, fitness sharing, reproduction.

Explicit fitness sharing. For an individual i of species S:
    f'_i = f_i / |S|
so that the total adjusted fitness of S is Σ_{i∈S} f'_i = f̄_S (the mean).
The number of offspring allotted to S is
    n_S = N · f̄_S / Σ_T f̄_T,
rounded by the largest-remainder method (Σ n_S = N exactly). A species thus
cannot take over the population merely through its size: this is what
protects topological innovations, initially poor performers, while their
weights are being optimized.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from .config import NEATConfig
from .genome import Genome, InnovationTracker


@dataclass
class Species:
    id: int
    representative: Genome
    created: int
    members: list[Genome] = field(default_factory=list)
    best_fitness: float = -math.inf
    last_improved: int = 0


@dataclass
class GenerationStats:
    generation: int
    best_fitness: float
    mean_fitness: float
    best_apples: float
    mean_apples: float
    max_apples: int
    n_species: int
    species_sizes: dict[int, int]
    threshold: float
    champion: Genome
    champion_hidden: int
    champion_conns: int
    mean_hidden: float
    species_champions: dict[int, Genome] = field(default_factory=dict)  # by decreasing species size
    elapsed: float = 0.0

    def row(self) -> dict[str, float]:
        return {
            "generation": self.generation,
            "best_fitness": round(self.best_fitness, 3),
            "mean_fitness": round(self.mean_fitness, 3),
            "best_apples": round(self.best_apples, 3),
            "mean_apples": round(self.mean_apples, 3),
            "max_apples": self.max_apples,
            "n_species": self.n_species,
            "threshold": round(self.threshold, 4),
            "champion_hidden": self.champion_hidden,
            "champion_conns": self.champion_conns,
            "mean_hidden": round(self.mean_hidden, 3),
            "elapsed_s": round(self.elapsed, 3),
        }


class Population:
    def __init__(self, cfg: NEATConfig, n_inputs: int, n_outputs: int, seed: int | None = None) -> None:
        self.cfg = cfg
        self.rng = np.random.default_rng(seed)
        self.tracker = InnovationTracker(n_inputs, n_outputs)
        self._next_key = 0
        self._next_species = 0
        self.generation = 0
        self.threshold = cfg.compat_threshold
        self.species: dict[int, Species] = {}
        self.best_ever: Genome | None = None
        self.genomes = [
            Genome.initial(self._new_key(), n_inputs, n_outputs, cfg, self.tracker, self.rng)
            for _ in range(cfg.population_size)
        ]
        self.speciate()

    def _new_key(self) -> int:
        self._next_key += 1
        return self._next_key - 1

    # ------------------------------------------------------------- cycle
    def run_generation(self, evaluate: Callable[[list[Genome]], None]) -> GenerationStats:
        """Evaluate the current generation, produce the next one, return the statistics."""
        evaluate(self.genomes)
        stats = self._statistics()
        self._reproduce()
        self.generation += 1
        self.speciate()
        return stats

    # -------------------------------------------------------- speciation
    def speciate(self) -> None:
        cfg = self.cfg
        for s in self.species.values():
            s.members = []
        for g in self.genomes:
            for s in self.species.values():
                if g.distance(s.representative, cfg) < self.threshold:
                    s.members.append(g)
                    g.species_id = s.id
                    break
            else:
                sid = self._next_species
                self._next_species += 1
                self.species[sid] = Species(sid, g, created=self.generation, members=[g],
                                            last_improved=self.generation)
                g.species_id = sid
        self.species = {k: s for k, s in self.species.items() if s.members}
        # Representative for the next generation: a random member of the current one.
        for s in self.species.values():
            s.representative = s.members[int(self.rng.integers(len(s.members)))]
        # Adaptive threshold (multiplicative, hence independent of the scale of δ):
        # feedback control of the number of species towards the target.
        n = len(self.species)
        if n < cfg.target_species:
            self.threshold = max(cfg.min_threshold, self.threshold * (1.0 - cfg.threshold_rate))
        elif n > cfg.target_species:
            self.threshold *= 1.0 + cfg.threshold_rate

    # --------------------------------------------------------- statistics
    def _statistics(self) -> GenerationStats:
        f = np.array([g.fitness for g in self.genomes], dtype=float)
        apples = np.array([g.stats.get("apples", 0.0) for g in self.genomes])
        champ = self.genomes[int(np.argmax(f))]
        by_size = sorted(self.species.values(), key=lambda s: len(s.members), reverse=True)
        species_champions = {s.id: max(s.members, key=lambda g: g.fitness).copy() for s in by_size}
        if self.best_ever is None or champ.fitness > self.best_ever.fitness:
            self.best_ever = champ.copy()
        return GenerationStats(
            generation=self.generation,
            best_fitness=float(f.max()),
            mean_fitness=float(f.mean()),
            best_apples=float(champ.stats.get("apples", 0.0)),
            mean_apples=float(apples.mean()),
            max_apples=int(max(g.stats.get("max_apples", 0) for g in self.genomes)),
            n_species=len(self.species),
            species_sizes={s.id: len(s.members) for s in self.species.values()},
            threshold=self.threshold,
            champion=champ.copy(),
            champion_hidden=champ.n_hidden,
            champion_conns=champ.n_enabled,
            mean_hidden=float(np.mean([g.n_hidden for g in self.genomes])),
            species_champions=species_champions,
        )

    # -------------------------------------------------------- reproduction
    def _reproduce(self) -> None:
        cfg, rng, gen = self.cfg, self.rng, self.generation
        species = list(self.species.values())
        champion = max(self.genomes, key=lambda g: g.fitness)

        for s in species:
            s.members.sort(key=lambda g: g.fitness, reverse=True)
            if s.members[0].fitness > s.best_fitness:
                s.best_fitness = s.members[0].fitness
                s.last_improved = gen

        # Extinction of stagnant species (except the best ones and the champion's).
        ranked = sorted(species, key=lambda s: s.best_fitness, reverse=True)
        protected = {s.id for s in ranked[: cfg.species_elitism]} | {champion.species_id}
        survivors = [s for s in species
                     if s.id in protected or gen - s.last_improved <= cfg.stagnation_limit]
        self.species = {s.id: s for s in survivors}

        # Allocation n_S ∝ f̄_S (explicit sharing), largest remainders.
        N = cfg.population_size
        means = np.array([np.mean([m.fitness for m in s.members]) for s in survivors])
        means -= min(0.0, means.min())
        total = means.sum()
        quotas = N * means / total if total > 0 else np.full(len(survivors), N / len(survivors))
        counts = np.floor(quotas).astype(int)
        remainder = N - int(counts.sum())
        counts[np.argsort(-(quotas - counts), kind="stable")[:remainder]] += 1
        ci = next(i for i, s in enumerate(survivors) if s.id == champion.species_id)
        if counts[ci] == 0:
            counts[int(np.argmax(counts))] -= 1
            counts[ci] = 1

        population = self.genomes
        new: list[Genome] = []
        for s, n in zip(survivors, counts):
            if n == 0:
                continue
            members = s.members
            children: list[Genome] = []
            # Elitism: the species champion is copied without mutation.
            if len(members) >= cfg.elitism_min_size or s.id == champion.species_id:
                children.append(members[0].copy(self._new_key(), keep_fitness=False))
            k = min(len(members), max(2, math.ceil(cfg.survival_threshold * len(members))))
            pool = members[:k]
            while len(children) < n:
                p1 = pool[int(rng.integers(k))]
                if rng.random() < cfg.p_crossover:
                    if rng.random() < cfg.p_interspecies:
                        p2 = population[int(rng.integers(len(population)))]
                    else:
                        p2 = pool[int(rng.integers(k))]
                    a, b = (p1, p2) if p1.fitness >= p2.fitness else (p2, p1)
                    child = Genome.crossover(self._new_key(), a, b, cfg, rng)
                else:
                    child = p1.copy(self._new_key(), keep_fitness=False)
                child.mutate(cfg, self.tracker, rng)
                children.append(child)
            new.extend(children)
        self.genomes = new
