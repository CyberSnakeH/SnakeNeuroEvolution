"""NEAT genotype: node genes, connection genes and historical markings.

A genome is a pair (𝒩, 𝒞): 𝒩 a set of nodes (input, output, hidden), each
with a bias, 𝒞 a set of connections (src → dst, weight, enabled,
innovation). The innovation number is a *historical marking*: two genes with
the same number describe the same structure, which allows two arbitrary
topologies to be aligned in O(|𝒞|) during crossover and distance computation.

Implementation choices (deliberate departures from the original paper):
  * innovations are indexed persistently by the pair (src, dst) instead of
    being reset every generation: the same structure always receives the
    same number, which is strictly more informative;
  * the network is constrained to remain acyclic (feed-forward): any
    connection that would close a cycle — including through disabled genes,
    which may later be re-enabled — is rejected. Crossover inherits structure
    only from the fitter parent, and therefore preserves acyclicity.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np

from .config import NEATConfig

INPUT, OUTPUT, HIDDEN = "input", "output", "hidden"


@dataclass(slots=True)
class NodeGene:
    id: int
    kind: str
    bias: float = 0.0


@dataclass(slots=True)
class ConnGene:
    innov: int
    src: int
    dst: int
    weight: float
    enabled: bool = True


class InnovationTracker:
    """Global registry of historical markings."""

    def __init__(self, n_inputs: int, n_outputs: int) -> None:
        self.next_node = n_inputs + n_outputs
        self.next_innov = 0
        self._conn: dict[tuple[int, int], int] = {}
        self._split: dict[tuple[int, int], int] = {}

    def connection(self, src: int, dst: int) -> int:
        key = (src, dst)
        innov = self._conn.get(key)
        if innov is None:
            innov = self._conn[key] = self.next_innov
            self.next_innov += 1
        return innov

    def split(self, innov: int, existing: Iterable[int]) -> int:
        """Id of the node created by splitting connection `innov`.

        Two genomes that split the same connection obtain the same node.
        If the genome already has that node (connection re-enabled then split
        again), move on to the k-th split of that connection.
        """
        existing = set(existing)
        k = 0
        while True:
            key = (innov, k)
            nid = self._split.get(key)
            if nid is None:
                nid = self._split[key] = self.next_node
                self.next_node += 1
            if nid not in existing:
                return nid
            k += 1


class Genome:
    __slots__ = ("key", "n_inputs", "n_outputs", "nodes", "conns", "fitness", "stats", "species_id")

    def __init__(self, key: int, n_inputs: int, n_outputs: int) -> None:
        self.key = key
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.nodes: dict[int, NodeGene] = {}
        self.conns: dict[int, ConnGene] = {}
        self.fitness: float | None = None
        self.stats: dict[str, float] = {}
        self.species_id: int | None = None

    # ---------------------------------------------------------- construction
    @classmethod
    def initial(cls, key: int, n_inputs: int, n_outputs: int, cfg: NEATConfig,
                tracker: InnovationTracker, rng: np.random.Generator) -> "Genome":
        """Minimal topology: inputs connected directly to outputs, no hidden nodes."""
        g = cls(key, n_inputs, n_outputs)
        for i in range(n_inputs):
            g.nodes[i] = NodeGene(i, INPUT, 0.0)
        for o in range(n_inputs, n_inputs + n_outputs):
            g.nodes[o] = NodeGene(o, OUTPUT, float(rng.normal(0.0, cfg.bias_init_sigma)))
        for i in range(n_inputs):
            for o in range(n_inputs, n_inputs + n_outputs):
                if cfg.initial_connection_prob >= 1.0 or rng.random() < cfg.initial_connection_prob:
                    innov = tracker.connection(i, o)
                    g.conns[innov] = ConnGene(innov, i, o, float(rng.normal(0.0, cfg.weight_init_sigma)))
        return g

    def copy(self, key: int | None = None, keep_fitness: bool = True) -> "Genome":
        g = Genome(self.key if key is None else key, self.n_inputs, self.n_outputs)
        g.nodes = {k: NodeGene(n.id, n.kind, n.bias) for k, n in self.nodes.items()}
        g.conns = {k: ConnGene(c.innov, c.src, c.dst, c.weight, c.enabled) for k, c in self.conns.items()}
        if keep_fitness:
            g.fitness = self.fitness
            g.stats = dict(self.stats)
            g.species_id = self.species_id
        return g

    # ------------------------------------------------------------- metrics
    @property
    def n_hidden(self) -> int:
        return sum(1 for n in self.nodes.values() if n.kind == HIDDEN)

    @property
    def n_enabled(self) -> int:
        return sum(1 for c in self.conns.values() if c.enabled)

    def distance(self, other: "Genome", cfg: NEATConfig) -> float:
        """Compatibility distance δ = c1·E/N + c2·D/N + c3·W̄."""
        a, b = self.conns, other.conns
        if not a and not b:
            return 0.0
        ka, kb = a.keys(), b.keys()
        matching = ka & kb
        cut = min(max(ka, default=-1), max(kb, default=-1))
        non_matching = ka ^ kb
        excess = sum(1 for k in non_matching if k > cut)
        disjoint = len(non_matching) - excess
        wbar = (sum(abs(a[k].weight - b[k].weight) for k in matching) / len(matching)) if matching else 0.0
        n = max(len(a), len(b)) if cfg.distance_norm == "max" else 1
        return cfg.c1 * excess / n + cfg.c2 * disjoint / n + cfg.c3 * wbar

    # ------------------------------------------------------------ mutations
    def mutate(self, cfg: NEATConfig, tracker: InnovationTracker, rng: np.random.Generator) -> None:
        if rng.random() < cfg.p_add_node:
            self.mutate_add_node(tracker, rng)
        if rng.random() < cfg.p_add_conn:
            self.mutate_add_conn(cfg, tracker, rng)
        if rng.random() < cfg.p_toggle:
            self.mutate_toggle(rng)
        if rng.random() < cfg.p_weight_mutate:
            self.mutate_weights(cfg, rng)
        if rng.random() < cfg.p_bias_mutate:
            self.mutate_biases(cfg, rng)

    def _perturb(self, values: np.ndarray, cfg: NEATConfig, rng: np.random.Generator) -> np.ndarray:
        n = len(values)
        touched = rng.random(n) < cfg.weight_gene_rate
        replace = touched & (rng.random(n) < cfg.p_weight_replace)
        perturb = touched & ~replace
        out = values.copy()
        out[perturb] += rng.normal(0.0, cfg.weight_sigma, int(perturb.sum()))
        out[replace] = rng.normal(0.0, cfg.weight_init_sigma, int(replace.sum()))
        return np.clip(out, -cfg.weight_clip, cfg.weight_clip)

    def mutate_weights(self, cfg: NEATConfig, rng: np.random.Generator) -> None:
        conns = list(self.conns.values())
        if not conns:
            return
        new = self._perturb(np.fromiter((c.weight for c in conns), float, len(conns)), cfg, rng)
        for c, w in zip(conns, new):
            c.weight = float(w)

    def mutate_biases(self, cfg: NEATConfig, rng: np.random.Generator) -> None:
        nodes = [n for n in self.nodes.values() if n.kind != INPUT]
        new = self._perturb(np.fromiter((n.bias for n in nodes), float, len(nodes)), cfg, rng)
        for n, b in zip(nodes, new):
            n.bias = float(b)

    def mutate_add_node(self, tracker: InnovationTracker, rng: np.random.Generator) -> None:
        """Split a connection a→b into a→h (weight 1) and h→b (old weight)."""
        enabled = [c for c in self.conns.values() if c.enabled]
        if not enabled:
            return
        c = enabled[int(rng.integers(len(enabled)))]
        c.enabled = False
        nid = tracker.split(c.innov, self.nodes.keys())
        self.nodes[nid] = NodeGene(nid, HIDDEN, 0.0)
        i1 = tracker.connection(c.src, nid)
        i2 = tracker.connection(nid, c.dst)
        self.conns[i1] = ConnGene(i1, c.src, nid, 1.0)
        self.conns[i2] = ConnGene(i2, nid, c.dst, c.weight)

    def mutate_add_conn(self, cfg: NEATConfig, tracker: InnovationTracker, rng: np.random.Generator) -> None:
        srcs = [n.id for n in self.nodes.values() if n.kind != OUTPUT]
        dsts = [n.id for n in self.nodes.values() if n.kind != INPUT]
        existing = {(c.src, c.dst): c for c in self.conns.values()}
        for _ in range(cfg.add_conn_attempts):
            s = srcs[int(rng.integers(len(srcs)))]
            d = dsts[int(rng.integers(len(dsts)))]
            if s == d:
                continue
            c = existing.get((s, d))
            if c is not None:
                if not c.enabled:
                    c.enabled = True
                    return
                continue
            if self.creates_cycle(s, d):
                continue
            innov = tracker.connection(s, d)
            self.conns[innov] = ConnGene(innov, s, d, float(rng.normal(0.0, cfg.weight_init_sigma)))
            return

    def mutate_toggle(self, rng: np.random.Generator) -> None:
        if self.conns:
            conns = list(self.conns.values())
            c = conns[int(rng.integers(len(conns)))]
            c.enabled = not c.enabled

    def creates_cycle(self, src: int, dst: int) -> bool:
        """True if adding src→dst would close a cycle (src reachable from dst)."""
        if src == dst:
            return True
        adj: dict[int, list[int]] = defaultdict(list)
        for c in self.conns.values():
            adj[c.src].append(c.dst)
        stack, seen = [dst], {dst}
        while stack:
            n = stack.pop()
            if n == src:
                return True
            for m in adj[n]:
                if m not in seen:
                    seen.add(m)
                    stack.append(m)
        return False

    # ------------------------------------------------------------- crossover
    @staticmethod
    def crossover(key: int, fitter: "Genome", other: "Genome", cfg: NEATConfig,
                  rng: np.random.Generator) -> "Genome":
        """Matching genes drawn at random; disjoint/excess genes from the fitter parent."""
        child = Genome(key, fitter.n_inputs, fitter.n_outputs)
        for nid, n in fitter.nodes.items():
            m = other.nodes.get(nid)
            bias = n.bias if (m is None or rng.random() < 0.5) else m.bias
            child.nodes[nid] = NodeGene(nid, n.kind, bias)
        for innov, c in fitter.conns.items():
            d = other.conns.get(innov)
            if d is None:
                child.conns[innov] = ConnGene(innov, c.src, c.dst, c.weight, c.enabled)
                continue
            w = c.weight if rng.random() < 0.5 else d.weight
            enabled = True
            if not (c.enabled and d.enabled):
                enabled = rng.random() >= cfg.p_inherit_disabled
            child.conns[innov] = ConnGene(innov, c.src, c.dst, w, enabled)
        return child

    # --------------------------------------------------------- serialization
    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "n_inputs": self.n_inputs,
            "n_outputs": self.n_outputs,
            "fitness": self.fitness,
            "stats": self.stats,
            "nodes": [[n.id, n.kind, n.bias] for n in self.nodes.values()],
            "conns": [[c.innov, c.src, c.dst, c.weight, c.enabled] for c in self.conns.values()],
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Genome":
        g = cls(d["key"], d["n_inputs"], d["n_outputs"])
        for nid, kind, bias in d["nodes"]:
            g.nodes[nid] = NodeGene(nid, kind, bias)
        for innov, s, t, w, en in d["conns"]:
            g.conns[innov] = ConnGene(innov, s, t, w, en)
        g.fitness = d.get("fitness")
        g.stats = d.get("stats", {})
        return g
