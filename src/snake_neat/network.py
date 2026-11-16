"""Phenotype: compilation of an acyclic genome into a layer-wise evaluable network.

Let G = (V, E) be the (acyclic) graph of enabled connections. Define the depth
d(v) = 0 for an input, and d(v) = 1 + max_{(u,v)∈E} d(u) otherwise
(d(v) = 1 if v has no predecessor). Nodes of equal depth depend only on
strictly shallower nodes; the forward pass therefore reduces to a sequence of
dense matrix-vector products:

    x[L_k] ← tanh(W_k · x + b_k),   k = 1, …, d_max

where W_k ∈ R^{|L_k|×|V|} holds the weights of the edges entering layer L_k.
The cost is O(d_max·|V|²) flops but only d_max numpy calls, which is much
faster than node-by-node evaluation in pure Python for the sizes encountered
here (|V| ≲ 50).

tanh(x) = 2σ(2x) − 1: same family as the steepened sigmoid σ(4.9x) of the paper.
"""

from __future__ import annotations

from collections import defaultdict, deque

import numpy as np

from .genome import HIDDEN, INPUT, OUTPUT, Genome


class FeedForwardNetwork:
    __slots__ = ("n_inputs", "n_outputs", "node_ids", "index", "depth", "layers", "values")

    def __init__(self, n_inputs: int, n_outputs: int, node_ids: list[int], depth: dict[int, int],
                 layers: list[tuple[np.ndarray, np.ndarray, np.ndarray]]) -> None:
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.node_ids = node_ids
        self.index = {nid: i for i, nid in enumerate(node_ids)}
        self.depth = depth
        self.layers = layers
        self.values = np.zeros(len(node_ids))

    @classmethod
    def from_genome(cls, g: Genome) -> "FeedForwardNetwork":
        inputs = sorted(n.id for n in g.nodes.values() if n.kind == INPUT)
        outputs = sorted(n.id for n in g.nodes.values() if n.kind == OUTPUT)
        hidden = sorted(n.id for n in g.nodes.values() if n.kind == HIDDEN)
        order = inputs + outputs + hidden
        index = {nid: i for i, nid in enumerate(order)}

        preds: dict[int, list[tuple[int, float]]] = defaultdict(list)
        succs: dict[int, list[int]] = defaultdict(list)
        for c in g.conns.values():
            if c.enabled:
                preds[c.dst].append((c.src, c.weight))
                succs[c.src].append(c.dst)

        # Kahn's topological sort + longest-path depth.
        indeg = {n: len(preds[n]) for n in order}
        depth = {n: (0 if g.nodes[n].kind == INPUT else 1) for n in order}
        queue = deque(n for n in order if indeg[n] == 0)
        seen = 0
        while queue:
            n = queue.popleft()
            seen += 1
            for m in succs[n]:
                depth[m] = max(depth[m], depth[n] + 1)
                indeg[m] -= 1
                if indeg[m] == 0:
                    queue.append(m)
        if seen != len(order):
            raise ValueError("genome contains a cycle: network is not feed-forward")

        n_nodes = len(order)
        by_depth: dict[int, list[int]] = defaultdict(list)
        for n in order:
            if depth[n] > 0:
                by_depth[depth[n]].append(n)
        layers = []
        for d in sorted(by_depth):
            members = by_depth[d]
            W = np.zeros((len(members), n_nodes))
            b = np.empty(len(members))
            for r, n in enumerate(members):
                b[r] = g.nodes[n].bias
                for src, w in preds[n]:
                    W[r, index[src]] += w
            layers.append((np.array([index[n] for n in members], dtype=np.intp), W, b))
        return cls(len(inputs), len(outputs), order, depth, layers)

    def activate(self, x: np.ndarray) -> np.ndarray:
        v = self.values
        v[: self.n_inputs] = x
        for idx, W, b in self.layers:
            v[idx] = np.tanh(W @ v + b)
        return v[self.n_inputs : self.n_inputs + self.n_outputs]

    def decide(self, x: np.ndarray) -> int:
        return int(np.argmax(self.activate(x)))
