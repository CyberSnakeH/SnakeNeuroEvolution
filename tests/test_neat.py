"""Unit tests (unittest, standard library).

    uv run python -m unittest discover -s tests -v
"""

from __future__ import annotations

import math
import unittest
from dataclasses import replace

import numpy as np

from snake_neat.config import GameConfig, NEATConfig
from snake_neat.evaluation import Evaluator, fitness, play_episode
from snake_neat.game import DIRS, SnakeGame
from snake_neat.genome import HIDDEN, INPUT, Genome, InnovationTracker
from snake_neat.network import FeedForwardNetwork
from snake_neat.population import Population

CFG = NEATConfig()


def random_genome(seed: int, n_mut: int = 300, n_in: int = 6, n_out: int = 3) -> Genome:
    rng = np.random.default_rng(seed)
    tr = InnovationTracker(n_in, n_out)
    g = Genome.initial(0, n_in, n_out, CFG, tr, rng)
    cfg = replace(CFG, p_add_node=0.3, p_add_conn=0.6, p_toggle=0.2)
    for _ in range(n_mut):
        g.mutate(cfg, tr, rng)
    return g


def naive_forward(g: Genome, x: np.ndarray) -> np.ndarray:
    """Reference recursive evaluation, directly from the definition."""
    memo: dict[int, float] = {}

    def val(n: int) -> float:
        if n in memo:
            return memo[n]
        node = g.nodes[n]
        if node.kind == INPUT:
            v = float(x[n])
        else:
            s = node.bias + sum(c.weight * val(c.src) for c in g.conns.values() if c.enabled and c.dst == n)
            v = math.tanh(s)
        memo[n] = v
        return v

    return np.array([val(o) for o in range(g.n_inputs, g.n_inputs + g.n_outputs)])


class TestNetwork(unittest.TestCase):
    def test_layered_forward_equals_recursive_definition(self):
        for seed in range(20):
            g = random_genome(seed)
            net = FeedForwardNetwork.from_genome(g)
            x = np.random.default_rng(seed).normal(size=g.n_inputs)
            np.testing.assert_allclose(net.activate(x), naive_forward(g, x), atol=1e-12)

    def test_mutations_preserve_acyclicity(self):
        for seed in range(20):
            g = random_genome(seed, n_mut=500)
            self.assertGreater(g.n_hidden, 0)
            FeedForwardNetwork.from_genome(g)  # raises ValueError on a cycle

    def test_cycle_detection(self):
        g = random_genome(1)
        c = next(c for c in g.conns.values() if g.nodes[c.src].kind == HIDDEN)
        self.assertTrue(g.creates_cycle(c.dst, c.src))


class TestGenome(unittest.TestCase):
    def test_distance_is_a_semimetric(self):
        a, b = random_genome(3), random_genome(4)
        self.assertEqual(a.distance(a, CFG), 0.0)
        self.assertAlmostEqual(a.distance(b, CFG), b.distance(a, CFG))
        self.assertGreater(a.distance(b, CFG), 0.0)

    def test_crossover_inherits_structure_of_fitter_parent(self):
        rng = np.random.default_rng(0)
        a, b = random_genome(5), random_genome(6)
        child = Genome.crossover(1, a, b, CFG, rng)
        self.assertEqual(set(child.conns), set(a.conns))
        self.assertEqual(set(child.nodes), set(a.nodes))
        for k, c in child.conns.items():
            allowed = {a.conns[k].weight} | ({b.conns[k].weight} if k in b.conns else set())
            self.assertIn(c.weight, allowed)
        FeedForwardNetwork.from_genome(child)

    def test_historical_markings_are_shared(self):
        tr = InnovationTracker(2, 1)
        self.assertEqual(tr.connection(0, 2), tr.connection(0, 2))
        self.assertNotEqual(tr.connection(0, 2), tr.connection(1, 2))
        n1 = tr.split(0, existing={0, 1, 2})
        self.assertEqual(n1, tr.split(0, existing={0, 1, 2}))
        self.assertNotEqual(n1, tr.split(0, existing={0, 1, 2, n1}))

    def test_serialization_roundtrip(self):
        g = random_genome(7)
        h = Genome.from_dict(g.to_dict())
        x = np.linspace(-1, 1, g.n_inputs)
        np.testing.assert_array_equal(FeedForwardNetwork.from_genome(g).activate(x),
                                      FeedForwardNetwork.from_genome(h).activate(x))


class TestGame(unittest.TestCase):
    def test_wall_death_and_observation(self):
        game = SnakeGame(GameConfig(width=7, height=7), seed=0)
        obs = game.observe()
        self.assertEqual(obs.shape, (24,))
        self.assertTrue(np.all((obs >= 0) & (obs <= 1)))
        self.assertAlmostEqual(obs[12 + 1], 1.0)  # backward ray: the neck is adjacent
        while game.alive:
            game.step(1)
        self.assertEqual(game.death, "wall")
        self.assertEqual(game.steps, 4)  # 7//2 = 3 cells before the edge, the 4th exits

    def test_eating_grows_and_resets_hunger(self):
        game = SnakeGame(GameConfig(width=9, height=9), seed=1)
        hx, hy = game.head
        dx, dy = DIRS[game.heading]
        game.food = (hx + dx, hy + dy)
        game.hunger = 5
        n = game.length
        game.step(1)
        self.assertEqual((game.apples, game.length, game.hunger), (1, n + 1, 0))
        self.assertNotIn(game.food, game.occupied)

    def test_moving_into_vacating_tail_is_legal(self):
        game = SnakeGame(GameConfig(width=9, height=9, initial_length=4), seed=2)
        game.food = (0, 0) if (0, 0) not in game.occupied else (8, 8)
        for a in (2, 2, 2):  # three right turns: the head returns onto the former tail
            game.step(a)
        self.assertTrue(game.alive)

    def test_starvation(self):
        cfg = GameConfig(width=9, height=9, hunger=12)
        game = SnakeGame(cfg, seed=3)
        game.food = None
        while game.alive:
            game.step(2 if game.steps % 2 else 1)
        self.assertEqual(game.death, "starvation")
        self.assertEqual(game.steps, 12)

    def test_rotation_invariance_of_observation(self):
        """On a square board, φ is invariant when the whole state is rotated by 90°."""
        cfg = GameConfig(width=11, height=11)
        g1 = SnakeGame(cfg, seed=4)
        for a in (1, 1, 2, 1, 0, 1):
            g1.step(a)
        rot = lambda p: (cfg.height - 1 - p[1], p[0])  # 90° clockwise rotation  # noqa: E731
        g2 = SnakeGame(cfg, seed=4)
        g2.body = type(g1.body)(rot(p) for p in g1.body)
        g2.occupied = set(g2.body)
        g2.heading = (g1.heading + 1) % 4
        g2.food = rot(g1.food)
        np.testing.assert_allclose(g1.observe(), g2.observe())


class TestEvolution(unittest.TestCase):
    def test_population_size_is_invariant(self):
        cfg = NEATConfig(population_size=40)
        pop = Population(cfg, 4, 2, seed=0)
        rng = np.random.default_rng(0)
        for _ in range(8):
            def evaluate(gs):
                for g in gs:
                    g.fitness = float(rng.random()) + g.n_hidden
            st = pop.run_generation(evaluate)
            self.assertEqual(len(pop.genomes), 40)
            self.assertEqual(sum(st.species_sizes.values()), 40)

    def test_evolution_learns_to_eat(self):
        """End-to-end test: 15 generations suffice to beat chance."""
        game = GameConfig(width=10, height=10)
        pop = Population(NEATConfig(population_size=60), game.n_inputs, game.n_outputs, seed=0)
        with Evaluator(game, episodes=2, workers=1, seed=0) as ev:
            for _ in range(15):
                st = pop.run_generation(lambda gs: ev(gs, pop.generation))
        net = FeedForwardNetwork.from_genome(st.champion)
        apples = np.mean([play_episode(net, game, 10**6 + k).apples for k in range(20)])
        self.assertGreater(apples, 3.0)

    def test_fitness_formula(self):
        self.assertEqual(fitness(3, 120), 3120.0)


if __name__ == "__main__":
    unittest.main()
