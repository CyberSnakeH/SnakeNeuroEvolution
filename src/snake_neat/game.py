"""Snake environment and observation function.

Formalization
-------------
Grid G = {0..W-1}×{0..H-1}, y axis pointing down. The state is
s = (body, heading, food, hunger) where body = (c_0, …, c_{L-1}) ∈ G^L, c_0 = head.

*Relative* actions a ∈ {0: left, 1: straight, 2: right}: the heading
evolves as h ← h + a − 1 (mod 4). This removes the "U-turn" action
(immediate suicide) and makes the action space rotation-equivariant.

Observation φ(s) ∈ R^24: 8 *egocentric* rays (forward, forward-right, right,
…, forward-left); for each ray, with step t = 1, 2, …:
    wall : 1/t_wall                 (t_wall = first step outside the grid)
    body : 1/t_body, 0 if no segment lies on the ray
    food : 1 − (t_food − 1)/max(W,H), 0 if the food is not on the ray
Expressing the rays in the snake's frame makes φ invariant under the action
of the cyclic group C4 of board rotations (up to boundary effects, if W = H):
the policy need not relearn the same strategy for each heading.
"""

from __future__ import annotations

from collections import deque

import numpy as np

from .config import GameConfig

# Absolute headings, ordered clockwise (y pointing down): N, E, S, W.
DIRS: tuple[tuple[int, int], ...] = ((0, -1), (1, 0), (0, 1), (-1, 0))
RAY_NAMES = ("F", "FR", "R", "BR", "B", "BL", "L", "FL")
CHANNEL_NAMES = ("wall", "body", "food")
ACTION_NAMES = ("Left", "Straight", "Right")


def input_labels(cfg: GameConfig) -> list[str]:
    labels = [f"{r} {c}" for r in RAY_NAMES for c in CHANNEL_NAMES]
    if cfg.food_compass:
        labels += ["food→fwd", "food→right"]
    return labels


def ray_directions(heading: int) -> list[tuple[int, int]]:
    """The 8 vision directions, clockwise starting from forward."""
    fx, fy = DIRS[heading]
    rx, ry = -fy, fx                      # "right" vector = clockwise rotation of forward
    return [
        (fx, fy), (fx + rx, fy + ry), (rx, ry), (rx - fx, ry - fy),
        (-fx, -fy), (-fx - rx, -fy - ry), (-rx, -ry), (fx - rx, fy - ry),
    ]


class SnakeGame:
    """A game of Snake, deterministic given the seed."""

    __slots__ = ("cfg", "rng", "body", "occupied", "heading", "food",
                 "alive", "steps", "apples", "hunger", "death")

    def __init__(self, cfg: GameConfig, seed: int | None = None) -> None:
        self.cfg = cfg
        self.rng = np.random.default_rng(seed)
        self.reset()

    # ----------------------------------------------------------------- state
    def reset(self) -> None:
        cfg = self.cfg
        self.heading = int(self.rng.integers(4))
        dx, dy = DIRS[self.heading]
        hx, hy = cfg.width // 2, cfg.height // 2
        self.body: deque[tuple[int, int]] = deque(
            (hx - i * dx, hy - i * dy) for i in range(cfg.initial_length)
        )
        self.occupied: set[tuple[int, int]] = set(self.body)
        self.alive = True
        self.steps = 0
        self.apples = 0
        self.hunger = 0
        self.death: str | None = None
        self.food: tuple[int, int] | None = None
        self._place_food()

    @property
    def head(self) -> tuple[int, int]:
        return self.body[0]

    @property
    def length(self) -> int:
        return len(self.body)

    def _place_food(self) -> None:
        W, H = self.cfg.width, self.cfg.height
        if len(self.occupied) >= W * H:
            self.food = None
            return
        # Rejection sampling (fast while the board is sparsely filled) …
        for _ in range(32):
            c = (int(self.rng.integers(W)), int(self.rng.integers(H)))
            if c not in self.occupied:
                self.food = c
                return
        # … then exact uniform draw among the free cells.
        free = [(x, y) for y in range(H) for x in range(W) if (x, y) not in self.occupied]
        self.food = free[int(self.rng.integers(len(free)))]

    def _die(self, reason: str) -> bool:
        self.alive = False
        self.death = reason
        return False

    # ------------------------------------------------------------ dynamics
    def step(self, action: int) -> bool:
        """Apply the relative action (0 left, 1 straight, 2 right)."""
        if not self.alive:
            return False
        self.heading = (self.heading + action - 1) % 4
        dx, dy = DIRS[self.heading]
        hx, hy = self.body[0]
        new = (hx + dx, hy + dy)
        self.steps += 1
        self.hunger += 1

        if not (0 <= new[0] < self.cfg.width and 0 <= new[1] < self.cfg.height):
            return self._die("wall")
        eat = new == self.food
        tail = self.body[-1]
        # The tail vacates its cell at the same instant, unless the snake grows.
        if new in self.occupied and not (new == tail and not eat):
            return self._die("body")

        if not eat:
            self.body.pop()
            self.occupied.discard(tail)
        self.body.appendleft(new)
        self.occupied.add(new)

        if eat:
            self.apples += 1
            self.hunger = 0
            self._place_food()
            if self.food is None:
                self.alive = False
                self.death = "win"
        elif self.hunger >= self.cfg.hunger_limit:
            return self._die("starvation")
        return self.alive

    # ------------------------------------------------------------ perception
    def cast_rays(self) -> list[tuple[int, int, int, int, int]]:
        """For each ray: (dx, dy, t_wall, t_body, t_food); 0 = not seen."""
        W, H = self.cfg.width, self.cfg.height
        hx, hy = self.body[0]
        occ, food = self.occupied, self.food
        out = []
        for dx, dy in ray_directions(self.heading):
            x, y, t, tb, tf = hx, hy, 0, 0, 0
            while True:
                x += dx
                y += dy
                t += 1
                if x < 0 or y < 0 or x >= W or y >= H:
                    break
                if not tb and (x, y) in occ:
                    tb = t
                if not tf and (x, y) == food:
                    tf = t
            out.append((dx, dy, t, tb, tf))
        return out

    def observe(self) -> np.ndarray:
        cfg = self.cfg
        span = max(cfg.width, cfg.height)
        v = np.zeros(cfg.n_inputs)
        for k, (_, _, tw, tb, tf) in enumerate(self.cast_rays()):
            v[3 * k] = 1.0 / tw
            if tb:
                v[3 * k + 1] = 1.0 / tb
            if tf:
                v[3 * k + 2] = 1.0 - (tf - 1) / span
        if cfg.food_compass and self.food is not None:
            fx, fy = DIRS[self.heading]
            rx, ry = -fy, fx
            ex, ey = self.food[0] - self.body[0][0], self.food[1] - self.body[0][1]
            v[24] = (ex * fx + ey * fy) / span
            v[25] = (ex * rx + ey * ry) / span
        return v
