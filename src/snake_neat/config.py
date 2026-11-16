"""Hyperparameters of the environment and of the NEAT algorithm.

Defaults follow Stanley & Miikkulainen (2002), "Evolving Neural Networks
through Augmenting Topologies", Evolutionary Computation 10(2), tuned
empirically for Snake (see README, "Protocol" section).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from typing import Any

N_RAYS = 8          # vision directions
N_CHANNELS = 3      # (wall, body, food) per direction
N_OUTPUTS = 3       # relative actions: left, straight, right


@dataclass(frozen=True)
class GameConfig:
    width: int = 15
    height: int = 15
    initial_length: int = 3
    hunger: int = 0               # steps without eating before death; 0 → width·height
    food_compass: bool = False    # 2 extra inputs: egocentric vector to the food

    @property
    def hunger_limit(self) -> int:
        return self.hunger if self.hunger > 0 else self.width * self.height

    @property
    def n_inputs(self) -> int:
        return N_RAYS * N_CHANNELS + (2 if self.food_compass else 0)

    @property
    def n_outputs(self) -> int:
        return N_OUTPUTS

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "GameConfig":
        names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in names})


@dataclass
class NEATConfig:
    population_size: int = 150

    # --- Speciation: δ = c1·E/N + c2·D/N + c3·W̄ --------------------------
    c1: float = 1.0
    c2: float = 1.0
    c3: float = 0.4
    distance_norm: str = "max"        # "max": N = max(|C1|,|C2|); "none": N = 1
    compat_threshold: float = 1.0     # initial δ_t, adapted dynamically
    target_species: int = 10
    threshold_rate: float = 0.05      # δ_t ← δ_t·(1 ± rate): multiplicative feedback control
    min_threshold: float = 0.05

    # --- Initialization -------------------------------------------------
    initial_connection_prob: float = 1.0   # 1 → minimal fully connected inputs→outputs topology
    weight_init_sigma: float = 1.0
    bias_init_sigma: float = 0.5

    # --- Mutations ------------------------------------------------------
    p_weight_mutate: float = 0.8      # probability that a genome undergoes weight mutation
    weight_gene_rate: float = 0.5     # per-gene probability, given weight mutation
    p_weight_replace: float = 0.1     # otherwise Gaussian perturbation
    weight_sigma: float = 0.4
    weight_clip: float = 8.0
    p_bias_mutate: float = 0.5
    p_add_conn: float = 0.10
    p_add_node: float = 0.03
    p_toggle: float = 0.01
    add_conn_attempts: int = 20

    # --- Reproduction ---------------------------------------------------
    p_crossover: float = 0.75
    p_interspecies: float = 0.01
    p_inherit_disabled: float = 0.75
    survival_threshold: float = 0.25   # fraction of a species' best allowed to reproduce
    elitism_min_size: int = 5          # champion copied unchanged if the species has ≥ 5 members
    stagnation_limit: int = 20
    species_elitism: int = 2           # number of species protected from stagnation extinction

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "NEATConfig":
        names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in names})
