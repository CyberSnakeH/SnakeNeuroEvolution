# API and usage reference

This page documents the command-line interface, the Python API, the hyperparameters and the genome file
format. The mathematical definitions referred to below (Definition 2.3, Proposition 4.4, …) are in
[theory.md](theory.md).

**Contents**

1. [Command-line interface](#1-command-line-interface)
2. [Programmatic use](#2-programmatic-use)
3. [Modules](#3-modules)
4. [Hyperparameters](#4-hyperparameters)
5. [Genome file format](#5-genome-file-format)
6. [Training logs](#6-training-logs)
7. [Reproducibility](#7-reproducibility)

---

## 1. Command-line interface

```text
snake-neat [gui] [options]          GUI with live evolution (default command)
snake-neat play GENOME.json         GUI replaying a saved genome
snake-neat train [options]          headless training
snake-neat eval GENOME.json         held-out evaluation with a 95 % confidence interval
```

Options shared by `gui`, `play` and `train`:

| Option | Default | Meaning |
|---|---|---|
| `--width`, `--height` | 15, 15 | board size $`W, H`$ |
| `--hunger` | 0 | hunger limit $`\eta_{\max}`$; 0 means $`WH`$ |
| `--compass` | off | add the two compass inputs $`\phi_{24}, \phi_{25}`$ |
| `--pop` | 150 | population size $`N_{\mathrm{pop}}`$ |
| `--species` | 10 | target number of species $`K^\star`$ |
| `--episodes` | 3 | episodes per evaluation $`K`$ |
| `--workers` | 0 | evaluation processes; 0 means all cores |
| `--seed` | 0 | master seed of the run |

Specific options:

| Command | Option | Default | Meaning |
|---|---|---|---|
| `gui` | `--view {champion,species}` | `champion` | initial view |
| `gui` | `--genome PATH` | | genome to replay at start-up |
| `train` | `--generations` | 200 | number of generations |
| `train` | `--out` | `runs/default` | output directory |
| `train` | `--checkpoint` | 25 | save the champion every this many generations |
| `train` | `--final-eval` | 200 | held-out episodes at the end; 0 disables it |
| `eval` | `--episodes` | 500 | held-out episodes |

GUI shortcuts: `Space` pauses or resumes evolution, `v` switches between the champion and species views,
`+` / `-` change the replay speed. In the species view, clicking a board shows that species' network.

---

## 2. Programmatic use

Train for 50 generations, then evaluate the champion on held-out seeds:

```python
from snake_neat.config import GameConfig, NEATConfig
from snake_neat.evaluation import Evaluator, benchmark
from snake_neat.population import Population


def main() -> None:
    game = GameConfig(width=15, height=15)
    neat = NEATConfig(population_size=150)
    pop = Population(neat, game.n_inputs, game.n_outputs, seed=0)

    with Evaluator(game, episodes=3, seed=0) as evaluate:      # one process per core
        for _ in range(50):
            stats = pop.run_generation(lambda genomes: evaluate(genomes, pop.generation))
            print(stats.generation, stats.best_apples, stats.n_species)

    print(benchmark(stats.champion, game, n_episodes=200))


if __name__ == "__main__":
    main()
```

> **The `if __name__ == "__main__":` guard is required.** Worker processes are started with the
> *spawn* method, which re-imports the main module in every worker. Without the guard each worker would
> start a training run of its own. For the same reason, the parallel `Evaluator` cannot be used from
> code piped on standard input; pass `workers=1` in that case.

Play one game with a saved genome:

```python
from snake_neat.evaluation import play_episode
from snake_neat.network import FeedForwardNetwork
from snake_neat.persistence import load_genome

genome, game, meta = load_genome("runs/demo/champion.json")
net = FeedForwardNetwork.from_genome(genome)
print(play_episode(net, game, seed=123))       # EpisodeResult(apples=..., steps=..., death=...)
```

Any callable that assigns `genome.fitness` to every genome can replace `Evaluator`. This makes the NEAT
core reusable for other tasks with 24 inputs and 3 outputs, or any other sizes passed to `Population`.

---

## 3. Modules

### `snake_neat.config`

| Name | Description |
|---|---|
| `GameConfig(width, height, initial_length, hunger, food_compass)` | immutable environment parameters; properties `hunger_limit`, `n_inputs`, `n_outputs` |
| `NEATConfig(...)` | algorithm hyperparameters, see [Section 4](#4-hyperparameters) |
| `to_dict()`, `from_dict(d)` | serialisation helpers on both classes |

### `snake_neat.game`

| Name | Description |
|---|---|
| `SnakeGame(cfg, seed)` | one game, deterministic given the seed (Definitions 2.1 to 2.3) |
| `SnakeGame.step(action) -> bool` | apply a relative action, 0 left, 1 straight, 2 right; returns whether the snake is alive |
| `SnakeGame.observe() -> ndarray` | observation $`\phi(s)`$ (Definition 3.2) |
| `SnakeGame.cast_rays()` | for each ray, `(dx, dy, t_wall, t_body, t_food)` with 0 meaning not seen |
| `SnakeGame.alive`, `apples`, `steps`, `hunger`, `death` | episode state; `death` is `"wall"`, `"body"`, `"starvation"` or `"win"` |
| `ray_directions(heading)` | the eight directions $`d_0, \dots, d_7`$ |
| `input_labels(cfg)` | input names such as `"FR body"` |
| `DIRS`, `RAY_NAMES`, `CHANNEL_NAMES`, `ACTION_NAMES` | constants |

### `snake_neat.genome`

| Name | Description |
|---|---|
| `NodeGene(id, kind, bias)` | `kind` is `"input"`, `"output"` or `"hidden"` |
| `ConnGene(innov, src, dst, weight, enabled)` | connection gene |
| `InnovationTracker(n_inputs, n_outputs)` | global registry $`\iota`$; `connection(src, dst)`, `split(innov, existing)` |
| `Genome.initial(key, n_in, n_out, cfg, tracker, rng)` | minimal fully connected genome |
| `Genome.mutate(cfg, tracker, rng)` | applies the mutation operators of Section 5 |
| `Genome.crossover(key, fitter, other, cfg, rng)` | crossover, structure of the fitter parent |
| `Genome.distance(other, cfg)` | compatibility distance $`\delta`$ (Definition 6.1) |
| `Genome.creates_cycle(src, dst)` | reachability test of Proposition 5.1 |
| `Genome.copy(key, keep_fitness)`, `to_dict()`, `from_dict(d)` | copying and serialisation |
| `Genome.n_hidden`, `Genome.n_enabled` | complexity measures |

### `snake_neat.network`

| Name | Description |
|---|---|
| `FeedForwardNetwork.from_genome(g)` | compiles the layers $`L_k, W_k, b_k`$; raises `ValueError` on a cycle |
| `activate(x) -> ndarray` | forward pass of Proposition 4.4; returns the 3 outputs |
| `decide(x) -> int` | $`\arg\max`$ of the outputs |
| `values`, `index`, `depth` | all activations, node index and depth, used by the GUI |

### `snake_neat.population`

| Name | Description |
|---|---|
| `Population(cfg, n_inputs, n_outputs, seed)` | initial population, already speciated |
| `run_generation(evaluate) -> GenerationStats` | evaluate, record statistics, reproduce, speciate |
| `species`, `threshold`, `generation`, `best_ever` | current state |
| `GenerationStats` | per-generation statistics; `row()` gives the CSV row |

### `snake_neat.evaluation`

| Name | Description |
|---|---|
| `fitness(apples, steps)` | $`F = 1000 A + T`$ (Definition 2.5) |
| `play_episode(net, cfg, seed) -> EpisodeResult` | one full game |
| `Evaluator(cfg, episodes, workers, seed)` | estimator $`\hat F_n`$ with common random numbers (Definition 7.1); context manager |
| `Evaluator.episode_seeds(generation)` | the $`K`$ seeds of a generation |
| `benchmark(genome, cfg, n_episodes, first_seed)` | held-out evaluation with seeds from $`10^6`$, mean, CI, median, max, deaths |

### `snake_neat.persistence`

| Name | Description |
|---|---|
| `save_genome(path, genome, game_cfg, meta)` | writes the JSON file of [Section 5](#5-genome-file-format) |
| `load_genome(path) -> (genome, game_cfg, meta)` | reads it back and checks the format tag |

### `snake_neat.gui`

Tkinter interface: `run_gui(game_cfg, neat_cfg, seed, episodes, workers, genome_path, view)`. Evolution
runs in a background thread that sends `GenerationStats` to the Tk loop through a queue. Each replayed
snake is a `Runner`, which holds a genome, its compiled network, a game and the current decision.

---

## 4. Hyperparameters

All fields of `NEATConfig`, with the symbols of [theory.md](theory.md).

| Field | Symbol | Default | Role |
|---|---|---|---|
| `population_size` | $`N_{\mathrm{pop}}`$ | 150 | number of genomes |
| `c1`, `c2`, `c3` | $`c_1, c_2, c_3`$ | 1.0, 1.0, 0.4 | weights of the compatibility distance |
| `distance_norm` | $`N`$ | `"max"` | `"max"`: $`N = \max(\lvert \mathcal{C}_1 \rvert, \lvert \mathcal{C}_2 \rvert)`$; `"none"`: $`N = 1`$ |
| `compat_threshold` | $`\delta_t^{(0)}`$ | 1.0 | initial speciation threshold |
| `target_species` | $`K^\star`$ | 10 | target number of species |
| `threshold_rate` | $`\alpha`$ | 0.05 | multiplicative step of the threshold controller |
| `min_threshold` | | 0.05 | lower bound of $`\delta_t`$ |
| `initial_connection_prob` | | 1.0 | density of the initial input→output wiring |
| `weight_init_sigma` | | 1.0 | std. of new and reset weights |
| `bias_init_sigma` | | 0.5 | std. of initial output biases |
| `p_weight_mutate` | | 0.8 | probability that a genome mutates its weights |
| `weight_gene_rate` | | 0.5 | probability per gene, given a weight mutation |
| `p_weight_replace` | | 0.1 | probability of a reset instead of a perturbation |
| `weight_sigma` | | 0.4 | std. of the Gaussian perturbation |
| `weight_clip` | | 8.0 | weights and biases are clipped to $`[-8, 8]`$ |
| `p_bias_mutate` | | 0.5 | probability that a genome mutates its biases |
| `p_add_conn` | | 0.10 | add-connection probability |
| `p_add_node` | | 0.03 | add-node probability |
| `p_toggle` | | 0.01 | enable/disable toggle probability |
| `add_conn_attempts` | | 20 | random endpoint pairs tried per add-connection |
| `p_crossover` | | 0.75 | probability that an offspring comes from crossover |
| `p_interspecies` | | 0.01 | probability that the second parent comes from any species |
| `p_inherit_disabled` | | 0.75 | probability that a gene disabled in a parent stays disabled |
| `survival_threshold` | | 0.25 | fraction of a species allowed to reproduce |
| `elitism_min_size` | | 5 | minimum species size for copying its champion |
| `stagnation_limit` | | 20 | generations without improvement before extinction |
| `species_elitism` | | 2 | number of best species protected from extinction |

---

## 5. Genome file format

Genomes are stored as UTF-8 JSON with the format tag `snake-neat/1`:

```json
{
  "format": "snake-neat/1",
  "game":   {"width": 15, "height": 15, "initial_length": 3, "hunger": 0, "food_compass": false},
  "genome": {
    "key": 41237, "n_inputs": 24, "n_outputs": 3,
    "fitness": 66827.0,
    "stats": {"fitness": 66827.0, "apples": 65.0, "steps": 1827.0, "max_apples": 71},
    "nodes": [[0, "input", 0.0], [24, "output", -0.31], [513, "hidden", 0.12]],
    "conns": [[0, 0, 24, 1.57, true], [97, 0, 513, 1.0, false]]
  },
  "meta": {"generation": 299, "fitness": 66827.0}
}
```

- `nodes` rows are `[id, kind, bias]`.
- `conns` rows are `[innovation, src, dst, weight, enabled]`.
- `game` stores the environment the genome was trained in, so that `play` and `eval` rebuild exactly the
  same observation map.
- The numeric values above are illustrative.

---

## 6. Training logs

`snake-neat train --out DIR` writes:

| File | Content |
|---|---|
| `config.json` | game and NEAT configuration, seed, episodes |
| `history.csv` | one row per generation (columns below) |
| `champion.json` | champion of the latest generation |
| `champion_gNNNN.json` | champion every `--checkpoint` generations |
| `final_eval.json` | held-out evaluation of the final champion |

`history.csv` columns: `generation`, `best_fitness`, `mean_fitness`, `best_apples` (champion, mean over
the $`K`$ training episodes), `mean_apples`, `max_apples` (best single episode), `n_species`, `threshold`
($`\delta_t`$), `champion_hidden`, `champion_conns`, `mean_hidden`, `elapsed_s`.

---

## 7. Reproducibility

- The population's random generator is seeded with `--seed`.
- The episode seeds of generation $`n`$ are derived from `SeedSequence([seed, n])`, so they do not depend
  on the number of worker processes or on scheduling.
- Held-out evaluation uses consecutive seeds starting at $`10^6`$, never used in training.
- The GUI replays with fresh random seeds, so a replayed game differs from the training games.

Given the same seed, numpy version and board settings, `train` therefore produces the same sequence of
generations, independently of `--workers`.
