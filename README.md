# Snake × NEAT

**NeuroEvolution of Augmenting Topologies applied to Snake, implemented from scratch with numpy only, with a live visualization of the evolving population.**

![Champion playing, its network lighting up, fitness and speciation curves](docs/demo.gif)

Every component is written from first principles: the game, the NEAT genome (historical markings, structural mutations, crossover), speciation with explicit fitness sharing, the layered phenotype evaluator, parallel evaluation, and the GUI. The only third-party dependency is **numpy**. Everything else comes from the Python standard library (Tkinter, multiprocessing, unittest). The project is managed with **uv**.

| | |
|---|---|
| Held-out performance (15×15 board) | **60.4 ± 0.8 apples** per game (95 % CI, 500 unseen games), best game 88 apples |
| Time to 20 apples | ≈ 8 generations |
| Generation time | ≈ 0.2 s (150 genomes × 3 games, 24 cores) |
| Champion network | 24 inputs → 3 outputs, ≈ 9 hidden neurons evolved from none |

---

## Contents

1. [Quick start](#quick-start)
2. [The interface](#the-interface)
3. [Problem formulation](#1-problem-formulation)
4. [Representation](#2-representation)
5. [The NEAT algorithm](#3-neat)
6. [Code layout](#4-code-layout)
7. [Experiments](#5-experiments)
8. [Limitations and future work](#6-limitations-and-future-work)

---

## Quick start

```bash
git clone https://github.com/CyberSnakeH/SnakeNeuroEvolution.git
cd SnakeNeuroEvolution
uv sync

uv run snake-neat                                     # GUI, evolution starts immediately
uv run snake-neat --view species                      # GUI, one snake per species
uv run snake-neat play runs/demo/champion.json        # watch the pre-trained champion
uv run snake-neat train --generations 300 --out runs/exp1   # headless: CSV log + genomes
uv run snake-neat eval runs/exp1/champion.json --episodes 500   # held-out evaluation, 95 % CI
uv run python -m unittest discover -s tests -v        # unit tests
uv run python experiments/ablation.py --generations 300       # ablation study (section 5)
```

Common options: `--width/--height` (board size, default 15×15), `--pop` (population, 150), `--episodes`
(games per evaluation, 3), `--species` (target number of species, 10), `--compass` (extra inputs, see
§2.2), `--workers` (processes, 0 = all cores), `--seed`.

> **Python version.** The project pins the *system* interpreter (`.python-version` = 3.15,
> `python-preference = "system"`). The Tk bundled with uv-managed Python builds is compiled without
> Xft and can only render bitmap fonts. Any Python ≥ 3.12 with a Tk built with Xft works.

## The interface

**Champion view** (default):
- **board**: the best genome of the latest generation plays live. Its 8 vision rays are drawn: dashed
  for empty, red when the apple is seen, an orange ring on the first body segment hit.
- **network**: the champion's topology. Inputs are labelled (`FR body` = front-right ray, body channel).
  Hidden neurons are arranged by topological depth. Edge colour shows the sign of the weight and
  thickness shows |w|. Each node is coloured by its activation in real time, and the chosen action is
  highlighted.
- **fitness**: best and mean fitness per generation.
- **speciation**: stacked area chart of species sizes. One colour per species, the thickness of a band
  is the size of that species.

**All-species view**: a mosaic with one board per living species, up to 16. Each board shows the best
individual of its species, and its border uses that species' colour in the speciation chart. Click a
board to display that species' network. Species appear and go extinct as evolution proceeds. At about
0.2 s per generation this turnover is fast, so pause evolution to let the species finish their games.

![One board per species](docs/species.gif)

Shortcuts: `Space` pauses or resumes evolution, `v` switches view, `+` / `-` change replay speed. The
**Save shown genome…** button exports the genome currently displayed, including a selected species'
champion.

---

## 1. Problem formulation

The board is G = {0,…,W−1} × {0,…,H−1}. A state is s = (body, heading, apple, hunger), where the body is
a sequence of cells c₀ (head), …, c_{L−1}. The game is a Markov decision process, deterministic except
for the apple position, which is drawn uniformly among free cells. An episode ends on collision (wall or
body), on starvation (h consecutive steps without eating, h = W·H by default), or on a win (board full).

We search for a policy π : φ(s) ↦ a that maximizes the expected fitness

$$F = 1000 \cdot \text{apples} + \text{steps survived}.$$

Since h ≤ W·H < 1000, F induces an almost **lexicographic order**. A snake eating k+1 apples always beats
one eating k, and survival time only breaks ties. Survival without eating, for example circling forever,
is capped by h. This removes the degenerate "loop forever" strategy.

## 2. Representation

### 2.1 Relative actions
a ∈ {left, straight, right}, with heading ← heading + a − 1 (mod 4). The U-turn, which is always fatal,
cannot be expressed. The network has 3 outputs and the action is their argmax.

### 2.2 Egocentric observation
Eight rays are cast from the head in the **snake's own frame**: forward, forward-right, right, …,
forward-left. Let t be the number of steps along a ray:

| channel | value | range |
|---|---|---|
| wall | 1 / t_wall (first step outside the board) | ]0, 1] |
| body | 1 / t_body, 0 if no segment on the ray | [0, 1] |
| food | 1 − (t_food − 1)/max(W,H), 0 if not on the ray | [0, 1] |

which gives φ(s) ∈ [0,1]²⁴.

**Why egocentric?** When W = H, the cyclic group C₄ of board rotations acts on states, and the optimal
policy is equivariant. Expressing φ in the snake's frame gives φ(ρ·s) = φ(s) for every rotation ρ
(checked by `test_rotation_invariance_of_observation`). Combined with relative actions, one learned
behaviour serves all four headings, which cuts the behaviour space to discover by a factor of four.
Inverse distances give a sharp signal when danger is close, while food is encoded linearly so it stays
detectable from afar.

`--compass` adds 2 inputs: the head→apple vector projected on the forward and right axes, normalized.
With rays only, the apple is seen only when it is aligned with one of the 8 directions. Section 5
measures the effect of this extra information.

## 3. NEAT

### 3.1 Genome and historical markings
A genome is a set of nodes (input, output, hidden, each with a bias) and connection genes
(src, dst, w, enabled, innovation). The **innovation number** comes from a global registry keyed by the
pair (src, dst). The same structure therefore always carries the same number, which aligns two
arbitrary topologies in O(|C|) without any graph matching. A node created by splitting connection i also
gets a canonical id (i, k).

**Initial population**: the minimal topology, with all 24 inputs wired to the 3 outputs (72 genes) and
no hidden node. Complexity grows only when selection rewards it, which is the *complexification*
principle of NEAT.

### 3.2 Mutations
- **weights and biases**: with probability 0.8 per genome, each gene is touched with probability 0.5.
  It is then perturbed by 𝒩(0, 0.4²) in 90 % of cases, or redrawn from 𝒩(0, 1) in 10 %. Values are
  clipped to [−8, 8].
- **add node** (p = 0.03): a→b becomes a→h (weight 1) then h→b (old weight), and a→b is disabled. The
  behaviour is almost preserved for small pre-activations, since tanh(x) ≈ x.
- **add connection** (p = 0.10): a random edge is added only if it **creates no cycle**. The check covers
  all genes, including disabled ones, because they may be re-enabled later. The network is always a DAG.
- **toggle** enabled/disabled (p = 0.01).

### 3.3 Crossover
Homologous genes (same innovation) are drawn at random from either parent. Disjoint and excess genes come
from the fitter parent. A gene disabled in either parent stays disabled with probability 0.75. The child
inherits the *structure* of the fitter parent, which preserves acyclicity with no extra check.

### 3.4 Speciation and fitness sharing
Compatibility distance:

$$\delta(g_1, g_2) = c_1 \frac{E}{N} + c_2 \frac{D}{N} + c_3 \bar W,$$

where E is the number of excess genes, D the number of disjoint genes, W̄ the mean weight difference of
homologous genes, and N = max(|C₁|, |C₂|). We use c₁ = c₂ = 1 and c₃ = 0.4. A genome joins the first
species whose representative (a random member of the previous generation) lies at distance < δ_t. The
threshold is **controlled multiplicatively**, δ_t ← δ_t·(1 ± 0.05), to keep about 10 species. This
controller does not depend on the scale of δ.

**Explicit fitness sharing**: f'_i = f_i / |S|, hence Σ_{i∈S} f'_i = f̄_S. Species S receives

$$n_S = N_{\text{pop}} \cdot \frac{\bar f_S}{\sum_T \bar f_T}$$

offspring, rounded with the largest-remainder method so that Σ n_S = N_pop exactly. A species cannot grow
merely because it is large. This protects a structural innovation while its weights are being tuned.

The rest of reproduction follows the paper. Only the top 25 % of a species reproduce. The champion of
every species with ≥ 5 members is copied unchanged. A species that has not improved for 20 generations
goes extinct, except the 2 best ones and the one holding the global champion. 75 % of offspring come from
crossover (1 % of them between species), and every offspring is then mutated.

### 3.5 Phenotype: layered forward pass
Depth d(v) = 0 for an input, otherwise 1 + the maximum depth of its predecessors (computed with Kahn's
topological sort). Nodes at the same depth only depend on strictly shallower nodes, so

$$x_{L_k} \leftarrow \tanh(W_k x + b_k), \quad k = 1, \dots, d_{\max},$$

with a dense W_k ∈ ℝ^{|L_k|×|V|}. This takes d_max numpy matrix-vector products instead of a Python loop
over nodes. Exact equality with the recursive definition is checked by
`test_layered_forward_equals_recursive_definition` on 20 heavily mutated random networks.

### 3.6 Evaluation and variance control
Fitness is a random variable, because the apple is placed at random. Three measures control it:
1. **averaging over K = 3 games**;
2. **common random numbers**: within a generation, every genome plays the *same* K seeds. A fitness
   difference between two individuals then reflects their policies rather than the luck of the draws;
3. **re-evaluating elites**: seeds change every generation, and a copied elite is evaluated again.
   Otherwise the retained champion would be the maximum of noisy estimates, biased upwards (the
   *winner's curse*), and the population could overfit one particular apple sequence.

Evaluation is distributed over all cores with `multiprocessing`. It uses the *spawn* context, which is
safe alongside the GUI thread. The performance reported in §5 is measured on **seeds never used during
training**.

## 4. Code layout

```
src/snake_neat/
  config.py       hyperparameters (GameConfig, NEATConfig)
  game.py         Snake environment, ray casting, observation φ
  genome.py       genes, historical markings, mutations, crossover, distance δ
  network.py      genome → layered network compiler (numpy)
  population.py   speciation, fitness sharing, offspring allocation, stagnation, elitism
  evaluation.py   fitness, episodes, parallel evaluation, benchmark with 95 % CI
  persistence.py  JSON save / load of genomes
  gui.py          Tkinter interface
  cli.py          gui / play / train / eval commands
tests/            15 unit tests (unittest)
experiments/      ablation study and SVG learning-curve plot
runs/demo/        pre-trained champion (300 generations) and its held-out evaluation
runs/ablation/    raw ablation data
```

## 5. Experiments

**Protocol.** 15×15 board, population 150, K = 3 games per evaluation, 300 generations, 3 independent
seeds per variant. Each training run takes about 50 s on 24 cores. We measure the champion of the last
generation on **200 test games** whose seeds were never used in training. Reproduce with
`uv run python experiments/ablation.py --generations 300 --seeds 0 1 2`. The raw data is in
`runs/ablation/results.json`.

### 5.1 Reference performance
The demo champion (`runs/demo/champion.json`, seed 0, 300 generations) scores **60.4 ± 0.8 apples**
(95 % CI, 500 test games). The median is 60 and the best game reaches 88 apples, a length of 91 cells,
or 40 % of the board. Causes of death: body 69 %, wall 26 %, starvation 4 %. A random snake eats about
0.1 apple per game. About **8 generations** are enough to exceed 20 apples.

### 5.2 Ablation

| Variant | test apples (mean of 3 seeds) | std. across seeds | Welch t vs baseline | champion hidden nodes |
|---|---|---|---|---|
| **NEAT (baseline)** | **60.8** | 5.9 | — | 9.3 |
| δ unnormalized (N = 1) | 56.3 | 3.0 | 1.2 | 13.0 |
| no speciation | 54.2 | 1.3 | 1.9 | 6.0 |
| fixed topology (weights only, 24→3) | 52.6 | 1.4 | 2.4 | 0 |
| + food compass (26 inputs) | 50.6 | 1.0 | 3.0 | 7.0 |

![Learning curves](docs/ablation_curves.svg)

**Critical reading.**
1. **The full configuration comes first, but the gap is not statistically established.** With 3 seeds
   per variant, Welch's t (about 2.2 degrees of freedom) does not reach the 5 % threshold for any
   variant: it would need t ≳ 4. The training curves overlap widely, and every variant reaches a
   plateau of 55–62 apples within about 100 generations. The careful conclusion is that **on this
   problem, topological complexification and speciation bring at most a modest gain**. At least 10 seeds
   per variant would be needed to decide.
2. **A perceptron with no hidden layer (24→3) is already very strong.** With a well-designed egocentric
   perception, a nearly linear policy eats about 50 apples. This is the premise of NEAT: start minimal,
   complexify only when selected. NEAT champions indeed recruit only 5 to 13 hidden neurons.
3. **The compass hurts performance**, and it is the clearest effect: low variance, and a curve
   consistently below the others after 100 generations. A hypothesis, not tested here: an always-available
   gradient towards the apple favours a greedy straight-line policy that traps the snake sooner. With
   rays only, the snake has to sweep the board, which keeps it more orderly. The compass is therefore off
   by default.
4. **Picking the final champion is noisy.** The mean training score over the last 50 generations (60 to
   62 apples for four variants) is often higher than the test score of the single champion retained. A
   more robust selection, such as re-evaluating the champions of the last N generations on a validation
   set, would reduce this variance.
5. **The current ceiling is about 60 apples on 15×15.** It comes from perception, not from the optimizer
   (§6): 69 % of deaths are collisions with the body, typically a self-trap that local rays cannot
   anticipate.

## 6. Limitations and future work

- **Local perception**: the 8 rays give neither the global shape of the body nor the position of the
  tail, so the snake cannot plan ahead to avoid enclosing itself. This explains most "body" deaths once
  the snake is long. Possible additions: tail direction, a local k×k map, or a reachability feature such
  as the size of the free connected region ahead of each action.
- **Recurrent connections** are excluded here. Allowing them would give memory, for example of the
  last known apple position.
- **Normalization of δ**: with 72 initial genes, the topological terms E/N and D/N are diluted. The
  "δ unnormalized" ablation measures the effect of a structure-sensitive speciation.
- **Cost**: about 0.1–0.5 s per generation on 24 cores. Cost grows with game length, hence with the
  skill of the snakes.

## Reference

K. O. Stanley and R. Miikkulainen. *Evolving Neural Networks through Augmenting Topologies.*
Evolutionary Computation 10(2):99–127, 2002.
