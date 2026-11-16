# Mathematical model and analysis

This document gives a self-contained formal description of the system implemented in `src/snake_neat/`:
the control problem, the observation map, the policy class, the NEAT variation and selection operators,
the fitness estimator, and the statistical protocol behind the experiments. Every statement is either a
definition matching the code, or a proposition with a proof. Remarks flag what is *not* guaranteed.

**Contents**

1. [Notation](#1-notation)
2. [The Snake control problem](#2-the-snake-control-problem)
3. [Observation map and rotational symmetry](#3-observation-map-and-rotational-symmetry)
4. [Policy class: genomes and phenotypes](#4-policy-class-genomes-and-phenotypes)
5. [Variation operators](#5-variation-operators)
6. [Speciation and selection](#6-speciation-and-selection)
7. [Fitness estimation](#7-fitness-estimation)
8. [The complete algorithm](#8-the-complete-algorithm)
9. [Computational complexity](#9-computational-complexity)
10. [Statistical methodology](#10-statistical-methodology)
11. [From mathematics to code](#11-from-mathematics-to-code)
12. [References](#12-references)

---

## 1. Notation

| Symbol | Meaning | Default |
|---|---|---|
| $`W, H`$ | board width and height | $`15, 15`$ |
| $`\mathcal{G}`$ | set of cells $`\{0, \dots, W-1\} \times \{0, \dots, H-1\}`$ | |
| $`u_h`$ | unit move of heading $`h \in \mathbb{Z}/4\mathbb{Z}`$ | |
| $`L_0`$ | initial body length | $`3`$ |
| $`\eta_{\max}`$ | hunger limit (steps without eating) | $`WH = 225`$ |
| $`B`$ | reward per apple in the fitness | $`1000`$ |
| $`A, T`$ | apples eaten, steps survived in one episode | |
| $`\phi`$ | observation map, $`\phi(s) \in [0, 1]^{24}`$ | |
| $`g = (\mathcal{N}, \mathcal{C})`$ | genome: node genes, connection genes | |
| $`N_{\mathrm{pop}}`$ | population size | $`150`$ |
| $`K`$ | episodes per fitness evaluation | $`3`$ |
| $`\delta, \delta_t`$ | compatibility distance, speciation threshold | $`\delta_t^{(0)} = 1`$ |
| $`K^\star`$ | target number of species | $`10`$ |

Coordinates follow screen conventions: $`x`$ grows to the right and $`y`$ grows **downwards**. Headings are
indexed clockwise,

```math
u_0 = (0, -1), \quad u_1 = (1, 0), \quad u_2 = (0, 1), \quad u_3 = (-1, 0),
```

and $`R(x, y) = (-y, x)`$ denotes the clockwise quarter turn, so that $`R\, u_h = u_{h+1}`$ for all $`h`$.

---

## 2. The Snake control problem

**Definition 2.1 (State).** A state is a tuple $`s = (\mathbf{c}, h, p, \eta)`$ where

- $`\mathbf{c} = (c_0, \dots, c_{L-1}) \in \mathcal{G}^L`$ is the body, with pairwise distinct cells, head $`c_0`$, and
  $`\lVert c_i - c_{i+1} \rVert_1 = 1`$ for consecutive segments;
- $`h \in \mathbb{Z}/4\mathbb{Z}`$ is the heading;
- $`p \in \mathcal{G} \setminus \{c_0, \dots, c_{L-1}\}`$ is the apple, or $`p = \bot`$ when the board is full;
- $`\eta \in \{0, \dots, \eta_{\max} - 1\}`$ is the number of steps since the last meal.

**Definition 2.2 (Initial distribution).** The heading is uniform, $`h \sim \mathcal{U}(\mathbb{Z}/4\mathbb{Z})`$, the head
is the central cell $`c_0 = (\lfloor W/2 \rfloor, \lfloor H/2 \rfloor)`$, the body is the straight segment
$`c_i = c_0 - i\, u_h`$ for $`i = 0, \dots, L_0 - 1`$, the hunger is $`\eta = 0`$, and the apple is uniform
over the free cells.

**Definition 2.3 (Dynamics).** The action set is $`\mathcal{A} = \{0, 1, 2\}`$ (left, straight, right). Given
$`s`$ and $`a \in \mathcal{A}`$, define the new heading and the candidate head

```math
h' \equiv h + a - 1 \pmod{4}, \qquad c' = c_0 + u_{h'} ,
```

and the eating indicator $`e = \mathbf{1}[c' = p]`$. The move is **fatal** if

```math
c' \notin \mathcal{G} \quad \text{or} \quad c' \in \{c_0, \dots, c_{L-2+e}\}.
```

The last body cell $`c_{L-1}`$ is excluded when $`e = 0`$ because the tail leaves it during the same step.
Otherwise the new body is

```math
\mathbf{c}' = \begin{cases} (c', c_0, \dots, c_{L-1}) & \text{if } e = 1, \\ (c', c_0, \dots, c_{L-2}) & \text{if } e = 0, \end{cases}
```

and if $`e = 1`$ a new apple is drawn as $`p' \sim \mathcal{U}(\mathcal{G} \setminus \mathbf{c}')`$ with $`\eta' = 0`$, while if
$`e = 0`$ the hunger becomes $`\eta' = \eta + 1`$ and the episode ends by starvation when $`\eta' = \eta_{\max}`$.
The episode ends by a win when $`\mathbf{c}'`$ covers $`\mathcal{G}`$.

This defines an episodic Markov decision process whose only randomness is the initial heading and the
apple placements.

**Proposition 2.4 (Episodes terminate).** Every episode ends after at most

```math
T \le \eta_{\max}\, (WH - L_0 + 1)
```

steps.

*Proof.* Split the trajectory at the steps where an apple is eaten. This gives $`A + 1`$ segments of
lengths $`\tau_1, \dots, \tau_{A+1}`$. Inside a segment the hunger increases by one per step and the
episode ends when it reaches $`\eta_{\max}`$, so $`\tau_i \le \eta_{\max}`$. Each meal increases the length by
one and the length never exceeds $`WH`$, so $`A \le WH - L_0`$. Hence $`T = \sum_i \tau_i \le \eta_{\max}(A + 1)`$, which
gives the bound. $`\square`$

**Definition 2.5 (Episode fitness).** The fitness of an episode is $`F = B A + T`$ with $`B = 1000`$.

**Proposition 2.6 (Structure of the fitness).** With the segment lengths $`\tau_i`$ of the proof above,

```math
F = \sum_{i=1}^{A} \big( B + \tau_i \big) + \tau_{A+1}, \qquad 1 \le \tau_i \le \eta_{\max},
```

and therefore

```math
(B + 1)\, A + 1 \;\le\; F \;\le\; (B + \eta_{\max})\, A + \eta_{\max}.
```

Consequently, for two episodes with $`A_1 > A_2`$, the inequality $`F_1 > F_2`$ is **guaranteed** by these bounds
when $`(B + 1)A_1 + 1 > (B + \eta_{\max})A_2 + \eta_{\max}`$. With $`A_1 = A_2 + 1`$ this condition reads

```math
A_2 < \frac{B + 2 - \eta_{\max}}{\eta_{\max} - 1} = \frac{777}{224} \approx 3.47 .
```

so the guarantee covers $`A_2 \le 3`$ only. Beyond that, the bounds allow $`F_1 < F_2`$.

*Proof.* The decomposition follows from $`T = \sum_i \tau_i`$, and the bounds from $`1 \le \tau_i \le \eta_{\max}`$. The
sufficient condition compares the smallest possible $`F_1`$ with the largest possible $`F_2`$. $`\square`$

> **Remark 2.7 (the fitness is not lexicographic).** $`F`$ ranks apples first only for weak snakes (at most
> 3 apples). Beyond that, a slow eater can outscore a faster one with one more apple, because each apple
> carries up to $`\eta_{\max}`$ extra survival points. An exact lexicographic order would need
> $`B > \eta_{\max}(WH - L_0 + 1)`$, i.e. $`B > 50\,175`$ here. The value $`B = 1000`$ is part of the task
> specification and is kept. In practice $`F/B`$ remains a close proxy for $`A`$: by Proposition 2.6,
> $`A \le F/B`$ and $`A \ge (F - \eta_{\max})/(B + \eta_{\max})`$.

---

## 3. Observation map and rotational symmetry

Let $`f = u_h`$ be the heading vector and $`r = R f`$ the snake's right-hand side. The eight ray directions,
listed clockwise from the front, are

```math
(d_0, \dots, d_7) = (f,\; f + r,\; r,\; r - f,\; -f,\; -f - r,\; -r,\; f - r).
```

**Definition 3.1 (Hitting times).** For ray $`k`$ and a set $`S \subseteq \mathbb{Z}^2`$, let
$`t_k(S) = \min \{ t \ge 1 : c_0 + t\, d_k \in S \}`$ with $`\min \emptyset = +\infty`$. Define

```math
t^{\mathrm{wall}}_k = t_k\big(\mathbb{Z}^2 \setminus \mathcal{G}\big), \qquad
t^{\mathrm{body}}_k = t_k\big(\{c_0, \dots, c_{L-1}\}\big), \qquad
t^{\mathrm{food}}_k = t_k\big(\{p\}\big).
```

Since the head lies inside the board, $`t^{\mathrm{wall}}_k`$ is finite and the scan stops there.

**Definition 3.2 (Observation map).** With the convention $`1/\infty = 0`$ and $`m = \max(W, H)`$,

```math
\phi_{3k}(s) = \frac{1}{t^{\mathrm{wall}}_k}, \qquad
\phi_{3k+1}(s) = \frac{1}{t^{\mathrm{body}}_k}, \qquad
\phi_{3k+2}(s) = \begin{cases} 1 - \dfrac{t^{\mathrm{food}}_k - 1}{m} & \text{if } t^{\mathrm{food}}_k < \infty, \\[4pt] 0 & \text{otherwise.} \end{cases}
```

for $`k = 0, \dots, 7`$. All features lie in $`[0, 1]`$, so $`\phi(s) \in [0, 1]^{24}`$. The optional compass adds
$`\phi_{24} = \langle p - c_0, f \rangle / m`$ and $`\phi_{25} = \langle p - c_0, r \rangle / m`$.

The inverse distance gives a large signal when an obstacle is adjacent and a small one when it is far.
The food channel is affine in the distance so that a distant apple remains clearly visible.

**Definition 3.3 (Rotation of states).** Assume $`W = H = n`$ and let $`\rho(x, y) = (n - 1 - y,\; x)`$ be the
clockwise quarter turn of the board. It acts on states by

```math
\rho \cdot (\mathbf{c}, h, p, \eta) = \big( (\rho(c_0), \dots, \rho(c_{L-1})),\; h + 1,\; \rho(p),\; \eta \big).
```

**Proposition 3.4 (Invariance of the observation).** For every state $`s`$ on a square board,
$`\phi(\rho \cdot s) = \phi(s)`$, and the same holds with the compass features.

*Proof.* The map $`\rho`$ is a bijection of $`\mathcal{G}`$ onto itself, and it is affine with linear part $`R`$:
$`\rho(z + v) = \rho(z) + R v`$. In $`\rho \cdot s`$ the heading vector is $`u_{h+1} = R f`$ and the right-hand vector
is $`R(R f) = R r`$. Each ray direction is a linear combination of $`f`$ and $`r`$, so every ray direction of
$`\rho \cdot s`$ is $`R\, d_k`$. The $`t`$-th cell scanned along ray $`k`$ in $`\rho \cdot s`$ is therefore

```math
\rho(c_0) + t\, R d_k = \rho\big(c_0 + t\, d_k\big),
```

the image of the $`t`$-th cell scanned in $`s`$. Since $`\rho`$ is a bijection of $`\mathcal{G}`$ that maps the body
onto the body and $`p`$ onto the apple, each membership test along the ray has the same outcome. All
hitting times coincide, hence all 24 features coincide. For the compass, $`R`$ is orthogonal, so
$`\langle R e, R f \rangle = \langle e, f \rangle`$ and likewise with $`r`$. $`\square`$

**Corollary 3.5 (Equivariance of the dynamics and of the policies).** Rotating the state commutes with
the dynamics: the transition kernel satisfies $`P(\rho \cdot s' \mid \rho \cdot s, a) = P(s' \mid s, a)`$, because relative
actions are rotation-invariant and the uniform apple law is invariant under the bijection $`\rho`$.
Every policy of the form $`\pi(s) = \psi(\phi(s))`$ is therefore automatically invariant under the group
$`C_4 = \{\rho^0, \rho^1, \rho^2, \rho^3\}`$. A single learned behaviour serves the four headings.

> **Remark 3.6 (partial observability).** $`\phi`$ is not injective. Two states that differ only in body
> segments lying off the eight rays have the same observation. The agent therefore acts in a partially
> observable process with a memoryless policy, and no optimality guarantee carries over from the MDP.
> This aliasing is the main source of self-trapping deaths (Section 5 of the README).

---

## 4. Policy class: genomes and phenotypes

**Definition 4.1 (Genome).** A genome is a pair $`g = (\mathcal{N}, \mathcal{C})`$.

- $`\mathcal{N} = \mathcal{N}_{\mathrm{in}} \sqcup \mathcal{N}_{\mathrm{out}} \sqcup \mathcal{N}_{\mathrm{hid}}`$ is a finite set of node genes,
  each with a bias $`b_v \in \mathbb{R}`$, with $`\lvert \mathcal{N}_{\mathrm{in}} \rvert = 24`$ and $`\lvert \mathcal{N}_{\mathrm{out}} \rvert = 3`$.
- $`\mathcal{C}`$ is a finite set of connection genes $`\gamma = (u, v, w, \epsilon, \iota)`$: source $`u`$, target $`v`$,
  weight $`w \in [-8, 8]`$, enabled flag $`\epsilon \in \{0, 1\}`$, and innovation number $`\iota \in \mathbb{N}`$.
- Innovation numbers come from a global injective registry $`\iota : (u, v) \mapsto \mathbb{N}`$. Two genes,
  possibly in different genomes, have the same innovation number if and only if they connect the same
  pair of nodes.

Write $`\mathcal{C}_{\mathrm{on}} = \{ \gamma \in \mathcal{C} : \epsilon = 1 \}`$ for the enabled genes.

**Definition 4.2 (Phenotype).** Assume the directed graph $`(\mathcal{N}, \mathcal{C}_{\mathrm{on}})`$ is acyclic. For an input
vector $`z \in \mathbb{R}^{24}`$, the activations are defined recursively by $`x_v = z_v`$ for $`v \in \mathcal{N}_{\mathrm{in}}`$ and

```math
x_v = \tanh\Big( b_v + \sum_{(u, v, w, 1, \iota) \in \mathcal{C}_{\mathrm{on}}} w\, x_u \Big) \qquad \text{for } v \notin \mathcal{N}_{\mathrm{in}} .
```

The policy is $`\pi_g(s) = \arg\max_{k \in \{0,1,2\}} x_{o_k}(\phi(s))`$, with ties broken by the smallest index.

The recursion is well defined because the graph is acyclic (Proposition 5.1). Note that
$`\tanh(z) = 2\sigma(2z) - 1`$, so the activation belongs to the same family as the steepened sigmoid
$`\sigma(4.9 z)`$ of Stanley and Miikkulainen.

**Definition 4.3 (Depth and layers).** Set $`d(v) = 0`$ for $`v \in \mathcal{N}_{\mathrm{in}}`$ and, for the other nodes,

```math
d(v) = 1 + \max \{ d(u) : (u, v, w, 1, \iota) \in \mathcal{C}_{\mathrm{on}} \}, \qquad \text{with } \max \emptyset = 0 .
```

The layers are $`L_k = \{ v : d(v) = k \}`$ for $`k = 1, \dots, d_{\max}`$.

**Proposition 4.4 (Layered evaluation is exact).** Let $`W_k \in \mathbb{R}^{\lvert L_k \rvert \times \lvert \mathcal{N} \rvert}`$ hold the weights
of the enabled genes entering $`L_k`$, and $`b_k`$ the biases of $`L_k`$. Starting from $`x_{\mathcal{N}_{\mathrm{in}}} = z`$ and
$`x_v = 0`$ elsewhere, the sequence of updates

```math
x_{L_k} \leftarrow \tanh\big( W_k\, x + b_k \big), \qquad k = 1, 2, \dots, d_{\max},
```

produces exactly the activations of Definition 4.2.

*Proof.* For every enabled edge $`u \to v`$ the definition of depth gives $`d(u) \le d(v) - 1 < d(v)`$. By induction
on $`k`$: when layer $`L_k`$ is updated, every predecessor of every $`v \in L_k`$ lies in a layer of index $`< k`$, or is
an input, and so already holds its final value. The non-zero entries of row $`v`$ of $`W_k`$ are exactly the
weights of the edges entering $`v`$, so the update computes Definition 4.2 for $`v`$. Entries of $`x`$ not yet
updated are multiplied by zero weights. $`\square`$

Depths are computed with Kahn's topological sort in $`O(\lvert \mathcal{N} \rvert + \lvert \mathcal{C} \rvert)`$. The sort visits every node if
and only if the graph is acyclic, which the code uses as a runtime check. The forward pass costs
$`d_{\max}`$ dense products, i.e. $`O(\lvert \mathcal{N} \rvert^2)`$ floating-point operations, but only $`d_{\max}`$ calls into numpy.
The equality is also verified numerically by `test_layered_forward_equals_recursive_definition`.

---

## 5. Variation operators

All operators act on the full gene set $`\mathcal{C}`$, enabled or not.

**Proposition 5.1 (Acyclicity is invariant).** If every genome of the initial population has an acyclic
graph $`(\mathcal{N}, \mathcal{C})`$ (all genes, enabled or not), then so does every genome produced by the algorithm.
In particular every phenotype is well defined.

*Proof.* Let $`\mathcal{I}`$ be the property "$`(\mathcal{N}, \mathcal{C})`$ is a DAG".

- *Initialisation.* The initial graph only has edges from inputs to outputs. It is bipartite, hence acyclic.
- *Add connection.* An edge $`u \to v`$ is added only if $`v`$ cannot reach $`u`$ in $`(\mathcal{N}, \mathcal{C})`$, which is checked
  by depth-first search. A new cycle would have to use the new edge and then a path from $`v`$ back to
  $`u`$, which does not exist.
- *Add node.* An edge $`u \to v`$ is disabled but kept in $`\mathcal{C}`$, and a fresh node $`h`$ with edges $`u \to h`$ and
  $`h \to v`$ is added. A cycle through $`h`$ would contain the path $`u \to h \to v`$ followed by a path from $`v`$
  to $`u`$. Together with the existing edge $`u \to v`$, that path would already form a cycle, contradicting $`\mathcal{I}`$.
- *Weights, biases, toggles.* These leave the set of edges unchanged.
- *Crossover.* The child receives exactly the innovation numbers of the fitter parent. Because $`\iota`$ is
  injective, each inherited gene has the same endpoints as in that parent, so the child's graph equals
  the fitter parent's graph.

Every operator preserves $`\mathcal{I}`$, and the enabled subgraph of a DAG is a DAG. $`\square`$

**Weight mutation.** With probability $`0.8`$ a genome mutates its weights. Each gene is then touched
independently with probability $`0.5`$ and updated by the mixture kernel

```math
w \leftarrow \mathrm{clip}_{[-8,\, 8]}(w'), \qquad
w' \sim \begin{cases} w + \mathcal{N}(0, 0.4^2) & \text{with probability } 0.9, \\ \mathcal{N}(0, 1) & \text{with probability } 0.1 . \end{cases}
```

Biases follow the same kernel with genome-level probability $`0.5`$. The perturbation explores locally,
while the reset gives the chain a positive probability of reaching any weight in a single step.

**Structural mutations.** Add node with probability $`0.03`$, add connection with probability $`0.10`$, and
toggle the enabled flag of one gene with probability $`0.01`$, each independently. Up to 20 random
endpoint pairs are tried for an add-connection. A pair whose edge already exists but is disabled gets
re-enabled, which is safe by Proposition 5.1.

**Proposition 5.2 (Adding a node is a small perturbation).** At the moment of the split, replacing $`u \to v`$ of weight $`w`$ changes
the pre-activation of $`v`$ from $`w\, x_u`$ to $`w \tanh(x_u)`$, and

```math
\big\lvert w \tanh(x_u) - w\, x_u \big\rvert \;\le\; \frac{\lvert w \rvert\, \lvert x_u \rvert^3}{3} \;\le\; \frac{\lvert w \rvert}{3} .
```

*Proof.* The new node has bias $`0`$ and a single incoming edge of weight $`1`$, so $`x_h = \tanh(x_u)`$. Let
$`q(z) = z - \tanh z`$ for $`z \ge 0`$. Then $`q(0) = 0`$ and $`q'(z) = \tanh^2 z \le z^2`$, so $`0 \le q(z) \le z^3/3`$.
The bound extends to all $`z`$ because $`q`$ is odd. Finally $`\lvert x_u \rvert \le 1`$, since inputs lie in $`[0, 1]`$ and other
activations in $`[-1, 1]`$. As $`\tanh`$ is 1-Lipschitz, the activation of $`v`$ moves by at most the same amount. $`\square`$

The new structure therefore starts close to the parent's function, and selection can tune it.

**Crossover.** Let $`g_1`$ be the fitter parent and $`g_2`$ the other one. For every gene $`\gamma \in \mathcal{C}_1`$:

- if $`g_2`$ has a gene with the same innovation number (*homologous*), the child takes the weight of
  either parent with probability $`1/2`$ each;
- otherwise (*disjoint* or *excess*) the child copies $`\gamma`$.

A homologous gene disabled in at least one parent is disabled in the child with probability $`0.75`$.
Biases of shared nodes are inherited uniformly from either parent.

---

## 6. Speciation and selection

**Definition 6.1 (Compatibility distance).** For genomes $`g_1, g_2`$ with gene sets $`\mathcal{C}_1, \mathcal{C}_2`$, let
$`M`$ be the set of shared innovation numbers and $`\iota^\ast = \min(\max \iota(\mathcal{C}_1), \max \iota(\mathcal{C}_2))`$. The
non-shared genes split into $`E`$ excess genes, with innovation number above $`\iota^\ast`$, and $`D`$ disjoint
genes, the others. Then

```math
\delta(g_1, g_2) = c_1 \frac{E}{N} + c_2 \frac{D}{N} + c_3\, \overline{W}, \qquad
N = \max\big( \lvert \mathcal{C}_1 \rvert, \lvert \mathcal{C}_2 \rvert \big), \qquad
\overline{W} = \frac{1}{\lvert M \rvert} \sum_{\iota \in M} \big\lvert w^{(1)}_\iota - w^{(2)}_\iota \big\rvert ,
```

with $`c_1 = c_2 = 1`$, $`c_3 = 0.4`$, and $`\overline{W} = 0`$ if $`M = \emptyset`$.

**Proposition 6.2 (Properties of the distance).** $`\delta`$ is non-negative, symmetric and satisfies
$`\delta(g, g) = 0`$. It is bounded by $`2 \max(c_1, c_2) + 16\, c_3`$. It is **not** a metric: the triangle
inequality can fail.

*Proof.* Non-negativity, symmetry and $`\delta(g, g) = 0`$ are immediate. For the bound, $`E + D \le \lvert \mathcal{C}_1 \rvert + \lvert \mathcal{C}_2 \rvert \le 2N`$ and
$`\overline{W} \le 16`$, since weights lie in $`[-8, 8]`$. Counter-example to the triangle inequality, with
$`c_1 = c_2 = 1`$ and identical weights: let $`g_1`$ hold gene $`\{1\}`$, $`g_2`$ gene $`\{2\}`$, and $`g_3`$ genes $`\{1, 2\}`$. Then

```math
\delta(g_1, g_2) = \tfrac{1}{1} + \tfrac{1}{1} = 2 > 1 = \tfrac{1}{2} + \tfrac{1}{2} = \delta(g_1, g_3) + \delta(g_3, g_2). \qquad \square
```

Speciation is a greedy *leader clustering*. Each genome joins the first species, in creation order, whose
representative $`g_S`$ satisfies $`\delta(g, g_S) < \delta_t`$, or founds a new species. Representatives are
random members of the previous generation. Because $`\delta`$ is not a metric, the partition depends on
the processing order and species need not be balls.

**Threshold control.** With $`K_n`$ species at generation $`n`$, a target $`K^\star`$ and a rate $`\alpha = 0.05`$,

```math
\log \delta_t^{(n+1)} = \log \delta_t^{(n)} + \begin{cases} \log(1 - \alpha) & \text{if } K_n < K^\star, \\ 0 & \text{if } K_n = K^\star, \\ \log(1 + \alpha) & \text{if } K_n > K^\star, \end{cases}
```

with $`\delta_t \ge 0.05`$. This is a sign (bang-bang) integral controller in logarithmic coordinates, so its
behaviour does not depend on the scale of $`\delta`$. It drives $`K_n`$ to oscillate around $`K^\star`$; it does
not make $`K_n`$ converge.

**Definition 6.3 (Explicit fitness sharing).** With $`\hat F_i`$ the estimated fitness of individual $`i`$ and
$`S(i)`$ its species, the shared fitness is

```math
f'_i = \frac{\hat F_i}{\lvert S(i) \rvert} ,
```

which is the sharing scheme of Goldberg and Richardson with the step sharing function
$`\mathrm{sh}(i, j) = \mathbf{1}[S(i) = S(j)]`$. The total shared fitness of a species is
$`\sum_{i \in S} f'_i = \bar F_S`$, its mean fitness.

**Proposition 6.4 (Offspring allocation).** Let $`q_S = N_{\mathrm{pop}}\, \bar F_S / \sum_T \bar F_T`$ be the quota of species $`S`$ and
$`r = N_{\mathrm{pop}} - \sum_S \lfloor q_S \rfloor`$. Giving $`n_S = \lfloor q_S \rfloor + 1`$ to the $`r`$ species with the largest
fractional parts $`q_S - \lfloor q_S \rfloor`$, and $`n_S = \lfloor q_S \rfloor`$ to the others (largest-remainder method), yields

```math
\sum_S n_S = N_{\mathrm{pop}} \qquad \text{and} \qquad \lvert n_S - q_S \rvert < 1 \quad \text{for every species } S.
```

*Proof.* Since $`\sum_S q_S = N_{\mathrm{pop}}`$, the sum of fractional parts is the integer $`r`$, and $`0 \le r`$ is smaller than the
number of species. Adding one unit to $`r`$ species restores the total. Each $`n_S`$ is $`\lfloor q_S \rfloor`$ or $`\lfloor q_S \rfloor + 1`$,
both within distance $`< 1`$ of $`q_S`$. The denominator is positive because $`F \ge 1`$ for every episode. $`\square`$

> **Remark 6.5.** If the species of the global champion receives $`n_S = 0`$, one offspring is moved to it
> from the largest allocation. The quota property may then fail by one unit for two species. This
> guarantees that the best genome found so far survives.

**Reproduction.** For a species $`S`$ with allocation $`n_S \ge 1`$:

1. its champion is copied unmutated if $`\lvert S \rvert \ge 5`$, or if $`S`$ holds the global champion;
2. the parent pool is the best $`\max(2, \lceil 0.25 \lvert S \rvert \rceil)`$ members, capped at $`\lvert S \rvert`$ (truncation selection);
3. each remaining offspring comes from crossover with probability $`0.75`$, the second parent being drawn
   from the whole population with probability $`0.01`$, and is otherwise a clone. In both cases it is
   then mutated.

**Stagnation.** A species whose best fitness has not improved for 20 generations is removed, except the
two species with the best historical fitness and the species of the current champion.

---

## 7. Fitness estimation

Let $`\omega`$ denote the randomness of an episode (initial heading and apple stream) and
$`X(g, \omega) = B\, A(g, \omega) + T(g, \omega)`$ the episode fitness. The target is the expected fitness
$`F(g) = \mathbb{E}_\omega[X(g, \omega)]`$, and its variance is $`\sigma^2(g)`$.

**Definition 7.1 (Estimator).** At generation $`n`$, with seeds $`\omega_{n,1}, \dots, \omega_{n,K}`$ drawn by a seed sequence
from $`(\text{master seed}, n)`$,

```math
\hat F_n(g) = \frac{1}{K} \sum_{j=1}^{K} X(g, \omega_{n,j}) .
```

**Proposition 7.2 (Bias and variance).** If the seeds are independent, $`\hat F_n(g)`$ is unbiased with variance
$`\sigma^2(g) / K`$.

*Proof.* Linearity of expectation, and additivity of the variance for independent terms. $`\square`$

**Proposition 7.3 (Common random numbers).** If every genome is evaluated on the *same* seeds,

```math
\mathrm{Var}\big[\hat F_n(g_1) - \hat F_n(g_2)\big] = \frac{\sigma^2(g_1) + \sigma^2(g_2) - 2\, \mathrm{Cov}\big[X(g_1, \omega), X(g_2, \omega)\big]}{K} ,
```

whereas independent seeds give the same expression without the covariance term. Sharing seeds therefore
reduces the variance of every pairwise comparison exactly when the covariance is positive.

*Proof.* Expand the variance of $`\frac{1}{K} \sum_j (X(g_1, \omega_j) - X(g_2, \omega_j))`$, a mean of $`K`$ independent copies. $`\square`$

> **Remark 7.4.** Positive covariance is plausible, since similar policies facing the same apple stream
> behave similarly, but it is not proved. Note also that the apple position is produced by rejection
> sampling, whose number of draws depends on the board occupancy. Once two trajectories diverge, their
> random streams desynchronise, so the coupling is strongest early in the episode.

**Proposition 7.5 (Optimizer's curse).** Let $`\hat F_1, \dots, \hat F_N`$ be unbiased estimators of $`F_1, \dots, F_N`$ and
$`\hat\imath = \arg\max_i \hat F_i`$ the selected individual. Then

```math
\mathbb{E}\big[\hat F_{\hat\imath}\big] \;\ge\; \max_i F_i \;\ge\; \mathbb{E}\big[F_{\hat\imath}\big] .
```

The estimated score of the selected individual is biased upwards with respect to its true fitness.

*Proof.* Pointwise $`\max_i \hat F_i \ge \hat F_j`$ for every $`j`$, so $`\mathbb{E}[\max_i \hat F_i] \ge F_j`$ for every $`j`$, which gives the
first inequality. The second holds because $`F_{\hat\imath} \le \max_i F_i`$ pointwise. $`\square`$

**Consequence for the implementation.** Elites are copied *without* their fitness and evaluated again on
the seeds of the next generation, which are independent of the selection event. Their new estimate is
unbiased, so a lucky genome cannot remain champion on the strength of one favourable draw. Changing the
seeds every generation also prevents overfitting to one particular apple sequence. Reported performance
is always measured on held-out seeds, starting at $`10^6`$, never used in training.

---

## 8. The complete algorithm

```text
Algorithm 1  NEAT for Snake (one run)
──────────────────────────────────────────────────────────────────────────────
Input : population size N_pop, episodes K, master seed σ, generations G
P ← N_pop minimal genomes (24 → 3, fully connected, weights ~ N(0, 1))
speciate(P)
for n = 0, 1, …, G − 1 do
    (ω_1, …, ω_K) ← SeedSequence(σ, n)                     ▷ common random numbers
    for each g ∈ P in parallel do
        F̂(g) ← (1/K) Σ_j [ B·A(g, ω_j) + T(g, ω_j) ]         ▷ Section 7
    record statistics, champion g* = argmax F̂
    update species best fitness; drop stagnant species       ▷ Section 6
    allocate n_S by largest remainders on mean fitness       ▷ Proposition 6.4
    P' ← ∅
    for each species S do
        P' ← P' ∪ elite(S) ∪ { mutate(crossover or clone of parents in top 25 %) }
    P ← P'
    speciate(P); adapt δ_t                                   ▷ Section 6
```

```mermaid
flowchart LR
    A[Population P_n] --> B[Evaluate K episodes per genome<br/>shared seeds, all cores]
    B --> C[Statistics and champion]
    C --> D[Species fitness, stagnation]
    D --> E[Offspring allocation<br/>largest remainders]
    E --> F[Elitism, truncation selection,<br/>crossover, mutation]
    F --> G[Speciation, threshold control]
    G --> A
```

---

## 9. Computational complexity

Let $`m = \max(W, H)`$, $`\lvert \mathcal{N} \rvert`$ the number of nodes, $`\bar T`$ the mean episode length, and $`P`$ the number of
worker processes.

| Operation | Cost |
|---|---|
| Observation, 8 rays with $`O(1)`$ set membership | $`O(m)`$ per step |
| Forward pass | $`O(d_{\max} \lvert \mathcal{N} \rvert)`$ numpy calls, $`O(\lvert \mathcal{N} \rvert^2)`$ flops per step |
| Evaluation of one generation | $`O\big(N_{\mathrm{pop}} K \bar T (m + \lvert \mathcal{N} \rvert^2) / P\big)`$ |
| Speciation | $`O(N_{\mathrm{pop}}\, K_n\, \lvert \mathcal{C} \rvert)`$ |
| Reproduction | $`O(N_{\mathrm{pop}}\, \lvert \mathcal{C} \rvert)`$ |

Evaluation dominates. Since $`\bar T`$ grows with skill (Proposition 2.4 bounds it by $`\eta_{\max}(A + 1)`$),
generations slow down as the snakes improve: from about 0.05 s to 0.3 s on 24 cores in our runs.

---

## 10. Statistical methodology

**Confidence interval.** For a genome evaluated on $`n`$ held-out episodes with apple counts
$`A_1, \dots, A_n`$, mean $`\bar A`$ and sample standard deviation $`s_A`$, the reported 95 % interval is the
normal-approximation interval

```math
\bar A \pm 1.96\, \frac{s_A}{\sqrt{n}} .
```

**Comparing variants.** Each variant is trained with $`n_1 = n_2 = 3`$ seeds, giving three test scores per
variant. Variances differ across variants, so we use Welch's test,

```math
t = \frac{\bar A_1 - \bar A_2}{\sqrt{s_1^2 / n_1 + s_2^2 / n_2}}, \qquad
\nu = \frac{\big( s_1^2 / n_1 + s_2^2 / n_2 \big)^2}{\dfrac{(s_1^2 / n_1)^2}{n_1 - 1} + \dfrac{(s_2^2 / n_2)^2}{n_2 - 1}} ,
```

where $`\nu`$ is the Welch–Satterthwaite degrees of freedom. The two-sided p-value under the Student law
with $`\nu`$ degrees of freedom is

```math
p = I_{\nu / (\nu + t^2)}\Big( \frac{\nu}{2}, \frac{1}{2} \Big),
```

where $`I_x(a, b)`$ is the regularized incomplete beta function. Four variants are compared with the
baseline, so p-values are adjusted with the Holm–Bonferroni procedure: sort $`p_{(1)} \le \dots \le p_{(4)}`$ and set
$`\tilde p_{(k)} = \max_{j \le k} \min\big(1, (5 - j)\, p_{(j)}\big)`$.

`experiments/significance.py` computes the table below from `runs/ablation/results.json`.

| Variant | mean | std. | $`t`$ | $`\nu`$ | $`p`$ | Holm $`\tilde p`$ |
|---|---|---|---|---|---|---|
| NEAT (baseline) | 60.8 | 5.9 | | | | |
| δ unnormalized ($`N = 1`$) | 56.3 | 3.0 | 1.19 | 2.95 | 0.322 | 0.389 |
| no speciation | 54.2 | 1.3 | 1.90 | 2.19 | 0.187 | 0.389 |
| fixed topology (weights only) | 52.6 | 1.4 | 2.37 | 2.21 | 0.130 | 0.389 |
| + food compass | 50.6 | 1.0 | 2.96 | 2.11 | 0.092 | 0.367 |

**Reading.** No difference is significant at the 5 % level, either before or after correction. With
three seeds and $`\nu \approx 2`$, the test has very low power: rejecting at 5 % would require
$`\lvert t \rvert`$ above the 97.5 % Student quantile, which is 4.30 for $`\nu = 2`$ and 3.18 for $`\nu = 3`$. The ordering of the means is suggestive, but it is not evidence. A conclusive
comparison would need on the order of ten seeds per variant.

---

## 11. From mathematics to code

| Object | Symbol | Code |
|---|---|---|
| Game parameters | $`W, H, L_0, \eta_{\max}`$ | `config.GameConfig` |
| NEAT hyperparameters | $`c_1, c_2, c_3, \delta_t, K^\star, \alpha`$, mutation rates | `config.NEATConfig` |
| Dynamics (Def. 2.3) | | `game.SnakeGame.step` |
| Rays and hitting times (Def. 3.1) | $`d_k, t_k`$ | `game.ray_directions`, `game.SnakeGame.cast_rays` |
| Observation map (Def. 3.2) | $`\phi`$ | `game.SnakeGame.observe` |
| Innovation registry | $`\iota`$ | `genome.InnovationTracker` |
| Genome (Def. 4.1) | $`g = (\mathcal{N}, \mathcal{C})`$ | `genome.Genome`, `NodeGene`, `ConnGene` |
| Cycle test (Prop. 5.1) | | `genome.Genome.creates_cycle` |
| Mutations, crossover (Sect. 5) | | `Genome.mutate*`, `Genome.crossover` |
| Distance (Def. 6.1) | $`\delta`$ | `genome.Genome.distance` |
| Depth, layers, forward pass (Prop. 4.4) | $`d, L_k, W_k`$ | `network.FeedForwardNetwork` |
| Speciation, threshold control | $`\delta_t`$ | `population.Population.speciate` |
| Sharing, allocation (Prop. 6.4) | $`f', n_S`$ | `population.Population._reproduce` |
| Estimator (Def. 7.1) | $`\hat F_n`$ | `evaluation.Evaluator` |
| Episode fitness (Def. 2.5) | $`F`$ | `evaluation.fitness` |
| Held-out evaluation, CI | | `evaluation.benchmark` |
| Welch test, Holm correction | $`t, \nu, p`$ | `experiments/significance.py` |

---

## 12. References

- K. O. Stanley and R. Miikkulainen. Evolving neural networks through augmenting topologies.
  *Evolutionary Computation*, 10(2):99–127, 2002.
- D. E. Goldberg and J. Richardson. Genetic algorithms with sharing for multimodal function optimization.
  *Proceedings of the Second International Conference on Genetic Algorithms*, 41–49, 1987.
- A. B. Kahn. Topological sorting of large networks. *Communications of the ACM*, 5(11):558–562, 1962.
- J. E. Smith and R. L. Winkler. The optimizer's curse: skepticism and postdecision surprise in decision
  analysis. *Management Science*, 52(3):311–322, 2006.
- B. L. Welch. The generalization of "Student's" problem when several different population variances are
  involved. *Biometrika*, 34(1–2):28–35, 1947.
- F. E. Satterthwaite. An approximate distribution of estimates of variance components.
  *Biometrics Bulletin*, 2(6):110–114, 1946.
- S. Holm. A simple sequentially rejective multiple test procedure. *Scandinavian Journal of Statistics*,
  6(2):65–70, 1979.
