# Pokémon Showdown project: how to frame the math (draft, 2026-10-11)

This builds on `final/Bayesian Memory and No-Regret Learning ... .txt`.

**Numerical checks:**
- L2 and L3 were checked in `scratchpad/pkmath/check.py`, on 2,000 random instances, using an exact LP solver.
- Other statements are standard results, cited rather than re-proved.
- L5 has not been checked.

## 0. The full game, and why we don't solve it

Gen 9 Random Battles form a **two-player zero-sum stochastic game**:
- **two-sided incomplete information:** each player knows only its own team;
- **simultaneous moves;**
- **chance:** damage rolls, accuracy, crits, secondary effects, speed ties.

The formal object is a factored-observation stochastic game (FOSG, Kovařík et al.), or a POSG. Its solution concept is Nash equilibrium. Because the game is two-player zero-sum, Nash equilibrium is the same as maximin, and the value is unique.

Solving this game exactly is hopeless. Every practical system approximates it with **depth-limited search plus a value function**, and that approximation should be stated openly. The theory is then about the **per-turn approximate game**, and about *what is lost* in each approximation step.

## 1. The per-turn game G(s, b, V̂)

| Object | Meaning |
|---|---|
| s | public state: revealed information, HP, field, our own team |
| θ ∈ Θ | opponent's hidden configuration: the sets of its Pokémon (moves, item, ability, tera) |
| b ∈ Δ(Θ) | our belief over θ |
| A | our legal actions (moves, tera variants, switches; up to about 13) |
| B_θ | opponent's legal actions, which depend on its type |
| M_θ[a, β] | E_chance[ V̂(next state) \| s, a, β, θ ]: one-turn simulation, then a leaf value |

**Bayesian game in agent form.** We pick x ∈ Δ(A). The opponent picks one strategy y_θ ∈ Δ(B_θ) *per type*, because it knows θ. The game is zero-sum, so our guaranteed value is

  v(b) = max_x f_b(x),  where f_b(x) := Σ_θ b(θ) · min_{β∈B_θ} (M_θᵀ x)_β.

This is an LP with |A| + |Θ| variables. Regret matching gives the same answer iteratively:
- we run regret matching on x;
- each type θ runs its own regret matching on y_θ;
- after T rounds, the average x̄ satisfies f_b(x̄) ≥ v(b) − ε_T, with ε_T = O(R(√|A| + max_θ√|B_θ|)/√T) (Hart–Mas-Colell; zero-sum self-play folklore).

Here R is the payoff range.

**What makes this "Bayesian" rather than "average the matrices":** the min is *inside* the sum over θ, because each opponent type best-responds separately. The opposite approach solves each sampled θ and averages the answers. That is Perfect Information Monte Carlo (PIMC), and it suffers from *strategy fusion* (Frank & Basin 1998; Long et al. 2010). The agent-form LP avoids strategy fusion by construction.

## 2. Lemmas worth proving (each is short)

**L1 (agent-form LP):**
- v(b) is the value of the Bayesian zero-sum per-turn game.
- The optimal x comes from one LP.
- Regret matching converges at the rate above.

*Status:* textbook. State it and cite it.

**L2 (error decomposition: the proposal's "Nash gap ≤ solver + belief + value" made precise).** Suppose we solve the game built from an estimated belief b̂ and estimated payoffs M̂, and output x̂ with solver gap ε_T. Then, in the *true* per-turn game (b, M),

  v(b) − f_b(x̂) ≤ ε_T + 2R · TV(b, b̂) + 2 · max_θ ‖M_θ − M̂_θ‖_∞.

*Proof:* f is R-Lipschitz in b in total-variation distance, and 1-Lipschitz in M in sup-norm. Apply that to both v and f, then use the triangle inequality.

*Check:* across 2,000 random instances, the worst observed gap/bound ratio was 0.48, with no violations.

*Sampled types:* with K sampled sets instead of exact b, add a uniform-convergence term O(R√(|A| log K / K)) using a covering argument over the simplex.

**L2′ (compression: the proposal's "memory sufficiency" question).** Storing per-attribute *marginals* instead of the *joint* belief over sets is a special case of L2: the loss is at most 2R · TV(b_joint, ∏ marginals).

A two-type toy shows the bound can be tight. For example, "Choice Scarf only appears with set A" makes the product of marginals put mass on impossible combinations, and that changes the best action.

This gives a precise answer to "ignoring correlations can be exploitable".

**L3 (value of information, one-shot).**
- v(b) is **convex** in b, because it is a max of functions linear in b. *Check:* 0 violations across 2,000 random instances.
- So for any information source that splits the prior into posteriors b_k with probabilities q_k, the value of information VOI = Σ q_k v(b_k) − v(b) is ≥ 0. This is Jensen's inequality.
- **VOI = 0 if and only if some x optimal at the prior stays optimal at every posterior.**

The last bullet is exactly the proposal's distinction between "information that changes the optimal strategy" and "information that merely reduces uncertainty".

*Proof of the iff:* f(x*, b) = Σ q_k f(x*, b_k) ≤ Σ q_k v(b_k), with equality iff x* is optimal at each b_k.

*Probing:* a probing action a_p (scouting, Protect, a safe pivot) is worth taking when

  VOI(a_p) > v(b) − (value of a_p in the current game).

**Caveat:** this holds only for *exogenous* information (chance and hard evidence). When the information comes from the opponent's own choices, see L4.

**L4 (endogenous information: the real theory question).** Beliefs can be updated with two kinds of evidence:
- **(i) hard evidence:** P(obs | θ, a, β), such as damage rolls, item activations, speed order, or the fact that move m exists in the set. This depends only on chance, so it is safe.
- **(ii) behavioral evidence:** P(β | θ, history), the opponent's *choice*. This depends on the opponent's strategy, so an equilibrium opponent can manipulate it.

What can be said about each:
- Updating on (i) only gives the correct posterior when the opponent's policy doesn't depend on its type given the public history. Otherwise we are throwing information away.
- Updating on (ii) with a *wrong* behavior model can be exploited. The opponent "bluffs".

The principled fix is **safe subgame solving / resolving gadgets** (Burch, Johanson & Bowling 2014; Moravčík et al. 2017 DeepStack; Brown & Sandholm 2017):
- let the opponent choose how likely each of its types is to reach the current state, within constraints;
- or carry public belief states, which capture both sides' beliefs (ReBeL, Brown et al. 2020).

*Research question:* a small Pokémon-like 2-turn game where hard-evidence-only updating loses a measurable amount of value and naive behavioral updating is exploitable, plus a measurement of how often this matters in real battles.

**L5 (bluffing / concealment = Aumann–Maschler).** Our own team is hidden too, so our actions leak information. In the repeated zero-sum game with one-sided incomplete information, the informed player's value is **cav u(p)**, the concave envelope of the non-revealing value (Aumann–Maschler).

"Bluffing" therefore has a precise meaning: playing type-independent (non-revealing) mixtures where u is not concave, and revealing where it is.

A 2-type, 2-turn toy can show when *not* using a super-effective coverage move (to hide it) is optimal.

*Status:* clean and course-relevant. Do it only as a toy.

**L6 (exploiting opponents across battles, safely).** Treat each battle as one round of an online-learning problem with a few experts:
- E₀: the equilibrium (L1) strategy;
- E₁…E_k: best responses to opponent models (aggressive, switch-heavy, ...).

**Feedback:** win/loss for the expert actually played (bandit feedback), so use EXP3.

**Guarantee:** average payoff ≥ max_k payoff(E_k) − O(√(k log k / N)) over N battles. Since E₀ is one of the experts, this inherits the equilibrium guarantee up to that regret.

**Relation to prior work:** this is a lighter-weight version of safe opponent exploitation (Ganzfried & Sandholm 2015) and restricted Nash response (Johanson et al. 2007).

## 3. What the proposal says that should change

1. **"No-regret search approaches equilibrium" is not true in general:**
   - It is true **in the per-turn approximate game**, and only for the time-averaged strategy (zero-sum ⇒ the average converges to Nash).
   - It is not true for the full game unless the subgame-solving issues in L4 are handled.
   - Say that explicitly.
2. **"Posterior over hidden sets is sufficient memory":**
   - This is the POMDP belief-state theorem (Åström 1965; Smallwood–Sondik), and it holds against a *fixed* opponent policy.
   - Against an adapting or equilibrium opponent, the sufficient statistic is the **public belief state**: both players' beliefs (Nayyar–Mahajan–Teneketzis common-information approach; ReBeL).
   - Don't make this a proof goal; cite it, and use L2′ for the compression loss.
3. **Random Battles make the prior essentially exact:**
   - Sample from Showdown's own team generator to get the *joint* prior over sets, so prior error is about 0.
   - The likelihoods of hard evidence are exact too: damage rolls are uniform over 16 values.
   - So the TV(b, b̂) term in L2 reduces entirely to *behavioral* modeling error (L4). This is a nice, clean point to make in the writeup.
4. **"Mixed strategy is optimal" vs. "argmax":**
   - Under L1, the gain from mixing equals v(b) − max_a f_b(e_a): the **value of randomization**.
   - It can be computed for every state, so you can report how often it is nonzero and how large it is.
