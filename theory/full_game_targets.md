# Full-game convergence: targets for the prove workflow (draft 2026-11-14)

Context: CS535 project, Gen 9 Pokémon Random Battles. Code in ../engine (nash.py, hidden_agents.py).
Per-turn results so far: RM+ Nash gap on real stage matrices ≈ T^-1.3 (median slope −1.26 at seed 0; −1.24 to −1.39 across seeds 1–3), worst-case theory
O(1/√T); agent-form vs PIMC; per-turn decomposition L2 in ../notes/math_framing.md.

## Model

G = finite-horizon two-player zero-sum stochastic game with simultaneous moves and chance.
- States s ∈ S (finite), terminal payoff u ∈ [-1, 1] (win 1, draw 0, loss -1), horizon cap H
  (300 turns in our engine; reaching the cap = draw, payoff 0). Stage t = 0..H-1.
- At a non-terminal state s, players pick a ∈ A(s), β ∈ B(s) simultaneously; next state
  s' ~ P(·|s, a, β) (chance: damage rolls, accuracy, crits).
- Q*_t(s, a, β) = E[V*_{t+1}(s')], V*_t(s) = val(Q*_t(s)) where val = matrix-game value
  (max over mixed x, min over mixed y). Shapley: V*_0(s0) is the game value; playing a stage
  equilibrium of Q*_t at every (t, s) is a Nash equilibrium.

Agent π (full information): at each (t, s) it builds an estimated stage matrix Q̂_t(s) (depth-d
lookahead + heuristic leaf) with ‖Q̂_t(s) − Q*_t(s)‖_∞ ≤ ε_v (sup over cells), and plays a mixed
x_t(s) that is ε_s-optimal in Q̂_t(s): min_y x^T Q̂ y ≥ val(Q̂) − ε_s.

Exploitability of π: expl(π) = V*_0(s0) − min_{σ} E_{π,σ}[u], σ ranging over all (history-dependent)
opponent strategies.

## Theorem A (full information)

expl(π) ≤ H · (ε_s + 2 ε_v).

Refinements wanted (state precisely, prove if true):
- A1: per-stage errors ε_s(t, s), ε_v(t, s) can vary; bound by sup over reachable states summed over
  stages: expl ≤ Σ_{t<H} sup_s (ε_s(t,s) + 2ε_v(t,s)).
- A2: can H be replaced by the expected number of turns under the worst-case opponent? (The adversary
  can prolong the game, so a careful statement is needed; give the correct version or a counterexample.)
- Key one-step lemma: if ‖A − B‖_∞ ≤ ε and x is ε_s-optimal in B, then min_y x^T A y ≥ val(A) − ε_s − 2ε.
  (Lean-checkable: finite action sets, mixed strategies as probability vectors.)
- Note: if Q̂ is built as one exact chance-averaged step plus a leaf V̂ with ‖V̂ − V*_{t+1}‖_∞ ≤ ε_leaf,
  then ε_v ≤ ε_leaf (averaging is non-expansive); deeper exact lookahead with exact val at inner
  nodes keeps ε_v ≤ ε_leaf (val is 1-Lipschitz in sup norm). No contraction without discounting.

## Corollary A' (rate in solver iterations)

With RM+ (alternating, linear averaging) for T iterations on an n×m stage matrix with payoff range R,
ε_s ≤ c · R · sqrt(max(n, m)) / sqrt(T) (cite the RM+/CFR+ regret bound: Tammelin 2015; Burch,
Moravčík, Schmid 2019). Hence expl(π) ≤ H (c R √n / √T + 2 ε_v). Empirical exponent on our stage
matrices: ε_s ≈ T^-1.3 (median slope −1.26 at seed 0; −1.24 to −1.39 across seeds 1–3).

## Theorem B (hidden information, one stage, sampled worlds)

Bayesian stage game f_b(x) = Σ_θ b(θ) min_β (M_θ^T x)_β, payoffs in [-R, R]; agent solves the
K-sample version f_K with θ_1..θ_K iid ~ b̂ and estimated matrices M̂ (‖M̂_θ − M_θ‖_∞ ≤ ε_v), getting
x̂ with ε_s-optimality in f_K. Then

  v(b) − f_b(x̂) ≤ ε_s + 2 R · TV(b, b̂) + 2 ε_v + 2 sup_{x∈Δ(A)} |f_b̂(x) − f_K(x)|,

and with probability ≥ 1 − δ, sup_x |f_b̂ − f_K| ≤ C R sqrt((|A| log K + log(1/δ)) / K)
(each θ ↦ min_β (M_θ^T x)_β is R-bounded and Lipschitz in x; covering argument over the simplex).
The first three terms were numerically checked earlier (L2, 2,000 random instances, max ratio 0.48).

## Theorem C (hidden information, full game; conditional, from literature)

State precisely what is known: depth-limited solving with values over public belief states
(Burch et al. 2014 CFR-D; Moravčík et al. 2017 DeepStack; Brown et al. 2020 ReBeL; Kovařík et al.
2023 value functions for depth-limited solving) gives expl ≤ (solver regret term) + (number of
public stages) × (PBS value error). Explain why a per-world state value V̂(s, θ) (what our agent uses)
is not a PBS value, so a non-locality / strategy-fusion bias term remains that does not vanish with
T or K. Give the bound in the form

  expl(π) ≤ H · (a · T^{-1/2} + b · sqrt(|A| log K / K) + 2 ε_v + η_NL),

with η_NL defined precisely (the gap between per-world state values and PBS values), and say which
parts are proved (A, B) and which are cited assumptions.

## Fit form for later (Modal)

expl(T, K, leaf) ≈ H_eff · (a T^-α + b K^-1/2 + c ε_v) + d, fitted against RL best-response
exploitability over a grid of T (RM+ iterations), K (worlds), leaf quality (heuristic vs value net,
depth 1/2).
