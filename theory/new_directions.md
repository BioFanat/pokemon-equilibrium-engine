# New directions: which mathematics to prove next (brainstorm, 2026-11-15)

Side note from an ideation sprint. Question: which *new* results should the project prove, building on Lecture 6
(experts, MWU, deterministic algorithms fail, the hustler) and Lecture 7 (MWU self-play gives
max_x xᵀAȳ − min_y x̄ᵀAy ≤ ε_R + ε_C; CCE) and on the lit notes. Scripts and raw outputs live in
`/private/tmp/claude-501/-Users-biofanat-Documents-classes-f26-cs535/2bbe670b-2f97-4d9a-bfad-e1139e0352c6/scratchpad/newtheory/`
(scratch, not project code; files `n1*.py`, `n2*.py`, …, outputs `n*.out`). Nothing here is a reportable number.
Every status label below is one of: **proved** (proof written here), **sketch** (gap named), **numeric**
(checked by brute force or simulation only), **conjecture**, **cited** (known; who proved it).

## 0. What the inventory changed

The seed list assumed less was done than is. `theory/search_rl_links.md` already contains the telescoping identity
(L0), the exact identity expl = sup_σ E[Σ ℓ_t] with the myopic lower side (L1), *safe Hedge* over experts with the
reset-vs-persistent comparison (L2(a)–(d)), the TD-residual form (L6), the exact folk identity for visit shares (L4) and
a UCB1 lockstep proposition (L4b); `theory/hedge_experts.md` and `theory/regret_theory.md` have the whole-game Hedge
composition (E2, E2′), the anytime constant 3/2 (R2.1), the k = 2 weight trajectory (R3a) and the exploiter bracket (R3c).
So three of the five seeds are corollaries of existing notes:

- **Seed C** (ε-safe expert in Hedge) is L2(b) with the safe expert's stage gap ε_s replaced by ε_s + ε·range, plus L2(c)/E3 for the gain. Two lines. Cite, do not prove.
- **Seed D** (Lecture 7 with RM+ weighted averaging) is FG Prop 4 + the BMS19 cross-term (Lean `rmplus_improvement`). What "breaks" is Lecture 7's identity under *alternating* updates: Burch, Moravčík & Schmid 2019, Observation 1 (both regrets zero after 2T alternating updates, average-strategy exploitability 0.5). Cite.
- **Seed A** (no-regret dynamics across a stochastic game) is the "per-state averaging fails" story: Kovařík & Lisý 2019 Thm 4.1 is literally a counterexample; the fixes are certified/nested policies (Bai, Jin & Yu 2020; Jin, Liu, Wang & Yu 2021 V-learning) or per-turn re-solving to convergence with a leaf (our Theorem 1 / L3). Cite.

What the inventory does **not** contain, and where this sprint found content: lower bounds and tightness
(Section 2), the UCB1 mechanism (Section 3), and the explicit carried-vs-reset cost (Section 4).

## 1. Ranked table

| # | Candidate | Verdict | Novelty | Proof status | Numeric check | What it adds to the paper |
|---|---|---|---|---|---|---|
| 1 | **S. Sampled-worlds term, sharp rates**: fixed prior → exponential decay in K; minimax over priors → Ω(1/√K) (Props S1, S2) | **prove now** | technique folklore (Hoeffding / two-point), statements new to the project; not found in Kov19b/Sch21 by the scouts | S1 proved; S2 proved (CLT step asymptotic) | `n2.out`: fixed-prior slope −0.021 per K vs predicted −2γ² = −0.020; √K·E[loss] → 0.0236 vs predicted 0.0231 | Shows Theorem B′ loses the *rate*, not just constants, by bounding the loss through the uniform deviation S (which is Θ(1/√K)): for a fixed interior prior the loss itself is exponentially small in K; the √(1/K) rate cannot be improved uniformly over priors. Finite-type assumption (the engine's set prior is a near-continuum; see §2 verdict) |
| 2 | **U. Decoupled UCB1 at a simultaneous node**: equalisation of conditional means (Prop U1); independence ⇒ Nash (Cor U2); Ω(1) plateau with Bernoulli payoffs (Conj U3); deterministic drift away from Nash (Obs U4) | **prove U1/U2 now; U3 as conjecture with exhibit** | U1/U2: standard bandit reasoning + Kovařík–Lisý's UPO idea, but no formal UCB1 statement exists (links_attribution.md:118); U3/U4 numerics new | U1 proved under a regularity hypothesis (named); U2 proved; U3 conjecture; U4 observation | `n1.out`, `n1b.out`: 3 seeds, N = 10⁶, gap 0.037–0.076 flat from 10³; TV(J, x̄ȳᵀ) 0.04–0.09 on plateaus vs 0.004 when gap → 0 | Replaces L4b's knife-edge example by a mechanism (asymptotic correlation = linear counterfactual regret) and a diagnostic; Lecture 6/7 link: equalised realised means, linear counterfactual regret |
| 3 | **R3d. Carried vs reset Hedge, closed-form cost** under R3a's constant-drift model: C(T) = dΣσ(−d√(8 ln 2 t)) → π²/(48 ln 2·d) ≈ 0.297/d | **prove now (one paragraph)** | folklore (fixed-schedule Hedge); corollary of R3a | proved | `n3.out`: closed form = simulation (4e-16); predicted mean weights 0.43/0.47/0.34/0.16 vs logged 0.38/0.45/0.34/0.14 | Gives seed B a quantitative statement: reset pays ≈ dH/2 per game, carried pays ≤ 0.297/d in total; reproduces four logged weights from two fitted drifts |
| 4 | **Second-order (AdaHedge) regret slack** for E2 | cite + number | de Rooij et al. 2014 | cited | `n4.out`: median per game 2.0 ranges (AdaHedge form) vs 6.8 (first order); data term 2√(V ln k) ≈ 0.3 range | Honest statement that no single-game slack is non-vacuous; amortisation (row 3) is the only route |
| 5 | **C. ε-safe expert in Hedge** (seed C) | cite only | L2 + GS15 §6.4 + Li 2026 Prop 1 + Jeary–Turrini 2023 | corollary (two lines, written in §5) | — | One corollary sentence after E2 |
| 6 | **D. RM+ averaging in Lecture 7's theorem** (seed D) | cite only | Tammelin 2014; BMS19 Thm 5, Obs 1; FG Prop 4 | Lean already | — | One sentence: alternation breaks the identity, BMS19 repairs it |
| 7 | **A. No-regret dynamics across the stochastic game** (seed A) | cite only | Bai–Jin–Yu 2020; Jin et al. 2021; Kovařík–Lisý Thm 4.1; DFG 2020 | — | — | One paragraph in related work: per-state averaging fails, per-turn re-solve is the Markov fix |
| 8 | Alternating PRM+ average iterate stalls | drop (note only) | Cai23/Ngu26k list it open; numerics match their suspicion | numeric | `n5.out`: median gap 3.5e-2 at T = 10³ and 10⁴ (85% > 1e-3); simultaneous PRM+ 7e-9 | None for this paper (solver is < 2% of time) |
| 9 | LP gain curve g(ε) of the ε-safe response (concave, slope = dual price) | cite only | LP sensitivity | — | — | Explains "88% of the top move's gain at ε = 2%" in one sentence |
| 10 | Policy regret vs the white-box exploiter (decoy-then-punish) | drop | Arora et al. 2012/2018; Ngu24c | — | — | Cite |
| 11 | Non-convexity of exploitability under per-turn mixing of experts | remark only | Kuhn 1953 (mixed vs behavioural) | 2-turn example in §5 | — | Motivates why the Hedge comparator must be stage-wise safe (already caveated in hedge_experts.md:185–201) |
| 12 | √(n log K) factor between S2 and Theorem B′ | drop | open | — | — | — |
| 13 | High-probability E2 via Azuma | do when writing | routine | — | — | — |
| 14 | UPO for RM selection at depth 2 (Kov15b Problem 5.17) | drop | open, hard | — | — | — |
| 15 | Σ_t η_NL ≈ oracle gap | drop (empirical) | — | conjecture (TS:117) | needs engine runs | — |

Recommendation: take **S1/S2** and **U1/U2 (+ exhibit)** to Plan; write **R3d** as a corollary of R3a.

---

## 2. Finalist S: sampled-worlds term, sharp rates

### Setting
Finite type set Θ, |Θ| = n_Θ. The agent plays one strategy x ∈ Δ(A) (agent form); the type-θ opponent knows θ and
replies. Stage matrices M_θ with all entries in an interval of length Δ. For a belief b ∈ Δ(Θ),

  g_θ(x) = min_j (xᵀM_θ)_j,   f_b(x) = Σ_θ b_θ g_θ(x),   v(b) = max_x f_b(x).

(This is Theorem B's f_b in `full_game.md`.) The agent samples K worlds iid from b, forms the empirical belief b̂_K,
and plays any x̂ ∈ argmax_x f_{b̂_K}(x). Loss L_K = v(b) − f_b(x̂) ∈ [0, Δ].

Theorem B′(ii) (`full_game.md:118–124`) gives E[L_K] ≤ 2·Δ√(2(n log K + log 2)/K) uniformly in b (n = |A|, K ≥ 9).

### Proposition S1 (fixed prior: exponential decay) — proved
**Assumptions.** (S1.1) finitely many types; (S1.2) b lies in the interior of a linearity cell of v: there is γ > 0
such that every b′ with TV(b′, b) < γ has an optimiser set for f_{b′} contained in the optimiser set for f_b.

**Statement.** E[L_K] ≤ Δ · P(TV(b̂_K, b) ≥ γ) ≤ Δ · 2^{n_Θ} e^{−2Kγ²}. For n_Θ = 2: E[L_K] ≤ 2Δe^{−2Kγ²}.

**Proof.** v(b) = max over (x, s) in the polyhedron P = {(x, s): x ∈ Δ(A), s_θ ≤ (xᵀM_θ)_j ∀θ, j} of bᵀs, so v is the
maximum of finitely many affine functions of b (one per vertex of P): convex, piecewise affine, with finitely many
cells. Fix a cell C on whose interior v(b′) = b′ᵀs_v for one vertex (x_v, s_v). If b′ is interior to C and x′ is any
optimiser of f_{b′}, then the affine function b″ ↦ v(b″) − f_{b″}(x′) is ≥ 0 on C and vanishes at the interior point
b′, hence vanishes on all of C; in particular f_b(x′) = v(b), i.e. x′ is optimal for b. (This is the content of
(S1.2); γ is the TV-radius of a ball around b inside C.) Therefore on the event {TV(b̂_K, b) < γ} we have L_K = 0,
and L_K ≤ Δ always, so E[L_K] ≤ Δ·P(TV(b̂_K, b) ≥ γ). The multinomial L1 bound of Weissman et al. (2003),
P(‖b̂_K − b‖₁ ≥ ε) ≤ (2^{n_Θ} − 2)e^{−Kε²/2} with ε = 2γ, gives the second inequality; for two types it is
Hoeffding's P(|b̂ − b| ≥ γ) ≤ 2e^{−2Kγ²}. ∎

**Remark (boundary priors).** If b lies on a cell boundary, the same argument gives L_K = 0 whenever b̂ falls in a cell
whose closure contains b; the loss then comes only from b̂ landing in a third cell, again exponentially unlikely
unless such a cell is within O(1/√K) of b. The uniform-in-b rate is governed by priors near a boundary (S2).

### Proposition S2 (minimax over priors: Ω(1/√K)) — proved (last step by the CLT)
**Instance.** Two types, M₁ = [[1, 0], [0, 1/4]], M₂ = [[1/4, 0], [0, 1]], Δ = 1, b = P(type 1).
Then g₁(x) = min(x₁, x₂/4), g₂(x) = min(x₁/4, x₂), and

  v(b) = max(0.2 − 0.15 b, 0.05 + 0.15 b),

with the unique optimiser x_L = (0.8, 0.2) for b < 1/2 and x_R = (0.2, 0.8) for b > 1/2. *Proof of the closed form:*
write x = (t, 1−t). For t ∈ [1/5, 4/5], g₁ = (1−t)/4 and g₂ = t/4, so f_b = (b + t(1−2b))/4 is affine in t with slope
(1−2b)/4: strictly increasing for b < 1/2, strictly decreasing for b > 1/2. For t < 1/5, f_b = bt + (1−b)t/4 is
increasing; for t > 4/5, f_b = b(1−t)/4 + (1−b)(1−t) is decreasing. Hence the maximiser is t = 4/5 (b < 1/2) or
t = 1/5 (b > 1/2), unique for b ≠ 1/2, and v(b) = max(0.2 − 0.15b, 0.05 + 0.15b): one kink at b₀ = 1/2, slopes ∓0.15,
jump s = 0.3 (the LP grid in `n2.out` agrees).

**Statement.** For b_K = 1/2 + c/√K with c = 1/4,

  E[L_K] ≥ s·(b_K − 1/2)·P(b̂_K < 1/2) = (0.3·c/√K)·P(Bin(K, b_K) < K/2),

and P(Bin(K, b_K) < K/2) → Φ(−c/√(b₀(1−b₀))) = Φ(−1/2) ≈ 0.3085. Hence

  liminf_K √K · sup_b E[L_K] ≥ 0.3 · 0.25 · 0.3085 ≈ 0.0231 > 0.

**Proof.** When b > 1/2 and b̂ < 1/2 the agent plays x_L, and L = v(b) − f_b(x_L) = (0.05 + 0.15b) − (0.2 − 0.15b)
= s(b − 1/2). When b̂ > 1/2 the agent plays x_R and L = 0; the tie b̂ = 1/2 contributes ≥ 0. Take expectations and
apply the CLT to Bin(K, b_K) with mean K/2 + c√K and standard deviation √(K b_K(1−b_K)) → √K/2. ∎

**Numeric check** (`n2_sampling_lb.py`, exact binomial sums, 7 s): √K·E[L_K] = 0.0315, 0.0269, 0.0258, 0.0243,
0.0237, 0.0236 at K = 10, 50, 100, 500, 2000, 4000, converging to 0.0231; the analytic expression
s(b_K − b₀)P(b̂ ≤ b₀) matches E[L_K] to ratio 1.000 at every K (the loss is exactly s·|b − b₀| on a crossing in this
instance). The grid supremum sup_b E[L_K] is 0.0362/√K, 0.0302/√K, 0.0277/√K, 0.0260/√K at K = 10, 50, 200, 1000.
For the fixed prior b = 0.6 (γ = 0.1): E[L_K] = 1.1e-2, 8.1e-4, 3.0e-12, 3.9e-39 at K = 10, 100, 1000, 4000, with
log-slope −0.021 per unit K against the predicted Hoeffding exponent −2γ² = −0.020; Theorem B′'s bound at those K is
1.03, 0.45, 0.17, 0.09.

### Verdict
- **Novelty:** new to the project; the matching scout item is `lit/scout_search_solvers.md:126–128` (item 4: agent-form
  K-world error as a function of K via TV concentration), which asks only for the upper bound — the lower bound and the
  exponential fixed-prior rate were not asked for anywhere in the notes. The technique is textbook (parametric LP,
  Hoeffding, two-point lower bound), so present it as a tightness proposition, not a theorem.
- **What it explains, and what it does not.** The measured "0.054Δ vs bound 0.91Δ" (`theory_section.md:88`,
  `full_game.md:126`) is E[S] with S = sup_x |f_b̂(x) − f_K(x)|, on *synthetic* few-type instances
  (`theory/checks/check_theorem_b.py`: 2–3 types, 2–3 actions, K ≤ 7 and 25). S is a supremum of sample-mean
  deviations and is genuinely Θ(1/√K); S1 does **not** explain that 17×. What S1 says is that bounding the loss by 2S
  (Theorem B) loses the rate: for a fixed interior prior the loss L_K is exponentially small while S is not. Prediction
  (testable on synthetic instances and, with care, on logged decisions): the sampling *loss* concentrates on beliefs
  within ≈ 1/√K of a boundary of v. **Limitation:** S1 needs finitely many types with a 2^{n_Θ} constant; the engine's
  prior over opponent sets is a near-continuum, where the cell structure is astronomically fine and S1's constant is
  vacuous — there the honest statement is only S2's uniform rate plus "loss is concentrated near decision boundaries".
- **Gap left:** the √(n log K) factor between S2 and B′(ii) (row 12). Not attempted.
- **Effort:** half a page; Lean optional (the Hoeffding step is library; the LP-cell argument is the only real step).
- **Risk:** low. S1.2 excludes measure-zero priors; S2's last step is asymptotic (a Berry–Esseen version is routine).

---

## 3. Finalist U: decoupled UCB1 at a simultaneous-move node

### Setting
Matrix A ∈ [0,1]^{n×m}; the row player maximises, the column player's reward is 1 − A. Both run UCB1 on their own
arms with their own realised rewards: after each arm once, i_t ∈ argmax_i μ̂¹_i(t−1) + √(2 ln t / n¹_i(t−1)) (any
fixed tie-break), similarly j_t for the column. Rewards r¹_t ∈ [0,1] with E[r¹_t | i_t, j_t, past] = A_{i_t j_t}
(Bernoulli or deterministic), r²_t = 1 − r¹_t. J_N(i,j) = (1/N)#{t ≤ N : (i_t, j_t) = (i,j)}; x̄_N, ȳ_N its marginals
(the visit shares); μ̂¹_i(N) = (1/n¹_i(N))Σ_{t ≤ N: i_t = i} r¹_t is the *realised conditional mean* of row i.
Nash gap of the shares: G_N = max_i (Aȳ_N)_i − min_j (x̄_Nᵀ A)_j. Proposition L4 gives G_N = (Reg¹_N + Reg²_N)/N with
the *counterfactual* regrets Reg¹_N = N max_i (Aȳ_N)_i − Σ_t A_{i_t j_t}, Reg²_N likewise.

### Proposition U1 (equalisation of realised conditional means) — proved under (U1.2)
**Assumptions.** (U1.1) N_s → ∞ is a sequence along which J_{N_s} → J; rows i, k have J_{i·}, J_{k·} > 0.
(U1.2) *Regularity:* the last time i is played before N_s, τ_s, satisfies N_s − τ_s = o(N_s), and likewise for k.

**Statement.** lim_s (μ̂¹_i(N_s) − μ̂¹_k(N_s)) = 0. The same holds for the column player.

**Proof.** At τ_s the index rule gives μ̂¹_i(τ_s − 1) ≥ μ̂¹_k(τ_s − 1) − √(2 ln τ_s / n¹_k(τ_s − 1)); since
n¹_k(τ_s − 1) ≥ (J_{k·} − o(1))τ_s, the bonus is O(√(ln τ_s/τ_s)) → 0. Between τ_s and N_s the arm i is played once
(at τ_s), so μ̂¹_i moves by at most 1/n¹_i(τ_s) → 0; μ̂¹_k is an average of n¹_k ≥ (J_{k·} − o(1))N_s values in [0,1]
of which at most N_s − τ_s = o(N_s) are added in the gap, so it moves by at most o(N_s)/((J_{k·} − o(1))N_s) → 0.
Hence liminf (μ̂¹_i − μ̂¹_k)(N_s) ≥ 0; exchanging i and k gives the limit. ∎

**Gap named.** (U1.2) is a hypothesis. I conjecture it holds automatically for UCB1 whenever J_{i·} > 0 (an arm with
positive frequency cannot be absent for a linear stretch because its bonus then exceeds the others' by Ω(√(ln N/N))
while realised means move by O(1/N) per step — but turning that into a proof needs control of the other player, and
I did not do it). Without (U1.2) the statement holds along the subsequence of times at which i is played.

### Corollary U2 (asymptotic independence ⇒ Nash) — proved
**Assumptions.** (U1.1), (U1.2) for all rows and all columns; every row and column has positive limiting frequency;
and the limit is a product: J = x̄ȳᵀ.

**Statement.** (x̄, ȳ) is a Nash equilibrium of A, i.e. lim G_{N_s} = 0.

**Proof.** r¹_t − A_{i_t j_t} is a bounded martingale difference, so by the martingale LLN
μ̂¹_i(N_s) − Σ_j (J_{N_s}(i,j)/J_{N_s}(i,·)) A_{ij} → 0 a.s. With J = x̄ȳᵀ the second term is (Aȳ)_i. U1 makes all
(Aȳ)_i equal (full support), and symmetrically all (x̄ᵀA)_j equal; mutual indifference over full supports is the
Nash condition. ∎

**Reading (the useful direction, under U1.2).** A persistent gap G_N ≥ g > 0 with all arms played linearly forces J ≠ x̄ȳᵀ:
the two players' choices stay *correlated*. In Kovařík & Lisý's terms UCB1 then violates unbiased payoff observations
(their Def. 5.11); by L4 the counterfactual regret Reg¹_N + Reg²_N is linear in N even though each player's realised
conditional means are equalised (its bandit-style regret on its own realised rewards is small). This is Lecture 7's
convergence proof failing at exactly one step: the realised average payoff is not (1/N)Σ_t (x_tᵀAy_t) with
independent x_t, y_t — the "ȳ" the row player effectively faces differs by row. Measured diagnostic: TV(J, x̄ȳᵀ).

### Conjecture U3 (Ω(1) plateau with Bernoulli payoffs) — numeric only
For A = [[1, 0], [0, 1/2]] (Nash x* = y* = (1/3, 2/3), v = 1/3) with Bernoulli rewards, liminf_N G_N > 0 a.s.

**Evidence** (`n1b.out`, 3 seeds, N = 10⁶, 16 s): G_N at N = 10³/10⁴/10⁵/10⁶ is 0.064/0.080/0.078/0.076 (seed 0),
0.067/0.056/0.064/0.066 (seed 1), 0.039/0.034/0.035/0.037 (seed 2) — flat over three decades, seed-dependent level.
Final shares x̄ ≈ ȳ ≈ (0.38, 0.62) vs Nash (0.333, 0.667); realised conditional means equal to 2e-3 (0.2725 vs 0.2743,
seed 0: U1 confirmed) while the counterfactual means (Aȳ) = (0.383, 0.309) differ by 0.07; TV(J, x̄ȳᵀ) = 0.085, 0.073,
0.038 (correlation persists). Contrast rand2x2_2 = [[0.544, 0.935], [0.816, 0.003]] where seed 0 converges
(G = 0.004, TV = 0.004) while seeds 1–2 plateau at G ≈ 0.06 with TV ≈ 0.04: the same game has both behaviours, so the
plateau is a long-lived correlated state, not a deterministic fixed point. Across 20 random 2×2 and 10 random 3×3 games
(`n1.out`, N = 10⁵) the mixed-equilibrium games have median log-log slope −0.21 (stochastic) with final gaps
0.0013–0.077; the pure-saddle games all converge (gap ≈ N^{−1/2}).

**Why no proof here:** a lower bound on the Nash gap of a coupled stochastic recursion needs a Lyapunov argument for
the correlation; I do not have one. State as a conjecture with the exhibit, or drop.

### Observation U4 (deterministic payoffs: drift *away* from Nash) — numeric only
With deterministic rewards r¹_t = A_{i_t j_t} the gap on mixed 2×2 games *grows* with N: mp_asym 0.09 → 0.116 →
0.238 → 0.244 → 0.249 at N = 10²..10⁶, with both players locking into the pure cell (2,2) (x̄ = (0.17, 0.83) at 10⁶);
rand2x2_2: 0.05 → 0.18, cell (1,1); rand2x2_18: 0.024 → 0.065, cell (2,1). This generalises L4b's phenomenon (knife-edge
round-robin on a symmetric game) to a different limit — a pure non-equilibrium cell on which both realised means are
"explained" by the correlated history — and is an *analogy* (not an instance) of Lecture 6's "a deterministic map from
history can be made to suffer linear regret": here the other UCB1 plays the adversary's role, but it does not observe
the row player's history, so Lecture 6's adversary model does not literally apply. A theorem for generic deterministic 2×2 games is
plausible but was not attempted (L4b's induction uses the symmetry).

### Verdict
- **Novelty:** U1/U2 are a specialisation of standard arguments (UCB1 index equalisation; Kovařík–Lisý's UPO property)
  to UCB1, which nobody seems to have written down (`lit/links_attribution.md:118`: "no formal theorem located";
  Shafiei et al. 2009 is empirical). U3/U4 are new numerical facts; F6 in `lit/next_directions.md` saw the gap decaying
  ≈ N^{−0.3} on *real* matrices with the engine's payoff transform, so the plateau is matrix-dependent.
- **What it explains:** why the MCTS baseline's root shares are not equilibrium shares (and gives the diagnostic
  TV(J, x̄ȳᵀ) to measure on poke-engine roots); why F2's "best response to uniform predicts MCTS's move 62%" is
  compatible with UCB1 equalising its realised means. It does not explain the engine's regime directly (one-node model,
  noisy leaf): say so.
- **Effort:** U1/U2 half a page; the exhibit one figure. **Risk:** U1.2 must be stated as a hypothesis; U3 must be
  labelled a conjecture. Don't call the plateau matrix "Shafiei's" — their matrix was not obtained (PDF 404).

---

## 4. Finalist R3d: carried vs reset Hedge, the explicit cost (seed B)

### Statement — proved (corollary of R3a(b))
**Assumptions.** k = 2 experts {F, N}; the shipped anytime FTRL η_t = √(8 ln 2/t) on normalised rewards
(`exp3_live.py:69`); constant drift r_{N,t} − r_{F,t} = d > 0 at every turn (R3a's model, with R3b's fitted
d₁ = 0.0133 vs MCTS, d₂ = 0.0291 vs the exploiter). Then w_F(t) = σ(−a√t), a = d√(8 ln 2) (R3a(b)), and the loss
relative to always playing N after T turns is C(T) = d Σ_{t ≤ T} σ(−a√t).

(i) C(T) ≤ C_∞ := d∫₀^∞ σ(−a√t) dt = (2d/a²)∫₀^∞ u σ(−u) du = (2d/a²)(π²/12) = π²/(48 ln 2 · d) ≈ 0.2966/d,
and C(T) = dT/2 − O(d a T^{3/2}) for T ≪ 1/a² = 1/(8 ln 2 d²).
(ii) Reset each game: cost G·C(H) over G games of H turns (≈ G d H/2 when H ≪ 1/a²). Carried: total cost C(GH) ≤ C_∞,
independent of G. The advantage of carrying is G·C(H) − C(GH).

**Proof.** σ(−a√t) is decreasing, so the sum is bounded by the integral; substitute u = a√t, dt = 2u du/a²;
∫₀^∞ u/(1 + e^u) du = η(2)Γ(2) = π²/12 (Fermi–Dirac integral). The small-T expansion is σ(−a√t) = 1/2 − a√t/4 + O(a³t^{3/2}). ∎

### Numeric check (`n3.out`)
Closed form vs the shipped update: max |diff| 4e-16. With H = 35 (median game length 32–39, RESULTS.md:501) and
R3b's drifts, the predicted mean F-weights are: reset vs exploiter **0.43** (logged 0.38), reset vs MCTS **0.47**
(0.45), carried games 1–30 vs MCTS **0.34** (0.34), carried games 31–60 vs exploiter **0.16** (0.14); predicted
carried weight 0.27 at the start of game 31 and 0.09 at the end of game 60 (RESULTS.md:512 text: 0.28 → 0.07).
Exploitation-phase cost over 30 games: reset 13.2 vs carried 4.9 normalised units (the single-drift bound C_∞ = 10.2
still applies to the two-phase carried run because the phase-1 drift d₁ > 0 only lowers w_F at every later turn); for d₂ the
saturation scale 1/a² ≈ 215 turns, so a 35-turn game is in the linear regime (C(35) = 0.44 ≈ 0.5·d₂·35 = 0.51).

### Verdict
Cite only, plus this corollary: the mechanism is textbook (a decaying-η learner cannot forget, so carrying amortises
the warm-up; Harris et al. 2022 "Meta-Learning in Games", ICLR 2023, is the general principle; Herbster & Warmuth 1998
fixed share is the tracking side; L2(d) is the bound comparison). What is new to the project is the closed form and
that two fitted drifts reproduce four logged weights within 0.05. It does **not** convert to the +0.146 win-rate
difference (normalised heuristic units ≠ win probability); do not claim that.

---

## 5. Cite-only items, written out

**C (ε-safe expert).** The ε-safe LP (`nash.py:244–259`) returns x with (xᵀÂ_t)_j ≥ v̂_t − ε·range_t for every column j,
so its stage gap is ≤ ε_s + ε·range_t. Plugging it into Theorem L2(b) in place of the ε_s-optimal expert:
Σ_g (V*₀ − u_g) ≤ Σ_t E[ε_s + ε·range_t] + E[R(T)] + 2ε_v E[T]; L2(c)/E3 gives the gain side (comparator = the exploit
expert at the realised states against the realised moves). Prior art: Ganzfried & Sandholm 2015 §6.4 (no-regret choice
between equilibrium and exploitation, "safe in the limit"), Li 2026 (arXiv 2607.28520, runtime certificate B_t ≤ ε_max,
Prop. 1; EFG, no regret bound, no stochastic games; "sound approximate best responses" is their open item), Jeary &
Turrini 2023 (arXiv 2307.12338, prime-safety, repeated play). Our L2 is the stochastic-game statement with leaf error;
the ε-expert adds nothing but a constant. The gain curve g_t(ε) = max{xᵀÂq̂ : (xᵀÂ)_j ≥ v̂ − ε·range} is concave
piecewise-linear in ε with slope equal to the LP dual price of the safety constraints — LP sensitivity analysis — which
is why 2% of range buys 88% of the top move's gain (F3). *Remark (non-convexity):* per-turn independent mixing of two
experts with zero game exploitability can be exploitable: in a 2-turn game where the turn-2 action must match the
turn-1 action (L↔l, R↔r, payoff +1 if matched, −1 otherwise), experts (L,l) and (R,r) each have exploitability 0 while
a 50/50 per-turn mixture has 1. So "expl(Hedge) ≤ min_e expl(e) + Reg" is false; the Hedge guarantee needs the
*stage-wise* safe expert (hedge_experts.md:185–201 already caveats the comparator).

**D (RM+ averaging).** Lecture 7's proof uses simultaneous updates; FG Prop 4 is the linearly-averaged version
(gap ≤ 2Δ(√n + √m)/√T). For alternating updates the folk identity fails (Burch, Moravčík & Schmid 2019, Observation 1)
and Theorem 5's improvement term restores the bound (Lean `rmplus_improvement`; `theory_section.md` Prop 2). Weighted
averaging: Tammelin 2014 (CFR+), Brown & Sandholm 2019 (DCFR polynomial weights), Farina, Kroer & Sandholm 2021 (PRM+,
quadratic weights). Nothing to prove.

**A (dynamics across the stochastic game).** Independent per-state no-regret learners in a zero-sum Markov game do not
yield a Nash *Markov* policy by averaging per-state iterates: Kovařík & Lisý 2019 Thm 4.1 (an HC selector whose averages
sit outside the 1/5-NE set), the certified/nested-policy constructions of Bai, Jin & Yu 2020 (Nash V-learning) and Jin,
Liu, Wang & Yu 2021 (V-learning), and Daskalakis, Foster & Golowich 2020 for independent policy gradient. Our engine
sidesteps the averaging problem by solving each stage to convergence with a leaf (Theorem 1 / L3); that is the right
one-paragraph framing. What "breaks" when moving Lecture 7 past one matrix is precisely the averaging step.

**Second-order slack (`n4.out`).** On the 17 non-capped logged games (601 two-sided decisions, 4 experts: top, sampled,
safest, ε-safe 2% with the BR-to-uniform predictor, rewards against the recorded MCTS move; the "sampled" and "top"
experts use the exact LP solution here where the engine uses RM+ at 1000 iterations): median per-game T = 26,
mean expert spread 2.95% of the matrix range, √V = 0.13 range with V = Σ_t spread_t²/4. First-order anytime bound
(R2.1, (3/2)√(T ln k/2)) = 6.8 ranges per game; an AdaHedge-type bound 2√(V ln k) + δ((16/3)ln k + 2) (de Rooij, van
Erven, Grünwald & Koolen 2014, Thm 8 — constants quoted from memory, check) = 2.0 ranges per game, of which the
data-dependent term is ≈ 0.3 and the additive δ-term ≈ 1.7. **Conclusion:** no known single-game bound is non-vacuous
(a game's payoff range is 1 range by definition); the data term says 0.3 range is the right order; only amortisation
across games (Section 4; E2′'s N) makes the slack small. Cite, don't prove.

**Alternating PRM+ (`n5.out`, `n5b.out`).** Independent re-implementation: on 40 random 3×3 games with quadratic
averaging, alternating PRM+ (prediction = last instantaneous regret) has median average-iterate gap 3.5e-2 at
T = 10³ and 10⁴ (85% > 1e-3); a "fresh prediction" variant (x predicts against the just-updated y) reaches 3e-3 at
T = 10⁴ (slower than plain RM+); simultaneous PRM+ 7e-9 at 10⁴. This replicates F5's stall and is consistent with
Cai et al. 2023 and Nguyen 2026 listing alternating PRM+ as having no known guarantee. A stall in one implementation is
not a counterexample (the averaging weights and the prediction choice both matter); off-theme for this paper. Note only.

---

## 6. Refuted / superseded

- **"Seed C is new"** — superseded: it is L2(b)+(c) with ε_s → ε_s + ε·range (two lines).
- **"Seed E: Ω(1) lower bound for UCB1 shares with stochastic payoffs is within reach as a theorem"** — downgraded:
  numerically true on specific games (U3, three seeds, 10⁶ pulls) and false on others (pure-saddle games converge; some
  mixed games converge for some seeds), so the correct statement is a conjecture about a long-lived correlated state;
  the provable part is the mechanism (U1/U2). F6's "gap ≈ N^{−0.3}" on real matrices is not contradicted: the plateau
  is matrix- and seed-dependent.
- **"Carried weights beat reset by a tracking/fixed-share argument"** — superseded: the measured effect is the
  warm-up amortisation of a decaying-η learner (R3d), not tracking; X6 already showed tracking is fast (0–3 battles).
- **"Prop 3's √(log K/K) is the rate the agent pays"** — refuted for fixed priors: the loss is exponentially small in K
  (S1); √(1/K) is only the uniform-in-prior rate (S2).
- **"expl(Hedge) ≤ min_e expl(e) + Reg"** — false (non-convexity example in §5); the stage-wise comparator is required.
- **"A per-turn normaliser makes the raw-unit Hedge bound sublinear"** — already refuted in hedge_experts.md:85–104
  (deficit S/3); the second-order route (§5) does not rescue a single game either.
- **My own first candidates** (observable whole-game certificate via telescoping; pure-policy exploitability sandwich;
  TD-residual leaf term) — superseded: they are L0/L2, L1(b) and L6 respectively.
- **Name fixes for the lit notes:** arXiv 2209.14110 is Harris, Anagnostides, Farina, Khodak, Wu & Sandholm, "Meta-Learning
  in Games", ICLR 2023 (not "Harvey"); arXiv 2307.12338 is Jeary & Turrini, "Safe Opponent Exploitation for Epsilon
  Equilibrium Strategies" (not "Jeanblanc"); arXiv 2607.28520's title is "Agents That Certify Their Own Exploits:
  Confidence-Scheduled Restricted Responses for Safe Opponent Exploitation" (author list not re-verified here).

## 7. Budget and provenance

CPU used: ≈ 1.5 min single-process (n1 6 s, n1b 16 s, n2 7 s, n3 < 1 s, n4 1 s, n5 14 s, n5b 12 s), numpy/scipy only,
`engine/.venv/bin/python`. No project code or `report/*.tex` touched; `results/experiments` not read beyond the mapper's
note that the SX run is in progress (its partial numbers are not used here).
