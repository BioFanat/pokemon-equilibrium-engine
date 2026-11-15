# Dynamics and single-algorithm results for the REAL game (brainstorm, 2026-11-15)

Side note from an ideation sprint. Question: which rigorous **dynamics** results (Lecture 7: no-regret self-play → minimax,
time-average vs last iterate, CCE, full vs bandit feedback) and **single-algorithm** results (Lecture 6: experts, Hedge,
adaptive opponents) can we prove *and* test in the real Pokémon game — with the priority set by the coordinator mid-sprint:
**the objective is a simple algorithm that actually plays end to end on our local Showdown server; toy-matrix theorems
count only when they support such an algorithm.**

Scripts and raw outputs: `/private/tmp/claude-501/-Users-biofanat-Documents-classes-f26-cs535/2bbe670b-2f97-4d9a-bfad-e1139e0352c6/scratchpad/dynamics/`
(`p1_policy_vs_external.py` → `p1.out`, `p2_switch_time.py` → `p2.out`, `b1_metagame.py` → `b1.out`,
`b3_bandit_vs_full.py` → `b3.out`, `a2_dual_price.py` → `a2.out`, `t_depth_composition.py` → `t.out`; total CPU ≈ 1 min).
Nothing here is a reportable number. Status labels: **proved** / **sketch** (gap named) / **numeric** / **conjecture** /
**cited**. Existing notation: `theory/search_rl_links.md` §0 (L0–L6), `theory/hedge_experts.md` (E1–E3, Lemma H),
`theory/new_directions.md` (S, U, R3d — being verified elsewhere; built on, not redone).

**What the adapter gives us** (`~/sandbox/ourbot/run_ourbot.py`, read 2026-11-15): one function `our_find_best_move(battle)`
called every turn with foul-play's battle object; it samples `K` worlds (`battle.mode.prepare_battles`), converts each to a
poke-engine `State`, builds `nash.full_matrix(state)` per world (13×13 at full option sets, 0.5–0.75 s per turn at K = 4
on the logged decisions), aligns rows/columns (`hidden_agents.align`) and solves one agent-form LP. The process serves many
games (`--run-count 100`), so **module-level learner state persists across real Showdown games**. The opponent's realised
move is available after each turn (`battle.opponent.last_used_move`, set in `fp/battle/protocol.py:396`; also the
`|move|`/`|switch|` protocol lines teed by `recording_receive`), so **full-information per-turn feedback for a Hedge layer
is available on Showdown**. Game outcome: `|win|` line (`fp/run_battle.py:33–43`). Coordinator's cost basis: 1–3 min per
local Showdown game, several in parallel via `run_fp_challenger.py`-style scripts.

---

## 0. Playable algorithms on Showdown

Each is a drop-in replacement of `our_find_best_move`. Per turn all of them compute the K world matrices exactly as the
adapter does now and the world-weighted, column-aligned matrix $\bar A_t=\sum_w p_w\,\mathrm{align}(\hat A_w)$ (or keep the
agent-form LP as the "sampled" expert). Costs below use 2 min/game and 4 parallel games unless stated.

### PA1 (single algorithm). `HedgeBot`: carried Hedge over stage experts, full-information per-turn feedback
- **Move choice.** Experts at turn $t$: `top` ($\arg\max$ of the agent-form LP $x$), `sampled` ($x$), `safest` (pure maximin
  row of $\bar A_t$), `safex(ε)` (`nash.safex_strategy` with ε = 0.3 % of range, §2.3). Play $p_t=\sum_e w_e q_{e,t}$, sample.
  After the opponent's move $b_t$ is revealed, credit $r_{e,t}=\mathrm{clip}((q_{e,t}^\top\bar A_te_{b_t}-\min\bar A_t)/\bar\Delta,0,1)$
  (exactly `exp3_live.Learner.observe`, `exp3_live.py:60–63`, with the world-weighted matrix) and update. Columns of worlds
  that do not contain $b_t$ are dropped before averaging (posterior zero); columns are aligned by canonical option name across the
  worlds that contain $b_t$. If **no** sampled world contains $b_t$ (e.g. a switch to an unguessed species) the turn gets no update and is
  **skipped**; unlike forced turns (L2 proof step 5, $\ell\le0$) skipped turns are *not* covered by L2's accounting, so the skipped-turn
  fraction is logged and the regret check below is stated over updated turns only. Weights live in the module and persist across games.
  Parallelism: four processes = four independent 75-game carried streams (state it as such); a single shared learner with delay is PA4's design.
- **Two variants** (the dynamics question inside the single algorithm): (i) *anytime FTRL* $\eta_t=\sqrt{8\ln k/t}$ on cumulative
  rewards (the shipped rule, `exp3_live.py:69`; never forgets; R3d amortisation); (ii) *Hedge + fixed share* ($\eta=0.2$,
  $\alpha=0.002$, the X6 constants).
- **Theorems it satisfies** (real setting, assumptions numbered in §2.1–2.2): Theorem L2(a)–(d) with $\hat A_i:=\bar A_t$
  (the hidden-information error enters $\varepsilon_v$; Theorem B′ of `full_game.md` bounds it); **Proposition P1** (what the
  logged external regret does and does not say about the win-rate gap to the best expert); **Proposition P2** (switching
  time of variant (i) is linear in the history, of variant (ii) history-free).
- **Measurable predictions on Showdown, vs stock foul-play (100 ms).** Let $W_e$ be each expert's own win rate and $W_H$
  HedgeBot's. (1) Logged per-turn external regret $\sum_t(r_{e,t}-w_t^\top r_t)\le R(T)$ for every $e$, with $R(T)=2(1.5\sqrt{T\ln k/2}+\sqrt{\ln k/8})$
  in normalised units (L2(H); pathwise, so it is checked on *one* run). (2) **Policy regret** $\max_eW_e-W_H$ is *not*
  controlled by (1): P1 says it equals the external term plus a continuation-mismatch term that can be $\Omega(1)$ with
  zero external regret, but is negligible on generic random games (§2.1 numeric). Prediction: $W_H\in[\min_eW_e-0.05,\max_eW_e+0.05]$
  and the realised external regret per expert is $\le 0$ for the leading expert. Engine priors agree: H1 `hedge2` vs MCTS-2000 0.560
  [0.473, 0.647] (n = 100) and `hedge_stream` persistent 0.533, both inside [sampled 0.44, top 0.58]. *Refuted if* $W_H<\min_eW_e-0.05$ with
  non-overlapping 95 % CIs (that would contradict L2(a) at the leaf-error level we assume) — or if the external regret bound (1)
  fails on the log (that would mean the reward normalisation leaves $[0,1]$, L2's refuted first draft).
  (2′) Skipped-turn fraction (no world contains $b_t$) — if above ~10 % the feedback model is semi-bandit and L2 needs the Exp3 variant. (3) Variant (ii) tracks an opponent switch (foul-play 100 ms → foul-play 2000 it, or → our `safex` bot) within
  $\approx150$ turns ≈ 4–5 battles; variant (i) needs $T_1d_1/d_2$ turns (P2): after 40 battles of phase 1 and the fitted
  drifts, ≈ 18 battles. *Refuted if* the crossing battle of variant (i) is within ±3 battles of variant (ii)'s.
- **n and cost.** Arms: HedgeBot(i) 300 games, HedgeBot(ii) 300, `top` 150, `sampled` 150, `safest` 150 (the last three give
  $W_e$; `safex` can be read from PA2). 1 050 games × 2 min / 4 parallel ≈ **9 h wall, ≈ 35 CPU-h** (local, $0). Switch test:
  2 × 80 games ≈ 1.5 h wall. Game-level CI at n = 300 is ±0.056, so (2) is a direction test, not an estimate; the sharp,
  cheap check is (1), which needs one run.

### PA2 (single algorithm). `SafexBot(ε)`: ε-safe exploitation with a non-vacuous certificate
- **Move choice.** `nash.safex_strategy(\bar A_t, ε)` (`nash.py:244–259`): maximise $x^\top\bar A_t\hat q$ against the predictor
  $\hat q$ = column best response to the uniform row (or foul-play's own most-visited move from a 100 ms search over our worlds,
  which is playable because it uses only our sampled worlds), subject to $(x^\top\bar A_t)_j\ge\hat v_t-\varepsilon\cdot\mathrm{range}_t\ \forall j$.
  **Scale convention:** run the LP on the tanh-squashed/clipped matrix in $[-1,1]$ (`nash.squash_matrix`), as L2 requires; ε is then a
  fraction of the *payoff* range (≤ 2) and the certificate arithmetic below holds. The shipped `safex_strategy` uses `ptp` of the raw
  heuristic matrix, for which the per-turn add-on in payoff units is not bounded by 2ε.
- **Theorem.** L2(b)/Theorem A with $\varepsilon_s\to\varepsilon_s+\varepsilon\cdot\mathrm{range}$ (cite only; written in
  `new_directions.md` §5). What is new is the **certificate arithmetic** (§2.3): the add-on is $\varepsilon\bar H$ in win-probability
  units ($\bar H\approx35$ non-forced turns), i.e. 0.105 at ε = 0.3 %, 0.35 at 1 %, **0.70 (vacuous) at the shipped 2 %**; and the
  **LP dual-price curve** $g(\varepsilon)$ on 300 real logged matrices: concave in all 300, recovering a median 35 % / 80 % / 100 %
  of the top move's gain at ε = 0.3 % / 1 % / 2 %.
- **Prediction.** $W_{\rm safex(ε)}\approx W_{\rm sampled}+f(\varepsilon)\,(W_{\rm top}-W_{\rm sampled})$ with $f=0.35,0.8,1.0$;
  with the engine numbers (SX: 0.510 / 0.580 vs MCTS-2000) that is 0.535 / 0.566 / 0.58. Against an exploiter (`lbr`-style bot
  we run ourselves on Showdown), the loss vs `sampled` is $\le\varepsilon\bar H$ = 0.105 at 0.3 %. *Refuted if* safex(1 %) ≤ sampled
  or safex(0.3 %) loses > 0.105 more than sampled to the exploiter (95 % CI).
- **n and cost.** 3 ε-arms × 200 games vs foul-play + 2 arms × 150 vs our exploiter ≈ 900 games ≈ **7.5 h wall**. The effect
  (0.03–0.06) is below the n = 200 CI (±0.07); **pre-screen in the engine** (5 s/game, `experiments.py`, SX-style, 1 000 games/arm
  ≈ 1.4 CPU-h) and use Showdown only to confirm the sign.

### PA3 (dynamics). `BanditRootBot`: bandit vs full feedback at the per-turn decision, everything else fixed
- **Move choice.** Same $\bar A_t$; instead of full-information RM+ (1 000 iterations, both sides see the whole matrix), run a
  **bandit self-play** at the root for $N$ = 2 000 iterations: each iteration samples one cell $(i,j)$ from the two players'
  current mixtures and reveals only $\mathrm{Bernoulli}(\bar A_t[i,j])$ (MCTS's feedback model, L4 Consequence 1); play the realised
  row shares $\bar x_N$ (variant UCB1: poke-engine's decoupled rule, variant Exp3: Hannan-consistent). Cost per turn ≈ 2 000 cell
  samples, cheaper than RM+.
- **Theorem.** L4's identity NashGap$(\bar x_N,\bar y_N)=(\mathrm{Reg}^1_N+\mathrm{Reg}^2_N)/N$ with the *realised counterfactual*
  regrets, plus Exp3.P / Exp3-IX high-probability bounds (Auer et al. 2002; Neu 2015): $\mathrm{Gap}\le c\sqrt{k\ln(k/\delta)/N}$
  w.p. $1-\delta$; for UCB1 no bound (U). **Corollary B3** (§2.4): the bound is 0.56 range at $k=9,N=2000$ (vacuous); the
  measured median gap on 120 real logged matrices is **0.154 for Exp3-2000 (realised shares; 0.150 for its mixed average), 0.120 for UCB1-2000, 0.0000 for RM+-1000**
  (82 % / 81 % / 0 % above 0.05 range). L1(b) turns stage gaps into a full-game *lower* bound along the myopic exploiter.
- **Prediction.** Against an exploiter (`lbr:sample`-style bot on Showdown, or our `safex(∞)` = pure best response to the
  predictor), BanditRootBot loses ≥ 0.05 more than the RM+ bot (the per-turn gap is 0.15 range on 80 % of turns; E6q saw sampled
  0.475 vs LBR). Against stock foul-play the difference may be small. *Refuted if* paired difference vs the exploiter is within
  ±0.03 at n = 300 — then feedback model does not matter at MCTS budgets in the real game, which would be a result in itself.
- **n and cost.** 2 solvers × {foul-play, exploiter} × 200 games = 800 games ≈ **6.7 h wall**. Engine pre-screen: 1 CPU-h.

### PA4 (dynamics). Two carried-Hedge bots in self-play across Showdown games (Lecture 7 on the meta-game)
- **Move choice.** Both seats run PA1 (possibly with different expert sets). Two feedback granularities: (α) *game level*: draw
  one expert per game, Exp3 on the game outcome (bandit; the meta-game $M$ of expert-vs-expert win rates); (β) *turn level*: PA1
  as is on both sides. Parallel games share one learner through a file lock → delayed feedback.
- **Theorem (α).** Lecture 7 in expectation with Exp3 pseudo-regret (Auer et al.): $\mathrm{Gap}_M(\mathbb E\bar x_G,\mathbb E\bar y_G)\le4\sqrt{K\ln K/G}$;
  Bernoulli noise adds an Azuma term of the same order; feedback delay $d$ multiplies regret by $\approx\sqrt{d+1}$
  (Weinberger & Ordentlich 2002; Joulani, György & Szepesvári 2013). All cite-only (§2.5). **Numeric finding B1:** with a
  realistic $M$ (entries 0.5 ± 0.1, the size of the engine's expert differences) **uniform play is already a 0.03-Nash of $M$**,
  the Exp3 bound at $B$ = 300 games is 0.54, and the realised gap is 0.043 / 0.042 / 0.024 at $B$ = 300 / 3 000 / 30 000 games
  (20 seeds); the realised average win rate is within 0.004 of the value at every budget. **Strategy convergence on the real
  meta-game is unobservable at any feasible $G$; value convergence is immediate because the game is flat.** Noisy full
  information at equal game budget is no better (0.064 / 0.055 / 0.024). Delay 4 costs nothing measurable.
- **Theorem (β).** L2(a) on both sides: each side's realised payoff is sandwiched by its own best-expert trajectory sums ± regret
  ± leaf. No theorem links the turn-level weights to the minimax of $M$ (P1 is the reason: the per-turn comparator is a
  one-step deviation, not a whole-game expert). In symmetric self-play the value statement is 0.5 by symmetry — vacuous.
- **What is worth running.** Not (α). Run (β) **asymmetrically**: seat A = PA1 with all four experts, seat B = `sampled` only
  (the agent-form equilibrium). Lecture 7's minimax content becomes the prediction **"a no-regret learner cannot beat the
  equilibrium expert by more than the equilibrium's per-turn leaf-level exploitability"**: $W_A\in[0.5-0.05,\ 0.5+0.05]$ after
  400 games (CI ±0.05). *Refuted if* $W_A>0.56$ — that would say the agent-form "equilibrium" is exploitable by *stage-wise*
  deviations, i.e. the leaf error $\varepsilon_v$ is larger than assumed (Theorem A's $2\varepsilon_v$ term is the culprit, a
  useful measurement). Also log the joint expert draws $(e_t,f_t)$: Lemma D (§2.6) predicts TV$(J,\frac1T\sum x_ty_t^\top)\lesssim km\sqrt{\ln(2km/\delta)/2T}$
  (independent sampling), the diagnostic that separates U's UCB1 correlation from sampling noise.
- **Cost.** 400 games ≈ **3.3 h wall** (both seats are ours, so no foul-play process is needed).

**One line on each named direction.** (a) Meta-game: Lecture 7 + Exp3 + Azuma, cite; rate unchanged by sampling noise; in the
real game the matrix is flat and convergence is unobservable (B1). (b) Within-game dynamics: the "right positive result" for
finite-horizon re-solving is our Theorem A / L1 (stage-wise solve-to-convergence); for *simultaneous* per-state learners it is
weighted averaging (Bai–Jin–Yu 2020 Nash V-learning, $\alpha_t=(H+1)/(H+t)$), whose mechanism we exhibit in T (§2.7). No new
theorem manufactured. (c) Search dynamics: L4 identity composes through depth with a factor 2 per level under uniform
averaging (Proposition T, sketch + numeric), consistent with Kovařík–Lisý's $C'=12(2^D-1)-8D$ and with their open Problem 5.17;
last iterate: cite Daskalakis–Panageas 2019, Wei–Lee–Zhang–Luo 2021, Cai–Zheng 2022; on real matrices OMWU's last iterate
is bimodal (median 0.009, p90 0.086) — most real stage matrices have pure saddles (§2.4). (d) Bandit vs full: PA3/B3 — bound
0.56 vs 0.13 range at MCTS budgets, measured 0.15 vs 0.0000. (e) CCE: in a zero-sum game the CCE deviation for the row player is
exactly L4's $\mathrm{Reg}^1_N/N$ and CCE marginals are maximin (standard), so CCE adds nothing beyond Nash except as the
correlation diagnostic — which is U's; Lemma D gives the sampling-noise scale for it. Single-algorithm bullets: Hedge across
games vs adaptive opponent → P1 (what is provable for carried weights is L2 + R3d + P2; policy regret is *not* provable without
the continuation term); ε-safe → cite + certificate arithmetic (PA2); tracking → Herbster–Warmuth cite + P2.

---

## 1. Ranked tables

### Dynamics line (Lecture 7)

| # | Candidate | Verdict | Novelty | Proof status | Numeric check | Real-game experiment | Cost |
|---|---|---|---|---|---|---|---|
| D1 | **PA3 / B3. Bandit vs full feedback at the per-turn decision** (BanditRootBot vs RM+ bot) | **play + measure** | theory cite (Auer 02, Neu 15, L4); real-matrix numbers new | corollary (L4 + Exp3.P), written §2.4 | `b3.out`: median gap Exp3-2000 (realised shares) 0.154, UCB1-2000 0.120, RM+ 0.0000 on 120 real matrices; bound 0.56 vs 0.13 | 2 solvers × {foul-play, exploiter} × 200 games; refute if paired Δ vs exploiter within ±0.03 | 6.7 h wall local; engine pre-screen 1 CPU-h |
| D2 | **PA4(β) asymmetric self-play**: PA1 vs equilibrium expert — "no-regret learner cannot beat the equilibrium by more than the leaf-level stage exploitability" | **play** (the only Lecture-7 minimax statement observable on Showdown) | L2(a)+(b) both sides; trivial corollary, honest framing | corollary | — (symmetric case is 0.5 by symmetry) | 400 games A vs B; refute if $W_A>0.56$ | 3.3 h wall |
| D3 | **PA4(α) meta-game Exp3 vs Exp3 across games** (Lecture 7 on $M$) | **do not run for convergence**; cite | Lecture 7 + Exp3 + Azuma + delay: all known | cited | `b1.out`: uniform is 0.03-Nash of a realistic $M$; realised gap 0.043 at 300 games, 0.024 at 30 000; bound 0.54 at 300 | — | — |
| D4 | **T. Depth composition under drifting child values**: uniform averaging compounds ×2 per level; BJY weights do not | theory only (supports "why MCTS visit shares are worse at depth"); demoted | Lis13/KL: no finite-N rate exists ("a practically useful bound… is missing"); KL Thm 5.12 $C'\sim2^D$; BJY weights known | **sketch** (uniform recursion proved; weighted = cite) | `t.out`: root err·√N 0.02/0.07/0.16/0.46 for D = 1..4 (uniform) vs 0.01–0.09 (weighted) | none playable (poke-engine is Rust, not edited) | 11 s |
| D5 | Last iterate vs average (OMWU) at per-turn matrices | cite + diagnostic | DP19, WLZL21, CZ22 | cited | `b3.out`: OMWU last median 0.0085, p90 0.086 (17 % > 0.05); Hedge avg 0.032 | none (solver < 2 % of time) | — |
| D6 | CCE of the averaged joint play | cite only | zero-sum CCE marginals are maximin (folklore); deviation = L4 regret | — | Lemma D (Azuma) gives the independence scale; U supplies the UCB1 numbers | logged $(e_t,f_t)$ in PA4 | 0 |

### Single-algorithm line (Lecture 6)

| # | Candidate | Verdict | Novelty | Proof status | Numeric check | Real-game experiment | Cost |
|---|---|---|---|---|---|---|---|
| S1 | **PA1 + P1. Carried Hedge over experts; external vs policy regret via the comparator's continuation $Q^{e,\sigma}$** | **play + prove** | technique standard (Kakade–Langford PDL; DFG20 Lemma 3; EKM09); statement + counterexample new to the project; lit note lists "policy regret in stochastic games with finite-horizon episodes" as unsearched (`scout_exploitation_pokemon.md:171`) | **proved** (identity, decomposition, bound, counterexample) | `p1.out`: identity 2.8e-16 on 200 random games; counterexample ext = 0, policy regret 0.5; random sweep: mismatch median 0.003 vs external 0.055 | HedgeBot(i),(ii) + 3 experts vs foul-play; check regret bound on the log; measure $\max_eW_e-W_H$ | 9 h wall |
| S2 | **P2. Switching time: anytime (non-forgetting) Hedge recovers in $T_1d_1/d_2$ turns, schedule-free; fixed share in ≤ $\ln(k/\alpha)/(\eta d_2)$, history-free** | **prove (one paragraph) + play as PA1 variant (ii)** | corollary of R3a + Herbster–Warmuth; exact crossing time new to the project | **proved** | `p2.out`: crossing 510/2040/8160 vs predicted 510/2040/8159; fixed share 102–158 turns for $T_1$ = 350–5 600; noise sd 0.1: 597 vs 150 | switch test, 2 × 80 games | 1.5 h wall |
| S3 | **PA2. ε-safe exploitation with a non-vacuous certificate** (ε ≤ 1 %) | **play**; theory cite | L2(b) + GS15 §6.4 + Li 2026 (cite); arithmetic + dual-price fractions new numbers | cited; LP sensitivity | `a2.out`: concave in 300/300; 35 % / 80 % / 100 % of top gain at 0.3 / 1 / 2 %; add-on 0.105 / 0.35 / 0.70 win prob | 3 ε × 200 vs foul-play + 2 × 150 vs exploiter; engine pre-screen | 7.5 h wall; 1.4 CPU-h |
| S4 | Tracking / dynamic regret vs a switching opponent | cite only + S2 | Herbster–Warmuth 1998; Zhang et al. 2020 | — | — | folded into S2 | — |
| S5 | Policy regret vs an *adaptive* (white-box) exploiter | cite (Arora 12/18, Nguyen 24c: Ω(T) lower bounds) | — | — | — | — | — |

Recommendation: take **S1 (PA1 + P1 + P2)** and **D1 (PA3/B3)** to Plan as the two playable algorithms with theorems and
measurable predictions; run **S3 (PA2)** and **D2 (PA4β)** behind them. Write D4 (T) as a half-page supporting proposition.

---

## 2. Finalists: statements → proofs → numeric checks → experiment designs

### 2.1 Proposition P1 (external vs policy regret of per-turn Hedge over experts) — proved

**Setting.** Finite-horizon simultaneous-move stochastic game $G$ as in `search_rl_links.md` §0 (stages $t<H$, chance, terminal
payoff $u\in[-1,1]$, cap 0). Fixed **Markov** opponent $\sigma$: $y_t=\sigma_t(s_t)$ (stock foul-play/MCTS is memoryless across
games and, given its search randomness, a Markov mixed strategy in the state it observes). Experts $e=1..k$ are Markov policies
$q_e(t,s)\in\Delta(A(s))$. The learner plays any history-dependent $p_t\in\mathrm{conv}\{q_e(t,s_t)\}$ (Hedge weights are
history-dependent). $Q^{e,\sigma}_t(s)$, $V^{e,\sigma}_t(s)$: the value of following $e$ against $\sigma$ from $(t,s)$;
$V(\pi,\sigma)=V^{\pi,\sigma}_0(s_0)$; $Q^*$ the minimax $Q$.

**Assumptions.** (P1.1) $a_t\perp b_t$ given $h_t$ (simultaneous moves; L0's only hypothesis). (P1.2) $\sigma$ Markov.
(P1.3) Experts Markov. (For the bound (iii): the per-turn matrix the learner scores with satisfies $\lVert\hat A_t-Q^*_t(s_t)\rVert_\infty\le\varepsilon_v$.)

**Statement.** For every expert $e$:
(i) *Identity.* $V(e,\sigma)-V(p,\sigma)=\mathbb E_{p,\sigma}\Big[\sum_{t<\tau}\big(q_e(t,s_t)-p_t\big)^\top Q^{e,\sigma}_t(s_t)\,\sigma_t(s_t)\Big].$
(ii) *Decomposition.* The right side equals
$\underbrace{\mathbb E\sum_t(q_e-p_t)^\top Q^*_t\sigma_t}_{\text{external term}}+\underbrace{\mathbb E\sum_t(q_e-p_t)^\top(Q^{e,\sigma}_t-Q^*_t)\sigma_t}_{\text{continuation mismatch}}.$
The external term is the expectation (over $b_t\sim\sigma_t$) of the per-turn external regret the Hedge layer measures when it
scores experts with the exact stage matrix: $\sum_t(q_e-p_t)^\top Q^*_te_{b_t}$.
(iii) *Bound.* $\mathrm{PolicyReg}(e):=V(e,\sigma)-V(p,\sigma)\le\mathbb E[\mathrm{ExtReg}_T(e)]+2\varepsilon_v\mathbb E[T]+\mathbb E\sum_t\lVert q_e-p_t\rVert_1\,\lVert Q^{e,\sigma}_t(s_t)-Q^*_t(s_t)\rVert_\infty$.
If $\sigma$ is $\delta$-minimax and $e$ is $\delta'$-maximin at every state (stage strategies $\delta$-optimal in $Q^*$), the last
term is $\le2H^2(\delta+\delta')$.
(iv) *Lower bound (the two regrets are incomparable).* There is a game with $H=3$, two experts and a Markov $\sigma$ in which the
external term is **exactly 0 for every weight vector** while $\mathrm{PolicyReg}(e_2)=w_{e_1}\in[0,1]$; carried Hedge with the
shipped anytime rule keeps $w=(\tfrac12,\tfrac12)$ forever and earns $+\tfrac12$ per game against $+1$ for $e_2$ alone.

**Proof.** (i) L0 with potential $\Phi=V^{e,\sigma}$: $\Phi=u$ on terminals, $\Phi_H=0$ at the cap, and $Q[\Phi]_t=Q^{e,\sigma}_t$
by definition. L0 gives $\Phi_0-V(p,\sigma)=\mathbb E_{p,\sigma}\sum_{t<\tau}(\Phi_t(s_t)-p_t^\top Q^{e,\sigma}_t\sigma_t)$, and
$\Phi_t(s)=V^{e,\sigma}_t(s)=q_e(t,s)^\top Q^{e,\sigma}_t(s)\sigma_t(s)$ because $e$ and $\sigma$ are Markov. (ii) Add and subtract
$Q^*_t$. The external-term identification is the tower rule over $b_t\sim\sigma_t(s_t)$ given $h_t$. (iii) Hölder on the mismatch
term; $|x^\top(\hat A-Q^*)e_b|\le\varepsilon_v$ twice for the $\hat A$-vs-$Q^*$ swap (as in L2 step 2). For the $\delta,\delta'$
clause: by L1(a) applied to $(e,\sigma)$ with potential $V^*$, $|V^{e,\sigma}_t-V^*_t|\le(H-t)(\delta+\delta')$ at every state,
hence $\lVert Q^{e,\sigma}_t-Q^*_t\rVert_\infty\le(H-t-1)(\delta+\delta')$, and $\lVert q_e-p_t\rVert_1\le2$. (iv) Construction
(`p1_policy_vs_external.py` [2]): $s_0$ with rows $a_1,a_2$; $a_1$ leads to an absorbing payoff-0 state, $a_2$ to $s_1$ = matching
pennies (±1) whose minimax value is 0, so $Q^*_0(s_0)\equiv0$. $\sigma$ plays the first column everywhere. Experts $e_1=(a_1;\ \text{match})$,
$e_2=(a_2;\ \text{match})$ differ only at $s_0$, where both score 0 under $Q^*$ against every column, and coincide at $s_1$; so every
per-turn reward vector is constant across experts and the external regret is identically 0. $V(e_2,\sigma)=1$, $V(e_1,\sigma)=0$,
so $V(p,\sigma)=1-w_{e_1}$. $\square$

**Reading.** The Hedge layer scores experts with the *minimax* continuation ($\hat A_t\approx Q^*$); an expert whose advantage
against *this* opponent only materialises later (a set-up move) is under-credited by exactly $Q^{e,\sigma}-Q^*$. That is the
quantity gray-box RL's critic estimates ($Q^{\pi,\sigma}$), and it is why external regret ≈ 0 on the Hedge logs is compatible
with a positive win-rate gap to the best single expert. It does **not** say the gap is large in Pokémon: on random games the
mismatch is two orders below the external term (§ numeric), so the effect is structural, and the Showdown measurement
$\max_eW_e-W_H$ is the test.

**Novelty.** PDL is Kakade & Langford 2002 (`search_rl_links.md` L1 Remark 3); the Markov-game PDL is Daskalakis–Foster–Golowich
2020 Lemma 3 (`links_attribution.md:193`); experts over policies in MDPs is Even-Dar–Kakade–Mansour 2009 (L2 Remark 1). Policy
regret with *adaptive* adversaries (Arora et al. 2012/2018; Nguyen 2024c Thm 1, $\Omega(T)$) is a different mechanism — ours is
against a *fixed* Markov opponent and the gap comes from the comparator's continuation. The lit notes record "policy regret in
stochastic games with finite-horizon episodes" as not searched (`scout_exploitation_pokemon.md:171`). Verdict: **technique
standard, statement and counterexample new to the project; present as a proposition, not a theorem.**

**Numeric check** (`p1.out`, 7 s). (1) Identity and decomposition on 200 random games ($H=3$, 3 states/stage, 3×3, Dirichlet
chance): max error $2.8\times10^{-16}$. (2) Counterexample: $Q^*_0(s_0)=0$ in all cells; external term $0.0$, policy regret
$0.50/0.90/0.10$ for $w_{e_1}=0.5/0.9/0.1$; 400 simulated games of carried anytime Hedge: mean payoff $+0.510$, final weights
$(0.5,0.5)$, realised external regret $(0,0)$. (3) Random sweep with $\sigma=(1-\lambda)\,$minimax$+\lambda\,$uniform and experts
{maximin, exact BR to $\sigma$, random}: median |mismatch| 0.000 / 0.001 / 0.002 / 0.003 vs external term 0.027 / 0.030 / 0.044 /
0.055 at $\lambda=0/0.25/0.5/1$; mismatch exceeds the external term in ≤ 3 % of games.

**Real-game experiment.** PA1 (§0). The sharp check is pathwise (one run): $\sum_t(r_{e,t}-w_t^\top r_t)\le2(1.5\sqrt{T\ln k/2}+\sqrt{\ln k/8})$
for every $e$ on the HedgeBot log; the win-rate check needs the three expert arms (n = 150 each) and HedgeBot (n = 300).

### 2.2 Proposition P2 (switching time of carried Hedge; fixed share) — proved

**Assumptions.** (P2.1) $k=2$ experts $F,N$, normalised rewards in $[0,1]$. (P2.2) R3a's drift model: $r_{N,t}-r_{F,t}=d_1>0$
for $t\le T_1$, $=-d_2<0$ for $t>T_1$ (deterministic; noise in the numeric check). (P2.3) Variant (i): any FTRL/Hedge of the
form $w_F(t)=\sigma(\eta_tD_t)$ with $D_t=\sum_{s\le t}(r_{F,s}-r_{N,s})$ and $\eta_t>0$ (the shipped anytime rule,
`exp3_live.py:69`, is $\eta_t=\sqrt{8\ln2/t}$). Variant (ii): Hedge + fixed share, $v\propto w\,e^{\eta r}$, $w=(1-\alpha)v+\alpha/k$.

**Statement.** (i) $w_F(t)\ge\tfrac12\iff D_t\ge0$, so the recovery time after the switch is $t^*-T_1=T_1d_1/d_2$ **for every
step-size schedule**: linear in the pre-switch history. The phase-2 regret against $F$ up to $t^*$ is $\ge\tfrac12 d_2(t^*-T_1)=\tfrac12T_1d_1$.
(ii) With fixed share, $w_F\ge\alpha/k$ always and the log-odds of $F$ grow by at least $\eta d_2$ per phase-2 turn, so the recovery
time is $\le\ln(k/\alpha)/(\eta d_2)$, **independent of $T_1$** (for $k=2$; with PA1's $k=4$ the mixing step can also pull a weight *down* toward $1/k$, so this is an upper bound only); the price is Lemma H's $\ln(1/(1-\alpha))/\eta\approx\alpha/\eta$ per turn
(0.035/turn at the X6 constants, `hedge_experts.md:59–60`). The two cross at $T_1=\ln(k/\alpha)/(\eta d_1)$.

**Proof.** (i) $D_t=-d_1t$ for $t\le T_1$ and $D_t=-d_1T_1+d_2(t-T_1)$ after; $\sigma(\eta_tD_t)\ge\tfrac12\iff D_t\ge0\iff t\ge T_1(1+d_1/d_2)$.
During recovery $w_N\ge\tfrac12$ and the per-turn loss vs $F$ is $d_2w_N$. (ii) The fixed-share step keeps $w\ge\alpha/k$;
multiplicatively, $\mathrm{logit}\,w_F$ increases by $\eta d_2$ before the mixing step, and mixing toward $1/k$ moves a weight below
$1/2$ upward, so the increase is at least $\eta d_2$ until $w_F=\tfrac12$. $\square$

**Novelty.** Corollary of R3a (`regret_theory.md`), Herbster & Warmuth 1998 for the tracking side (`hedge_experts.md:199`). The
exact schedule-free crossing time is new to the project; it is the flip side of R3d's amortisation (a decaying-$\eta$ learner
cannot forget, so it cannot switch either). Cite + corollary.

**Numeric check** (`p2.out`, 1.5 s). Deterministic drifts $d_1=0.0133$, $d_2=0.0291$ (R3b's fits): variant (i) crosses at
$t=510/2040/8160$ for $T_1=350/1400/5600$ (predicted 510/2040/8159), i.e. 4.6 / 18.3 / 73.1 battles of 35 turns. Fixed share
(X6 constants): 102 / 156 / 158 turns (2.9–4.5 battles), flat in $T_1$; the history-free bound 1 187 is loose because under
drift $d_1$ fixed share holds $w_F\approx\alpha/(k\eta d_1)=0.38$, not $\alpha/k$. With reward noise sd 0.1 (20 seeds): 597 ± 237 vs
150 ± 83 turns. Break-even $T_1=2\,597$ turns ≈ 74 battles. **Consistency with X6:** X6 used fixed share (`--eta 0.2 --alpha 0.002`,
`exp3_live.py:12`) and switched in ≤ 5 battles — predicted; the carried run of `hedge_stream.py` used the anytime rule but its
phase-2 drift favoured the same expert, so no crossing was observable there. No tension.

**Real-game experiment.** PA1 switch test (§0): 40 battles vs foul-play 100 ms then 40 vs an opponent that favours the other
expert (foul-play 2000 it, or our `safex` bot); log the crossing battle for variants (i) and (ii). Prediction: (ii) in ≤ 5
battles, (i) in ≈ $T_1d_1/d_2$ ≈ 18 battles if the drifts are as fitted (they will differ on Showdown; the *ratio* test
"(i) − (ii) grows with $T_1$" is schedule-free). 160 games ≈ 1.5 h wall.

### 2.3 Certificate arithmetic for ε-safe exploitation (PA2) — cite + numbers

**Statement (corollary of L2(b)/Theorem A, cite).** With $\hat A_t$ clipped to $[-1,1]$ (so range ≤ 2; the raw-`ptp` LP of `nash.py:244–259` must be run on the squashed matrix) and the ε-safe expert in every $\hat A_t$, $\mathbb E\sum_g(V^*_0-u_g)\le(\varepsilon_s+\varepsilon\cdot\mathrm{range}+2\varepsilon_v)\mathbb E[T]+\mathbb E[R(T)]$.
In win-probability units (payoff range 2 ↔ 1), the ε add-on per game is $\varepsilon\bar H$ with $\bar H\approx35$ non-forced turns
(`RESULTS.md:514`): 0.105 (ε = 0.3 %), 0.35 (1 %), **0.70 (2 %, the shipped SX value, vacuous)**. The base term $\bar H(\varepsilon_s+2\varepsilon_v)$
is vacuous already (`new_directions.md` §5), so "certified" here means *the ε add-on is non-vacuous*, nothing more.

**Gain curve (LP sensitivity, cite).** $g_t(\varepsilon)=\max\{x^\top\hat A_t\hat q:(x^\top\hat A_t)_j\ge\hat v_t-\varepsilon\,\mathrm{range}\}$
is concave piecewise-linear with slope = dual price of the safety constraints.

**Numeric check** (`a2.out`, 1.2 s; 300 real logged matrices, predictor $\hat q$ = BR to uniform): 108 have a positive top-move gain
(median 0.038 range); median fraction recovered 0.35 / 0.80 / 1.00 / 1.00 at ε = 0.3 / 1 / 2 / 5 %; 0 concavity violations.
(F3's "88 % at 2 %" is on a different set; consistent.)

**Experiment.** PA2 (§0). Prediction 0.535 / 0.566 / 0.58 vs foul-play if the engine's 0.51/0.58 transfers; the n = 200 CI
(±0.07) does not resolve 0.03, hence engine pre-screen (1 000 games/arm, 1.4 CPU-h) and Showdown for the sign and the
exploiter-side loss (≤ 0.105 at 0.3 %).

### 2.4 Corollary B3 (bandit vs full feedback at the decision node; last iterate diagnostic) — cite + real-matrix numbers

**Statement (cite).** At a node with payoff matrix $A\in[0,1]^{n\times m}$, both sides sampling independently given the history,
NashGap$(\bar x_N,\bar y_N)=(\mathrm{Reg}^1_N+\mathrm{Reg}^2_N)/N$ (L4, exact). With Exp3.P/Exp3-IX selection on both sides, w.p.
$\ge1-\delta$ each realised counterfactual regret is $O(\sqrt{Nk\ln(k/\delta)})$ (Auer et al. 2002 Thm 6.3; Neu 2015), so the gap is
$O(\sqrt{k\ln(k/\delta)/N})$; with full-information Hedge it is $O(\sqrt{\ln k/N})$; with decoupled UCB1 there is no bound (U).
At $k=9$, $N=2000$: $4\sqrt{2k\ln k/N}=0.56$ range vs $2\sqrt{2\ln k/1000}=0.13$ — **both vacuous at MCTS budgets**, which is why
the measurement matters.

**Numeric check** (`b3.out`, 9 s; 120 two-sided real logged matrices, median 5×6, normalised to range 1):

| solver | median gap | p90 | frac > 0.05 |
|---|---|---|---|
| RM+ full info, 1 000 it | 0.0000 | 0.0001 | 0.00 |
| Exp3 bandit, N = 500 (shares) | 0.208 | 0.351 | 0.97 |
| Exp3 bandit, N = 2 000 (shares) | 0.154 | 0.241 | 0.82 |
| UCB1 decoupled, N = 500 / 2 000 | 0.180 / 0.120 | 0.311 / 0.222 | 0.86 / 0.81 |
| Hedge avg / last, η = 0.1, 1 000 it | 0.032 / 0.0087 | 0.045 / 0.092 | 0.02 / 0.21 |
| OMWU avg / last, η = 0.1, 1 000 it | 0.032 / 0.0085 | 0.044 / 0.086 | 0.01 / 0.17 |

Pure gap of the top move (what a most-visited rule pays): RM+ median 0 (p90 0.087), Exp3-2000 0.016 (p90 0.108), UCB1-2000 0.003
(p90 0.085). Reading: (1) at the real decision sizes the feedback model costs 0.12–0.15 range of stage gap at N = 2 000 — the
L1(b) lower side with a myopic exploiter turns this into a full-game loss along the exploiter's trajectory; (2) the last-iterate
distributions are bimodal because most real stage matrices have pure saddles (RM+'s top-move gap has median 0), so
"average vs last iterate" matters on a minority (~20 %) of turns; OMWU's last iterate is not better than plain Hedge's here
(1 000 iterations, untuned η) — cite the theory, do not claim an engine effect.

**Experiment.** PA3 (§0).

### 2.5 Lecture 7 on the meta-game with sampled payoffs and parallel games (PA4α) — cite + numeric

**Statement (cite).** Meta-game $M\in[0,1]^{K\times L}$ (win probabilities of expert policies; stationary only if the experts
carry no in-game state across games — PA1's carried weights violate this, so use fixed experts). Exp3 vs Exp3 over $G$ games:
$\mathrm{Gap}_M(\mathbb E\bar x_G,\mathbb E\bar y_G)\le(\mathbb E\mathrm{Reg}^1+\mathbb E\mathrm{Reg}^2)/G\le4\sqrt{K\ln K/G}$
(Lecture 7's two inequalities hold in expectation because Exp3's pseudo-regret bound is against every fixed arm under adaptive
losses; Auer et al. 2002). Bernoulli outcomes add a martingale term $O(\sqrt{\ln(K/\delta)/G})$ (Azuma). Noisy full information
($K+L$ games per round) at equal game budget $B$ gives $O(\sqrt{(K+L)\ln K/B})$ — the same order: Exp3's $\sqrt K$ *is* the
budget split. Delay $d$ (parallel games sharing one learner) multiplies the regret by $\approx\sqrt{d+1}$ (Weinberger & Ordentlich
2002; Joulani et al. 2013). The meta-game value is PSRO's restricted population value, not $V^*_0$.

**Numeric check** (`b1.out`, 21 s; $M=0.5+S$ with a cyclic $S$ of entries ±0.03–0.10, value 0.500, mixed minimax; 20 seeds):

| games B | Exp3 gap | avg win − v | full-info (K+L)/round gap | Exp3 delay-4 gap | bound |
|---|---|---|---|---|---|
| 300 | 0.043 | −0.002 | 0.064 | 0.045 | 0.544 |
| 3 000 | 0.042 | −0.002 | 0.055 | 0.041 | 0.172 |
| 30 000 | 0.024 | 0.000 | 0.024 | 0.028 | 0.054 |

Uniform play has gap 0.030. **Finding:** convergence of the time-average meta-strategies is not observable below $\sim10^4$
games because the real meta-game is flat (expert win-rate differences ≤ 0.1, cf. SX/E1fix); value convergence is immediate for
the same reason. Delay is free. Do not run PA4(α) for convergence; run PA4(β) asymmetric (§0).

### 2.6 Lemma D (realised joint play of independently randomising learners) — proved, trivial

**Statement.** Two learners sample $a_t\sim x_t$, $b_t\sim y_t$ independently given $h_t$. Let $\hat J_T(i,j)=\frac1T\#\{t:(a_t,b_t)=(i,j)\}$
and $\bar J_T=\frac1T\sum_tx_ty_t^\top$. Then for all $\delta$, w.p. $\ge1-\delta$, $\max_{ij}|\hat J_T-\bar J_T|\le\sqrt{\ln(2nm/\delta)/(2T)}$,
hence TV$(\hat J_T,\bar J_T)\le\tfrac{nm}2\sqrt{\ln(2nm/\delta)/(2T)}$. The CCE deviation of $\bar J_T$ for the row player is
$\max_i\frac1T\sum_t(Ay_t)_i-\frac1T\sum_tx_t^\top Ay_t$ = the full-information external regret; in a zero-sum game $\bar J_T$'s
marginals are therefore $(\varepsilon_1+\varepsilon_2)$-minimax (Lecture 7) and $\bar J_T$ is a mixture of products.
**Proof.** Azuma–Hoeffding on $\mathbb 1[(a_t,b_t)=(i,j)]-x_t(i)y_t(j)$, a bounded martingale difference by independence given $h_t$;
union bound over cells. $\square$ **Use.** At $T=10^6$ on 2×2 the scale is $0.003$; U's UCB1 plateaus show TV 0.04–0.09 — rigorously
not sampling noise (UCB1 is deterministic given $h_t$, so the lemma does not apply to it). Cite Lecture 7 §3.2's last sentence.

### 2.7 Proposition T (depth composition of the node identity under drifting child values) — sketch + numeric

**Model.** Depth-$D$ tree, full information, synchronous: every live node runs Hedge (both sides) on the matrix $\hat A^{(n)}$ whose
cells are the children's *current* value estimates (chance children averaged); terminals exact. Mean propagation: uniform
$\hat V_n=\frac1n\sum_{k\le n}x_k^\top\hat A^{(k)}y_k$ (Lecture 7's time average) or weighted $\hat V_n=(1-\alpha_n)\hat V_{n-1}+\alpha_nx_n^\top\hat A^{(n)}y_n$,
$\alpha_n=(D+1)/(D+n)$ (Bai–Jin–Yu 2020). $\mathrm{err}_d(n)=\max_{\text{level-}d\text{ nodes}}|\hat V_n-V^*|$; $R_n$ = a Hedge regret
bound ($R_n\le c_0\sqrt n$ with the anytime rule).

**Statement (uniform).** $\mathrm{err}_d(n)\le\frac{R_n}{n}+\frac1n\sum_{k\le n}\mathrm{err}_{d+1}(k)$; hence with $\mathrm{err}_D\equiv0$,
$\mathrm{err}_d(N)\le c_0(2^{D-d}-1)/\sqrt N$ and the root average strategy's gap is $\le(2c_0+4c_0(2^{D-1}-1))/\sqrt N$: **a factor 2 per
level**. With chance and exact terminals, $\mathrm{err}_{d+1}(k)$ is replaced by its chance-average over live children, so the
constant is survival-weighted as in L5's $\rho_D$.
**Proof (one side).** $\hat V_n\ge\max_i\frac1n\sum_k(\hat A^{(k)}y_k)_i-\frac{R_n}n\ge\max_i(A\bar y_n)_i-\frac1n\sum_k\lVert\hat A^{(k)}-A\rVert_\infty-\frac{R_n}n\ge\mathrm{val}A-\frac1n\sum_k\mathrm{err}_{d+1}(k)-\frac{R_n}n$,
and symmetrically above; $\frac1n\sum_{k\le n}C/\sqrt k\le2C/\sqrt n$ gives the doubling. $\square$ **Gap named:** whether the factor 2
is tight for Hedge (errors are signed and may cancel), and the weighted-averaging bound — Bai–Jin–Yu prove the $(1+1/H)$-per-level
compounding for their weights with a matched learning-rate schedule (Nash V-learning); here it is **cite + numeric**.

**Novelty.** Lisý et al. 2013 Thm 4.11 and Kovařík–Lisý Thm 5.4/5.12 are asymptotic ("the rate of the convergence is not given",
`links_attribution.md:90`; "a practically useful bound on this speed of convergence is missing", `scout_search_solvers.md:88`);
KL's UPO constant $C'=12(2^D-1)-8D$ is exponential in $D$ (`links_attribution.md:100`) and their Problem 5.17 asks whether linear is
possible (Prop 5.15: not for their theorems). Our finite-$N$ recursion in the *synchronous full-information* model is new to the
project; the mechanism (uniform averaging of a $1/\sqrt k$ drift doubles it) is the content; the fix is BJY's. Not playable
(poke-engine is Rust and not edited); it supports the reading "MCTS visit shares at depth inherit a compounding drift error".

**Numeric check** (`t.out`, 11 s; 2×2 nodes, chance ½/½, 4 seeds): root $|\hat V_N-V^*|\sqrt N$ at $N=4000$: uniform 0.017 / 0.068 /
0.159 / 0.463 for $D=1..4$ (≈ ×2–3 per level); weighted 0.008 / 0.010 / 0.048 / 0.077; suffix-half 0.008 / 0.017 / 0.081 / 0.079.
Root strategy gap·√N: uniform 0.06 → 0.20, weighted 0.03 → 0.11. The compounding is real for the time average and absent for the
weighted average; the bound's ×2 is of the right order at these depths.

---

## 3. Cite-only answers, written out

- **(a) Meta-game / PSRO / fictitious play.** §2.5. Fictitious play vs Hedge: a no-regret learner facing a myopic best-responder
  earns the value and its average is maximin (Freund & Schapire 1999) — this is L2(b) with the white-box one-turn exploiter, already
  in `search_rl_links.md`. PSRO's inner loop is an RL best response (`noregret_vs_rl.md:103`); our meta-game is its outer loop with
  fixed experts.
- **(b) Per-state averaging.** The lit map corrects the task wording: Kovařík–Lisý's failure (Thm 4.1) is HC selection on *observed
  payoffs*; averaging over joint actions is their fix (`links_attribution.md:97`). The Markov-game positive results are Bai–Jin–Yu
  2020 (weighted averaging, certified policies) and Jin et al. 2021 (V-learning); ours is Theorem A for stage-wise re-solving.
- **(c) Last iterate.** Daskalakis & Panageas 2019 (OMWU last iterate, unique NE), Wei, Lee, Zhang & Luo 2021 (linear rate, unique
  NE), Cai & Zheng 2022 ($O(1/\sqrt T)$ without uniqueness). Alternation breaks Lecture 7's identity (Burch, Moravčík & Schmid 2019
  Obs. 1; `new_directions.md` D).
- **(d) Bandit vs full.** Auer, Cesa-Bianchi, Freund & Schapire 2002 (Exp3, Exp3.P); Neu 2015 (Exp3-IX high probability);
  Kovařík–Lisý UPO (open for every concrete selector, `links_attribution.md:102`).
- **(e) CCE.** Zero-sum CCE marginals are maximin (e.g. Roughgarden, *Twenty Lectures*, Ex. 13); Lecture 7 §3.2 last sentence for
  sampled actions; Lemma D above.
- **Policy regret.** Arora, Dekel & Tewari 2012; Arora et al. 2018 (external and policy regret compatible when both play no-regret);
  Nguyen 2024c (Markov games: $\Omega(T)$ against adaptive adversaries). P1 is the fixed-Markov-opponent case.
- **Safe exploitation.** Ganzfried & Sandholm 2015 §6.4; Johanson et al. RNR; Jeary & Turrini 2023; Li 2026 (runtime certificate
  $B_t\le\varepsilon_{\max}$, EFG only). Ours: L2(b) with ε, stochastic game with leaf; the certificate arithmetic is §2.3.
- **Tracking.** Herbster & Warmuth 1998; Zhang, Lu & Yang 2020 (dynamic vs adaptive regret); AdaHedge (de Rooij et al. 2014) for the
  second-order slack (`new_directions.md` row 4).
- **Delayed feedback.** Weinberger & Ordentlich 2002; Joulani, György & Szepesvári 2013 — for parallel Showdown games sharing a learner.

## 4. Refuted / superseded

- **"Lecture 7 on the meta-game of repeated Showdown matches is testable at a few hundred games"** — refuted numerically (B1) *under a
  meta-matrix of the measured expert-difference size (entries 0.5 ± 0.03–0.10; the real 4×4 $M$ is unmeasured — a 16-cell × 50-game engine
  brute force ≈ 1 CPU-h would pin it)*: such a meta-matrix is flat (uniform play is a 0.03-Nash), the Exp3 bound at 300 games is 0.54, and the realised
  time-average gap stays ≈ 0.04 up to 3 000 games. Only value convergence is observable, and it is trivial.
- **"Zero external regret of the Hedge layer ⇒ it does as well as the best expert in win rate"** — false in general (P1 (iv):
  external regret identically 0, policy regret 1/2); generically small on random games (mismatch two orders below the external term).
- **"Fixed share recovers in $\ln(k/\alpha)/(\eta d_2)$ turns"** — that is an upper bound only; under drift $d_1$ the fixed-share
  weight sits at $\approx\alpha/(k\eta d_1)$ (0.38 at the X6 constants), so recovery is ~8× faster than the bound (`p2.out` [B]).
- **"X6's fast switch (0–3 battles) contradicts a linear-in-history switching cost"** — no tension: X6 ran *fixed share*
  (`exp3_live.py:12`), to which P2(ii) applies; P2(i) applies to the anytime carried run, where no switch was observable.
- **"Full information at game level (replaying all experts) beats Exp3 at equal game budget"** — same order (B1: 0.064 vs 0.043
  at 300; equal at 30 000); Exp3's $\sqrt K$ is the budget split.
- **"OMWU's last iterate beats the time average on the engine's matrices"** — not at 1 000 untuned iterations: last-iterate gaps are
  bimodal (p90 0.086) because ~half the real stage matrices have pure saddles; cite the theory only.
- **"The depth-composition bound can be linear in $D$ with uniform averaging"** — the $\times2$-per-level mechanism is visible
  numerically (0.02 → 0.46 over $D=1..4$); KL's Prop 5.15 says their constants cannot be linear either. Weighted averaging is the fix.
- **Coordinator's re-prioritisation** superseded the sprint's first ranking (depth composition T as the dynamics finalist): T is
  demoted to supporting theory because poke-engine is not edited and T is not a move-choice function.

## 5. Budget and provenance

CPU ≈ 1 min total (`p1` 7.5 s, `p2` 1.5 s, `b1` 21 s, `b3` 9 s, `a2` 1.2 s, `t` 11 s), numpy/scipy only, `engine/.venv/bin/python`,
single process. Real matrices from `scratchpad/brainstorm/data.pkl` (1 629 logged decisions vs MCTS-2000, as used by `n4`). No
project code, engine code, paper or Showdown process touched; no Modal. Infrastructure facts from the RESULTS/engine mapper
(recomputed from `results/experiments/*.jsonl`: `wall_s` 4–13 s per engine game for matrix bots; SX 0.530/0.510/0.580 at n = 200;
hedge_stream paired +0.146 [+0.039, +0.253] with the caveat that its CI treats sequential games as independent) and from reading
`~/sandbox/ourbot/run_ourbot.py` and `~/sandbox/foul-play/fp/run_battle.py` directly.
