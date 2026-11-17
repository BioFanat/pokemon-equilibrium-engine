# Theory map: every result at its level, in its verified form

Consolidation, 2026-11-15. This file supersedes the statements in the source notes wherever they differ; the source
notes keep the long proofs. Sources: `theory_section.md`, `full_game.md`, `hedge_experts.md`, `regret_theory.md`,
`search_rl_links.md` (L0–L6), `new_directions.md` (S, U, R3d), `dynamics_directions.md` (P1, P2, B3, D, T, PA1–PA4),
`bridge_plan.md`, `CHECK.md`, `../lit/next_directions.md`, and `../engine/RESULTS.md` for every number. Lectures:
`../lecture6.pdf` (experts, MWU, adversary model) and `../lecture7.pdf` (no-regret dynamics, minimax, CCE).

How to read it.

- §0 notation, canonical names and the status vocabulary.
- §1 levels M0–M4 and the two course views.
- §2 assumptions A1–A12, numbered once and cited by number everywhere below.
- §3 the bridges M0→M1→M2→M3→M4.
- §4–§7 results, grouped by view, then by level. Each result has a status tag, a level tag, a view tag, its
  assumptions, a statement with quantifiers in order, a proof (idea → steps → conclusion) or a precise pointer, and
  its real-game test.
- §8 playable algorithms. §9 status table (as of 2026-11-15). §10 refuted / superseded. §11 tightening results and
  real-Showdown open items (2026-11-16), each with its literature status and the experiment that bears on it. §12 live tests and follow-up (2026-11-16).

Corrections from the verifier and the correlation probe (2026-11-15) are applied in place and override the source
notes; each one is also listed in §10 so the old versions are not re-derived.

---

## 0. Notation, units, names, status vocabulary

### 0.1 Notation (one meaning per symbol)

| Symbol | Meaning |
|---|---|
| $G$ | finite-horizon two-player zero-sum stochastic game with simultaneous moves and chance (`full_game.md` §1) |
| $H$; $\tau$ | horizon, counted in decision nodes: $H=300$ nodes in our engine loop (A6, A12; `game_model.md` §1.6); Showdown's limit is a tie after turn 1000; realised length, $\tau\le H$ |
| $s,\ a,\ b$ | state; our action (row, maximiser); opponent action (column) |
| $h_t$ | public history $(s_0,a_0,b_0,\dots,s_t)$ |
| $x_t,\ p_t$ | our stage mixture (per-turn agent; Hedge learner) |
| $\sigma$ | opponent strategy, $\sigma(h_t)\in\Delta(B(s_t))$; $\sigma_{\rm BR}$ a best response to our policy |
| $V^*_t,\ Q^*_t$ | game value and true stage matrix on the payoff scale |
| $\hat Q_t,\ \hat A_t$ | the stage matrix we build (`nash.full_matrix`); $\hat A_t$ when a learner scores experts with it |
| $\operatorname{val}(A)$ | $\max_x\min_y x^\top Ay$ |
| $\Delta$ | payoff **range** of a matrix (max entry − min entry); $\Delta=2$ on $[-1,1]$ |
| $\mathcal P(X)$ | the probability simplex on a finite set $X$ (written $\Delta(X)$ in the sources; renamed here to avoid the clash with the range) |
| $\varepsilon_s$ | stage exploitability: $\operatorname{val}\hat Q-\min_y x^\top\hat Qy$ |
| $\varepsilon_v$ | stage-matrix error, A7 |
| $\varepsilon_{\rm sim}$ | simulator-fidelity term, A5 |
| $\delta(t,s)$ | per-turn error $\varepsilon_s+2\varepsilon_v$ (Theorem A). Confidence levels are written $\vartheta$; no other $\delta$ |
| $\bar H$ | $\mathbb E_{\pi,\sigma_{\rm BR}}[\tau]$, expected length against a best response (in scored turns where stated: turns on which our agent had at least two legal actions and the opponent's action was observed and matched a sampled world, turns where the opponent had one option included). $\bar H_\sigma$: against opponent $\sigma$. The sources' $H_{\rm eff}$ and $\bar L$ are this quantity |
| $k$; $\eta$; $\alpha$ | number of experts; Hedge step; fixed-share rate |
| $d,\ d_1,\ d_2$ | per-turn normalised-stage-payoff drift between two experts (R3a model); $d_1$ vs MCTS, $d_2$ vs the exploiter |
| $\operatorname{lgs}(z)$ | logistic $1/(1+e^{-z})$ |
| $\mathrm{TV}(p,q)$ | $\tfrac12\lVert p-q\rVert_1$ |
| $J_N,\ \bar x_N,\ \bar y_N$ | empirical joint visit distribution at a node and its marginals (visit shares) |
| $\eta_{\rm NL}$ | value of revealing the opponent's team to both sides (ours; it bounds the error Long et al. call non-locality; `full_game.md` §6.2) |

### 0.2 Units (A9 fixes them; used at every number below)

- Payoff $u\in[-1,1]$; score $s=(u+1)/2\in[0,1]$. Exploitability in score units is half the exploitability in payoff
  units.
- An exploiter with mean score $Z$ against our bot gives, at the point estimate, the lower bound
  $\operatorname{expl}\ge 2(Z-\tfrac12)$ in payoff units (a 95% lower confidence bound once $Z$ is replaced by its lower confidence limit) and $\ge Z-\tfrac12$ in score units.
- Engine matrices are in heuristic engine units. The map to the payoff scale is unknown; its error is part of
  $\varepsilon_v$ (A7).

### 0.3 Canonical names and aliases

| Canonical name here | Aliases in the sources |
|---|---|
| Lemma 1 (one step) | `theory_section.md` Lemma 1; `full_game.md` Lemma 1; Lean `lemma1_one_step` |
| Theorem A (per-turn composition) | `theory_section.md` Theorem 1; `search_rl_links.md` L1(c); "Theorem 7" in `bridge_plan.md` §1 |
| Proposition 4 / Corollary A′ (RM+ rate) | `theory_section.md` Proposition 2 |
| Theorem B′ (uniform sampling error) | `theory_section.md` Proposition 3 |
| Theorem C, Corollary C.1 (hidden information) | `theory_section.md` "general case" and "headline bound"; `bridge_plan.md`'s "Prop K" is not defined in any source file read here and is taken to mean Theorem C / Corollary C.1 |
| Proposition L2 (safe Hedge) | "L2 safe Hedge" in `bridge_plan.md` |
| Proposition U-fin | the verifier's finite-$N$ inequality (new here) |
| Hypothesis H4 | the probe's "exploration" explanation of the engine's root gap (new name here) |

### 0.4 Status vocabulary

**Lean**: the statement's core is machine-checked in `lean/FullGame/` (the gap between the Lean statement and the
game model is stated). **proved**: a complete proof exists here or at the pointer. **proved with named hypothesis**:
complete proof given a hypothesis that is stated by name and not itself proved. **sketch**: proof idea with a named
gap. **conjecture**: believed, not proved; evidence given. **numeric**: brute force, simulation or measurement only.
**cite**: known result; we use it without re-proof. "Measured (probe level, n small)" in §3.1 is **numeric** with small $n$, not a paper number.

Level tags M0–M4 (§1). View tags: **L6** (single algorithm), **L7** (dynamics), **S** (structural: perturbation,
telescoping and measurement results that belong to neither view; forcing them into a view would overstate them).

---

## 1. Levels and views

### 1.1 Levels

| Level | Object | Where the error terms enter |
|---|---|---|
| **M0** | one exact matrix $A$ (Lecture 7's setting) | solver/dynamics gap only |
| **M1** | one estimated stage matrix $\hat Q\approx Q^*$ | + matrix error $\varepsilon_v$ (A7) |
| **M2** | full-information finite-horizon stochastic game (poke-engine, A6) | + composition over $\bar H$ decision nodes |
| **M3** | hidden information: Bayesian-LP solve over $K$ sampled worlds | + sampling, belief error (A2), non-locality (A3), prior (A4) |
| **M4** | real Showdown | + simulator mismatch $\varepsilon_{\rm sim}$ (A5); measured only so far |

Pure online-learning results with no game in them (Lemma H, R1, R2a) are tagged **M0** and listed where they are
applied (M2).

### 1.2 The two course views

**L6, single algorithm (Lecture 6).** One learner, an arbitrary opponent, a regret guarantee against fixed
comparators. The adversary model is Lecture 6 §2.1 and §2.3: losses "may depend on everything that happened earlier,
but not on our random choice for the current day"; the adversary "may know our algorithm, and hence our distribution
… but it does not know which random choice was made". That is assumption A1 below. Lecture 6's guarantee for MWU,
$L_{\rm MWU}\le(1+\eta)L^*+\ln n/\eta$, is the template for Lemma H, R2a and L2. Lecture 6 §2.3 (deterministic
algorithms suffer linear regret) is the template for R1 and is the analogy, not an instance, behind U4.

**L7, dynamics (Lecture 7).** All players learn; the object is the joint outcome. In a zero-sum game with both players
on full-information no-regret dynamics, Lecture 7 §2.2 gives
$$\max_x x^\top A\bar y-\min_y\bar x^\top Ay\le\epsilon_R+\epsilon_C .$$
In a general game the time-average joint play is an $\epsilon$-CCE (§3.2); its last sentence notes that for the
*sampled* actions "the analogous statement requires an additional expectation or a high-probability guarantee", which
is the role of Lemma D. Proposition L4 is the §2.2 inequality as an identity, with full-information regrets and bandit
feedback; B3 measures what the feedback model costs; U explains when decoupled UCB1 breaks the §2.2 argument.

---

## 2. Assumptions (numbered once)

**A1 (adversary model).** At every live history $h_t$ the opponent's action is drawn as $b_t\sim\sigma(h_t)$, where
$\sigma(h_t)$ may depend on the whole public history (including our past realised actions, the game index, and our
current mixture $p_t$, which is a function of $h_t$ and our known algorithm), and
- (A1.i) $a_t\perp b_t\mid h_t$ (simultaneous moves; no shared randomness), and
- (A1.ii) the law $\sigma(h_t)$ does not react to the current realised $a_t$: the opponent may know our mixture, not
  our sample.

L0 and everything built on it (Theorem A, L1, L2, L6, E2, P1) uses (A1.i); (A1.ii) is the Lecture 6 reading that
makes (A1.i) hold when our sample is drawn with private randomness. An opponent that sees $a_t$ breaks L2(b) in
simulation (`search_rl_links.md:77`). P1 additionally needs $\sigma$ to be the *same* strategy in every game (named
P1.2 there).

*Matrix game at every node (Lecture 7's setting at each stage).* One-player nodes (forced switches, waiting) are
$n\times1$ or $1\times m$ matrix games, so the one-step lemma and Theorem A still apply. The exception is U-turn/Volt
Switch pivots: poke-engine exposes the opponent's committed move as a state field, while Showdown hides it until it
runs. So full-information (M2) results and the oracle give the pivoting side information it does not have on Showdown.
This is covered by A11 at M3/M4, and `hidden.py` resamples the pending move (`game_model.md` §5, §6 item 3).

**A2 (non-revealing opponent).** In the M3 model (`full_game.md` §6.2), $\sigma_\theta(h)(b)$ does not depend on the
private type $\theta$ among types for which $b$ is legal. Then the true posterior equals the hard-evidence belief. It
holds trivially under full information.

**A3 (world-only values).** The continuation value after the lookahead depends on the true position, not on beliefs:
$V_{t+1}(s',b')=\sum_\theta b'(\theta)V_{t+1}(s',\delta_\theta)$. Then $\eta_{\rm NL}\equiv0$ and concealment has
no value.

**A4 (exact prior).** Opponent types are drawn from the prior the agent samples from. In the engine this is only
partly exact (`game_model.md` §6 item 2). Within a species the set posterior is exact. For unrevealed slots,
`Belief.sample_unrevealed` is NOT the true conditional of `randbats.sample_team` (toy check: true 3/8, 3/8, 1/4 vs the
sampler's 1/3, 1/3, 1/3; size at 509 species unmeasured). `Belief` uses only legality evidence (revealed moves, tera,
item gone), no damage likelihoods, so Theorem C's TV term is not zero even under A2. On Showdown both E1 arms use
Foul Play's count-weighted prior, which is a third prior; its distance to Showdown's generator is unmeasured.

*Crosswalk:* `theory_section.md`'s (A1), (A2), (A3) are A2, A3, A4 here; `game_model.md` §5's new A10, A11, A12 are the assumptions below the A9 entry.

**A5 (simulator fidelity).** For each stage $t$ let
$\varepsilon_{{\rm sim},t}=\sup_{s,a,b}\mathrm{TV}\big(P_{\rm pe}(\cdot\mid s,a,b),P_{\rm sd}(\cdot\mid s,a,b)\big)$,
the worst one-step mismatch between poke-engine's transition kernel and Showdown's, and
$$\varepsilon_{\rm sim}:=\sum_{t<H}\varepsilon_{{\rm sim},t}\cdot\operatorname{span}(V^*_{{\rm pe},t+1}),\qquad\operatorname{span}(V)=\max V-\min V\le\Delta .$$
A5 is the assumption that $\varepsilon_{\rm sim}$ is small. It is its **own term**, compounded over the horizon by the
simulation lemma (Proposition SIM, §7). It is not a leaf error: a kernel mismatch changes $Q^*$ itself, not our
estimate of it. **Resolution:** the TV is taken between the kernels pushed forward to a coarse partition of states
(KO events, or HP bins); at exact-state resolution $\varepsilon_{{\rm sim},t}\ge0.89$ per damaging move and the bound is
vacuous (§3.1). The coarsening adds a within-cell term $\omega$ (oscillation of $V$ within a cell; Proposition SIM, §7).

**A6 (finite horizon with cap).** Stages $t<H$; terminal states are absorbing with payoff $u\in[-1,1]$; a game still
live at $H$ pays $u_H\equiv0$ (a draw at the cap). The cap counts decision nodes, not turns: $H=300$ nodes in our
engine loop, and a draw is scored 0 at the cap. Showdown declares a tie only after turn 1000 (checked from turn 100
on), and Endless Battle Clause is not in gen9randombattle; our Showdown harness has no 300 cap. The form of A6 (finite
horizon, draw $=0$ at the cap) survives; the number and the unit differ by level (`game_model.md` §1.6, §6 item 1).
Backward induction is then exact. Without a cap, stage-wise optimality is not enough (R6c).
*Draw convention:* when both sides run out simultaneously, Showdown (Gen ≥ 5) gives the win to the side whose Pokémon
fainted last; poke-engine's search makes side one lose; our engine loop scores a draw (`game_model.md` §6 item 5).

**A7 (matrix error).** At every live $(t,s)$ the matrix we solve satisfies
$\lVert\hat Q_t(s)-Q^*_t(s)\rVert_\infty\le\varepsilon_v(t,s)$ on the payoff scale; $\varepsilon_v=\sup\varepsilon_v(t,s)$.
A per-state affine fit $a_{t,s}\hat Q+c_{t,s}$ ($a>0$) may be used for Theorem A (R6a); the Hedge learner
needs one global affine fit (E2′).

**A8 (experts are fixed stage policies).** The expert set $\{1,\dots,k\}$ is fixed for the match. Expert $e$'s stage
strategy at turn $t$ is $q_{e,t}=q_e(\hat A_t)$, a fixed function of the current stage matrix (top move, RM+ mixture,
pure maximin row, ε-safe LP). P1 also allows history-dependent experts, fixed across games. **Hypothesis (E)**, used by the
safety statements: some expert $e^\star$ is $\varepsilon_s$-optimal in every $\hat A_t$.

**A9 (bounded rewards, normalisation).** Units as in §0.2. Learner rewards are
$r_{t,e}=\operatorname{clip}\big((q_{e,t}\cdot\hat A_t[:,b_t]-\ell)/\bar\Delta,0,1\big)$ with a **fixed** normaliser
$\bar\Delta$ (engine: $\bar\Delta=400$ heuristic units; L2's idealised form: $\hat A\in[-1,1]$ and $(\cdot+1)/2$). A
per-turn normaliser $\operatorname{range}(\hat A_t)$ is not allowed (counterexample, §10). Lecture 6 losses are
$1-r$; Lecture 7 matrices are rescaled to $[0,1]$; poke-engine's MCTS rewards are in $[0,1]$ via
$\operatorname{sigmoid}(0.0125(h-h_{\rm root}))$.

**A10 (menu and payoff alignment).** For every state reachable in both simulators, the action menus
$A_{\rm pe}(s)=A_{\rm sd}(s)$, $B_{\rm pe}(s)=B_{\rm sd}(s)$ and the terminal payoffs agree. Violated on measure-small
sets (`game_model.md` §5): Struggle, PP (real PP vs the engine's 16 PP), simultaneous wipe-outs. Proposition SIM is
stated under A10.

**A11 (information at pivot nodes).** At a pivot node (U-turn, Volt Switch) the switching player does not see the
opponent's committed action. True on Showdown; false in poke-engine, where the action is a state field. At M3/M4
`hidden.py` resamples the pending move; the oracle and every M2 result give the pivoting side this information.

**A12 (horizon unit).** $H$ counts decision nodes (engine `step` calls). Showdown's turn counter counts full turns, and
replacement and pivot nodes do not advance it. Results stated with $\bar H$ in decision nodes transfer to Showdown with
$\bar H$ measured in decision nodes, not turns.

---

## 3. Bridges M0 → M1 → M2 → M3 → M4

| Bridge | Theorem that crosses it | Assumptions | Status | Real-game test |
|---|---|---|---|---|
| **B01** M0→M1 | Lemma 1 (+ Lemma 2(a)) | A7 | Lean | **Only exact-endgame $\varepsilon_v$ is measured** (abstracted endgames: affine-invariant stage-matrix error $\varepsilon_v^{\rm aff}$ max 0.8–1.0, the best global affine fit is constant, $\varepsilon_v^{\rm glob}\approx1$; RESULTS "Exact endgames"). **Mid-game $\varepsilon_v$ is unmeasured**: poke-engine MCTS values are root-relative, so they are not an absolute $Q^*$ reference. What the measured exploitability *requires* is bounded below: path-average $\varepsilon_v\ge0.0041$ per decision (W1 point; §11, $\varepsilon_v$ lower bound) |
| **B12** M1→M2 | Theorem A (with L1's sandwich and L6's residual form) | A1, A6, A7 | Lean (abstract induction) + proved | Stage gaps along the exploiter's path, computed in $\hat Q$ ($Q^*$ unavailable). For sampled play they are $\approx0$ (RM+ converges), so the test **isolates the residual $2\varepsilon_v\bar H$** that must explain the measured exploitability; **it does not confirm the bound**. Probe done (§3.1): residual isolated for sampled play. Units first: exploitability $=2\times$(exploiter score $-0.5$) $=$ **0.746 / 0.426 / 0.534** for top move / sampled / safex 2% (W1 0.873, W1 0.713, SXW 0.767). Log-only rerun, about 1 CPU-h |
| **B23** M2→M3 | Theorem C; Corollary C.1; Theorem B′; S1, S2 | A2, A3, A4 | proved (C, C.1, B′, S2); proved with named hypothesis (S1) | On Showdown we control both bots, so true opponent sets are known: measure TV(Foul Play belief, truth) per turn and the oracle-vs-Bayesian LP gap in real games (from harness logs) |
| **B34** M3→M4 | Proposition SIM (simulation lemma) | A5 | sketch (standard lemma, cite) | From recorded Showdown battles: rebuild each turn with the true teams, predict the outcome distribution of the played move pair with poke-engine, score the observed outcome (calibration, HP-damage error). Probe done (§3.1): mean KO-gap $\approx0.012$ per move, 0.011–0.015 per turn at KO resolution (sup unmeasured, observed max 0.31); $\ge0.89$ at exact-state resolution |

### 3.1 Bridge measurements (status: measured, probe level, n small; leads, not paper numbers)

Source: scratch probes in
`/private/tmp/claude-501/-Users-biofanat-Documents-classes-f26-cs535/2bbe670b-2f97-4d9a-bfad-e1139e0352c6/scratchpad/bridges/`
(`README.txt`, `b12/analysis_out.txt`, `b34/final_out.txt`). Not re-verified.

**B12 (M1→M2).** The W1/SXW games were replayed from their logged seeds, 6 pairs (12 games) per target. All 36
replays reproduce the logged games exactly (winner and turn count). Per game,
$-z=\sum_t(R_t+S_t-D_t)-\varphi(v_1)$, with $R$ the realised stage gain given the exploiter's column ($R\le$ stage gap $G$), $S$ the
target's sampling term, $D$ the drift term. It is an exact identity; the closure is a code check, and it closes to 3
decimals under both units conversions used (logistic fit $P(\text{win})=\operatorname{lgs}(v/98)$, and
$\tanh(v/400)$). Caveat: for sampled play, $-\sum D=1.02$ is offset by $R=-0.50$ and $S=-0.27$ ($S$ has mean zero), so at
$n=12$ the channel split is noisy.

| Target | Exploitability on these 12 games, $2(s-\tfrac12)$ (full log) | $\sum_t g_t$ per game, range units, median | Reading |
|---|---|---|---|
| sampled | 0.250 (0.426) | **0.001** [0.000, 0.001]; exploiter's realised stage gain $\le0$ (median $-0.69$ to $-1.08$) | all measured exploitability goes through the drift channel: the $2\varepsilon_v\bar H$ residual. **Supported** |
| top move | 0.833 (0.746) | **1.95** [1.23, 3.41]; 0/597 turns mixed | the stage channel explains 0.68–1.04 of the exploitability, depending on the units conversion |
| ε-safe 2% | 0.667 (0.534) | 0.33; per-turn gap capped at exactly 0.020 | exploitation goes through drift |

Reading: deterministic play is exploited through stage gaps; mixed play is exploited through value error along the
line chosen by the exploiter's lookahead. This **isolates** the residual; it does not confirm Theorem A's constant.
Caveat: on these 6 pairs the sampled target's exploitability (0.250) is well below the full log's 0.426, so the
subsample is not representative in level. Extension (2026-11-16, verified): 284 in-engine games (36 B12 + 48 exploiter extension +
100 MCTS-2000 + 100 self-play) and 599 real-Showdown LOG games; see §11, signed pathwise identity.

**B34 (M3→M4, A5).** 21 instrumented Foul Play-vs-Foul Play games on a local Showdown server, both true teams known;
519 damage events scored against the 16-roll grid reconstructed by the bridge scorer from poke-engine's max roll (poke-engine itself uses one average roll per branch), poke-engine 0.0.48.

- Observed damage vs poke-engine's 16-roll support (519 events):

  | Definition | $\pm0$ HP | $\pm1$ HP | $\pm2$ HP |
  |---|---|---|---|
  | Inside the union $[\min,\max]$ of branch ranges | 96.5% (501/519) | 98.8% | 99.0% (514/519) |
  | Per-branch range | $\approx$96.2% | — | 98.8% (513/519) |
  | Exactly on one of the 16 roll values | 89.0% (462/519) | — | — |

  The 96.5% is zero tolerance against the range, not an exact match.
- Calibration is uniform: PIT KS $D=0.040$, $p=0.37$.
- Mean absolute error of the predicted mean: 2.05 HP% (uncensored events; median 1.38).
- KO-indicator gap $|P_{\rm pe}(\mathrm{KO})-P_{\rm sd}(\mathrm{KO})|$: mean 0.0122 over 516 events; 0.0054 is the mean over
  the 300 uncapped events (not a lower bound) and 0.0217 over the 216 capped events. The three 0.5 events are scorer
  artefacts (Focus Sash / Sturdy ignored by the 16-roll model; the engine was right); without them the mean is 0.0094,
  and the observed maximum after removing artefacts is 0.31. Per turn: 0.011–0.015. So the **mean over self-play events
  is $\approx0.012$ per move, 0.011–0.015 per turn, at KO resolution; the sup (which A5 defines) is unmeasured
  (observed max 0.31)**.
- **At exact-state resolution $\varepsilon_{{\rm sim},t}\ge0.89$** per damaging move: poke-engine applies a single
  average damage roll by design, so its point kernel matches the observed HP exactly on only 10.6% of single-value
  events. With that resolution Proposition SIM is vacuous.
- Genuine mechanic mismatches: 2 of 519 (< 0.5%): drain rounding, and a suspected crit/Unaware issue ($n=1$); counting
  these as genuine is a judgement call. Errors in rebuilding the state from Foul Play's tracker are about as common.

**Consequence for A5.** A5 must be stated at a **coarsened resolution**: the TV in $\varepsilon_{{\rm sim},t}$ is
taken between the kernels pushed forward to a coarse partition of states (KO events, or HP bins), and the simulation
lemma then carries an explicit within-cell term $\omega$ (the oscillation of $V$ within a cell; Proposition SIM, §7).
At KO resolution the measured per-turn mean mismatch (0.011–0.015) times $\bar H\approx35$ is about 0.4–0.5 in TV-sum
units before the span factor: already large, so even the coarsened bound is near vacuous over a full game; it is
informative per turn.

What is already known about transfer (empirical bridge results):

- **U does not transfer to the engine at practical budgets.** At $10^3$–$10^4$ iterations the real poke-engine root's
  joint visits are indistinguishable from the product of the marginals; the gap is unequalised exploration (H4,
  §5.2). U stays an M0, large-$N$ result.
- **The 2% ε-safe best-response bound does not survive M1→M2.** SXW: the depth-2 informed best response scores 0.767 against the 2% ε-safe best response,
  worse for us than 0.713 against sampled (W1; different seeds, unpaired). The stage-level cap bounds loss only in
  $\hat Q$; the exploiter wins through $\varepsilon_v$. The bound's ε-add-on is non-vacuous only for
  $\varepsilon\lesssim1\%$ (§6.3).
- **Leaf rescaling did not help:** LF, $\tanh(h/400)$ with exact terminals vs the current leaf, $-0.040$
  $[-0.098,+0.018]$, n.s. Lowering the TD residual (L6) did not move the win rate: $\varepsilon_v$ is not a scale
  problem.
- **Gray-box RL did not beat black-box:** 0.075 vs 0.125 win rate at 8M decisions (single runs, unpaired, about
  2.7 SE). Knowing the opponent's move distribution without a payoff model was not enough (§6.5).

---

## 4. Structural results (view S)

### 4.1 M1: one estimated matrix

**Lemma 1 (one step).** [M1; S; **Lean**] *Assumption:* A7 at one stage. *Statement.* For all $A,B\in\mathbb R^{n\times m}$
with $\lVert A-B\rVert_\infty\le\varepsilon$ and every $x$ that is $\varepsilon_s$-optimal in $B$:
$\min_y x^\top Ay\ge\operatorname{val}(A)-\varepsilon_s-2\varepsilon$. The factor 2 is tight ($2\times1$ instance).
*Proof idea:* bilinear forms move by at most $\varepsilon$; apply twice (value, then guarantee). Steps:
`full_game.md:34–38`; Lean `lemma1_one_step`, `lemma1_mixed`, tightness `example`.
*Real-game test:* B01 (exact endgames only).

**Lemma 2 (Lipschitz, lookahead, no contraction).** [M1–M2; S; (a) **Lean**, (b)–(d) **proved**] (a) $|\operatorname{val}A-\operatorname{val}B|\le\lVert A-B\rVert_\infty$
(Lean `val_lipschitz`, `exists_val`). (b) Chance averaging is non-expansive. (c) Exact depth-$d$ lookahead with leaf
error $\varepsilon_{\rm leaf}$ gives $\varepsilon_v\le\varepsilon_{\rm leaf}$ for every $d\ge1$. (d) The bound in (c) is
attained for every $d$ in games where no transition before stage $t+d$ reaches a terminal state. `full_game.md:40–46`.

**Lemma R6a (affine invariance; affine-invariant Theorem A).** [M1–M2; S; **proved**, numeric check] For $a>0$,
`rm_plus(aM+c)` returns the strategies of `rm_plus(M)`; Theorem A holds with per-turn error
$a_{t,s}\varepsilon_s+2\varepsilon_v^{\rm aff}(t,s)$; as $a\to0$ the per-turn error $2\varepsilon_v^{\rm aff}$ tends to the trivial $\operatorname{range}(Q^*)$ ($\varepsilon_v^{\rm aff}$ itself tends to the error of the best constant fit, half the range).
`regret_theory.md:403–420`.

**Lemma R6b (inversions; exact terminals).** [M1; S; **proved**, numeric] (1) $\varepsilon_v^{\rm aff}(M,Q)\ge\tfrac12\max\{Q_i-Q_j:M_i\le M_j\}$;
(2) the best affine fit is constant iff some argmax-$Q$ cell is scored at or below some argmin-$Q$ cell; (3) with exact
terminal values, $|\hat Q-Q^*|\le\pi_{\rm live}\,\varepsilon_{\rm live}$ cellwise. Explains the constant best fit on all six exact
endgames (one mis-scored won terminal suffices). `regret_theory.md:422–460`. *Scope:* "exact terminals" holds only for `squash_matrix` and `vnet_matrix`; `nash.full_matrix` scores terminal children with the heuristic evaluation (`game_model.md` §6 item 6), so part (3) applies only to those.

### 4.2 M2: composition over the game

**Lemma L0 (potential telescoping).** [M2; S; **proved**, brute force] *Assumptions:* A1(i), A6. *Statement.* For every
real $\Phi_t(h_t)$ with $\Phi=u$ on terminals and $\Phi_H=0$, and all history-dependent $\pi,\sigma$:
$\Phi_0-V(\pi,\sigma)=\mathbb E_{\pi,\sigma}\big[\sum_{t<\tau}(\Phi_t(h_t)-x_t^\top Q[\Phi]_t(h_t)y_t)\big]$.
*Idea:* telescope on each path; A1(i) turns $\mathbb E[\Phi_{t+1}\mid h_t]$ into $x_t^\top Q[\Phi]_ty_t$; tower rule.
`search_rl_links.md:37–45`. Theorem A, L1, L6, L3 and P1 are L0 with potentials $V^*$, $\hat V$, $V^{\pi,\sigma_k}$, $V^{e,\sigma}$.

**Proposition 0 (value).** [M2; S; **proved**] $V^*_0(s_0)$ is the value of $G$ under A6 (Shapley 1953, cite for the
name). `full_game.md:28`.

**Theorem A (per-turn composition).** [M2; S; **Lean** (abstract induction) + **proved** (model link by reading)]
*Assumptions:* A1(i), A6, A7; a Markov agent playing at each live $(t,s)$ an $\varepsilon_s(t,s)$-optimal strategy of
$\hat Q_t(s)$. *Statement.* For every history-dependent opponent $\sigma$,
$$W_0^{\pi,\sigma}(s_0)\ge V^*_0(s_0)-\mathbb E_{\pi,\sigma}\Big[\sum_{t<\tau}\delta(t,s_t)\Big],\qquad\delta=\varepsilon_s+2\varepsilon_v ,$$
hence
$$\operatorname{expl}(\pi)\le\mathbb E_{\pi,\sigma_{\rm BR}}\Big[\sum_{t<\tau}\delta(t,s_t)\Big]\le\bar\delta\,\bar H\le H(\varepsilon_s+2\varepsilon_v).$$
$\bar H$ (length against a best response) cannot be replaced by the length under *some* equilibrium opponent, under
self-play, or under typical opponents (counterexample with $\operatorname{expl}=H\delta$ while an equilibrium
opponent ends at once; equilibrium opponents need not be unique). *Proof idea:* backward induction on histories;
Lemma 1 at each node against whatever $y=\sigma(h_t)$ is played. `full_game.md:50–69`; Lean `backward_induction`,
`deficit_le_sum`, `theoremA_abstract`. Hedge-learner version (history-dependent agent): the identity in E2 step (i).
*Real-game test:* B12 (§3.1).

**Theorem L1 (identity and sandwich).** [M2; S; **proved**, brute force] *Assumptions:* A1(i), A6. (a)
$V^*_0-V(\pi,\sigma)=\mathbb E[\sum_{t<\tau}\ell_t]$ exactly, so $\operatorname{expl}=\sup_\sigma\mathbb E[\sum\ell_t]$.
(b) $\mathbb E_{\pi,\sigma_g}[\sum e_t]\le\operatorname{expl}(\pi)\le\mathbb E_{\pi,\sigma_{\rm BR}}[\sum e_t]$, with $e_t$ the stage
exploitability in $Q^*$ and $\sigma_g$ the myopic $Q^*$-exploiter; the gap can approach $\Delta$ (Example L1b, gap
$2-\delta$). (c) Theorem A as a corollary. `search_rl_links.md:57–70`.

**Theorem L5 (depth with exact terminals; applies to `squash_matrix`/`vnet_matrix` only, not `nash.full_matrix`, which scores terminal children with the heuristic evaluation).** [M2; S; **proved**, brute force, bound attained] $|V^D_t-V^*_t|\le\varepsilon_{\rm leaf}\,\rho_D(t,s)$
with $\rho_D$ the cooperative survival probability; Theorem A holds with per-turn error $\varepsilon_s+2\varepsilon_{\rm leaf}\rho_D$.
Mid-game $\rho_D\approx1$: depth alone does not shrink the leaf term. `search_rl_links.md:193–215`.

**Theorem L6 (minimax TD residual).** [M2; S; **proved**, brute force] *Assumptions:* A1(i), A6; leaf $\hat V$ with exact
terminals (`squash_matrix`/`vnet_matrix` only), agent $\varepsilon_s$-optimal in $Q[\hat V]$. *Statement.*
$\operatorname{expl}(\pi)\le\mathbb E_{\pi,\sigma_{\rm BR}}[\sum(\delta_t+\varepsilon_s)]-\mathbb E_{\pi^*,\hat\sigma}[\sum\delta_t]\le\sum_{t<H}(2\sup_s|\delta_t|+\varepsilon_s)$,
$\delta_t=\hat V_t-\operatorname{val}Q[\hat V]_t$. Incomparable with Theorem A (examples both ways); mis-scored terminals (the terminal term applies only to `squash_matrix`/`vnet_matrix`, not `nash.full_matrix`)
add $\mathbb E_{\pi,\sigma_{\rm BR}}[m]-\mathbb E_{\pi^*,\hat\sigma}[m]$ (note the minus). `search_rl_links.md:219–246`.
*Measured (engine, diagnostic):* mean $\delta^2$ minimal at $\tanh(h/k)$, $k\approx320$–$400$, 1.7× lower than $k=160$; the
bound is vacuous (median $2\sum|\delta_t|=4.2$ per game). *Real-game test:* LF (played $k=400$): $-0.040$, n.s.

**Proposition R6c (non-termination).** [M2; S; **proved**] In an uncapped undiscounted game, a stationary agent with
per-turn error $\delta\equiv0$ in the true $V^*$ can have exploitability 0.3; with the cap (A6) Theorem A holds with
equality and charges the loss to the last turn; residual form for fixed points. Engine instance: exploitability
0.029 in an abstracted endgame loop. `regret_theory.md:462–510`.

**Corollary R4 (normalised-stage-payoff-to-true transfer).** [M1–M2; S; **proved**, numeric] Under A7 (affine-invariant),
trajectory sums of stage payoffs move by at most $\sum_t\varepsilon_v(t)$ (one strategy) or $2\sum_t\varepsilon_v(t)$
(differences, values). With the best fit constant on every exact endgame, the transfer bound is vacuous. `regret_theory.md:297–321`.

**Propositions R5a–c (exploitability bracket).** [M2/M4; S; **proved**] (a) any fixed pre-registered exploiter gives
$\operatorname{expl}\ge\bar Z-\tfrac12-\sqrt{\ln(m/\vartheta)/(2n)}$ (score units; $m$ exploiters tried, $n$ pairs);
fresh seeds remove the $\ln m$. (b) $\mathrm{gain}\le\operatorname{expl}(\pi)+\operatorname{expl}(\beta)$. (c) bracket with
Theorem A's upper side, which is vacuous for current leaves. `regret_theory.md:325–397`. *Measured (engine):* W1
95% lower confidence bounds 0.250 (top move), 0.090 (sampled); SXW 0.145 (ε-safe 2%), all score units, one exploiter each.

### 4.3 M3: hidden information

**Theorem B (one-stage decomposition).** [M3; S; **proved**, tight example, numeric] *Assumptions:* finite type set; common
entry interval of length $\Delta$; A7 per world. *Statement.* $v(b)-f_b(\hat x)\le\varepsilon_s+2\Delta\,\mathrm{TV}(b,\hat b)+2\varepsilon_v+2S$
($2\Delta\mathrm{TV}=4\,\mathrm{TV}$ on $[-1,1]$); $2\Delta\mathrm{TV}$ is tight. `full_game.md:95–112`.

**Theorem B′ (uniform sampling error).** [M3; S; **proved**] For $K\ge9$, $n=|A|$:
$\mathbb ES\le\Delta\sqrt{2(n\log K+\log2)/K}$, and the high-probability form. Vacuous unless $K/\log K\gtrsim2n$
(measured $\mathbb ES=0.054\Delta$ vs bound $0.91\Delta$ at $n=3$, $K=25$, synthetic). `full_game.md:118–128`.

**Proposition S1 (fixed prior: exponential rate).** [M3; S; **proved with named hypothesis** (S1.2)]
*Assumptions:* (S1.1) finitely many types, $|\Theta|=n_\Theta$; (S1.2) there are a linearity cell $C$ of $v$ and
$\gamma>0$ with **$\gamma\le\min_\theta b_\theta$** such that the TV ball $\{b':\mathrm{TV}(b',b)<\gamma\}\cap\mathcal P(\Theta)$
lies in the relative interior of $C$. The agent plays **any** $\hat x\in\arg\max_x f_{\hat b_K}(x)$, ties included.
*Statement.* For every $K\ge1$: $\mathbb E[L_K]\le\Delta\,\Pr(\mathrm{TV}(\hat b_K,b)\ge\gamma)\le\Delta\,(2^{n_\Theta}-2)\,e^{-2K\gamma^2}$.
*Proof.* Idea: inside the relative interior of one cell every optimiser for $\hat b$ is optimal for $b$.
1. $v(b')=\max_{(x,s)\in P}b'^\top s$ over a polyhedron, so $v$ is convex piecewise affine with finitely many cells.
2. $\gamma\le\min_\theta b_\theta$ makes every $b'$ in the ball fully supported (one coordinate can drop by at most
   the TV), so the ball lies in the relative interior of the simplex.
3. Let $b'$ be in the ball and $x'$ any optimiser of $f_{b'}$ (ties included). The affine function
   $b''\mapsto v_C(b'')-f_{b''}(x')$ is $\ge0$ on $C$ and vanishes at the relative-interior point $b'$, so it vanishes
   on $C$; at $b''=b$ this says $f_b(x')=v(b)$.
4. So $L_K=0$ on $\{\mathrm{TV}(\hat b_K,b)<\gamma\}$ and $L_K\le\Delta$ always. The multinomial $L_1$ bound
   $\Pr(\lVert\hat b_K-b\rVert_1\ge\epsilon)\le(2^{n_\Theta}-2)e^{-K\epsilon^2/2}$ (Weissman et al. 2003; **constant recalled,
   not checked against the paper**) with $\epsilon=2\gamma$ gives the claim. $\square$

*Counterexample without $\gamma\le\min_\theta b_\theta$:* $M_1=[[1],[0]]$, $M_2=[[1],[1]]$, $b=P(\text{type }1)=0.05$.
$v\equiv1$ is one cell, but $\hat b_K=0$ (all samples type 2, probability $0.95^K$) makes every $x$ optimal, and the
tie choice $x=e_2$ loses $0.05$. *Limitation:* the engine's set prior is a near-continuum, where the $2^{n_\Theta}$
constant is vacuous. `new_directions.md:64–82` (superseded there by this form).

**Proposition S2 (uniform over priors: $\Omega(K^{-1/2})$).** [M3; S; **proved** (last step by the CLT)] *Instance:*
two types, $M_1=[[1,0],[0,\tfrac14]]$, $M_2=[[\tfrac14,0],[0,1]]$, $\Delta=1$, $v(b)=\max(0.2-0.15b,0.05+0.15b)$ with one
kink at $b_0=\tfrac12$ and slope jump $s=0.3$. *Statement.* For every $c>0$, with $b_K=\tfrac12+c/\sqrt K$:
$\liminf_K\sqrt K\sup_b\mathbb E[L_K]\ge0.3\,c\,\Phi(-2c)$; at $c=\tfrac14$ this is **0.0231**, and the maximiser
$c^*=0.376$ gives **0.0255**. The liminf does not depend on how ties $\hat b_K=\tfrac12$ are broken.
*Proof.* Idea: a prior $c/\sqrt K$ past the kink is misclassified with constant probability. When $b>\tfrac12$ and
$\hat b<\tfrac12$ the unique optimiser is $x_L$ and $L=s(b-\tfrac12)$; the CLT gives
$\Pr(\mathrm{Bin}(K,b_K)<K/2)\to\Phi(-2c)$. $\square$ `new_directions.md:84–107`.
*Exact finite-$K$ values* (binomial sums, computed for this map; $\sqrt K\,\mathbb E[L_K]$ at $b_K$; favourable tie-break
counts $\hat b=\tfrac12$ as no loss, adverse as a loss):

| $K$ | 10 | 50 | 100 | 500 | 2000 | 4000 | limit |
|---|---|---|---|---|---|---|---|
| $c=\tfrac14$, favourable | 0.0153 | 0.0195 | 0.0205 | 0.0220 | 0.0226 | 0.0227 | 0.0231 |
| $c=\tfrac14$, adverse | 0.0315 | 0.0269 | 0.0258 | 0.0243 | 0.0237 | 0.0236 | 0.0231 |
| $c=0.376$, favourable | 0.0154 | 0.0208 | 0.0221 | 0.0240 | 0.0247 | 0.0250 | 0.0255 |
| $c=0.376$, adverse | 0.0361 | 0.0303 | 0.0289 | 0.0270 | 0.0263 | 0.0260 | 0.0255 |

The source's numeric table equals the adverse-tie-break row; it is not a property of the instance (§10).

**Theorem C (hidden information, full game, five terms).** [M3; S; **proved**] *Assumptions:* A1(i), A6, A7, A4
(exact prior $b_0$); one-sided private information; fresh unobserved samples. *Statement.* For every type-informed
history-dependent $\sigma$, $\operatorname{expl}(\pi)\le\sup_\sigma\mathbb E[\sum_{t<\tau}\delta_C(t,h_t)]$ with
$\delta_C=\varepsilon_s+2\varepsilon_v+2\mathbb ES_t+\eta_{\rm NL}(t)+2\Delta\,\mathrm{TV}(p_h,b^{\rm hard}_h)$.
`full_game.md:164–175`.

**Corollary C.1 (headline bound).** [M3; S; **proved**] Under A2 (TV term $=0$), A3 ($\eta_{\rm NL}=0$), A4 and the RM+ and
$K$-sample rates, with $\Delta=2$:
$$\operatorname{expl}(\pi)\le\bar H\Big(8\sqrt{n/T}+2\varepsilon_v+4\sqrt{2n\log K/K}\Big).$$
Against arbitrary (revealing) opponents the TV term is necessary: three-team Example C.1 (repaired) has
$\operatorname{expl}=1$ while $\sum_t\eta_{\rm NL}=3/7$ and all other terms are 0. At least 3 teams are needed at $H=2$.
`full_game.md:176–186`; `theory_section.md:52–60`. Two-sided information adds the value of concealing our team once
(`full_game.md:188`).

**PIMC-averaging loss bound (PIMC averaging vs Bayesian LP, one stage).** [M3; S; **numeric** ($4\delta_H$), **sketch** ($2\delta_H$)] The Bayesian LP is the maximin LP of PokaiTrainer's Bayesian matrix game, the one-sided special case of Ponssard and Sorin's (1980) LP for zero-sum games with incomplete information ("Bayesian matrix game" is PokaiTrainer's name). PIMC averaging solves each sampled world as if it were known and plays the $\beta$-weighted average of the per-world equilibria. PIMC-averaging
loss $\le4\delta_H$ held on all 46,890 logged decisions (max 0.098 of the bound); a hand argument gives $2\delta_H$, not
yet through `prove`. RESULTS "PIMC-averaging loss vs the bound" and "Relation to prior work".

**Conjecture ($\eta_{\rm NL}$ + TV floor).** [M3; S; **conjecture**] $\sum_t(\eta_{\rm NL}(t)+4\,\mathrm{TV}_t)$ along typical
play is of the order of the measured oracle gap ($\approx0.1$ score). Test: B23 on Showdown.

---

## 5. Dynamics results (view L7)

### 5.1 M0: one exact matrix

**Lecture 7 theorem.** [M0; L7; **cite** (Lecture 7 §2.2, §3.2)] Full-information no-regret self-play on $A\in[0,1]^{n\times m}$:
$\max_xx^\top A\bar y-\min_y\bar x^\top Ay\le\epsilon_R+\epsilon_C$; in general games the time-average joint play is an
$\epsilon$-CCE. **Zero-sum CCE** [cite]: its marginals are maximin and its deviation gain equals the external regret,
so CCE adds nothing beyond Nash here except as a correlation diagnostic (Roughgarden, *Twenty Lectures*).

**Proposition 4 / Corollary A′ (RM+ as coded).** [M0–M1; L7; **proved**, key step **Lean**] RM+ with alternating updates
and linear averaging, exactly as `nash.rm_plus`: $\varepsilon_s\le\text{gap}\le2\Delta(\sqrt n+\sqrt m)/\sqrt T$.
*Idea:* Lecture 7's two inequalities with weighted regrets (Lemma 3, $\lVert Q_T\rVert_2\le\Delta\sqrt{nT}$) plus a cross
term from alternation that is $\le0$ term by term (Burch–Moravčík–Schmid 2019 Thm 5; Lean `rmplus_improvement`).
`full_game.md:73–91`, `CHECK.md` finding 2. *Measured:* gap $\approx T^{-1.3}$ on 35 real stage matrices (median slope
$-1.26$; $-1.24$ to $-1.39$ across seeds; median gap $7.0\cdot10^{-5}$ at $T=1000$); gap $\le0.04$ of the bound on real
matrices; 0.12–0.20 of it on synthetic matrices; max ratio 0.123 (code) / 0.172 (simultaneous) over 1,920 cases
(`CHECK.md`). **Cite-only companion:** Lecture 7 with RM+ weighted averaging is BMS19 Thm 5; alternation breaks the
plain identity (BMS19 Observation 1: both regrets 0, average exploitability 0.5).

**Proposition L4 (exact folk identity for visit shares).** [M0; L7; **proved**] At a node with payoff matrix
$\hat A\in[0,1]^{n\times m}$ and any selection rule,
$\operatorname{DualityGap}(\bar x_N,\bar y_N)=(\mathrm{Reg}^1_N+\mathrm{Reg}^2_N)/N$ with full-information (mean-reward) regrets.
Lecture 7 §2.2 as an identity. Convergence iff the *sum* is $o(N)$ (one side can be negative). `search_rl_links.md:149–155`.

**Proposition L4b (UCB1 lockstep on symmetric games).** [M0; L7; **proved**] Deterministic rewards,
$\hat A(a,b)+\hat A(b,a)=1$, same option order: decoupled UCB1 plays round-robin, so shares stay uniform even with a
dominant row (gap 0.4 on $[[.5,.9],[.1,.5]]$). Knife-edge: noise breaks it. `search_rl_links.md:180–189`.

**Setting U** (`new_directions.md` §3). $A\in[0,1]^{n\times m}$; both sides run UCB1 on their own realised rewards
$r^1_t$ ($\mathbb E[r^1_t\mid i_t,j_t,\text{past}]=A_{i_tj_t}$) and $r^2_t=1-r^1_t$; $\hat\mu^1_i(N)$ is row $i$'s realised
conditional mean; $J_N$ the joint visit distribution; $G_N$ the share gap.

**Proposition U1 (equalisation of realised means).** [M0; L7; **proved with named hypothesis** (U1.2)] *Assumptions:*
(U1.1) $J_{N_s}\to J$ along $N_s\to\infty$, and rows $i,k$ have $J_{i\cdot},J_{k\cdot}>0$. (U1.2) *Regularity, for both
arms:* the last play of $i$ before $N_s$, $\tau^i_s$, satisfies $N_s-\tau^i_s=o(N_s)$, and the same for $k$.
*Statement.* $\lim_s(\hat\mu^1_i(N_s)-\hat\mu^1_k(N_s))=0$; likewise for columns.
*Proof.* Idea: at $i$'s last play its index beat $k$'s, and both bonuses vanish.
1. At $\tau=\tau^i_s$: $\hat\mu_i+\sqrt{2\ln\tau/n_i}\ge\hat\mu_k+\sqrt{2\ln\tau/n_k}$, so
   $\hat\mu_i(\tau-1)\ge\hat\mu_k(\tau-1)-\sqrt{2\ln\tau/n_i(\tau-1)}$ (**$i$'s** bonus is subtracted).
2. By (U1.1) and (U1.2) for $i$, $n_i(\tau-1)\ge(J_{i\cdot}-o(1))\tau$ and $\tau\sim N_s$, so the bonus is
   $O(\sqrt{\ln N_s/N_s})\to0$.
3. From $\tau$ to $N_s$, $\hat\mu_i$ moves by $\le1/n_i$ and $\hat\mu_k$ by $\le o(N_s)/((J_{k\cdot}-o(1))N_s)$, using
   (U1.2) for $i$. So $\liminf(\hat\mu_i-\hat\mu_k)\ge0$.
4. Exchanging $i$ and $k$ uses (U1.2) for $k$ and gives the limit. $\square$

Without (U1.2) only the one-sided inequality of step 1 holds, at $i$'s play times. *Numeric support for (U1.2):* the
irregularity statistic $(N-\tau)/N$ falls roughly like $N^{-1/2}$. The realised-mean spread equals the bonus
difference to 4 decimals in simulation.

**Corollary U2 (independence ⇒ Nash).** [M0; L7; **proved with named hypothesis** (U1.2 for all arms)] *Assumptions:*
(U1.1), (U1.2) for every row and column, full-support limit, and $J=\bar x\bar y^\top$. *Statement.* $(\bar x,\bar y)$ is
a Nash equilibrium; $G_{N_s}\to0$. *Idea:* the martingale LLN makes $\hat\mu^1_i$ track the conditional mean
$\sum_jJ(i,j)A_{ij}/J(i,\cdot)$; it needs only a conditional mean, so deterministic payoffs are included (the
difference is then 0). With a product limit that is $(A\bar y)_i$; U1 equalises it over rows, and likewise columns:
mutual indifference on full supports. `new_directions.md:165–174`.
*Consequence (gap ⇒ correlation)* [**proved with named hypothesis** (U1.2), contrapositive of U2]: under U2's other hypotheses, $\lim G_N>0$ forces
$J\ne\bar x\bar y^\top$, i.e. $\mathrm{TV}(J,\bar x\bar y^\top)>0$. *The converse fails* [**numeric**]: on RPS,
$\mathrm{TV}$ 0.2–0.67 while the gap tends to 0.

**Proposition U-fin (finite-$N$ inequality).** [M0; L7; **proved**; numeric: 199,652 random $(J,A)$ pairs plus 300 adversarial optimisations, maximum violation $2.2\cdot10^{-16}$; 180/180 bandit checkpoints]
*Statement.* For every $N$ with all shares positive,
$$G_N\le\operatorname{spread}(c)+\operatorname{spread}(c')+\mathrm{TV}(J_N,\bar x_N\bar y_N^\top)\Big(\frac1{\min_i\bar x_i}+\frac1{\min_j\bar y_j}\Big),$$
where $c_i=\sum_jJ_{ij}A_{ij}/\bar x_i$ and $c'_j=\sum_iJ_{ij}A_{ij}/\bar y_j$ are the rows' and columns' conditional mean payoffs under $J_N$, both in row-payoff units, and $\operatorname{spread}=\max-\min$.
The row-wise form, with $\max_iD_i/(2\bar x_i)+\max_jD'_j/(2\bar y_j)$ in place of the TV terms ($D_i=\sum_j|\bar x_i\bar y_j-J_{ij}|$, $D'_j=\sum_i|\bar x_i\bar y_j-J_{ij}|$), is tighter still.
*Proof.* (1) $|(A\bar y)_i-c_i|\le D_i/(2\bar x_i)\le\mathrm{TV}/\bar x_i$: with
$(A\bar y)_i-c_i=\sum_j(A_{ij}-\tfrac12)(\bar x_i\bar y_j-J_{ij})/\bar x_i$, centring at $\tfrac12$ is free because each row of
$\bar x\bar y^\top-J$ sums to 0; columns likewise. (2) $\mu=\sum_i\bar x_ic_i=\sum_j\bar y_jc'_j$ (both are the realised average payoff), so
$\max_ic_i-\min_jc'_j\le\operatorname{spread}(c)+\operatorname{spread}(c')$. (3) Combine the bounds on $\max_i(A\bar y)_i$ and $\min_j(\bar x^\top A)_j$. $\square$ *Remark:* under deterministic
payoffs $c$ equals the realised means $\hat\mu$; under stochastic payoffs they differ by martingale noise (U2's step).

**Conjecture U3 (plateau with Bernoulli payoffs).** [M0; L7; **conjecture**, numeric] On
$A=[[1,0],[0,\tfrac12]]$ with Bernoulli rewards, $\liminf_NG_N>0$ a.s. Evidence: plateau 0.04–0.09, seed-dependent;
6/6 seeds flat over $N=10^4$–$10^6$. Some mixed games converge for some seeds; pure saddles always converge.

**Observation U4 (deterministic payoffs).** [M0; L7; **numeric**] With deterministic rewards on mixed $2\times2$ games
the gap grows with $N$, and play settles into a **correlated full-support mixture**, not lock-in to a pure cell. An
analogy to Lecture 6 §2.3 (deterministic maps from history suffer linear regret), not an instance.

**Corollary B3 (bandit vs full feedback at the decision node).** [M0–M1; L7; L4 part **proved**, rest **cite** + **numeric**]
*Statement.* (a) For any two selection rules sampling independently given the history (A1(i)), the visit-share gap is
exactly $(\mathrm{Reg}^1_N+\mathrm{Reg}^2_N)/N$ (L4). (b) The bandit high-probability bounds (Exp3-IX, Neu 2015; Exp3.P,
Auer et al. 2002) are $\ge1$ range at $k=9$, $N=2000$, $\vartheta=0.05$ (Exp3-IX 1.04, Exp3.P 1.58; at $\vartheta=0.1$ they are 0.95 and 1.47): **vacuous** (the Exp3.P constant is from Bubeck & Cesa-Bianchi 2012 Thm 3.2). (c) The full-information tuned Hedge bound,
$2\sqrt{\ln k/(2\cdot1000)}=0.066$ range at $T=1000$ (two players, tuned regret $\sqrt{T\ln k/2}$ each), is **not vacuous** (measured Hedge average gap 0.032). (d) Measured on
real logged stage matrices (normalised to range 1): Exp3 with forced exploration 0.14–0.15, plain Exp3 $\approx0.105$,
UCB1 0.12–0.125, RM+ (1000 it, full information) 0.0000; gap $>0.05$ on about 80% of **two-sided** decisions (74% of
all turns). *Reading:* the feedback model costs 0.10–0.15 of stage gap at MCTS budgets, depending on the bandit
algorithm. L1(b) turns stage gaps into a full-game *lower* bound along the myopic exploiter. `dynamics_directions.md`
§2.4 (numbers superseded by these). *Real-game test:* E4 (§8).

**Lemma D (realised joint play).** [M0; L7; **proved**] *Assumption:* $a_t\sim x_t$, $b_t\sim y_t$ independent given
$h_t$ (A1(i)). *Statement.* For all $\vartheta\in(0,1)$, with probability $\ge1-\vartheta$,
$\max_{ij}|\hat J_T(i,j)-\bar J_T(i,j)|\le\sqrt{\ln(2nm/\vartheta)/(2T)}$, so
$\mathrm{TV}(\hat J_T,\bar J_T)\le\tfrac{nm}2\sqrt{\ln(2nm/\vartheta)/(2T)}$, where $\bar J_T=\frac1T\sum_tx_ty_t^\top$.
*Proof:* Azuma–Hoeffding with conditional range 1 for $\mathbb 1[(a_t,b_t)=(i,j)]-x_t(i)y_t(j)$; union bound. At
$T=10^6$, $2\times2$, $\vartheta=0.05$: TV $\le0.0032$. **Scope:** it bounds $\mathrm{TV}(\hat J,\bar J)$, **not** U's
$\mathrm{TV}(J,\bar x\bar y^\top)$; $\bar J$ (an average of products) differs from $\bar x\bar y^\top$ by time variation,
so Lemma D does **not** separate correlation from time variation. For UCB1 (pure $x_t,y_t$) it is trivial. Lecture 7
§3.2's last sentence is the context.

### 5.2 M1–M2: the real engine and the search tree

**Probe result H4 (U does not explain the engine root at practical budgets).** [M1–M2; L7; **numeric** (probe, not
re-verified)] On real poke-engine roots at $10^3$–$10^4$ iterations the joint visit distribution is indistinguishable
from the product of its marginals; 92–99% of the share gap is **unequalised realised means**, i.e. exploration
(hypothesis H4). Correlation appears only at $N\ge10^5$, and the engine gap keeps decaying through $10^6$. **So U1–U4
are M0, large-$N$ results; they do not explain the MCTS baseline at practical budgets.** Source: correlation probe
(`bridge_plan.md` header). The one-node model F6 (`../lit/next_directions.md`) also sees the gap decay roughly like
$N^{-0.3}$ on real matrices.

**Proposition T (depth composition).** [M2; L7; **sketch**] Synchronous full-information tree, every node running Hedge on
children's current estimates: with **uniform** averaging, $\mathrm{err}_d(n)\le R_n/n+\frac1n\sum_{k\le n}\mathrm{err}_{d+1}(k)$,
which compounds by a factor 2 per level (measured: root error$\cdot\sqrt N$ 0.02 / 0.07 / 0.16 / 0.46 for $D=1..4$).
Gap: tightness of the factor 2. **Weighted averaging** removes the compounding: cite Bai–Jin–Yu 2020 (Nash V-learning,
$\alpha_t=(H+1)/(H+t)$). `dynamics_directions.md:353–380`.

**Per-state averaging in stochastic games.** [M2; L7; **cite**] Independent per-state no-regret learners whose iterates
are averaged need not yield a Nash Markov policy: Kovařík & Lisý (Thm 4.1); fixes by certified/nested policies (Bai,
Jin & Yu 2020; Jin, Liu, Wang & Yu 2021 V-learning). Our agent sidesteps it by re-solving each stage (Theorem A).

**Last iterate (OMWU).** [M0; L7; **cite**] Daskalakis–Panageas 2019; Wei–Lee–Zhang–Luo 2021; Cai–Zheng 2022. On real
stage matrices (1000 untuned iterations) the last iterate is bimodal (median 0.0085, p90 0.086) because many matrices
have pure saddles; no engine effect claimed.

**Meta-game dynamics (PA4α).** [M2 meta; L7; **cite** + **numeric**] Exp3 vs Exp3 over games: Lecture 7 in expectation,
plus Azuma and delay terms (cite). On a meta-matrix of the measured expert-difference size, uniform play is already
a 0.03-Nash and the realised gap is 0.043 / 0.042 / 0.024 at 300 / 3,000 / 30,000 games: strategy convergence is
unobservable at feasible budgets (B1). Not run.

**PA4β asymmetric self-play statement.** [M2; L7; **proved** (corollary of Theorem A and L2(a))] Any opponent, a no-regret
learner (PA1) included, gains against the equilibrium-sampled bot at most that bot's exploitability, which Theorem A
(Corollary C.1 under hidden information) bounds by $\bar H(\varepsilon_s+2\varepsilon_v)$ (+ sampling terms). L2(a)
sandwiches the learner's own payoff by its best expert's trajectory sums $\pm$ regret $\pm$ leaf. The Lecture 7
content that survives is this one-sided statement; the symmetric self-play value is $\tfrac12$ by symmetry and says
nothing. The bound is vacuous in absolute terms ($\varepsilon_v\approx1$); the test (E5, §8) reads a large learner edge as
a measurement of $\varepsilon_v$, not as a refutation of a theorem.

---

## 6. Single-algorithm results (view L6)

### 6.1 M0: online learning, no game

**Lemma H (Hedge + fixed share, pathwise).** [M0; L6; **proved**, summation step **Lean** (`hedge_telescope`, per-step
Hoeffding as a hypothesis)] For every reward sequence $r_t\in[0,1]^k$ that may depend on $w_1,\dots,w_t$ (A1), and every
expert $e$: $\sum_{t\le S}(r_{t,e}-w_t\cdot r_t)\le\ln k/\eta+\eta S/8+S\ln\frac1{1-\alpha}/\eta$; shifting comparator with
$m$ switches pays $m\ln(k/\alpha)/\eta$ more. At the X6 constants ($k=3$, $\eta=0.2$, $\alpha=0.002$): $5.49+0.035S$,
where $0.035=\eta/8+\ln\frac1{1-\alpha}/\eta=0.025+0.010$; the share term is $\approx\alpha/\eta=0.010$. Linear in $S$.
`hedge_experts.md:44–62`.

**Proposition R1 (incremental anytime MW has linear regret).** [M0; L6; **proved**, numeric] On the two-phase sequence
with switch at $\alpha T$: $\mathrm{Reg}_T/T\to\kappa(\alpha)=\min(2\alpha,1-2\alpha)^+$, independent of the step constant
and $k$; $\kappa(0.3)=0.40$. Lecture 6 §2.3's phenomenon for a decreasing-step mirror descent (cf. Orabona & Pál 2018).
`regret_theory.md:38–99`.

**Lemma R2a (anytime FTRL, shipped indexing).** [M0; L6; **proved**] $w_t\propto\exp(\beta_tL_{t-1})$ with
$\beta_t=\sqrt{8\ln k/(t-1)}$: $\mathrm{Reg}_T\le\tfrac32\sqrt{T\ln k/2}+\sqrt{\ln k/8}$ on every path, adaptive
opponents included. Cesa-Bianchi & Lugosi Thm 2.3 (constant 2, cited from memory) applies to their indexing, not
literally to the code. `regret_theory.md:105–136`.

### 6.2 M2: the Hedge learner in the game

**Proposition L2 (safe Hedge in the full game).** [M2; L6; **proved**, simulation; (b) **proved with named hypothesis** (E)] *Assumptions:* A1, A6, A7 with
$\hat A_i$ clipped to $[-1,1]$, A8, A9 with normalised rewards $(q^\top\hat Ae_b+1)/2$, anytime FTRL (R2a), so
$R(T)=2(1.5\sqrt{T\ln k/2}+\sqrt{\ln k/8})$. *Statement.* (a) $\mathbb E[\sum_g(V^*_0-u_g)]\le\mathbb E[\min_e\sum_i\hat\ell_i(q_{e,i},b_i)]+\mathbb E[R(T)]+2\varepsilon_v\mathbb E[T]$.
(b) Under (E): $\le(\varepsilon_s+2\varepsilon_v)\mathbb E[T]+\mathbb E[R(T)]$. (c) gain side of (a). (d) carried vs reset as a
comparison of bounds. *Idea:* L1(a) per game, Lemma 1 for $\hat A$ vs $Q^*$, pathwise regret. `search_rl_links.md:74–103`.
Prior art: Ganzfried & Sandholm 2015 §6.4 ("safe in the limit"), Hannan consistency; new is the finite slack and the
stochastic-game form.

**Propositions E1–E3 and Theorem E2 (code-level safety).** [M2; L6; E1 **Lean** (`claim1_weighted`, `claim1_fixed`),
E2 per-turn step **Lean** (`composition_step`) + **proved with named hypothesis** (E), E3 **proved**] With a fixed
normaliser and (E): match-average $V^*-\mathbb E[u]\le\bar H_\sigma(\varepsilon_s+2\varepsilon_v)+\bar\Delta\,\mathbb E[\mathrm{Reg}_S]/N$
for every adaptive $\sigma$. Bounds the average over games, not the exploitability of any single game's policy. E3: the
learner matches the best expert along the realised trajectory only (external, not policy, regret).
`hedge_experts.md:64–201`.

**Theorem E2′ (shipped learner, carried weights).** [M2; L6; **proved with named hypothesis** (E) and (B2) global
affine fit $a>0$] Adds the clipping loss $C_S$ and the R2a constant. **Does not apply on real leaves:** the best global
affine fit is constant, $\varepsilon_v^{\rm glob}\approx1$ on 6/6 exact endgames. `regret_theory.md:138–183`.

**Lemma R3a (weight identities, $k=2$).** [M2; L6; **proved**] $\ln(w_N/w_F)=\eta_tG_t$; constant drift gives
$w_F=\operatorname{lgs}(-d\sqrt{8\ln2}\sqrt t)$; per-turn deficit to the equilibrium expert is $400\,w_{F,t}\tilde d_t$.
`regret_theory.md:195–204`.

**R3b (fitted drifts).** [M2; L6; **numeric**] From `hedge_stream` logs: $d_1=0.0133\pm0.0023$ (vs MCTS),
$d_2=0.0291\pm0.0014$ (vs exploiter); the constant-slope model **reproduces** the logged weights **in-sample** (slopes
fitted on the same data); "exploiter edge grows with $p_F$" not supported; no per-game link to score
(corr $-0.17$). `regret_theory.md:206–263`.

**Proposition R3c (bracket on the loss to the depth-2 informed best response).** [M2; L6; **proved**] $w_F\gamma_F\le\operatorname{val}\hat A-p\cdot\hat A[:,b]\le w_F\,\mathrm{gap}_F+w_N\varepsilon_s$.
`regret_theory.md:265–285`.

**Proposition R3d (carried vs reset cost).** [M2; L6; **proved**] *Assumptions:* $k=2$; shipped anytime FTRL with the
code's indexing, so turn $t$ plays $w_F(t)=\operatorname{lgs}(-a\sqrt{t-1})$, $a=d\sqrt{8\ln2}$; constant drift $d>0$.
Let $C(T)=d\sum_{t\le T}\operatorname{lgs}(-a\sqrt{t-1})$ (loss relative to always playing $N$).
*Statement.* (i) For every $T$: $C(T)\le C_\infty+d/2$, with $C_\infty:=d\int_0^\infty\operatorname{lgs}(-a\sqrt t)\,dt=\pi^2/(48\ln2\cdot d)\approx0.2966/d$;
$\lim_TC(T)\in[C_\infty,C_\infty+d/2]$, so $C_\infty$ is the limit to leading order. For $T\ll1/a^2$, $C(T)\approx dT/2$.
(ii) Reset each game: $G\cdot C(H)$; carried: $C(GH)\le C_\infty+d/2$, independent of $G$.
(iii) Two phases (drift $d_1$ for $T_1$ turns, then $d_2$): the phase-2 cost is bounded by $C_\infty(d_2)+d_2/2$ only
when $d_1\ge d_2/2$ or over a bounded horizon; in general (for $d_2\ge d_1$) by $(d_2/d_1)\,C_\infty(d_1)$ (+ the turn-1
term).
*Proof.* Idea: a decreasing summand is bounded by its integral. Turn 1 contributes $d\operatorname{lgs}(0)=d/2$; the
rest is $d\sum_{s\ge1}\operatorname{lgs}(-a\sqrt s)\le d\int_0^\infty$; substitute $u=a\sqrt t$ and use
$\int_0^\infty u/(1+e^u)\,du=\pi^2/12$. (iii) with $G_t\ge d_1t$, $w_F\le\operatorname{lgs}(-a_1\sqrt{t-1})$ and per-turn
cost $d_2w_F\le(d_2/d_1)\,d_1w_F$. $\square$
*At the fitted drifts* $d_1=0.0133<d_2/2=0.0146$, so the $d_1\ge d_2/2$ clause of (iii) fails; whether the bounded-horizon clause covers the 60-game run is not checked here. The general bound is
$(0.0291/0.0133)\times0.2966/0.0133\approx48.8$ normalised units, far above the simulated carried phase-2 cost 4.9
(reset 13.2). *Data agreement (numeric):* with actual scored-turn counts, like-for-like within 0.02: reset vs
exploiter end-of-game $w_F$ model value 0.394 (in-sample drifts) vs logged 0.380. Does not convert to the +0.146 win-rate difference
(heuristic units are not win probability). Mechanism: warm-up amortisation of a decaying-$\eta$ learner (Harris,
Anagnostides, Farina, Khodak, Wu & Sandholm, ICLR 2023, arXiv 2209.14110, "Meta-Learning in Games", general principle).

**Proposition P1 (policy vs external regret).** [M2; L6; **proved**, numeric]
*Assumptions:* A6; (P1.1) A1, both clauses: $a_t\perp b_t\mid h_t$ and $b_t$'s law given $h_t$ is $\sigma(h_t)$;
(P1.2) $\sigma$ is one history-dependent strategy, the same in every game (this makes $V(e,\sigma)$ well defined);
(P1.3) experts are history-dependent policies $q_e(h_t)$, fixed across games. The learner plays any history-dependent
$p_t$. Define the **history-indexed continuation** $V^{e,\sigma}(h_t)$, the expected payoff from $h_t$ when the learner
follows $e$ from $h_t$ on and $\sigma$ continues, and $Q^{e,\sigma}(h_t)(a,b)=\mathbb E[V^{e,\sigma}(h_tabs')]$;
$V(e,\sigma)=V^{e,\sigma}(h_0)$.
*Statement.* For every expert $e$:
(i) *Identity.* $V(e,\sigma)-V(p,\sigma)=\mathbb E_{p,\sigma}\big[\sum_{t<\tau}(q_e(h_t)-p_t)^\top Q^{e,\sigma}(h_t)\,\sigma(h_t)\big]$.
(ii) *Decomposition.* This equals an external term $\mathbb E\sum_t(q_e-p_t)^\top Q^*_t\sigma(h_t)$ plus a continuation
mismatch $\mathbb E\sum_t(q_e-p_t)^\top(Q^{e,\sigma}(h_t)-Q^*_t)\sigma(h_t)$. By the tower rule the external term is the
expectation of $\sum_t(q_e-p_t)^\top Q^*_te_{b_t}$. It equals the expectation of Hedge's measured external regret when,
in addition, $p_t=\sum_ew_{e,t}q_e(h_t)\in\operatorname{conv}\{q_e\}$ and Hedge scores with $Q^*$ ($\hat A$ within
$\varepsilon_v$ costs $2\varepsilon_v$ per turn).
(iii) *Bound.* Under A7, $\mathrm{PolicyReg}(e)\le\mathbb E[\mathrm{ExtReg}_T(e)]+2\varepsilon_v\mathbb E[T]+\mathbb E\sum_t\lVert q_e-p_t\rVert_1\lVert Q^{e,\sigma}(h_t)-Q^*_t\rVert_\infty$.
If $\sigma(h_t)$ is $\delta$-minimax and $e$ is $\delta'$-maximin in $Q^*_t$ at every history, the last term is
$\le H(H-1)\max(\delta,\delta')$ $(\le H(H-1)(\delta+\delta'))$.
(iv) *Incomparability.* There is a game with **two decision stages**, two experts and a fixed $\sigma$ in which the
external term is exactly 0 for every weight vector while $\mathrm{PolicyReg}(e_2)=w_{e_1}$.
*Proof.* Idea: L0 with the potential $V^{e,\sigma}$.
1. (i) $\Phi_t(h_t)=V^{e,\sigma}(h_t)$ meets L0's boundary conditions and $Q[\Phi]=Q^{e,\sigma}$; following $e$ at $h_t$ gives
   $\Phi_t(h_t)=q_e(h_t)^\top Q^{e,\sigma}(h_t)\sigma(h_t)$. No Markov property is used; the Markov state-indexed form is
   the special case.
2. (ii) add and subtract $Q^*_t$; tower rule over $b_t\sim\sigma(h_t)$ given $h_t$ (P1.1).
3. (iii) Hölder on the mismatch. For the $\delta$-clause the bounds are **one-sided**: $\sigma$ $\delta$-minimax gives
   $V^{e,\sigma}(h_t)\le V^*_t+(H-t)\delta$ (mirror of Theorem A), $e$ $\delta'$-maximin gives
   $V^{e,\sigma}(h_t)\ge V^*_t-(H-t)\delta'$; so $|V^{e,\sigma}-V^*_t|\le(H-t)\max(\delta,\delta')$ and
   $\lVert Q^{e,\sigma}-Q^*_t\rVert_\infty\le(H-t-1)\max(\delta,\delta')$. With $\lVert q_e-p_t\rVert_1\le2$:
   $\sum_{t<H}2(H-t-1)=H(H-1)$.
4. (iv) Stage 0 at $s_0$: rows $a_1,a_2$; $a_1$ leads to an absorbing payoff-0 state, $a_2$ to $s_1$, matching pennies
   ($\pm1$, value 0), so $Q^*_0(s_0)\equiv0$. Stage 1 at $s_1$. $\sigma$ plays the first column everywhere. Both experts
   play the **pure move $a_1$ at $s_1$** ("match" = $a_1$); $e_1$ plays $a_1$ at $s_0$, $e_2$ plays $a_2$. Every per-turn
   reward vector is constant across experts, so external regret is identically 0; $V(e_2,\sigma)=1$, $V(e_1,\sigma)=0$,
   $V(p,\sigma)=w_{e_2}$. Carried anytime Hedge keeps $w=(\tfrac12,\tfrac12)$ and earns $\tfrac12$ per game. $\square$

*Numeric:* identity error $2.8\cdot10^{-16}$ on 200 random games; on random games the mismatch is about one order of
magnitude below the external term (**18–30×**), not two. Engine logs: median mismatch 0.001–0.003 vs external term 0.030–0.055 ($\lambda>0$), about an order of magnitude. *Applying P1 to Showdown logs requires:* the scoring matrix
$\bar A_t$ not depending on $b_t$ (PA1's dropping of worlds that lack $b_t$ violates this), skipped turns counted, and
payoff units. `dynamics_directions.md:175–235` (superseded there by this form).

**Proposition P2 (switch time).** [M2; L6; **proved**, numeric] *Assumptions:* (P2.1) normalised rewards in $[0,1]$;
(P2.2) drift model: the leading expert out-earns $F$ by $d_1>0$ per turn for $t\le T_1$, and $F$ out-earns it by $d_2>0$
afterwards.
*Statement.*
(i) *Anytime FTRL ($k=2$, code indexing, any step schedule).* $w_F\ge\tfrac12$ iff $F$'s cumulative reward leads, so
recovery takes $\lceil T_1d_1/d_2\rceil+1$ played turns after the switch; the phase-2 regret against $F$ until then is
$\ge\tfrac12T_1d_1$. More generally, for any $k$, the FTRL top-weighted expert is the cumulative-reward leader
(immediate from $w_e\propto e^{\beta_tL_e}$), so "time until top-weighted" is schedule-free.
(ii) *Fixed share.* For $k=2$: recovery within $\lceil\ln((2-\alpha)/\alpha)/(\eta d_2)\rceil$ turns; $\ln(k/\alpha)/(\eta d_2)$
is a valid bound. **For $k\ge3$ the old statement ("$w_F\ge\tfrac12$ within $\ln(k/\alpha)/(\eta d_2)$") is false.** Correct
versions:
- (a) if $F$ out-earns **every** other expert by $\ge d_2$ per turn, $F$ becomes **top-weighted** within
  $\ln(k/\alpha)/(\eta d_2)$ turns, for any $k$;
- (b) $w_F\ge\tfrac12$ only if $\eta d_2>\alpha(2-4/k)$ (exact; necessity **proved**; at the threshold $w_F$ plateaus at 0.4995). The time estimate $\ln(k/\alpha)/(\eta d_2-\alpha(2-4/k))$ is a first-order **upper** bound that overstates by 4×–300× (539 actual vs 2000 at 2.9× the threshold) [**sketch**].
*Proof.* Idea: track log-ratios. (i) $D_t=-d_1T_1+d_2(t-T_1)\ge0$ iff $t-T_1\ge T_1d_1/d_2$; the played weight uses one
more update; before crossing $w_N\ge\tfrac12$, so each turn costs $\ge d_2/2$. (ii) For $k=2$ from the floor $\alpha/2$
the log-odds must rise by $\ln((2-\alpha)/\alpha)$ and rise by $\ge\eta d_2$ per turn (mixing moves a weight below
$\tfrac12$ up). (a) For each $e$, the ratio $w_F/w_e\ge\alpha/k$ at the floor gains a factor $\ge e^{\eta d_2}$ in the Hedge
step, and mixing toward $1/k$ can only move a ratio below 1 toward 1. (b) Near $w_F=\tfrac12$ the share step lowers
$F$'s log-odds by $\approx\alpha(\tfrac12-\tfrac1k)/\tfrac14=\alpha(2-4/k)$ per turn. $\square$
*Numeric:* anytime crossings for $T_1=350/1400/5600$: update index 510 / 2040 / 8160 predicted ($=\lceil T_1(1+d_1/d_2)\rceil$); played turns 511 / 2041 / 8161. X6's fast switch (0–3 battles under fixed share) is **consistent with** (ii), not predicted by it.
*Refuted:* "0.035/turn $=\alpha/\eta$" ($\alpha/\eta=0.010$); fixed-share steady state 0.38 and break-even at 74 battles are in §10. `dynamics_directions.md:237–270` (superseded).

### 6.3 ε-safe best response

**ε-safe expert inside Hedge.** [M2; L6; **cite**] The ε-safe LP's stage gap is $\le\varepsilon_s+\varepsilon\cdot\operatorname{range}$, so
L2(b) holds with that per-turn error and L2(c)/E3 give the gain side. Cite: L2; Ganzfried & Sandholm 2015 §6.4; Li 2026
(arXiv 2607.28520, runtime certificate); Jeary & Turrini 2023 (arXiv 2307.12338, "Safe Opponent Exploitation for
Epsilon Equilibrium Strategies"). Non-convexity remark [**proved**, 2-turn example]: per-turn mixing of two zero-exploitability
experts can have exploitability 1, so "$\operatorname{expl}(\mathrm{Hedge})\le\min_e\operatorname{expl}(e)+\mathrm{Reg}$" is false.

**ε-add-on arithmetic.** [M2; L6; **proved** (arithmetic on L2(b)) + **numeric**] With $\hat A$ on the payoff scale
(range $\le2$; the shipped LP must run on the squashed matrix), the ε add-on is $\varepsilon\bar H$ in score units; at
$\bar H\approx35$ scored turns: 0.105 (ε = 0.3%), 0.35 (1%), **0.70 (2%, vacuous)**. So the add-on is non-vacuous
only for $\varepsilon\lesssim1\%$. The base term $\bar H(\varepsilon_s+2\varepsilon_v)$ is vacuous already, so the bound
the ε-safe bound says only that its add-on is non-vacuous. Gain curve (LP sensitivity, cite): concave in ε (300/300 real matrices),
recovering a median 35% / 80% / 100% of the top move's stage gain at 0.3% / 1% / 2%.
*Empirical (engine):* SX ε-safe 2% 0.530 vs MCTS-2000 (top move 0.580, sampled 0.510; paired differences n.s.); SXW: the
ε-safe best-response bound did not translate into game-level robustness (§3).

### 6.4 RL best responses

**Theorem L3 (RL best response = per-state Hedge in $M_\pi$).** [M2; L6; **proved**] Averaged iterate value minus a Markov
comparator's value $\le\sum_{t,s}d^\circ_t(s)\mathrm{Reg}_K(t,s)/K+$ critic term; horizon factor $\bar H(\sigma^\circ)$.
Corollaries L3a (RL gives a lower bound with computable slack), L3b (distribution mismatch $2HC\varepsilon_{\rm crit}$).
`search_rl_links.md:107–128`.

**Proposition L3c (undirected exploration).** [M2; L6; **proved**] A chain with $|B|^D$ miss probability: uniform-start
exploiters need $\sim|B|^D$ games; informed-best-response backward induction needs none. `search_rl_links.md:130–140`.
*Empirical:* RL BR vs Foul Play reached 0.125 at 8M decisions (black-box); gray-box (opponent's move distribution in
the observation) 0.075 at 8M (single runs, unpaired, $\approx2.7$ SE): **gray-box did not beat black-box**; the depth-2 informed best response gives 95% lower confidence bounds $\ge0.25$ / $\ge0.09$ (top / sampled) with no training. Reading fixed before the run: $q$ without a
payoff model is not enough; it does not say knowing the policy is useless.

**AdaHedge second-order slack.** [M2; L6; **cite** + numeric] de Rooij et al. 2014 (constants from memory): about 2.0
ranges per game vs 6.8 first-order; no single-game bound is non-vacuous; only amortisation across games helps.

---

## 7. M4: real Showdown

Measured on Showdown so far: the bridge probes of §3.1 and the playable-algorithm runs of §8 (E1, E2, E2fs, E4 in four arms, E5, BH, AF16; 2026-11-15), and the open items of §11 (LOG, E3a/E3f, the exploiter arm X*, E5t/E5s, the $\varepsilon$ sweep SX*, RQ*, leaf exposure LSs/LSn; 2026-11-16). They test predictions of §4–§6; they do not measure the quantities ($\varepsilon_v$, $\varepsilon_{\rm sim}$, $\bar H$) in the bounds.

**Proposition SIM (simulation lemma for exploitability).** [M4; S; **exact resolution: proved**; coarsened resolution: **proved with the $\omega$ term**; trajectory form: **sketch** (standard lemma: cite Kearns & Singh 2002)]
*Assumptions:* A1(i), A6, A5; the same action sets and terminal payoffs in both kernels. This precondition fails on the structural rows of `game_model.md` §3.2: Struggle, PP, the pending move at a pivot, simultaneous wipe-out payoffs, horizon, and team prior. These are outside A5/SIM and are stated as A10–A12.
*Statement (exact resolution).* For every policy $\pi$,
$$\operatorname{expl}_{\rm sd}(\pi)\le\operatorname{expl}_{\rm pe}(\pi)+\sum_{t<H}\varepsilon_{{\rm sim},t}\Big(\operatorname{span}V^*_{{\rm pe},t+1}+\sup_\sigma\operatorname{span}V^{\pi,\sigma}_{{\rm pe},t+1}\Big)\le\operatorname{expl}_{\rm pe}(\pi)+2\Delta\sum_{t<H}\varepsilon_{{\rm sim},t}.$$
The sum runs over all $t<H$ ($H=300$ decision nodes) with a sup over states in $\varepsilon_{{\rm sim},t}$, so it is linear in the horizon. It is tight at $H=1$ (the $2\Delta$ form). The first part of the sum is A5's $\varepsilon_{\rm sim}$. Numerics: 700 random games, 0 violations.
*Proof.* (1) Values: $V^*_{{\rm sd},t}-V^*_{{\rm pe},t}$ is bounded by Lemma 2(a) and
$Q_{\rm sd}-Q_{\rm pe}=P_{\rm sd}(V^*_{\rm sd}-V^*_{\rm pe})+(P_{\rm sd}-P_{\rm pe})V^*_{\rm pe}$, with
$|(P_{\rm sd}-P_{\rm pe})V|\le\mathrm{TV}\cdot\operatorname{span}V$; unroll. (2) Policy values: L0 in the Showdown game with
potential $V^{\pi,\sigma}_{\rm pe}$ gives the same per-stage term for each $\sigma$. (3) Subtract. Under hidden
information apply it per world. $\square$
*Statement (coarsened resolution).* The displayed bound with TV taken on a partition is **false** without a within-cell
term (150/300 counterexamples). The correct form is
$$\operatorname{expl}_{\rm sd}\le\operatorname{expl}_{\rm pe}+\sum_t\varepsilon^\Pi_t\big(\operatorname{span}V^*_{{\rm pe},t+1}+\sup_\sigma\operatorname{span}V^{\pi,\sigma}_{{\rm pe},t+1}\big)+\sum_t\mathrm{TV}^{\rm exact}_t\big(\omega^*_{t+1}+\sup_\sigma\omega^{\pi,\sigma}_{t+1}\big),$$
where $\varepsilon^\Pi_t$ is the TV between the kernels pushed forward to the partition $\Pi$, $\mathrm{TV}^{\rm exact}_t$ the TV at exact-state resolution, and
$\omega_{t+1}=\max_C\operatorname{osc}_C V$ is the maximum over cells $C$ of the oscillation of $V$ within $C$. This held on all 600 tests.
*Reading.* $\mathrm{TV}^{\rm exact}\approx0.89$–1 per damaging move, so at KO resolution the $\omega$ term is roughly the full
within-cell value oscillation and the coarse bound is uninformative. A Wasserstein variant,
$|(P_{\rm sd}-P_{\rm pe})V|\le W_1\cdot\mathrm{Lip}_{\rm HP}(V)$, fits the measured 2 HP% damage error and is the better route
[**sketch**, not checked].
*Trajectory form (for using $\bar H$).* [**sketch**, derived, not checked numerically]
$V^*_{\rm sd}-V^*_{\rm pe}\le\mathbb E_{\rm sd}\big[\sum_{t<\tau}(P_{\rm sd}-P_{\rm pe})V_{\rm pe}\big]$ along $(x^*_{\rm sd},y^*_{\rm pe})$, and the
policy term along $(\pi,\mathrm{BR}_{\rm sd})$. These lengths and TVs are along **adversarial** trajectories, not self-play.
*Measured (probe):* mean KO-gap $\approx0.012$ per move, 0.011–0.015 per turn at KO resolution (sup unmeasured); $\ge0.89$ at exact-state resolution (vacuous).

**What has been measured at M3 in the engine and awaits M4:** Bayesian LP vs PIMC 0.585 [0.540, 0.630] (400 games,
equal compute); oracle gap $\approx0.1$; X5 hidden-information exploiter gain 0.00 / 0.03 (n.s.).

---

## 8. Playable algorithms on Showdown

Each is a drop-in for the adapter's per-turn move choice (`our_find_best_move`), built on the $K$ sampled-world
matrices. Predictions are from `bridge_plan.md` §2 and `dynamics_directions.md` §0, restated against the verified
results. All are M4 tests. Run on real Showdown (2026-11-15; Modal except E1): E1, E2, E2fs, E4 (shares and most-visited actions, head-to-head against RM+), E5, BH and AF16; results below the table. Not run on 2026-11-15: E3, PA2, and E4 against an exploiter; all three were run on 2026-11-16 (E3a/E3f; the $\varepsilon$ sweep SX1–SX30 of `bayes_safex`, a PA2-type sweep without the squashed LP; exploiters XU, XUmv, XR, XEmv), results in §11. Sources: `engine/showdown/results/<TAG>/<TAG>_summary.json`; interpretation in `engine/RESULTS.md`, section "Real-Showdown experiments (Modal, 2026-11-15)".

| # | Algorithm | Theorem it instantiates (level) | Real-Showdown prediction | Refuted if | Games |
|---|---|---|---|---|---|
| **E1** | Bayesian LP vs PIMC, both using Foul Play's set sampler as belief | Theorem C / Cor C.1 and the PIMC-averaging loss bound (M3), carried to M4 via A4, A5 | Bayesian LP > PIMC (engine: 0.585) | paired $\Delta\le0$ at $n=300$ | 600 |

**E1 result (measured).** On real Showdown, Bayesian LP vs PIMC, both arms sharing Foul Play's belief: pair score **0.559 [0.521, 0.597]** over 298 paired seeds. Status: measured; it transfers the in-engine 0.585. Source: `engine/RESULTS.md`, section "E1: Bayesian LP vs PIMC on real Showdown".
| **E2** (PA1) | **HedgeBot**: carried Hedge over {top, sampled, pure maximin, ε-safe 0.3%}, per-turn full-information feedback (opponent's realised move) | L2(a)–(d), R2a pathwise bound (M2); P1 (M2) for what external regret does not say | (1) pathwise external regret $\le R(T)$ on every log (one run suffices); (2) $W_H\in[\min_eW_e-0.05,\max_eW_e+0.05]$ | (1) fails on any log (rewards left $[0,1]$), or $W_H<\min_eW_e-0.05$ with non-overlapping 95% CIs. Log the skipped-turn fraction (no world contains $b_t$); above ~10% the feedback is semi-bandit and L2 needs Exp3. P1 applies to the log only if $\bar A_t$ does not depend on $b_t$ | 300 + 150/expert |
| **E3** | **Switch test**: anytime FTRL vs fixed-share HedgeBot ($k=4$), opponent switches after $T_1$ games | P2 (M2), updated: measure **time until top-weighted** (P2(ii)(a)), not time to $w_F\ge\tfrac12$ | fixed share: favoured expert top-weighted within $\ln(k/\alpha)/(\eta d_2)$ turns (an upper bound, valid for any $k$ if it out-earns every other expert by $\ge d_2$; at X6 constants and $d_2=0.0291$: $\le1306$ turns $\approx37$ battles); anytime: top-weighted exactly when its cumulative reward leads (schedule-free), i.e. linear in $T_1$ | fixed share exceeds its bound while the premise holds on the log; or the anytime-minus-fixed-share gap does not grow with $T_1$ (or the two crossings are within ±3 battles) | 160 |
| **E4** (PA3) | **BanditRootBot**: per-turn bandit self-play (UCB1 / Exp3, $N=2000$, MCTS's feedback model) vs full-information RM+ on the same matrices | L4 identity + B3 (M0–M1), L1(b) lower side (M2) | bandit root loses $\ge0.05$ more against an exploiter (stage gap 0.10–0.15 on ~80% of two-sided decisions) | $|\Delta|\le0.03$ at $n=300$ (then feedback model does not matter at MCTS budgets: a result in itself) | 800 |
| **E5** (PA4β) | **Asymmetric self-play**: HedgeBot (all experts) vs equilibrium-sampled | PA4β statement (M2; Theorem A + L2(a)) | $W_A\in[0.45,0.55]$ | $W_A>0.56$: read as a measurement of a larger $\varepsilon_v$ (stage-wise deviations profit), not a theorem failure | 400 |
| **PA2** | **ε-safe best-response bot (SafexBot, ε)**, ε $\le1\%$, LP on the squashed $[-1,1]$ matrix | L2(b) with $\varepsilon_s\to\varepsilon_s+\varepsilon\cdot$range (M2, cite); bound arithmetic | win rate $\approx W_{\rm sampled}+f(\varepsilon)(W_{\rm top}-W_{\rm sampled})$, $f=0.35,0.8$ at 0.3%, 1% (engine pre-screen first; effects 0.03–0.06 are below the $n=200$ CI) | safex(1%) $\le$ sampled. Against an exploiter, only a direction test: the theorem bounds each bot's loss separately, not the *difference* to sampled, and SXW already shows the 2% bound not surviving M1→M2 | 900 (pre-screen 1.4 CPU-h) |
| **E4 _mv** (B8) | **Most-visited root**: `bandit_exp3_mv`, `bandit_ucb1_mv` (the MCTS decision rule) vs `rmplus_avg` | B8 (`experts_extensions.md`; identity proved, real-opponent form a hypothesis) | written before the runs (`experts_extensions.md` B8); quantified pre-registration recorded before results were read (`scratchpad/verify_ext/prereg_B8.txt`): _mv scores far above the shares roots against RM+ (deficit 5–8× smaller) | _mv $\le$ shares at $n=300$ | 600 |
| **BH** | **hedge_bayes**: E2's HedgeBot with the ε-safe expert's prediction $\hat q$ from the style posterior | B1 (Bayes = Hedge on log loss), B5, B6 (`experts_extensions.md`) | no win-rate prediction (B6's whole-game bound is vacuous per game); paired with E2 on the same seeds | — | 800 |
| **AF16** | **Bayesian LP with $K=16$ worlds** vs Foul Play | B10, B11 (`experts_extensions.md`) | B11(iv): no win-rate gain derivable (in-engine X2, 8 vs 4 worlds: 0.495 n.s.); paired with AF | — | 600 |

**Results (measured, real Showdown; pair score of the first-named bot [95% CI], pairs).** Status: measured. Paired differences are over common seeds (same teams and battle PRNG).
- **E1** Bayesian LP vs PIMC: **0.559 [0.521, 0.597]**, 298. Prediction met (transfers the in-engine 0.585).
- **E2** hedge vs Foul Play: **0.3108 [0.2804, 0.3412]**, 399; **E2fs** hedge_fs vs Foul Play: **0.3033 [0.2729, 0.3336]**, 399. Prediction (1) met: on all 10 E2 matches (a match, also called a stream in the logs: a sequence of games over which Hedge weights and the posterior are carried) the carried learner's regret on normalised stage payoffs per turn stays below the R2a bound (median 0.0042, IQR 0.0039–0.0046, vs bound 0.0277 at $T=2056$; slope −0.15 vs −0.51; rewards all in $[0,1]$; reset weights 0.0057; weight-replay check 0.0092). Prediction (2) met for the one expert run alone (AF 0.3144). 16.43% / 16.83% of feedback turns skipped (> 10%: L2 holds only approximately).
- **E4** (head-to-head vs `rmplus_avg`, not vs an exploiter as designed): shares roots **E4e 0.0625 [0.0430, 0.0820]**, 296; **E4u 0.0502 [0.0312, 0.0691]**, 299 (median top-move probability 0.315 / 0.225 vs 0.78 for RM+; median stage gap 0.0585 / 0.0578). Most-visited roots **E4emv 0.4949 [0.4547, 0.5351]**, 295; **E4umv 0.5933 [0.5551, 0.6316]**, 300. B8's pre-registered ordering held (the verifier's point guess 0.25–0.45 undershot both): the decision rule, not the feedback model, drives the outcome. E4umv > 0.5 shows `rmplus_avg` is not a stage equilibrium of the bandit's matrix (B8(iii) is a hypothesis for it). The exploitability of max-visit play was untested on 2026-11-15; the 2026-11-16 model-based exploiters exploit it at most marginally (XUmv +0.052 [−0.002, 0.106], match-clustered [0.007, 0.097]; XEmv −0.053 [−0.107, 0.002] over RM$^+$; §11).
- **E5** hedge vs Bayesian LP: **0.4583 [0.4210, 0.4956]**, 300. Inside the predicted band $[0.45,0.55]$, not refuted; the CI lies below 0.5, which PA4β (an upper-side bound) does not address. Controls (2026-11-16, §11): top move alone 0.5500, Bayesian LP self-play 0.5185; the split of the deficit is not identifiable from the logs.
- **BH** hedge_bayes vs Foul Play: **0.3329 [0.3022, 0.3637]**, 398; paired BH − E2 +0.024 [−0.018, +0.066] (397), BH − E2fs +0.029 [−0.013, +0.071] (397), BH − AF +0.017 [−0.032, +0.066] (297): n.s.
- **AF16** Bayesian LP $K=16$ vs Foul Play: **0.3278 [0.2923, 0.3632]**, 299; paired AF16 − AF +0.017 [−0.031, +0.065] (298): n.s., as B11(iv) expects; the B10 absent fractions at $K=16$ were not scored.
- Reading (interpretation): better opponent prediction (BH) and calibration (AF16) do not translate into significant wins, consistent with the leaf evaluation error dominating the bound ($\varepsilon_v\approx1$ dominates Theorems A and C).

Not run (cite only): meta-game Exp3 vs Exp3 (B1: too flat to observe), OMWU last iterate, CCE.

---

## 9. Status table

View: S structural, L6 single algorithm, L7 dynamics. "Test" is the real-game test or the measurement that bears on it.

| Result | Level | View | Status | Source | Real-game test |
|---|---|---|---|---|---|
| Lemma 1 (one step, factor 2 tight) | M1 | S | Lean | `full_game.md:34` | B01 (endgames only) |
| Lemma 2(a) val Lipschitz | M1 | S | Lean | `full_game.md:41` | — |
| Lemma 2(b)–(d) lookahead, no contraction | M1–M2 | S | proved | `full_game.md:42` | L5 / depth |
| Lemma R6a affine invariance, affine-invariant Thm A | M1–M2 | S | proved | `regret_theory.md:403` | endgames |
| Lemma R6b inversions, constant-fit criterion | M1 | S | proved | `regret_theory.md:422` | endgames (best fit constant on 6/6) |
| Lemma L0 telescoping | M2 | S | proved | `search_rl_links.md:37` | — |
| Proposition 0 value | M2 | S | proved | `full_game.md:28` | — |
| Theorem A per-turn composition, $\bar H$ refinement | M2 | S | Lean + proved (model link by reading) | `full_game.md:52` | B12 (probe: residual isolated) |
| Theorem L1 identity + sandwich | M2 | S | proved | `search_rl_links.md:57` | B12 |
| Theorem L5 depth with exact terminals (squash/vnet only) | M2 | S | proved | `search_rl_links.md:203` | endgames |
| Theorem L6 TD residual | M2 | S | proved | `search_rl_links.md:227` | LF (n.s.) |
| Proposition R6c non-termination | M2 | S | proved | `regret_theory.md:462` | endgame loop 0.029 |
| Corollary R4 transfer | M1–M2 | S | proved | `regret_theory.md:297` | — |
| Propositions R5a–c bracket | M2/M4 | S | proved | `regret_theory.md:330` | W1, SXW LCBs |
| Theorem B one stage | M3 | S | proved | `full_game.md:105` | — |
| Theorem B′ sampling | M3 | S | proved | `full_game.md:118` | — |
| Proposition S1 fixed-prior rate | M3 | S | proved with named hypothesis | `new_directions.md:64` (corrected here) | logged beliefs near boundaries |
| Proposition S2 lower bound | M3 | S | proved | `new_directions.md:84` (corrected here) | — |
| Theorem C five terms | M3 | S | proved | `full_game.md:166` | B23 |
| Corollary C.1 headline bound; Example C.1 | M3 | S | proved | `full_game.md:176` | B23, E1 |
| PIMC-averaging loss bound $4\delta_H$ | M3 | S | numeric | RESULTS | E1 |
| PIMC-averaging loss bound $2\delta_H$ | M3 | S | sketch | RESULTS | — |
| $\sum(\eta_{\rm NL}+4\mathrm{TV})\approx$ oracle gap | M3 | S | conjecture | `full_game.md:190` | B23 |
| Proposition SIM (exact resolution) | M4 | S | proved | this map §7 | B34 (probe) |
| Proposition SIM (coarsened, with $\omega$ term) | M4 | S | proved with the $\omega$ term | this map §7 | B34 (probe) |
| Proposition SIM trajectory form ($\bar H$) | M4 | S | sketch | this map §7 | — |
| B12 decomposition | M1→M2 | S | numeric (probe) | §3.1 | — |
| B34 kernel fidelity | M3→M4 | S | numeric (probe) | §3.1 | — |
| Lecture 7 theorem; zero-sum CCE | M0 | L7 | cite | Lecture 7 | — |
| Prop 4 / Cor A′ RM+ as coded | M0–M1 | L7 | proved (key step Lean) | `full_game.md:81` | — |
| RM+ weighted averaging, alternation (BMS19) | M0 | L7 | cite | `new_directions.md:274` | — |
| Proposition L4 identity | M0 | L7 | proved | `search_rl_links.md:149` | E4e/E4u (measured): shares roots 0.0625 / 0.0502 vs RM+; median gap 0.058 |
| Proposition L4b lockstep | M0 | L7 | proved | `search_rl_links.md:180` | — |
| Proposition U1 | M0 | L7 | proved with named hypothesis | `new_directions.md:148` (corrected) | not at engine budgets (H4) |
| Corollary U2 (incl. gap ⇒ TV>0, contrapositive, U1.2) | M0 | L7 | proved with named hypothesis | `new_directions.md:165` | — |
| Converse of gap ⇒ TV fails | M0 | L7 | numeric | verifier | — |
| Proposition U-fin (tighter form; row-wise tighter still) | M0 | L7 | proved | this map §5.1 | — |
| Conjecture U3 plateau | M0 | L7 | conjecture | `new_directions.md:183` | — |
| Observation U4 | M0 | L7 | numeric | `new_directions.md:199` (corrected) | — |
| H4 engine root = exploration | M1–M2 | L7 | numeric | probe | — |
| Corollary B3 | M0–M1 | L7 | numeric (L4 part proved; bounds cite) | `dynamics_directions.md:290` (corrected) | E4e/E4u (measured, head-to-head; exploiter arm run 2026-11-16, §11); E4emv/E4umv 0.4949 / 0.5933: decision rule, not feedback, drives the outcome (B8) |
| Lemma D | M0 | L7 | proved | `dynamics_directions.md:342` (scope corrected) | — |
| Proposition T depth composition | M2 | L7 | sketch | `dynamics_directions.md:353` | — |
| Per-state averaging failure | M2 | L7 | cite | `new_directions.md:280` | — |
| OMWU last iterate | M0 | L7 | cite | `dynamics_directions.md:393` | — |
| Meta-game Exp3 dynamics (B1) | M2 | L7 | cite | `dynamics_directions.md:319` | not run |
| PA4β statement | M2 | L7 | proved | §5.2 | E5 (measured): 0.4583 [0.4210, 0.4956], inside $[0.45,0.55]$; CI below 0.5 |
| Lemma H | M0 | L6 | proved (summation step Lean) | `hedge_experts.md:44` | — |
| Proposition R1 | M0 | L6 | proved | `regret_theory.md:47` | — |
| Lemma R2a | M0 | L6 | proved | `regret_theory.md:105` | E2 (measured): pathwise bound held on all 10 matches; median 0.0042 vs 0.0277 at $T=2056$ |
| Proposition L2 (b: proved with named hypothesis (E)) | M2 | L6 | proved | `search_rl_links.md:84` | E2/E2fs (measured): 0.3108 / 0.3033 vs AF 0.3144, within 0.05; 16.4–16.8% of feedback skipped |
| Propositions E1, E3 | M2 | L6 | Lean (E1) / proved (E3) | `hedge_experts.md:64` | E2 (measured, see L2) |
| Theorem E2 | M2 | L6 | Lean (`composition_step`) + proved with named hypothesis (E) | `hedge_experts.md:106` | E2 (measured; bound vacuous in level) |
| Theorem E2′ | M2 | L6 | proved with named hypothesis | `regret_theory.md:138` | does not apply (best global affine fit constant) |
| Lemma R3a | M2 | L6 | proved | `regret_theory.md:195` | — |
| R3b fitted drifts | M2 | L6 | numeric | `regret_theory.md:206` | — |
| Proposition R3c | M2 | L6 | proved | `regret_theory.md:265` | — |
| Proposition R3d | M2 | L6 | proved | `new_directions.md:225` (corrected) | hedge_stream (in-sample); E2 logs: carried 0.0042 vs reset 0.0057 regret per turn, win rate unchanged |
| Proposition P1 | M2 | L6 | proved | `dynamics_directions.md:175` (corrected) | E2 run; policy regret not separately measured |
| Proposition P2 (necessity in (ii)(b) proved; time estimate sketch) | M2 | L6 | proved | `dynamics_directions.md:237` (corrected) | E3a/E3f (measured 2026-11-16, §11): fixed share − Hedge +0.003 [−0.066, 0.072]; flips median 273 / 245 updates, analog bound not exceeded |
| ε-safe expert in Hedge | M2 | L6 | cite | `new_directions.md:259` | PA2 |
| Non-convexity remark (Hedge vs min expl) | M2 | L6 | proved | §6.3 | — |
| ε-add-on arithmetic, gain curve | M2 | L6 | proved (arithmetic) / numeric (curve) | `dynamics_directions.md:272` | PA2; SXW failure |
| Theorem L3, L3a, L3b | M2 | L6 | proved | `search_rl_links.md:115` | rlbr |
| Proposition L3c | M2 | L6 | proved | `search_rl_links.md:130` | gray-box vs black-box |
| AdaHedge slack | M2 | L6 | cite | `new_directions.md:287` | — |
| A10 menu and payoff alignment | M3–M4 | S | assumption | this map §2; `game_model.md` §5 | B34 (probe) |
| A11 information at pivot nodes | M3–M4 | S | assumption | this map §2; `game_model.md` §5 | — |
| A12 horizon unit (decision nodes) | M2–M4 | S | assumption | this map §2; `game_model.md` §5 | — |
| E1 Bayesian LP vs PIMC on Showdown | M4 | S | measured (numeric) | `engine/RESULTS.md` (E1) | 0.559 [0.521, 0.597], 298 pairs |
| E2 / E2fs Hedge over experts vs Foul Play | M4 | L6 | measured (numeric) | `engine/showdown/results/{E2,E2fs}/` | 0.3108 [0.2804, 0.3412] / 0.3033 [0.2729, 0.3336], 399 pairs each |
| E4e / E4u bandit roots (visit shares) vs RM+ | M4 | L7 | measured (numeric) | `engine/showdown/results/{E4e,E4u}/` | 0.0625 [0.0430, 0.0820], 296 / 0.0502 [0.0312, 0.0691], 299 |
| E4emv / E4umv most-visited roots vs RM+ (B8, pre-registered) | M4 | L7 | measured (numeric) | `engine/showdown/results/{E4emv,E4umv}/` | 0.4949 [0.4547, 0.5351], 295 / 0.5933 [0.5551, 0.6316], 300; ordering confirmed |
| E5 Hedge vs Bayesian LP | M4 | L7 | measured (numeric) | `engine/showdown/results/E5/` | 0.4583 [0.4210, 0.4956], 300 pairs |
| BH hedge_bayes vs Foul Play | M4 | L6 | measured (numeric) | `engine/showdown/results/BH/` | 0.3329 [0.3022, 0.3637], 398; vs E2 +0.024 [−0.018, +0.066] n.s. |
| AF16 Bayesian LP $K=16$ vs Foul Play | M4 | S | measured (numeric) | `engine/showdown/results/AF16/` | 0.3278 [0.2923, 0.3632], 299; vs AF +0.017 [−0.031, +0.065] n.s. |

**Count by status (76 rows):** Lean 5; proved 36; proved with named hypothesis 4; proved with the $\omega$ term 1; sketch 3;
conjecture 2; numeric 8; cite 7; assumption 3; measured 7 (E1, E2/E2fs, E4e/E4u, E4emv/E4umv, E5, BH, AF16). (Rows with a mixed tag are counted by the first-named tag.)

---

## 10. Refuted / superseded (kept so they are not re-derived)

*Corrections applied in this map (verifier, probe, coordinator, 2026-11-15).*
- **S1 without $\gamma\le\min_\theta b_\theta$:** false (counterexample $M_1=[[1],[0]]$, $M_2=[[1],[1]]$, $b=0.05$; loss 0.05
  with probability $0.95^K$).
- **S2 numeric table** ($\sqrt K\,\mathbb E[L_K]=0.0315\dots0.0236$ and the grid supremum $0.0362/\sqrt K$, …): an adverse
  tie-break artefact; exact values in §4.3. The constant 0.0231 is improved to 0.0255 at $c^*=0.376$.
- **U1 index inequality subtracting $k$'s bonus:** typo; $i$'s bonus is subtracted. **(U1.2) for one arm:** insufficient;
  needed for both.
- **U4 "both players lock into a pure cell":** play is a correlated full-support mixture.
- **"U explains the MCTS baseline's root shares":** refuted at practical budgets by the real-engine probe (H4).
- **Lemma D "separates U's correlation from sampling noise / time variation"** (`dynamics_directions.md` §2.6 and PA4β's
  logging rationale): it bounds $\mathrm{TV}(\hat J,\bar J)$ only.
- **R3d:** "the fitted drifts predict the weights" → reproduce them in-sample; $C(T)\le C_\infty$ → $C_\infty+d/2$ with the code's
  indexing, and $C_\infty$ is the limit to leading order; "$C_\infty(d_2)=10.2$ still bounds the two-phase carried run" →
  holds only when $d_1\ge d_2/2$ or over a bounded horizon (the first clause fails at the fitted drifts; the second is unchecked for the 60-game run); "four logged weights within 0.05"
  (0.43 vs 0.38) → like-for-like 0.394 vs 0.380 with actual scored-turn counts.
- **P1:** Markov-only statement → history-indexed; $2H^2(\delta+\delta')$ → $H(H-1)\max(\delta,\delta')$; the $H=3$
  counterexample with "match" read as a strategy → two decision stages, pure $a_1$ at $s_1$; "mismatch two orders below
  the external term" → 18–30×.
- **P2:** recovery "$T_1d_1/d_2$" → $\lceil T_1d_1/d_2\rceil+1$; fixed-share "$w_F\ge\tfrac12$ within $\ln(k/\alpha)/(\eta d_2)$" for
  $k\ge3$ → false (see (a), (b)); steady state 0.38 → 0.25; break-even 74 battles → fixed share is never slower;
  "0.035/turn $=\alpha/\eta$" → $\alpha/\eta=0.010$; "X6's switch predicted" → consistent with.
- **B3:** Exp3-2000 0.154 / UCB1 0.120 / 82% and 81% above 0.05, "bound 0.56 range", "both bounds vacuous" → the corrected
  numbers of §5.1; the full-information bound (0.066) is not vacuous.
- **T as a proved proposition** → sketch (weighted case cited).
- **Citation names:** arXiv 2209.14110 is Harris et al. (ICLR 2023), not "Harvey"; arXiv 2307.12338 is Jeary & Turrini,
  not "Jeanblanc".
- **PA2 "loss vs sampled $\le\varepsilon\bar H$ against an exploiter" as a theorem consequence:** not implied (the theorem
  bounds each bot, not the difference).
- **Empirical bridge failures:** 2% ε-safe best-response bound as game-level robustness (SXW 0.767 vs 0.713); leaf rescaling
  (LF $-0.040$, n.s.); gray-box over black-box RL (0.075 vs 0.125 at 8M).
- **A5 at exact-state resolution:** vacuous ($\ge0.89$ per damaging move); A5 is stated at a coarsened resolution.

*Refuted 2026-11-16 (§11; sources `tightening/*.md`).*
- **"A common $\varepsilon_v$ cancels in the comparison of two agents on the same leaf":** false; $\hat Q=0$ ($2\times2$),
  $Q=\varepsilon\begin{psmallmatrix}1&-1\\1&1\end{psmallmatrix}$: both strategies are exactly optimal in $\hat Q$ and differ by
  $2\varepsilon_v$ in stage loss. What holds is the comparative identity of §11.
- **"The coded forgetting (`Posterior` with $\lambda<1$, predict then discount) has discounted regret $\le\ln k$", and
  "it holds for $\lambda\ge\tfrac12$":** false (T1); the ½ was an artefact of a non-optimal build-up.
- **"E3 tests posterior tracking":** E3a/E3f run `hedge`/`hedge_fs` (no posterior).

*Earlier refutations in the source notes (still refuted).*
- RM+ slope $T^{-1.46}$ (`full_game.md` §4, §7): not reproducible; superseded by $T^{-1.3}$ (median $-1.26$).
- Two-type Example C.1 as a counterexample to the four-term form: wrong ($\eta_{\rm NL}(0)=3/5$); replaced by the three-team example.
- Four-term form against arbitrary opponents: false. Targets' $2R\,\mathrm{TV}$ with payoffs in $[-R,R]$: should be $2\Delta\,\mathrm{TV}$.
- "Expected length under an equilibrium opponent" in Theorem A: equilibrium opponents need not be unique; "some".
- Cor A′ "not proved for the alternating code": proved (cross term $\le0$).
- Kovařík et al. author list; ReBeL Thm 3 constants (`CHECK.md` findings 3, 4).
- Raw-unit Claim 1 with a per-turn normaliser: false (deficit $S/3$); per-turn affine fit in E2: false.
- Incremental anytime MW as anytime Hedge: linear regret (R1); CBL Thm 2.3 verbatim for the code: superseded by R2a.
- "Exploiter's normalised-stage-payoff edge grows with $p_F$": not supported.
- Stage-wise equilibria of $V^*$ unexploitable without a cap: false (R6c).
- L4 "converges iff both regrets sublinear"; "0.75 rule lies between the experts" in exploitability; L2 (H) first draft; L6
  Remark 4 sign; L1 Remark 2 example (`search_rl_links.md` refuted list).
- "$\operatorname{expl}(\mathrm{Hedge})\le\min_e\operatorname{expl}(e)+\mathrm{Reg}$": false (non-convexity).
- "Prop 3's $\sqrt{\log K/K}$ is the rate the agent pays": false for fixed interior priors (S1).
- "Lecture 7 on the meta-game is testable at a few hundred games": refuted numerically (B1).
- "Zero external regret ⇒ as good as the best expert": false (P1(iv)).
- "Carried beats reset by a tracking argument": superseded by warm-up amortisation (R3d).

---

## 11. Tightening results (2026-11)

Added 2026-11-16. Sources: `tightening/bounds_epsv.md`, `tightening/bayes_dynamics.md`, `tightening/voi.md`,
`tightening/verify/leaf/README.txt` (theory, all "Verified 2026-11-16, corrections applied"); literature labels and
citation keys from `../lit/tightening_related_work.md` (a targeted search of about 25 queries, not a systematic
review: "new" below means "no prior statement found in that search"); real-Showdown numbers from
`../engine/analysis/open_items/REPORT.md` (independently verified, corrections applied) and the leaf-exposure runs
`../engine/showdown/results/{LSs,LSn}/`. Status words as in §0.4. Name collisions: the open-item analyses are called
"open-item B1–B6" here; `experts_extensions.md`'s B1 (Bayes = Hedge) is "B1 (Bayes)"; the bridge M1→M2 stays "bridge
B12" and `bayes_dynamics.md`'s two-learner result is "T4 (B12 two learners)".

| Result | Statement (one line) | Level; assumptions | Status | Literature | Experiment that supports / tests / bounds it (number; source) |
|---|---|---|---|---|---|
| **Lemma 1′** (profile-restricted one step) | $\operatorname{val}Q^*-\min_yx^\top Q^*y\le\varepsilon_s+x^{*\top}(Q^*-B)\hat y+\max_yx^\top(B-Q^*)y\le\varepsilon_s+2\lVert B-Q^*\rVert_\infty$; Theorem A′ = Theorem A with this per-state budget | M1 (A′: M2); A7 at one stage (A′: A1(i), A6, A7) | proved (numeric: 2000/2000 random checks, 0 violations) | adaptation / "new but small" (Per15 error propagation has the factor 2 and sup over profiles; Kov19b states no quantitative leaf-error bound) | exact abstracted endgames: along-chain budget/exploitability 1.6–9.4× against 7.6–39× for Theorem A; 89× vs 29× on the seat-2 top-move chain (`tightening/bounds_epsv.md` §2.1, one realisation; BR chain not deterministic) |
| **Signed pathwise identity** (Theorem A-res) | $z-\varphi(v_1)=\sum_t(-R_t-S_t+D_t)$ on every path; averaged over $s_0$ under SYM, $\mathbb E\operatorname{expl}=\mathbb E\sum_t(R_t+S_t-D_t)$ along $\sigma_{\rm BR}$ | M2 (M3 with $v_t$ the Bayesian LP value); A1(i), A6; SYM for the root term | identity proved (telescoping); root-term removal proved with named hypothesis SYM | known in structure (trajectory telescoping; Lob24 uses a signed decomposition then discards it; Per15 bounds sup quantities); new: the three-way split and its signed measurement | in-engine: closes to $10^{-14}$ on 284 games; drift per decision $-0.0127$ (exploiter, 60 games), $-0.0113$ (MCTS-2000, 100), $+0.0020$ (self-play, 100) (`bounds_epsv.md` §2.2). Real Showdown, open-item B1 (LOG, hedge_bayes vs Foul Play, 599 games): closes to $4.4\times10^{-16}$, **by construction** (telescoping); $\mathbb E[D]=-0.0548$ logit per decision (SE 0.0015 game-clustered). These are *not* Theorem A's terms here: the logged Bayesian LP strategy $x=Q[1]$ (Hedge's sampled expert) is not a maximin strategy of the world-averaged $\bar A$ (gap median 0.12, mean 4.6 engine units; above 1% of the payoff range on a third of decisions), and the played Hedge mixture has gap median 1.85, mean 6.12 (`engine/analysis/open_items/verify/b1_gap.py`), and leaf, mixture and opponent differ. Same formula on a different leaf and opponent; neither a replication nor a contradiction (`engine/analysis/open_items/REPORT.md` B1) |
| **$\varepsilon_v$ lower bound** (Theorem A backwards) | $\mathbb E_{\pi,\sigma}[\sum_{t<\tau}(\varepsilon_s+2\varepsilon_v)]\ge V^*_0-W^{\pi,\sigma}$ for any $\sigma$ (here the informed BR): the leaf error the measured exploitability *requires* | M2; Theorem A's hypotheses | proved (arithmetic on Theorem A) | not classified in the lit notes (bears on Kov19b's open question) | in-engine W1 (depth-2 informed best response, sampled target; updated 2026-11-17): Theorem A holds for every $\sigma$, so the exploiter's gain gives $\mathbb E[\sum_{t<\tau}(\varepsilon_s+2\varepsilon_v)]\ge0.18$ payoff (one-sided 95%; point 0.425); over the exploiter's mean battle length 57.2 turns (SE 5.9) $\varepsilon_v$ averages at least $\approx0.0015$ per turn (0.0037 at the point gain; joint 95% 0.0011); over the 52 non-forced turns of the 60 replays 0.0041 (point), 0.0017 (LCB). A necessary condition along the exploiter's battles, not a measurement of $\varepsilon_v$; sup-norm $\varepsilon_v$ is far below the endgames' $\varepsilon_v^{\rm aff}\approx1$ (a different set of states, so no fixed ratio) (`bounds_epsv.md` §1.3). **Not updated by LOG** (reason in the row above) |
| **Comparative identity**; "common $\varepsilon_v$ cancels" | for Markov $\pi_1,\pi_2$ under A1(i), A6: $V(\pi_2,\sigma)-V(\pi_1,\sigma)=\mathbb E_{\pi_1,\sigma}\sum_t(x_{2,t}-x_{1,t})^\top Q^{\pi_2,\sigma}_t\sigma_t$; the leaf enters only through row oscillation on the support difference. The naive statement that a shared $\varepsilon_v$ cancels between two agents is **refuted** | M2; A1(i), A6 | identity proved; cancellation refuted by counterexample ($\hat Q=0$, $Q=\varepsilon\begin{psmallmatrix}1&-1\\1&1\end{psmallmatrix}$: two exactly optimal strategies differ by $2\varepsilon_v$) | not classified in the lit notes | in-engine bridge-B12 replays: drift channel differs between agents sharing $\hat Q$ (top $-0.08\pm0.25$, sampled $+0.66\pm0.12$ per game) (`bounds_epsv.md` §1, rejected list). Oscillation term not measured |
| **B1 (Bayes) = Hedge, $\eta=1$, log loss** | $\sum_t-\ln\hat q_t(b_t)\le\min_s\sum_t-\ln q_{s,t}(b_t)+\ln k_s$, $k_s=5$ | M0; $\lambda=1$, uniform prior | cite + adaptation | known (Ces06 exponentially weighted forecasters; Rei14: constant regret $\ln\lvert\Theta\rvert/\eta$ iff $\eta$-mixable, log loss $\eta\le1$) | see N1 |
| **N1 regret identity** | $R_T(s)=\ln5+\ln P_T(s)$ for every $T$ and style $s$ | M0; $\lambda=1$, uniform prior | proved | not separately classified (exact form of the Bayes-mixture bound above) | BH: final regret 1.609 on all 8 matches with complete logs; open-item B6 (RQ), pre-registered P1: **pass to logging rounding** (anytime excess over $\ln5$ at most $3.9\times10^{-4}$; final regret 1.609 to 3 dp in 18 of 29 matches with $P_T\ge0.999$) (`REPORT.md` B6) |
| **T1 discounted Bayes** (coded "predict, then discount", $\lambda<1$) | no bound $R^\lambda_T\le f(k,\lambda)$ with unbounded losses; with the shipped cap $B=\ln(n/\mathrm{MIX})$ a 2-turn example exceeds $\ln k$ for $\lambda\le\lambda^*_2(n)$ ($0.951$ at $n=13$, $0.985$ at $n=44$); data-dependent bound $\ln k+(1-\lambda)\sum\lambda^{T-t}\operatorname{range}D_{t-1}$; **fix**: discount-then-predict (store $\lambda(\log p+\log\text{lik})$), bound $\ln k$ on every path | M0; L6 | (a)–(c) proved; (d) measured; (e) conjecture (violation for every $\lambda<1$ as $T$ grows) | new (small; no prior analysis of the predict-then-discount order found); discount-first $\ln k$ is known (Che10); $\eta>1$ non-mixable is known (Rei14) | no game run with $\lambda<1$ (shipped `LAM=1.0`, `behavior.py:29`); realisable violations on the 57 real style matrices at $\lambda\in\{0.5,0.7,0.9\}$ (`bayes_dynamics.md` §1.1) |
| **T2** MAP = FTL; oscillation | MAP is Follow-the-Leader on log loss; time on the wrong side $\approx\sigma^2/(2\mu^2)$ | M0 meta; L6 | (a),(c) proved; (b) sketch | known / adaptation (Ces06, Hut05, Kal05); measured data new | BH logs: 649 MAP-wrong turns vs 563 predicted (match means; order of magnitude only); FTL regret 5–18 nats vs mixture 1.609 (`bayes_dynamics.md` §2) |
| **T3 Bhattacharyya concentration** | realisable: $\mathbb E\sum_t(1-P_{t-1}(s^*))\le\sum_{s\ne s^*}1/(1-\rho_s)$, constant in $T$; misspecified Chernoff-½ version (vi) | M0 (M3 inputs); L6; T3.1 realisability, $\lambda=1$ | (i)–(v) proved; (vi) proved, hypothesis fails on the mismatch proxy | device known (Wal04, Zha06, Lat13 Lemmas 13/14); instantiation and calibration are adaptation | 57 real matrices under exact realisability: bound holds, 2–5× loose (`bayes_dynamics.md` §3.1). RQ realised $\sum_t(1-P_{t-1}(s^*))$, mean per match: 589.0 (stage equilibrium), 78.5 (cautious), 16.2 (uniform) (`REPORT.md` B6 P5). vs the i.i.d. T3 bound 15.9 / 55.3 / 38.0: holds for uniform (exactly calibrated), fails for stage equilibrium (realisability fails: q_safe out-predicts q_eq, 1.62 vs 2.20 nats) and on average for cautious (near-tie, overconfident model) (`analysis/open_items/t3/T3_RQ.md`) |
| **T4 (B12 two learners)** | two $\varepsilon$-safe players on a shared matrix: every iterate is a $2\varepsilon\Delta$-NE, time average a $2\varepsilon\Delta$-CCE; mismatch widens the band to $\varepsilon\Delta+2\delta$; Hedge vs safex: nothing bounds $\bar x_T$ | M0 shared / M1 mismatch; L7 + S | (a)–(c) proved (elementary); (d) measured; (e) conjecture | elementary / known by definition of $\varepsilon$-safety (Gan15); contrast with MWU/FTRL last iterate (Mer17, Das18); Jea23, Mul25 guarantee only cumulative quantities | 57 matrices × 400 rounds: band attained exactly ($\pm0.0030$); with $0.1\Delta$ noise the mismatched side loses $0.017\Delta$ per round (`bayes_dynamics.md` §5). Not run on Showdown (no safex-vs-safex tag) |
| **T5 detection delay after a switch** | $\lambda=1$: detection delay $d^*\approx\max_{s\ne b}t_0(\mu_{a\to b}-\mu_{a\to s})^+/\mu_{b\to s}$, linear in $t_0$; fixed share and discount-then-predict: independent of $t_0$ | M0; L6; i.i.d. turn matrices | (a) proved for the mean log-odds crossing; (b)–(c) sketch; **live-tested 2026-11-16 (prereg_style_switch.md)** | not separately classified (fixed-share comparator Her95) | simulated switches: $\lambda=1$ delays within 6% of (a) (`bayes_dynamics.md` §6). Open-item B2 (E3a/E3f) ran **Hedge without a posterior (anytime and fixed-share), not the posterior**, so T5 is not tested by that run (the live test is in §12); a reward-space analog gives realised crossing / analog median ratio 0.91 (E3a, 9 matches) and 0.97 (E3f, 4 matches) (`REPORT.md` B2) |
| **VOI $=\eta_{\rm NL}$; Theorem C at $H=2$** | VOI$_{\rm total}=\sum_\theta b_0V_0(\delta_\theta)-V_0(b_0)$ and $\eta_{\rm NL}(0)$ computed exactly by sequence-form LP in one-sided 1v1 tera subgames; sandwich $V_0\le v_{A3}\le V_0+\eta_{\rm NL}(0)$ | M3, $H=2$, $\varepsilon_s=\varepsilon_v=S=0$ by construction | numeric (exact LP per matchup); Theorem C proved (§4.3) | LP known (KMS94, vS96); applying it to measure non-locality/VOI in a real game's subgames is new empirical application (Lon10 used synthetic trees) | 46 subgames: sandwich, (C.1), VOI $\ge0$, expl $\ge0$ hold **46/46**; VOI $>0.1\%\Delta$ in 14/46, max 5.2%; expl(A3) max 3.1%, $>0.1\%\Delta$ in 3/46 (`tightening/voi.md` §2.1) |
| **Leaf nondeterminism** | `nash._leaf` → `pe.id(child, 0)` is time-budgeted; under CPU load its background search thread occasionally returns a complete deeper search (`depth_searched` $\ge2$, or a wrapped i8), and terminal children then score $100\cdot\text{depth}\cdot$`battle_is_over`; depth 1 is never truncated; fix: retry until `depth_searched == 1` (`OURBOT_DET_LEAF=1`) | engine (affects every M2–M4 number) | measured | known in computer-chess practice (TalkChess threads); not found peer-reviewed; new as a quantification for this agent (Vol20, Aga21 for reporting) | load probe: 16/240 (1v1) and 17/240 (6v6) children change value with 12 busy processes on 14 cores, 0/480 unloaded (`voi.md` §2.5). Real games: LSs (Bayesian LP vs Foul Play, 2-CPU containers, 120 games) 432 non-depth-1 returns in 7,246,430 leaf calls = 0.0060%, $\ge1$ in 59/120 games; LSn (Bayesian LP vs PIMC averaging, 1-CPU, 120 games) 850 / 16,172,097 = 0.0053% (both bots combined: the two bots share one process and one counter), $\ge1$ in 91/120; about 1 in 17,000–19,000 calls (1 in 40,000 in AFdet), similar rates in the two container classes. Scores LSs 0.2833 [0.2068, 0.3599], LSn 0.5917 [0.4981, 0.6853], 60 pairs each (`engine/showdown/results/{LSs,LSn}/*_summary.json`). Deterministic-leaf control AFdet: 0.3067 vs 0.3144, paired diff −0.008 [−0.057, 0.040], n = 299; no detectable effect (`analysis/open_items/out_afdet_paired.json`) |

T3 against the RQ data: the i.i.d. bound Σ_{s≠s*} 1/(1−mean BC_s) from the logged per-turn style predictives is 15.9 (stage equilibrium) / 55.3 (cautious) / 38.0 (uniform) vs realised 589.0 / 78.5 / 16.2. Holds for uniform (9/10 matches; exactly calibrated: cross-entropy 2.291 = model entropy). Fails for stage equilibrium (0/10) because realisability fails — q_eq on our world-averaged Abar predicts the stage-equilibrium opponent worse than q_safe (log loss 2.20 vs 1.62 nats/update), so the posterior correctly concentrates on cautious (explains P4). Fails on average for cautious (5/10): cautious is best but only 0.055 nats/turn ahead of best response to uniform, model overconfident (entropy 1.85 vs realised 2.05), one match at 268; not clearly significant with 10 matches. Source: `engine/analysis/open_items/t3/T3_RQ.md` (log-loss ordering independently re-checked).

Leaf: deterministic-leaf control AFdet (Bayesian LP vs Foul Play, `OURBOT_DET_LEAF=1`, seeds 1–300, same teams in 598/598 games): 0.3067 [0.2693, 0.3440] vs AF 0.3144; seed-paired difference −0.008 [−0.057, 0.040] (match-clustered [−0.070, 0.053], n = 299). Retry fired on all 826 non-depth-1 returns in 33,222,584 leaf calls (829 retries, 0 fallbacks). No detectable effect; results stand. Source: `engine/analysis/open_items/out_afdet_paired.json` (`afdet_paired.py`).

**Real-Showdown open items that test §8 predictions** (all from `../engine/analysis/open_items/REPORT.md`; pair
scores with 95% CI; match-clustered CIs have only 10 clusters per tag).
- **E3 (open-item B2), P2.** E3a (`hedge`) and E3f (`hedge_fs`) vs an opponent that switches from Foul Play to
  `exploit_top` after 8 of 16 seeds per match. Fixed share is no better than anytime Hedge: paired
  $+0.0032$ [$-0.066$, $0.072$] (157 seeds). The post-switch best expert reaches the maximum weight in 8/10 matches in
  both runs, median 273 (E3a) and 245 (E3f) updates after the switch; the fixed-share analog bound (median 5639 / 4895
  updates) is not exceeded and is far from tight. Weights stay within 0.2–0.35, so the "flip" is a ranking change,
  not a concentration. Only one $T_1$ was run, so the E3 refutation clause "the gap does not grow with $T_1$" is not
  testable from these runs.
- **E4 exploiter arm (open-item B3).** Model-based exploiters (pure best reply to the victim's predicted column; they
  know the victim's algorithm but not its sampled worlds or RNG) vs RM$^+$'s score against the same victim: only XU
  (vs `bandit_ucb1`, visit shares) gains clearly, $+0.0354$ [0.014, 0.057]; XUmv $+0.0518$ [$-0.002$, 0.106] (match-clustered [0.007, 0.097]: borderline); XEmv ($-0.0527$ [$-0.107$, 0.002]) and XR (vs RM$^+$, $0.4683$ vs 0.5 by symmetry) within noise.
- **E5 controls (open-item B4).** E5 0.4583, E5t (Bayesian LP, top move vs Bayesian LP) 0.5500 [0.513, 0.587], E5s (Bayesian LP vs
  Bayesian LP) 0.5185 [0.480, 0.557]; E5 − E5t $-0.0917$, E5 − E5s $-0.0621$ (seed-paired). Descriptive only: a causal
  split of the deficit (mixing weights vs expert quality) is not identifiable from the logs. On the 25.9% of decisions
  where top and pure maximin disagree, Hedge played top's move 59.7%, pure maximin's 31.3%; cause open.
- **PA2-type $\varepsilon$ sweep (open-item B5).** `bayes_safex` vs Foul Play at $\varepsilon\in\{0.001,0.003,0.01,0.03\}$:
  0.2000, 0.2275, 0.2450, 0.2538; logistic slope $+0.087$ per $\ln\varepsilon$ ($p=0.018$ clustered by tag×match;
  $p=0.040$ by match only); all four below Bayesian LP on the same seeds (0.3291, $n=199$) by 0.076–0.128.
- **RQ pre-registration (open-item B6; `bayes_dynamics.md` §4, committed fb3cc8e before the runs).** P1 pass to
  rounding; P2 partial (4 of 6 clauses pass; "$P\ge0.99$ by turn 50 in every match" 8/10 and "MAP-wrong turns $\le10$
  per match" 5/10 fail); P3 high-mismatch branch (selected by the pre-registered rule: gap 0.048 nats/turn $<0.1$)
  **fails** (final MAP = cautious 10/10), and neither branch strictly covers 0.048; point prediction (MAP = cautious in
  8–10/10) passes 10/10; P4 pass (P_T(stage equilibrium) < 0.1 in 10/10; speculative score 0.50–0.60 missed: 0.670); P5 pass.
- **Audit caveats (open-item A).** 14 error games across 9 tags (each drops its seat-swapped partner). In 12 matches
  the carried Hedge/posterior state counts more games than game files exist; hypothesis (not a finding): an aborted
  retry attempt updated the carried state before the resume rule (`modal_showdown.py:163`, skip seeds whose two game
  JSONs exist) reran it. Config fields are constant across invocations. Scores are unaffected.

---

## 12. Live tests and follow-up (2026-11-16)

Added 2026-11-16. Sources: `../research/prereg_style_switch.md` ("Results"), `../research/prereg_mw.md`,
`../research/joint_posterior.md` (corrected numbers), `../research/info_gathering_concealment.md` (corrected),
`../research/s1_results.md`, `../lit/joint_belief_voi_concealment.md` (known / adapted / new labels; the lit file calls
the middle label "adaptable"). Numbers and run details: `../engine/RESULTS.md`, "Benchmark and follow-up studies
(integration branch, 2026-11-16)". Status words as in §0.4.

| Item | Result | Status | Label | Evidence (source) |
|---|---|---|---|---|
| **T5 live-confirmed** | the P3 per-match test passes: median tau_map/tau_hat 1.06, Spearman rho = 0.86 over 20 matches (band [0.7, 1.4], rho >= 0.5); the P6 forgetting point prediction passes: SW4p (lambda = 0.98) median tau_map 37.5 against the predicted 35, ratio to SW4 0.17; P2 (delay scales with pre-switch evidence): ratio 2.09. One false alarm (SW2 match 7), below the refutation threshold; the SW4d arm was not run | live-tested 2026-11-16 (`prereg_style_switch.md`) | adapted (T5 is the §11 detection-delay result; the live test is new) | `prereg_style_switch.md` "Results"; `switch_results/out_style_switch.md` |
| **T3 / realisability** | per-world styles still do not make the equilibrium style realisable: stage equilibrium is the final MAP in 0/10 MWeq matches (R1; at most 4/10 predicted). The T3 realisability hypothesis still fails for a stage-equilibrium opponent | measured; T3's hypothesis fails (unchanged from §11) | adapted | `prereg_mw.md` "Results", R1 |
| **VOI (S1 measurement)** | the planner that updates the posterior inside the lookahead gains about 0 over its frozen-posterior control: mean 0.0022 % of range, A1 and A2 choose different actions in 3/3,620 decisions. Reason: poke-engine branches only on crit/kill, so soft evidence is almost absent and information arrives as hard eliminations that both arms credit. The clairvoyance bound at the children (A0L - A1) is mean 0.35 %. This is consistent with the small eta results in `tightening/voi.md` (along-play eta-hat on the order of 1 % of range, median near 0; 87 % tera-zero), which already say that inside the agent eta = 0 by A3 and the capturable value is small. Null accepted | measured (engine simulation); pre-registered E1b, E2a fail, E4 pass | adapted (the VOI = eta identification is §11; the S1 measurement is new) | `s1_results.md` Headline, §4, Decision; `tightening/voi.md` |
| **Concealment** | its value is measured: ex-ante mean 1.113 %, median 0.241 % of range (species known). The omniscient-opponent model (each world's opponent knows our true team) is the safe side by maxmin: x_AF guarantees v_AF against any opponent, including all type-blind ones. The comparison 0.948 % vs 3.4 % is not the argument | measured; safe side proved by maxmin (not by the numbers) | known (the principle) / new (the measured value in a real game) | `info_gathering_concealment.md` Headline 2, §1.3 |
| **Joint set x style** | the measured correlation term is about 0: a carried joint (active set x style) posterior adds +0.001 ± 0.001 nats over the factorised one. The real gains are restricting the likelihood to each world's legal actions (+0.324 ± 0.031 vs Foul Play; pooled +0.325 ± 0.027) and a per-world style likelihood (+0.028 ± 0.011). The analogy to HBA (Albrecht and Ramamoorthy 2013) is structural only | measured (offline, 18 games) | adapted (HBA-style beliefs); new (the measurement) | `joint_posterior.md` Corrections, §3.1 |
| **Two-sided regrets along real games** | the LOG run (hedge_bayes vs Foul Play, 599 games, 17,271 usable decisions) logs both sides. In our estimated stage games: our regret vs the best fixed expert 0.0044 ± 0.0002 per turn; the opponent's regret vs our equilibrium style 0.0213 ± 0.0006 (positive in 10/10 matches), negative vs the other four styles; per-turn best-response regrets 0.030 (ours, vs the realised column) and 0.041 (theirs, vs our mixture), not a symmetric pair. Reading: model mismatch, not the opponent's regret: a literal maximin opponent (CEQ) shows the same 0.021 ± 0.001, almost all of it mismatch (+0.025 ± 0.001); ε-safe opponents (CSS) show 0.011–0.015. Hedge's per-match worst-case bound is about 0.04 per turn | measured; CCE only with respect to small comparator classes in estimated games | new (measurement) | `../research/cce/cce_two_sided.md` (Corrections, §2.1, §2.3); `../research/cce/verification.md` |
| **Regret along real games (final paper)** | RM+ self-play median gap/range 7.0e-5 at T=1000 vs bound 0.38 on 35 in-engine stage matrices (slope −1.26 on T∈[10,2000], −1.05 on [100,2000]); Hedge regret per match 6.8–8.6, 0.13–0.16 of the anytime bound (3/2)√(T ln4/2)+√(ln4/8), 0.0045 per counted turn over 17,305 updates, growth exponent 0.82 on t∈[200,1600]; memory final regret 1.6094–1.6100 nats on 9 matches = ln5+lnΠ_T within 6.6e-4 (match 9 excluded, 12 unlogged updates); exact endgames: Theorem bound ≥ exact exploitability on all 22 rows (ratio 6.86–155.39) | measured; bounds hold on every logged row | new (measurement) | `../report/figures/fig_regret_numbers.json`, `fig_main_numbers.json`; `../engine/results/endgame/eps.jsonl` |

What this does not change: the leaf evaluation error dominating the bound (§11; the loss is charged to $\varepsilon_v$). Better prediction (restricting
the likelihood to each world's legal actions; `hedge_bayes_mw`) did not move the win rate in the replication (+0.0075 for seeds 201-400), the same
pattern as BH.

### 12.1 Verifier corrections (2026-11-17)

Dated 2026-11-17. These override the statements of the source notes where they differ.

- (a) $\eta_{\mathrm{NL}}$ does not shrink with a better belief: it depends on the game and on the per-world leaf values. The claim that only the leaf error does not shrink holds under full information (Theorem A), not for the hidden-team theorem (`full_game.md` §6, Theorem C).
- (b) The hidden-team theorem needs worlds drawn i.i.d. with equal weights from the legality-restricted prior, fresh and unseen by the opponent, and the matrix-error assumption holding per world.
- (c) The tight form of the whole-battle bound is $\bar H\cdot\bar\delta_{\mathrm{BR}}$, the average per-turn loss along the best response's path, which equals $\mathbb E_{\mathrm{BR}}[\sum\delta]$. This is what the endgame figure plots (`engine/endgame_eps.py`).
- (d) The Hedge reward uses a fixed scale $\bar\Delta=400$ evaluation units (at least the stage range on about 90% of Showdown turns). Per-turn normalisation admits linear payoff regret; the counterexample is in `hedge_experts.md` and `../research/`.
- (e) The Hedge-across-games safety part holds in estimated-matrix units only. A global affine calibration is degenerate on the endgames ($\alpha\le2\times10^{-9}$). On Showdown only the exploitation part carries over.
- (f) The Bayes = Hedge identity requires $\hat q$ to use the same per-world likelihood as the update.
- (g) The detection-delay formula needs i.i.d. per-turn inputs and $\mu_{b\to\theta}>0$, and covers $\lambda=1$ only.

### 12.2 Final paper to notes map

| Paper result | Notes |
|---|---|
| Theorem MWU / Hedge; no-regret dynamics; CCE | Background (cited; lectures 6 and 7) |
| Lemma, sum of per-turn losses | `search_rl_links.md` L1 |
| Whole-battle theorem | `full_game.md` Theorem A |
| Corollary, no-regret solver | `full_game.md` §4 (RM+ bound) |
| Hedge across games | `hedge_experts.md` E1–E3; `search_rl_links.md` L2; `regret_theory.md` E2′ |
| RL remark | `search_rl_links.md` L3 |
| Hidden-team theorem | `full_game.md` Theorem C |
| Leaf remark | `full_game.md` Lemma 2 |
| Bayes = Hedge equation | `experts_extensions.md` B1 |
| Detection-delay equation | this file (T5, §11); `experts_extensions.md` |
