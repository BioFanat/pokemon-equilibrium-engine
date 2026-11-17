# Results: equilibrium play and Bayesian-LP solving in Pokémon Random Battles

Status 2026-11-15 (real-Showdown section added). All numbers below come from files in `results/` and are recomputed by
`figures/make_figures.py` and the `summary` commands of `experiments.py` / `hidden_experiments.py`.
Terms follow `figures/FIGURE_VOCAB.md`. Literature cross-references and citations are added
in the section "Relation to prior work", after the literature review.

## Headline

1. **Hidden information: the Bayesian-LP solver beats PIMC averaging.** It scores
   **0.585 [0.540, 0.630]** over 400 games with equal compute (4 sampled worlds each).
2. **The PIMC-averaging loss bound holds on real play.** Across 46,890 real decisions, PIMC-averaging loss never
   exceeded the 4·δ_H bound; the largest loss was 0.098 of it.
   - PIMC averaging loses value in 61% of decisions, but the median loss is only 0.19% of the payoff range.
   - The tail carries the effect: 27.7% of decisions lose at least 1%, and 0.7% lose at least 10%.
3. **Full information: matrix solving beats MCTS only with 25× more time, and not by mixing.**
   Against MCTS at 2,000 iterations (2.7 ms per move), both deterministic bots win:
   - Equilibrium, top move: **0.58 [0.54, 0.62]**
   - Pure maximin move: **0.56 [0.52, 0.60]**

   Against time-matched MCTS (75 ms per move, X1) both lose clearly: top move **0.345
   [0.289, 0.401]**, Pure maximin move 0.263 [0.210, 0.315]. Depth-2 matrix solving is not competitive
   with MCTS per unit of compute. Sampling the equilibrium scores only 0.44 [0.40, 0.48] vs MCTS
   at 2,000 iterations. Against an opponent that doesn't adapt, randomizing only costs value.
4. **Infrastructure finding.** poke-engine's iterative-deepening matrix is mostly pruned: 55–80% of
   cells are missing. Treating it as a payoff matrix silently breaks every equilibrium method; our
   first round of experiments (E1–E3) was invalidated this way.
5. **Plain self-play RL is cheap per decision but weak.** After 30M decisions (25 min on a laptop),
   PPO wins 5–13% against MCTS (200 iterations requested, 1,000 run: poke-engine rounds up to multiples of 1,000), in a simpler 12-species, full-information setting.

## Setup

- **Simulator.** poke-engine 0.0.48 (Rust), built for Gen 9 with Terastallization. Each turn we
  enumerate every chance outcome with its probability, sample one, and apply it.
- **Teams.** Showdown's published Gen 9 Random Battle set data (`data/`, 509 species, 873 roles),
  loaded by `randbats.py`. Sampling picks six species, then per species a role uniformly, 4 moves
  uniformly from the role's pool, and an ability, item and Tera type.
  - This approximates Showdown's generator; it skips constraints such as guaranteed same-type
    attacks.
  - Engine coverage: every species, move and ability is supported; 3 rare items (7 of 873 roles)
    have no effect.
- **Payoff matrix (`nash.full_matrix`).** An unpruned depth-2 expectiminimax matrix. Each cell is
  the probability-weighted average, over chance outcomes, of the heuristic evaluation function.
  - The heuristic evaluation function is the midpoint of the two sides' maximin values on the child position, taken
    from the engine's depth-1 search.
  - Side one's maximin alone, which is the engine's own leaf rule, made the matrix strongly
    asymmetric: swapping sides changed cells by a median 32% of the payoff range. With the midpoint
    the swap is exact in the median position (worst 31%).
  - Cost: about 61 ms per matrix. Cells agree with the engine's own computed cells within a median
    2.2% of the payoff range (worst 9%, from damage-roll branching). See `test_matrix.py`.
- **Solvers.** Regret Matching+ (alternating updates, linear averaging) and an exact LP. RM+ at 100
  iterations is indistinguishable from the LP in play: 0.501 and 0.504 head-to-head over 400 games
  each (E3; on the pruned matrix).
- **Hidden information (`hidden.py`, `hidden_agents.py`).**
  - Each side sees its own team and what the opponent has revealed: species on the field, moves
    when used, Tera when used. HP is shown as a percentage. Items and abilities stay hidden unless
    an item is consumed or knocked off.
  - The belief is the exact posterior over each revealed Pokémon's sets given the revealed moves and
    Tera type. Unrevealed slots are drawn from the prior.
  - Sampled worlds are full states consistent with the view. With the belief set to the true sets,
    rebuilding reproduces the true state exactly in 922 test positions. Two exceptions: 37 differ
    only in choice-lock flags, and 33 positions with in-battle stat changes were skipped.
  - The four agents:
    - **Oracle:** solves the true state.
    - **Most-likely set:** solves one world built from each revealed Pokémon's most probable set.
    - **PIMC averaging:** averages the equilibria of 4 sampled worlds.
    - **Bayesian LP:** solves max_x Σ_k β_k min_j (M_kᵀx)_j over the same 4 worlds as one LP (the maximin LP of the Bayesian matrix game, the one-sided special case of Ponssard and Sorin's (1980) LP for zero-sum games with incomplete information; "Bayesian matrix game" is PokaiTrainer's name), with
      the opponent best-responding in each world.
- **Evaluation.** Games come in pairs with the same teams and seed and sides swapped. Score = wins +
  ½ draws, and 95% CIs are taken over pairs. Games stop after 300 decision nodes (engine steps, including forced switches), which counts as a draw.

## Results

### Fixing the payoff matrix, full information (Fig. 1a; `results/experiments/E1fix.jsonl`, `E1.jsonl`)

| Agent vs MCTS (2,000 iterations) | Pruned matrix | Full matrix | Time per move (agent / MCTS) |
|---|---|---|---|
| Pure maximin move | 0.434 [0.393, 0.475] | **0.561 [0.518, 0.604]** | 67 ms / 2.7 ms |
| Equilibrium, sampled | 0.302 [0.257, 0.348] | 0.443 [0.402, 0.483] | 74 ms / 2.7 ms |
| Equilibrium, top move | — | **0.581 [0.540, 0.623]** | 72 ms / 2.7 ms |

- Equilibrium, sampled vs Pure maximin move: 0.516 [0.475, 0.558], a tie.
- The pruned matrix's signature was a side split. The bot did fine as side one and collapsed as
  side two: Pure maximin move scored 0.618 as side one and 0.250 as side two (E1, 200 games per side).
  Side-two maximin needs the cells that pruning drops.
- The full-matrix bots use about 25× more time per move than MCTS at 2,000 iterations. At
  matched time (X1, below) MCTS wins.
- 10–13% of games between the deterministic bots and MCTS hit the 300-decision-node cap.

### Hidden information (Fig. 2; `results/hidden/E4.jsonl`, `E4am.jsonl`)

| Matchup (score of first-named) | Sampled play | Top-move play |
|---|---|---|
| Bayesian LP vs PIMC averaging | **0.585 [0.540, 0.630]**, n=400 | 0.540 [0.452, 0.628], n=100 |
| Bayesian LP vs Most-likely set | 0.440 [0.338, 0.542], n=100 | 0.525 [0.452, 0.598], n=100 |
| PIMC averaging vs Most-likely set | 0.535 [0.455, 0.615], n=100 | 0.560 [0.466, 0.654], n=100 |
| Oracle vs Bayesian LP | 0.600 [0.513, 0.687], n=100 | 0.620 [0.525, 0.715], n=100 |

- **Bayesian LP vs PIMC** is the only matchup run at full size. The first 50 pairs scored 0.60 and
  the next 150 scored 0.58, so the result is stable.
- **Equal compute in that matchup:** both use 4 worlds at about 340 ms per move.
- **Oracle:** knowing the hidden sets is worth about 0.1 in score under both kinds of play. That gap
  is the room available to better belief modelling.
- **Most-likely set** uses one world (about 85 ms per move against 320–340 ms). Its matchups have
  only 100 games and are inconclusive.
- **Top-move play** removes the cost of sampling but adds draws: 10–12% of games hit the turn cap
  in two matchups.

### PIMC-averaging loss vs the bound (Fig. 3; `results/hidden/E5_decisions.jsonl`)

Every Bayesian LP and PIMC decision in E4's sampled play was logged: 46,890 decisions.

| Quantity | Value |
|---|---|
| Decisions where PIMC averaging loses value (> 1e-6) | 61.0% |
| PIMC-averaging loss, median / 90th / 99th percentile / max (share of payoff range) | 0.19% / 2.6% / 8.8% / 25.0% |
| Decisions with loss ≥ 1% / ≥ 10% of range | 27.7% / 0.7% |
| Bound violations (loss > 4·δ_H) | **0** |
| Largest loss / bound | 0.098 |
| Top move differs between Bayesian LP and PIMC averaging | 14.9% of decisions |
| Median loss when the top move differs / is the same | 1.33% / 0.06% |

- The bound is valid but loose, by a factor of 10 or more.
- The Bayesian-LP agent's advantage comes from a minority of decisions where the worlds disagree enough that
  averaging per-world equilibria picks a different action (PIMC assumes it can use a different strategy in each world although the worlds belong to one information set: strategy fusion).

### Self-play RL baseline (Fig. 1b; `results/ppo_pilot.jsonl`)

- **Setup:** PPO self-play with a 256-256 MLP, full information, the 12-species hand-built pool;
  30M decisions in 25 min on 12 cores.
- **Result:** beats random at about 95%, but wins only 5–13% against MCTS (200 iterations requested, 1,000 run: poke-engine rounds up to multiples of 1,000), rising
  slowly.
- **Comparability:** it ran in a simpler setting than everything above, so it is a cost and
  learning-curve reference only.

### Regret along real games (`report/figures/fig_regret_numbers.json`, `fig_main_numbers.json`; scripts `make_regret_figure.py`, `make_headline_figure2.py`, `make_wins_figure.py`)

- **RM+ self-play on in-engine stage matrices.** On 35 stage matrices from the engine, the median gap divided by the payoff range is 7.0e-5 at T = 1000, against a bound of 0.38. The median gap decays with log-log slope −1.26 on T ∈ [10, 2000] and −1.05 on T ∈ [100, 2000].
- **Hedge regret per match.** Regret against the best fixed expert is 6.8–8.6 per match, which is 0.13–0.16 of the anytime bound (3/2)√(T ln4/2) + √(ln4/8). Pooled over 17,305 updates it is 0.0045 per counted turn; the growth exponent on t ∈ [200, 1600] is 0.82.
- **Memory (style posterior).** The final regret of the memory expert is 1.6094–1.6100 nats on 9 matches, equal to ln5 + lnΠ_T within 6.6e-4. Match 9 is excluded because 12 of its updates are unlogged.
- **Exact endgames.** The Theorem bound is at least the exact exploitability on all 22 rows (`results/endgame/eps.jsonl`, `endgame_eps.py`); the ratio is 6.86–155.39.

### No-regret dynamics: last iterate vs average (`theory/dynamics_directions.md` §2.4, `b3.out`)

On 120 in-engine stage matrices (median 5×6, range 1), the average iterate of Hedge / OMWU (η = 0.1, 1,000 iterations) has median gap 0.032 / 0.032, and the last iterate has median 0.0087 / 0.0085 (p90 0.092 / 0.086). RM+ has median gap 0.0000. Most real stage matrices have pure saddles, which is why the last-iterate distribution is bimodal. For the Hedge-vs-equilibrium-expert comparison (E5, 0.4583 [0.4210, 0.4956], 300 pairs, inside the predicted band [0.45, 0.55]) see "Real-Showdown experiments" below (E5 row).

## Conclusions

1. **Hidden information:** the Bayesian-LP agent beats PIMC averaging at equal compute,
   0.585 [0.540, 0.630]. This is the project's central claim and holds at 400 games. The
   Bayesian matrix game itself comes from PokaiTrainer [Yu26]; its maximin LP is the one-sided special case of Ponssard and Sorin's (1980) LP for zero-sum games with incomplete information; "Bayesian matrix game" is PokaiTrainer's name; the head-to-head comparison is ours.
2. **The bound holds but is loose.** 4·δ_H always upper-bounds PIMC-averaging loss on real decisions, and
   a hand argument tightens it to 2·δ_H (see Relation to prior work). As a gate for when to run the
   joint LP, δ_H is weak (X3), and it isn't needed: Bayesian LP costs the same as PIMC.
3. **No-regret solving is enough per turn.** On the full matrix, RM+ reaches a duality gap of
   2·10⁻⁴ of the payoff range in 1,000 iterations (`test_matrix.py`). In play, RM+ at 100 iterations
   tied the exact LP (0.501 [0.462, 0.541], E3), but E3 used the pruned matrix, so that is evidence
   about solvers only. The bottleneck is the matrix (depth, heuristic evaluation function, hidden information), not
   the solver.
4. **Randomizing trades value for robustness.** Against fixed MCTS, sampling the equilibrium costs
   about 0.14 compared with its top move (0.44 vs 0.58). Against an exploiter that knows the bot's
   algorithm (E6q), the ranking flips: sampled 0.475, pure maximin 0.367, top move 0.317. Quick test,
   60 games each; the exploiter looks one turn ahead.
5. **Information is valuable.** The oracle gains about 0.1 over Bayesian LP. Better beliefs (item and
   ability inference, behaviour-based updates) are the clearest lever.
6. **Methodology:** search matrices from pruning engines are not payoff matrices. Any equilibrium
   approach built on such an engine must recompute the pruned cells.
7. **The leaf evaluation error dominates the bound (consistent with the evidence), and a cheap learned value doesn't fix it.** Time-matched MCTS
   beats depth-2 matrix play (0.345, X1). A value net fit to game outcomes predicts no better than
   the engine heuristic and makes a weaker bot (0.235 vs time-matched MCTS, V1).

## Theory status: proposal goals vs evidence

Framing: each turn is a finite zero-sum Bayesian matrix game G(s, b, V̂). No-regret dynamics
(RM+ for us, one regret learner per opponent world) converge on average to its equilibrium, which
is our maximin value. Being "no-regret" is a property of the solver, not of Pokémon. Across
battles, choosing among whole strategies is an experts/bandit problem (L6 in
`../notes/math_framing.md`).

| Proposal goal | Formal statement | Evidence | Status |
|---|---|---|---|
| No-regret search approaches equilibrium | L1: RM+ average → Nash of the per-turn game, gap O(R/√T) | Gap 2·10⁻⁴·R at T=1000; RM+100 ≈ LP in play (pruned matrix) | Holds, per-turn game only. Says nothing about the full game (L4). |
| stage exploitability ≤ solver + belief + value error | L2: v(b) − f_b(x̂) ≤ ε_T + 2Δ·TV(b, b̂) + 2‖M − M̂‖_∞ (Δ = payoff range; 2Δ·TV = 4·TV on [−1,1]) | Random instances: max ratio 0.48. In play: solver term ≈ 0; belief term ≈ the 0.10–0.12 oracle gap; value term dominates (X1, V1) | Bound checked; the ranking of terms is measured, not proved |
| Optimizing per hidden set is exploitable | Strategy fusion; PIMC-averaging loss ≤ 4·δ_H (hand: 2·δ_H) | Bayesian LP beats PIMC 0.585 [0.540, 0.630]; bound holds on 46,890 decisions | Strongest result; the 2·δ_H proof still needs the `prove` workflow |
| Memory sufficiency / compression | Belief-state theorem (cite); L2′ marginals loss ≤ 2Δ·TV (Δ = payoff range; = 4·TV on [−1,1]) | Belief is the exact joint posterior over Random Battle roles; no marginals ablation run | Not tested |
| Value of information | L3: v(b) convex, VOI ≥ 0, = 0 iff the prior optimum stays optimal | Oracle gap bounds total VOI at ≈ 0.1 score; exact VOI in one-sided tera subgames at H = 2 (`theory/tightening/voi.md` §2.1) | Measured at H = 2 (46 subgames); full games not measured |
| Behavioural evidence / bluffing | L4 (safe resolving, public belief states), L5 (Aumann–Maschler cav u) | Hard evidence only; leaves ignore beliefs, so concealment has zero value in our model | Not tested; needs belief-dependent leaves |
| When to randomize | Value of randomization v(b) − max_a f_b(e_a) | Top move beats sampled vs fixed MCTS (0.58 vs 0.44); vs a one-turn exploiter sampled 0.475, top move 0.317 (E6q) | Tested (quick): mixing costs vs fixed opponents, protects vs exploiters |
| Adapt across battles | L6: EXP3 over {equilibrium, exploiters}, regret O(√(k log k / N)) | Offline replay (X4, X4b): regret within bounds; with bandit feedback, switch tracking needs far more battles. Live with per-turn feedback (X6): matches the best expert vs MCTS (0.600), switches to sampled within ~5 battles of meeting an exploiter | Tested (small runs) |


## Caveats

- Matrices are depth 2, with payoffs in the engine's heuristic evaluation units, not win
  probability.
- The 0.58 win over MCTS is not compute-matched; at matched time MCTS wins (X1).
- Most hidden-information matchups have only 100 games (CI about ±0.09), and each matchup is a
  single run.
- Simplifications in the hidden-information model:
  - Moves count as revealed when chosen, even if they never execute.
  - Items and abilities are never inferred.
  - Choice-lock flags are re-derived (exact in 885 of 922 checked positions).
- The team generator is an approximation of Showdown's.
- Long games: 2–30% of games hit the 300-decision-node cap (up to 40% for LBR in the informed-best-response runs), and they count as draws.

## Exploratory runs (2026-11-14)

- **X1, time-matched MCTS** (`results/experiments/X1.jsonl`, 200 games each, MCTS at 75 ms per
  move): Equilibrium, top move 0.345 [0.289, 0.401]; Pure maximin move 0.263 [0.210, 0.315]. The E1fix
  win came from compute, not from the matrix method.
- **X2, 8 vs 4 sampled worlds** (`results/hidden/X2.jsonl`, both Bayesian LP with top-move play,
  100 games): 0.495 [0.414, 0.576]. There's no sign that more worlds help, at 2× the time per move
  (735 vs 367 ms).
- **X3, δ_H as a gate (offline on E5 decisions):**
  - Running Bayesian LP on the half of decisions with the largest δ_H/range catches 70% of
    top-move changes and 81% of PIMC-averaging loss. Spearman(δ_H, loss) is 0.53; the AUC for spotting a
    changed top move is 0.65.
  - Gating also saves almost nothing: Bayesian LP and PIMC cost the same per move (343 vs 344 ms),
    because building the matrices dominates and δ_H needs the same matrices. Direction A is
    dropped: always run Bayesian LP.

- **X4, EXP3 over whole-battle experts (offline replay, `exp3_replay.py`, `results/exp3/replay.json`):**
  - Setup: experts are Equilibrium top move, Equilibrium sampled and Pure maximin move. Each round's
    reward is a recorded game score drawn with replacement; seeds differ across matchups, so this
    is a bootstrap stochastic bandit. 2,000 repeats per scenario.
  - Results, mean score per battle (regret vs the best fixed expert, in battles):

    | Scenario | Best fixed | EXP3 | Hedge (full info) | Uniform |
    |---|---|---|---|---|
    | vs MCTS-2000, N = 400 | 0.581 | 0.541 (16.0) | 0.566 (6.2) | 0.528 (21.2) |
    | vs MCTS-2000, N = 4,000 | 0.581 | 0.557 (95.4) | 0.575 (26.7) | 0.528 (212.2) |
    | MCTS-2000, then MCTS 75 ms, 2 experts, N = 400 | 0.463 | 0.440 (9.2) | 0.448 (5.9) | 0.438 (10.2) |

  - Reading: regret stays well inside the bounds (EXP3 95.2 / 301.0 / 61.7). EXP3 drops the sampled
    expert (17% → 4% of picks in the last quarter), but even after 4,000 battles it can't separate
    top move from Pure maximin move (a 0.02 gap). Experts-across-battles only pays off when experts
    differ a lot against a given opponent, i.e. against an exploitable opponent with an exploiting
    expert. The switching scenario is uninformative: the top move is best in both phases.

- **E6q, quick exploitability test (`results/experiments/E6q.jsonl`, 60 games each).**
  - Exploiter: a local best response (`lbr:*`) that knows the target's algorithm, rebuilds the
    same full matrix each turn, computes the target's strategy and plays the best reply in it.
    Only one turn deep, so this is a lower bound on exploitability.
  - Score of the target vs its exploiter:

    | Target | Score vs exploiter | vs MCTS-2000 (E1fix) |
    |---|---|---|
    | Equilibrium, sampled | **0.475 [0.377, 0.573]** | 0.443 |
    | Pure maximin move | 0.367 [0.268, 0.466] | 0.561 |
    | Equilibrium, top move | 0.317 [0.232, 0.401] | 0.581 |

  - Per turn, the best reply to the equilibrium mix gains exactly 0 (checked on test positions),
    against up to 15% of the payoff range for the top move.
  - Reading: the deterministic bots lose about 0.13–0.18 to an exploiter; the sampled bot is not
    distinguishable from 0.5. The ranking flips relative to fixed MCTS. That is the
    robustness–exploitation trade-off theory predicts, and it matches PokaiTrainer's informed best
    response (77.7% vs argmax, 49.6% vs the full mix). The CIs are wide; the sampled-vs-top
    difference (0.16) is about at the edge of significance at 60 games each.
  - Draws: 18 of 60 top-move games hit the 300-decision-node cap (deterministic loops).
- **X4b, EXP3 against the exploiter (`results/exp3/replay_vs_exploiter.json`, 1,000 repeats).**
  The exploiter knows which expert is played each battle.

  | Scenario | Best fixed | Best per phase | EXP3 | EXP3.S | Hedge | Fixed-share Hedge | Uniform |
  |---|---|---|---|---|---|---|---|
  | exploiter only, N = 400 | 0.475 | — | 0.406 | 0.399 | 0.454 | 0.444 | 0.386 |
  | MCTS-2000 for 300, then exploiter for 300 | 0.464 | 0.528 | 0.457 | 0.459 | 0.460 | 0.478 | 0.458 |

  - Against the exploiter, all learners move to the sampled expert. Hedge plays it 98% of the time
    in the last quarter; EXP3 only 56%.
  - When the opponent switches, plain EXP3/Hedge track only the best fixed expert (barely above
    uniform). Fixed-share Hedge partly tracks the switch (0.478, 63% sampled at the end). With
    bandit feedback, EXP3.S doesn't, at 300 battles per phase.
  - Implication for L6: the experts layer works when feedback is rich. With win/loss bandit
    feedback, it needs far more battles than a phase of an adaptive opponent lasts.

- **X5, hidden-information exploiter (2026-11-14, `results/hidden/X5.jsonl`).**
  - Exploiter `xlbr:<target>`: knows the target's algorithm and the true state. Each turn it
    rebuilds the target's own view and belief, recomputes the target's strategy q with its own
    resampled worlds (a separate RNG, so the game's randomness is unchanged), and plays the best
    reply to q in the true full matrix. One turn deep.
  - Targets: Bayesian LP with top-move play and with sampled play, 20 pairs (40 games) each.
  - Control: the exploiter differs from the Oracle only in knowing q. So exploitation gain =
    (target vs Oracle) − (target vs exploiter). Oracle results: Bayesian LP scores 0.38 with
    top-move play (E4am) and 0.40 with sampled play (E4).
  - Prediction: the gain is larger for top-move play than for sampled play. q is computed from
    resampled worlds, so even top-move play is only partly predictable.
  - Code: `hidden_agents.Agent` (`xlbr` kind, `exploit`); `play_hidden` passes the opponent's
    view and belief to `xlbr` agents.
  - Result (score of Bayesian LP, 40 games each; CIs about ±0.13–0.16):

    | Bayesian LP play | vs exploiter (X5) | vs Oracle (control) | Exploitation gain |
    |---|---|---|---|
    | Sampled | 0.400 [0.270, 0.530] | 0.400 (E4) | 0.00 |
    | Top move | 0.350 [0.193, 0.507] | 0.380 (E4am) | 0.03 |

  - Reading: knowing the agent's algorithm adds nothing detectable beyond knowing the true teams.
    Under hidden information, even top-move play is effectively randomized: the move depends on
    which 4 worlds the agent samples, and the exploiter can't reproduce that draw. The full-info
    gap (0.18 for top move, E6q) doesn't carry over at this sample size; the 0.03 is well inside
    the noise.
- **X6, live experts learner with per-turn feedback (2026-11-14, `exp3_live.py`,
  `results/exp3/live_seed{0,1,2}.jsonl`).**
  - Experts: top move, sampled and pure maximin, all from one full matrix per turn, so the learner costs
    the same as one matrix bot.
  - Each turn: play the mixture p = Σ w_e q_e. After seeing the opponent's actual move b, reward
    every expert r_e = (q_e · A[:, b] − min A) / range(A).
  - Update: Hedge (η = 0.2) plus fixed share (α = 0.002) so it can track a switch. This is the
    "richer feedback" alternative to bandit EXP3, which gets one win/loss bit per battle.
  - Opponents in sequence: MCTS-2000 for 40 battles, then an exploiter that knows p and plays
    argmin_b p · A[:, b] in the same matrix. 3 seeds, sides alternate.
  - Prediction: the weight moves to top move against MCTS and to sampled against the exploiter,
    within a few battles rather than the hundreds bandit EXP3 needed (X4b). Compare phase scores
    with the fixed experts (MCTS: 0.58 / 0.44 / 0.56; own exploiter: 0.32 / 0.48 / 0.37).
  - Caveats: η and α are not tuned. The per-turn reward uses the heuristic matrix, not win
    probability. The exploiter is myopic.
  - Result (fixed exploiter, 3 seeds × 80 battles; unit-checked: the exploiter's reply is the
    best reply in 15 forced and 248 mixed positions):

    | Phase | Learner score | Fixed experts in the same setting |
    |---|---|---|
    | vs MCTS-2000, battles 0–39 | 0.600 ± 0.084 | top move 0.58, pure maximin 0.56, sampled 0.44 |
    | vs exploiter, battles 40–49 | 0.417 | vs own exploiter: sampled 0.475, pure maximin 0.367, top move 0.317 |
    | vs exploiter, battles 50–59 | 0.400 | |
    | vs exploiter, battles 60–79 | 0.458 | |

  - Mean weights over seeds (top move / sampled / pure maximin):
    - start 0.35 / 0.31 / 0.34;
    - battle 9: 0.43 / 0.23 / 0.34;
    - battle 39: 0.43 / 0.23 / 0.34;
    - battle 44 (5 after the switch): 0.27 / 0.47 / 0.27;
    - battle 59: 0.20 / 0.61 / 0.19.
  - Reading: with per-turn full-information feedback, the learner matches the best expert against
    MCTS. It moves to the sampled expert within about 5 battles of the switch, where bandit EXP3
    on win/loss didn't move in 300 (X4b). Its exploiter-phase score approaches the sampled
    expert's level by the last 20 battles. Small n: phase CIs are about ±0.08, sub-phase ones
    wider.
  - An earlier run with a buggy exploiter (it played its first option whenever the learner was
    forced) is kept in `results/exp3/buggy_exploiter/` with a note; don't use it.
- **Convergence rates (2026-11-14, computed inline, no new battles).**
  - Per-turn solver, RM+ on 35 full matrices (duality gap as a share of the payoff range):

    | Iterations | 10 | 100 | 1,000 | 2,000 |
    |---|---|---|---|---|
    | Median gap | 2.0e-2 | 7.2e-4 | 7.0e-5 | 2.8e-5 |
    | 90th percentile | 3.9e-2 | 2.5e-3 | 2.0e-4 | 1.1e-4 |

    - Gap ≈ T^−1.3 (median slope −1.26 at seed 0; −1.24 to −1.39 across seeds 1–3; per-position slope
      quartiles −1.51 / −1.26 / −1.14, T ≥ 10). Source: report/figures/data/rmplus_gap.json (35 real stage
      matrices, seed 0). On these matrices the gap is at most 0.04 of the proved bound 2Δ(√n+√m)/√T;
      the 0.12–0.20 ratio is from synthetic random matrices (theory/checks/check_rmplus*.py).
    - An earlier inline computation reported −1.46; it is not reproducible and is superseded by report/figures/data/rmplus_gap.json.
      This is much faster than the worst-case O(1/√T) guarantee, as usually seen for RM+ with
      alternation and linear averaging.
  - Experts learner (X6): per-turn Hedge with fixed share.
    - Theory: log(w_i/w_j) drifts by η·Δr per turn, where Δr is the mean per-turn reward gap. So
      the time to switch after an opponent change is about |log ratio at the switch| /
      (η·Δr·turns per battle). Fixed share keeps every log weight ≥ ln(α/k), so recovery is at
      most ln(k/α)/(η·Δr) turns.
    - Fitted drift of log(w_sampled/w_top) after the switch: 0.12–0.23 per battle, which implies
      Δr ≈ 0.011–0.025 of the payoff range per turn.
    - Predicted switch time: about 1–3 battles. Observed: the sampled weight passed top move at
      battles 40, 43 and 42 (0–3 battles after the switch). Worst-case recovery bound: about
      30 battles.
    - With fixed η the regret bound is ln k/η + ηT/8 per turn, linear in T. That is the price of
      tracking; a decaying η would give the √T rate but adapt more slowly.
- **Deferred:** E6q at 200 pairs per target (≈1,000 more games, ≈45 min on 6 workers).

## Theory and infrastructure, continued (2026-11-14)

### Which parts of the theory are new
Most pieces are known results put together for this agent:
- Lemma 1 is a standard perturbation bound.
- Theorem 1's induction is Shapley 1953 plus the RL simulation lemma.
- The RM+ rate is Burch–Moravčík–Schmid 2019.
- The sampling term is uniform convergence.
- η_NL is the value of revealing the opponent's team to both sides (ours; it bounds the error Long et al. 2010 call non-locality, kept separate from strategy fusion; cf. Frank & Basin 1998; Kovařík et al. 2023).

Ours are three things: the H̄ = best-response-length refinement with its counterexample, the necessity of the TV term (the 3-team counterexample), and the Hedge composition below.

### Hedge over experts: safety theorem (`theory/hedge_experts.md`)
- **Proposition E1 (safety, holds against any adaptive opponent).** With a fixed reward normaliser Δ̄ and an ε_s-equilibrium expert in the pool, the normalised stage payoff of the played mixtures p_t against the opponent's realised moves is at least Σ_t (val(Â_t) − ε_s) minus Δ̄·Reg_S.
- **Theorem E2.** Averaged over N games, V* − E[u] ≤ H̄_σ(ε_s + 2ε_v) + Δ̄√(H ln k/(2N)) for every opponent σ, where H̄_σ is the mean game length against σ (η fixed and tuned to N·H; anytime tuning (the code's indexing) multiplies the tuned-η term by 3/2 and adds Δ̄√(ln k/8)/N, Cesa-Bianchi & Lugosi 2006 Thm 2.3). It bounds the average over games, not the exploitability of any single game's policy.
- **Proposition E3 (exploitation).** The learner matches the best expert only along the realized trajectory. That is external regret, not policy regret (Arora, Dekel & Tewari 2012).
- **Two parts of the original sketch were false:**
  - The code's per-turn normaliser range(A_t) has a linear-regret counterexample: alternating A and 0.01·A loses S/3.
  - X6's constant η = 0.2 gives the bound 5.49 + 0.035·S, which is linear.
- **Checks:**
  - Lean, `theory/lean/FullGame/Hedge.lean`: no `sorry`, standard axioms only. The per-step Hoeffding inequality is a hypothesis.
  - Numerics, `theory/checks/hedge_check.py`: 0/32 violations, and the counterexample confirmed.
  - Checked by an independent verifier (2026-11-14). E1–E3 hold for constant η. The anytime update as first coded did not satisfy them; it is now fixed (below).
- **Code (`exp3_live.Learner`):** rewards (q_e·A[:,b] − min A)/400, clipped to [0,1]. range(Â) > 400 (the clipping term is nonzero) on about 6% of learner turns (95% CI 3–10%) and 8.5% of random start positions; a reward is actually clipped on about 0.5% of learner turns (verifier scratch runs, 2026-11-15, not yet a logged experiment; an independent resample of 2,272 positions gives p99 ≈ 405–417 and median ≈ 230, not the 166/203 quoted earlier). The anytime step was first coded as `logw += η_t r_t`, which is not anytime Hedge and has linear regret (0.40·T on a phase-switch sequence; E1(b) fails at S=100). It is now fixed to the FTRL form `logw = η_t·Σ_s r_s`: regret on the same sequence is 1.46 at T = 10³–10⁵. H1 and the first persistent-weights run used the flawed update; both were rerun with the fix (Hedge bullet below).
  - X6 is reproduced by `--eta 0.2 --alpha 0.002 --scale 0`.

### Exact endgames: stage-matrix error and bound tightness (`endgame_eps.py`, `results/endgame/eps.jsonl`)
- **Not exact after all.** No endgame fits 20k states exactly, because of PP combinatorics. All rows use PP abstraction plus HP buckets (/10 or /20), so V* and the exploitability are exact only for the abstracted game. Skipped: kingambit vs clefable, the 2v2s, and 5 of 8 extra pairs.
- **Bound ≥ exploitability on every evaluated row**, but the bound is nearly vacuous: refined bound / exploitability is 6.9–155 (primary refined bound; the 22 rows with exploitability > 1e-3 and a finite bound) where exploitability > 0. It is 5.1 if the short-tie-break bound is used where the primary bound is infinite.
- **The heuristic evaluation function carries the error.** The affine-invariant stage-matrix error ε_v^aff := inf over a>0, c of ‖aQ̂ + c − Q*‖∞ (best per-state a·M + c fit to Q* on the [-1,1] scale) is 0.8–1.0 at its maximum nearly everywhere. The best fit is constant (a → 0) at 9–67% of live states, meaning the heuristic carries almost no information about Q* in these endgames. ε_s is about 0.
- **True exploitability of the matrix bots, selected rows (equilibrium sampled):**

  | Case (abstracted) | Seat | Exploitability | H̄ | Refined bound |
  |---|---|---|---|---|
  | garchomp vs rotomwash 100% | 1 / 2 | 0.338 / 0.602 | 3.2 / 3.0 | 4.5 / 4.6 |
  | greattusk vs gholdengo 100% | 1 / 2 | 0.215 / 0.120 | 1.1 / 1.0 | 2.0 / 2.0 |
  | gholdengo vs heatran 100% | 1 / 2 | 0.074 / 0.425 | 2.0 / 31.4 | 1.6 / 16.4 |
  | greattusk vs rotomwash 100% | 1 / 2 | 0.190 / 0.859 | 2.6 / 6.7 | 2.9 / 10.8 |

  Pure maximin reaches 1.60 and top move 1.26 (greattusk vs rotomwash, seat 2).
- **One failed sanity check.** The exact-equilibrium row for garchomp vs rotomwash 100%, seat 1, has exploitability 0.029 (should be about 0) and infinite H̄. Shapley hit its sweep cap with a final change of 3e-8. Diagnosed: under (exact stage equilibrium, best response), two reachable states form an endless loop: earthquake into a levitating Rotom-Wash vs pain split, which the PP abstraction never ends. Valuing endless play at V* gives exploitability 1e-16. Valuing it as a draw, the best-response convention, gives 0.029. The stage-wise equilibria of V* are optimal only if endless play is valued at V*. This is the non-termination problem of undiscounted stochastic games, and it is also why Shapley needs 5,000 sweeps. The other 11 exact-equilibrium rows are below 2.5e-10.
- **Reading.** This supports the theory's main claim that ε_v dominates, and is consistent with the X1 / V1 finding that the leaf evaluation error dominates the bound. It also shows that per-state worst-case ε_v makes the bound uninformative; a typical-state (mean) version would be tighter.

### PSRO plus RL best response on Modal (`psro.py`, `modal_psro.py`, `psro_cost.py`, `results/psro/COST.md`)
- **Scaffold.**
  - Population of fixed bots plus PPO nets.
  - Meta-matrix of win rate − ½ from paired seeds, solved by LP for the meta-Nash.
  - PPO best response against the meta-Nash mixture, warm-started from `results/ppo_pilot.pt`.
  - `rl_br` mode: a lower bound on a fixed agent's exploitability (the hedge learner, equilibrium sampled).
  - Budget guard: refuses to launch over `--max-usd` and stops mid-run when the running estimate exceeds it.
- **Environment.**
  - Modal 1.6.4 is installed in `.venv`.
  - The image compiles poke_engine 0.0.48 from source with the tera feature.
  - `modal run modal_psro.py::check` played one game per bot against the PPO net on Modal; all four ran.
  - The main run is now `modal run modal_psro.py::main ...`.
- **Cost estimate** (Modal pricing 2026-11-14, throughput from our logs):

  | Run | Cost |
  |---|---|
  | Smoke | $0.77 |
  | First real run | $6.52 (up to ≈$9.8 at 40% utilization) |
  | Fuller run | $114 |

  About 90% of the cost is training best responses against matrix bots, which take about 74 ms per move.
- **Smoke run.** Stopped by hand after about 9 minutes, while still filling the opening table between the 4 starting bots. No results were written. Spend was under the $1 cap.
- **Why it was slow, and the fixes before the next run.** A GPU won't help: the time goes to game simulation, not the 256×256 network.
  1. Raise the containers' idle timeout from 10 s to about 60 s, to avoid repeated cold starts.
  2. Stop waiting on the slowest game in each training round: take results as they arrive, or use more and smaller rounds.
  3. Fill the opening table in one batch instead of sequential batches.
  4. Use 64 containers.
  5. Optionally train best responses against cheap opponents only, and keep matrix bots for evaluation. This is the biggest saving, but it changes the method.
- **Known risk: seeded games don't replay exactly.** In 1 of 6 local replays the same position produced a different matrix at turn 27; local and Modal results for the same seed also differed. Teams and full_matrix on 60 fixed positions were identical across processes, so the source isn't found yet. Each game is still a valid sample, but paired-seed pairing is approximate and pairs shouldn't be compared one-to-one across machines.

### MCTS with Foul Play's final-move rule, and Hedge over {MCTS, equilibrium} (`results/experiments/E7.jsonl`, `E6f.jsonl`, `H1.jsonl`)
- **Agent `foulplay_*` (MCTS with Foul Play's final-move rule; "MCTS (Foul Play's rule)" in the tables).** It reproduces Foul Play's final-move rule (`fp/search/main.py` `select_move_from_mcts_results`): MCTS visit fractions, keep the options with ≥ 75% of the top fraction, then sample in proportion. One world, full information.
- **Exploiter for MCTS targets.** For an MCTS target (with or without Foul Play's rule), LBR estimates the target's q from its own MCTS run.
- **Scores** are A's score; the unit is a pair (both seats, same seed); 95% CIs.

| Exp | A | B | Games | Score | 95% CI | ms/move A / B |
|---|---|---|---|---|---|---|
| E7 | MCTS (Foul Play's rule), 2,000 it | MCTS, 2,000 it (sanity check) | 100 | 0.505 | [0.417, 0.593] | 2.9 / 2.8 |
| E7 | MCTS (Foul Play's rule), 2,000 it | Equilibrium, sampled | 100 | 0.490 | [0.399, 0.581] | 2.8 / 75 |
| E7 | MCTS (Foul Play's rule), 2,000 it | Equilibrium, top move | 100 | 0.310 | [0.235, 0.385] | 2.8 / 74 |
| E7 | MCTS (Foul Play's rule), 75 ms (time-matched) | Equilibrium, top move | 100 | 0.650 | [0.571, 0.729] | 72 / 71 |
| E6f | MCTS (Foul Play's rule), 2,000 it | its LBR exploiter | 60 | 0.275 | [0.161, 0.389] | 2.9 / 71 |
| E6f | MCTS, 2,000 it (most visited) | its LBR exploiter | 60 | 0.258 | [0.166, 0.351] | 2.6 / 75 |
| E6q | Equilibrium, sampled | its LBR exploiter | 60 | 0.475 | [0.377, 0.573] | 74 / 73 |
| E6q | Equilibrium, top move | its LBR exploiter | 60 | 0.317 | [0.232, 0.401] | 77 / 77 |
| H1 | Hedge {MCTS (Foul Play's rule), equilibrium sampled} | MCTS, 2,000 it | 100 | 0.560 | [0.473, 0.647] | ~74 / 2.6 |
| H1 | Hedge {MCTS (Foul Play's rule), equilibrium sampled} | depth-2 informed best response (knows p) | 60 | 0.408 | [0.294, 0.522] | 36 / 38 (recorded; the plan's cost splits by seat) |

- **Time-matched follow-ups** (`E7t.jsonl`, `E6ft.jsonl`):
  - MCTS with Foul Play's final-move rule at 75 ms per move vs equilibrium sampled: **0.760** [0.677, 0.843] (100 games, 70 vs 61 ms per move).
  - The same bot at 75 ms per move vs its LBR exploiter: **0.533** [0.437, 0.629] (60 games).
- **Correction: an LBR score mixes exploitation with strength.** The LBR exploiter is itself a depth-2 matrix bot, so its score is not pure exploitation. The fair measure is the exploitation gain: the score against a non-exploiting opponent of the same strength, minus the score against LBR. For MCTS with Foul Play's rule, that opponent is equilibrium top move: same matrix as LBR, but it doesn't model its opponent. For the equilibrium bots, it is the bot itself (0.5 by symmetry).

  | Bot | vs same-strength non-exploiter | vs LBR | Exploitation gain | 95% CI |
  |---|---|---|---|---|
  | MCTS (Foul Play's rule), 2,000 it (2.8 ms) | 0.310 | 0.275 | 0.035 (CIs overlap) | [−0.10, 0.17] |
  | MCTS (Foul Play's rule), 75 ms | 0.650 | 0.533 | 0.117 | [−0.01, 0.24] |
  | Equilibrium, top move | 0.5 | 0.317 | 0.183 | [0.10, 0.27] |
  | Equilibrium, sampled | 0.5 | 0.475 | 0.025 | [−0.07, 0.12] |
  | MCTS, 2,000 it (most visited) | 0.419 (E1fix, cross-experiment) | 0.258 | 0.160 | [0.06, 0.26] |

  - The earlier reading, "MCTS with Foul Play's rule is as exploitable as plain MCTS", was confounded: at 2,000 iterations LBR mainly wins by being the stronger bot.
  - Only top move's gain (z = 4.3), plain MCTS's gain (z = 3.1) and the top-move minus sampled difference (0.158 [0.03, 0.29]) are significant. The lower gain of MCTS with Foul Play's rule is suggestive (vs plain MCTS z = 1.45; vs top move z = −1.8) and depends on the baseline: with equilibrium sampled as its baseline its gains become 0.21–0.23.
  - The gains come from different experiments with CIs around ±0.1, so they are indicative only.
- **Strength vs robustness.** At equal time, MCTS with Foul Play's final-move rule beats equilibrium play (0.650 vs top move, 0.760 vs sampled), consistent with the leaf evaluation error dominating the bound. Equilibrium sampling is less exploitable. The result is a trade-off, not a dominance, which motivates combining them (Hedge, below) and a better leaf value. A depth-2 informed best response confirms the ordering at 100 pairs (W1, below): top move 0.873 vs sampled 0.713, difference +0.160 [+0.091, +0.229].
- **RL best response on Modal** (`results/psro/rlbr_exploit/`, config `configs/rlbr_exploit.json`):
  - Setup: 100k PPO decisions per target, warm-started from the PPO pilot net on the 12-species full-information pool. Each net is evaluated on 200 held-out games, alongside the untrained start net on the same seeds.

    | Target | Untrained net | Trained net | Gain |
    |---|---|---|---|
    | MCTS (Foul Play's rule), 2,000 it | 0.043 | 0.088 | +0.045 |
    | Equilibrium, sampled | 0.108 | 0.123 | +0.015 |
    | Equilibrium, top move | 0.135 | 0.158 | +0.023 |

    95% CIs are about ±0.02–0.04 per win rate. Wall time 0.73 h; the budget meter estimated $2.65 against a $1.19 plan. The meter is conservative: it bills min(inputs, 64) warm containers per map, and actual task time was 8.3 core-hours. Check the real bill in the Modal dashboard.
  - **Reading: not informative yet.** The best response stays far below 0.5 against every target, so the PPO arm's lower bounds are negative. Any fixed exploiter gives a valid lower bound, so LBR's scores were valid positive bounds throughout (win rate − 0.5: top move ≥ 0.18, Foul Play 2,000 it ≥ 0.22, sampled ≈ 0.03, n.s.). The gains are within noise, except possibly against Foul Play (z ≈ 2.4 under independence; seeds are shared, so a paired test would be tighter).
  - The pilot net is too weak a starting point; it won 5–13% against MCTS (200 iterations requested, 1,000 run: poke-engine rounds up to multiples of 1,000) in the pilot. A useful RL exploitability measurement needs a much stronger base policy, or a far larger budget. A depth-2 informed best response is the better tool (below).
- **RL best response at scale (rlbr_scale)** (`results/psro/rlbr_scale/curve_full.json`, logs `results/psro/run_rlbr_scale.log` and `results/psro/run_rlbr_scale_resume.log`, configs `configs/rlbr_scale.json` and `configs/rlbr_scale_resume.json`):
  - Setup: PPO best response vs MCTS with Foul Play's final-move rule (`foulplay_it:2000`), warm-started from the PPO pilot net, planned for 8M decisions; 200 held-out games per curve point, same seeds throughout.

    | Decisions | Games trained | Win rate | 95% CI |
    |---|---|---|---|
    | 0 | 0 | 0.060 | [0.036, 0.084] |
    | 1.00M | 22,016 | 0.065 | [0.042, 0.088] |
    | 3.01M | 63,232 | 0.090 | [0.064, 0.117] |
    | 4.94M | 97,536 | 0.100 | [0.071, 0.129] |
    | 6.00M | 115,200 | 0.1225 | [0.088, 0.157] |
    | 8.01M | 147,456 | 0.125 | [0.097, 0.153] |

  - The run was in two parts. The first stopped at its $2.92 budget guard at 4.94M decisions; a resume from that checkpoint (fresh Adam state, constant lr 3e-4, new training seeds, games_per_input 16) reached 8.01M with no guard stop, about 68× the previous run's games (100k decisions ≈ 2.2k games, 0.043 → 0.088). The 4.94M checkpoint was evaluated on Modal (0.100) and locally (0.0975) on the same seeds. Final exploitability lower bound −0.375 [−0.403, −0.347].
  - Cost: meter smoke $0.074 + first $2.922 + resume $0.969 = $3.965 total; 13.2 worker core-hours, 2.5 h wall.
  - Note: `psro_cost` under-predicted this run type by about 2×. Container utilization was 16%: billed time is dominated by warm containers waiting on the slowest 8-game input. The resume cost about $0.32 per 1M decisions vs $0.59 for the first part, because halving the containers (games_per_input 16) halved the idle billing.
  - **Reading:** 68× more training moves the win rate from 0.06 to 0.125. It stays far below 0.5, so RL gives no useful exploitability bound here. The depth-2 informed best response gives 95% lower confidence bounds on exploitability of ≥ 0.25 (top move) / ≥ 0.09 (sampled) with no training.
- **Gray-box RL best response (rlbr_gray, planned 2026-11-15; written before the run).** Same as rlbr_scale, except that the PPO observation also includes the target's move distribution q for the current turn (14 action slots). This is the exact distribution Foul Play then samples from, not a fresh estimate.
  - The init is the pilot net with zero weights on the new inputs, so curve point 0 must reproduce 0.060 on the same held-out seeds.
  - Config `configs/rlbr_gray.json`: 8M decisions in one run, games_per_input 16, curve points at 1M/3M/5M/6M/8M. Cap $4.
  - Tests the claim in `theory/search_rl_links.md` (L3) that knowing the opponent's policy, not more games, is what separates informed-best-response search from RL.
  - Interpretation limits, fixed in advance:
    - The gray-box net gets q only. The depth-2 informed best response gets q plus the exact stage matrix. A null result therefore means "q without a value model is not enough", not "knowing the policy doesn't matter".
    - Confound: the black-box curve was two runs (games_per_input 8, then 16 with a fresh Adam state). This is one run.
  - **Result (2026-11-15, `results/psro/rlbr_gray/rl_br_foulplay_it_2000.json`, log `results/psro/run_rlbr_gray.log`):**

    | Decisions | Gray-box win rate | 95% CI | Black-box (rlbr_scale) |
    |---|---|---|---|
    | 0 | 0.055 | [0.033, 0.077] | 0.060 |
    | 1.01M | 0.0925 | [0.062, 0.123] | 0.065 |
    | 3.01M | 0.0975 | [0.070, 0.125] | 0.090 |
    | 5.00M | 0.0825 | [0.057, 0.108] | 0.100 |
    | 6.00M | 0.0825 | [0.056, 0.110] | 0.1225 |
    | 8.01M | 0.075 | [0.051, 0.099] | 0.125 |

    - 146,432 training games; Modal estimate $1.90, 13.7 worker core-hours, 1.8 h wall.
    - Reading: the gray-box net leads early, then flattens and ends below black-box (unpaired difference −0.05, about 2.7 SE; single runs). Point 0 is 0.055 vs 0.060: the same zero-padded policy, different MCTS randomness. As written above before the run, this says the opponent's distribution alone is not enough without a payoff model; it does not say knowing the policy is useless.
- **Hedge** (fixed FTRL-form anytime learner; the runs with the flawed update are archived in `results/experiments/buggy_anytime_hedge/`). H1 rows above are the rerun.
  - Per-game Hedge (weights reset each game; median 32–39 turns, p90 51–64): 0.560 vs MCTS, 0.408 vs the depth-2 informed best response that knows its mixture. Mean final weight on the MCTS expert 0.46 vs MCTS, 0.38 vs the exploiter.
  - **Weights carried across games** (`hedge_stream.py`, the setting Thm E2 covers; 4 seeds × 30 games vs MCTS, then 30 vs the depth-2 informed best response, same game seeds as a per-game-reset control; `results/experiments/hedge_stream_seed*.jsonl`):

    | Weights | Opponent | Games | Score | 95% CI (seeds, t₃) | MCTS-expert weight (end of game, mean over games) |
    |---|---|---|---|---|---|
    | Carried | MCTS, 2,000 it | 120 | 0.533 | [0.355, 0.712] | 0.34 |
    | Carried | depth-2 informed best response | 120 | 0.533 | [0.441, 0.625] | 0.14 |
    | Reset | MCTS, 2,000 it | 120 | 0.471 | [0.401, 0.541] | 0.45 |
    | Reset | depth-2 informed best response | 120 | 0.388 | [0.308, 0.467] | 0.38 |

    Paired on (seed, game): carried − reset = +0.062 [−0.050, +0.175] vs MCTS, **+0.146 [+0.039, +0.253] vs the exploiter** (95% CIs over 120 paired games, iid; t₃ over the 4 seed means: [−0.085, +0.210] and [+0.019, +0.273]).
  - Reading: carrying weights costs nothing against MCTS and makes the learner significantly harder to exploit. The MCTS-expert weight drifts slowly against MCTS (0.45 → 0.28 over 30 games), then falls faster once the exploiter starts (0.28 → 0.07). The per-game learner restarts near uniform each game, so the exploiter always faces a large MCTS share.
  - Caveat: both experts are rewarded under the heuristic matrix A, which favours the equilibrium expert by construction. The drift is therefore toward "best under A", not toward the expert that wins more games.
  - With the flawed incremental update, the carried learner had collapsed to the equilibrium expert within about 10 games (weight 0.10 by game 29), and the carried − reset difference was −0.10 vs MCTS. That was an artefact of the bug.

### Leaf value and exploiter brainstorms (2026-11-14; preliminary, scratch harness)
Scripts and logs: `/private/tmp/claude-501/-Users-biofanat-Documents-classes-f26-cs535/2bbe670b-2f97-4d9a-bfad-e1139e0352c6/scratchpad/leaf/` and `.../scratchpad/rlbr/`. No project code changed.
- **poke-engine MCTS runs at least 1,000 iterations.** Requests of 1, 50, 200 or 1,000 all run exactly 1,000 root visits (checked). So every "MCTS-200" in this file, including the PPO pilot's opponent and the psro `tiny` preset `mcts_it:200`, actually ran MCTS-1,000.
- **The heuristic leaf mis-scores finished games.** Through `pe.id(child, 0)`, a terminal child is scored by the winner's remaining material, not as a win. On greattusk vs gholdengo, P1-won terminals score −50 to +185 while live children span ±215, so a won position can rank below a live one (checked). In MCTS self-play on randbats, 18 of 151 two-sided decisions (12%, all late-game) have a terminal child; in 15 of them the leaf inverts a win or loss against a live child. One mis-scaled terminal cell also explains why the ∞-norm affine fit in `endgame_eps.py` often collapses to a = 0.
- **Candidate leaves on the exact endgames.** Mean affine-invariant stage-matrix error ε_v^aff and exploitability of equilibrium sampled (seat 1 / seat 2):

  | Endgame | Current leaf | Exact terminals + tanh(h/160) | MCTS-1,000 leaf | Existing value net |
  |---|---|---|---|---|
  | greattusk vs gholdengo, ε_v | 0.153 | 0.046 | 0.008 | 0.057 |
  | exploitability | 0.215 / 0.120 | 0.000 / 0.000 | 0 / 0 | 0 / 0 |
  | gholdengo vs heatran, ε_v | 0.164 | 0.068 | 0.035 | 0.078 |
  | exploitability | 0.074 / 0.425 | 0.074 / 0.020 | 0.005 / 0.008 | 0.074 / 0.020 |
  | garchomp vs rotomwash, ε_v | 0.475 | 0.397 | 0.331 | 0.548 |
  | exploitability | 0.338 / 0.602 | 0.306 / 0.601 | 0.431 / 0.816 | 0.193 / 0.873 |

  - Exact terminals plus a bounded leaf remove nearly all exploitability on two of three endgames, at no extra cost per move. On the long, balanced endgame they barely help: there the heuristic lacks information, not an affine rescaling.
  - An MCTS leaf is the best leaf on the easy endgames and the worst on the balanced one. MCTS values are also relative to the root (`mcts.rs` rollout = sigmoid(0.0125·(eval(leaf) − eval(root)))), so they can't replace the heuristic mid-game.
  - The maximum ε_v stays about 1 for every candidate leaf, so only the refined, along-the-game form of the bound can become informative.
  - Mid-game strength: 20 pairs vs MCTS with Foul Play's final-move rule at 75 ms per move gave the current leaf 0.375 ± 0.15 and the fixed leaf 0.312 ± 0.14. No measurable change, as expected, since the fix only acts on about 12% of decisions.
- **Depth-2 informed best response (`wb2`).** It knows the target's exact policy. For its top-K actions it expands chance branches, then best-responds to the target's exact policy one turn deeper. For a fixed target, any such exploiter gives a valid lower bound: win rate − 0.5.

  | Exploiter | Target | Pairs | Score | 95% CI | Lower bound |
  |---|---|---|---|---|---|
  | LBR (one turn), same seeds | Equilibrium, top move | 20 | 0.700 | [0.590, 0.810] | 0.20 |
  | wb2 | Equilibrium, top move | 20 | 0.825 | [0.718, 0.932] | 0.325 |
  | wb2 | Equilibrium, sampled | 10 | 0.850 | [0.700, 1.000] | 0.35 |
  | LBR + MCTS leaf | Equilibrium, top move | 20 | 0.475 | [0.363, 0.587] | −0.03 |

  - wb2 − LBR on the same 20 seeds: +0.125 [+0.010, +0.240]. Part of this is fewer 300-decision-node-cap draws (6/40 vs LBR's 16/40).
  - Preliminary reading (10 pairs), refuted by W1 below.
  - The RL exploiter used only 1.2–2.2k games per target, against the pilot's 340k self-play games. For a known, deterministic target, informed-best-response search is far cheaper than RL. RL remains the fallback for black-box or hidden-information targets.
- **Confirmation (W1).** `wb2` (now `wb2_policy` in `engine/nash.py`), 100 pairs per cell (200 games), on fresh seeds disjoint from all earlier runs. Both cells use the same pruning: top K=4 own actions, 80% target support, chance branches to 95% mass. (The preliminary 20-pair top-move run used K=5.) Source: `results/experiments/W1.jsonl`.

  | Target | Pairs | Score | 95% CI | Lower bound (score − 0.5) | 95% LCB (Hoeffding, one exploiter) | W-L-D | 300-decision-node-cap draws |
  |---|---|---|---|---|---|---|---|
  | Equilibrium, top move | 100 | 0.873 | [0.836, 0.909] | 0.373 | 0.250 | 160-11-29 | 29 |
  | Equilibrium, sampled | 100 | 0.713 | [0.654, 0.771] | 0.213 | 0.090 | 136-51-13 | 13 |

  - Difference, top move − sampled: +0.160 [+0.091, +0.229]. The seeds are disjoint, so the cells are independent. Hoeffding radius sqrt(ln 20/200) = 0.122. Wall time 59 min on 5 local processes.
  - Caveat: the 80% support truncation only affects the sampled cell, since the argmax target has a single support action. So part of the gap may come from pruning. A K=5/95% rerun of the sampled cell would separate the two.
  - Reading: the preliminary small-sample reading (a depth-2 exploiter exploits equilibrium sampled about as much as top move, 0.850 at 10 pairs) is refuted. With 100 pairs on fresh seeds, top move is clearly more exploitable. Sampling does buy robustness against a depth-2 informed best response too, not only against one-turn LBR.
  - Both targets now have positive 95% lower confidence bounds on exploitability: ≥ 0.25 for top move and ≥ 0.09 for sampled, at 95% with one pre-registered exploiter, so no selection correction is needed. This is the bracket's lower side; the upper side stays vacuous.

### ε-safe best response and leaf rescaling (2026-11-15; `results/experiments/{SX,LF,SXW}.jsonl`, log `run_SX_LF_SXW.log`)
- **ε-safe best-response bot (`safex:0.02`).** Each turn: v* = LP value of the full matrix A; predicted opponent column q = argmin of A's column means (best response to our uniform mix); play max x·A·q subject to (xA)_j ≥ v* − 0.02·range(A) for every column j (`nash.safex_strategy`; idea from Ganzfried & Sandholm 2015). Worst-case stage loss vs the equilibrium is at most 2% of the matrix range by construction.
- **SX: vs MCTS 2,000 iterations, 100 pairs, same seeds for all three bots (paired):**

  | Bot | Score | 95% CI | ms/move |
  |---|---|---|---|
  | Equilibrium, top move | 0.580 | [0.518, 0.642] | 71 |
  | ε-safe best response, ε = 2% | 0.530 | [0.466, 0.594] | 63 |
  | Equilibrium, sampled | 0.510 | [0.449, 0.571] | 70 |

  Paired: ε-safe best response − top move −0.050 [−0.134, +0.034]; ε-safe best response − sampled +0.020 [−0.066, +0.106]. Both n.s. Sampled scores 0.510 here vs 0.44 in E1fix (different seeds and n).
- **SXW: depth-2 informed best response (`wb2:safex:0.02:4:0.8`, knows the bot's algorithm) vs the ε-safe best-response bot, 100 pairs:** exploiter scores **0.767** [0.718, 0.817]; Hoeffding 95% lower bound 0.645, so exploitability ≥ 0.145 in score units. W1 on different seeds: 0.873 vs top move, 0.713 vs sampled (unpaired).
- **LF: leaf rescaling vs MCTS with Foul Play's final-move rule, 75 ms per move, 100 pairs, paired seeds:** current leaf 0.355 [0.301, 0.409]; tanh(h/400) with exact ±1 terminals 0.315 [0.259, 0.371]; paired difference −0.040 [−0.098, +0.018], n.s.
- **Reading:**
  - The ε-safe best-response bot's per-turn gain does not show as a win-rate gain at 100 pairs. Its exploitability sits between top move and sampled: the 2% stage-level cap bounds loss only in the estimated matrix, and the exploiter wins through the leaf error, as predicted in `lit/next_directions.md`.
  - Rescaling the heuristic leaf toward the TD-residual minimiser does not help; the leaf evaluation error is not a scale problem.

### E1: Bayesian LP vs PIMC on real Showdown (2026-11-15; `showdown/`, outputs in `$OURBOT_WORK/replays_headless/E1*`)
- Setup: Showdown's own simulator (`pokemon-showdown simulate-battle`, commit 7e74376) run headless by `showdown/headless.py`; both seats use Foul Play's battle tracker and set sampler (K = 4 sampled worlds, equal compute), then our matrix (`nash.full_matrix` per world); Bayesian LP = one LP over the worlds, PIMC averaging = LP per world, strategies averaged by weights. Paired seeds: same teams and battle PRNG, seats swapped. 300 seeds.
- Result: Bayesian LP's pair score **0.559 [0.521, 0.597]** over 298 pairs (596 games); 1 game hung (seed 44, hit the 3,600 s cap; per-game cap now 600 s, longest E1 game 108 s); mean 32.7 turns; 0.315 / 0.314 CPU-s per decision; 28.2 CPU-s per game.
- The two methods choose a different top move on ~11% of decisions (shadow solve on the same worlds, smoke run).
- Reading: the in-engine result (0.585 [0.540, 0.630], our own belief tracker) transfers to real Showdown with Showdown's hidden information and Foul Play's belief model. The comparison isolates the formulation: both arms share the belief.

## Real-Showdown experiments (Modal, 2026-11-15)

Outputs in `showdown/results/<TAG>/`.

- Setup: as E1 (Showdown commit 7e74376 headless, Foul Play's tracker and set sampler, K = 4 worlds unless stated, our matrix per world). Seat-swapped pairs share seed, teams and battle PRNG; the result is player A's pair score (win = 1, draw = ½) with a 95% CI over pairs. "Foul Play" = the Foul Play MCTS client (decoupled UCT) at 100 ms per sampled battle. Runs used 10 matches per tag on Modal (E1 local); a match (also called a stream in the logs): a sequence of games over which Hedge weights and the posterior are carried. A scored turn is a turn on which our agent had at least two legal actions and the opponent's action was observed and matched a sampled world (turns where the opponent had one option are included); Hedge and the posterior update on exactly these turns.
- Sources: `showdown/results/<TAG>/<TAG>_summary.json` (scores, CIs, pairs, costs, skip and gap fields); per-game logs in `showdown/results/<TAG>/replays_headless/`; E1 in `showdown/results/E1_summary.json`.

| Tag | Matchup (A vs B) | Question | Pairs | Pair score of A [95% CI] | Modal cost |
|---|---|---|---|---|---|
| E1 | Bayesian LP vs PIMC averaging | Does Bayesian LP beat PIMC averaging on Showdown? | 298 | **0.5587** [0.5209, 0.5966] | local |
| AF | Bayesian LP vs Foul Play | How strong is the base bot? | 299 | 0.3144 [0.2792, 0.3495] | $0.50 |
| E2 | `hedge` vs Foul Play | Does carried Hedge over 4 experts match its experts? | 399 | 0.3108 [0.2804, 0.3412] | $0.72 |
| E2fs | `hedge_fs` vs Foul Play | Same with fixed-share Hedge | 399 | 0.3033 [0.2729, 0.3336] | $0.75 |
| E4e | `bandit_exp3` (visit shares) vs `rmplus_avg` | Does bandit feedback at the root cost value? | 296 | 0.0625 [0.0430, 0.0820] | $0.45 |
| E4u | `bandit_ucb1` (visit shares) vs `rmplus_avg` | Same, UCB1 | 299 | 0.0502 [0.0312, 0.0691] | $0.38 |
| E4emv | `bandit_exp3_mv` (most-visited action) vs `rmplus_avg` | Is it the feedback model or the decision rule? (B8) | 295 | 0.4949 [0.4547, 0.5351] | $0.51 |
| E4umv | `bandit_ucb1_mv` vs `rmplus_avg` | Same, UCB1 (B8) | 300 | **0.5933** [0.5551, 0.6316] | $0.56 |
| E5 | `hedge` vs Bayesian LP | Can a no-regret learner beat the equilibrium expert? (PA4β) | 300 | 0.4583 [0.4210, 0.4956] | $0.56 |
| BH | `hedge_bayes` (style posterior in the ε-safe expert) vs Foul Play | Does a better opponent prediction win games? | 398 | 0.3329 [0.3022, 0.3637] | $0.68 |
| AF16 | Bayesian LP, K = 16 vs Foul Play | Do better-calibrated worlds win games? | 299 | 0.3278 [0.2923, 0.3632] | $1.61 |

Paired differences (same seeds, so same teams and battle PRNG; trajectories diverge, so these are not identical games; mean ± 1.96 sd/√n over common seeds, recomputed from `per_seed`):

| Comparison | Difference [95% CI] | Common seeds |
|---|---|---|
| BH − E2 | +0.024 [−0.018, +0.066] | 397 |
| BH − E2fs | +0.029 [−0.013, +0.071] | 397 |
| BH − AF | +0.017 [−0.032, +0.066] | 297 |
| AF16 − AF | +0.017 [−0.031, +0.065] | 298 |

Modal cost: the ten summaries' `usd_estimate` fields sum to **$6.73** (of a $15 budget). This excludes partial first attempts (several summaries have `games_run` below the games scored, i.e. resumed runs) and pilots; the full bill was reported as about $8.3, which these files cannot confirm.

Per experiment (Lecture 6 = one no-regret learner against an adaptive adversary; Lecture 7 = no-regret dynamics give minimax / CCE). Predictions are those pre-registered in `theory/bridge_plan.md` §2 and `theory/theory_map.md` §8, and for B8 in `theory/experts_extensions.md` (pre-registration `verify_ext/prereg_B8.txt`).
- **E1 (structural, M3→M4).** Prediction: Bayesian LP > PIMC; refuted if the paired Δ ≤ 0. **Met**: 0.5587 replicates the in-engine 0.585 [0.540, 0.630] with a different belief model (both arms share Foul Play's).
- **E2 / E2fs (Lecture 6, R2a and L2).** Prediction (1): the pathwise regret bound holds on every log. **Met** on all 10 E2 matches: on the normalised stage payoffs (all in [0, 1]), the carried learner's average regret per turn stays below the R2a bound everywhere (largest excess over the bound −0.0229); at T = 2056 scored turns (shortest match) the median is 0.0042 (IQR 0.0039–0.0046) against the bound 0.0277; fitted slope of the median −0.15 against the bound's −0.51 (headline figure panel B; 10 matches, 800 games, 22,858 turns; weight-replay consistency check max |w_logged − w_replayed| = 0.0092; recomputed from the E2 logs with `report/make_headline_figure.py`). Weights reset at each game give 0.0057, so carrying lowers regret (consistent with R3d's warm-up amortisation), but this does not move the win rate. Prediction (2): W_H within 0.05 of its experts. **Met for the one expert run alone**: E2 0.3108 and E2fs 0.3033 vs Bayesian LP 0.3144; the other three experts were not run alone. Caveat: 16.43% (E2) and 16.83% (E2fs) of feedback turns were skipped (opponent action in no world, fainted, `cant`, no action), above the ~10% at which L2 needs semi-bandit feedback, so L2 applies only approximately. Regret here is on normalised stage payoffs along our own trajectory, not on game outcomes (P1).
- **E4e / E4u (Lecture 7, L4 + B3: bandit vs full-information feedback).** Pre-registered design (bridge_plan §2): the bandit root loses ≥ 0.05 more **against an exploiter**; not run as designed. The runs are head-to-head against `rmplus_avg` on the same kind of matrix. Shares-playing roots lose heavily (0.0625, 0.0502). Their decision logs show diffuse play: median top-move probability 0.315 (Exp3) and 0.225 (UCB1) vs 0.78 for RM+ (0.783 / 0.788); median duality gap 0.0585 / 0.0578, mean 0.086 / 0.093 (A01 units; `theory/experts_extensions.md` B7, verified with corrections). Theorem A's summed gap (2.6–2.8 per game) is vacuous and does not explain the scores (B7(iii)).
- **E4emv / E4umv (B8).** B8's prediction and refutation criterion were written before the runs (`theory/experts_extensions.md`); the verifier's quantified pre-registration was recorded before any result was read (`scratchpad/verify_ext/prereg_B8.txt`, timestamped). Prediction: against the RM+ root, the max-visit roots lose much less than the shares roots (per-decision deficit 5–8× smaller; verifier's point guess 0.25–0.45); refuted if _mv ≤ shares at n ≈ 300. **Ordering confirmed**: 0.4949 and 0.5933 vs 0.0625 and 0.0502; the point guess undershot both. So at MCTS budgets the decision rule (play shares vs the most-visited action), not the feedback model, drives the outcome. E4umv > 0.5 also shows that `rmplus_avg` is not a stage equilibrium of the bandit's matrix (it solves its own worlds' matrix, an ε_v-type difference; B8(iii) is a hypothesis for this opponent, not a theorem). Reading (ours): deterministic play beats an opponent that does not adapt, a parallel to Lecture 7 §4.1's probability matching (matching the frequencies scores 58% where always predicting the majority scores 70%) and to the in-engine top move vs sampled (0.581 vs 0.443). Its cost in exploitability was untested as of 2026-11-15: Lecture 6 §2.3 says a deterministic rule can be exploited and B9(i) gives the max-visit row a positive duality gap on strictly mixed matrices, but on 57 real matrices the two rules' duality gaps are similar (0.053 vs 0.058; B8 verifier note), and no exploiter had been run; the 2026-11-16 exploiters ("Open items (2026-11)", B3) find a borderline gain against the most-visited UCB1 root only. (The `gap` fields in these summaries, median 0.104 / 0.125, are the duality gap of the visit shares at the _mv bots' positions, not of the arm they played.)
- **E5 (Lecture 7 minimax content, PA4β).** Prediction W_A ∈ [0.45, 0.55]; refuted if > 0.56. **Not refuted**, point estimate 0.4583 inside the band, but the whole CI lies below 0.5: the learner was out-played by the equilibrium expert at 95% confidence, which the design did not anticipate. PA4β bounds only the upper side, so this probes the band's lower edge.
- **BH (style posterior, `experts_extensions.md` B1, B5, B6).** No win-rate prediction was registered; B6's whole-game bound is vacuous per game. Result 0.3329, paired +0.024 vs E2 and +0.017 vs AF, n.s. 17.36% of feedback turns skipped. The posterior predicts the opponent's move much better than the old predictor (log loss 1.56–1.81 vs 3.23–3.32 in the 24-pair pilots BMstock / BMaf; `report/current_report.tex` Table 5, `tab:behav`) but this does not show up as a significant win-rate gain.
- **AF16 (K = 16 worlds, B10/B11).** B11(iv): no win-rate gain is derivable from the bounds (in-engine X2, 8 vs 4 worlds: 0.495, n.s.). Result 0.3278, paired +0.017 vs AF, n.s.; the K = 16 absent-fraction predictions of B10 were not scored in this run (no belief logging). Cost 2.9× AF's CPU per game (100.3 vs 34.5 CPU-s).
- **Overall reading (ours).** Better prediction (BH) and better-calibrated worlds (AF16) give small positive, non-significant edges. This is consistent with the leaf evaluation error dominating the bound: by Theorems A and C the remaining loss is charged to ε_v, which the exact endgames put near 1, so improving the belief or the opponent model moves the estimated matrix but not the error in it.

## Public ladder: preliminary model vs the Bayesian model (merged from `feat/mechanics-belief-tracker`)

Gen 9 Random Battles on the public Showdown ladder, account `bot_bonp`, against human players. Batches are sequential
and rated (not paired), so opponents and ratings differ between rows; Elo changes are not causal per-change estimates.
Sources are on branch `feat/mechanics-belief-tracker` (commit `bf4355e`): `docs/final-ladder-bot-handoff.md`,
`docs/100-game-tactical-evaluation.md`, `research/credit_v2_50game_evaluation.json`,
`research/native_mcts_v1_ladder_results.json`, `research/stratified_mcts_climb_100completed_results.json`.
Paper: final5.tex "On the public ladder" (`sec:exp-ladder`); deck: "Out in the world: the public ladder".

| Bot | Games | W–L | Elo start → end | Peak | Source |
|---|---:|---|---|---:|---|
| Hedge + Bayesian memory (`hedge_bayes_mw`, merged joint-belief) | 100 | 52–48 | 1577 → 1521 | – | 100-game-tactical-evaluation.md (raw JSON local only) |
| Corrected per-world Hedge | 50 | 25–25 | 1521 → 1470 | – | credit_v2_50game_evaluation.json |
| Bayesian model without stratification | 50 | 33–17 | 1470 → 1677 | 1741 | native_mcts_v1_ladder_results.json |
| Bayesian model (final): native MCTS + item/ability-stratified worlds | 100 | 66–34 | 1677 → 2090 | 2141 | stratified_mcts_climb_100completed_results.json |

- Final bot: `mcts_native`, `item_stratified` sampler, 4 worlds, 1000 ms total native search (250 ms per world),
  action = argmax of prior-weighted, per-world-normalised legal root-visit shares (deterministic). No Hedge, no
  equilibrium solve, no carried opponent memory. Top 500 was reported by the operator; the rank itself was not logged.
- Five interrupted attempts with unknown outcomes (connection loss) are excluded from W–L; the server rating fell
  1984 → 1920 during that gap and the endpoint above includes it.
- Our theory does not cover this bot: Theorem 6 needs an ε_s-optimal stage strategy, and argmax of root visits is
  randomised only through sampling and search noise, with no bound on its solver error. Its exploitability is unmeasured.
- Stratified sampling (worlds grouped by the active opponent's effective item and ability) changes finite-sample
  coverage, not the prior: in a 5000-draw Dragalge diagnostic, plain 4-world sampling omitted Choice Specs 33.8% of the
  time, stratified sampling never.
- Ladder audit of the Hedge agent: its pure-maximin expert minimised columns of the averaged matrix Ā and picked a
  strictly worse per-world security action in 495 of 2,951 decisions (16.8%); the corrected per-world version
  (`argmax_a Σ_k w_k min_b M_k[a,b]`) still went 25–25.

- Paper/deck report the two rows of each model combined: preliminary model 77–73 (150 games, 1577 → 1470); Bayesian
  model 99–51 (150 games, 1470 → 2090, peak 2141; stratified worlds from game 51).
- Price of determinism (final5.tex, Proposition `prop:det`): a policy that puts probability ≥ 1−ρ on one action loses at
  least g_t − ρΔ_t per turn to the myopic exploiter, g_t = value − best pure guarantee of the true stage game. Brute
  force: `scratchpad/verify_det/det.py` (602 random games, worst slack 0, sometimes tight) and `det_rho.py` (597 games,
  ρ ∈ {0, 0.05, 0.2, 0.5}). On the 120 b3 stage matrices, 42% have g_t > 0 (no optimal pure strategy for us), median
  g_t = 0.024 of the payoff range (`scratchpad/verify_det/gap_stats.py`; all 1,477 matrices: 35%, median 0.022).

## No-regret over the Bayesian model, and an adversary for it (branch `test/noregret-ladder`, 2026-11-17)

**Hedge over the Bayesian model** (`hedge_mcts`: experts = Bayesian argmax, its visit shares, Bayesian-LP x on
per-world depth-2 matrices; credited on the matrix of the highest-weight world containing the opponent's move,
Δ̄ = 400; weights carried per stream) vs the Bayesian model (`mcts_native`, `item_stratified`, 1000 ms), local headless
Showdown games, seat-swapped pairs, seeds 1–50, 5 streams (`engine/showdown/batch.py --tag h2h_hedge_mcts ...
--workers 5 --streams 5`; the run was paused once after seed 30 and resumed, see the branch's
`engine/results/noregret/`):

- Score of `hedge_mcts`: **0.310 [0.218, 0.402]** (95% CI over 50 pairs, 100 games); lost both seats on 24 seeds.
- Realised regret per stream 3.9–5.0 vs the anytime bound 26.9–28.7 (568–648 credited turns); best expert on every
  stream: `lp_sampled`; final weights LP 0.60–0.71, argmax 0.23–0.33, shares 0.05–0.08.
- Off-argmax probability per decision 0.33 in lost games vs 0.28 in won games; 64% of off-argmax plays are on turns
  where the LP strategy is mixed.
- Reading: Hedge is no-regret with respect to the depth-2 matrices, where the LP expert is optimal by construction, so
  the guarantee points it at the weakest expert in real play (the evaluation error ε_v of the whole-battle bound).
- Source: `analysis_h2h_hedge_mcts.json`, `h2h_hedge_mcts_summary.json` (analysis by
  `engine/showdown/analyze_noregret_h2h.py`). The interrupted resume means each stream's carried weights include a few
  partial-game updates.

**Adversary** (`engine/experiments.py ADV`, in-engine, full information, 100 paired seeds per matchup, seeds
63,000,000+, `results/experiments/ADV.jsonl` on the branch): a one-turn local best response that reruns the
target's MCTS (5,000 iterations) to predict it.

| Target | vs predicting exploiter | vs uniform-prediction exploiter | Gain from predicting (paired) |
|---|---|---|---|
| MCTS argmax (the Bayesian model, one world) | 0.300 [0.241, 0.359] | 0.787 [0.730, 0.845] | +0.487 [+0.399, +0.576] |
| MCTS sampling its visit shares | 0.077 [0.042, 0.113] | 0.087 [0.048, 0.127] | +0.010 [−0.039, +0.059] |

- Exploitability lower bound for the argmax: exploiter 0.70 − 0.5 − Hoeffding 0.12 = 0.08 (score units).
- Sampling the visit shares removes the predictability (Proposition 9, price of determinism) but plays far worse.

**Local Elo (round robin, seat-swapped pairs, local headless Showdown):** preliminary model (`hedge_bayes_mw`) vs the
Bayesian model 0.167 [0.081, 0.252] (30 pairs, seeds 101–130); preliminary vs `hedge_mcts` 0.533 [0.419, 0.648] (30
pairs, seeds 101–130); `hedge_mcts` vs the Bayesian model 0.310 (50 pairs, above). Bradley–Terry/Elo fit with the
preliminary model at 0: Bayesian model **+203** [+135, +283], Hedge over it **+27** [−44, +105] (95% bootstrap over
pairs, 400 resamples). Work dirs: scratchpad `rr_rr_prelim_bayes`, `rr_rr_prelim_hedge` (batch.py, 6 workers,
`--streams 6`).

## Open items (2026-11)

Caveat for all earlier results: 33 games dropped as error games came from harness bugs that were later fixed, and they are not a random sample (see "Harness bugs and the error-game caveat" in the next section).

Showdown runs of 2026-11-16 and their analyses. Full tables, scripts and caveats: `analysis/open_items/REPORT.md`
(independently verified; corrections applied; scripts `analysis/open_items/*.py`, outputs `out_*.json`). Pair score of
A with a 95% CI over pairs (normal); match-clustered CIs use only 10 matches per tag. Theory cross-references:
`theory/theory_map.md` §11.

| Item | Runs (A vs B) | Result | Reading |
|---|---|---|---|
| A. Audit | all 16 tags | 14 error games across 9 tags (partner game dropped); in 12 matches the carried Hedge/posterior state counts more games than game files | hypothesis, not a finding: an aborted retry attempt updated the carried state before the resume rule (`showdown/modal_showdown.py:163`) reran the seed; config fields constant across invocations; scores unaffected |
| B1. LOG identity | `hedge_bayes` vs Foul Play, 599 games | pair 0.3077 [0.273, 0.343]; $z-\varphi(v_1)=\sum(-R-S+D)$ closes to 4.4e-16 (by construction: telescoping); mean drift per decision $D=-0.0548$ logit (SE 0.0015, game-clustered) | the logged Bayesian LP strategy $x=Q[1]$ (Hedge's sampled expert) is not a maximin strategy of the world-averaged $\bar A$ (gap median 0.12, mean 4.6 engine units; above 1% of the payoff range on a third of decisions), and the played Hedge mixture has gap median 1.85, mean 6.12 (`engine/analysis/open_items/verify/b1_gap.py`), so these are not Theorem A's terms; the $\varepsilon_v$ lower bound stays sourced to the in-engine runs (`theory/tightening/bounds_epsv.md` §1.3). Same formula on a different leaf and opponent: not a replication, not a contradiction |
| B2. E3 switch | E3a `hedge` / E3f `hedge_fs` vs Foul Play → `exploit_top` after 8 of 16 seeds per match | fixed share − Hedge +0.0032 [−0.066, 0.072] (157 seeds); flip to the post-switch best expert in 8/10 matches, median 273 (E3a) / 245 (E3f) updates | fixed share no better than Hedge; reward-space analog of T5 within median ratio 0.91 / 0.97 (realised crossing / analog). T5 is for the Bayes posterior; E3 ran Hedge without a posterior (anytime and fixed-share) |
| B3. Exploiters | XU, XUmv, XR, XEmv: model-based best reply vs the victim | gain over RM$^+$ vs the same victim: XU +0.0354 [0.014, 0.057]; XUmv +0.0518 [−0.002, 0.106]; XEmv −0.0527 [−0.107, 0.002]; XR 0.4683 vs 0.5 | only XU's gain is clearly positive; XUmv's gain is borderline (match-clustered CI [0.007, 0.097]); XEmv and XR within noise |
| B4. E5 controls | E5t Bayesian LP, top move vs Bayesian LP; E5s Bayesian LP vs Bayesian LP | E5 0.4583, E5t 0.5500 [0.513, 0.587], E5s 0.5185 [0.480, 0.557]; E5 − E5t −0.0917, E5 − E5s −0.0621 | descriptive; a causal split of the deficit is not identifiable from the logs. Where top and pure maximin disagree (25.9% of decisions), Hedge played top 59.7%, pure maximin 31.3%; cause open |
| B5. ε sweep | `bayes_safex` vs Foul Play, ε = 0.001 / 0.003 / 0.01 / 0.03 (SX1–SX30), seeds 1–200 | 0.2000 / 0.2275 / 0.2450 / 0.2538 | monotone trend: logistic slope +0.087 per ln ε (p = 0.018 clustered by tag×match; p = 0.040 by match only); all below Bayesian LP on the same seeds (0.3291, n = 199) by 0.076–0.128 |
| B6. RQ pre-registration | `hedge_bayes` vs single-style bots (stage equilibrium / cautious / uniform) | 0.6700 / 0.8550 / 0.9948 | predictions committed in fb3cc8e before the runs (`theory/tightening/bayes_dynamics.md` §4). P1 pass (to rounding); P2 partial (4 of 6 clauses); P3 high-mismatch branch **fail** (final MAP = cautious 10/10), neither branch strictly covers the observed 0.048-nat gap, point prediction passes; P4 pass; P5 pass |
| Leaf exposure | LSs Bayesian LP vs Foul Play (2-CPU containers, 120 games); LSn Bayesian LP vs PIMC averaging (1-CPU, 120 games) | non-depth-1 leaf returns: LSs 432 / 7,246,430 calls = 0.0060% (≥1 in 59/120 games); LSn 850 / 16,172,097 = 0.0053% (≥1 in 91/120; both bots combined: the two bots share one process and one counter); scores LSs 0.2833 [0.2068, 0.3599], LSn 0.5917 [0.4981, 0.6853] (60 pairs each) | about 1 in 17,000–19,000 calls in real games (1 in 40,000 in AFdet), similar rates in the two container classes; far below the 16/240 and 17/240 changed children of the 12-process load probe (`theory/tightening/voi.md` §2.5). Mechanism: `theory/tightening/verify/leaf/README.txt`. Sources `showdown/results/{LSs,LSn}/*_summary.json` |
| Det-leaf control | AFdet Bayesian LP vs Foul Play, `OURBOT_DET_LEAF=1`, seeds 1–300 | 0.3067 [0.2693, 0.3440] vs AF 0.3144; paired diff −0.008 [−0.057, 0.040] (n = 299) | retry fired on all 826 non-depth-1 returns in 33.2M calls, 0 fallbacks; no detectable effect (`analysis/open_items/out_afdet_paired.json`) |

## Benchmark and follow-up studies (integration branch, 2026-11-16)

What was run: a ladder-style round-robin tournament of nine bots (G1), a live test of the style-posterior switching theory (T5), a live
test of the style posterior with the per-world style likelihood (`hedge_bayes_mw`) with a replication, offline measurements of a joint
set x style posterior, and an engine-simulation test of a planner that updates the posterior inside the lookahead (S1) with measurements of scouting
and concealment value. All new runs are local (the two-sided regret analysis reuses an earlier Modal run). After the harness fixes below, every new run listed here has 0 error games
(G1: 0 in 1,800). Full write-ups are under `research/`; sources are cited per subsection.

### Round-robin tournament G1 (`engine/showdown/results/gauntlet/G1_ratings.{md,json}`, `research/gauntlet_fp_iterations.md`)

Bradley-Terry ratings on an Elo scale, anchored at Bayesian LP = 0. "Foul Play at N ms" is Foul Play (decoupled UCT) with N ms per
sampled battle.

| bot | rating | 95% CI |
|---|---|---|
| Foul Play at 1000 ms | 192 | 152 to 233 |
| Foul Play at 100 ms | 160 | 119 to 200 |
| Foul Play at 400 ms | 150 | 110 to 192 |
| Foul Play at 25 ms | 135 | 94 to 179 |
| Bayesian LP, top move | 43 | -4 to 92 |
| Bayesian LP | 0 | 0 to 0 |
| PIMC averaging | -41 | -90 to 5 |
| Max damage | -286 | -367 to -220 |
| Uniform random | -911 | -1495 to -752 |

Score of the Bayesian-LP agent against Foul Play at each search time (60 pairs each, 0 error games):

| Foul Play search time (ms per sampled battle) | Bayesian LP score | 95% CI | Foul Play median iterations (p10 to p90) |
|---|---|---|---|
| 25 | 0.3167 | 0.242 to 0.392 | 17000 (7000 to 24000) |
| 100 | 0.3167 | 0.242 to 0.400 | 57000 (25000 to 83000) |
| 400 | 0.275 | 0.200 to 0.342 | 157000 (77000 to 252000) |
| 1000 | 0.225 | 0.158 to 0.292 | 377000 (196000 to 646000) |

The Bayesian-LP agent also scores 0.5375 against PIMC averaging (40 pairs), 0.55 against Bayesian LP, top move (20 pairs), 0.8 against Max damage and 1.0
against Uniform random (20 pairs each).

Meta-game (symmetric zero-sum, P = score - 0.5): the equilibrium puts all weight on Foul Play at 1000 ms (sigma = 1.000; value
-0.0000, exploitability 0.0e+00). The bootstrap is loose: Foul Play at 1000 ms's 95% CI is 0.000 to 1.000 and P(sigma > 0.01) is
0.623.

Reading (ours): the agent sits below Foul Play even at its shortest search time (rating 0 against 135 for Foul Play at 25 ms; the equivalent Foul Play
search time is out of range, reported as such in G1_ratings.md). Foul Play's strength saturates early: the four
search times span only 135 to 192, and Foul Play at 25 ms against Foul Play at 1000 ms scores 0.475 (20 pairs). The agent is not separable from PIMC and the top move at the round-robin tournament's sample size (0.5375 against PIMC over 40 pairs, consistent with the larger E1 result): their intervals (-4 to 92 and -90 to 5) overlap the agent's 0. The meta-game puts its weight on Foul Play at its longest search time,
which scores above 0.5 against every other bot in the table (only the intervals against Foul Play at the other three search times include 0.5).

### Style-switch test, live T5 (`research/prereg_style_switch.md` "Results"; `research/switch_results/out_style_switch.md`)

Runs SW2 (seeds 1-100), SW4 (1-180), SW4p (1-80, lambda = 0.98, the shipped predict-then-discount). 0 error games; all
30 matches pass the posterior reconstruction check (max error 2.3e-4 against the 1e-3 clean threshold). **SW4d was not
run.** Predictions were fixed before the runs.

| P | prediction | observed | verdict |
|---|---|---|---|
| P1 | median tau_map SW2 in [70,170], SW4 in [130,330] | 107.5 [77, 138]; 225 [150, 274] | pass |
| P2 | tau_map(SW4)/tau_map(SW2) in [1.5, 2.7] | 2.09 | pass |
| P3 (primary) | median tau_map/tau_hat in [0.7, 1.4], Spearman >= 0.5 (20 matches) | 1.06, rho = 0.86 | pass |
| P4 | median tau_90 - tau_map in [0, 25] | 4 (SW2), 6.5 (SW4) | pass |
| P5 | X: SW2 in [20,60], SW4 in [45,110] | 29.2, 66.2 | pass |
| P6 | SW4p median tau_map in [15,70] (pt 35), ratio to SW4 <= 0.4; false alarms 0/10 per arm | 37.5 [26, 62], ratio 0.17; false alarms SW2 1/10, SW4 0/10, SW4p 0/10 | pass |
| P7 | score differences within +-0.05 (post) / +-0.10 (pre) | post +0.006, +0.009; pre 0.000, 0.000 | as predicted |
| P8 | identity error <= 1e-3 (reconstruction), <= 5e-3 (logged) | 6.3e-4; 4.9e-3 | pass |

Reading: this is the first live test of T5. The posterior's detection delay after a style switch is predicted per
match from its own switch-time posterior and the measured drift rates (P3), and it scales with the pre-switch evidence
(P2). Forgetting at lambda = 0.98 cuts the delay about 6x, matching the point prediction (35 vs 37.5). One false
alarm: SW2 match 7 (P(uniform) = 0.067 at the switch, so the posterior had not yet separated the cautious style from uniform), which
misses the 0/10 point prediction but is below the refutation threshold (>= 2).

### hedge_bayes_mw (`research/prereg_mw.md` "Results" and "Replication results"; `research/mw_w1_verification.md`; `research/out_mw.md`)

Style posterior with the per-world style likelihood (each style's likelihood computed on each world's own columns). Runs MWs, BHs
(seeds 1-200), MWeq (1-100), replication MWs2 / BHs2 (seeds 201-400). 0 error games.

| P | observed | predicted | verdict |
|---|---|---|---|
| L1 | 0.297 nats/turn [0.281, 0.314] | [0.25, 0.40] | pass |
| L1-rep | 0.311 nats/turn [0.291, 0.331] | [0.25, 0.40] | pass (replicates) |
| L2 | 0.270 [0.233, 0.307] | [0.20, 0.45] | pass |
| L3 | 0.2499 [0.224, 0.276] | [0.25, 0.45] | not refuted (< 0.15 refutes); 1e-4 below the interval |
| R1 | stage equilibrium is the final MAP in 0/10 MWeq matches | at most 4/10 | pass; realisability still fails with per-world styles |
| W1 | +0.0625 [0.003, 0.122] (iid); match-clustered t(9) [-0.011, 0.136] | [-0.03, 0.03] | not refuted, interval missed; not robust |
| W1-rep | +0.0075 (MWs2 0.3425 vs BHs2 0.3350); iid SE 0.0316, one-sided lower bound -0.045; match-clustered SE 0.0358; sign-flip one-sided p = 0.45 | supported if one-sided lower > 0 | **not supported** |
| W2 | +0.095 [0.014, 0.176] vs RQeq | [-0.05, 0.05] | descriptive; confounded (RQeq ran on Modal, MWeq locally) |
| C1 | +2.0 % CPU per decision | within 10 % | as predicted |

The prediction gain replicates (0.297 then 0.311 nats/turn). The win-rate gain does not: W1 was not robust to match
clustering (`mw_w1_verification.md`) and the replication gives +0.0075. BHs2 (0.335) is in line with earlier Modal runs
of the same method (0.318-0.348), so BHs (0.2825) was a low draw. Pooled over seeds 1-400: +0.035, iid 95% [-0.008,
+0.078], match-clustered t(9) [-0.010, +0.080] (reported alongside, not the test). Same pattern as BH: better
prediction, no detectable win-rate change against Foul Play.

### Joint set x style posterior (`research/joint_posterior.md`, corrected numbers only)

Offline, on logged worlds from 18 games (10 vs Foul Play, 8 vs style_eq), paired differences in log loss (nats per turn),
game-clustered:

| difference | pooled |
|---|---|
| restricting the likelihood to each world's legal actions (current minus the averaged predictor restricted to each world's legal actions) | +0.325 ± 0.027 (vs Foul Play +0.324 ± 0.031) |
| per-world style likelihood (restricted averaged predictor minus per-world style likelihood) | +0.028 ± 0.011 |
| true joint (option iii minus set-conditional style) | +0.001 ± 0.001 |

Restricting the likelihood to each world's legal actions is the real gain: current 1.623 vs the restricted averaged predictor 1.299 nats vs Foul Play (372 turns). The old
"correlation term" is a per-world style likelihood, worth about +0.028; a joint (active set x style) posterior
carried across turns adds nothing measurable. Verdict: do not build the joint posterior; restricting the likelihood to each world's legal actions is what to
keep. The link to HBA (Albrecht & Ramamoorthy 2013) is structural only. Per-world `style_dists` costs about 4x the
averaged-matrix call (1.0 vs 3.9 ms per decision).

### Scouting and concealment (`research/info_gathering_concealment.md`, corrected; `research/s1_results.md`)

Value of revealing each hidden variable along 6v6 play in the engine simulation, per hidden variable (fraction of the depth-2 payoff range; per-world best
action disagrees in the given share of decisions):

| hidden variable | mean | median | per-world argmax disagrees |
|---|---|---|---|
| opponent unrevealed moves (active) | 0.53 % | 0 | 16 % |
| opponent item (active) | 0.45 % | 0 | 6 % |
| opponent tera type | 0.22 % | 0 | 9 % |
| all opponent sets jointly | 1.2 % | 0.3 % | 37 % |

Concealment value (what the opponent's ignorance of our set is worth): ex-ante mean 1.113 %, median 0.241 % of range
(species known); mean 1.515 %, median 0.73 % with back-row species unknown (the comparison is noise-dominated, so the
pre-registered "at least 2x" prediction is not decided). The LP argmax changes in 21/91 decisions, with one-stage
regret above 1 % of range in 4/91. Assuming an omniscient opponent is the safe side by theory (maxmin): x_AF is maxmin
against opponents who can condition on our type, a set that includes all type-blind opponents, so it guarantees v_AF
against any opponent. The valid-model strategy is the exploitative side; it would lose mean 3.4 % (median 0.9 %)
against a best-responding omniscient opponent, against 0.948 % (median 0) lost by the current assumption against one
solver-picked equilibrium. The safe-side claim rests on the theory, not on that comparison.

S1 (a planner that updates the posterior inside the lookahead, engine simulation, 3,620 decisions, 32 games): V_A1(a*) - V_A2(a*) has mean 0.0022 % of
range, median 0, and A1 and A2 choose different actions in 3/3,620 decisions (0.08 %).

| score | prediction | outcome | verdict |
|---|---|---|---|
| E1b | mean 0.5-1.5 % of range | 0.0022 % | **fail** |
| E2a | A1, A2 differ in 5-12 % of decisions | 0.08 % (3/3,620) | **fail** |
| E4 (a, b, c) | keys per cell median 2-3 (moves) and 1 (switches); merged children > 50 % | 2, 1, 67.9 % | **pass** |

Why: poke-engine branches only on crit or kill (average damage rolls), so a public observation is almost always either
impossible in a world or equally likely in every world that can produce it. Information arrives as hard eliminations,
which the matched frozen-posterior control credits to both arms, so E1 measures soft updating only and finds almost
none. E4's merging happened but bought no soft information. The clairvoyance bound at the children (A0L - A1) is mean
0.35 %, median 0. Decision (2026-11-16): accept the null. Do not port S1 live and do not build S2; damage-roll
branching is a large engine change for at most the 0.35 % bound, and no principled alternative control was found.

### Two-sided regrets along real games (`research/cce/cce_two_sided.md`, `research/cce/verification.md`)

The LOG run (hedge_bayes vs Foul Play, Modal, `OURBOT_LOG_ABAR=1`, 599 games, 17,271 usable decisions; one error game,
LOG-s4-g2, excluded) logs Ā_t, our mixture, the four experts, the five opponent styles and the realised opponent
column, so both sides' regrets are computable in our estimated stage games (units: Hedge's 400-point reward scale, per
turn).

| quantity | LOG (vs Foul Play) |
|---|---|
| our regret vs best fixed expert (top) | 0.0044 ± 0.0002 |
| opponent regret vs our equilibrium style | 0.0213 ± 0.0006 (positive in 10/10 matches); negative vs the other four styles |
| per-turn best-response regrets | 0.030 (ours, vs the realised column), 0.041 (theirs, vs our mixture); not a symmetric pair |
| our mean stage exploitability | 0.014 |
| Hedge worst-case bound at the per-match horizons | about 0.04 |

σ_T is a coarse correlated equilibrium of the realised sequence of estimated stage games only with respect to these
small comparator classes. A control in which the opponent literally plays its own maximin (CEQ, local) shows the same
0.021 ± 0.001, almost all of it model mismatch (+0.025 ± 0.001; sampling −0.001 ± 0.001; TV(y, q_eq) = 0.44; the two
sides' stage values differ by about 15 engine units). So ε_C measured through our model does not identify Foul Play's
regret; other statistics through the same model (posterior style weights, log loss, dynamic regret) do separate Foul
Play from the literal equilibrium player. ε-safe opponents (CSS, local) show a lower gap of 0.011–0.015. Every number
was recomputed independently in `verification.md`; use only its "Allowed wording".

### Harness bugs and the error-game caveat (`research/harness_bugs.md`)

Three real-Showdown harness bugs were found during G1 and fixed (`engine/showdown/headless.py`,
`engine/showdown/revival_guard.py`).
1. **Simulator startup race (local only).** Each simulate-battle launch ran `node build`, which rewrites `dist/`; a
   concurrent start could load a half-written file and hang. 5 G1 seeds lost and rerun; Modal runs not exposed. Fix:
   `--skip-build`, fail fast, guard `proc.kill()`.
2. **Revival Blessing (all methods, all runs).** The request for a fainted reserve is not understood by Foul Play or
   poke-engine: Foul Play crashes with ZeroDivisionError, our solvers with AttributeError, and others pick a living
   reserve that Showdown rejects. Past occurrences: 10 ZeroDivisionError (Foul Play), 15 invalid_choice_stall, 8
   AttributeError, i.e. 33 games. Fix: `revival_guard.guard` answers with the first fainted reserve.
3. **Forced recharge while trapped.** poke-engine returns 'No Move'. Fix: answer `recharge`.

**Caveat for earlier results:** those 33 games were dropped as error games, but not at random: they are games where a
Revival Blessing user was on the team and used the move, and seed 4 / seat p1 (Pawmot) recurs. The direction of the bias
on any one pair score is not established. After the fixes the 11 lost G1 seeds were rerun and G1 has 0 error games in
1,800. The fixes leave decisions in all other states unchanged.

### Ladder handoff

`engine/showdown/LADDER.md` documents running the Bayesian-LP agent (`agentform`) on the public Showdown ladder. It is untested against the public
server; one local rated game ran clean after the guards were added.

### Known / adapted / new (`lit/joint_belief_voi_concealment.md`)

- **Known:** Bayesian posteriors over fixed opponent types (Albrecht and Stone survey, Baker and Cowling 2007) and over
  strategies with hidden cards marginalised (Southey et al. 2005, Ganzfried and Sun 2016); PIMC strategy fusion
  (Long et al. 2010, Frank and Basin 1998); ISMCTS; public belief states; the concealment principle (Schmid et al. 2021).
- **Adapted:** the style-type posterior and its switching test (T5) in a real random-battle game; ISMCTS-style
  information-set search for the S1 planner (the closest ISMCTS results need translation to a depth-2 matrix planner).
- **Gap:** no prior measurement of the loss from a product posterior versus a joint (private type, style) posterior in
  a real imperfect-information game; our measurement is restricting the likelihood to each world's legal actions (+0.325) and an about-zero joint term.
- **Gap:** no measurement of the value of information (oracle-vs-belief gap decomposed into strategy fusion,
  non-locality and belief error) in a real game; no scouting in any Pokemon agent.
- **Gap:** no Pokemon agent models what the opponent infers about our hidden set, and no measured value of concealment
  in a real game; the PokaiTrainer paper concedes our information instead (known).

## Next experiments

**What X1 implies.** The matrix bot looks two turns ahead and then scores positions with
poke-engine's hand-written evaluation. In the same 75 ms, MCTS searches deeper with that same
evaluation and wins. This is consistent with the leaf evaluation error dominating the bound, not the equilibrium solver. That points
to search plus a learned value function, the pattern behind the strong systems in the literature:
- PokaiTrainer reaches parity with decoupled UCT only by adding a value net to CFR.
- Wang 2024 uses MCTS guided by PPO.
- Metamon uses offline RL on human battles.

Plain policy RL is not the route at our budget: our PPO pilot wins 5–13% against MCTS at 200
iterations, even in the easier 12-species, full-information setting. A value net also attacks the
speed problem. Most of the 72 ms per move is Python engine expansion, which one batched forward pass
over the children replaces. Caveat: the matrix builder is Python and MCTS is Rust, so X1 partly
measures implementation speed.

1. **V1, value-net leaf (done, negative):**
   - Data: 30k MCTS-1000 self-play games (ε=0.1), 1.1M positions labelled with the game outcome.
     Antisymmetric 256-256 MLP; `results/value/`.
   - The net predicts outcomes no better than the engine heuristic: held-out sign accuracy 0.726
     vs 0.732. More data (8k → 30k games) didn't help, and val loss bottoms out within one epoch.
   - Played as the leaf of a depth-1 matrix (RM+, top move, 38 ms per move; 200 games each,
     `results/experiments/V1.jsonl`):

     | Opponent | Score |
     |---|---|
     | MCTS, time-matched (35 ms) | 0.235 [0.185, 0.285] |
     | MCTS, 2,000 iterations (2.7 ms) | 0.315 [0.260, 0.370] |
     | Equilibrium, top move, depth-2 heuristic matrix | 0.273 [0.224, 0.321] |

   - Reading: a value net fit to game outcomes doesn't replace search. The ~37 positions of a game
     share one label, so it mostly learns team strength, not the consequences of a single move.
     Untested: depth 1 vs depth 2 isn't separated (no depth-1 heuristic control). Next: train on
     lower-variance targets (MCTS root values, or TD bootstrapping in self-play, i.e. actual RL),
     and run the depth-1 heuristic control.
2. **Belief-conditioned values for hidden information:** feed the belief, not one sampled world, to
   the value net, inside the Bayesian-LP solve. This keeps the joint-solve advantage, closes part of
   the 0.1 oracle gap, and is the only way concealment can have value (direction E).
3. **Exploitability (E6), using RL as the measuring tool** (scaffolded: `psro.py --mode rl_br`, see above):
   train PPO best responses against frozen agents (top move vs sampled). The opponent is fixed, so this is much easier than learning a
   strong policy. It tests whether mixing buys robustness.
4. **Belief upgrades:** item and ability inference from damage and speed, and behaviour-based
   updates. Measure how much of the oracle gap closes.
5. The E4 exploiter arm: bandit (shares and most-visited) and RM+ roots against an exploiter, to measure the exploitability cost of most-visited play (see "Real-Showdown experiments"). Done 2026-11-16: see "Open items (2026-11)", B3.

## Relation to prior work

Full reviews with quotes and BibTeX are in `../lit/pokemon_ai.md` (Pokémon AI; §7 cross-references
each conclusion) and `../lit/game_theory_noregret.md` (game theory and no-regret learning; §4b and
§4c). Cite keys below match the game-theory file's BibTeX block. Most quotes were extracted by
Undermind PDF readers and still need checking against the PDFs. PokaiTrainer's arXiv record was
checked directly: arXiv:2608.29197, M. Yu, 2026-08-29.

**What is not new: the formulation.**
- PokaiTrainer [Yu26] already solves every Pokémon VGC decision as a Bayesian matrix game: one
  strategy for the acting agent across hidden-information worlds, the opponent best-responding in
  each world. It uses linear CFR at 100–300 iterations. The LP for zero-sum games with one-sided
  incomplete information is classical [Pon80].
- We should present Bayesian-LP solving as an existing method that we evaluate, not as a new one.
- Other prior art we should cite rather than claim:
  - maximin pure play (PokéChamp; Sarantinos 2022);
  - "alpha-beta is unsound for simultaneous moves" (Sarantinos 2022; pruned search computes only
    bounds on cells [Saf12, Bos16]);
  - "the top move suffices against fixed opponents, mixing is for adaptive ones" [Yu26, Gan16];
  - evidence-based set filtering (Foul Play);
  - search on top of RL (PokaiTrainer, Metamon, Wang 2024).

**What appears new** (not found in the located literature):
1. **A head-to-head of Bayesian LP vs PIMC averaging at equal compute,** with an effect size of
   0.585 [0.540, 0.630]. Yu26 has no PIMC baseline. Strategy-fusion theory [Fra98, Lon10]
   predicts the direction. It also fits ISMCTS beating determinized MCTS in Pokémon [Iha18].
2. **A per-decision bound on PIMC-averaging loss, checked on real decisions.**
   - Lon10 explicitly gives no loss bound. Arj24 only shows that strategy fusion shrinks with
     lookahead. Li24e bounds risk under wrong type beliefs with L1 and diameter terms, the closest
     analogue.
   - The argument is elementary, and a hand check tightens our constant. Pick, for each world, a
     column within the pairwise Hausdorff distance d_kl of each column of world k, then use
     concavity of the minimum and Bayesian LP value ≤ Σ β_k v_k. The result is PIMC-averaging loss ≤
     Σ_{k,l} β_k β_l d_kl ≤ max d_kl ≤ 2·δ_H, where δ_H in our code is the radius around a central
     column set. That is tighter than the 4·δ_H we tested.
   - Empirically the largest loss is 0.39·δ_H, consistent with this.
   - The argument has been checked by hand, not formally verified. Send it to `prove` before
     stating it in a paper.
3. **Quantifying poke-engine's pruned matrix** (55–80% of cells missing) and showing that it breaks
   equilibrium bots. The poke-engine README prints NaN cells without explaining them.
4. **A direct measurement of the cost of sampling against a fixed opponent** (top move 0.58 vs
   sampled 0.44 against MCTS). Yu26 argues the point and measures the other side: an informed local
   best response wins 77.7% against its top move but only 49.6% against its full mix (Table 9).
   Together these give both sides of the trade-off that E6 should measure in our setting.

**What is consistent with prior work:**
- MCTS is strong in Pokémon: Foul Play, built on poke-engine and using decoupled UCT over sampled
  opponent sets, won the PokeAgent Gen 9 OU final.
- Model-free self-play is far below decision-time search unless it gets offline data, imitation
  initialisation or very large scale [Yu26, Ang25, Per22; Metamon].
- RM+ reaching LP-level play within 100 iterations matches the iteration counts used by Yu26.
  RM+ and OMWU last-iterate theory is weak or slow [Cai23, Far23, Cai24].
- Beliefs as the main lever: Yu26 reports +8–12 points from a hidden-stat belief summary.
- No located paper contradicts any of our findings.

**New no-regret directions** (details and supporting papers in `../lit/game_theory_noregret.md` §4
and §4c):
- **A. δ_H-gated Bayesian LP.** Predict the 15% of decisions where PIMC averaging changes the top move,
  from δ_H or per-world top-move disagreement. Calibrate on `E5_decisions.jsonl` and solve the
  joint LP only there. Related: Lon10's a priori predictors, Bro17's selective re-solving.
- **B. E6 exploitability and regret-gated exploitation.** Replicate Yu26's local-best-response
  protocol against top move, maximin, sampled equilibrium and Bayesian LP. Then choose online
  between top move and the mix with a restricted-response or confidence test [Joh09, Li26j],
  tracing the win-vs-MCTS against win-vs-best-response front [Li24e].
- **C. Decompose the 0.1 oracle gap.** Compare Bayesian LP with the true posterior, the current
  belief and a prior-only belief; add item and ability inference.
- **D. Warm-started and predictive RM+.** Reuse the previous turn's regrets to reach LP-level play
  in 10–20 iterations, so Bayesian LP costs no more than PIMC [Bro16, Far23].
- **E. Belief-indexed leaves and depth 3.** Give concealment a nonzero value, which is impossible
  with depth-1 scalar leaves [Bro20, Sch21].
- **F. MMD toward a prior.** A smooth dial between the top move and the equilibrium, scored against
  both MCTS and a best response [Sok22].

## Reproduce

```bash
cd final/engine
.venv/bin/python test_matrix.py && .venv/bin/python test_hidden.py
.venv/bin/python experiments.py E1fix --workers 6
.venv/bin/python hidden_experiments.py E4 --only 0 --workers 6 --log-decisions   # Bayesian LP vs PIMC, 200 pairs
.venv/bin/python hidden_experiments.py E4 --pairs 50 --workers 6 --log-decisions # other matchups, 50 pairs
.venv/bin/python hidden_experiments.py E4am --workers 6
.venv/bin/python figures/make_figures.py
```
