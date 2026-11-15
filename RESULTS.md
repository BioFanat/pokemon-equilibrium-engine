# Results: equilibrium play and Bayesian agent-form solving in Pokémon Random Battles

Status 2026-11-14. All numbers below come from files in `results/` and are recomputed by
`figures/make_figures.py` and the `summary` commands of `experiments.py` / `hidden_experiments.py`.
Terms follow `figures/FIGURE_VOCAB.md`. Literature cross-references and citations are added
in the section "Relation to prior work", after the literature review.

## Headline

1. **Hidden information: the agent-form Bayesian solver beats PIMC blending.** It scores
   **0.585 [0.540, 0.630]** over 400 games with equal compute (4 sampled worlds each).
2. **The blend-loss bound holds on real play.** Across 46,890 real decisions, blend loss never
   exceeded the 4·δ_H bound; the largest loss was 0.098 of it.
   - Blending loses value in 61% of decisions, but the median loss is only 0.19% of the payoff range.
   - The tail carries the effect: 27.7% of decisions lose at least 1%, and 0.7% lose at least 10%.
3. **Full information: matrix solving beats MCTS only with 25× more time, and not by mixing.**
   Against MCTS at 2,000 iterations (2.7 ms per move), both deterministic bots win:
   - Equilibrium, top move: **0.58 [0.54, 0.62]**
   - Safest move: **0.56 [0.52, 0.60]**

   Against time-matched MCTS (75 ms per move, X1) both lose clearly: top move **0.345
   [0.289, 0.401]**, Safest move 0.263 [0.210, 0.315]. Depth-2 matrix solving is not competitive
   with MCTS per unit of compute. Sampling the equilibrium scores only 0.44 [0.40, 0.48] vs MCTS
   at 2,000 iterations. Against an opponent that doesn't adapt, randomizing only costs value.
4. **Infrastructure finding.** poke-engine's iterative-deepening matrix is mostly pruned: 55–80% of
   cells are missing. Treating it as a payoff matrix silently breaks every equilibrium method; our
   first round of experiments (E1–E3) was invalidated this way.
5. **Plain self-play RL is cheap per decision but weak.** After 30M decisions (25 min on a laptop),
   PPO wins 5–13% against MCTS (200 iterations) (actually 1,000; poke-engine rounds up to multiples of 1,000), in a simpler 12-species, full-information setting.

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
  the probability-weighted average, over chance outcomes, of a leaf value.
  - The leaf value is the midpoint of the two sides' maximin values on the child position, taken
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
    - **PIMC blend:** averages the equilibria of 4 sampled worlds.
    - **Agent-form:** solves max_x Σ_k b_k min_j (M_kᵀx)_j over the same 4 worlds as one LP, with
      the opponent best-responding in each world.
- **Evaluation.** Games come in pairs with the same teams and seed and sides swapped. Score = wins +
  ½ draws, and 95% CIs are taken over pairs. Games stop at 300 turns, which counts as a draw.

## Results

### Fixing the payoff matrix, full information (Fig. 1a; `results/experiments/E1fix.jsonl`, `E1.jsonl`)

| Agent vs MCTS (2,000 iterations) | Pruned matrix | Full matrix | Time per move (agent / MCTS) |
|---|---|---|---|
| Safest move | 0.434 [0.393, 0.475] | **0.561 [0.518, 0.604]** | 67 ms / 2.7 ms |
| Equilibrium, sampled | 0.302 [0.257, 0.348] | 0.443 [0.402, 0.483] | 74 ms / 2.7 ms |
| Equilibrium, top move | — | **0.581 [0.540, 0.623]** | 72 ms / 2.7 ms |

- Equilibrium, sampled vs Safest move: 0.516 [0.475, 0.558], a tie.
- The pruned matrix's signature was a side split. The bot did fine as side one and collapsed as
  side two: Safest move scored 0.618 as side one and 0.250 as side two (E1, 200 games per side).
  Side-two maximin needs the cells that pruning drops.
- The full-matrix bots use about 25× more time per move than MCTS at 2,000 iterations. At
  matched time (X1, below) MCTS wins.
- 10–13% of games between the deterministic bots and MCTS hit the 300-turn cap.

### Hidden information (Fig. 2; `results/hidden/E4.jsonl`, `E4am.jsonl`)

| Matchup (score of first-named) | Sampled play | Top-move play |
|---|---|---|
| Agent-form vs PIMC blend | **0.585 [0.540, 0.630]**, n=400 | 0.540 [0.452, 0.628], n=100 |
| Agent-form vs Most-likely set | 0.440 [0.338, 0.542], n=100 | 0.525 [0.452, 0.598], n=100 |
| PIMC blend vs Most-likely set | 0.535 [0.455, 0.615], n=100 | 0.560 [0.466, 0.654], n=100 |
| Oracle vs Agent-form | 0.600 [0.513, 0.687], n=100 | 0.620 [0.525, 0.715], n=100 |

- **Agent-form vs PIMC** is the only matchup run at full size. The first 50 pairs scored 0.60 and
  the next 150 scored 0.58, so the result is stable.
- **Equal compute in that matchup:** both use 4 worlds at about 340 ms per move.
- **Oracle:** knowing the hidden sets is worth about 0.1 in score under both kinds of play. That gap
  is the room available to better belief modelling.
- **Most-likely set** uses one world (about 85 ms per move against 320–340 ms). Its matchups have
  only 100 games and are inconclusive.
- **Top-move play** removes the cost of sampling but adds draws: 10–12% of games hit the turn cap
  in two matchups.

### Blend loss vs the bound (Fig. 3; `results/hidden/E5_decisions.jsonl`)

Every agent-form and PIMC decision in E4's sampled play was logged: 46,890 decisions.

| Quantity | Value |
|---|---|
| Decisions where blending loses value (> 1e-6) | 61.0% |
| Blend loss, median / 90th / 99th percentile / max (share of payoff range) | 0.19% / 2.6% / 8.8% / 25.0% |
| Decisions with loss ≥ 1% / ≥ 10% of range | 27.7% / 0.7% |
| Bound violations (loss > 4·δ_H) | **0** |
| Largest loss / bound | 0.098 |
| Top move differs between agent-form and blend | 14.9% of decisions |
| Median loss when the top move differs / is the same | 1.33% / 0.06% |

- The bound is valid but loose, by a factor of 10 or more.
- The agent-form advantage comes from a minority of decisions where the worlds disagree enough that
  averaging per-world equilibria (strategy fusion) picks a different action.

### Self-play RL baseline (Fig. 1b; `results/ppo_pilot.jsonl`)

- **Setup:** PPO self-play with a 256-256 MLP, full information, the 12-species hand-built pool;
  30M decisions in 25 min on 12 cores.
- **Result:** beats random at about 95%, but wins only 5–13% against MCTS (200 iterations) (actually 1,000; poke-engine rounds up to multiples of 1,000), rising
  slowly.
- **Comparability:** it ran in a simpler setting than everything above, so it is a cost and
  learning-curve reference only.

## Conclusions

1. **Hidden information:** the agent-form game beats PIMC blending at equal compute,
   0.585 [0.540, 0.630]. This is the project's central claim and holds at 400 games. The
   formulation itself comes from PokaiTrainer [Yu26]; the head-to-head comparison is ours.
2. **The bound holds but is loose.** 4·δ_H always upper-bounds blend loss on real decisions, and
   a hand argument tightens it to 2·δ_H (see Relation to prior work). As a gate for when to run the
   joint LP, δ_H is weak (X3), and it isn't needed: agent-form costs the same as PIMC.
3. **No-regret solving is enough per turn.** On the full matrix, RM+ reaches a Nash gap of
   2·10⁻⁴ of the payoff range in 1,000 iterations (`test_matrix.py`). In play, RM+ at 100 iterations
   tied the exact LP (0.501 [0.462, 0.541], E3), but E3 used the pruned matrix, so that is evidence
   about solvers only. The bottleneck is the matrix (depth, leaf value, hidden information), not
   the solver.
4. **Randomizing trades value for robustness.** Against fixed MCTS, sampling the equilibrium costs
   about 0.14 compared with its top move (0.44 vs 0.58). Against an exploiter that knows the bot's
   algorithm (E6q), the ranking flips: sampled 0.475, safest 0.367, top move 0.317. Quick test,
   60 games each; the exploiter looks one turn ahead.
5. **Information is valuable.** The oracle gains about 0.1 over agent-form. Better beliefs (item and
   ability inference, behaviour-based updates) are the clearest lever.
6. **Methodology:** search matrices from pruning engines are not payoff matrices. Any equilibrium
   approach built on such an engine must recompute the pruned cells.
7. **The leaf value is the bottleneck, and a cheap learned value doesn't fix it.** Time-matched MCTS
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
| Nash gap ≤ solver + belief + value error | L2: v(b) − f_b(x̂) ≤ ε_T + 2Δ·TV (Δ = payoff range; = 4·TV on [−1,1])(b, b̂) + 2‖M − M̂‖_∞ | Random instances: max ratio 0.48. In play: solver term ≈ 0; belief term ≈ the 0.10–0.12 oracle gap; value term dominates (X1, V1) | Bound checked; the ranking of terms is measured, not proved |
| Optimizing per hidden set is exploitable | Strategy fusion; blend loss ≤ 4·δ_H (hand: 2·δ_H) | Agent-form beats PIMC 0.585 [0.540, 0.630]; bound holds on 46,890 decisions | Strongest result; the 2·δ_H proof still needs the `prove` workflow |
| Memory sufficiency / compression | Belief-state theorem (cite); L2′ marginals loss ≤ 2Δ·TV (Δ = payoff range; = 4·TV on [−1,1]) | Belief is the exact joint posterior over Random Battle roles; no marginals ablation run | Not tested |
| Value of information | L3: v(b) convex, VOI ≥ 0, = 0 iff the prior optimum stays optimal | Oracle gap bounds total VOI at ≈ 0.1 score; no probing experiment | Not tested |
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
- Long games: 2–30% of games hit the 300-turn cap (up to 40% for LBR in the white-box runs), and they count as draws.

## Exploratory runs (2026-11-14)

- **X1, time-matched MCTS** (`results/experiments/X1.jsonl`, 200 games each, MCTS at 75 ms per
  move): Equilibrium, top move 0.345 [0.289, 0.401]; Safest move 0.263 [0.210, 0.315]. The E1fix
  win came from compute, not from the matrix method.
- **X2, 8 vs 4 sampled worlds** (`results/hidden/X2.jsonl`, both agent-form with top-move play,
  100 games): 0.495 [0.414, 0.576]. There's no sign that more worlds help, at 2× the time per move
  (735 vs 367 ms).
- **X3, δ_H as a gate (offline on E5 decisions):**
  - Running agent-form on the half of decisions with the largest δ_H/range catches 70% of
    top-move changes and 81% of blend loss. Spearman(δ_H, loss) is 0.53; the AUC for spotting a
    changed top move is 0.65.
  - Gating also saves almost nothing: agent-form and PIMC cost the same per move (343 vs 344 ms),
    because building the matrices dominates and δ_H needs the same matrices. Direction A is
    dropped: always run agent-form.

- **X4, EXP3 over whole-battle experts (offline replay, `exp3_replay.py`, `results/exp3/replay.json`):**
  - Setup: experts are Equilibrium top move, Equilibrium sampled and Safest move. Each round's
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
    top move from Safest move (a 0.02 gap). Experts-across-battles only pays off when experts
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
    | Safest move | 0.367 [0.268, 0.466] | 0.561 |
    | Equilibrium, top move | 0.317 [0.232, 0.401] | 0.581 |

  - Per turn, the best reply to the equilibrium mix gains exactly 0 (checked on test positions),
    against up to 15% of the payoff range for the top move.
  - Reading: the deterministic bots lose about 0.13–0.18 to an exploiter; the sampled bot is not
    distinguishable from 0.5. The ranking flips relative to fixed MCTS. That is the
    robustness–exploitation trade-off theory predicts, and it matches PokaiTrainer's informed best
    response (77.7% vs argmax, 49.6% vs the full mix). The CIs are wide; the sampled-vs-top
    difference (0.16) is about at the edge of significance at 60 games each.
  - Draws: 18 of 60 top-move games hit the 300-turn cap (deterministic loops).
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
  - Targets: Agent-form with top-move play and with sampled play, 20 pairs (40 games) each.
  - Control: the exploiter differs from the Oracle only in knowing q. So exploitation gain =
    (target vs Oracle) − (target vs exploiter). Oracle results: Agent-form scores 0.38 with
    top-move play (E4am) and 0.40 with sampled play (E4).
  - Prediction: the gain is larger for top-move play than for sampled play. q is computed from
    resampled worlds, so even top-move play is only partly predictable.
  - Code: `hidden_agents.Agent` (`xlbr` kind, `exploit`); `play_hidden` passes the opponent's
    view and belief to `xlbr` agents.
  - Result (score of Agent-form, 40 games each; CIs about ±0.13–0.16):

    | Agent-form play | vs exploiter (X5) | vs Oracle (control) | Exploitation gain |
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
  - Experts: top move, sampled and safest, all from one full matrix per turn, so the learner costs
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
    | vs MCTS-2000, battles 0–39 | 0.600 ± 0.084 | top move 0.58, safest 0.56, sampled 0.44 |
    | vs exploiter, battles 40–49 | 0.417 | vs own exploiter: sampled 0.475, safest 0.367, top move 0.317 |
    | vs exploiter, battles 50–59 | 0.400 | |
    | vs exploiter, battles 60–79 | 0.458 | |

  - Mean weights over seeds (top move / sampled / safest):
    - start 0.35 / 0.31 / 0.34;
    - battle 9: 0.43 / 0.23 / 0.34;
    - battle 39: 0.43 / 0.23 / 0.34;
    - battle 44 (5 after the switch): 0.27 / 0.47 / 0.27;
    - battle 59: 0.20 / 0.61 / 0.19.
  - Reading: with per-turn counterfactual feedback, the learner matches the best expert against
    MCTS. It moves to the sampled expert within about 5 battles of the switch, where bandit EXP3
    on win/loss didn't move in 300 (X4b). Its exploiter-phase score approaches the sampled
    expert's level by the last 20 battles. Small n: phase CIs are about ±0.08, sub-phase ones
    wider.
  - An earlier run with a buggy exploiter (it played its first option whenever the learner was
    forced) is kept in `results/exp3/buggy_exploiter/` with a note; don't use it.
- **Convergence rates (2026-11-14, computed inline, no new battles).**
  - Per-turn solver, RM+ on 35 full matrices (Nash gap as a share of the payoff range):

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
- η_NL is strategy fusion / non-locality (Frank & Basin 1998; Kovařík et al. 2023).

Ours are three things: the H̄ = best-response-length refinement with its counterexample, the necessity of the TV term (the 3-type counterexample), and the Hedge composition below.

### Hedge over experts: safety theorem (`theory/hedge_experts.md`)
- **Proposition E1 (safety, holds against any adaptive opponent).** With a fixed reward normaliser Δ̄ and an ε_s-equilibrium expert in the pool, the surrogate payoff of the played mixtures p_t against the opponent's realised moves is at least Σ_t (val(Â_t) − ε_s) minus Δ̄·Reg_S.
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

### Exact endgames: leaf error and bound tightness (`endgame_eps.py`, `results/endgame/eps.jsonl`)
- **Not exact after all.** No endgame fits 20k states exactly, because of PP combinatorics. All rows use PP abstraction plus HP buckets (/10 or /20), so V* and the exploitability are exact only for the abstracted game. Skipped: kingambit vs clefable, the 2v2s, and 5 of 8 extra pairs.
- **Bound ≥ exploitability on every evaluated row**, but the bound is nearly vacuous: refined bound / exploitability is 6.9–155 (primary refined bound; the 22 rows with exploitability > 1e-3 and a finite bound. It is 5.1 if the short-tie-break bound is used where the primary bound is infinite) where exploitability > 0.
- **The leaf is the error.** The calibrated ε_v (best per-state a·M + c fit to Q* on the [-1,1] scale) is 0.8–1.0 at its maximum nearly everywhere. The best fit is a constant (a = 0) at 9–67% of live states, meaning the heuristic carries almost no information about Q* in these endgames. ε_s is about 0.
- **True exploitability of the matrix bots, selected rows (equilibrium sampled):**

  | Case (abstracted) | Seat | Exploitability | H̄ | Refined bound |
  |---|---|---|---|---|
  | garchomp vs rotomwash 100% | 1 / 2 | 0.338 / 0.602 | 3.2 / 3.0 | 4.5 / 4.6 |
  | greattusk vs gholdengo 100% | 1 / 2 | 0.215 / 0.120 | 1.1 / 1.0 | 2.0 / 2.0 |
  | gholdengo vs heatran 100% | 1 / 2 | 0.074 / 0.425 | 2.0 / 31.4 | 1.6 / 16.4 |
  | greattusk vs rotomwash 100% | 1 / 2 | 0.190 / 0.859 | 2.6 / 6.7 | 2.9 / 10.8 |

  Safest reaches 1.60 and top move 1.26 (greattusk vs rotomwash, seat 2).
- **One failed sanity check.** The exact-equilibrium row for garchomp vs rotomwash 100%, seat 1, has exploitability 0.029 (should be about 0) and infinite H̄. Shapley hit its sweep cap with a final change of 3e-8. Diagnosed: under (exact stage equilibrium, best response), two reachable states form an endless loop: earthquake into a levitating Rotom-Wash vs pain split, which the PP abstraction never ends. Valuing endless play at V* gives exploitability 1e-16. Valuing it as a draw, the best-response convention, gives 0.029. The stage-wise equilibria of V* are optimal only if endless play is valued at V*. This is the non-termination problem of undiscounted stochastic games, and it is also why Shapley needs 5,000 sweeps. The other 11 exact-equilibrium rows are below 2.5e-10.
- **Reading.** This supports the theory's main claim that ε_v dominates, and the X1 / V1 finding that the leaf is the bottleneck. It also shows that per-state worst-case ε_v makes the bound uninformative; a typical-state (mean) version would be tighter.

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

### Foul-play-style agent and Hedge over {MCTS, equilibrium} (`results/experiments/E7.jsonl`, `E6f.jsonl`, `H1.jsonl`)
- **Agent `foulplay_*`.** It reproduces foul-play's selection rule (`fp/search/main.py` `select_move_from_mcts_results`): MCTS visit fractions, keep the options with ≥ 75% of the top fraction, then sample in proportion. One world, full information.
- **Exploiter for MCTS targets.** For a foul-play or MCTS target, LBR estimates the target's q from its own MCTS run.
- **Scores** are A's score; the unit is a pair (both seats, same seed); 95% CIs.

| Exp | A | B | Games | Score | 95% CI | ms/move A / B |
|---|---|---|---|---|---|---|
| E7 | foul-play, 2,000 it | MCTS, 2,000 it (sanity check) | 100 | 0.505 | [0.417, 0.593] | 2.9 / 2.8 |
| E7 | foul-play, 2,000 it | Equilibrium, sampled | 100 | 0.490 | [0.399, 0.581] | 2.8 / 75 |
| E7 | foul-play, 2,000 it | Equilibrium, top move | 100 | 0.310 | [0.235, 0.385] | 2.8 / 74 |
| E7 | foul-play, 75 ms (time-matched) | Equilibrium, top move | 100 | 0.650 | [0.571, 0.729] | 72 / 71 |
| E6f | foul-play, 2,000 it | its LBR exploiter | 60 | 0.275 | [0.161, 0.389] | 2.9 / 71 |
| E6f | MCTS, 2,000 it (most visited) | its LBR exploiter | 60 | 0.258 | [0.166, 0.351] | 2.6 / 75 |
| E6q | Equilibrium, sampled | its LBR exploiter | 60 | 0.475 | [0.377, 0.573] | 74 / 73 |
| E6q | Equilibrium, top move | its LBR exploiter | 60 | 0.317 | [0.232, 0.401] | 77 / 77 |
| H1 | Hedge {foul-play MCTS, equilibrium sampled} | MCTS, 2,000 it | 100 | 0.560 | [0.473, 0.647] | ~74 / 2.6 |
| H1 | Hedge {foul-play MCTS, equilibrium sampled} | white-box exploiter (knows p) | 60 | 0.408 | [0.294, 0.522] | 36 / 38 (recorded; the plan's cost splits by seat) |

- **Time-matched follow-ups** (`E7t.jsonl`, `E6ft.jsonl`):
  - foul-play at 75 ms vs equilibrium sampled: **0.760** [0.677, 0.843] (100 games, 70 vs 61 ms per move).
  - foul-play at 75 ms vs its LBR exploiter: **0.533** [0.437, 0.629] (60 games).
- **Correction: an LBR score mixes exploitation with strength.** The LBR exploiter is itself a depth-2 matrix bot, so its score is not pure exploitation. The fair measure is the exploitation gain: the score against a non-exploiting opponent of the same strength, minus the score against LBR. For foul-play, that opponent is equilibrium top move: same matrix as LBR, but it doesn't model its opponent. For the equilibrium bots, it is the bot itself (0.5 by symmetry).

  | Bot | vs same-strength non-exploiter | vs LBR | Exploitation gain | 95% CI |
  |---|---|---|---|---|
  | foul-play, 2,000 it (2.8 ms) | 0.310 | 0.275 | 0.035 (CIs overlap) | [−0.10, 0.17] |
  | foul-play, 75 ms | 0.650 | 0.533 | 0.117 | [−0.01, 0.24] |
  | Equilibrium, top move | 0.5 | 0.317 | 0.183 | [0.10, 0.27] |
  | Equilibrium, sampled | 0.5 | 0.475 | 0.025 | [−0.07, 0.12] |
  | MCTS, 2,000 it (most visited) | 0.419 (E1fix, cross-experiment) | 0.258 | 0.160 | [0.06, 0.26] |

  - The earlier reading, "foul-play is as exploitable as plain MCTS", was confounded: at 2,000 iterations LBR mainly wins by being the stronger bot.
  - Only top move's gain (z = 4.3), plain MCTS's gain (z = 3.1) and the top-move minus sampled difference (0.158 [0.03, 0.29]) are significant. Foul-play's lower gain is suggestive (vs plain MCTS z = 1.45; vs top move z = −1.8) and depends on the baseline: with equilibrium sampled as foul-play's baseline its gains become 0.21–0.23.
  - The gains come from different experiments with CIs around ±0.1, so they are indicative only.
- **Strength vs robustness.** At equal time, foul-play-style MCTS beats equilibrium play (0.650 vs top move, 0.760 vs sampled), consistent with the leaf bottleneck. Equilibrium sampling is less exploitable. The result is a trade-off, not a dominance, which motivates combining them (Hedge, below) and a better leaf value. A depth-2 white-box exploiter confirms the ordering at 100 pairs (W1, below): top move 0.873 vs sampled 0.713, difference +0.160 [+0.091, +0.229].
- **RL best response on Modal** (`results/psro/rlbr_exploit/`, config `configs/rlbr_exploit.json`):
  - Setup: 100k PPO decisions per target, warm-started from the PPO pilot net on the 12-species full-information pool. Each net is evaluated on 200 held-out games, alongside the untrained start net on the same seeds.

    | Target | Untrained net | Trained net | Gain |
    |---|---|---|---|
    | foul-play, 2,000 it | 0.043 | 0.088 | +0.045 |
    | Equilibrium, sampled | 0.108 | 0.123 | +0.015 |
    | Equilibrium, top move | 0.135 | 0.158 | +0.023 |

    95% CIs are about ±0.02–0.04 per win rate. Wall time 0.73 h; the budget meter estimated $2.65 against a $1.19 plan. The meter is conservative: it bills min(inputs, 64) warm containers per map, and actual task time was 8.3 core-hours. Check the real bill in the Modal dashboard.
  - **Reading: not informative yet.** The best response stays far below 0.5 against every target, so the PPO arm's lower bounds are negative. Any fixed exploiter gives a valid lower bound, so LBR's scores were valid positive bounds throughout (win rate − 0.5: top move ≥ 0.18, foul-play 2,000 it ≥ 0.22, sampled ≈ 0.03, n.s.). The gains are within noise, except possibly against foul-play (z ≈ 2.4 under independence; seeds are shared, so a paired test would be tighter).
  - The pilot net is too weak a starting point; it won 5–13% against MCTS-200 (actually 1,000; poke-engine rounds up to multiples of 1,000) in the pilot. A useful RL exploitability measurement needs a much stronger base policy, or a far larger budget. A white-box search exploiter is the better tool (below).
- **RL best response at scale (rlbr_scale)** (`results/psro/rlbr_scale/curve_full.json`, logs `results/psro/run_rlbr_scale.log` and `results/psro/run_rlbr_scale_resume.log`, configs `configs/rlbr_scale.json` and `configs/rlbr_scale_resume.json`):
  - Setup: PPO best response vs foul-play-style MCTS (`foulplay_it:2000`), warm-started from the PPO pilot net, planned for 8M decisions; 200 held-out games per curve point, same seeds throughout.

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
  - **Reading:** 68× more training moves the win rate from 0.06 to 0.125. It stays far below 0.5, so RL gives no useful exploitability bound here. The white-box exploiter certifies ≥ 0.25 (top move) / ≥ 0.09 (sampled) with no training.
- **Gray-box RL best response (rlbr_gray, planned 2026-11-15; written before the run).** Same as rlbr_scale, except that the PPO observation also includes foul-play's move distribution q for the current turn (14 action slots). This is the exact distribution foul-play then samples from, not a fresh estimate.
  - The init is the pilot net with zero weights on the new inputs, so curve point 0 must reproduce 0.060 on the same held-out seeds.
  - Config `configs/rlbr_gray.json`: 8M decisions in one run, games_per_input 16, curve points at 1M/3M/5M/6M/8M. Cap $4.
  - Tests the claim in `theory/search_rl_links.md` (L3) that knowing the opponent's policy, not more games, is what separates white-box search from RL.
  - Interpretation limits, fixed in advance:
    - The gray-box net gets q only. The white-box exploiter gets q plus the exact stage matrix. A null result therefore means "q without a value model is not enough", not "knowing the policy doesn't matter".
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
  - Per-game Hedge (weights reset each game; median 32–39 turns, p90 51–64): 0.560 vs MCTS, 0.408 vs the white-box exploiter that knows its mixture. Mean final weight on the MCTS expert 0.46 vs MCTS, 0.38 vs the exploiter.
  - **Weights carried across games** (`hedge_stream.py`, the setting Thm E2 covers; 4 seeds × 30 games vs MCTS, then 30 vs the white-box exploiter, same game seeds as a per-game-reset control; `results/experiments/hedge_stream_seed*.jsonl`):

    | Weights | Opponent | Games | Score | 95% CI (seeds, t₃) | MCTS-expert weight (end of game, mean over games) |
    |---|---|---|---|---|---|
    | Carried | MCTS, 2,000 it | 120 | 0.533 | [0.355, 0.712] | 0.34 |
    | Carried | white-box exploiter | 120 | 0.533 | [0.441, 0.625] | 0.14 |
    | Reset | MCTS, 2,000 it | 120 | 0.471 | [0.401, 0.541] | 0.45 |
    | Reset | white-box exploiter | 120 | 0.388 | [0.308, 0.467] | 0.38 |

    Paired on (seed, game): carried − reset = +0.062 [−0.050, +0.175] vs MCTS, **+0.146 [+0.039, +0.253] vs the exploiter**.
  - Reading: carrying weights costs nothing against MCTS and makes the learner significantly harder to exploit. The MCTS-expert weight drifts slowly against MCTS (0.45 → 0.28 over 30 games), then falls faster once the exploiter starts (0.28 → 0.07). The per-game learner restarts near uniform each game, so the exploiter always faces a large MCTS share.
  - Caveat: both experts are rewarded under the heuristic matrix A, which favours the equilibrium expert by construction. The drift is therefore toward "best under A", not toward the expert that wins more games.
  - With the flawed incremental update, the carried learner had collapsed to the equilibrium expert within about 10 games (weight 0.10 by game 29), and the carried − reset difference was −0.10 vs MCTS. That was an artefact of the bug.

### Leaf value and exploiter brainstorms (2026-11-14; preliminary, scratch harness)
Scripts and logs: `/private/tmp/claude-501/-Users-biofanat-Documents-classes-f26-cs535/2bbe670b-2f97-4d9a-bfad-e1139e0352c6/scratchpad/leaf/` and `.../scratchpad/rlbr/`. No project code changed.
- **poke-engine MCTS runs at least 1,000 iterations.** Requests of 1, 50, 200 or 1,000 all run exactly 1,000 root visits (checked). So every "MCTS-200" in this file, including the PPO pilot's opponent and the psro `tiny` preset `mcts_it:200`, actually ran MCTS-1,000.
- **The heuristic leaf mis-scores finished games.** Through `pe.id(child, 0)`, a terminal child is scored by the winner's remaining material, not as a win. On greattusk vs gholdengo, P1-won terminals score −50 to +185 while live children span ±215, so a won position can rank below a live one (checked). In MCTS self-play on randbats, 18 of 151 two-sided decisions (12%, all late-game) have a terminal child; in 15 of them the leaf inverts a win or loss against a live child. One mis-scaled terminal cell also explains why the ∞-norm affine fit in `endgame_eps.py` often collapses to a = 0.
- **Candidate leaves on the exact endgames.** Mean calibrated ε_v and exploitability of equilibrium sampled (seat 1 / seat 2):

  | Endgame | Current leaf | Exact terminals + tanh(h/160) | MCTS-1,000 leaf | Existing value net |
  |---|---|---|---|---|
  | greattusk vs gholdengo, ε_v | 0.153 | 0.046 | 0.008 | 0.057 |
  | exploitability | 0.215 / 0.120 | 0.000 / 0.000 | 0 / 0 | 0 / 0 |
  | gholdengo vs heatran, ε_v | 0.164 | 0.068 | 0.035 | 0.078 |
  | exploitability | 0.074 / 0.425 | 0.074 / 0.020 | 0.005 / 0.008 | 0.074 / 0.020 |
  | garchomp vs rotomwash, ε_v | 0.475 | 0.397 | 0.331 | 0.548 |
  | exploitability | 0.338 / 0.602 | 0.306 / 0.601 | 0.431 / 0.816 | 0.193 / 0.873 |

  - Exact terminals plus a bounded leaf remove nearly all exploitability on two of three endgames, at no extra cost per move. On the long, balanced endgame they barely help: there the heuristic lacks information, not calibration.
  - An MCTS leaf is the best leaf on the easy endgames and the worst on the balanced one. MCTS values are also relative to the root (`mcts.rs` rollout = sigmoid(0.0125·(eval(leaf) − eval(root)))), so they can't replace the heuristic mid-game.
  - The maximum ε_v stays about 1 for every candidate leaf, so only the refined, along-the-game form of the bound can become informative.
  - Mid-game strength: 20 pairs vs foul-play at 75 ms gave the current leaf 0.375 ± 0.15 and the fixed leaf 0.312 ± 0.14. No measurable change, as expected, since the fix only acts on about 12% of decisions.
- **White-box search exploiter (`wb2`).** It knows the target's exact policy. For its top-K actions it expands chance branches, then best-responds to the target's exact policy one turn deeper. For a fixed target, any such exploiter gives a valid lower bound: win rate − 0.5.

  | Exploiter | Target | Pairs | Score | 95% CI | Lower bound |
  |---|---|---|---|---|---|
  | LBR (one turn), same seeds | Equilibrium, top move | 20 | 0.700 | [0.590, 0.810] | 0.20 |
  | wb2 | Equilibrium, top move | 20 | 0.825 | [0.718, 0.932] | 0.325 |
  | wb2 | Equilibrium, sampled | 10 | 0.850 | [0.700, 1.000] | 0.35 |
  | LBR + MCTS leaf | Equilibrium, top move | 20 | 0.475 | [0.363, 0.587] | −0.03 |

  - wb2 − LBR on the same 20 seeds: +0.125 [+0.010, +0.240]. Part of this is fewer 300-turn draws (6/40 vs LBR's 16/40).
  - Preliminary reading (10 pairs), refuted by W1 below.
  - The RL exploiter used only 1.2–2.2k games per target, against the pilot's 340k self-play games. For a known, deterministic target, white-box search is far cheaper than RL. RL remains the fallback for black-box or hidden-information targets.
- **Confirmation (W1).** `wb2` (now `wb2_policy` in `engine/nash.py`), 100 pairs per cell (200 games), on fresh seeds disjoint from all earlier runs. Both cells use the same pruning: top K=4 own actions, 80% target support, chance branches to 95% mass. (The preliminary 20-pair top-move run used K=5.) Source: `results/experiments/W1.jsonl`.

  | Target | Pairs | Score | 95% CI | Lower bound (score − 0.5) | Certified 95% LCB (Hoeffding, one exploiter) | W-L-D | 300-turn-cap draws |
  |---|---|---|---|---|---|---|---|
  | Equilibrium, top move | 100 | 0.873 | [0.836, 0.909] | 0.373 | 0.250 | 160-11-29 | 29 |
  | Equilibrium, sampled | 100 | 0.713 | [0.654, 0.771] | 0.213 | 0.090 | 136-51-13 | 13 |

  - Difference, top move − sampled: +0.160 [+0.091, +0.229]. The seeds are disjoint, so the cells are independent. Hoeffding radius sqrt(ln 20/200) = 0.122. Wall time 59 min on 5 local processes.
  - Caveat: the 80% support truncation only affects the sampled cell, since the argmax target has a single support action. So part of the gap may come from pruning. A K=5/95% rerun of the sampled cell would separate the two.
  - Reading: the preliminary small-sample reading (a depth-2 exploiter exploits equilibrium sampled about as much as top move, 0.850 at 10 pairs) is refuted. With 100 pairs on fresh seeds, top move is clearly more exploitable. Sampling does buy robustness against a depth-2 white-box exploiter too, not only against one-turn LBR.
  - Both targets now have certified positive exploitability lower bounds: ≥ 0.25 for top move and ≥ 0.09 for sampled, at 95% with one pre-registered exploiter, so no selection correction is needed. This is the bracket's lower side; the upper side stays vacuous.

### Safe exploitation and leaf rescaling (2026-11-15; `results/experiments/{SX,LF,SXW}.jsonl`, log `run_SX_LF_SXW.log`)
- **Safe-exploit bot (`safex:0.02`).** Each turn: v* = LP value of the full matrix A; predicted opponent column q = argmin of A's column means (best response to our uniform mix); play max x·A·q subject to (xA)_j ≥ v* − 0.02·range(A) for every column j (`nash.safex_strategy`; idea from Ganzfried & Sandholm 2015). Worst-case stage loss vs the equilibrium is at most 2% of the matrix range by construction.
- **SX: vs MCTS 2,000 iterations, 100 pairs, same seeds for all three bots (paired):**

  | Bot | Score | 95% CI | ms/move |
  |---|---|---|---|
  | Equilibrium, top move | 0.580 | [0.518, 0.642] | 71 |
  | Safe exploit, ε = 2% | 0.530 | [0.466, 0.594] | 63 |
  | Equilibrium, sampled | 0.510 | [0.449, 0.571] | 70 |

  Paired: safe exploit − top move −0.050 [−0.134, +0.034]; safe exploit − sampled +0.020 [−0.066, +0.106]. Both n.s. Sampled scores 0.510 here vs 0.44 in E1fix (different seeds and n).
- **SXW: white-box depth-2 exploiter (`wb2:safex:0.02:4:0.8`, knows the bot's algorithm) vs the safe-exploit bot, 100 pairs:** exploiter scores **0.767** [0.718, 0.817]; Hoeffding 95% lower bound 0.645, so exploitability ≥ 0.145 in score units. W1 on different seeds: 0.873 vs top move, 0.713 vs sampled (unpaired).
- **LF: leaf rescaling vs foul-play 75 ms, 100 pairs, paired seeds:** current leaf 0.355 [0.301, 0.409]; tanh(h/400) with exact ±1 terminals 0.315 [0.259, 0.371]; paired difference −0.040 [−0.098, +0.018], n.s.
- **Reading:**
  - The safe-exploit bot's per-turn gain does not show as a win-rate gain at 100 pairs. Its exploitability sits between top move and sampled: the 2% stage-level cap bounds loss only in the estimated matrix, and the exploiter wins through the leaf error, as predicted in `lit/next_directions.md`.
  - Rescaling the heuristic leaf toward the TD-residual minimiser does not help; the leaf bottleneck is not a scale problem.

## Next experiments

**What X1 implies.** The matrix bot looks two turns ahead and then scores positions with
poke-engine's hand-written evaluation. In the same 75 ms, MCTS searches deeper with that same
evaluation and wins. So the bottleneck is the leaf value, not the equilibrium solver. That points
to search plus a learned value function, the pattern behind the strong systems in the literature:
- PokaiTrainer reaches parity with decoupled MCTS only by adding a value net to CFR.
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
   the value net, inside the agent-form solve. This keeps the joint-solve advantage, closes part of
   the 0.1 oracle gap, and is the only way concealment can have value (direction E).
3. **Exploitability (E6), using RL as the measuring tool** (scaffolded: `psro.py --mode rl_br`, see above):
   train PPO best responses against frozen agents (top move vs sampled). The opponent is fixed, so this is much easier than learning a
   strong policy. It tests whether mixing buys robustness.
4. **Belief upgrades:** item and ability inference from damage and speed, and behaviour-based
   updates. Measure how much of the oracle gap closes.
5. Full-size runs for the remaining E4 matchups.

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
- We should present agent-form solving as an existing method that we evaluate, not as a new one.
- Other prior art we should cite rather than claim:
  - maximin pure play (PokéChamp; Sarantinos 2022);
  - "alpha-beta is unsound for simultaneous moves" (Sarantinos 2022; pruned search computes only
    bounds on cells [Saf12, Bos16]);
  - "the top move suffices against fixed opponents, mixing is for adaptive ones" [Yu26, Gan16];
  - evidence-based set filtering (Foul Play);
  - search on top of RL (PokaiTrainer, Metamon, Wang 2024).

**What appears new** (not found in the located literature):
1. **A head-to-head of agent-form vs PIMC blending at equal compute,** with an effect size of
   0.585 [0.540, 0.630]. Yu26 has no PIMC baseline. Strategy-fusion theory [Fra98, Lon10]
   predicts the direction. It also fits ISMCTS beating determinized MCTS in Pokémon [Iha18].
2. **A per-decision bound on blend loss, checked on real decisions.**
   - Lon10 explicitly gives no loss bound. Arj24 only shows that strategy fusion shrinks with
     lookahead. Li24e bounds risk under wrong type beliefs with L1 and diameter terms, the closest
     analogue.
   - The argument is elementary, and a hand check tightens our constant. Pick, for each world, a
     column within the pairwise Hausdorff distance d_kl of each column of world k, then use
     concavity of the minimum and agent-form value ≤ Σ b_k v_k. The result is blend loss ≤
     Σ_{k,l} b_k b_l d_kl ≤ max d_kl ≤ 2·δ_H, where δ_H in our code is the radius around a central
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
- **A. δ_H-gated agent-form.** Predict the 15% of decisions where blending changes the top move,
  from δ_H or per-world top-move disagreement. Calibrate on `E5_decisions.jsonl` and solve the
  joint LP only there. Related: Lon10's a priori predictors, Bro17's selective re-solving.
- **B. E6 exploitability and regret-gated exploitation.** Replicate Yu26's local-best-response
  protocol against top move, maximin, sampled equilibrium and agent-form. Then choose online
  between top move and the mix with a restricted-response or confidence test [Joh09, Li26j],
  tracing the win-vs-MCTS against win-vs-best-response front [Li24e].
- **C. Decompose the 0.1 oracle gap.** Compare agent-form with the true posterior, the current
  belief and a prior-only belief; add item and ability inference.
- **D. Warm-started and predictive RM+.** Reuse the previous turn's regrets to reach LP-level play
  in 10–20 iterations, so agent-form costs no more than PIMC [Bro16, Far23].
- **E. Belief-indexed leaves and depth 3.** Give concealment a nonzero value, which is impossible
  with depth-1 scalar leaves [Bro20, Sch21].
- **F. MMD toward a prior.** A smooth dial between the top move and the equilibrium, scored against
  both MCTS and a best response [Sok22].

## Reproduce

```bash
cd final/engine
.venv/bin/python test_matrix.py && .venv/bin/python test_hidden.py
.venv/bin/python experiments.py E1fix --workers 6
.venv/bin/python hidden_experiments.py E4 --only 0 --workers 6 --log-decisions   # agent-form vs PIMC, 200 pairs
.venv/bin/python hidden_experiments.py E4 --pairs 50 --workers 6 --log-decisions # other matchups, 50 pairs
.venv/bin/python hidden_experiments.py E4am --workers 6
.venv/bin/python figures/make_figures.py
```
