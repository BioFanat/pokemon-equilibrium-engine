# Bridge plan: from matrix theory to the real game (2026-11-15)

Inputs: `new_directions.md` (S, U, R3d, verified with fixes), `dynamics_directions.md` (PA1–PA4, P1, P2, B3, D),
the correlation probe (U does not explain the engine at 1e3–1e4 iterations), RESULTS.md (SX, LF, SXW, gray-box).

## 1. Levels and bridges

| Level | Object | Proved results that live here |
|---|---|---|
| M0 | one exact matrix | Lecture 7 (self-play → minimax), RM+ rates, U1/U2 (UCB1 node), D (CCE), B3 (bandit vs full feedback) |
| M1 | one estimated stage matrix | one-step lemma (ε_s + 2ε_v; Lean) |
| M2 | full-information stochastic game (poke-engine) | Theorem 7, L0–L6, L2 safe Hedge, R3d, P1 (policy vs external regret), P2 (switch time) |
| M3 | hidden information (sampled worlds) | Prop K, S1/S2 (rates), general case (η_NL + TV) |
| M4 | real Showdown | measured only |

Each bridge: the assumption it needs, its proof status, and a real-game MEASUREMENT that tests it.

| Bridge | Assumption | Status | Real-game test | Cost |
|---|---|---|---|---|
| B01 M0→M1 | matrix error ‖Q̂ − Q*‖∞ ≤ ε_v | lemma proved (Lean) | exact-endgame ε_v (already in RESULTS) is the only honest measurement; poke-engine MCTS values are root-relative, so they cannot serve as an absolute Q* reference. Mid-game ε_v is stated as UNMEASURED | 0 |
| B12 M1→M2 | opponent sees our mixture, not our sample (L6 adversary); finite horizon | Theorem 7 proved | stage gaps along the exploiter's path, computed in Q̂ (Q* is unavailable): for sampled play they are ≈ 0 (RM+ converges), so the test ISOLATES the residual 2ε_v·H̄ that explains the measured exploitability; it does not confirm the bound. Convert units first: exploitability in ±1 payoff units = 2 × (exploiter score − 0.5), i.e. 0.746 / 0.426 / 0.534 for top / sampled / safex | log-only rerun, ~1 CPU-h |
| B23 M2→M3 | A2 non-revealing opponent, A3 world-only values, A4 exact prior | Prop K + general case proved | on Showdown we control both bots, so the TRUE opponent sets are known: measure TV(foul-play belief, truth) per turn and the oracle-vs-agent-form gap in real games | from harness logs |
| B34 M3→M4 | **new A5: simulator fidelity** — poke-engine's transition kernel P_pe ≈ Showdown's P_sd | not stated yet. It is NOT a leaf error: a kernel mismatch enters as Σ_t sup_{s,a,b} ‖P_pe(·|s,a,b) − P_sd(·|s,a,b)‖_TV × (range of V*) (simulation lemma), i.e. its own term ε_sim with its own bound, compounded over H̄ | from recorded Showdown battles: rebuild each turn's state with true teams, predict the outcome distribution of the played move pair with poke-engine, score the observed Showdown outcome (calibration, HP-damage error) | from replays, ~0.5 CPU-h |

What is already known about transfer (extensibility check so far):
- U (UCB1 correlation) holds at M0 but does NOT transfer to the engine at 1e3–1e4 iterations (probe: gap is unequalised exploration, joint ≈ product). Keep as an M0 remark.
- The ε-safe certificate holds at M1 but does not survive to M2 at ε = 2% (SXW: exploiter 0.767, worse than sampled 0.713): exploitation goes through ε_v. Certificate non-vacuous only for ε ≤ ~1% (dynamics_directions A2).
- Leaf rescaling (lower TD residual) did not move win rate (LF −0.040, n.s.): ε_v is not a scale problem.

## 2. Theory experiments on the real game (Showdown), ranked

| # | Algorithm (drop-in for the adapter's move choice) | Theory it tests | Prediction / refutation | Games |
|---|---|---|---|---|
| E1 | Agent-form vs PIMC (our central claim) under Showdown's own hidden information. Both arms use foul-play's set sampler as the belief, not our hidden.py tracker: a cleaner test of the formulation, a less direct replication of the in-engine 0.585 | M3→M4 transfer of the headline result | agent-form > PIMC; refuted if paired Δ ≤ 0 at n = 300 | 600 |
| E2 | PA1 HedgeBot: carried Hedge over {top, sampled, safest, safex 0.3%}, per-turn full-info feedback (opponent's realised move) | L6 regret bound pathwise (proved); P1 external vs policy regret | pathwise bound holds on every log; W_H within [min_e W_e − 0.05, max_e W_e + 0.05] | 300 + 150/expert |
| E3 | P2 switch test: anytime vs fixed-share Hedge, opponent switches after T₁ games | P2 (proved, k = 2; upper bound k = 4) | anytime crosses in ≈ T₁d₁/d₂ games, fixed share in ≈ 4–5 | 160 |
| E4 | PA3 BanditRootBot: per-turn bandit self-play (UCB1 / Exp3, N = 2000, MCTS's feedback model) vs full-info RM+ on the same matrices | L4 identity + B3 (Lecture 7 with bandit vs full feedback) | bandit root loses ≥ 0.05 more against an exploiter; refuted if |Δ| ≤ 0.03 at n = 300 | 800 |
| E5 | PA4β asymmetric self-play: HedgeBot vs equilibrium-sampled | the testable Lecture-7 minimax content: a no-regret learner can't beat the equilibrium expert by more than leaf-level exploitability | W_A ∈ [0.45, 0.55]; > 0.56 measures a larger ε_v | 400 |

Not run (cite only): meta-game Exp3-vs-Exp3 convergence (meta-game too flat to observe, B1), OMWU last iterate, CCE.

**Priority (what survives a time crunch):** (1) consolidation with the verifier/probe fixes → (2) log-only bridges B12, B34 → (3) E1 on real Showdown. E2–E5 are second tier. P1, P2, B3 and Lemma D are from one brainstormer and are UNVERIFIED: they go to the verifier before any of their predictions enter the paper (P2 is proved for k = 2 only; PA1 uses k = 4).

## 3. Infrastructure

1. **Headless real-Showdown harness.** `pokemon-showdown simulate-battle` (stdin/stdout, seeded `>start`), adapter
   that feeds each player's stream into foul-play's battle tracker (no server, no websocket), our move-choice
   functions plugged in. Paired seeds: same teams and PRNG, sides swapped. Every game writes a replay (.html) and a
   decision log (matrix, mixture, true opponent sets for B23).
   **Gate:** one paired game end to end locally (same seed, sides swapped, both replays render, decisions legal)
   before any Modal image. **Fallback** if the adapter is not working after a bounded effort: server-in-container
   (the route that already works), unpaired games, larger n.
2. **Local calibration**: 20 games → s/game, CPU, correctness (replays render; decisions legal).
3. **Modal**: one image (Node 22 + Showdown at the pinned commit + bot venv with pinned deps + engine code), a
   `.map` over game seeds, budget guard as in `modal_psro.py`, run ATTACHED. Estimate before launch (psro_cost under-predicted 2× last time); ask before spending.

## 4. Consolidation (after measurements)

`theory/theory_map.md`: every assumption numbered once (A1–A5 plus experts/adversary), every result tagged with its
level (M0–M4), the bridge it needs, proof status (Lean / proved / sketch / conjecture / numeric) and its real-game
test. Apply the verifier's fixes (S1 boundary condition γ ≤ min_θ b_θ; S2 constant 0.0231 → 0.0255 with c*;
U1 typo + U1.2 for both arms; R3d off-by-one + C ≤ C_∞ + d/2, "reproduce in-sample" not "predict"; U4 wording)
and the probe's narrowing of U. Then rebuild the paper's theory section and appendix on this structure.
