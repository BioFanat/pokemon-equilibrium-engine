# Independent verification of cce_two_sided.md (2026-11-16)

Verifier scripts: session scratch verify_cce/scripts (vload, vcompute, vagg, vhedge, vopp, vcal, vcorr_ceq, vcss, vdist).

## Verification of research/cce/cce_two_sided.md

I recomputed every measured number from the raw logs with my own loader and math. I did not reuse two_sided.py or opp_side_check.py. The two differences in method: stage values come from the column player's LP rather than the row LP, and SEs come from a game bootstrap as well as the cluster formula. All headline numbers reproduce to the last printed digit. The problems are in what the numbers are said to mean: the "floor", "indistinguishable", the Cesa-Bianchi–Lugosi (CBL) bound, Prop 3's "conclusion holds", and "CCE" used without qualification.

Scripts are in `/private/tmp/claude-501/-Users-biofanat-Documents-classes-f26-cs535/2bbe670b-2f97-4d9a-bfad-e1139e0352c6/scratchpad/verify_cce/scripts/`:
- `vload.py`: loader
- `vcompute.py`: per-decision quantities, column-player LP, bootstrap SE
- `vagg.py`: aggregates
- `vhedge.py`: Hedge normalisation, weight replay, per-stream bound
- `vopp.py`: CEQ/CSS decomposition and oracle gap
- `vcal.py`: calibration, posterior, permutation nulls
- `vcorr_ceq.py`: per-turn-product test
- `vcss.py`: ε-safe band
- `vdist.py`: Foul Play vs a literal eq player

Per-decision outputs are in `.../verify_cce/out/*.pkl`. The CEQ/CSS archive is extracted to `.../verify_cce/data/`, and the archive itself is untouched.

### Highest-risk issues first
1. **"Indistinguishable from an equilibrium player": refuted.** The two opponents match on only one statistic. Through the same model, other statistics separate Foul Play (LOG) from the literal eq player (CEQ) (`vdist.py`):
   - the posterior: P(br_unif) is 0.54 vs 0.09 (stream-clustered z ≈ 3.5; game-clustered z = 16);
   - br_unif log loss: 1.505 vs 1.647 (z = −5.0);
   - q̂ log loss: 1.492 vs 1.580 (z = −3.1);
   - ε_C^dyn: 0.0408 vs 0.0369 (z = 2.5).

   ε_C(eq) is the one that matches: 0.0213 vs 0.0211, difference +0.0003 ± 0.0013 (z = 0.2). Even that match only bounds the difference to about ±12% of the value, so "exactly at" is too strong.
2. **"Floor" / "resolution" is the wrong word.** CSS shows ε-safe opponents at ε_C(eq) = 0.0106 and 0.0152, below 0.021. The report admits this in §2.3. So 0.021 is a reference value for one opponent type, not a lower bound. The attribution to "information asymmetry" is also an interpretation: each side's own sampling of K = 4 worlds is an alternative explanation that has not been ruled out.
3. **The CBL Thm 2.3 bound is applied with the wrong T.** Hedge is carried separately per stream, with T_s = 1600–1940.
   - LOG: the correct per-turn bound is 0.038–0.042 per stream, 0.0403 pooled. The report uses 0.0127. Measured regret is 11% of the bound, not 35%.
   - CEQ: the correct bound is 0.075, not 0.0235.
   - Consequence: Prop 4's "guaranteed ε_R" is about 0.04. That is larger than the measured dynamic regret (0.030), so the guaranteed-CCE statement is weak.
   - The guarantee also holds on the learner's own reward (world-k matrix M_k, clipped), not on the Ā-based ε_R^E. 10.2% of turns have ptp(Ā) > 400, and 1.8% of turns have a clipped reward entry.
   - Minor: the code's step size is η = √(8 ln4 / t) applied before decision t+1, one step ahead of CBL's η_{t+1}. This changes only the constant.
4. **Prop 3, "the conclusion holds numerically": no.** The conclusion is O(1) cumulative regret, i.e. a per-turn gap that goes to 0. On CEQ the per-turn gap goes from 0.00137 (first half of each stream) to 0.00105 (second half). The cumulative gap is 4.18 Δ̄ over 3,501 turns and grows roughly linearly. What the data supports is a small per-turn gap of about the width of the ε-slab.
   - The hypothesis is better refuted directly: min over styles of TV(y^B, q_s) = 0.36, and the posterior puts only 0.008 on eq against a literal eq opponent. ‖q̂ − y^B‖₁ = 1.21 is weaker evidence.
   - (B), "ρ_s < 1 uniformly", is literally false on this data. On 2,553 LOG turns (580 CEQ) m_t = 1 and all five styles are identical. It needs "for turns with m_t ≥ 2".
5. **I1 and I2 are tautologies.** "The two-sided Lecture 7 identity closes" must not be presented as a finding. My recomputation gives an I1 residual of exactly 0 and I2 of 2e-17. The bracket "equalities" hold by definition.
6. **Provenance error.** For "stream 9 is resumed" the report cites synthesis.md §3.1(a′), but that passage is about E2 stream 9. For LOG, the Hedge state shows 1940 updates vs 1928 logged, and the weight replay breaks at seed 100. Stream 3 also mismatches (1811 vs 1776), but that is because the excluded error game LOG-s4-g2 is in stream 3 and has 34 usable updates. Including it leaves ε_R^E at 0.00444.

### Per-claim verdicts
**1. Fields: CONFIRMED (with a scope note).** Every decision where we have at least 2 legal options logs:
- `abar` (2 dp), `labels`, `legal`, `pfull` (4 dp), `Q` (4×n), `qs` (5×m), `bayes.qhat`.

The realised column is in the next record's `feedback.bayes.j`. Code: `final-integration/engine/showdown/run_ourbot.py:541–551` and `382–420`.
- Counts: 20,657 decisions with feedback, 3,386 skipped, **17,271 usable**, 599 games (LOG-s4-g2 excluded for a ZeroDivisionError). This matches `LOG_summary.json` ("updates": 17271).
- Decisions where we have a single legal option log no Ā; "every decision" should say that.
- synthesis.md §3.1 is stale. It was committed 2026-11-15 23:20. The LOG decision files date from 00:03 and the summary from 00:23 on 11-16, and the run's wall time was 3,012 s, so it started around 23:33. "No run logs Ā_t" is now false. The scoped sentence "not computable from E2/E5/BH" is still true.

**2. Definitions / CCE reading: PARTIALLY TRUE.**
- Lecture 7 §3.1 defines a CCE for one fixed game, with deviations to a fixed pure strategy s′_i. What the report computes is a restricted-deviation (Φ-type) coarse correlated gap of σ_T over the realised sequence of *our estimated* stage games. The deviation classes are tiny (|E| = 4, |S| = 5). Deviations are counterfactual per stage and ignore their effect on later states. The opponent's deviations are scored in our matrices, not in Foul Play's objective.
- The experts and styles are not "fixed policies". The safex expert depends on q̂ (the posterior history), and br_us depends on our Hedge weights. They are known before the turn, so the regret is valid, but they are history-dependent advice.
- The row player's dynamic comparator max_i Ā[i, j_t] sees the opponent's *realised* action. That is a clairvoyant deviation, not a coarse one, and it is inflated by the opponent's randomisation. The column player's comparator min_j p_tᵀĀe_j sees only our mixture. So 0.030 vs 0.041 is not a symmetric pair.
- "Exact" is true only of the arithmetic.

**3. LOG numbers: CONFIRMED.**

| Quantity | Recomputed |
|---|---|
| ε_R^E (best expert: top in 10/10 streams; per stream 0.0039–0.0050) | 0.00444, boot SE 0.00020, cluster SE 0.00020 |
| ε_C^S (eq) | 0.02133 ± 0.00062 |
| ε_C(eq) positive by stream | 10/10 (0.0184–0.0227) |
| Opponent regret vs br_unif / br_us / safe / uniform | −0.0257 / −0.0120 / −0.0150 / −0.1012 |
| ε_R^dyn | 0.0301 |
| ε_C^dyn | 0.0408 |
| Our exploitability | 0.01413 |
| I1 residual | 0 |
| I2 residual | 2e-17 |
| Hedge regret from logged r (CEQ: 0.00231) | 0.00448 |

- Normalisation matches `exp3_live.py:63–64` exactly, r = clip((Q·M_k[:, j] − min M_k)/400, 0, 1). Replaying anytime-FTRL weights from the logged r reproduces the logged weights to 1e-4 in 8/10 streams; streams 3 and 9 deviate for the reasons in issue 6.
- The Ā-based ε_R^E drops the min shift, which cancels in regret, and drops the clipping. It agrees with the logged-r figure, 0.00444 vs 0.00448.
- Sign convention is correct: we maximise; the opponent's payoff is −p_tᵀĀ_t·(column); ε_C = ḡ − min_s c̄_s.

**4. CEQ decomposition: numbers CONFIRMED, interpretation REFUTED.**

| Quantity | Recomputed |
|---|---|
| Paired turns | 3,501 (report 3,507; 6 have zero overlapping mass) |
| ε_C(eq) on paired turns | 0.0234 ± 0.0014 |
| Sampling term | −0.0012 ± 0.0007 |
| Mismatch term | +0.0246 ± 0.0013 |
| TV(y^B, q_eq) (median 0.40) | 0.441 |
| val_A + val_B | −0.0377, i.e. −15.1 engine units (median absolute value 12.1) |
| ε_C^S over all CEQ turns | 0.0211 ± 0.0011 |

style_eq uses the same 5% uniform mix, so the 0.44 TV is not a smoothing artefact. "Indistinguishable" holds only in the sense "its regret against the eq style equals a literal eq player's"; see issue 1.

**5. CSS: PARTIALLY TRUE.** Mean stage exploitability is 0.00134 (side A) and 0.00133 (side B). But 53–54% of turns exceed 0.0015, with a maximum of 0.0048 / 0.0076. The per-turn guarantee is ε·ptp(Ā_t), which holds up to ≤0.06 engine units of rounding. The report's "up to 1.5 engine units" overstates the excess. Correct wording: "run-mean exploitability 0.0013 in its own model."

**6. Log loss vs payoff ranking: CONFIRMED, with caveats.** By log loss, safe 1.4994 < br_unif 1.5053 < br_us 1.700 < eq 2.242 < uniform 2.255. Payoff ranks eq first. But safe vs br_unif is within noise. The payoff result is close to structural: q_eq = 0.95·maximin + 0.05·uniform, so c_eq ≲ v_t. The two rankings score different objects (a predictor vs a policy), so this is not a paradox.

**7. Posterior-learner statements:**
- Prop 4: the citation is right. CBL 2006 Thm 2.3 is the time-varying-η exponentially weighted forecaster. The bound is misapplied (issue 3), and the guarantee does not transfer to the Ā-based CCE.
- Prop 3: the proof steps check out (three-term split; Lipschitz Δ_t/2·‖·‖₁; ‖q̂ − q*‖₁ ≤ 2(1 − P(s*))). The hypothesis (B) problem and "conclusion holds" are covered in issue 4.
- Calibration: PARTIALLY TRUE. Bins reproduce (0.24 → 0.32, 0.38 → 0.45), and ECE is 0.0147. But predictions are *over*-confident at 0.79 → 0.66 and 0.996 → 0.98. "Top-1 accuracy 0.44" is wrong: 0.443 is the mean predicted top-1 probability, and the hit rate is 0.521. Excluding m_t = 1 turns, the figures are 0.347 vs 0.438. So it is under-confident only for favourites predicted below 0.5.
- Kalai–Lehrer 1993 (Econometrica 61), Foster–Vohra 1997 (GEB 21), Fudenberg–Levine 1995 (JEDC 19), Nachbar 1997, Foster–Young 2001: citations correct. Say "its hypotheses do not hold" (no grain of truth, a changing stage game, ε-safe rather than Bayes-optimal play), not "fails".

**8. Correlation: numbers CONFIRMED; the null is uninformative.**
- TV: 0.0541 (LOG) / 0.0711 (CEQ). Permutation null mean: 0.0044 / 0.0080. 95th percentile: 0.0076 / 0.0141. p = 1/1001. A within-game permutation null gives 0.006 / 0.012.
- Any time-average of varying per-turn products is non-product, so this null rejects a hypothesis nobody holds. On CEQ, where the opponent is stateless and does not learn, the test still rejects. σ_T sits at TV 0.0094 from the logged per-turn product, against a sampling null of 0.0091. This is tautological for a stateless opponent, but it shows the test cannot detect learning-induced correlation.

### Allowed wording
**5-page paper (one sentence):** "Along 17,271 logged decisions against Foul Play, Hedge's external regret against its best fixed expert is 0.0044 ± 0.0002 per turn, and in our estimated stage games the opponent's play falls 0.021 ± 0.001 short of our LP-equilibrium style. A literal equilibrium opponent acting on its own information shows the same 0.021 gap, so this number reflects mismatch between our model and the opponent's, not Foul Play's regret (units: Hedge's 400-point reward scale)."

**Long report (paragraph):** "The LOG run (hedge_bayes vs Foul Play, 599 games, 17,271 usable decisions) logs Ā_t, our mixture, the four experts, the five opponent styles and the realised opponent column. This makes both sides' regrets computable in our estimated stage games. Against fixed comparators, our regret vs the best expert is 0.0044 ± 0.0002 per turn. The opponent's regret vs our LP-equilibrium style is 0.0213 ± 0.0006 (positive in 10/10 streams) and is negative against the other four styles. Against per-turn best responses the regrets are 0.030 (ours, against the realised column) and 0.041 (theirs, against our mixture). Our mean stage exploitability is 0.014. Equivalently, σ_T is a coarse correlated equilibrium of the realised sequence of estimated stage games only with respect to these small comparator classes. Hedge's worst-case guarantee at these per-stream horizons is about 0.04 per turn. A control in which the opponent literally plays its own maximin (CEQ) shows the same 0.021 ± 0.001. Almost all of it is mismatch: the true mixture's cost in our model exceeds our q_eq's by +0.025 ± 0.001, sampling contributes −0.001 ± 0.001, TV(y, q_eq) = 0.44, and the two sides' stage values differ by about 15 engine units. So ε_C measured through our model does not identify the opponent's regret. Other statistics through the same model (posterior style weights, log loss, dynamic regret) do separate Foul Play from the literal equilibrium player. ε-safe opponents (CSS) show a lower gap of 0.011–0.015."

### Must NOT be claimed
- "Foul Play is indistinguishable from an equilibrium player through our model," or that it "sits exactly at" a floor.
- "Floor", "resolution" or "information-asymmetry floor" for 0.02.
- "Measured regret is 35% of the bound," or the bound 0.0127 / 0.0235.
- "σ_T is a CCE" without naming the comparator classes and the estimated games.
- ε_R^E as "guaranteed" on Ā.
- "The Lecture 7 identity closes" as evidence.
- Prop 3's "conclusion holds numerically" / O(1) regret observed.
- "Under-confident on favourites" without the 0.7–1.0 over-confidence; "top-1 accuracy 0.44".
- "Each ε-safe side is ≤0.0015-exploitable" per turn (say run-mean 0.0013).
- The meta-level correlation as evidence of anything learning-related.
- ε_R^dyn vs ε_C^dyn as a symmetric pair.
- Foul Play's regret in its own objective.
- Stream 9's resumed status cited to synthesis.md §3.1(a′).
