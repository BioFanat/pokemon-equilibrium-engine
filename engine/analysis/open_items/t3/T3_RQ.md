# T3 on the real RQ runs

Scripts: `t3_rq.py` (BC, LHS, log-likelihoods; writes `t3_out.json`), `t3_bounds.py` (bounds from `t3_out.json`), `t3_check.py` (q-vs-p_style consistency).
Data: `engine/showdown/results/{RQeq,RQsafe,RQunif}/replays_headless/<tag>-s<seed>-g<1|2>.A.decisions.jsonl`; streams = seeds with (s-1)%10, games in order (seed, g1, g2), as `verify/v_b6.py`.
Per-style predictive over ALL opponent columns IS logged: decision record `bayes.q[style]` (32-label vector, 4 dp, `bayes.labels`), paired with the feedback of the next record (`feedback.for_turn == turn`, `bayes.j`, `bayes.p_style`). q is built on our world-averaged Abar (behavior.style_dists). Non-skipped feedback turns only (the turns that update the posterior). Pairing check: log q[s][j] vs logged p_style differs by <=0.037 nats (4-dp rounding) on all 14,513 turns.
True style s*: eq / safe / uniform. BC capped at 1; forced turns (n_cols=1) have BC=1. Averages are over the 10 streams (mean of per-stream values).

## 1. LHS = sum_t (1 - P_{t-1}(s*)), P_0 = 0.2, lambda=1 (rebuilt from p_style)
Reproduces known values: RQeq 589.02, RQsafe 78.48, RQunif 16.24 (stream T = 479-650 / 409-608 / 292-387 updates).
Per stream: eq [589.8 623.8 519.3 607.8 478.3 598.8 574.1 648.8 649.5 600.1]; safe [118.2 22.7 87.2 59.5 6.4 267.5 110.1 11.7 92.4 9.2]; unif [3.2 7.1 29.3 14.2 14.2 8.5 41.2 23.1 4.0 17.7].

## 2-3. BC statistics and bounds (stream-averaged)
mean BC / sup BC over ALL update turns (forced included) / fraction of turns with BC>0.99; "nf" = non-forced turns only (the form T3(v) says to use).

| tag | other s | mean BC | sup BC | frac BC>.99 | mean BC nf | sup BC nf | frac>.99 nf |
|---|---|---|---|---|---|---|---|
| RQeq (s*=eq) | br_unif | 0.754 | 1.000 | 0.125 | 0.722 | 0.998 | 0.010 |
| | br_us | 0.777 | 1.000 | 0.124 | 0.748 | 0.997 | 0.008 |
| | safe | 0.791 | 1.000 | 0.129 | 0.764 | 0.9995 | 0.014 |
| | uniform | 0.611 | 1.000 | 0.117 | 0.560 | 0.919 | 0.001 |
| RQsafe (s*=safe) | eq | 0.799 | 1.000 | 0.132 | 0.773 | 0.999 | 0.019 |
| | br_unif | 0.953 | 1.000 | 0.209 | 0.946 | 1.000 | 0.107 |
| | br_us | 0.953 | 1.000 | 0.329 | 0.947 | 1.000 | 0.242 |
| | uniform | 0.868 | 1.000 | 0.128 | 0.851 | 0.999 | 0.015 |
| RQunif (s*=uniform) | eq | 0.638 | 1.000 | 0.104 | 0.596 | 0.936 | 0.000 |
| | br_unif | 0.919 | 1.000 | 0.126 | 0.909 | 0.997 | 0.024 |
| | br_us | 0.924 | 1.000 | 0.162 | 0.915 | 1.000 | 0.065 |
| | safe | 0.897 | 1.000 | 0.140 | 0.885 | 1.000 | 0.040 |

| tag | LHS (mean) | iid bound sum 1/(1-meanBC), all turns | iid bound, nf turns | sup bound (all turns) | sup bound (nf) | realised-product bound sum_t sum_s min(1,prod BC) | holds? (iid-nf / realprod, streams of 10) |
|---|---|---|---|---|---|---|---|
| RQeq | 589.0 | 15.9 | 14.1 | inf | 4.3e6 (vacuous) | 8.7 | NO (0/10, 0/10) |
| RQsafe | 78.5 | 55.3 | 48.9 | inf | inf | 50.7 | NO on average (holds in 5/10 streams vs the all-turn bound 55.3, 4/10 vs the non-forced 48.9; recounted in verify/t3_check.py); stream LHS 6-268 |
| RQunif | 16.2 | 38.0 | 34.0 | inf | inf | 34.0 | YES (9/10, 9/10); mean 16.2 <= 34.0 |

(iid-all per stream: eq 15.4-16.4, safe 49.7-61.3, unif 35.1-40.6; sup-all is inf because forced turns have BC=1.) Sup bound is vacuous everywhere (inf) except RQeq-nf where it is finite but astronomically large (one style's sup 0.9995 -> ~2000/stream, the average is dominated by outlier streams), so it "holds" trivially.

## 4. Realisability: mean log loss -log q_s(b_actual) (nats/update, all turns; nf in parentheses), stream-avg
| tag | eq | br_unif | br_us | safe | uniform | best style (cum. log-lik, all 10 streams) |
|---|---|---|---|---|---|---|
| RQeq | 2.202 (2.493) | 1.684 (1.907) | 1.695 (1.919) | **1.616 (1.829)** | 2.230 (2.525) | safe, 10/10 |
| RQsafe | 3.033 (3.425) | 1.863 (2.103) | 1.931 (2.181) | **1.815 (2.049)** | 2.158 (2.438) | safe, 10/10 (true) |
| RQunif | 3.826 (4.271) | 2.346 (2.619) | 2.365 (2.641) | 2.448 (2.732) | **2.053 (2.291)** | uniform, 10/10 (true) |

RQeq per-turn mean log-likelihood ratio log q_eq(b)/q_safe(b): -0.586 (all turns), -0.664 (nf), i.e. q_safe out-predicts q_eq by 0.59 nats/update; q_eq is also behind br_unif (-0.518) and br_us (-0.507); only beats uniform (+0.028). q_safe beats q_eq on 56% of nf turns. Model-expected gap if realisable would be KL(q_eq||q_safe) nf = +0.812 (positive, favouring eq); realised is -0.664: sign flipped. Mean nf entropy of q_eq = 0.714 nats vs realised cross-entropy 2.493 (underdispersed model: eq opponent's real actions are far from what our q_eq predicts).
RQsafe: true safe is best, margins vs eq +1.377, uniform +0.389, br_us +0.132, br_unif +0.055 nats/nf turn (br_unif/br_us nearly tie: KL(q_safe||q_brunif) 0.208, frac BC>0.99 up to 0.24 on nf turns). Realised cross-entropy 2.049 vs mean entropy of q_safe 1.846 (model slightly overconfident, 0.2 nats).
RQunif: true uniform is best, margins vs eq +1.98, safe +0.441, br_us +0.350, br_unif +0.328 nats/nf turn. Realised 2.291 = mean entropy 2.291 (exactly calibrated: opponent really samples uniform over the columns).

## 5. Verdicts
- RQeq: T3 bound FAILS (LHS 589 vs 14-16 iid bound, 8.7 realised-product, sup bound vacuous). Cause: realisability fails. In our model class safe (and br_unif, br_us) predicts the eq opponent's actions better than q_eq by ~0.59 nats/update (10/10 streams), so the posterior correctly goes to safe (P_T(eq)=0) and 1-P(eq) stays ~1 for the whole run. The T3 martingale premise E[sqrt(LR)] = prod BC would give decay at rate meanBC~0.75-0.79, but the realised LR eq-vs-safe drifts the wrong way. This is the misspecified case (vi), s° = safe, not (i)-(v).
- RQsafe: bound FAILS on average (LHS 78.5 vs 48.9 iid-nf / 50.7 realised product; sup inf); holds in 4/10 streams (those with LHS<~50; stream 6 has 268, ~5x the bound). Realisability approximately holds (safe is the best predictor in 10/10 streams) but the model is overconfident and safe/br_unif/br_us nearly tie (margin 0.055-0.13 nats/turn, BC mean 0.95 -> per-style term 1/(1-0.95)=~20 each), so the bound is large and the heavy-tailed per-stream LHS exceeds it; it is a finite-sample/misspecification-by-calibration gap (expectation bound over 10 streams with heavy tail), not a wrong-best-style failure. Mean-over-streams exceeds the bound by 1.6x (48.9 vs 78.5); 10-stream noise is large (stream SD ~ 78).
- RQunif: bound HOLDS (mean LHS 16.2 <= 34.0 iid-nf, 34.0 realised product, 9/10 streams); uniform is the best predictor and exactly calibrated (realised cross-entropy = model entropy). Bound is 2.1x loose. Sup bound vacuous (inf, forced turns and BC~1 turns).

Caveats: q logged to 4 dp (renormalised; <=0.037 nat discrepancy vs logged p_style); BC uses the posterior-update turns only (skipped turns, 134 per stream in RQeq state, are excluded since they do not move the posterior); stream-concatenated across games with the carried posterior (as in the verified LHS). Expectation bound vs 10 realised streams: 'holds' is per-stream count and stream-mean, no CI computed.
