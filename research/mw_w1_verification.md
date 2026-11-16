# Independent verification of the W1 result (hedge_bayes_mw vs hedge_bayes, seeds 1-200), 2026-11-16

Scripts and outputs: `research/verify/mw/` (copied from the verifier's scratch; run with ~/sandbox/bot-venv/bin/python).

- **W1 recomputes exactly:** +0.0625 [0.0032, 0.1218] (iid over 200 seed pairs; 0 error games; pairing by seed
  confirmed: identical teams, battle seeds, stream positions in both arms).
- **Not robust to stream clustering:** the hedge learner carries state within a stream. t(9) stream-clustered CI
  [-0.011, 0.136]; exact stream sign-flip p = 0.10; stream-bootstrap lower bound 0.000. Dropping the single most
  favourable seed puts 0 inside the iid CI.
- **Pairing gives almost no variance reduction:** our bots' RNG and foul-play's world sampling are unseeded, so the
  arms never share sampled worlds; per-seed correlation 0.02-0.08.
- **The control looks low:** the earlier Modal run of the same method (tag BH) on the same seeds 1-200 scored 0.3475;
  BHs scored 0.2825 (BHs - BH = -0.065 [-0.123, -0.007]); MWs - BH = -0.0025. The W1 gap is the size of the gap between
  two runs of the same method. Load was equal between MWs and BHs (stock MCTS iterations per decision differ by
  -0.65 % [-4.3 %, +3.0 %]); our bot used about 2x the CPU per decision locally vs on Modal (load-dependent leaf).
- **The method does change behaviour:** final MAP style safe in 9/10 MWs streams vs br_unif in 8/10 BHs streams;
  mean safex Hedge weight 0.268 vs 0.179; safex goes from the worst to the second-best expert by cumulative reward.
- **Post hoc:** the gain sits in early stream positions (seeds 1-100: +0.105; 101-200: +0.020; difference n.s.).
- **Code sanity:** 0 errors, 0 invalid choices; the per-world safex LP is always feasible and bounded (argument plus
  3000 synthetic cases, 0 failures), reduces to nash.safex_strategy at K = 1.

Reading: W1 is not evidence of a win-rate gain. The pre-registered replication (prereg_mw.md Addendum A, seeds 201-400)
is the test. Note for it: the stream-clustered interval is the honest one for any learner arm.
