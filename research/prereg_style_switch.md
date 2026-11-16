# Pre-registration: style-posterior switching test (live test of T5)

Written: 2026-11-16 11:41 EST. Status: pre-registered; no switch run with a Bayes posterior has been launched. Commit this file
(as `research/prereg_style_switch.md`) before launching any arm; the predictions in §4 are not edited afterwards
(results go in an appended "Results" section).

Target: `theory/tightening/bayes_dynamics.md` §6 (Proposition T5, tracking after a style switch), calibrated on
the RQ runs (`engine/analysis/open_items/out_B6_rq.json`; §4 of the theory file, scored in
`engine/analysis/open_items/REPORT.md` B6). E3a/E3f (`engine/RESULTS.md` "Open items (2026-11)" B2) switched the
opponent against plain Hedge, which carries no posterior, so T5 has had no live test.

## 1. Runs

Bot: `hedge_bayes` (role A; style posterior of `engine/showdown/behavior.py` plus Hedge weights, both carried per
stream in `$OURBOT_WORK/hedge_state/<tag>-stream<k>-A{.bayes,}.json`). Opponent (role B): a `batch.py` switch spec
`switch:<m1>@<T1>:<m2>` (T1 in seeds = stream positions; a seed is two paired games with seats swapped), so the switch
is after game 2*T1 of the stream. Streams: `--streams 10` (stream k plays seeds k+1, k+11, k+21, ... in order), the
same layout as the RQ runs (seeds 1-100, 10 streams), so each seed of an arm has the same teams and battle PRNG as the
same seed in RQsafe / RQunif. Trajectories are not replays (`run_ourbot.py:45` seeds its move RNG from the clock), so
the controls are paired on seed (teams + battle seed), as E1/E3 were.

Switch pair: **style_safe -> style_uniform**. Both styles were identified by the shipped lambda = 1 posterior in RQ
(final MAP safe 10/10, uniform 10/10). Drifts measured on RQ (mean over the 10 streams of
$\mu_{x\to s}=\overline{\ell_{s}-\ell_{x}}$ per update, nats; `mean_loss_style` in out_B6_rq.json):

| truth x | eq | br_unif | br_us | safe | uniform |
|---|---|---|---|---|---|
| safe (RQsafe) | 1.218 | 0.048 | 0.117 | 0 | 0.344 |
| uniform (RQunif) | 1.773 | 0.293 | 0.312 | 0.395 | 0 |

T5(a) competitor terms $t_0(\mu_{a\to b}-\mu_{a\to s})^+/\mu_{b\to s}$ for a = safe, b = uniform: safe 0.87,
br_unif 1.01, br_us 0.73, eq 0 (in units of $t_0$), so $\tau^*\approx1.0\,t_0$ (over all 100 RQsafe x RQunif
stream pairs: median 1.00, 10-90% 0.77-1.36). The reverse direction (uniform -> safe) is dominated by a 2.1 $t_0$
br_unif term that rests on the 0.048 drift (0.006-0.09 per stream) and is not run.

Arms (local; `~/sandbox/bot-venv/bin/python <worktree>/engine/showdown/batch.py --k 4 --search-time-ms 100 --streams 10`
with `OURBOT_WORK=$HOME/sandbox/ourbot/sw`; the copies in `~/sandbox/ourbot/*.py` are stale and must not be used):

| tag | --a | --b | --seeds | switch after | phase 1 (t0, updates) | post-switch | env |
|---|---|---|---|---|---|---|---|
| SW2 | hedge_bayes | switch:style_safe@2:style_uniform | 1-100 | game 4 of each stream | 2 seeds, ~112 (range ~70-145) | 8 seeds, ~290 uniform updates | default (lambda = 1) |
| SW4 | hedge_bayes | switch:style_safe@4:style_uniform | 1-180 | game 8 | 4 seeds, ~220 | 14 seeds, ~500 | default |
| SW4p | hedge_bayes | switch:style_safe@4:style_uniform | 1-80 | game 8 | 4 seeds, ~220 | 4 seeds, ~145 | OURBOT_BAYES_LAMBDA=0.98 (shipped predict-then-discount forgetting) |
| SW4d (optional) | hedge_bayes | switch:style_safe@4:style_uniform | 1-80 | game 8 | as SW4p | as SW4p | OURBOT_BAYES_LAMBDA=0.98 OURBOT_BAYES_DTP=1 (discount-then-predict; needs the §5 code change) |

SW4p tests T5(c)'s flip time on shipped code: the shipped forgetting (`behavior.py:77`, $D_t=\lambda D_{t-1}+\ell_t$,
predict with $e^{-D_{t-1}}$) and discount-then-predict (predict with $e^{-\lambda D_{t-1}}$) share the recursion for
$D_t$, hence the same MAP trajectory on the same data; they differ only in the sharpness of q-hat (T1: no ln k bound
for the shipped variant). SW4d adds the log-loss contrast and is run only if cores are free.

Controls (no new runs): RQsafe (safe throughout, seeds 1-100) pairs with phase 1 of every arm; RQunif (uniform
throughout, seeds 1-100) is the $t_0=0$ reference for identification time (its t_sustained_P90 per stream:
8, 40, 62, 22, 21, 16, 67, 34, 24, 25) and the win-rate reference on post-switch seeds <= 100. SW4's seeds 101-180
have no RQ control (descriptive only). Phase 1 of every arm is its own no-switch segment for false alarms.
Updates per seed in RQ: safe 57/55/59/49 at positions 0-3; uniform 37/43/34/38/40/27/29/27/33 at positions 1-9.

## 2. Hypotheses

H1 (T5(a), lambda = 1): after the switch the posterior re-concentrates on `uniform` with delay linear in $t_0$,
$\tau^*\approx t_0$ updates for this pair, i.e. about 3 seeds (6 games) in SW2 and 6 seeds (12 games) in SW4: the
carried posterior does not track within a game or two.

H2 (T5(c), lambda = 0.98): delay about 35 updates (max over competitors of
$\ln((\mu_{a\to b}-\mu_{a\to s}+\mu_{b\to s})/\mu_{b\to s})/\ln(1/\lambda)$: safe 31, br_unif 35, br_us 27),
independent of $t_0$, i.e. within one seed.

H3 (descriptive): win rate. The control is at the ceiling against uniform (RQunif 0.9948), so no detectable effect
of the stale posterior on the post-switch score is expected.

## 3. Metrics (per stream k; unit = non-skipped posterior update, forced updates included)

Log fields. Per-decision records `<tag>-s<seed>-g<game>.A.decisions.jsonl` (also `decisions.A` of the game JSON):
`bayes.posterior` (pre-decision $P_{t-1}$, rounded to 4 decimals), `feedback.skipped`, `feedback.bayes.p_qhat`
(pre-update predictive probability of the realised column, 6 decimals), `feedback.bayes.p_style[s]` (6 decimals,
never below MIX/n_cols), `feedback.bayes.posterior` (post-update, 4 decimals), `feedback.bayes.forced`,
`feedback.bayes.n_cols`; records with `final: true` carry a game's last feedback. Game JSON: `seed`, `game`,
`stream`, `methods.B` (style actually played), `winner_role`, `error`. Stream order:
`engine/analysis/synthesis/common.py:streams`.

Reconstruction. Because the logged posterior is rounded to 4 decimals ($P(\text{uniform})\approx e^{-38}$ at the
switch logs as 0), every posterior quantity is recomputed by replaying the arm's recursion on the logged
`p_style` sequence (lambda = 1: $\log P_t\propto\log P_{t-1}+\log p_{s,t}$; lambda = 0.98 PTD: $z=\lambda\log P_{t-1}+\log p_{s,t}$;
DTP: $z=\lambda(\log P_{t-1}+\log p_{s,t}$)), from the uniform prior; the logged posteriors are only the validation
target (max abs difference in probability <= 1e-3 at every update; measured on RQsafe streams 0, 5 and RQunif streams 0, 5: <= 9.4e-5; a larger difference marks the stream
`unclean` and it is excluded, as B6's `clean` rule).

- $t_0^{(k)}$: updates while `methods.B == style_safe`; $T^{(k)}$: all updates; $P_{t_0}$: reconstructed posterior
  after the last phase-1 update.
- **tau_map**: first post-switch index $\tau$ (1-based from the switch) at which the reconstructed MAP is `uniform`
  for 20 consecutive updates (the T5 measurement rule). Right-censored at $T-t_0$.
- **tau_90 / tau_99**: first $\tau$ with $P(\text{uniform})\ge0.9$ / $\ge0.99$ for 20 consecutive updates.
- **tau_leave**: first $\tau$ with $P(\text{safe})\le0.1$ for 20 consecutive updates.
- **Per-stream T5 prediction** (uses only $P_{t_0}$ and the RQunif drifts, nothing post-switch):
  $\hat\tau^{(k)}=\max_{s\ne u}[\ln P_{t_0}(s)-\ln P_{t_0}(u)]^+/\mu_{u\to s}$, $\mu_{u\to s}$ = safe 0.395,
  br_unif 0.293, br_us 0.312, eq 1.773. For lambda = 0.98:
  $\hat\tau^{(k)}=\max_s\ln\big(1+(1-\lambda)G_{t_0}(s)/\mu_{u\to s}\big)/\ln(1/\lambda)$ with $G_{t_0}(s)$ the
  reconstructed discounted log-odds of s against uniform at the switch.
- **tau_leave prediction**: $\hat\tau_{\rm leave}^{(k)}=\min_{s\ne\text{safe}}[\ln P_{t_0}(\text{safe})-\ln P_{t_0}(s)]/(\mu_{u\to\text{safe}}-\mu_{u\to s})$
  over s with positive denominator (descriptive comparison only).
- **Post-switch excess log loss** $X^{(k)}=\sum_{t>t_0}(-\ln p_{\hat q,t}+\ln p_{u,t})$ (q-hat vs the `uniform` style
  on the post-switch segment). Oracle-switch reference: a posterior restarted at the switch has segment excess
  $\le\ln5=1.609$ (B1); the stale posterior's extra cost is $X-1.609$. For lambda = 1 the Bayes identity gives
  exactly $X^{(k)}=\ln P_T(u)-\ln P_{t_0}(u)$ on the reconstruction (consistency check, 1e-3).
- **False alarm**: a phase-1 index at which the reconstructed MAP is `uniform` for 20 consecutive updates (same rule
  as tau_map; the first 20 updates after a fresh prior are excluded).
- **Scores**: pair score of A per seed (`winner_role`; seeds with an error game dropped from scores only), split by
  phase; paired difference against RQsafe (phase-1 seeds) and RQunif (post-switch seeds <= 100) on the same seed.
- **Uncertainty**: medians over the 10 streams with bootstrap percentile intervals (2000 resamples over streams);
  ratios as median of per-stream ratios; Spearman rank correlation for P3. Carried-state check: `meta.updates` of the
  final `.bayes.json` equals the counted updates (B6 `clean`).

## 4. Predictions and refutation criteria (fixed before any run)

P1 (H1, absolute). SW2: median tau_map in [70, 170] updates (point 112 = nominal $t_0$); SW4: [130, 330] (point 220).
In games: SW2 flips during post-switch games 3-10 (seeds 2-5 of the stream), SW4 during games 5-18 (seeds 3-9).
Refuted if the median of tau_map/$t_0^{(k)}$ is < 0.4 or > 2.0 in either arm, or if >= 3 streams of an arm flip
within 30 updates of the switch (the posterior was not carried). SW4's post-switch window (~500) covers 2.0 $t_0$
(~440); SW2's (~290) covers 2.5 $t_0$.

P2 (H1, linearity). median tau_map(SW4) / median tau_map(SW2) in [1.5, 2.7] (point 2.0). Refuted if < 1.25.

P3 (H1, mechanism; the sharp test). Over the 20 lambda = 1 streams: median of tau_map/$\hat\tau^{(k)}$ in [0.7, 1.4];
Spearman correlation between $\hat\tau^{(k)}$ and tau_map >= 0.5. Refuted if the median ratio is outside [0.5, 2.0]
or the correlation is <= 0.

P4 (H1, mass). median(tau_90 - tau_map) in [0, 25] updates (log-odds must grow a further ln 9 = 2.2 nats at
>= 0.29 per update). tau_leave vs $\hat\tau_{\rm leave}^{(k)}$ reported descriptively (the model says br_unif usually
overtakes safe before uniform does; which happens first depends on the stream's phase-1 safe-br_unif gap).

P5 (H1, log loss). median $X^{(k)}$: SW2 in [20, 60] nats (point $0.344\,t_0+1\approx40$), SW4 in [45, 110]
(point about 77). Refuted if < 10 nats in either arm (posterior not stale).

P6 (H2). SW4p: median tau_map in [15, 70] updates (point 35); median tau_map(SW4p) / median tau_map(SW4) <= 0.4;
refuted if median tau_map(SW4p) > 100 or the ratio is > 0.7. Per-stream P3-type check with the discounted
$\hat\tau^{(k)}$: median ratio in [0.5, 2.0]. False alarms (sustained-20 rule) in phase 1: 0/10 streams in every arm;
2 or more streams with a phase-1 false alarm refutes the "uniform is well separated" premise. If SW4d is run: median
$X^{(k)}$ in [5, 30] nats (point about 15 = 0.395 x 35 plus tail), at least 2.5x below SW4's; SW4p's $X^{(k)}$ is
predicted >= SW4d's (T1: the shipped variant over-weights, so it pays more when wrong) but no interval is claimed.

P7 (H3, descriptive, no refutation). Post-switch paired score difference vs RQunif in [-0.05, 0.05]; phase-1 difference
vs RQsafe in [-0.10, 0.10] (10-40 pairs per arm).

P8 (implementation check, as B6 P1). lambda = 1 arms, on the reconstruction: at every update
$\sum_{u\le t}(-\ln p_{\hat q,u}+\ln p_{s,u})=\ln5+\ln P_t(s)$ for every style to 1e-3 on the reconstruction (measured on RQ: <= 2.0e-4) and to 5e-3 against the logged 4-decimal posterior (B6 measured 0.0013-0.0049;
only where it is >= 0.01). A violation refutes the implementation or the log, not T5.

Censoring: a right-censored tau_map counts against P1 only if the censoring point already exceeds the interval's upper
end; otherwise the stream is inconclusive and the median uses the censored value as a lower bound. Arms are not killed
mid-seed (an aborted seed updates the carried state before the resume rule reruns it: RESULTS.md Open items A).

## 5. Code change (optional arm SW4d only; H1, H2-flip-time and H3 run on the shipped code)

`engine/showdown/behavior.py`: opt-in discount-then-predict behind `OURBOT_BAYES_DTP=1` (`Posterior.update`:
`z = self.lam * (self.logp + np.log(lik))`, the one-line fix of theory §1), `config()` adds the key `dtp` only when
on (default state files and logs byte-identical). `engine/showdown/modal_showdown.py`: `--bayes-lambda`,
`--bayes-dtp` into `extra_env` (only needed on Modal). No logging change: every field of §3 is already written.

## 6. Cost and runtime

Local: SW2 100 + SW4 180 + SW4p 80 = 360 seeds = 720 games (SW4d +80 seeds). About 2 min per seed on an idle core
(RQ: ~50 decisions per game at ~1.1 s wall each, 48 CPU-s per game), 3 min while the gauntlet occupies 10 cores
(load 21 on 14 cores at writing). Stream lengths: SW2 10 seeds (~20-30 min), SW4 18 seeds (~36-54 min), SW4p 8 seeds
(~16-24 min). Schedule: SW2 and SW4p now with `--workers 2` each (4 cores): done in ~2.5 h; SW4 when the gauntlet ends
(~3 h) with `--workers 10`: ~40 min. All results in ~3.7 h. Modal alternative (`modal_showdown.estimate_cost`,
1-CPU streams, 48 CPU-s per game, 10 streams): SW2 $0.57, SW4 $1.02, SW4p $0.46, SW4d $0.46; total $2.05 ($2.51
with SW4d) over the exhausted $15 budget.

## 7. Launch checklist

1. Copy this file to `research/prereg_style_switch.md` in the worktree and commit it before any arm starts.
2. `OURBOT_WORK=$HOME/sandbox/ourbot/sw`; assert `$OURBOT_WORK/hedge_state` has no `SW*` files (`behavior.load_posterior`
   silently resumes an existing `<tag>-stream<k>-A.bayes.json` with a matching config) or use fresh tags.
3. Invoke `~/sandbox/bot-venv/bin/python /Users/biofanat/Documents/classes/f26/cs535/final-integration/engine/showdown/batch.py`
   (never the stale `~/sandbox/ourbot/batch.py`); always `--streams 10` (`batch.py main()` only guards `hedge`/`hedge_fs`
   against stream-less parallel runs, not `hedge_bayes`).
4. Discounted arms: log-odds at the switch for SW4p are the plain `logp` differences of the reconstruction; for SW4d
   (DTP stores -lambda D) divide the `logp` differences by lambda before applying the per-stream prediction.
5. Controls: RQsafe / RQunif game logs exist only in the main checkout
   (`/Users/biofanat/Documents/classes/f26/cs535/final/engine/showdown/results/<tag>/replays_headless`, gitignored); point
   the analysis script's control root there.

## Results (appended 2026-11-16; predictions above unchanged)

Runs: SW2 (seeds 1-100), SW4 (1-180), SW4p (1-80, lambda = 0.98, shipped predict-then-discount), all local, 0 error
games; SW4d not run. Full tables: `research/switch_results/out_style_switch.md` (+ `.json`), produced by
`research/scripts/switch/analyze_style_switch.py`. All 30 streams pass the posterior reconstruction check (max error
2.3e-4 vs the 1e-3 clean threshold).

| P | prediction | observed | verdict |
|---|---|---|---|
| P1 | median tau_map SW2 in [70,170] (pt 112), SW4 in [130,330] (pt 220) | 107.5 [77, 138]; 225 [150, 274]; tau_map/t0 1.13 and 1.09 | pass |
| P2 | tau_map(SW4)/tau_map(SW2) in [1.5, 2.7] | 2.09 | pass |
| P3 (primary) | median tau_map/tau_hat in [0.7, 1.4], Spearman >= 0.5 (20 streams) | 1.06, rho = 0.86 | pass |
| P4 | median tau_90 - tau_map in [0, 25] | 4 (SW2), 6.5 (SW4) | pass |
| P5 | X: SW2 in [20,60], SW4 in [45,110] | 29.2, 66.2 | pass |
| P6 | SW4p median tau_map in [15,70] (pt 35), ratio to SW4 <= 0.4; false alarms 0/10 per arm (refuted at >= 2) | 37.5 [26, 62], ratio 0.17; false alarms SW2 1/10, SW4 0/10, SW4p 0/10 | pass (the 0/10 point prediction missed in SW2: stream 7 had P(uniform) = 0.067 at the switch, i.e. the posterior had not yet separated safe from uniform; below the refutation threshold) |
| P7 | descriptive score differences within +-0.05 (post) / +-0.10 (pre) | post +0.006, +0.009; pre 0.000, 0.000 | as predicted |
| P8 | identity error <= 1e-3 (reconstruction), <= 5e-3 (logged) | 6.3e-4; 4.9e-3 | pass |

Reading: this is the first live test of T5 (bayes_dynamics.md §6). The posterior's re-concentration time after a
style switch is predicted per stream from its own switch-time posterior and the RQ drift rates (P3: median ratio 1.06,
rank correlation 0.86 over 20 streams), and scales linearly with the pre-switch evidence (P2). Forgetting at
lambda = 0.98 cuts the delay about 6x, matching the T1-based point prediction (35 vs 37.5).
