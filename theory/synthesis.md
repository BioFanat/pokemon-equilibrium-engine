# Synthesis: what the project says about no-regret learning (Lecture 6) and no-regret dynamics (Lecture 7)

Written 2026-11-15 as the final development-and-connection pass. Sources read in full: `../lecture6.pdf`, `../lecture7.pdf`,
the proposal, `theory_map.md` (status table §9), `game_model.md`, `experts_extensions.md` (B1–B12), `bridge_plan.md`,
`dynamics_directions.md`, `new_directions.md`, `../engine/RESULTS.md` ("Real-Showdown experiments"),
`../engine/results/belief/report.md`, `../report/current_report.tex`, `../report/final.tex`, and the decision logs under
`../engine/showdown/results/<TAG>/replays_headless/`. Nothing in `report/*.tex` or the engine was edited.

**Status words.** *measured* = recomputed here from the logs (script named); *recorded* = taken from a verified source
(theory_map / RESULTS, not recomputed); *estimated* = arithmetic on measured numbers with a stated unit assumption;
*speculative* = an interpretation. Units are kept apart as in `theory_map.md` §0.2: engine units (stage matrix $\bar A_t$,
median range $\Delta_t\approx230$), Hedge reward units ($/400$), A01 units (bandit roots), payoff $[-1,1]$, score $[0,1]$.

**New scripts and outputs** (all in `/private/tmp/claude-501/-Users-biofanat-Documents-classes-f26-cs535/2bbe670b-2f97-4d9a-bfad-e1139e0352c6/scratchpad/synthesis/`):
`common.py` (stream reconstruction: stream $k$ = sorted seeds$[k::10]$, games g1 then g2, as `batch.split_streams`),
`bh_regret.py` → `out_bh_regret.json` (BH log-loss regret curves), `bayes_identity.py` → `out_bayes_identity.txt`,
`hedge_dyn.py` → `out_hedge_dyn.json`, `out_hedge_dyn.txt` (E2/E2fs/E5/BH Hedge dynamics), `e4_gap.py` →
`out_e4_gap.json` (node-level gaps), `checks2.py` → `out_checks2.txt` (E4 filter reconciliation; BH non-forced log loss),
`voi_toy.py` → `out_voi_toy.txt`, `weight_replay.py` → `out_weight_replay.txt` (per-stream log-integrity check). Total CPU under 10 s.

---

## 1. The learning questions

The lectures supply two objects. **Lecture 6**: one learner, $n$ experts, losses chosen by Nature that "may depend on
everything that happened earlier, but not on our random choice for the current day"; regret $=$ our loss $-\min_i\sum_t l_i^t$;
MWU gives $L_{\rm MWU}\le(1+\eta)L^*+\ln n/\eta$, i.e. regret $\le2\sqrt{T\ln n}$; deterministic rules suffer linear regret
(§2.3). **Lecture 7**: both players no-regret with full information on one matrix $A$; the *time averages* satisfy
$\max_xx^\top A\bar y-\min_y\bar x^\top Ay\le\epsilon_R+\epsilon_C$ ("daily play may continue to oscillate, but … every limit
point of the empirical averages is a minimax equilibrium"); in general games the time-average joint play is an
$\epsilon$-CCE; sampled actions need "an additional expectation or a high-probability guarantee" (§3.2); humans
probability-match instead (§4.1: 58% vs 70%).

The project asks nine precise questions about these two objects in a real game.

| # | Question | Lens |
|---|---|---|
| Q1 | Does per-turn no-regret self-play (RM$^+$ on the estimated stage matrix) reach the stage equilibrium, at the rate Lecture 7's argument gives? | L7, M0–M1 |
| Q2 | Does a carried-weight experts learner keep its regret below the Lecture 6 bound against a real, adaptive opponent, on every path? | L6, M4 |
| Q3 | Does bandit feedback at a search node break Lecture 7's convergence (time-average gap $=(\mathrm{Reg}^1+\mathrm{Reg}^2)/N$)? | L7, M0–M4 |
| Q4 | Does the decision rule (play the time-average mixture vs its mode) matter more than the feedback model? | L7 §4.1, L6 §2.3 |
| Q5 | Can a no-regret learner over stage experts beat the equilibrium expert (Lecture 7's minimax content, PA4$\beta$)? | L7, M2–M4 |
| Q6 | Is a Bayesian opponent model literally Lecture 6's MWU, and does it keep its regret below $\ln k$ on real opponents? | L6, M0 (M3 inputs) |
| Q7 | Does a better opponent prediction (Q6) or a better-calibrated belief (more worlds) convert into wins? | S/L6, M4 |
| Q8 | How do belief error and value error enter the full-game guarantee (the proposal's "Nash gap $\le$ solver regret + belief error + value error")? | S, M1–M3 |
| Q9 | Is a posterior over sets (and tendencies) a sufficient memory, and when does information have value? | S, M3 |

---

## 2. Evidence map

Numbers tagged *measured* were recomputed here; *recorded* ones come from `theory_map.md` §8–§9 or RESULTS.

| Q | Lecture result it builds on | Our theorem (status) | Experiment (numbers, 95% CI) | Verdict | Weakest link |
|---|---|---|---|---|---|
| Q1 | Lecture 7 §2.2: $\max_xx^\top A\bar y-\min_y\bar x^\top Ay\le\epsilon_R+\epsilon_C$ with time averages | Prop. 4 / Cor. A′ (RM$^+$ as coded: gap $\le2\Delta(\sqrt n+\sqrt m)/\sqrt T$; proved, key step Lean); Lemma 1 (Lean) for the estimated matrix | *recorded*: 35 real stage matrices, median gap $7.0\cdot10^{-5}$ at $T=1000$, slope $-1.26$ vs $-0.5$, $\le0.04$ of the bound (headline Fig. A) | **Answered** (at M0–M1) | The equilibrium reached is that of $\hat Q$, not $Q^*$; $\varepsilon_v\approx1$ on exact endgames (B01) |
| Q2 | Lecture 6 §4: $L\le(1+\eta)L^*+\ln n/\eta$; §2.1 adversary model (= A1) | Lemma R2a (anytime FTRL, pathwise, proved); L2(a)–(d) and Theorem E2 (proved with (E)); P1 (external $\ne$ policy regret, proved) | *measured* (`hedge_dyn.py`): on all 40 carried streams of E2, E2fs, E5, BH the anytime external regret never exceeds the R2a bound; max ratio regret/bound over all $(T,\text{stream})$: 0.178 (E2), 0.204 (E2fs), 0.138 (E5), 0.166 (BH). Per turn at each stream's end: median 0.0041 (E2), 0.0048 (E2fs), 0.0032 (E5), 0.0038 (BH) reward units vs bound 0.026–0.030. Win rate *recorded*: E2 0.3108 [0.2804, 0.3412] vs AF 0.3144 [0.2792, 0.3495] | **Answered** for surrogate regret; **open** for policy regret | Regret is on surrogate stage rewards along our own trajectory (P1), with 16–17% of turns skipped (semi-bandit); the four experts differ by only 0.011 reward/turn (§3.4) |
| Q3 | Lecture 7 §2.2 as an identity; §3.2 last sentence (sampled actions) | Prop. L4 (identity, proved); Cor. B3 (numeric; bandit high-probability bounds vacuous at $k=9$, $N=2000$); Lemma D (proved) | *measured* (`e4_gap.py`): logged time-average gap at the bandit root, all 17,820 E4e decisions median 0.0585 (matches RESULTS); restricted to two-sided nodes (15,277) median 0.083, mean 0.098; grows with columns ($n<8$: 0.053; $n\ge27$: 0.200). RM$^+$ side: 0.0000. *recorded*: E4e 0.0625 [0.0430, 0.0820], E4u 0.0502 [0.0312, 0.0691] vs RM$^+$ | **Answered**: bandit feedback leaves a 0.06–0.08 gap at $N=2000$ that full information does not; but see Q4 | Head-to-head, not vs an exploiter as designed; the gap is exploitability against a best response, which the RM$^+$ root does not play (B8) |
| Q4 | Lecture 7 §4.1 (probability matching 58% vs 70%); Lecture 6 §2.3 (deterministic rules are exploitable) | B8 (deficit vs an equilibrium opponent $=\sum_{i\notin\mathrm{BR}}\bar x_i\,\mathrm{slack}_i$; (i)–(ii) proved, (iii) hypothesis; pre-registered); B9 (max-visit never consistent on strictly mixed matrices) | *recorded*: E4emv 0.4949 [0.4547, 0.5351], E4umv 0.5933 [0.5551, 0.6316] vs 0.0625 / 0.0502 for the shares; median top probability 0.315 / 0.225 vs 0.78 | **Answered** against a non-adapting opponent: the rule dominates the feedback | Exploitability of max-visit play untested (no exploiter arm); E4umv $>0.5$ means neither root plays an equilibrium of the other's matrix |
| Q5 | Lecture 7 §2.2 one-sided: Row's average payoff $\ge v_C-\epsilon_R$ | PA4$\beta$ (proved: a learner gains at most the equilibrium bot's exploitability); L2(a) sandwich | *recorded*: E5 0.4583 [0.4210, 0.4956], band $[0.45,0.55]$ not refuted, CI below 0.5. *measured*: in E5 the top expert earns most surrogate reward in all 10 streams (0.323 vs sampled 0.3201 per turn), Hedge ends with $w_{\rm top}\in[0.37,0.45]$ | **Partially**: no gain over the equilibrium (as PA4$\beta$ allows); the loss below 0.5 has no theorem | PA4$\beta$ bounds only the upper side; the lower side needs the sampled expert alone vs agent form (a 0.5 control) and top alone vs agent form |
| Q6 | Lecture 6 §3–4: MWU; mixability replaces the Hoeffding step | B1 (Bayes $=$ exponential weights with $\eta=1$ on log loss; pathwise regret $\le\ln5$; cite + adaptation); N1 equality case (cite) | *measured* (`bh_regret.py`): on the 8 clean BH streams (0 and 6 excluded: logs overwritten by reruns, 2266 vs 2344 and 2069 vs 2114 updates) the final regret vs the best style is 1.609 on every stream $=\ln5$ to 3 decimals; anytime max 1.6099 (6-dp rounding); update counts match the state files; cumulative log-likelihood gaps match the state files to $\le0.016$ nats | **Answered**; the equality case $\mathrm{Reg}_T(s)=\ln5+\ln P_T(s)$ (§4, N1, standard) explains why the bound is attained | The comparator class is five softmax styles; realisability fails (the best style changes between streams: *safe* on 6, *br_unif* on 2, 26–65 nats apart over $\approx2{,}200$ turns) |
| Q7 | (none directly; Lecture 6's "hustler": exploit a flawed opponent) | B5 (safex loss Lipschitz in the prediction, proved); B6 (whole-game bound for hedge_bayes, vacuous per game); B10/B11 ($K$-dependence; no win-rate gain derivable) | *recorded*: BH 0.3329 [0.3022, 0.3637], BH$-$E2 $+0.024$ [$-0.018$, $+0.066$]; AF16 0.3278, AF16$-$AF $+0.017$ [$-0.031$, $+0.065$]. *measured*: BH prediction at the full run (15,456 non-forced turns, 8 streams): log loss $\hat q$ 1.773 vs best style 1.772, old predictor 3.328, uniform 2.677; argmax accuracy 0.431 vs 0.420 | **Answered negatively** at $n\approx300$–400 pairs: prediction improves by 1.5 nats/turn, wins by $<0.04$ | Theorems A/C charge the remaining loss to $\varepsilon_v$; B6's conversion needs a global calibration $a>0$, which exact endgames refute ($a=0$) |
| Q8 | Lecture 7 at each stage (Q1) composed by backward induction | Lemma 1 (Lean), Theorem A (Lean + proved), Theorem B/B′, Theorem C and Cor. C.1 (proved): $\delta_C=\varepsilon_s+2\varepsilon_v+2\mathbb ES_t+\eta_{\rm NL}+2\Delta\,\mathrm{TV}$ | *recorded*: B12 replay isolates the $2\varepsilon_v\bar H$ channel (sampled play: summed stage gap 0.001/game, exploitability 0.426 payoff units); calibrated $\varepsilon_v$ 0.8–1.0 on exact endgames; E1 agent form vs PIMC 0.5587 [0.5209, 0.5966] (engine 0.585) | **Answered** as a theorem; **vacuous in level** | $\varepsilon_v\approx1$ makes every absolute bound trivial; mid-game $\varepsilon_v$ unmeasured |
| Q9 | — (proposal goals ii, iii) | A2 (hard-evidence belief $=$ posterior for non-revealing opponents); Example C.1 (three types: TV term necessary, expl $=1$); A3 and $\eta_{\rm NL}$ (`full_game.md`:160–162: "the value, to the agent, of having $\theta$ revealed") | *recorded*: belief calibration ECE 0.01–0.04 (posterior) vs 0.09–0.16 ($K=4$ worlds), truth absent from all worlds 3–10%; oracle gap $\approx0.1$ score (engine). *measured* (`voi_toy.py`): VOI $=\eta_{\rm NL}$ identity and the A3 agent's loss on a 2-type toy | **Partially** (sufficiency: delivered via A2 + Example C.1 + E1; VOI: identified with $\eta_{\rm NL}$, no experiment) | Belief-dependent continuation values do not exist in the engine (A3), so VOI is identically zero inside the agent |

---

## 3. Gaps and missed connections, with what was fixed

### 3.1 Lecture 7's time averages and CCE: what the project measures and what it cannot

**Where a time average of the real dynamics exists.** Lecture 7's objects $\bar x_T,\bar y_T$ need one fixed game played
repeatedly. In the battle the stage matrix changes every turn, so a game-level $\bar x$ does not exist. Time averages of
real dynamics exist at exactly three places, and all three are measured:

1. **Inside a turn** (RM$^+$ on $\hat Q_t$): the output *is* the time average; its gap is Lecture 7's left-hand side.
   *recorded*: median $7\cdot10^{-5}$ at $T=1000$ (Fig. A).
2. **At a bandit root** (E4): visit shares are $\bar x_N,\bar y_N$ and the logged `gap` is exactly
   $\max_i(A^{01}\bar y_N)_i-\min_j(\bar x_N^\top A^{01})_j=(\mathrm{Reg}^1_N+\mathrm{Reg}^2_N)/N$ (Prop. L4) on
   **17,820 real decisions**. *measured* (`e4_gap.py`): median 0.0585 (all decisions), 0.083 on two-sided nodes;
   RM$^+$ root 0.0000; the shares' gap at the max-visit bots' positions is larger (E4emv 0.121, E4umv 0.141 median)
   because those games reach positions with more opponent columns. This is the one place where the project measures
   Lecture 7's §2.2 quantity on a real-game trajectory, and it should be said so in the report (edit E7 below).
   Reconciliation: `experts_extensions.md` B7 calls the 17,820 decisions "two-sided"; 2,543 of them have one side with a
   single option (`abar_shape` has a 1; the gap is still $>0$ on most of them, from exploration mass on dominated rows). The medians 0.0585 / 0.0578 are over all logged bandit decisions.
3. **The meta-game over experts** (E2, E2fs, E5, BH): the learner's time-average weights $\bar w_T$ and its two regrets.
   *measured* (`hedge_dyn.py`), per stream at its own $T$ (1,685–2,450 feedback turns):

   | Tag | ext. regret / $T$ (median) | R2a bound / $T$ | dynamic regret / $T$ (per-turn best expert) | expert spread / $T$ | $w_T(\text{top})$ | $\bar w_T(\text{top})$ | sd of $w_t(\text{top})$, 2nd half |
   |---|---|---|---|---|---|---|---|
   | E2 | 0.0041 | 0.0263 | 0.0139 | 0.0109 | 0.45–0.57 | 0.40–0.49 | 0.008–0.037 |
   | E2fs | 0.0048 | 0.0263 | 0.0139 | 0.0112 | 0.39–0.50 | 0.38–0.42 | 0.017–0.027 |
   | E5 | 0.0032 | 0.0297 | 0.0147 | 0.0075 | 0.37–0.45 | 0.32–0.40 | 0.009–0.029 |
   | BH | 0.0038 | 0.0265 | 0.0138 | 0.0091 | 0.47–0.54 | 0.38–0.44 | 0.012–0.026 |

   Reading. (a) The Lecture 6 comparator (a fixed expert) is also Lecture 7 §3.1's CCE deviation (every *fixed*
   $s'_i$), so the external regret of 0.004 per turn is the row player's CCE deviation gain in the surrogate meta-game.
   The per-turn comparator ("deviate to the best expert of each turn") is *dynamic* regret, a strictly stronger
   comparator with no equilibrium meaning; it measures how non-stationary the best expert is along the trajectory. The
   two differ by $3.4\times$ and both are small: 0.014 reward units is $\approx5.6$ engine units per turn against a
   median stage range of $\approx230$.
   (a′) *Log integrity* (`weight_replay.py`): the same update-count test that excludes BH streams 0 and 6 also fails on
   E2 stream 9 (2411 logged vs 2448 in the state file), E2fs stream 9 (2353 vs 2390) and E5 streams 1 and 4 (1779 vs
   1837, 1685 vs 1723): resumed runs. Excluding them moves no median (E2 0.0041 → 0.0041, E5 0.0032 → 0.0030, BH
   0.0038 → 0.0039). The report's weight-replay discrepancy of 0.0092 (headline panel B caption) localises entirely to
   E2 stream 9; the other nine E2 streams, and all count-clean E5 and BH streams, replay to 0.0001 (4-dp rounding). So
   0.0092 is a log artefact of a resumed stream, not a property of the learner (edit E15).
   (b) **Last iterate vs average**: the carried anytime-FTRL weights do *not* oscillate; $w_t$ drifts with sd 0.01–0.04
   over the second half and the leader changes 0–25 times per stream, with the top expert the cumulative leader in
   every stream. Lecture 7's "daily play may oscillate" regime is absent here because the opponent (stock foul-play,
   or agent form) does not learn in the meta-game; what oscillates in this project is the Bayesian posterior
   (§3.2), not the Hedge weights.

**What cannot be computed from the logs.** The opponent's side of Lecture 7's identity needs its counterfactual
payoffs $p_t^\top\bar A_t[:,j]$ for every column $j$. No run logs $\bar A_t$ (only `abar_shape`; schema map of the
decision records), so the opponent's payoff regret vs any comparator class (fixed column labels, the five styles, or
the stage best response) is **not computable** from E2/E5/BH. The one-sided statements above are all the logs support.
A log-only rerun that writes $\bar A_t$ and the five style vectors $q_{s,t}$ (already written in BH) would make the
two-sided surrogate identity $\bar g_T-\bar v_T=\epsilon_C^{\rm opp}-\epsilon_R^{\rm us}$ along the real trajectory
computable at zero theory cost (≈ \$0.7 Modal or 1 CPU-h per tag). That is the smallest fix for "we never measure a
two-sided Lecture 7 quantity above the node level".

**CCE specifically.** In a zero-sum game a CCE's marginals are maximin and its deviation gain equals the external
regret (theory_map §5.1), so "CCE" adds nothing to the minimax statement except as a *correlation* diagnostic
($\mathrm{TV}(J_N,\bar x_N\bar y_N^\top)$, U-fin). The only correlation measurement is the H4 probe: at $10^3$–$10^4$
iterations the engine root's joint visits are indistinguishable from the product of marginals. The report should say
"CCE = minimax here; the only CCE-specific object is correlation, which we measured and found absent at practical budgets".

### 3.2 Regret curve of the Bayesian posterior (fixed: computed)

*measured* (`bh_regret.py`, `bayes_identity.py`). Per stream, feedback turns only (the posterior updates only there),
$\hat\ell_t=-\ln p_{\hat q}(b_t)$ and $\ell_{s,t}=-\ln q_{s,t}(b_t)$ from `feedback.bayes` (6 dp), regret
$R_T=\sum_{t\le T}\hat\ell_t-\min_s\sum_{t\le T}\ell_{s,t}$.

| stream | updates (logs / state) | final $R_T$ | anytime $\max_TR_T$ | best style | state$-$log log-lik. gap (nats) |
|---|---|---|---|---|---|
| 1 | 2159 / 2159 | 1.609 | 1.609 | safe | 0.009 |
| 2 | 2376 / 2376 | 1.609 | 1.609 | safe | 0.013 |
| 3 | 2173 / 2173 | 1.609 | 1.610 | safe | 0.015 |
| 4 | 1969 / 1969 | 1.609 | 1.610 | safe | 0.015 |
| 5 | 2404 / 2404 | 1.609 | 1.609 | safe | 0.016 |
| 7 | 2234 / 2234 | 1.609 | 1.609 | br_unif | 0.009 |
| 8 | 2396 / 2396 | 1.609 | 1.609 | br_unif | 0.014 |
| 9 | 2450 / 2450 | 1.609 | 1.610 | safe | 0.006 |
| 0 (excluded) | 2266 / 2344 | 1.610 | 1.610 | br_unif | 71.8 |
| 6 (excluded) | 2069 / 2114 | 5.029 | 5.029 | safe | 58.4 |

$\ln5=1.6094$. On the 8 clean streams the bound holds anytime up to 6-dp rounding (max excess $5\cdot10^{-4}$) and is
**attained** at the end of every stream: the posterior has concentrated on one style ($P_T(s^*)=1.0000$). Streams 0 and 6
fail the update-count check (the B1 verifier's caveat) and are excluded; stream 6's 5.03 is an artefact of two runs'
logs interleaved, not a violation. The identity of §4 (N1) explains the equality: $R_T(s)=\ln5+\ln P_T(s)$, verified
to 0.005 nats on all 18,161 clean-stream turns with $P_T(s)\ge0.01$.

Two qualitative facts for the narrative. (i) The two near-tied styles (*safe*, *br_unif*) are 26–65 nats apart over
$\approx2{,}200$ turns (0.011–0.030 nats per turn); *eq* and *uniform* trail by 1,450–2,090 nats. (ii) The posterior's
**last iterate** is a vertex in all 10 streams while its **time average** is mixed (stream 0: 0.50/0.50; 3: 0.41/0.59;
4: 0.40/0.60; 8: 0.59/0.41; 9: 0.62/0.38 on br_unif/safe): the posterior switched between the two near-tied styles
during the run. This is Lecture 7's "daily play may oscillate; the averages converge" shown on a Lecture 6 object, and
it is exactly what B4(iii) warns about: regret $\le\ln5$ says nothing about which style is "true".

**Table 5 of the report at full scale** (*measured*, clean streams, 15,456 non-forced feedback turns; the 24-pair pilot
had 1,128): log loss $\hat q$ 1.773, best single style 1.772 (*safe*), *br_unif* 1.79, *br_us* 2.01, *eq* 2.69, uniform
2.677, previous predictor 3.328; argmax accuracy $\hat q$ 0.431, *safe* 0.428, *br_unif* 0.420, *eq* 0.410, previous
0.420. Including forced turns (one legal column, log loss 0 for everyone) and all 10 streams: $\hat q$ 1.503, best style 1.502, old 2.816, uniform 2.277 (stream means; not the same set as the non-forced numbers).

**Proposed headline panel D** ("Opponent model: Bayes as MWU"): $x$ = feedback turns $T$ (linear, 0–2,450), $y$ =
cumulative regret $R_T$ vs the best style, one line per clean stream (8), a horizontal line at $\ln5$, and a second
$y$-axis or inset with $\ln P_T(s^*)$ to show the identity; caption: "the bound is a constant, is met with equality once
the posterior concentrates, and is attained on every stream". Data: `out_bh_regret.json` (`curves`). It is the fourth
place where the project uses no-regret learning and the only one whose bound does not grow with $T$.

### 3.3 E4 and Lecture 7: mixture vs mode of the time average (fixed: framing corrected, proposition stated)

The task's phrasing "last iterate vs time average" is not what E4 contrasts. The shares bots play $\bar x_N$ (the time
average, Lecture 7's object); the `_mv` bots play $\arg\max_i\bar x_N(i)$, the **mode of the same time average**, not
the last iterate $x_N$. The right reading is:

**Proposition N3 (mixture vs mode of the average against a non-adapting opponent).** Let $\bar x_N$ be the visit shares
at a node with matrix $A$, $y^*$ an equilibrium column mixture with value $v$, $\mathrm{BR}(y^*)$ the rows attaining $v$
against $y^*$, $i_{\rm mv}\in\arg\max\bar x_N$. Under A1(ii) (the opponent's law does not react to our realised action)
and the hypothesis that the opponent plays $y^*$ at this node (B8(iii)):
(i) $v-\bar x_N^\top Ay^*=\sum_{i\notin\mathrm{BR}(y^*)}\bar x_N(i)\,(v-(Ay^*)_i)\le\mathrm{gap}_N=(\mathrm{Reg}^1_N+\mathrm{Reg}^2_N)/N$
(B8(i), L4): the average pays for its exploration mass;
(ii) $v-(Ay^*)_{i_{\rm mv}}=0$ whenever $i_{\rm mv}\in\mathrm{BR}(y^*)$, in particular at a pure saddle whose row carries the
maximal share; on strictly mixed matrices $i_{\rm mv}$ has stage gap $\ge\gamma_A>0$ against a best response for every
$N$ (B9(i)).
(iii) Against an **adaptive** opponent, $i_{\rm mv}$ is a deterministic function of the public history, so Lecture 6
§2.3 applies: the ordering of (i)–(ii) can reverse. No theorem fixes by how much; on 57 real matrices the two rules'
stage gaps are 0.053 vs 0.058 (recorded), so the reversal is not predicted to be large either.
*Proof.* (i) and (ii) are B8(i)–(ii) with $x=\bar x_N$ and $x=e_{i_{\rm mv}}$; the inequality in (i) is L4. (iii) is Lecture 6
§2.3 verbatim. $\square$ *Numbers* (recorded): shares median top probability 0.315 / 0.225 (Exp3 / UCB1) vs 0.78 for RM$^+$;
scores 0.0625 / 0.0502 (shares) vs 0.4949 / 0.5933 (mode).

**Connection to Lecture 7 §4.1.** Playing $\bar x_N$ against a fixed opponent *is* probability matching: the shares
reproduce the frequencies the bandit explored with, as the human subjects reproduce 70/30; the mode is "always guess
green". Lecture 7's arithmetic (58% vs 70%) is the one-node version of E4e vs E4emv. The lecture adds the caution the
project has not tested: a no-regret learner "would eventually perform nearly as well" as the fixed best prediction only
when its regret has vanished; at $N=2000$ the shares' gap of 0.06–0.08 has not.

**Connection to Lecture 7 §2.2.** "Daily play may oscillate; the averages converge" is about $x_t$ vs $\bar x_t$ inside
the node; the project has a diagnostic (recorded): OMWU's last iterate is bimodal on real matrices (median 0.0085, p90
0.086), so the solver returns averages. E4 shows that *averages carry exploration* and the MCTS rule discards it.

### 3.4 E5 below 0.5 and E2 $=$ AF: the cleanest readings, with numbers

**E2 $=$ AF (0.3108 vs 0.3144).** Three facts from the logs make this expected rather than disappointing.
(a) *measured*: the four experts earn almost the same surrogate reward: top 0.3253, sampled 0.3207, safest 0.3167,
safex 0.3142 per turn (E2 average over streams), a spread of 0.011 reward units $=4.4$ engine units per turn
(*estimated* with the fixed normaliser 400) against a median stage range of 230, i.e. $\approx2\%$ of the range. All
four experts act on the *same* $\hat Q_t$, so by Theorem A they share the same $2\varepsilon_v\bar H$ residual; they can
differ only through $\varepsilon_s$-type terms, and those are 2% of the range per turn. (b) *measured*: Hedge trails the
best expert (top, in all 40 streams across E2/E2fs/E5/BH) by 0.0041 reward units per turn, well under the bound; its
earned 0.3211 lies between top and sampled. (c) P1: external regret along our trajectory does not bound policy regret;
the two differ by a constant in a two-stage example. So L2(a) predicts $W_{\rm Hedge}\approx W_{\rm best\ expert}$ up to
terms that are small *in the surrogate*, and the surrogate differences are themselves small; with $\varepsilon_v\approx1$
the only large term is common to all experts. The honest caveat stays: only the sampled expert (AF) was run alone on
Showdown, so "E2 $=$ AF" tests the sandwich at one of its four corners.

**E5 $<0.5$ (0.4583 [0.4210, 0.4956]).** Hedge's own expert "sampled" *is* the opponent's strategy (same algorithm,
different world samples), so following it would score 0.5 by symmetry. Hedge instead puts $\bar w_{\rm top}\approx0.32$–0.40
(final 0.37–0.45) on the deterministic top move, plus mass on safest and safex, because *in the surrogate* top earns
more than sampled even against the equilibrium opponent (0.323 vs 0.320 per turn, *measured*). Two theorems say why
that need not convert: (i) B8(i): against a stage-equilibrium column $y^*$ of the **opponent's** matrix, a pure row
costs its slack unless it is a best response to $y^*$; top is a best response in *our* $\hat Q_t$ (it is in the support of
our equilibrium) but not necessarily in the opponent's, which is built from different $K=4$ worlds (E4umv $>0.5$ shows
the two bots' matrices disagree in this way); (ii) P1(iv): zero external regret is compatible with a constant policy
regret. The regret cost itself is not the explanation: at E5's $T\approx1{,}800$ feedback turns the measured regret is
0.0032 reward units per turn (bound 0.030), i.e. 1.3 engine units per turn, under 1% of the range (*estimated*). The
"small-horizon" reading applies only to a reset learner: with $T\approx25$ turns per game the R2a bound is 0.25 reward units
per turn (*estimated*), which is why carrying weights matters (R3d). Deficit $-0.042$ [$-0.079$, $-0.004$] is inside the
band that the world-sampling noise between the two bots' matrices allows, but no theorem gives that band a number;
the smallest fix is two control runs, *sampled alone vs agent form* (should give 0.5) and *top alone vs agent form*
(300 pairs each, ≈ \$1.1 Modal at E5's cost), which would split E5's deficit between the deterministic expert and the
mixing.

### 3.5 The proposal's open goals

**Value of information (fixed as far as theory goes; experiment impossible in the current engine).** The proposal
asked "when is it rational to sacrifice immediate damage to reveal an item". Theorem C already contains the object:
$\eta_{\rm NL}(t,s',b')=\sum_\theta b'(\theta)V_{t+1}(s',\delta_\theta)-V_{t+1}(s',b')$ is "the value, to the agent, of
having $\theta$ revealed at $(s',b')$" (`full_game.md`:160–162). That is the value of *perfect* information at the next
node, and because $V_{t+1}(s',\cdot)$ is convex in the belief (Lemma 7(a)) it is $\ge0$ with equality iff the optimal
strategy does not depend on the type (Jensen). What is **missing** is precisely stated: the agent's continuation values
are world-only (A3), i.e. $V_{t+1}(s',b')$ is replaced by $\sum_\theta b'(\theta)V_{t+1}(s',\delta_\theta)$, which is
*linear* in $b'$. Under A3 every action's VOI is identically zero, and the agent behaves as if it will learn the type
for free at the next node (strategy fusion one level down). So no probing experiment can be run until a
belief-dependent leaf exists (a PBS value as in DeepStack/ReBeL, cited in `full_game.md` §6.1).

*Toy statement, checked numerically* (`voi_toy.py`, N4): two types, prior $b$; stage 1 "probe" (cost $c$, signal of
accuracy $\alpha$) or "skip"; stage 2 pick a row, payoff 1 iff it matches the type, so $v(b')=\max(b',1-b')$. Then
$\mathrm{VOI}(\alpha)=\mathbb E[v(b')]-v(b)\ge0$ (0 violations on 10,000 chords), $=\alpha-\tfrac12$ at $b=\tfrac12$;
$\eta_{\rm NL}$ at the skip node $=1-v(b)=$ VOI of a perfect reveal; the A3 agent values every continuation at 1,
never probes, and loses $\max(0,\mathrm{VOI}(\alpha)-c)$: 0.5 at $(b,\alpha,c)=(\tfrac12,1,0)$, 0.2 at $(\tfrac12,0.7,0)$,
0 whenever $c\ge\mathrm{VOI}$. The engine-level upper bound on total VOI along play is the oracle gap, $\approx0.1$ score
(recorded, engine, 100-game matchups): information is worth at most that much, and the conjecture
$\sum_t(\eta_{\rm NL}+4\mathrm{TV})\approx$ oracle gap (theory_map §4.3) is the open quantitative link.

**Memory sufficiency (delivered, with one named hole).** Positive side: under A2 (non-revealing opponent) the
hard-evidence belief over sets equals the true posterior, so the belief is a sufficient statistic of the history for
the stage game, and Theorem C's TV term vanishes; the agent-form solve over that belief beats the PIMC blend that
*ignores correlations between the opponent's choice and its type* (E1 0.5587 [0.5209, 0.5966] on Showdown, 0.585 in
the engine; blend loss $\le4\delta_H$ on 46,890 decisions). Negative side: Example C.1 (three types, repaired) has
exploitability 1 against a revealing opponent with every other term 0, so a belief over *sets alone* is not sufficient
memory against opponents whose play depends on their type; that is why the style posterior (tendencies) is the second
memory component the proposal asked for. The hole: the proposal's "bounds on the loss from compressing that belief"
has only the sampling rates S1/S2 (compression to $K$ samples) and no marginals-vs-joint ablation; the belief report
shows the $K=4$ compression is the lossy step (ECE 0.09–0.16 vs 0.01–0.04; truth absent from all worlds 3–10%).

### 3.6 Theorems without experiments, experiments without theorems

**Theorems with no real-game test** (status table §9, "—" or "not run"), grouped with the smallest fix:
- *Testable on existing logs with one rerun that logs $\bar A_t$*: P1 (policy vs external regret: needs $\bar A_t$ and
  skipped turns), the two-sided Lecture 7 identity (§3.1). One log-only rerun of E2 fixes both.
- *Needs a short new run*: P2 (switch time; E3 not run, 160 games); B5/B6's `bayes_safex` (no game run); PA2 ε-sweep;
  B8(iii)/B9 exploitability of max-visit (E4 `_mv` vs the white-box exploiter, the only arm that tests Lecture 6 §2.3
  on the MCTS rule); E5 controls (§3.4).
- *M0 results that are not meant to have a game test*: Lemma H, R1, R2a (tested through E2), L4b, U1–U4, U-fin,
  Lemma D, Prop. T, B2, B3(b), S1, S2. Keep as theory; say so.
- *Would need belief-dependent leaves*: everything about $\eta_{\rm NL}$, VOI, concealment.

**Experiments with no theorem or registered prediction**, and the smallest statement that covers each:
- E5's CI below 0.5: no lower-side theorem (PA4$\beta$ is upper-side). Fix: the two control runs; theory: B8(i) applied
  to each expert against the opponent's $y^*$ (needs the opponent's matrix, loggable).
- E4umv $>0.5$: B8(iii) is a hypothesis for `rmplus_avg`; the fact that it fails says the two roots solve different
  matrices (world sampling). Fix: log both sides' $\bar A_t$ in one rerun and compute $\|\bar A^A_t-\bar A^B_t\|_\infty$,
  which is a direct measurement of the world-sampling part of $\varepsilon_v$ (Theorem B's $S$ term) on Showdown.
- BH and AF16 (n.s. gains): B6 and B11 say "no win-rate prediction derivable"; the honest theorem-side statement is
  Theorem C's decomposition with the measured ranking of terms (solver $\approx0$, belief/sampling $\approx0.1$,
  value $\approx1$), i.e. a *prediction of no detectable effect*, which the runs confirm. Register it as such.
- Belief calibration (ECE, absent fractions): covered by B10 (Jensen) for absent fractions; ECE has no theorem and
  needs none (it is a measurement of A4/sampling).

---

## 4. Small new results (each checked numerically)

**N1 (Bayes regret identity; cite + adaptation, checked).** Standard prequential identity (Cesa-Bianchi & Lugosi 2006 §9, the source B1 cites); new here only as the equality case of B1(ii) and as the statement behind the proposed figure panel. With $P_0$ uniform on $k$ styles and $P_t(s)\propto P_{t-1}(s)q_{s,t}(b_t)$,
for every $T$ and every style $s$,
$$\sum_{t\le T}\big(\hat\ell_t-\ell_{s,t}\big)=\ln k+\ln P_T(s).$$
*Proof.* $\prod_{t\le T}\hat q_t(b_t)=\sum_{s'}\tfrac1k\prod_t q_{s',t}(b_t)$ (B1's chain rule) and
$P_T(s)=\prod_tq_{s,t}(b_t)\big/\sum_{s'}\prod_tq_{s',t}(b_t)$; divide and take $-\ln$. $\square$
*Consequences.* B1(ii) with its equality case: $R_T(s)\le\ln k$ with equality iff $P_T(s)=1$; the regret curve against
any comparator is the comparator's log-posterior shifted by $\ln k$, so the headline panel D can show either. *Checked*:
max error 0.0048 nats over 18,161 turns $\times$ styles with $P_T(s)\ge0.01$ (4-dp rounding of the logged posterior).
Scope: $\lambda=1$ only (B3 shows the shipped $\lambda<1$ breaks the telescoping).

**N2 (BH pathwise regret, measured).** On the 8 clean BH streams the regret against the best style is $\le\ln5$ anytime
(to $5\cdot10^{-4}$ rounding) and equals $\ln5$ to three decimals at the end of every stream; update counts match the
state files; cumulative log-likelihood gaps match the state files to $\le0.016$ nats. Streams 0 and 6 excluded (count
mismatch from overwritten logs; stream 6 reads 5.03 for that reason). Full-run prediction quality: §3.2.

**N3 (mixture vs mode of the time average; proved from B8/B9/L4).** Statement and proof in §3.3. Numeric support
recorded (E4 scores; stage gaps 0.053 vs 0.058 on 57 real matrices).

**N4 (VOI $=\eta_{\rm NL}$; A3 sets it to zero; toy checked).** Statement in §3.5; `voi_toy.py` verifies the three
identities on a 36-point grid and convexity on 10,000 chords.

**N5 (expert spread; measured).** Across 40 carried streams the four Hedge experts differ by 0.0075–0.0112 reward units
per turn (3–4.5 engine units; $\approx1.3$–2% of the median stage range), top is best in hindsight in 40/40, Hedge
trails it by 0.0032–0.0048 per turn, and the dynamic regret (per-turn best expert) is 0.0138–0.0147 per turn. This is
the quantitative content of "the experts share the leaf bottleneck": the learning layer has 2% of the range to work
with per turn, and it uses it as the bound says.

Not fixed (stated as open): the two-sided Lecture 7 identity on the real trajectory (needs $\bar A_t$ logged); E5's lower
side; max-visit exploitability; any VOI experiment.

---

## 5. Recommended final narrative (one page)

The paper should tell one story with two lectures as its spine and three honest negatives as its load-bearing walls.

**The spine.** Lecture 6 gives one learner a guarantee against an adversary who knows its distribution but not its
draw; Lecture 7 shows that when both sides learn, the *time averages* reach minimax even though daily play may not.
We used no-regret learning at four places in a real imperfect-information game and asked, at each, whether the
lecture's guarantee survives contact with the game: (A) inside a turn, RM$^+$ self-play on the estimated stage matrix
reaches a Nash gap of $7\cdot10^{-5}$ in 1,000 iterations, forty times under its bound; (B) across 600–800 real Showdown
games per run, a carried Hedge learner over four stage experts keeps its pathwise external regret under the Lecture 6
bound on all 40 streams of four experiments (0.003–0.005 reward units per turn vs 0.026–0.030); (C) at a search node,
the visit shares of bandit self-play are Lecture 7's time averages and their gap is the players' summed regret, measured
at 0.06–0.08 on 17,820 real decisions where full-information RM$^+$ reaches 0; (D) the Bayesian opponent model is
Lecture 6's MWU with $\eta=1$ on log loss, and on 8 real streams its regret against the best of five styles equals
$\ln5$ at the end of every stream, because the regret is exactly $\ln5+\ln P_T(s)$ and the posterior has concentrated.
All four lecture-level predictions held, pre-registered where a run was involved.

**What the game adds.** Composing (A) over a battle gives Theorem A/C: the Nash gap of play is bounded by solver regret
plus twice the matrix error plus belief and sampling terms, which is the proposal's target inequality. The three
honest negatives then follow from one measured fact: the matrix error $\varepsilon_v$ is near 1 on exact endgames, so
every absolute bound is vacuous and every remaining loss is charged to the leaf. First, the learning layer wins at its
own level but not in games: Hedge matches its best expert within 0.004 reward units per turn, yet its four experts
differ by only 2% of the stage range and E2 scores what its sampled expert scores alone (0.311 vs 0.314); the posterior
halves the previous predictor's log loss (1.77 vs 3.33 nats) and wins 0.024 [−0.018, 0.066] more. Second, Lecture 7's
average is the wrong thing to *play* against a non-adapting opponent: the same bandit scoring 0.05–0.06 with its shares
scores 0.49–0.59 with their mode, Lecture 7 §4.1's probability-matching lesson at a search node; what the mode costs
against an adaptive opponent (Lecture 6 §2.3) is untested. Third, a no-regret learner cannot beat the equilibrium
expert (E5 0.458, as PA4$\beta$ allows) and in fact lost to it at 95% confidence, for which we have a mechanism (pure
experts pay their slack against an equilibrium of a *different* sampled matrix) but no theorem.

**What is new, and what is not.** The theory is assembly: Lecture 7 at every stage (Lean), Lecture 6 along the
trajectory (pathwise, anytime, with the code's indexing), Bayes as Lecture 6 (mixability instead of Hoeffding), the
exploration-deficit identity behind E4, and the levels M0–M4 that say which assumption carries each result to the
next. The experiments are the first real-Showdown tests of these statements with pre-registered refutation criteria.
The one structural win is memory: the agent-form solve over the belief beats the PIMC blend on Showdown (0.559), the
strategy-fusion loss Theorem C assigns to PIMC. The proposal's value of information is identified with Theorem C's
non-locality term and is identically zero under world-only leaves, which names the one component (a belief-dependent
leaf) that would make the next version of this agent able to probe, bluff, and be worth more than its leaf.

---

## 6. Concrete edit list (not applied)

### `report/current_report.tex`

- **E1 (§2 "How we use them", after "Lemma D is the high-probability statement…")**: add one sentence: "Lecture 7's time
  average exists at exactly three places in our system (inside RM$^+$, at a bandit root, and over the experts), and the
  node-level gap of 17,820 real E4 decisions (median 0.0585) is Lecture 7's left-hand side measured on a real
  trajectory; a game-level $\bar x$ does not exist because the stage game changes every turn."
- **E2 (§2, "In zero-sum games a CCE's marginals are maximin…")**: append "the only CCE-specific object is correlation of
  joint play, which the H4 probe found absent at practical budgets."
- **E3 (§4.1, after Prop. `prop:style`)**: add the identity N1 as a corollary with its one-line proof, and the sentence
  "so $R_T(s)\le\ln|\mathcal S|$ with equality iff the posterior has concentrated on $s$; on the eight clean BH streams
  the final regret is 1.609 on every stream (§\ref{sec:behav})".
- **E4 (§4.2, after Prop. `prop:B8`)**: replace "B8 predicts that a bandit root loses…" paragraph's first sentence with
  the N3 framing: "The shares bot plays the time average $\bar x_N$ of Lecture 7; the `_mv` bot plays its mode. Against
  a non-adapting opponent the average pays its exploration mass (B8(i)) and the mode pays nothing on best-response
  rows (B8(ii)); against an adaptive opponent the mode is a deterministic rule (Lecture 6 §2.3), untested here."
- **E5 (§6.2, E2 paragraph)**: add the measured spread: "the four experts earn 0.325 / 0.321 / 0.317 / 0.314 reward
  units per turn (E2, stream average), a spread of 4.4 engine units against a median stage range of 230; Hedge earns
  0.321 and trails top, the best expert in hindsight on all 10 streams, by 0.0041 per turn." Add "the per-turn
  best-expert (dynamic) regret is 0.014 per turn, 3.4× the external regret and still 2.4% of the range".
- **E6 (§6.2, E5 paragraph)**: add the mechanism and the missing controls: "In the surrogate, top still out-earns
  sampled against the equilibrium opponent (0.323 vs 0.320 per turn) and Hedge ends top-weighted (0.37–0.45); a pure
  row costs its slack against an equilibrium of the opponent's matrix, which is built from different worlds (B8(i);
  E4umv $>0.5$ shows the matrices disagree). The 0.042 deficit has no theorem; sampled-alone and top-alone vs agent
  form (300 pairs each) would split it."
- **E7 (§6.2, E4e/E4u paragraph)**: state the measured gap on two-sided nodes (0.083 median, 0.098 mean, 15,277
  decisions) alongside the all-decisions 0.0585, and that the latter includes 2,543 nodes where one side has a single option; cite Prop. L4 as
  the reason the logged `gap` *is* Lecture 7's quantity.
- **E8 (§6.3 Behavioural model, Table 5)**: add a third column pair "vs stock, full BH run (8 streams, 15,456 non-forced
  turns)": posterior 1.773 / 0.431; *safe* 1.772 / 0.428; *br_unif* 1.79 / 0.420; *eq* 2.69 / 0.410; uniform 2.677 /
  —; previous 3.328 / 0.420. Add: "the pathwise regret vs the best style is $\le\ln5$ anytime on all eight clean streams
  and equals $\ln5$ at the end of each (streams 0 and 6 excluded: logs overwritten by reruns, update counts 2266 vs
  2344 and 2069 vs 2114)"; and the last-iterate/time-average observation (posterior final = vertex, time average
  0.4–0.6 between *safe* and *br_unif* on 4 of 10 streams).
- **E9 (headline figure)**: add panel D as specified in §3.2; caption text there. Update the figure caption's first
  line to "at the four places we use no-regret learning".
- **E10 (§7 Discussion)**: after "Lecture 7's prescription works per stage but not as a search rule", insert "it is the
  *average* that carries exploration; the MCTS rule plays its mode, Lecture 7 §4.1's majority guess", and after "The
  Bayesian style model … chooses the best-calibrated style" insert "and its regret meets Lecture 6's bound with
  equality, because for Bayes the regret is $\ln|\mathcal S|+\ln P_T(s)$".
- **E11 (§8 Limitations, item 3 "Not run")**: add "a log-only rerun writing $\bar A_t$, which would give the opponent's
  surrogate regret and the two-sided Lecture 7 identity along the real trajectory, and the direct measurement
  $\|\bar A^A_t-\bar A^B_t\|_\infty$ of the world-sampling error between two bots".
- **E12 (§8 item 8, proof goals)**: replace with "Proof goal (iii): the value of information is Theorem C's
  $\eta_{\rm NL}$, the value of a perfect reveal at the next node; it is $\ge0$ by convexity and identically 0 under
  A3, so no probing experiment is possible with world-only leaves (toy: a probe worth 0.5 is never taken). Proof goal
  (ii): sufficiency holds under A2 (A2 + E1 + Example C.1); the compression bound exists only for $K$-sampling (S1, S2);
  no marginals ablation."
- **E13 (Table `tab:goals`)**: row "Value of information, bluffing": state → "identified ($=\eta_{\rm NL}$), zero under A3";
  row "Proof goal (iii) VOI": "formal object named; no experiment possible without belief-dependent leaves".
- **E15 (headline figure caption, panel B; §6.2 E2 paragraph)**: "The logged weights match those replayed from the
  rewards to within 0.0092" → "to within 0.0001 on nine streams; stream 9 (a resumed run whose log has 2411 of the
  state file's 2448 updates) differs by 0.0092". Optionally drop stream 9 from panel B (medians unchanged).
- **E14 (Provenance appendix)**: add the scripts of this note with their outputs.

### `report/final.tex` (the short in-engine paper)

- **F1 (§3 Preliminaries, Prop. `prop:selfplay`)**: it cites `CS535L6` for Lecture 7 content; the bibliography has a single
  `CS535L6` entry covering both lectures. Either split into `CS535L6`/`CS535L7` or rename the key. Add after the proof
  idea: "Daily play may oscillate; only the averages converge, which is why our solver returns the average and why the
  MCTS rule (the mode of the average) is a different object."
- **F2 (§6.1 "parallels the lectures' probability matching")**: the parallel is with the *shares vs mode* contrast, not
  only top vs sampled; one sentence pointing to the Showdown E4 result if the short paper is to be updated, otherwise
  leave.
- **F3 (§4.2 Relation to the lectures)**: after "the comparator is the best expert along our own trajectory (external,
  not policy, regret)", add the number from this note's E5/E2 logs only if Showdown results are admitted into
  `final.tex`; otherwise no change.
- **F4 (§7 Discussion)**: add the sentence "Hedge's four experts act on one matrix and so share its leaf error; their
  surrogate rewards differ by about 2% of the stage range per turn, which is the room the learning layer has."

---

## 7. Remaining open gaps (ranked by what they would settle)

1. **Two-sided Lecture 7 identity on the real trajectory** — needs $\bar A_t$ logged (one rerun). Would also give P1's
   policy regret and the inter-bot matrix distance (world-sampling part of $\varepsilon_v$).
2. **Exploitability of the max-visit rule** (B8(iii)/B9(iv), Lecture 6 §2.3 on the MCTS rule) — `_mv` vs the white-box
   exploiter; the only untested direction of E4.
3. **E5's lower side** — sampled-alone and top-alone vs agent form.
4. **Mid-game $\varepsilon_v$** — unmeasured; everything absolute hangs on it.
5. **VOI / concealment** — impossible without a belief-dependent leaf; the toy says what to build.
6. **Realisability of the style class** — the best style changes between streams; B4(iii) means the posterior's
   accuracy has no guarantee; a run against the five styles themselves (realisable by construction) would separate
   misspecification from estimation.
7. **E3 (switch test) and PA2 (ε-sweep)** — not run; P2 and the certificate arithmetic remain theory-only.
