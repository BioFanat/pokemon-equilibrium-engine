# Two-sided no-regret, minimax and CCE quantities along real games

*Research note, 2026-11-16. Scripts: `research/cce/scripts/` (`cce_load.py`, `two_sided.py`, `opp_side_check.py`);
aggregates: `research/cce/out/*.json`; raw decision logs: `~/sandbox/ourbot/research_logs/cce_logs_2026-11-16.tgz`
(CEQ, CSS) and `final/engine/showdown/results/LOG/replays_headless/` (LOG). Status words as in `theory/theory_map.md` §0.4.*

## Corrections (independent verification, 2026-11-16)

An independent recomputation from the raw logs (`research/cce/verification.md`; own loader, column-player LP, game
bootstrap) reproduces every headline number to the last printed digit. The corrections are to what the numbers are
said to mean; the body below has been edited accordingly. For paper wording use only the "Allowed wording" of
`verification.md`.

Highest-risk issues:
- **"Indistinguishable from an equilibrium player" is refuted.** Foul Play (LOG) and the literal eq player (CEQ) match
  on one statistic only, $\epsilon_C(\text{eq})$ 0.0213 vs 0.0211 (difference $+0.0003\pm0.0013$, bounding it only to
  about $\pm12\%$). Other statistics through the same model separate them: posterior P(br\_unif) 0.54 vs 0.09,
  br\_unif log loss 1.505 vs 1.647, $\hat q$ log loss 1.492 vs 1.580, $\epsilon^{\rm dyn}_C$ 0.0408 vs 0.0369.
- **"Floor" / "resolution" was the wrong word.** CSS shows ε-safe opponents at 0.0106 and 0.0152, below 0.021, so 0.021
  is a reference value for one opponent type, not a lower bound. "Information asymmetry" is an interpretation; each
  side's own sampling of $K=4$ worlds is an alternative not ruled out.
- **The CBL Thm 2.3 bound was applied with the wrong $T$.** Hedge is carried per stream ($T_s=1600$–$1940$): the bound
  is 0.038–0.042 per turn per stream (0.0403 pooled) for LOG, not 0.0127, so measured regret is 11% of it, not 35%;
  CEQ's is 0.075, not 0.0235. The guaranteed $\epsilon_R\approx0.04$ exceeds the measured dynamic regret (0.030), so
  the guaranteed-CCE statement is weak, and the guarantee is on the learner's clipped $M_k$ reward, not on the
  $\bar A$-based $\epsilon^E_R$ (10.2% of turns have $\operatorname{ptp}(\bar A)>400$, 1.8% a clipped entry). The
  code's step $\eta=\sqrt{8\ln4/t}$ is one step ahead of CBL's $\eta_{t+1}$ (constant only).
- **Prop 3's "conclusion holds numerically" is withdrawn.** On CEQ the per-turn oracle gap goes 0.00137 → 0.00105 from
  the first to the second half of each stream; the cumulative gap ($4.18\bar\Delta$ over 3,501 turns) grows roughly
  linearly, not O(1). Hypothesis (B) is false on $m_t=1$ turns (2,553 LOG, 580 CEQ) and needs "$m_t\ge2$". The
  hypothesis is best refuted by $\min_s\mathrm{TV}(y^B,q_s)=0.36$ and posterior weight 0.008 on eq against an eq
  opponent.
- **I1 and I2 are tautologies**; "the two-sided Lecture 7 identity closes" is not a finding.
- **Provenance:** stream 9's resumed status was cited to `synthesis.md` §3.1(a′), which is about E2 stream 9. For LOG
  the evidence is LOG's own Hedge state (1940 updates vs 1928 logged; weight replay breaks at seed 100). Stream 3 also
  mismatches (1811 vs 1776) because the excluded error game LOG-s4-g2 (34 usable updates) is in it; including it leaves
  $\epsilon^E_R$ at 0.00444.

Per-claim corrections (verification items 1–8):
1. *Fields*: confirmed; Ā is logged on decisions with $\ge2$ legal options only (not "every non-forced decision").
   17,271 usable decisions of 20,657 with feedback, 599 games.
2. *CCE reading*: what is computed is a restricted-deviation (Φ-type) coarse correlated gap over the realised sequence
   of *our estimated* stage games, with tiny classes ($|E|=4$, $|S|=5$), per-stage counterfactual deviations, and the
   opponent scored in our matrices. Experts and styles are history-dependent advice, not fixed policies. The row
   dynamic comparator sees the realised column (clairvoyant); the column one sees only our mixture, so 0.030 vs 0.041
   is not a symmetric pair. "Exact" applies to the arithmetic only. Say "CCE only with respect to these comparator
   classes".
3. *LOG numbers*: confirmed (Table 2.1).
4. *CEQ decomposition*: numbers confirmed (paired turns 3,501, not 3,507; 6 have zero overlapping mass);
   interpretation "indistinguishable" refuted (above).
5. *CSS*: run-mean exploitability 0.0013 (0.00134 / 0.00133); 53–54% of turns exceed 0.0015 (max 0.0048 / 0.0076);
   the per-turn guarantee is $\epsilon\Delta_t$, held up to $\le0.06$ engine units of rounding, not "up to 1.5".
6. *Log loss vs payoff ranking*: confirmed; safe vs br\_unif is within noise; the payoff result is near-structural
   ($q_{\rm eq}=0.95\cdot$maximin$+0.05\cdot$uniform); predictor vs policy, not a paradox.
7. *Posterior-learner statements*: Prop 4 citation right, bound misapplied (above). Calibration: ECE 0.0147 and the
   low bins reproduce, but $\hat q$ is over-confident at 0.7–1.0 (0.79 → 0.66, 0.996 → 0.98); 0.443 is the mean
   predicted top-1 probability and the top-1 hit rate is 0.521 (0.347 vs 0.438 excluding $m_t=1$). Under-confident only
   for favourites predicted below 0.5. Kalai–Lehrer: say "its hypotheses do not hold", not "fails".
8. *Correlation*: numbers confirmed; the permutation null is uninformative (it rejects even for the stateless CEQ
   opponent), so the meta-level TV is not evidence of anything learning-related.

## 0. Verdict on "not computable from the logs"

**"Impossible" was too quick — it was true of E2/E5/BH and is now stale.** The sentence "No run logs $\bar A_t$ (only
`abar_shape`)" (`theory/synthesis.md:114–116`, last committed 2026-11-15 23:20) was correct when written; the `LOG`
summary is dated one hour later (2026-11-16 00:23) and `b1_gap.py` / theory_map's open-item B1 row already consume
it. The `LOG` tag (hedge_bayes vs Foul Play, seeds 1–300, Modal, `OURBOT_LOG_ABAR=1`,
`engine/showdown/results/LOG/LOG_summary.json:381`) writes, on every decision where we have at least 2 legal options (single-option decisions log no $\bar A_t$), exactly the fields the
passage asks for: `abar` (2 dp), `labels`, `legal`, `pfull`, `Q` (the four experts) and `qs` (the five style vectors),
plus `bayes.qhat` (`engine/showdown/run_ourbot.py:541–551`); the realised opponent column is in the next record's
`feedback.bayes.j` / `feedback.b` (`run_ourbot.py:382–420`). `b1_gap.py` already consumed these fields for a different
purpose. So every quantity below is computable from existing data at zero theory cost; the only new runs are the
two local control runs (CEQ, CSS; 10 CPU-min each) needed to *interpret* the opponent-side numbers.

Three things were genuinely not possible and still are not: (i) the opponent's regret *in its own objective*
(Foul Play's MCTS values are not logged and are not our matrix); (ii) a payoff-regret *guarantee* for the posterior
learner without the Hedge wrapper or realisability (§3); (iii) separating "correlation through learning" from
"correlation through the shared stage" in the time-averaged joint play with one sample per stage (§4).

## 1. Definitions

**Setting (S).** Index by $t=1,\dots,T$ our decisions along the real trajectory (all games of a stream concatenated,
game boundaries ignored as in E2), restricted to decisions with $\ge2$ legal options for us and a feedback that was
not skipped (the opponent's realised action is one of our matrix's columns). At $t$ the log gives:

| symbol | logged field | meaning |
|---|---|---|
| $\bar A_t\in\mathbb R^{n_t\times m_t}$ | `abar` | world-averaged stage matrix, rows our legal options, columns the union of the $K=4$ worlds' opponent labels; engine units; we maximise, the opponent minimises |
| $p_t\in\Delta_{n_t}$ | `pfull` | the mixture we played |
| $q_{e,t}\in\Delta_{n_t}$, $e\in E=\{$top, sampled, safest, safex$\}$ | `Q` | the four Hedge experts (hedge methods) |
| $q_{s,t}\in\Delta_{m_t}$, $s\in S=\{$eq, br\_unif, br\_us, safe, uniform$\}$ | `qs` | the five styles, computed on $\bar A_t$ (bayes methods) |
| $\hat q_t=\sum_sP_{t-1}(s)q_{s,t}$ | `bayes.qhat` | the posterior's prediction |
| $j_t\in[m_t]$ | next record's `feedback.bayes.j` | the opponent's realised column |
| $\Delta_t=\operatorname{ptp}(\bar A_t)$, $\bar\Delta=400$ | — | stage range; Hedge's fixed normaliser (`exp3_live.py:34`) |

All of $\bar A_t,p_t,q_{e,t},q_{s,t},\hat q_t$ are functions of the history before the opponent's action at $t$ is
revealed (they are computed at the decision, `run_ourbot.py:460–490`), so they are predictable with respect to the
realised columns. Unless stated otherwise every payoff below is divided by $\bar\Delta=400$, so that a value of
$0.004$ is "$0.004$ of Hedge's reward unit per turn", the unit of E2's regret figure. The per-turn range is
$\Delta_t/\bar\Delta\approx0.67$ (median $\Delta_t$: 269 LOG, 266 CEQ, 273 CSS).

**Definition 1 (stage quantities).** For each $t$ (the four experts and five styles are history-dependent advice, known before the turn: safex depends on
$\hat q_t$, br\_us on our Hedge weights; so the regrets below are valid, but the comparators are not fixed policies):
$g_t=p_t^\top\bar A_te_{j_t}$ (realised stage payoff: our mixture against their realised column);
$e_{k,t}=q_{k,t}^\top\bar A_te_{j_t}$ (what fixed expert $k$ would have earned);
$c_{s,t}=p_t^\top\bar A_tq_{s,t}$ (what fixed style $s$ would have cost the opponent against our $p_t$);
$v_t=\max_x\min_yx^\top\bar A_ty$ (stage minimax value, LP);
$\operatorname{expl}_t=v_t-\min_jp_t^\top\bar A_te_j\ge0$ (our stage exploitability; `b1_gap.py`'s "played Hedge pfull" gap);
$\mathrm{br}^R_t=\max_i\bar A_t[i,j_t]$, $\mathrm{br}^C_t=\min_jp_t^\top\bar A_te_j$ (per-turn best responses).

**Definition 2 (regrets along the trajectory).** With $\bar g_T=\frac1T\sum_tg_t$,
$$\epsilon^E_R=\max_{k\in E}\tfrac1T\textstyle\sum_te_{k,t}-\bar g_T,\qquad
\epsilon^{\rm dyn}_R=\tfrac1T\textstyle\sum_t\mathrm{br}^R_t-\bar g_T,\qquad
\epsilon^S_C=\bar g_T-\min_{s\in S}\tfrac1T\textstyle\sum_tc_{s,t},\qquad
\epsilon^{\rm dyn}_C=\bar g_T-\tfrac1T\textstyle\sum_t\mathrm{br}^C_t .$$
$\epsilon^E_R$ is Lecture 6's external regret against fixed experts (what Hedge bounds); $\epsilon^S_C$ is the same
object for the opponent with the five styles as *its* experts; the dynamic versions compare with the per-turn best
response (Zinkevich 2003 / Besbes–Gur–Zeevi 2015 "dynamic regret") and dominate the restricted ones,
$\epsilon^{\rm dyn}_R\ge\epsilon^E_R$, $\epsilon^{\rm dyn}_C\ge\epsilon^S_C$, since $\mathrm{br}^C_t\le c_{s,t}$ and
$\mathrm{br}^R_t\ge e_{k,t}$ term by term.

**Which deviation classes are meaningful.** The stage matrix changes every turn and its column labels are
per-matchup (move names and species), so "a fixed column label across turns" is not a well-defined deviation (a
label legal at $t$ is usually absent at $t'$); we drop that class. What survives is exactly Lecture 7's structure
with "fixed pure strategy" replaced by "fixed *policy*": a map from the stage to an action. The experts are such maps
for us, the styles for the opponent, and the per-turn best response is the largest class (all maps). Lecture 7's
single-matrix setting is the special case where constant maps are the deviations.

**Definition 3 (empirical joint play and its CCE gap).** Let $\sigma_T$ be the distribution of $(t,a,b)$ obtained by
drawing a round $t$ uniformly, our action $a\sim p_t$ and the opponent's realised action $b=j_t$. For deviation
classes $D_R$ (maps $t\mapsto x_t\in\Delta_{n_t}$) and $D_C$ (maps $t\mapsto y_t\in\Delta_{m_t}$) the CCE gaps of
$\sigma_T$ in the sequence of stage games $(\bar A_t)_t$ are
$\max_{\phi\in D_R}\mathbb E_{\sigma_T}[\phi_t^\top\bar A_te_b]-\mathbb E_{\sigma_T}[e_a^\top\bar A_te_b]$ and the
mirror for the column player. By linearity these are $\epsilon_R^{D_R}$ and $\epsilon_C^{D_C}$: *"$\sigma_T$ is an
$(\epsilon_R,\epsilon_C)$-CCE with respect to $(D_R,D_C)$" is the same statement as "the two regrets are
$(\epsilon_R,\epsilon_C)$"* — Lecture 7 eq. (4) with policy deviations, in the sampled-action form of its §3.2 last
sentence on the opponent's side (our side enters through $p_t$, so it carries no sampling noise).

**Proposition 1 (restricted identity, I1).** $\max_k\frac1T\sum_te_{k,t}-\min_s\frac1T\sum_tc_{s,t}=\epsilon^E_R+\epsilon^S_C$.

**Proposition 2 (stage-value identity, I2).** $\bar g_T-\bar v_T=\epsilon^{\rm dyn}_C-\overline{\operatorname{expl}}_T$,
with $\bar v_T=\frac1T\sum_tv_t$ and $\overline{\operatorname{expl}}_T=\frac1T\sum_t\operatorname{expl}_t$; both terms on
the right are non-negative.

*Proofs.* Both are rearrangements of Definitions 1–2: for I2, $g_t-v_t=(g_t-\mathrm{br}^C_t)-(v_t-\mathrm{br}^C_t)$ and
$v_t\ge\mathrm{br}^C_t$ because $p_t$ is one feasible $x$. $\square$

*Remarks.* (i) I1 is Lecture 7 §2.2's $\max_xx^\top A\bar y-\min_y\bar x^\top Ay\le\epsilon_R+\epsilon_C$ as an identity
(the mechanism of Prop. L4): with regrets *defined* through comparator classes the inequality is an equality, and the
left side is the empirical duality gap of the joint play with respect to those classes. (ii) I2 is the form quoted in
`synthesis.md` §3.1, $\bar g_T-\bar v_T=\epsilon^{\rm opp}_C-\epsilon^{\rm us}_R$, with the per-turn best response as
the deviation class and our stage exploitability as "our" term; it says *our realised payoff exceeds the mean stage
value by the opponent's dynamic regret minus our exploitability*. (iii) The mirror decomposition
$\bar g_T-\bar v_T=\frac1T\sum_t(\mathrm{br}^R_t-v_t)-\epsilon^{\rm dyn}_R$ also holds but its first term is the
exploitability of a *pure* realised action and is not informative. (iv) I1 and I2 are tautologies (they hold by definition); their closure to $10^{-16}$
only checks the implementation (every term is computed from the same logged, 2-dp-rounded $\bar A_t$) and is not
evidence about the game. The content is the *sizes* of the terms. (v) A consistency check that is *not* an identity: Hedge's own reward increments
(`feedback.r`) are computed on the world-$k$ matrix $M_k$ and column index of the hit world
(`run_ourbot.py:405`), not on $\bar A_t$; $\epsilon^E_R$ recomputed from `feedback.r` and from $\bar A_t$ should agree
only approximately.

**What the opponent-side numbers mean.** Payoffs are *our* model (world-averaged depth-1 engine evaluation).
$\epsilon^S_C$ and $\epsilon^{\rm dyn}_C$ are the opponent's regrets *as measured through our matrices*: how far its
realised actions fall short, in our evaluation, of its best fixed style / of the best reply to our actual mixture.
Foul Play is not a learner in this game and never sees $\bar A_t$; the statement "the joint play is an
$(\epsilon_R,\epsilon_C)$-CCE of the sequence of estimated stage games *with respect to the comparator classes $E$ and
$S$*" is a statement about $\sigma_T$ and $(\bar A_t)$ only (exact only in its arithmetic), and says nothing about Foul
Play's own objective. It is a restricted-deviation (Φ-type) coarse correlated gap, not Lecture 7 §3.1's CCE of one fixed
game: the classes are tiny ($|E|=4$, $|S|=5$), deviations are counterfactual per stage and ignore their effect on later
states, and the opponent's deviations are scored in our matrices. The decisive question is the *scale*: an opponent
that plays the eq style *on its own information* has $\epsilon^S_C(\text{eq})\ne0$ in our model because its matrix
differs from ours. The CEQ control (§2.3) measures that value for one opponent type; it is a reference value, not a
lower bound (ε-safe opponents in CSS sit below it).

## 2. Measurements

Runs. **LOG**: hedge_bayes vs Foul Play (stock, 100 ms MCTS), seeds 1–300, 10 streams, 599 games, 20,657 decisions
with feedback, 17,271 usable (16.4% skipped: `not_in_worlds` 1,533, fainted 1,003, no action 431, cant 419; one error
game LOG-s4-g2 excluded). **CEQ** (new, local): hedge_bayes vs `style_eq`, seeds 1–80, 10 streams, both sides with
`OURBOT_LOG_ABAR=1`. **CSS** (new, local): `bayes_safex` vs `bayes_safex`, seeds 1–40, 10 streams, both sides logged —
the two-posterior-learner tag that `theory_map.md` T4 lists as "not run on Showdown". Commands:
```
OURBOT_WORK=<scratch> OURBOT_LOG_ABAR=1 ~/sandbox/bot-venv/bin/python engine/showdown/batch.py --tag CEQ --a hedge_bayes --b style_eq --seeds 1-80 --streams 10 --workers 10
OURBOT_WORK=<scratch> OURBOT_LOG_ABAR=1 ~/sandbox/bot-venv/bin/python engine/showdown/batch.py --tag CSS --a bayes_safex --b bayes_safex --seeds 1-40 --streams 10 --workers 10
~/sandbox/bot-venv/bin/python research/cce/scripts/two_sided.py <results_dir> <tag> [--side B] --out research/cce/out/<tag>.json
~/sandbox/bot-venv/bin/python research/cce/scripts/opp_side_check.py <results_dir> <tag>
```
Pair scores (A): LOG 0.308 [0.272, 0.343] (n = 299); CEQ 0.694 [0.631, 0.756] (n = 80); CSS 0.500 [0.380, 0.620] (n = 40).
SEs below are game-clustered; "streams" gives the min–max over the 10 carried-state streams.

### 2.1 Two-sided quantities (units: $\bar\Delta=400$ engine units per usable turn)

| quantity | LOG (vs Foul Play), $T=17{,}271$ | CEQ (vs style_eq), $T=5{,}036$ | CSS side A (safex vs safex), $T=2{,}680$ | CSS side B, $T=2{,}596$ |
|---|---|---|---|---|
| $\bar g_T$ | $-0.1128$ | $+0.1364$ | $-0.1180$ | $+0.0659$ |
| $\bar v_T$ (mean stage value) | $-0.1395$ | $+0.1102$ | $-0.1351$ | $+0.0448$ |
| **ours** $\epsilon^E_R$ (best expert) | $0.00444\pm0.00020$ (top; streams 0.0039–0.0050; stream 9 is the minimum at 0.0039 and is a resumed stream by LOG's own Hedge state: 1940 updates in the state file vs 1928 logged, weight replay breaks at seed 100; the other nine span 0.0040–0.0050) | $0.00240\pm0.00040$ (top in 9/10 streams; 0.0012–0.0038) | — (no experts) | — |
| ours $\epsilon^{\rm dyn}_R$ | $0.0301$ | $0.0381$ | $0.0383$ | $0.0366$ |
| ours $\overline{\operatorname{expl}}_T$ | $0.01413$ (streams 0.0123–0.0164) | $0.01074$ | $0.00134\pm0.00004$ | $0.00133$ |
| **opponent** $\epsilon^S_C$ (best style) | $0.02133\pm0.00061$ (eq; 10/10 streams; 0.0184–0.0227) | $0.02106\pm0.00113$ (eq; 10/10; 0.0177–0.0256) | $0.01062\pm0.00119$ (eq) | $0.01523\pm0.00127$ (eq) |
| opponent $\epsilon^{\rm dyn}_C$ | $0.0408\pm0.0007$ | $0.0369\pm0.0014$ | $0.0184\pm0.0012$ | $0.0224\pm0.0013$ |
| I1 residual | $3.5\cdot10^{-17}$ | $-4.9\cdot10^{-17}$ | — | — |
| I2 residual ($\bar g-\bar v-\epsilon^{\rm dyn}_C+\overline{\rm expl}$) | $-1.7\cdot10^{-17}$ | $0.0$ | $3.5\cdot10^{-18}$ | $-3.5\cdot10^{-18}$ |
| Hedge regret from logged `feedback.r` (remark v) | $0.00448$ | $0.00231$ | — | — |
| both-sided turns only ($n_t,m_t\ge2$) | $T=14{,}718$: $\epsilon^E_R$ 0.0052, $\epsilon^S_C$ 0.0250, $\epsilon^{\rm dyn}_C$ 0.0479, expl 0.0166 | $T=4{,}456$: 0.0027, 0.0238, 0.0417, 0.0121 | $T=2{,}355$: expl 0.0015, $\epsilon^S_C$ 0.0121 | $T=2{,}286$: 0.0015, 0.0173 |

Per-expert and per-style means (LOG): experts top $-0.1084$, sampled $-0.1133$, safest $-0.1172$, safex $-0.1186$;
our regret against each: top $+0.0044$, sampled $-0.0005$, safest $-0.0044$, safex $-0.0058$ (we beat two of the
four experts; sampled, $-0.0005$, is a tie within noise). Style costs: eq $-0.1341$, br\_us $-0.1008$, safe $-0.0978$, br\_unif $-0.0871$, uniform $-0.0116$; the
opponent's regret against each: eq $+0.0213$, br\_us $-0.0120$, safe $-0.0150$, br\_unif $-0.0257$, uniform $-0.1012$.
Expert exploitabilities (mean $v_t-\min_jq_{k,t}^\top\bar A_te_j$): top 0.0316, safest 0.0201, sampled 0.0100, safex 0.0013.

Readings (measured):
1. **Our side** reproduces E2/BH: $\epsilon^E_R=0.0044$ per turn against the best expert (top), 10/10 streams within
   0.0039–0.0050; recomputed from $\bar A_t$ it agrees with the learner's own $M_k$-based increments to $4\cdot10^{-5}$
   (0.00444 vs 0.00448), so the world-averaging makes no difference to the regret. The anytime-Hedge bound
   (Cesa-Bianchi–Lugosi 2006 Thm 2.3, $k=4$), $\sqrt{2\ln4/T}+\sqrt{\ln4/8}/T$, must be applied per stream, because
   Hedge is carried separately in each of the 10 streams ($T_s=1600$–$1940$): it is $0.038$–$0.042$ per turn per
   stream ($0.0403$ pooled), so measured regret is about 11% of it. (CEQ: bound $\approx0.075$.)
2. **Opponent side vs fixed styles.** In all four runs the *eq* style is the only style with positive regret
   (the opponent would have done better playing our LP equilibrium every turn) and it is the best style in
   10/10 streams of every run. Foul Play: $\epsilon^S_C=0.0213\pm0.0006$ (8.5 engine units per turn, 3.2% of the
   stage range), i.e. $4.8\times$ our own regret. Against the other four styles the opponent's regret is negative:
   its actual play beats br\_unif, br\_us, safe and uniform in our model. Note the ordering differs from log loss:
   in prediction, *safe* out-predicts *eq* (1.499 vs 2.242 nats, as in T3/RQ), while in payoff *eq* is the style the
   opponent should have played. Log loss and payoff regret rank the styles differently (new observation; safe vs
   br\_unif in log loss is within noise). This is not a paradox: the two rankings score different objects (a predictor
   vs a policy), and the payoff result is close to structural, since $q_{\rm eq}=0.95\cdot$maximin$+0.05\cdot$uniform
   gives $c_{{\rm eq},t}\lesssim v_t$.
3. **Dynamic regrets** are an order of magnitude larger on both sides (0.030 ours, 0.041 theirs, LOG). They are not a
   symmetric pair: our comparator $\max_i\bar A_t[i,j_t]$ sees the opponent's *realised* action (a clairvoyant
   deviation, inflated by the opponent's randomisation), while theirs, $\min_jp_t^\top\bar A_te_j$, sees only our
   mixture. Our stage exploitability (0.014) is a third of the opponent's dynamic regret, so by I2 we earn
   $0.027$ above the mean stage value even while losing 69% of games ($\bar g_T<0$: in our model we are behind on
   average because Foul Play is the stronger player; the *excess over value* is what the opponent leaves on the table).
4. **Lecture 7 bracket.** $v^R_T-\epsilon^E_R\le\bar g_T\le v^C_T+\epsilon^S_C$ with $v^R_T$ the best expert's mean and
   $v^C_T$ the best style's cost: $-0.1084-0.0044=-0.1128=-0.1341+0.0213$. These equalities hold by definition (I1 is a tautology) and are not a
   finding.

### 2.2 The posterior learner as a player (hedge_bayes's `safex` expert $=\epsilon$-safe best reply to $\hat q_t$, $\epsilon=0.003$)

| | LOG | CEQ | CSS A (plays it) | CSS B |
|---|---|---|---|---|
| payoff $\frac1T\sum_tq_{\text{safex},t}^\top\bar A_te_{j_t}$ | $-0.1186$ | $0.1340$ | $-0.1180$ | $0.0659$ |
| regret vs best fixed expert | $0.0102$ | $0.0048$ | — | — |
| dynamic regret | $0.0359$ | $0.0405$ | $0.0383$ | $0.0366$ |
| stage exploitability | $0.00130$ | $0.00129$ | $0.00134$ | $0.00133$ |
| opponent's $\epsilon^S_C$ against it (eq) | $0.0147$ | $0.0179$ | $0.0106$ | $0.0152$ |
| opponent's $\epsilon^{\rm dyn}_C$ against it | $0.0222$ | $0.0250$ | $0.0184$ | $0.0224$ |

The $\epsilon$-safe constraint is visible: the run-mean exploitability $0.0013$ (0.5 engine units) is below
$\epsilon\,\overline{\Delta_t}/\bar\Delta\approx0.0020$ on every run (T4's band, now on Showdown). This is a run-mean statement: per turn, 53–54% of CSS turns exceed
$0.0015$ (maximum 0.0048 side A, 0.0076 side B), because the per-turn guarantee is $\epsilon\,\Delta_t$, which scales
with the stage range; it holds up to $\le0.06$ engine units of rounding (`Q` at 4 dp, $\bar A_t$ at 2 dp). The
safety comes at the price of the largest regret against the top expert (0.0102 in LOG: the safe
player forgoes exploitation that the top expert takes). As a Hedge expert it is the worst of the four in payoff
(LOG) while being the one that keeps Hedge's mixture safe.

### 2.3 Calibrating the opponent-side numbers: a literal equilibrium opponent as reference (CEQ and CSS, both sides logged)

`style_eq` plays $y^B_t=q_{\rm eq}(\bar A'_t)$ on *its own* world-averaged matrix $\bar A'_t$ and logs it
(`pfull` on the B side). On turns with exactly one decision per side (CEQ: 3,507 paired, of which 3,501 have overlapping mass with our columns and enter the table, 1,298 unpaired/multi; CSS: 1,849 / 649), aligned
by label string (mean mass of $y^B_t$ on options absent from our columns 4.5% / 3.7%):

| | CEQ (eq opponent) | CSS (safex opponent) |
|---|---|---|
| $\mathrm{TV}(y^B_t,\,q_{{\rm eq},t})$ (ours) | $0.441\pm0.014$ (median 0.40; vs uniform 0.80) | $0.461\pm0.018$ |
| $\mathrm{TV}(y^B_t,\,\hat q_t)$ | $0.607\pm0.008$ | $0.648\pm0.012$ |
| $\frac1T\sum_t[\operatorname{val}(\bar A_t)+\operatorname{val}(\bar A'_t)]$ (0 in a shared game) | $-0.0376\pm0.0024$ (median $|\cdot|$ 12 engine units) | $-0.0340\pm0.0036$ |
| sampling term $\frac1T\sum_t(g_t-p_t^\top\bar A_ty^B_t)$ | $-0.0012\pm0.0007$ | $-0.0002\pm0.0005$ |
| **mismatch term** $\frac1T\sum_t(p_t^\top\bar A_ty^B_t-c_{{\rm eq},t})$ | $+0.0246\pm0.0013$ | $+0.0112\pm0.0015$ |
| $\epsilon^S_C(\text{eq})$ on the paired turns ($=$ sum of the two) | $0.0234\pm0.0014$ | $0.0110\pm0.0015$ |
| mismatch vs $\hat q$: $\frac1T\sum_t(p_t^\top\bar A_ty^B_t-p_t^\top\bar A_t\hat q_t)$ | $-0.0150\pm0.0013$ | $-0.0299\pm0.0020$ |

So: (a) the sampling noise in $\epsilon^S_C$ averages out (as it must, $y^B_t$ is predictable); (b) **the entire
$\epsilon^S_C(\text{eq})\approx0.02$ of a literal equilibrium opponent is model mismatch** — its equilibrium on its
information is 0.44 TV away from our equilibrium of its options, and the two sides' stage values disagree by
14–15 engine units on average (each side thinks it is ahead by more than the other concedes: $\bar g-\bar v$ is
positive on *both* sides in CSS, $+0.017$ and $+0.021$, which is impossible in a shared zero-sum game and is
consistent with the value disagreement: their sum 0.038 vs $-\overline{\rm val\,sum}=0.034$, on slightly different turn sets); (c) therefore **Foul Play's $\epsilon^S_C=0.0213$ matches the literal equilibrium
player's** ($0.0211\pm0.0011$; difference $+0.0003\pm0.0013$, which bounds it only to about $\pm12\%$ of the value),
so $\epsilon_C$ measured through our model reflects model mismatch and does not identify the opponent's regret. This
is a match on one statistic only: other statistics through the same model separate Foul Play from the eq player —
posterior weight on br\_unif 0.54 vs 0.09, br\_unif log loss 1.505 vs 1.647, $\hat q$ log loss 1.492 vs 1.580,
$\epsilon^{\rm dyn}_C$ 0.0408 vs 0.0369 (verification.md issue 1). It is not a lower bound either: a safex opponent
(CSS) sits *below* it (0.011–0.015) although its TV from our $q_{\rm eq}$ is the same (0.46 vs 0.44): it is closer in
*payoff*, not in TV, and TV is not the payoff metric; the mechanism is not identified here. Attributing the mismatch
to information asymmetry is an interpretation: each side's own sampling of $K=4$ worlds is an alternative that has
not been ruled out.

### 2.4 Calibration of $\hat q$ (Foster–Vohra hypothesis), LOG

Reliability of $\hat q_t(j)$ over all 242,861 (turn, column) pairs: ECE 0.015, but the curve is systematically
tilted — predicted 0.006 → realised 0.0014; 0.014 → 0.006; 0.032 → 0.019; 0.071 → 0.065; 0.14 → 0.17; 0.24 → 0.32;
0.38 → 0.45; 0.58 → 0.57; 0.79 → 0.66; 0.996 → 0.98. Mean predicted top-1 probability 0.443 vs top-1 hit rate 0.521
(excluding $m_t=1$ turns: 0.347 vs 0.438). $\hat q$ is *under-confident for favourites predicted below 0.5 and
over-confident at 0.7–1.0* (0.79 → 0.66, 0.996 → 0.98), and over-spread on the tail (the 5% uniform `MIX` and the
10%-range softmax temperature spread mass the opponent never uses). CEQ/CSS show the same shape (ECE 0.008–0.015, mean
predicted top-1 0.42–0.44 vs hit rate 0.52–0.55). So the calibration hypothesis of Foster–Vohra does not hold, mildly
(temperature / mix are tunable; not done here).

### 2.5 Realisability test of Proposition 3's oracle gap (CEQ, CSS)

On paired turns the opponent's true mixture $y^B_t$ is known, so Proposition 3's left side can be computed exactly:
the expected shortfall of the posterior player's $\epsilon$-safe reply $\hat x_t$ against the $\epsilon$-safe reply
$x^*_t$ to $y^B_t$ itself, $(x^*_t-\hat x_t)^\top\bar A_ty^B_t$. CEQ: $0.0012\pm0.0001$ per turn (median
$10^{-7}$; 16% of turns slightly negative: $\hat x_t$ was solved on the unrounded $\bar A_t$ and logged at 4 dp, $x^*_t$ on the 2-dp log, so the two $F_\epsilon$ slabs differ at logging precision); CSS: $0.0015\pm0.0001$. The Lipschitz bound
$\Delta_t\lVert\hat q_t-y^B_t\rVert_1$ holds on 100% of turns and averages $0.88$ — $700\times$ loose, because
$\lVert\hat q_t-y^B_t\rVert_1$ averages 1.21 (realisability fails: the posterior's prediction is far from the true
mixture) while the $\epsilon$-safe reply is insensitive to the prediction (its feasible set is a thin slab around the
equilibrium). **The oracle gap is small for a reason that has nothing to do with learning**: $\epsilon=0.003$ makes
every $\epsilon$-safe reply nearly the same strategy.

## 3. Posterior learners: what can be said

**Proposition 3 (payoff regret of the $\epsilon$-safe reply to the posterior, realisable case).** *Sketch; new
assembly of known parts.* Assume (R) a fixed true style $s^*\in S$ with $b_t\sim q_{s^*,t}$ independently given the
history, $\lambda=1$, uniform prior (T3.1 of `bayes_dynamics.md` §3), and (B) history-conditional Bhattacharyya affinities bounded uniformly,
$\sum_j\sqrt{q_{s,t}(j)q_{s^*,t}(j)}\le\rho_s<1$ for all $t$ with $m_t\ge2$ and $s\ne s^*$ (on $m_t=1$ turns all
styles coincide, so (B) cannot hold uniformly over all $t$: 2,553 such LOG turns, 580 CEQ) (`bayes_dynamics.md` §3 proves the bound in its
i.i.d.-matrix form (v); the uniform-conditional form used here is a one-line supermartingale extension that is not
written there). Let $F_\epsilon(\bar A_t)$ be the $\epsilon$-safe
set, $\hat x_t=\arg\max_{x\in F_\epsilon}x^\top\bar A_t\hat q_t$ and $x^*_t=\arg\max_{x\in F_\epsilon}x^\top\bar A_tq_{s^*,t}$.
Then
$$\sum_{t\le T}\mathbb E\big[(x^*_t-\hat x_t)^\top\bar A_tq_{s^*,t}\big]\ \le\ 2\Delta_{\max}\sum_{t\le T}\mathbb E[1-P_{t-1}(s^*)]\ \le\ 2\Delta_{\max}\sum_{s\ne s^*}\frac1{1-\rho_s},$$
a constant independent of $T$; hence the posterior player's expected regret against the oracle $\epsilon$-safe reply
— and against every fixed policy $\phi$ with $\phi_t\in F_\epsilon(\bar A_t)$, in particular the stage maximin
strategy — is $O(1/T)$. Pathwise, add Azuma: realised payoffs deviate from their conditional means by at most
$\Delta_{\max}\sqrt{2T\ln(1/\delta)}$ with probability $1-\delta$ (Lemma D of `dynamics_directions.md`).

*Proof idea.* Three-term split $x^{*\top}Aq^*-\hat x^\top Aq^*=[x^{*\top}A(q^*-\hat q)]+[x^{*\top}A\hat q-\hat x^\top A\hat q]+[\hat x^\top A(\hat q-q^*)]$;
the middle term is $\le0$ by optimality of $\hat x$ on $F_\epsilon$ (and $x^*\in F_\epsilon$); each outer term is
$\le\frac{\Delta_t}2\lVert\hat q_t-q_{s^*,t}\rVert_1$ (B5's Lipschitz step, `experts_extensions.md` §6 row 1); and
$\lVert\hat q_t-q_{s^*,t}\rVert_1=\lVert\sum_sP_{t-1}(s)(q_{s,t}-q_{s^*,t})\rVert_1\le2(1-P_{t-1}(s^*))$; T3 bounds
the sum. $\square$

*Status and relevance.* Ingredients known: B5 (LP sensitivity), T3 (Bhattacharyya concentration, device of
Walker 2004 / Zhang 2006), the "posterior concentrates ⇒ track the oracle" argument is Kalai–Lehrer-type with a
finite type space. The assembly is ours. **Hypothesis (R) is refuted on our data** for the eq opponent in both
prediction (T3 against RQ: $q_{\rm safe}$ out-predicts $q_{\rm eq}$) and payoff (§2.3: the eq opponent's true mixture
costs it 0.025 more than $q_{\rm eq}$ predicts). The most direct refutation: $\min_s\mathrm{TV}(y^B,q_s)=0.36$, and
against a literal eq opponent the posterior puts only 0.008 on eq ($\lVert\hat q-y^B\rVert_1=1.21$ is weaker
evidence). The *conclusion* (O(1) cumulative regret, a per-turn gap going to 0) is **not** observed: on CEQ the
per-turn oracle gap goes from 0.00137 (first half of each stream) to 0.00105 (second half), and the cumulative gap is
$4.18\bar\Delta$ over 3,501 turns, growing roughly linearly. What the data supports is a small per-turn gap of about
the width of the $\epsilon$-slab, for the trivial reason that $F_\epsilon$ is thin. So Proposition 3 is a correct
theorem whose hypotheses fail on our data and whose conclusion is not observed.

**Proposition 4 (Hedge wrapper; known).** hedge_bayes plays $p_t=\sum_kw_{k,t}q_{k,t}$ with $q_{\text{safex},t}=\hat x_t$,
weights from anytime Hedge on rewards $(q_{k,t}^\top M_{k_t}e_{j}-\min M_{k_t})/\bar\Delta$ clipped to $[0,1]$
(`exp3_live.py:55–66`). Cesa-Bianchi–Lugosi 2006 Thm 2.3 gives, pathwise and for every opponent, a per-stream regret bound
$\sqrt{2\ln4/T_s}+\sqrt{\ln4/8}/T_s$ with $T_s$ the stream's own horizon (Hedge is carried per stream): about 0.04
per turn at $T_s=1600$–$1940$. The guarantee holds on the learner's own reward (world-$k$ matrix $M_k$, clipped), not
on the $\bar A$-based $\epsilon^E_R$: 10.2% of LOG turns have $\operatorname{ptp}(\bar A_t)>400$ and 1.8% have a
clipped reward entry. (The code applies $\eta=\sqrt{8\ln4/t}$ before decision $t+1$, one step ahead of CBL's
$\eta_{t+1}$; this changes only the constant.) Consequently, by Definition 3, $\sigma_T$ is an
$(\epsilon^E_R,\epsilon^S_C)$-CCE of the estimated stage games *only with respect to the small comparator classes $E$
and $S$*, with $\epsilon^E_R$ measured (0.0044) and its worst-case guarantee weak ($\approx0.04$, larger than even the
measured dynamic regret 0.030); the posterior contributes only through the comparator class (its expert is in $E$). This is the only unconditional payoff statement available for
the posterior learner, and it is a statement about the Hedge player, not the posterior.

**Known results and whether they apply.**
- *Foster–Vohra 1997* (Games Econ. Behav. 21, "Calibrated learning and correlated equilibrium"): if every player
  best-responds to a calibrated forecast of the others' play, the empirical joint distribution converges to the set of
  correlated equilibria (one player alone: no internal regret). Needs calibration — which **does not hold** here (§2.4),
  mildly — and exact best response — we play the $\epsilon$-safe reply, which breaks the best-response step unless the
  best reply is already in $F_\epsilon$. Conditional form: "if $\hat q$ were calibrated and $\epsilon$ were 0, the
  posterior player's play in our model would have vanishing internal regret"; the calibration measurement is new.
- *Kalai–Lehrer 1993* (Econometrica 61): Bayesian learners with a grain of truth converge to Nash play; its
  hypotheses do not hold (no grain of truth — B12: the type space does not contain the other learner; a changing
  stage game; ε-safe rather than Bayes-optimal play); Nachbar 1997, Foster–Young 2001 impossibility.
- *Fudenberg–Levine 1995* (J. Econ. Dyn. Control 19, "Consistency and cautious fictitious play"): smooth fictitious
  play (perturbed best reply to empirical frequencies) is universally consistent (Hannan). Not ours: the posterior
  predicts with a 5-style model, not empirical frequencies, and the $\epsilon$-safe constraint is not a stochastic
  smoothing. Plain fictitious play is not no-regret (Lecture 6 §5), which is B12's point. The Hedge wrapper
  (Proposition 4) is the "cautious" ingredient that restores consistency, over experts rather than actions.
- *Hart–Mas-Colell 2000* regret matching → correlated equilibrium: a different algorithm; not ours.
- *Two $\epsilon$-safe players* (T4, `bayes_dynamics.md` §5): on a shared matrix every iterate is a $2\epsilon\Delta$-NE.
  CSS is its Showdown instance *without* a shared matrix: each side's run-mean exploitability in its own model is
  0.0013 (per turn the bound is $\epsilon\Delta_t$, and 53–54% of turns exceed 0.0015), but the two models disagree by 0.034 in value and each side's dynamic regret in the other's model is
  0.018–0.022, i.e. T4(e)'s mismatch term $\delta$ is $\approx0.02$ here, ten times $\epsilon\Delta$. Measured, not proved.

**Known vs new.** Definitions 1–3 and Propositions 1–2 are Lecture 7 §2.2/§3.2 with policy deviations (known in
form; the measurement along a real Showdown trajectory with the opponent's counterfactuals is new). Proposition 3
is a new assembly of B5 + T3 with a refuted hypothesis. Proposition 4 is CBL Thm 2.3 (applied per stream). The reference calibration of §2.3
(an equilibrium opponent on its own information has regret 0.021 in our model, almost all of it model mismatch) and
the log-loss-vs-payoff ranking difference of the styles are new measurements.

## 4. Correlation diagnostic

Categorical coding of $\sigma_T$: our action class $\in\{$top-only, safest-only, top=safest, other$\}$ (mass of
$p_t$ on each), the opponent's realised class $\in\{$argmax $q_{\rm eq}$, argmax $q_{\rm br\_unif}$, argmax $q_{\rm br\_us}$,
argmax $q_{\rm safe}$, other$\}$ (first match). $\mathrm{TV}(\sigma_T,\bar x\otimes\bar y)$: LOG 0.054, CEQ 0.071, against
a permutation null (opponent classes shuffled across turns) of $0.004$ / $0.008$ (95th pct 0.008 / 0.013),
$p<0.005$ (within-game permutation null: 0.006 / 0.012). **This null is uninformative.** Every stage is a product by
construction (simultaneous moves), and any time average of varying per-turn products is non-product, so the test
rejects a hypothesis nobody holds. It rejects even on CEQ, where the opponent is stateless and does not learn; there
$\sigma_T$ sits at TV 0.0094 from the logged per-turn product against a sampling null of 0.0091, which is tautological
for a stateless opponent but shows the test cannot detect learning-induced correlation. The meta-level TV is
therefore not evidence of anything learning-related. The within-node H4 probe (indistinguishable from product at
$10^3$–$10^4$ iterations) remains the only test of learning-induced correlation.

## 5. Ranked options for the papers

| # | option | 5-page paper | long report | cost / risk |
|---|---|---|---|---|
| 1 | **Replace the "not computable" paragraph** (`synthesis.md` §3.1) with the two-sided numbers: our $\epsilon^E_R=0.0044$, Foul Play's $\epsilon^S_C=0.021$ (eq style, 10/10 streams); one sentence that a literal equilibrium opponent shows the same 0.021, so it reflects model mismatch, not Foul Play's regret | one sentence: the "Allowed wording" of `verification.md` (our regret 0.0044 ± 0.0002 per turn vs the best expert; the opponent 0.021 ± 0.001 short of our equilibrium style in our estimated stage games; a literal equilibrium opponent shows the same gap) | new subsection with Table 2.1 and §2.3 | done; zero theory risk; numbers from existing LOG |
| 2 | **Reference calibration (CEQ/CSS)** as the interpretation key: $\epsilon_C$ through our estimated matrices is dominated by model mismatch (0.021 for a literal eq opponent, 0.011–0.015 for ε-safe ones), so it does not identify the opponent's regret | half a sentence | §2.3 table and the both-sides-above-value observation | done; needs the two 10-min runs to be archived |
| 3 | **Posterior learner: Proposition 4 + the honest Proposition 3** (theorem true, hypothesis refuted, conclusion not observed; small per-turn gap at $\epsilon=0.003$) with the calibration result for Foster–Vohra | one clause: "the posterior's payoff guarantee comes from the Hedge wrapper, not from the posterior" | Propositions 3–4, §2.2, §2.4, §2.5 | sketch-level; would need the `prove` workflow before an appendix |
| 4 | **Log loss vs payoff rank the styles differently** (safe wins prediction, eq wins payoff) | skip | one paragraph next to T3/RQ | done |
| 5 | **Meta-level correlation diagnostic** | skip | skip (the permutation null is uninformative, §4) | done; not evidence of anything learning-related |
| 6 | Extend LOG-type logging to a `style_eq` vs Foul Play or `bayes_safex` vs Foul Play run to put the posterior player's own two-sided numbers against the real opponent (not through Hedge) | — | optional | 1 CPU-h; low value given §2.5 |

What would change the ranking: if the paper's Lecture 7 paragraph already commits to "we never measure a two-sided
quantity above the node level", option 1 is mandatory (the sentence is now false); if space is the constraint,
options 1–2 are one sentence each and 3 is a clause.

## 6. Rejected / not supported

- *Fixed column labels across turns* as the opponent's deviation class: not well defined (labels are per-matchup).
- *"Foul Play is a no-regret learner in our game"*: not supported; what is measured is regret through our model, which matches a literal equilibrium opponent's and is dominated by model mismatch.
- *"The posterior learner is no-regret in payoff"*: not supported without Hedge (Proposition 4) or realisability (Proposition 3, refuted hypothesis).
- *Foster–Vohra applied as is*: calibration does not hold (§2.4) and the reply is $\epsilon$-safe, not best.
- *Learning-induced correlation at the meta level*: not identifiable from one sample per stage (§4).
- *Hedge's logged `r` as a closure check of the $\bar A$-based identity*: different matrix ($M_k$); the 0.00448 vs 0.00444 agreement is a consistency check only.
