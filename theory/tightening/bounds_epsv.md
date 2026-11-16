# Tightening the game-level bounds past $\varepsilon_v\approx1$: measured options, ranked

Brainstorm output (2026-11-16). Scripts: `theory/tightening/scripts/`; raw outputs in the session scratchpad
`.../scratchpad/tightening/` (`slack_small.json`/`.log` (two cases), `slack_garchomp.json`/`.log`, `neutral_mcts.jsonl`, `neutral_self.jsonl`,
`replay_sampled_{a,b,c}.jsonl`, `drift_summary.json`). Nothing outside `theory/tightening/` was edited.
Status words follow `theory_map.md` §0.4. Every number is tagged **measured** / **estimated** / **speculative**;
"payoff units" means the $[-1,1]$ scale, "range" means the matrix range $\Delta=2$ on that scale.

## 0. The one-paragraph diagnosis

Theorem A is a chain of two exact identities (L0 telescoping, L1(a)) and one inequality applied once per visited
node: Lemma 1, $\ell_t\le\varepsilon_s+2\lVert\hat Q_t-Q^*_t\rVert_\infty$. **All the vacuity is in that step**, and it has two
separable parts, which the exact endgames now let us measure (§2.1):

1. *sup over cells*: replacing $2\varepsilon_v$ by the error on the cells the two equilibrium profiles actually weight
   (Lemma 1′, §1.2) cuts the along-path budget by 3–6× on the endgames;
2. *signs*: the remaining slack is a cancellation. Along the best-response chain the value-side error
   $V^*-\operatorname{val}B$ and the guarantee-side error $x^\top(B-Q^*)\sigma$ are both large and of opposite sign
   (e.g. $+4.86$ and $-4.13$ summed over a 31-node chain, gholdengo vs heatran seat 2, exploitability 0.425). No
   absolute-value or sup-norm statement can see this; only the signed identity does.

On real mid-game play the same picture holds (§2.2–2.3, **measured**): the signed per-decision drift of the
calibrated depth-2 value is $\approx-0.012$ payoff per non-forced decision against *both* the white-box exploiter and
MCTS-2000 (0.6% of range, vs Theorem A's $2\varepsilon_v=2$ per decision), while the per-decision absolute drift
$\mathbb E|D_t|\approx0.04$–$0.056$ sums to 2.3–3.1 per game, above the range 2. So: **the path-average
*signed* residual explains the exploitability and is $\approx150$× smaller than $2\varepsilon_v$; every sup/absolute
restatement, including per-state and visited-state versions, stays vacuous.** The honest tightenings are therefore
(i) identities with one unmeasurable root term that symmetry removes, and (ii) Lemma 1′; not a smaller $\varepsilon_v$.

## 1. Ranked options

Ranking = (what it would say on our data) × (proof status) ÷ (cost). The first two are recommended for `Plan`.

| # | Option | Statement (short) | Proof status | On our data | Cost |
|---|---|---|---|---|---|
| 1 | **Signed path-residual identity, averaged over $s_0$** (§1.1) | $\mathbb E_{s_0}[V^*_0-V(\pi,\sigma)]=\mathbb E[\sum_t(R_t+S_t-D_t)]$ for every $\sigma$; with $\sigma=\sigma_{\rm BR}$ it is the average-case exploitability | identity: **proved** (pathwise algebra, checked to $10^{-14}$ on 284 games); root-term removal: **proved with named hypothesis** SYM | measured along the exploiter's path (60 games): drift channel $0.66\pm0.12$/game (sampled), per decision $-0.013$; along MCTS-2000 (100 games): $0.64\pm0.10$, $-0.011$; self-play (100): $-0.11\pm0.09$, $+0.002$ | none (the agent already computes $v_t$); one log field |
| 2 | **Lemma 1′ / Theorem A′: profile-restricted budget** (§1.2) | $\ell_t\le\varepsilon_s+x^{*\top}(Q^*-B)\hat y+\max_y x^\top(B-Q^*)y\le\varepsilon_s+2\varepsilon_v$ | **proved** (3 lines from L5.0; 2000/2000 random checks, 0 violations); composes through Theorem A's induction unchanged | exact endgames (7 rows): budget/exploitability 1.6–9.4× (vs 7.6–39× for Theorem A), 29× vs 89× on the 49-node chain | none to state; mid-game needs $Q^*$, so it is a theorem about *where the slack is*, not a mid-game number |
| 3 | **Theorem A backwards: necessary path-average $\varepsilon_v$** (§1.3) | $\mathbb E_{\pi,\sigma_{\rm BR}}[\overline{2\varepsilon_v}]\ge(\operatorname{expl}-\varepsilon_s\bar H)/\bar H$ | **proved** (arithmetic on Theorem A's trajectory form) | path-average $\varepsilon_v\ge0.0067$ payoff per decision (point, $0.426/(2\cdot32)$), $\ge0.0028$ (R5a LCB 0.18): a leaf 100× better than $\varepsilon_v=1$ in sup norm would still allow all of the measured exploitability | none |
| 4 | **Comparative identity with the continuation value** (§1.4) | $V(\pi_2,\sigma)-V(\pi_1,\sigma)=\mathbb E_{\pi_1,\sigma}\sum_t(x_{2,t}-x_{1,t})^\top Q^{\pi_2,\sigma}_t\sigma_t$; $\operatorname{expl}(\pi_1)-\operatorname{expl}(\pi_2)\le$ this at $\sigma=\sigma_{\rm BR}(\pi_1)$; the leaf enters only through $\lVert x_1-x_2\rVert_1\cdot$ row-oscillation of $(Q^{\pi_2,\sigma}-\hat Q)\sigma$ | **proved** (P1(i) + one inequality); the naive "common $\varepsilon_v$ cancels" version is **false** (counterexample + B12) | not run; the oscillation term is a policy-evaluation quantity (rollouts, no minimax) | 1–4 CPU-h of rollouts for one pair of agents (estimated) |
| 5 | **Typical-state / visited-state $\varepsilon_v$** (direction 3 as stated) | Theorem A's trajectory form with $\varepsilon_v(t,s)$ on reached states | already in Theorem A | exact endgames: reach-mean $\varepsilon_v^{\rm cal}$ 0.16–0.49 vs max 1.0, but the reach-weighted budget is still 9–39× the exploitability (`refined_bound`) | none; **insufficient alone** |
| 6 | **Opponent-class ("restricted") exploitability** (direction 5b) | $\operatorname{expl}_\Sigma(\pi)=\sup_{\sigma\in\Sigma}[V^*-V(\pi,\sigma)]$ with the option-1 identity per $\sigma$ | definitional; identity proved | vs MCTS-2000: $0.11\pm0.10$ payoff; vs wb2: 0.426 (W1); the identity splits each into stage / sampling / drift channels | none beyond logging |
| 7 | **Calibrated or learned leaf** (direction 4) | the leaf's figure of merit is the signed drift along adversarial paths (= minimax TD residual with adversarial visitation, L6 Rem. 2), not $\varepsilon_v$ | L6 **proved**; training is empirical | sup $\varepsilon_v$ stays $\approx1$ for every candidate (RESULTS:546); exact terminals cut mean $\varepsilon_v^{\rm cal}$ 3× and exploitability to $\approx0$ on 2/3 endgames but not the balanced one; the value net's residual is above tanh's | high (training), uncertain |
| 8 | **Depth-$d$ vs $d{+}1$ consistency as the mid-game residual** (direction 1b) | $D_t$ in option 1 *is* the depth-1→2 deepening correction at the realised child (expectation over chance) | identity | measured (option 1 numbers); a depth 2→3 version is proposed, not run | 2→3: $\approx3$–4 s per state, 500 states ≈ 30 min CPU (estimated) |

Rejected (one line each):
- *Deeper search or self-play rollouts as a $V^*$ reference.* Lemma L5: mid-game $\rho_D\approx1$, depth does not
  contract the leaf error; rollouts of $(\pi,\sigma)$ estimate $V^{\pi,\sigma}$, which is neither above nor below $V^*$.
- *Naive comparative bound where $\varepsilon_v$ cancels.* False: two agents $\varepsilon_s$-optimal in the same
  $\hat Q$ can differ by $2\varepsilon_v$ in stage loss (column-only error $\hat Q=I$, $Q=\hat Q+\mathbf 1(0,-10)$: losses
  0.5 vs 0); and B12 shows the drift channel differs across agents sharing $\hat Q$ (top $-0.08\pm0.25$, sampled $+0.66\pm0.12$).
- *Mean instead of sup $\varepsilon_v$.* Still 9–39× vacuous on endgames (option 5).
- *Absolute along-path residual $\sum|D_t|$.* Measured 2.3–3.1 per game $>$ range 2: vacuous on every path.
- *Global affine calibration.* $a=0$ on 6/6 endgames (R6b); E2′ does not apply.
- *Fixing the terminal inversions to shrink the real-play loss.* The last-decision drift (terminal mis-scoring) is
  $-0.13$ to $+0.03$ per game, i.e. the mid-game deepening drift, not the terminal scoring, carries the loss (§2.2).

### 1.1 Option 1: the signed path-residual identity (Theorem A-res)

*Setting.* Our agent plays $x_t$, $\varepsilon_s$-optimal in the depth-2 matrix $\hat A_t$ (engine units); $v_t=\operatorname{val}\hat A_t$;
$\varphi:\mathbb R\to[-1,1]$ any odd monotone map (we use the logistic $2\operatorname{lgs}(v/98)-1$ fitted on the B12 games and
$\tanh(v/400)$); $z\in\{-1,0,1\}$ the outcome; at a non-forced decision the opponent plays column $j_t$, we play row
$i_t$. Define, with $\varphi(v_{T+1}):=z$,
$$R_t=\varphi(v_t)-\varphi((x_t\hat A_t)_{j_t}),\qquad S_t=\varphi((x_t\hat A_t)_{j_t})-\varphi(\hat A_t[i_t,j_t]),\qquad D_t=\varphi(v_{t+1})-\varphi(\hat A_t[i_t,j_t]).$$

**Proposition (pathwise identity).** On every path, $z-\varphi(v_1)=\sum_t(-R_t-S_t+D_t)$. *Proof.* Telescoping:
the three terms sum to $\varphi(v_{t+1})-\varphi(v_t)$. $\square$ (Checked to $10^{-14}$ on all 284 logged games: 36 B12 + 48 extension + 100 MCTS + 100 self-play.)

**Theorem A-res.** Under A1(i), A6, for every history-dependent $\sigma$:
$$V^*_0(s_0)-V(\pi,\sigma)=\big[V^*_0(s_0)-\varphi(v_1)\big]+\mathbb E_{\pi,\sigma}\Big[\sum_t(R_t+S_t-D_t)\Big].$$
With $\sigma=\sigma_{\rm BR}$ the left side is $\operatorname{expl}(\pi;s_0)$. Under
**(SYM)** — the team generator is side-symmetric and $V^*_0$ and $\varphi\circ v_1$ are antisymmetric under the side
swap — $\mathbb E_{s_0}[V^*_0-\varphi(v_1)]=0$ and
$$\mathbb E_{s_0}\operatorname{expl}(\pi;s_0)=\mathbb E_{s_0}\mathbb E_{\pi,\sigma_{\rm BR}}\Big[\sum_t(R_t+S_t-D_t)\Big].$$
*Proof.* $\mathbb E_{\pi,\sigma}[z]=V(\pi,\sigma)$ (payoff units); take expectations in the pathwise identity and subtract
from $V^*_0$. SYM: $\mathbb E_{s_0}V^*_0=0$ by zero-sum symmetry,
$\mathbb E_{s_0}\varphi(v_1)=0$ by antisymmetry of the mid leaf ($\operatorname{val}$ of the mirrored root is $-\operatorname{val}$ to
$10^{-14}$ on a test state; $\varphi$ odd). $\square$ Status: identity **proved**; SYM is a **named hypothesis**
(speed ties, the draw convention and the generator's side asymmetries are the known violations, all measure-small;
$\mathbb E\varphi(v_1)$ is directly measurable and was $+0.000$ to $+0.004$ on the seat-balanced logs). *Sharper:* on
seat-paired logs (W1 is paired; the neutral runs alternate seats by seed) the root term cancels *exactly* across each
pair under antisymmetry of $V^*$ and of the mid leaf, so SYM reduces to the engine's own side asymmetries (speed
ties, draw convention), not to an assumption on the team generator.

*Reading.* $R_t$ is the opponent's realised stage gain in $\hat A_t$; $R_t\le$ the stage gap, which is $\approx0$ for RM+
play, and the measured $\sum R$ is *negative* ($-0.26$ to $-0.43$ per game) because the opponent's column is not the
$\hat A_t$-minimiser against our mixture. $S_t$ is our own sampling term (mean 0 for affine $\varphi$; a Jensen term
otherwise, measured $-0.06$ to $-0.13$). $-D_t$ is the **drift**: the one-step Bellman residual of $\varphi\circ v$ along
the retained (non-forced) decision sequence — the calibrated depth-2 value at the next decision minus the depth-1
cell that predicted it. For affine $\varphi$ and no forced node in between, $\mathbb E[D_t\mid s_t,i_t,j_t]$ is exactly the
depth-1→2 deepening correction at the children of the played cell; so option 1 is direction 1b (Bellman consistency
along the trajectory) made exact, up to the Jensen and forced-node terms, which the pathwise identity absorbs.
What Theorem A charges to $2\varepsilon_v$ per decision is here a measured signed quantity.

*What it does not do.* It is not a certified upper bound on $\operatorname{expl}$: the sup over $\sigma$ is not computable,
and the logged exploiter is a lower-bound proxy for $\sigma_{\rm BR}$. It gives (a) an exact decomposition of every
measured exploitation, (b) under SYM an average-case identity whose right side is measurable *for the true BR* in
principle, and (c) the right per-decision quantity to report in place of $2\varepsilon_v$.

### 1.2 Option 2: Lemma 1′ (profile-restricted one-step lemma) and Theorem A′

**Lemma 1′.** Let $Q^*,B\in\mathbb R^{n\times m}$, $x$ $\varepsilon_s$-optimal in $B$, $x^*$ maximin in $Q^*$, $\hat y$ minimax in $B$. Then
$$\operatorname{val}Q^*-\min_yx^\top Q^*y\ \le\ \varepsilon_s+x^{*\top}(Q^*-B)\hat y+\max_yx^\top(B-Q^*)y\ \le\ \varepsilon_s+2\lVert B-Q^*\rVert_\infty .$$
*Proof.* Split $\operatorname{val}Q^*-\min_yx^\top Q^*y=[\operatorname{val}Q^*-\operatorname{val}B]+[\operatorname{val}B-\min_yx^\top By]+[\min_yx^\top By-\min_yx^\top Q^*y]$.
The first bracket is $\le x^{*\top}(Q^*-B)\hat y$ by L5.0 (`search_rl_links.md`, proved); the second is $\le\varepsilon_s$; for the
third, with $y^\circ$ a minimiser of $x^\top Q^*y$, $\min_yx^\top By\le x^\top By^\circ$, so it is $\le x^\top(B-Q^*)y^\circ\le\max_yx^\top(B-Q^*)y$.
Each middle term is bounded by $\lVert B-Q^*\rVert_\infty$. $\square$ Numeric: 2000 random $(Q^*,E)$ with dense, row-only,
column-only and single-cell errors, 0 violations of either inequality; median ratio of the new budget to the old
0.16 (`scripts/check_lemma1_profile.py`).

**Theorem A′.** Theorem A holds with the per-state budget $\delta'(t,s)=\varepsilon_s+x^{*\top}(Q^*-B)\hat y+\max_yx^\top(B-Q^*)y$,
$B=a_{t,s}\hat Q+c_{t,s}$ any per-state calibration (R6a). *Proof.* Theorem A's induction applies Lemma 1 one node at a
time; substitute Lemma 1′. $\square$ **Proved.**

*Exact three-term split (used in §2.1).* For the actual column $\sigma$: $V^*-x^\top Q^*\sigma=T_1+T_2+T_3$ with
$T_1=V^*-\operatorname{val}B$, $T_2=\operatorname{val}B-x^\top B\sigma$ ($\le\varepsilon_s$), $T_3=x^\top(B-Q^*)\sigma$. $T_1$ and $T_3$ are the
two faces of the leaf error (value side, guarantee side); Lemma 1 bounds each by $\varepsilon_v$ *separately*; on the data
they cancel.

*What it buys.* On the endgames (§2.1) the along-chain budget falls from 9–39× the exploitability to 1.6–9.4×.
It cannot be evaluated mid-game (needs $Q^*$), so its role is to say precisely which part of Lemma 1 is loose.
Direction for a mid-game proxy: the guarantee-side term $\max_yx^\top(B-Q^*)y$ only involves our own mixture's rows —
a one-sided, support-restricted error that a leaf could be trained to keep small (option 7).

### 1.3 Option 3: Theorem A backwards

From Theorem A's trajectory form, $\operatorname{expl}(\pi)\le\mathbb E_{\pi,\sigma_{\rm BR}}[\sum_t(\varepsilon_s+2\varepsilon_v(t,s_t))]$, so the
BR-path average of $2\varepsilon_v(t,s)$ is at least $(\operatorname{expl}-\varepsilon_s\bar H)/\bar H$. With $\varepsilon_s\approx0$ (RM+),
$\operatorname{expl}\ge0.426$ (W1 point) or $\ge0.18$ (R5a LCB, payoff) and $\bar H=32$ non-forced decisions (median over the
60 replayed W1 sampled-target games; the mean, 52, is inflated by 3 capped games): path-average
$\varepsilon_v\ge0.0067$ (point) / $0.0028$ (LCB) payoff — 0.3%/0.14% of range. **Proved**, honest, and weak: it says a
leaf with sup error 0.006 would already permit the whole measured exploitability. Its use is rhetorical: the
sup-norm number 1.0 is $\approx150$× larger than anything the data require, so lowering $\varepsilon_v$ is not the lever.

### 1.4 Option 4: comparative statements that are actually true

Direction 2 asked whether Theorems A/C can be restated for the *difference* between two agents on the same leaf so
that $\varepsilon_v$ cancels. **Not in general** (rejected list). What holds:

**Proposition (comparison against a fixed opponent; P1(i) restated).** Under A1(i), A6, for Markov $\pi_1,\pi_2$ and any
$\sigma$: $V(\pi_2,\sigma)-V(\pi_1,\sigma)=\mathbb E_{\pi_1,\sigma}\big[\sum_t(x_{2,t}-x_{1,t})^\top Q^{\pi_2,\sigma}_t\sigma_t\big]$, where
$Q^{\pi_2,\sigma}$ is the continuation matrix under $(\pi_2,\sigma)$. Hence
$\operatorname{expl}(\pi_1)-\operatorname{expl}(\pi_2)\le V(\pi_2,\sigma_{\rm BR}(\pi_1))-V(\pi_1,\sigma_{\rm BR}(\pi_1))$. Writing
$Q^{\pi_2,\sigma}=\hat Q+E$: the summand is $(x_2-x_1)^\top\hat Q\sigma$ (a logged $\hat Q$-stage quantity) plus
$(x_2-x_1)^\top E\sigma\le\tfrac12\lVert x_1-x_2\rVert_1\operatorname{osc}_{\rm rows}(E\sigma)$ — the leaf error enters only through
its **row variation on the support difference**, against the actual column. *Proof.* L0 with potential $V^{\pi_2,\sigma}$
(P1(i)); $\operatorname{expl}(\pi_1)-\operatorname{expl}(\pi_2)=V(\pi_2,\sigma_{\rm BR2})-V(\pi_1,\sigma_{\rm BR1})\le V(\pi_2,\sigma_{\rm BR1})-V(\pi_1,\sigma_{\rm BR1})$;
Hölder. $\square$ **Proved.**

*Caveats.* (i) $E$ is the error relative to the *matchup's continuation value*, not $Q^*$ — a different unknown, but one
that policy-evaluation rollouts estimate without any minimax solve. (ii) For Hedge vs its experts P1(iv) still
applies (external regret is not policy regret); the continuation-mismatch term was measured at an order of magnitude
below the external term on engine logs (`theory_map.md` P1 numeric), which is the quantitative content behind
"E2 $=$ AF" (synthesis §3.4). (iii) Cost of measuring the oscillation term: rollouts of $(\pi_2,\sigma)$ from the
children of logged decisions; at $\approx3$–15 s per rollout, 100 decisions × 10 rollouts ≈ 1–4 CPU-h (estimated).

### 1.5 Theorem C (hidden information) and the Hedge-level results

The pathwise identity of §1.1 holds verbatim at M3 with $v_t$ the agent-form root value (`agent_form_lp`) and
$\hat A_t[i_t,j_t]$ the cell of the scoring world: Theorem C's $2\varepsilon_v$ term gets the same restatement, and its
hidden-information terms ($2\mathbb ES_t$, $2\Delta\,\mathrm{TV}$, $\eta_{\rm NL}$) all sit *inside* the drift channel
$-\sum D$ — the identity does not separate them. `results/hidden/E5_decisions.jsonl` logs `v_AF` but not the realised
cell, so one more logged field per decision makes the channel split available on Showdown at no compute cost. Two
caveats: under hidden information the root value depends on each seat's belief, so antisymmetry of $\varphi(v_1)$ across
seats is only approximate and SYM is weaker (measure $\mathbb E\varphi(v_1)$ directly); and the scoring world is chosen
after seeing $b_t$ (B6.1), which is harmless for a pathwise identity. For the Hedge layer (L2, E2′, B6): the identity
sidesteps the global-calibration requirement for the *channel decomposition* (any $\varphi$, per path), but converting
Hedge's regret from reward units to payoff still needs a global $a>0$ (E2′'s (B2)), which the endgames refute; so the
Hedge statements become "surrogate identities plus a measured drift channel", not payoff bounds. No new experiments.

## 2. Experiments run (all local, `engine/.venv/bin/python`; wall ≈ 30 min, CPU ≈ 60–70 min across workers, **estimated**)

### 2.1 Where Theorem A's slack lives on the exact endgames — `scripts/endgame_slack.py`

Reuses `engine/endgame_eps.py` unchanged (enumeration, Shapley, per-state `full_matrix` + per-state affine
calibration, exact best response, BR chain with the longest tie-break = `refined_bound`'s chain). Along the chain it
sums, for "equilibrium sampled" (and top move), the true loss $\ell_t$ (L1(a); sums to the exact exploitability — the
code check passed on every row to $10^{-4}$), the stage exploitability $e_t$ in $Q^*$ (L1(b) upper sandwich), the
Theorem A budget $\varepsilon_s+2\varepsilon_v^{\rm cal}$ (= `refined_bound`), the Lemma 1′ budget, and the split $T_1+T_2+T_3$.
Output: `scratchpad/tightening/slack_small.{json,log}`, `slack_garchomp.{json,log}`; the `expl` and Theorem A
budget columns reproduce `results/endgame/eps.jsonl` (`expl`, `refined_bound`) to 3 decimals on every row. **Measured** (abstracted endgames, PP* and HP buckets as in
RESULTS "Exact endgames"):

| Case (seat, agent) | expl | $\bar H$ | $\sum e_t$ | Thm A budget | Lemma 1′ budget | $T_1$ | $T_2$ | $T_3$ | $\varepsilon_v^{\rm cal}$ reach max / mean |
|---|---|---|---|---|---|---|---|---|---|
| greattusk vs gholdengo 100% (1, sampled) | 0.215 | 1.1 | 0.215 | 2.000 | 0.335 | +0.80 | 0.00 | −0.58 | 1.00 / 0.49 |
| same (2, sampled) | 0.120 | 1.0 | 0.120 | 2.000 | 0.335 | −0.80 | 0.00 | +0.92 | 1.00 / 1.00 |
| gholdengo vs heatran 100% (1, sampled) | 0.074 | 2.0 | 0.074 | 1.583 | 0.522 | −0.40 | −0.21 | +0.69 | 0.96 / 0.45 |
| same (2, sampled) | 0.425 | 31.4 | 0.512 | 16.42 | 4.01 | +4.86 | −0.30 | −4.13 | 1.00 / 0.16 |
| same (2, top move) | 0.426 | 49.2 | 0.512 | 37.93 | 12.45 | +14.98 | −0.30 | −14.25 | 1.00 / 0.21 |
| garchomp vs rotomwash 100% (1, sampled) | 0.338 | 3.2 | 0.338 | 4.498 | 0.912 | −0.46 | −0.05 | +0.85 | 1.00 / 0.44 |
| same (2, sampled) | 0.602 | 3.0 | 0.602 | 4.583 | 1.088 | −0.22 | −0.01 | +0.83 | 0.92 / 0.45 |
| same (1, top move) | 0.356 | 3.3 | 0.491 | 4.675 | 1.064 | −0.41 | +0.05 | +0.71 | 0.78 / 0.49 |

Reading: (a) Lemma 1′ removes 3–6× of the slack (budget/expl 9.3→1.6, 16.7→2.8, 21→7.0, 39→9.4, 89→29, 13.3→2.7, 7.6→1.8); on the
three short chains it is within a factor 1.6–2.8 of the exact exploitability; (b) what remains is the
signed cancellation $T_1\approx-T_3$: the value-side and guarantee-side errors are each large (the leaf is far from
$Q^*$ at every reached state) and nearly cancel along the path; (c) the L1(b) sandwich $\sum e_t$ is within 20% of the
exploitability, so the right per-state quantity is the *stage exploitability of our mixture in the true matrix*,
not any norm of the matrix error; (d) the visited-state mean $\varepsilon_v^{\rm cal}$ (0.16–0.49) is far below the max
but the mean-form budget is still vacuous (option 5).

### 2.2 Drift along adversarial vs neutral paths — `scripts/drift_analysis.py`, `scripts/neutral_drift.py`, `scripts/replay_more.py`

Data: (i) the B12 replay (`scratchpad/bridges/b12/replay.jsonl`, 36 games, white-box exploiter wb2 vs three targets,
per-turn $\hat A_t,x_t,v_t,i_t,j_t$; replays reproduce the logged games exactly); (ii) extension of the sampled-target
replay to W1 pairs 6–29 (48 more games; `replay_more.py` wraps the B12 script unchanged; 2 of 48 diverge from the
logged winner/length, the known cross-run replay issue of RESULTS:428, and are kept as valid exploiter games); (iii) 100 new games of
the sampled target against MCTS-2000 (`neutral_drift.py`, seat alternating, seeds 8000–8099); (iv) 100 self-play
games (seeds 9000–9099). Forced decisions (one option) are excluded from per-decision statistics. Both $\varphi$ maps
reported; the identity closes to $10^{-14}$ on every game. **Measured**:

| Path (target = equilibrium sampled) | games | score | $-\mathbb Ez$ | $\sum R$ | $\sum S$ | $-\sum D$ (SE) | mid / last | per-decision $\mathbb ED$ | $\mathbb E|D|$ | $\sum|D|$/game | decisions/game |
|---|---|---|---|---|---|---|---|---|---|---|---|
| wb2 exploiter, logit | 12 | 0.375 | 0.250 | −0.39 | −0.28 | +0.91 (0.31) | +1.05 / −0.13 | −0.0127 | 0.040 | 2.86 | 72 |
| wb2 exploiter, tanh400 | 12 | 0.375 | 0.250 | −0.46 | −0.13 | +0.84 (0.34) | +0.92 / −0.08 | −0.0117 | 0.032 | 2.33 | 72 |
| wb2 exploiter, pairs 0–29, logit | 60 | 0.358 | 0.283 (0.12) | −0.26 | −0.13 | +0.66 (0.12) | +0.71 / −0.05 | −0.0127 | 0.056 | 2.91 | 52 |
| wb2 exploiter, pairs 0–29, tanh400 | 60 | 0.358 | 0.283 (0.12) | −0.21 | −0.06 | +0.55 (0.12) | +0.55 / +0.00 | −0.0106 | 0.042 | 2.16 | 52 |
| MCTS-2000, logit | 100 | 0.445 | 0.110 (0.10) | −0.43 | −0.10 | +0.64 (0.10) | +0.63 / +0.01 | −0.0113 | 0.054 | 3.06 | 57 |
| MCTS-2000, tanh400 | 100 | 0.445 | 0.110 (0.10) | −0.30 | −0.05 | +0.45 (0.10) | +0.41 / +0.04 | −0.0080 | 0.040 | 2.27 | 57 |
| self-play, logit | 100 | 0.525 | −0.050 | 0.00 | +0.06 | −0.11 (0.09) | −0.11 / −0.01 | +0.0020 | 0.057 | 3.18 | 56 |
| self-play, tanh400 | 100 | 0.525 | −0.050 | 0.00 | +0.03 | −0.08 (0.09) | −0.06 / −0.02 | +0.0015 | 0.042 | 2.34 | 56 |

For the other B12 targets (12 games each, logit, same convention): top move $-\sum D=-0.08\pm0.25$ with $\sum R=+0.92$
(its loss is the stage channel; 0/597 mixed decisions); safex 2% $-\sum D=+1.14\pm0.76$, $\sum R=-0.55$ (3 of 12 games
capped at 300 turns, hence the SE).

Reading (sampled target): (a) the exploitation runs through a **signed** drift of $\approx-0.012$ payoff per
non-forced decision — against the exploiter *and* against MCTS-2000, i.e. any competent opponent steers play to the
children where one more ply of search lowers our value; (b) $\sum|D|$ is 2.3–3.1 per game $>$ range: the
absolute along-path residual is vacuous, exactly like L6's sup form (median 4.2 in the earlier probe); (c) the
terminal-scoring part (last decision) is $-0.13$ to $+0.03$ per game: on real paths the loss is mid-game drift, not the
terminal inversions that drive $\varepsilon_v^{\rm cal}\approx1$ on the endgames; (d) the $\varphi$ dependence is material
for the channel split (logit vs tanh move $\sum R$ and $-\sum D$ by 0.1–0.2) but not for the conclusions above.

### 2.3 Self-play control (by symmetry $\mathbb Ez=0$, so the identity forces $\sum R+\sum S=\sum D$)

100 games, sampled vs sampled, seat alternating (`neutral_self.jsonl`). **Measured**: $-\sum D=-0.11\pm0.09$ per game
(logit; $-0.08\pm0.09$ tanh), per-decision $\mathbb ED=+0.002$, $\mathbb E\varphi(v_1)=+0.004$ — consistent with SYM and
with zero signed drift, while $\mathbb E|D|=0.057$ and $\sum|D|=3.2$ per game are the *same* as on the adversarial paths.
So the absolute residual is a property of the leaf everywhere; the signed drift ($-0.012$ per decision) is created by
the opponent's selection of children and is absent in symmetric play. $\sum R=0$ exactly: against a column drawn from
the same matrix's equilibrium, the realised stage gain vanishes (support property of the equilibrium), as it should.

### 2.4 Lemma 1′ numeric check — `scripts/check_lemma1_profile.py`

2000 random instances; 0 violations of $\ell\le$ Lemma 1′ budget $\le$ Lemma 1 budget; the exact three-term split
holds to $10^{-9}$; budget ratio new/old: median 0.16, 90th percentile 0.62, max 0.998 (**measured**, synthetic).

## 3. What would change the ranking

- Self-play drift came out $\approx0$ ($-0.11\pm0.09$), as SYM requires; had it not, option 1's root-term removal
  would have been in doubt.
- The extended exploiter replay (60 games) gives the same per-decision drift as MCTS-2000 ($-0.0127$ vs $-0.0119$):
  the drift is what any competent opponent extracts, not a white-box artefact; the exploiter's extra edge over MCTS on
  the full W1 log (0.426 vs 0.11) is not resolved by these 60 games ($0.283\pm0.12$) and must come from more decisions
  per game and the stage/sampling channels, which this sample cannot split.
- If a depth-2→3 deepening residual (option 8, not run) is as large as the 1→2 one, the drift is a property of the
  heuristic, not of shallow depth, and the only leaf lever is learning (option 7).

## 4. Recommendation

Take to `Plan`: **option 1** (restate B12 as Theorem A-res; add $v_t$, $\hat A_t[i_t,j_t]$ to the per-turn log of every
engine run so the drift channel is a standard column; report per-decision signed drift and $\sum|D|$ side by side
with $2\varepsilon_v$) and **option 2** (Lemma 1′/Theorem A′ as the replacement for Lemma 1 in the report, with the
endgame slack table as its evidence). Option 3 as one sentence in the report's limitations. Options 4 and 8 are the
next experiments if CPU is available; option 7 only with the drift as its training metric.
