# Theory: the live experts learner (Hedge + fixed share over stage-game experts)

Companion to `theory_section.md` (Lemma 1, Theorem 1) and `full_game.md` (§1 model, §3 proof of
Theorem A). Learner: `engine/exp3_live.py` (`Learner.plan`, `Learner.update`). Numeric check:
`checks/hedge_check.py`. Lean: `lean/FullGame/Hedge.lean`.

## Setting

Same game as `full_game.md` §1: finite horizon $H$, simultaneous moves, chance, terminal payoff in
$[-1,1]$, values $V^*_t$, true stage matrices $Q^*_t(s)$, payoff range $\Delta=2$. A **match** is $N$
games played in sequence. The learner's weights carry over from one game to the next.
Game $g$ lasts $\tau_g$ turns (as in Theorem 1). $S\le\sum_g\tau_g$ is the number of turns, over
the whole match, on which the learner has more than one option. Only these turns update the weights
(`exp3_live.py` `Learner.update`), and all regret sums run over them.

**Learner.** At turn $t$ it builds $\hat A_t=\hat Q_t(s_t)$ (`nash.full_matrix`, own
perspective) and $k=3$ expert strategies $q_{e,t}$, all computed from $\hat A_t$:

- the top move of the RM+ strategy;
- the RM+ strategy $x_t$ itself (1000 iterations);
- the safest pure row $\arg\max_i\min_j\hat A_t[i,j]$.

It plays $p_t=\sum_e w_{t,e}q_{e,t}$. The opponent's move $b_t$ may depend on everything public,
including $p_t$, but not on the learner's sampled action $a_t\sim p_t$. Rewards are
$$r_{t,e}=\frac{q_{e,t}\cdot\hat A_t[:,b_t]-\min\hat A_t}{\Delta_t},\qquad\Delta_t=\operatorname{range}(\hat A_t),$$
and the update is Hedge plus fixed share: $v\propto w_t e^{\eta r_t}$, then $w_{t+1}=(1-\alpha)v+\alpha/k$.
Code defaults: scale $=400$ (fixed normaliser), anytime $\eta_t=\sqrt{8\ln k/t}$ in FTRL form (`logw = η_t · cumulative reward`), $\alpha=0$. The X6 setting is `--eta 0.2 --alpha 0.002 --scale 0`. The first anytime implementation (`logw += η_t r_t`) is outside Lemma H and has linear regret. Counterexample: two experts, expert 0 rewarded for the first 30% of turns and expert 1 afterwards, gives regret $0.40\,T$. This is now fixed. **Variant (fixed normaliser):** the same learner with
$r_{t,e}=(q_{e,t}\cdot\hat A_t[:,b_t]-\ell)/\bar\Delta$, where every $\hat A_t$ has its entries in
$[\ell,\ell+\bar\Delta]$.

| Symbol | Meaning |
|---|---|
| $\varepsilon_s,\varepsilon_v$ | as in Theorem 1: solver gap in $\hat A_t$, and $\lVert\hat A_t-Q^*_t\rVert_\infty$ |
| $g_t$ | guarantee of the equilibrium expert: $\min_b q^*_t\cdot\hat A_t[:,b]\ \ge\ \operatorname{val}(\hat A_t)-\varepsilon_s$ |
| $\mathrm{Reg}_S$ | regret of the weights, in **normalised** reward units (Lemma H) |
| $\bar H_\sigma$ | mean game length in the match against opponent $\sigma$: $\mathbb E_\sigma[\sum_g\tau_g]/N$ (all live turns) |

**Hypothesis (E).** At every turn one expert $e^*$ is an $\varepsilon_s$-equilibrium strategy of
$\hat A_t$. In the code this is the RM+ expert, with $\varepsilon_s$ from Proposition 2. The top-move
and safest experts are not $\varepsilon_s$-optimal in general.

## Results

**Lemma H (Hedge + fixed share, pathwise).** Fix any sequence of reward vectors $r_t\in[0,1]^k$, where
$r_t$ may depend on $w_1,\dots,w_t$ (adaptive opponent). For every expert $e$:
$$\sum_{t\le S}\big(r_{t,e}-w_t\cdot r_t\big)\;\le\;\frac{\ln k}{\eta}+\frac{\eta S}{8}+\frac{S\ln\frac1{1-\alpha}}{\eta}=:\mathrm{Reg}_S.$$
With $\alpha=0$ and $\eta=\sqrt{8\ln k/S}$: $\mathrm{Reg}_S\le\sqrt{S\ln k/2}$. Against a comparator
that switches experts $m$ times, the bound is
$\big[\ln k+m\ln\frac k\alpha+(S-m)\ln\frac1{1-\alpha}\big]/\eta+\eta S/8$ (shifting regret).

*Proof.* Let $L_t=\ln w_{t,e}$ and $Z_t=\sum_{e'}w_{t,e'}e^{\eta r_{t,e'}}$. Hoeffding's lemma gives
$\ln Z_t\le\eta\,w_t\cdot r_t+\eta^2/8$. The share step gives $w_{t+1,e}\ge(1-\alpha)w_{t,e}e^{\eta r_{t,e}}/Z_t$. So
$L_{t+1}\ge L_t+\eta(r_{t,e}-w_t\cdot r_t)-\eta^2/8-\ln\frac1{1-\alpha}$. Sum over $t$, using
$L_1=-\ln k$ and $L_{S+1}\le0$. For the shifting bound, at a switch $e\to e'$ use
$w_{t+1,e'}\ge\alpha/k\ge(\alpha/k)\,v_e$ instead. The argument is pathwise and $w_t$ is fixed before
$r_t$ is revealed, so an adaptive opponent changes nothing. $\square$

The constants are derived here (cf. Freund & Schapire 1997; Cesa-Bianchi & Lugosi 2006, Thm 2.2;
Herbster & Warmuth 1998). **For the code's parameters** ($k=3$, $\eta=0.2$, $\alpha=0.002$):
$\mathrm{Reg}_S\le5.49+0.035\,S$. This grows linearly in $S$: $3.5\%$ of each turn's range, of which
$0.025$ is $\eta/8$ and $0.010$ is the share term. Each switch of the comparator costs
$\ln(k/\alpha)/\eta\approx36.6$.

**Proposition E1 (safety; Claim 1).** Assume (E). Against any opponent, for every realisation
(sums over the $S$ non-forced turns; a turn with constant $\hat A_t$ has $\varepsilon_s=0$ and
gives every expert reward 0, and in (a) it is dropped, or $\Delta_t:=1$ as in the code):

- (a) **as coded (per-turn normaliser):**
$$\sum_{t\le S}\frac{p_t\cdot\hat A_t[:,b_t]-\operatorname{val}(\hat A_t)+\varepsilon_s}{\Delta_t}\;\ge\;-\mathrm{Reg}_S.$$
- (b) **fixed normaliser $\bar\Delta$:**
$$\sum_{t\le S}p_t\cdot\hat A_t[:,b_t]\;\ge\;\sum_{t\le S}\big(\operatorname{val}(\hat A_t)-\varepsilon_s\big)-\bar\Delta\,\mathrm{Reg}_S,$$
  and with tuned $\eta$ and $\alpha=0$ the regret term is $\bar\Delta\sqrt{S\ln k/2}$.

*Proof.*

1. Because $\sum_e w_{t,e}=1$, the weighted reward is the normalised payoff of the mixture:
   $w_t\cdot r_t=(p_t\cdot\hat A_t[:,b_t]-\min\hat A_t)/\Delta_t$ (Lean `learner_reward_linear`).
2. Lemma H with $e=e^*$ gives $\sum_t(q^*_t\cdot\hat A_t[:,b_t]-p_t\cdot\hat A_t[:,b_t])/\Delta_t\le\mathrm{Reg}_S$.
3. Hypothesis (E) holds for every column, the realised $b_t$ included, so
   $q^*_t\cdot\hat A_t[:,b_t]\ge g_t\ge\operatorname{val}(\hat A_t)-\varepsilon_s$. Substituting gives (a).
4. For (b), $\Delta_t\equiv\bar\Delta$ is a constant, so multiply through by it.

Lean: `claim1_weighted` and `claim1_fixed`. $\square$

**False as first stated.** The raw-unit form "$\sum_t p_t\cdot\hat A_t[:,b_t]\ge\sum_t(\operatorname{val}-\varepsilon_s)-\Delta_{\max}\mathrm{Reg}_S$"
**fails for the per-turn normaliser**. It fails even for the code's own three experts and an
opponent that ignores $p$.

*Counterexample.* Alternate $\hat A=\begin{pmatrix}1&-1\\-1&1\end{pmatrix}$ (high stakes) and
$\epsilon\hat A$ (low stakes, $\epsilon=0.01$).

- The experts are: top move $=$ row 1, RM+ $=(\frac12,\frac12)$ exactly, safest $=$ row 1. Every turn
  has $\operatorname{val}=0$ and $\varepsilon_s=0$.
- The opponent plays column 2 on high-stakes turns and column 1 on low-stakes turns.
- In normalised units each of the three experts earns exactly $1$ per pair of turns, so Hedge never
  moves away from the non-equilibrium experts.
- In raw units the learner loses about $\frac23$ on every high-stakes turn and wins back only
  $O(\epsilon)$. The total deficit is $S/3$. Measured: $132$, $528$ and $2112$ at $S=400$, $1600$ and
  $6400$, against $\Delta_{\max}\mathrm{Reg}_S=30$, $59$ and $119$.
- With the fixed normaliser $\bar\Delta=2$ the equilibrium expert pulls ahead in reward. The deficit
  becomes $15$, $30$ and $60$, sublinear and inside the bound.

The per-turn normaliser counts low-stakes turns as much as high-stakes ones, which lets an opponent
shift the learner's losses onto high-stakes turns.

**Theorem E2 (composition with Theorem 1; Claim 2).** Use the fixed-normaliser variant with
$\hat A_t$ on the $[-1,1]$ scale, so $\bar\Delta\le2+2\varepsilon_v$. Assume (E) and
$\lVert\hat A_t-Q^*_t\rVert_\infty\le\varepsilon_v$. Let $\sigma$ be any opponent for the whole
match. It may depend on the entire history of all games, see $p_t$, and know the algorithm. Then
$$\frac1N\sum_{g=1}^N\Big(V^*_0(s^g_0)-\mathbb E_\sigma[u_g]\Big)\;\le\;\bar H_\sigma(\varepsilon_s+2\varepsilon_v)+\frac{\bar\Delta\,\mathbb E_\sigma[\mathrm{Reg}_S]}{N}.$$
With $\alpha=0$ and $\eta=\sqrt{8\ln k/(NH)}$, the last term is at most $\bar\Delta\sqrt{H\ln k/(2N)}$.
If $\eta$ is tuned to the true $\mathbb E[S]$ instead, it is at most $\bar\Delta\sqrt{\bar H_\sigma\ln k/(2N)}$.
With anytime $\eta_t=\sqrt{8\ln k/t}$ in FTRL form, the regret term is $2\sqrt{S\ln k/2}+\sqrt{\ln k/8}$ (Cesa-Bianchi & Lugosi 2006, Thm 2.3), so E2's last term doubles.
Fixed share $\alpha>0$ adds $\bar\Delta\bar H_\sigma\ln\frac1{1-\alpha}/\eta$.

*Proof.*

- **(i) Telescoping identity, for any history-dependent learner.** Theorem 1 assumes a Markov agent.
  Here $p_t$ depends on past games and on earlier turns of the same game, so Theorem 1 is not cited;
  its induction is redone directly. For live $s_t$ put
  $\delta(h_t)=V^*_t(s_t)-p_t^\top Q^*_t(s_t)\,y_t$, where $y_t=\sigma(h_t)$. Moves are simultaneous:
  $a_t\sim p_t$ and $b_t\sim y_t$ are independent given $h_t$. Hence
  $\mathbb E[V^*_{t+1}(s_{t+1})\mid h_t]=p_t^\top Q^*_t y_t$, with the convention $V^*=u$ on terminal
  states. Summing over $t<\tau_g$ gives the exact identity
  $V^*_0(s^g_0)-\mathbb E[u_g]=\mathbb E[\sum_{t<\tau_g}\delta(h_t)]$, with no Markov assumption.
  (This is `full_game.md` (A.0) with equality, at the opponent's actual mixed action.) Then sum over $g$.
- **(ii) Per turn.** By Lemma 1(i) and Lemma 2(a),
  $V^*_t-p_t\cdot Q^*_t[:,b]\le(\operatorname{val}\hat A_t-p_t\cdot\hat A_t[:,b])+2\varepsilon_v$ for
  every column $b$ (Lean `composition_step`). Average over $b\sim y_t$ to get
  $\delta(h_t)\le\mathbb E_{b\sim y_t}[\operatorname{val}\hat A_t-\varepsilon_s-p_t\cdot\hat A_t[:,b]]+\varepsilon_s+2\varepsilon_v$.
- **(iii) Regret.** Proposition E1(b) bounds the sum of the bracket at the *realised* $b_t$ by
  $\bar\Delta\mathrm{Reg}_S$ on every path. Given $h_t$, $b_t\sim y_t$, so the tower rule replaces
  $\mathbb E_{b\sim y_t}$ by the realised $b_t$ inside the expectation. All sums are finite, since
  $S\le NH$.
- **(iv) Forced turns.** These have $\varepsilon_s=0$ and contribute at most $2\varepsilon_v$.
- **Conclusion.** The budget term $\varepsilon_s+2\varepsilon_v$ is charged on every live turn, so it
  totals $\mathbb E[\sum_g\tau_g](\varepsilon_s+2\varepsilon_v)=N\bar H_\sigma(\varepsilon_s+2\varepsilon_v)$.
  The regret term is $\mathrm{Reg}_S$, and $\mathrm{Reg}$ is increasing in its argument with
  $\mathbb E[S]\le N\bar H_\sigma$. Divide by $N$. For tuning to $H$: $\ln k/\eta+\eta N\bar H_\sigma/8\le\sqrt{NH\ln k/2}$. $\square$

**What exactly is bounded.**

- The bound holds for *every* opponent $\sigma$, so it bounds the learner's worst-case $N$-game
  average gap. One can call this the match-average exploitability of the learning algorithm: the
  opponent knows the algorithm and adapts across games.
- Take $\sigma$ to be a per-game best responder to the learner's game-$g$ strategy $\pi_g$, which is
  fixed by the weights at the start of game $g$. The left side is then
  $\frac1N\sum_g\mathbb E[\operatorname{expl}(\pi_g)]$.
- It is a bound **in expectation**, over chance, the learner's sampling and the opponent's
  randomness. A high-probability version for the realised $\frac1N\sum_g u_g$ should follow from
  Azuma's inequality, because each turn changes $V^*$ by at most 2. That would add
  $O(\sqrt{H\log(1/\delta)/N})$, but it is **not derived here**.
- $\bar H_\sigma$ is the mean length against the opponent in question, as in Theorem 1's (A.2). It
  cannot be replaced by typical game length (same counterexample).

**What is not provable.**

1. **No bound on the exploitability of a single game's policy or of the final weights.** Regret only
   controls the average. Suppose the weights concentrate on the top-move expert, as they may after
   the MCTS phase. A fresh best responder then wins about
   $(1-w_{e^*})\times(\text{top-move gap})$ per turn until the weights move, which takes
   about $\ln(1/w_{e^*})/\eta$ turns.
2. **No raw-unit composition for the code as written.** With the per-turn normaliser only
   Proposition E1(a) holds. The counterexample above, placed in a game whose payoff adds up per-turn
   stage payoffs, keeps the average gap away from 0 for every $N$. That last step is checked by
   reading, not by brute force.
3. **No sublinear guarantee with the code's constant $\eta,\alpha$.** The regret term per turn stays
   at $0.035\bar\Delta$, so the bound is $\bar H_\sigma(\varepsilon_s+2\varepsilon_v+0.07)+11/N$ with
   $\bar\Delta=2$.
4. **Units.** `full_matrix` returns heuristic engine units, not $[-1,1]$. Theorem E2 needs $\hat A$ on
   the payoff scale. As in `full_game.md` §2, the mapping error is part of $\varepsilon_v$, and any
   fixed normaliser needs a known bound on the range.

**Sanity check.** With $k=1$ (only the equilibrium expert), $\mathrm{Reg}_S=0$ and Theorem E2 reduces
to Theorem 1, (A.2), for the opponent $\sigma$.

**Proposition E3 (exploitation; Claim 3).** Against any opponent and on every path, the learner's
surrogate payoff is at least that of the best fixed expert minus the regret term:
$\sum_t p_t\cdot\hat A_t[:,b_t]\ge\max_e\sum_t q_{e,t}\cdot\hat A_t[:,b_t]-\bar\Delta\mathrm{Reg}_S$
(fixed normaliser). With the per-turn normaliser the same holds in $1/\Delta_t$-weighted units.
Against a comparator that switches $m$ times, use the shifting bound of Lemma H.

*Proof.* Lemma H, then step 1 of the proof of Proposition E1. $\square$

The comparator is the expert's surrogate payoff **at the states the learner reached and against the
moves the opponent actually made**. It is not the game payoff of "always follow expert $e$". Two
counterfactuals separate them:

- The **opponent's reaction.** Against an adaptive opponent external regret is not policy regret,
  which can be linear (Arora, Dekel & Tewari 2012).
- The **state trajectory.** Even a stationary opponent with $b_t=\beta(s_t)$ sees different states
  if the learner plays differently. A game-level comparison with the best expert played throughout
  needs online-MDP results (e.g. Even-Dar, Kakade & Mansour 2009, *Math. Oper. Res.*; **check**).

Proposition E3 is therefore exact only as a per-turn statement along the realised trajectory. Safety
(Proposition E1, Theorem E2) needs neither counterfactual: the $\varepsilon_s$ guarantee is a property
of the stage game at whatever state is reached, against whatever column is played.

Fixed share buys tracking. If the opponent changes (for example MCTS, then the exploiter), regret
against the best expert in each phase costs $\ln(k/\alpha)/\eta$ per switch (Herbster & Warmuth
1998). The price is the $S\ln\frac1{1-\alpha}/\eta$ term against a fixed comparator.

## Related work

- **Safe opponent exploitation** (Ganzfried & Sandholm, ACM TEAC 2015): exploit an opponent while
  guaranteeing at least the game value, by risking only gifts already won. Our guarantee is the
  regret-relaxed version: value minus $\varepsilon_s+2\varepsilon_v$ per turn, minus $\mathrm{Reg}_S/N$.
- **Double oracle** (McMahan, Gordon & Blum, ICML 2003) and **PSRO** (Lanctot et al., NeurIPS 2017):
  the "growing expert pool" version, where best responses are added as new strategies and a
  meta-equilibrium is solved over the pool. Our expert set is fixed with $k=3$.
- **Hedge / fixed share**: Freund & Schapire 1997 (JCSS); Herbster & Warmuth 1998 (Machine Learning,
  "Tracking the best expert"); Cesa-Bianchi & Lugosi 2006 (book). **Policy regret**: Arora, Dekel &
  Tewari, ICML 2012.

## Verification status

| Claim | Status |
|---|---|
| Lemma H (fixed-share Hedge, pathwise, adaptive) | Summation step in Lean (`hedge_telescope`, with the per-step Hoeffding inequality as a hypothesis); per-step inequality checked by reading; numerically, max regret ≤ 0.23 of the bound for the code's parameters (5 seeds, $S=2000$) |
| Proposition E1(a), per-turn normaliser | Lean (`claim1_weighted`, `learner_reward_linear`); numerically, the weighted deficit is at most 0.196 of the bound against an adaptive best-responding opponent |
| Proposition E1(b), fixed normaliser, $\sqrt S$ order | Lean (`claim1_fixed`); numerically 0/32 violations (uses a constant tuned $\eta$; the anytime update is checked separately); max regret$/\sqrt{S\ln k}$ is 0.06, 0.10, 0.15 and 0.24 at $S=100$, 400, 1600 and 6400 (bound 0.71). This confirms the bound, which Lemma H forces, not the empirical rate: the ratio is still rising as $\eta$ shrinks |
| Anytime update: incremental form has linear regret (counterexample, $0.40\,T$); FTRL form regret 1.46 at $T=10^3$–$10^5$ | numeric check (scratchpad verify/anytime.py; exp3_live fix) |
| Raw-unit Claim 1 with per-turn normaliser | **False**: counterexample brute-forced, deficit $=S/3$ |
| Theorem E2, per-turn step | Lean (`composition_step`, from `colPay_close` and `val_ge`) |
| Theorem E2, telescoping identity and expectation step | Checked by reading. Random layered game (5 stages, 2×3 stages, $\varepsilon_v=0.05$, $N=4000$): average $V^*-u=0.039\pm0.013$ vs average $\sum_t\delta_t=0.032$; bound $0.275+0.041=0.316$ |
| "Not a bound on the final policy"; the composition counterexample at game level | Checked by reading |
| Proposition E3, policy-regret and trajectory caveats | Checked by reading; ADT 2012 cited |

Lean: `lean/FullGame/Hedge.lean` imports `Basic.lean` (unchanged) and is built by `lake build`. It
has no `sorry`, and its axioms are only propext, Classical.choice and Quot.sound.
`lean/FullGame.lean` now also imports `FullGame.Hedge`.

Note: Lean `claim1_fixed` uses a constant `lo`, while the code's per-turn $\min\hat A_t$ is equivalent because it cancels across experts.
