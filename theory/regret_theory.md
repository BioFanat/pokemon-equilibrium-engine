# Regret theory of the live experts learner: anytime updates, weight dynamics, leaf structure

Companion to `hedge_experts.md` (Lemma H, Prop. E1–E3, Thm E2), `full_game.md` (§1 model, Lemma 1,
Lemma 2, Theorem A with (A.0)–(A.2)) and `theory_section.md`. Code: `engine/exp3_live.py`
(`Learner.update`), `engine/experiments.py` (`Hedge2`, `XHedge2`), `engine/hedge_stream.py`,
`engine/nash.py:47` (`rm_plus`), `engine/endgame_eps.py`. Checks: `theory/checks/rt*.py`, run with
`engine/.venv/bin/python theory/checks/<name>.py` (numpy/scipy only, no battles; each runs in seconds).

Statement labels: **R1–R6**. "Proved" means proved in this note. "Cited" means taken from the
reference without re-proof. "Checked" means a numeric check script, named in each case.

## 0. Notation and conventions

| Symbol | Meaning |
|---|---|
| $k$ | number of experts ($k=2$ in `Hedge2`: $F$ = foul-play MCTS, $N$ = RM+ equilibrium, sampled) |
| $r_t\in[0,1]^k$ | reward vector at learner turn $t$; $r_{t,e}=\operatorname{clip}\big((q_{e,t}\cdot\hat A_t[:,b_t]-\min\hat A_t)/\bar\Delta,0,1\big)$, $\bar\Delta=400$ |
| $L_t=\sum_{s\le t}r_s$ | cumulative reward (`Learner.cum`), $L_0=0$ |
| $w_t$ | weights used at turn $t$; $p_t=\sum_e w_{t,e}q_{e,t}$ |
| $\mathrm{Reg}_T$ | $\max_e L_{T,e}-\sum_{t\le T}w_t\cdot r_t$ (pathwise, normalised units) |
| $\eta_t$ | $\sqrt{8\ln k/t}$ |
| $\beta_t$ | step actually used at turn $t$ by the shipped FTRL: $\beta_1$ arbitrary, $\beta_t=\eta_{t-1}$ for $t\ge2$ |
| $G_t$ | $L_{t,N}-L_{t,F}$: cumulative surrogate reward gap of the equilibrium expert over the MCTS expert |
| $d_t$ | $r_{t,N}-r_{t,F}$: per-turn surrogate advantage |
| $\hat A_t$ | the heuristic stage matrix (`nash.full_matrix`, own perspective, engine units) |
| $Q^*_t$ | true stage matrix on the $[-1,1]$ payoff scale (`full_game.md` §1) |
| $\varepsilon_s,\varepsilon_v$ | solver gap and leaf error as in `full_game.md` §1 |
| $\varepsilon_v^{\rm cal}(M,Q)$ | $\min_{a\ge0,c}\lVert aM+c-Q\rVert_\infty$ (calibrated leaf error; LP in `endgame_eps.py:131`) |
| $\operatorname{lgs}(z)$ | logistic function $1/(1+e^{-z})$ ($\sigma$ is reserved for opponent strategies) |
| $\vartheta$ | confidence level in §R5 ($\delta$ is reserved for the stage budget of Theorem A) |
| $v^*$ | game value in win-rate units ($=1/2$ for seat-swapped pairs) |

Units. Payoff $u\in[-1,1]$ and score $s=(u+1)/2\in[0,1]$; exploitability in score units is half of
exploitability in payoff units. "Normalised units" are engine units divided by $\bar\Delta=400$.

---

## R1. Incremental anytime multiplicative weights has linear regret

**Setting.** Incremental anytime MW (the first anytime implementation in `exp3_live.py`):
$\lambda_1=0$, $\lambda_{t+1}=\lambda_t+\eta'_t r_t$ with $\eta'_t=c/\sqrt t$ for a constant $c>0$,
and $w_t\propto e^{\lambda_t}$. Equivalently $\lambda_t=\sum_{s<t}\eta'_s r_s$: this is
follow-the-leader on rewards **re-weighted by $c/\sqrt s$**, so early rounds count more than late
ones. It is entropic mirror descent with a decreasing step. It is not FTRL, which uses
$\beta_t L_{t-1}$ (Thm R2).

**Proposition R1.** Fix $k\ge2$, $c>0$ and $\alpha\in(0,1)$. For horizon $T$ let $T_1=\lfloor\alpha T\rfloor$
and let the rewards be $r_t=e_1$ for $t\le T_1$ and $r_t=e_2$ for $t>T_1$ (oblivious, rewards in $\{0,1\}^k$). Then
$$\lim_{T\to\infty}\frac{\mathrm{Reg}_T}{T}=\kappa(\alpha):=\begin{cases}2\alpha,&0<\alpha\le\tfrac14,\\ 1-2\alpha,&\tfrac14\le\alpha\le\tfrac12,\\ 0,&\alpha\ge\tfrac12.\end{cases}$$
$\kappa$ does not depend on $c$ or $k$. Its maximum is $\kappa(1/4)=1/2$, and $\kappa(0.3)=0.40$.

*Idea.* Expert 1's lead in $\lambda$ after phase 1 is $c\sum_{s\le T_1}s^{-1/2}\approx2c\sqrt{\alpha T}$.
Phase 2 erodes it at rate $c/\sqrt t$, so it lasts until $\sqrt t\approx2\sqrt{\alpha T}$, i.e.
$t\approx4\alpha T$. The lead is $\Theta(\sqrt T)$ away from that point, so the weights are
essentially $0/1$. The learner collects phase 1, misses phase 2 up to $\min(4\alpha T,T)$, and the
best expert in hindsight is expert 2 with $T-T_1$.

*Proof.* Write $s_n=\sum_{j=1}^n j^{-1/2}$. Comparing with $\int x^{-1/2}dx$ gives
$2\sqrt{n+1}-2\le s_n\le2\sqrt n$ and $s_n-s_m\le2(\sqrt n-\sqrt m)$ for $n\ge m$.

1. *Gap formula.* For $t>T_1$, $\lambda_t(1)=cs_{T_1}$, $\lambda_t(2)=c(s_{t-1}-s_{T_1})$ and $\lambda_t(j)=0$ for $j\ge3$.
   So $\Lambda_t:=\lambda_t(1)-\lambda_t(2)=c(2s_{T_1}-s_{t-1})\ge2c(2\sqrt{T_1}-\sqrt t-2)$ by the bounds above.
2. *Lower bound, $\alpha<1/2$.* Fix $\theta\in(0,1)$. For $T_1<t\le4\theta^2T_1$, $\sqrt t\le2\theta\sqrt{T_1}$, so
   $\Lambda_t\ge\gamma_T:=2c(2(1-\theta)\sqrt{T_1}-2)$, and the learner's reward
   $w_t(2)\le e^{-\Lambda_t}\le e^{-\gamma_T}$. The learner's reward is at most $T_1$ in phase 1, at most
   $Te^{-\gamma_T}$ on $(T_1,\min(T,4\theta^2T_1)]$, and at most $(T-4\theta^2T_1)^+$ afterwards. Expert 2 earns $T-T_1$. Hence
   $\mathrm{Reg}_T\ge T-2T_1-(T-4\theta^2T_1)^+-Te^{-\gamma_T}$. Divide by $T$. Since $T_1/T\to\alpha$ and $\gamma_T\to\infty$,
   $\liminf\mathrm{Reg}_T/T\ge1-2\alpha-(1-4\theta^2\alpha)^+$. Let $\theta\uparrow1$:
   $\liminf\ge1-2\alpha-(1-4\alpha)^+=\kappa(\alpha)$.
3. *Upper bound.* In phase 1, $w_t(1)\ge1-(k-1)e^{-cs_{t-1}}$ and $\sum_t e^{-cs_{t-1}}\le\sum_t e^{-2c(\sqrt t-1)}<\infty$,
   so the learner earns at least $T_1-O(1)$. For $\theta\in(0,1)$ and $t\ge4T_1/\theta^2$, step 1's formula with
   $s_{t-1}\ge2\sqrt t-2$ and $s_{T_1}\le2\sqrt{T_1}$ gives $\Lambda_t\le c(4\sqrt{T_1}(1-1/\theta)+2)\to-\infty$. Also
   $\lambda_t(2)-\lambda_t(j)=c(s_{t-1}-s_{T_1})\to\infty$ uniformly there, for $j\ge3$. So $w_t(2)\ge1-o(1)$ uniformly, and
   $\mathrm{Reg}_T\le T-2T_1-(T-4T_1/\theta^2)^+(1-o(1))+O(1)$. Let $T\to\infty$ and then $\theta\uparrow1$:
   $\limsup\mathrm{Reg}_T/T\le\kappa(\alpha)$.
4. *$\alpha\ge1/2$.* The best expert earns $\max(T_1,T-T_1)\le T_1+1$, and step 3 gives the learner $\ge T_1-O(1)$,
   so $\limsup\mathrm{Reg}_T/T\le0$. Conversely $4\alpha\ge2>1$, so step 2's bound $\Lambda_t\ge\gamma_T$ (with any $\theta\in(1/(2\sqrt\alpha),1)$, which is needed so that $4\theta^2\alpha>1$) covers every $t\le T$
   in phase 2, and the learner earns at most $T_1+Te^{-\gamma_T}$. Hence $\mathrm{Reg}_T\ge-Te^{-\gamma_T}-1$ and $\liminf\mathrm{Reg}_T/T\ge0$. $\square$

**Contrast with FTRL.** Under FTRL, $w_t\propto e^{\beta_tL_{t-1}}$. The sign of $\lambda_t(1)-\lambda_t(2)$
is the sign of $L_{t-1,1}-L_{t-1,2}$, which flips at $t=2T_1$ for every step schedule. Thm R2 bounds the
regret by $O(\sqrt{T\ln k})$ for every sequence.

**Numeric check** (`checks/rt1_incremental_mw.py`; $k=3$, $c=\sqrt{8\ln3}$ as in the code).
Regret$/T$ at $T=10^4,10^5,10^6$:

| $\alpha$ | 0.05 | 0.10 | 0.20 | 0.25 | 0.30 | 0.40 | 0.45 | 0.50 |
|---|---|---|---|---|---|---|---|---|
| $\kappa(\alpha)$ | 0.100 | 0.200 | 0.400 | 0.500 | 0.400 | 0.200 | 0.100 | 0 |
| $T=10^6$ | 0.0994 | 0.1991 | 0.3987 | 0.4985 | 0.4000 | 0.2000 | 0.1000 | 0.0000 |

For $\alpha<1/4$ the convergence is $O(T^{-1/2})$, from the crossing window. The same $0.4000$ appears
for $k=2,3,10$. Running `exp3_live.Learner`'s incremental branch itself on the $30/70$ sequence gives
$0.4001\,T$ at $T=10^4$ (the incremental branch is reached with `Learner(alpha=1e-300)`; the share step is then negligible), which is fact (i). The shipped FTRL form has regret $\le0.0014\sqrt{T\ln k}$ on every
$\alpha$ at $T=10^6$ (about 1.5 in absolute terms).

**New vs known.** That mirror descent with a decreasing step (as opposed to FTRL / dual averaging)
can have linear regret on the simplex is known qualitatively. Orabona's monograph *A Modern Introduction to Online
Learning* discusses the same phenomenon. Orabona & Pál (2018, Thm 4 in the arXiv version) prove Regret $\ge T/6$ for entropic mirror descent with step $1/\sqrt t$ on the same kind of two-phase sequence (phases $T/3$, $2T/3$). New here: the exact limit $\kappa(\alpha)=\min(2\alpha,1-2\alpha)^+$ for every $\alpha$ ($\kappa(1/3)=1/3$ vs their bound $1/6$), and its independence of $c$ and $k$.

---

## R2. Anytime FTRL as shipped, and Theorem E2′

**Lemma R2a (anytime FTRL, shipped indexing).** Let $\beta_1\ge\beta_2\ge\dots>0$ and
$w_t\propto\exp(\beta_tL_{t-1})$. The rewards $r_t\in[0,1]^k$ may depend on $w_1,\dots,w_t$ (adaptive
opponent). Then on every path
$$\mathrm{Reg}_T\le\frac{\ln k}{\beta_T}+\frac18\sum_{t=1}^T\beta_t.$$
The shipped learner (`exp3_live.py:69`, `logw = η_t·cum` after the $t$-th update) plays turn $t$ with
$\beta_t=\eta_{t-1}=\sqrt{8\ln k/(t-1)}$ for $t\ge2$, and uniformly at $t=1$, so $\beta_1:=\beta_2$ is admissible. For it,
$$\mathrm{Reg}_T\le3\sqrt{\frac{T\ln k}{8}}+\sqrt{\frac{\ln k}{8}}=\tfrac32\sqrt{\tfrac{T\ln k}{2}}+\sqrt{\tfrac{\ln k}{8}}.\tag{R2.1}$$

*Proof.* Let $\Phi_t(\beta)=\frac1\beta\ln\big(\frac1k\sum_ie^{\beta L_{t,i}}\big)$, so $\Phi_0\equiv0$.

1. $\Phi_T(\beta_T)\ge\max_iL_{T,i}-\ln k/\beta_T$: keep only the largest term of the sum.
2. $\Phi_t(\beta_t)-\Phi_{t-1}(\beta_t)=\frac1{\beta_t}\ln\sum_iw_{t,i}e^{\beta_tr_{t,i}}\le w_t\cdot r_t+\beta_t/8$.
   The equality uses $w_t\propto e^{\beta_tL_{t-1}}$; the inequality is Hoeffding's lemma for a $[0,1]$ variable.
3. $\beta\mapsto\Phi_t(\beta)$ is nondecreasing, because $\big(\frac1k\sum_ie^{\beta L_i}\big)^{1/\beta}$ is the
   $L^\beta$ norm of $e^{L}$ under the uniform measure (Lyapunov's inequality). Hence
   $\Phi_{t-1}(\beta_t)\le\Phi_{t-1}(\beta_{t-1})$.
4. Telescope:
   $\Phi_T(\beta_T)=\sum_{t=1}^T[\Phi_t(\beta_t)-\Phi_{t-1}(\beta_t)]+\sum_{t=2}^T[\Phi_{t-1}(\beta_t)-\Phi_{t-1}(\beta_{t-1})]$.
   The second sum is $\le0$ by step 3, so step 2 gives $\Phi_T(\beta_T)\le\sum_tw_t\cdot r_t+\frac18\sum_t\beta_t$.
   Combining with step 1 gives the general bound.
5. Shipped schedule, $T\ge2$: $\ln k/\beta_T=\sqrt{(T-1)\ln k/8}$ and
   $\frac18\sum_t\beta_t=\sqrt{\ln k/8}\,(1+s_{T-1})\le\sqrt{\ln k/8}\,(1+2\sqrt{T-1})$, which sums to (R2.1).
   For $T=1$ the bound is $2\sqrt{\ln k/8}$, which is also $\le$ (R2.1).

Every step is pathwise, and $w_t$ is fixed before $r_t$ is revealed, so adaptivity changes nothing. $\square$

**Cited.** Cesa-Bianchi & Lugosi 2006, *Prediction, Learning, and Games*, Thm 2.3: with
$w_t\propto e^{\eta_tL_{t-1}}$ (step $\eta_t$, **not** $\eta_{t-1}$, at turn $t$),
$\mathrm{Reg}_T\le2\sqrt{(T/2)\ln k}+\sqrt{\ln k/8}$ (constant quoted from memory, **check**). The shipped
code is one index off from that schedule: it uses $\eta_{t-1}$ with $L_{t-1}$. So Thm 2.3 does not literally
apply to it, and (R2.1) is proved here instead. Lemma R2a is the standard FTRL-with-decreasing-rates
argument (cf. Orabona's monograph, ch. 7). Applied to CBL's own schedule $\beta_t=\eta_t$, Lemma R2a gives $\le3\sqrt{T\ln k/8}$ under CBL's indexing ($\tfrac32\sqrt{T\ln k/2}+\sqrt{\ln k/8}$), so on equal terms its constant $3/2$ is smaller than CBL's $2$.

**Theorem E2′ (shipped learner, carried weights).** Assume the setting of Thm E2 (`hedge_experts.md`)
with these differences, all of which match the code:

- (B1) the rewards are as coded: fixed normaliser $\bar\Delta=400$, clipped to $[0,1]$, with the FTRL step of Lemma R2a carried across games;
- (B2) one **global** calibration $(a,c)$, $a>0$, with $\lVert a\hat A_t+c-Q^*_t\rVert_\infty\le\varepsilon_v^{\rm glob}$ at every learner turn;
- (B3) hypothesis (E): the RM+ expert is $\varepsilon_s$-optimal in $\hat A_t$, in engine units.

Then for every match-long opponent $\sigma$ (adaptive, sees $p_t$, knows the algorithm), in payoff units,
$$\frac1N\sum_{g=1}^N\big(V^*_0(s^g_0)-\mathbb E_\sigma[u_g]\big)\le\bar H_\sigma\big(a\varepsilon_s+2\varepsilon_v^{\rm glob}\big)+400a\Big(3\sqrt{\tfrac{\bar H_\sigma\ln k}{8N}}+\tfrac1N\sqrt{\tfrac{\ln k}8}\Big)+\frac{\mathbb E_\sigma[C_S]}N,$$
where $C_S=\sum_{t\le S}a\big(\operatorname{range}(\hat A_t)-400\big)^+$ is the clipping loss. $C_S=0$ when every
range is $\le400$, which holds on about 99% of turns (code docstring).

*Proof.*

1. *Units.* With $\tilde A_t=a\hat A_t+c$ the coded reward is
   $\operatorname{clip}\big((q_e\cdot\tilde A_t[:,b]-\min\tilde A_t)/(400a),0,1\big)$. So the learner is exactly the
   fixed-normaliser learner on $\tilde A_t$ with $\bar\Delta'=400a$. RM+ is invariant to this map (Lemma R6a), and its gap becomes $a\varepsilon_s$.
2. *Clipping.* Let $x_{t,e}=(q_{e,t}\cdot\tilde A_t[:,b_t]-\min\tilde A_t)/\bar\Delta'\ge0$, so $r_{t,e}=\min(x_{t,e},1)$.
   The learner's normalised payoff is $\sum_ew_{t,e}x_{t,e}\ge\sum_ew_{t,e}\min(x_{t,e},1)=w_t\cdot r_t$. By (E),
   $x_{t,e^*}\ge g_t:=(\operatorname{val}\tilde A_t-a\varepsilon_s-\min\tilde A_t)/\bar\Delta'$, so $r_{t,e^*}\ge g_t-(g_t-1)^+$, and
   $\bar\Delta'(g_t-1)^+\le a(\operatorname{range}\hat A_t-400)^+$. Lemma R2a with $e=e^*$ then gives E1(b) with the
   regret term $\bar\Delta'\mathrm{Reg}_S$ plus $C_S$.
3. *Composition.* Steps (i)–(iv) of the proof of Thm E2 apply verbatim with $\tilde A_t$, $\varepsilon_v^{\rm glob}$ and $a\varepsilon_s$.
4. *Expectation.* $\mathbb E\sqrt S\le\sqrt{\mathbb ES}\le\sqrt{N\bar H_\sigma}$ by Jensen. Divide by $N$. $\square$

**What is cited and what is proved.**

| Part | Status |
|---|---|
| The CBL constant $2\sqrt{(S/2)\ln k}+\sqrt{\ln k/8}$ | cited, for CBL's indexing |
| (R2.1) for the code's indexing | proved |
| Clipping term, unit map, global-calibration requirement | proved |
| Telescoping and composition | proved in `hedge_experts.md` (Thm E2), reused |

**Remarks.**

1. *Persistent vs reset.* With carried weights the regret term per game is $O(\sqrt{\bar H\ln k/N})\to0$.
   A learner reset every game pays (R2.1) once per game, a constant $\approx400a\cdot3\sqrt{\bar H\ln k/8}$ per game.
   In normalised units per learner turn the bound evaluates to $0.0178$ (persistent, $S=2505$ turns over 60 games) vs
   $0.137$ (reset, 45.8 turns per game) (`checks/rt3_weight_dynamics.py`, last block).
2. *Why global, not per-turn, calibration.* A per-turn slope $a_t$ turns the fixed normaliser back into a
   per-turn normaliser, and `hedge_experts.md` shows the per-turn normaliser has a linear raw-unit counterexample.
   Explicit counterexample: identical $\hat A$ every turn with $Q^*=a_t\hat A$, $a_t$ alternating between $1/200$ and $0.01/200$. Per-turn calibration is exact ($\varepsilon_v=\varepsilon_s=0$), yet the deficit in $Q^*$ units is 0.248 per turn, linear in $S$.
   Theorem A, by contrast, allows per-state calibration (Lemma R6a), because each stage is solved separately.
3. *Data.* The global LP in `endgame_eps.py` returns $a=0$ and $\max\varepsilon_v^{\rm glob}=1.0$ (mean 0.94–1.0)
   on all six exactly solved endgames (`results/endgame/eps.jsonl`, `global_a_c`). So E2′ does not apply (hypothesis $a>0$ fails; as $a\to0^+$ its budget is vacuous) for the heuristic leaf. Lemma R6b explains why.

**Numeric check** (`checks/rt2_anytime_ftrl.py`).

- The shipped `Learner` matches $\beta_t=\sqrt{8\ln k/(t-1)}$ FTRL to $10^{-15}$.
- The regret$/$(R2.1) ratio is at most 0.66 over five sequences: phase switch, alternation, two adaptive adversaries and iid. It is 0.66 for alternation at $T=10^4$, so (R2.1) is within a factor 1.5 of tight there.
- $\Phi$ is monotone in $\beta$ on 2000/2000 random draws.

---

## R3. FTRL weight dynamics: backing out the surrogate advantage from the data

**Lemma R3a (identities, $k=2$).** Under the shipped FTRL, after $t$ learner turns:

- (a) $\ln(w_N/w_F)=\eta_tG_t$, so $G_t=\ln(w_N/w_F)/\eta_t$ is recoverable from logged weights and `t_learner`.
- (b) If $d_s\equiv d$ on turns $s\le t$, then $w_F=\operatorname{lgs}\big(-d\sqrt{8\ln2}\,\sqrt t\big)$, a stretched exponential in $t$.
- (c) On every turn, $p_t\cdot\hat A_t[:,b_t]=q_{N,t}\cdot\hat A_t[:,b_t]-400\,w_{F,t}\,\tilde d_t$, where $\tilde d_t=(q_N-q_F)\cdot\hat A_t[:,b_t]/400$ ($=d_t$ when no clipping occurs).

  The learner's per-turn surrogate deficit relative to the equilibrium expert is therefore exactly $400\,w_{F,t}\tilde d_t$ engine units ($\tilde d_t$ is the clipped gap; equal to $d_t$ without clipping).

*Proof.* (a) `exp3_live.py:69` sets $\mathrm{logw}=\eta_tL_t$, and the weights are its softmax. (b) Substitute $G_t=dt$ into (a).
(c) $p=w_Fq_F+w_Nq_N$ with $w_F+w_N=1$, so $p-q_N=w_F(q_F-q_N)$. The min-subtraction is common to both experts and cancels. $\square$

**Corollary R3b (data).** From `hedge_stream_seed{0..3}.jsonl`, end-of-game weights and `t_learner`
(`checks/rt3_weight_dynamics.py`). Rounding the weights to 4 decimals perturbs $G$ by up to 0.021.

1. *The gap is linear in turns within each phase.* A least-squares fit of $G$ on $t$ per seed (phase 1 through
   the origin, phase 2 continuing from the phase-1 endpoint) gives:

   | | seed 0 | seed 1 | seed 2 | seed 3 | mean ± sd |
   |---|---|---|---|---|---|
   | $d_1$ (vs MCTS) | 0.0150 | 0.0143 | 0.0099 | 0.0139 | 0.0133 ± 0.0023 |
   | $d_2$ (vs exploiter) | 0.0303 | 0.0273 | 0.0302 | 0.0288 | 0.0291 ± 0.0014 |
   | $R^2$, phase 1 / 2 | 0.987 / 0.981 | 0.987 / 0.954 | 0.861 / 0.947 | 0.960 / 0.980 | |

   In engine units this is $5.3$ and $11.7$ points per turn: the exploiter phase roughly doubles the
   equilibrium expert's surrogate advantage ($d_2/d_1=1.9$–$3.1$).
2. *Roughly constant within a phase.* There are 120 per-game slopes per phase (turn-weighted).
   - The trend per game is $-0.00011\pm0.00027$ (MCTS phase) and $-0.00032\pm0.00032$ (exploiter phase).
   - The exploiter-phase trend is borderline: the first half averages 0.0325, the second 0.0247.
   - Per-game slopes are noisy (sd 0.013–0.015) around a stable phase mean.
3. *It predicts the decay in fact (ii).* The constant-slope model with pooled $(d_1,d_2)$ and each seed's own $t$:

   | game | 0 | 9 | 19 | 29 | 39 | 49 | 59 |
   |---|---|---|---|---|---|---|---|
   | mean $t$ | 45 | 332 | 640 | 1028 | 1462 | 1934 | 2505 |
   | logged $w_F$ | 0.452 | 0.359 | 0.306 | 0.278 | 0.162 | 0.106 | 0.072 |
   | predicted | 0.450 | 0.362 | 0.312 | 0.268 | 0.166 | 0.106 | 0.065 |

   With per-seed slopes the per-game RMS error is 0.015–0.022. The slow phase-1 drift is
   Lemma R3a(b) with a small $d$: the log-odds grow like $\sqrt t$, not $t$. The faster phase-2 fall
   comes from the $\sqrt t$ damping. The log-odds of $F$ are $-\sqrt{8\ln2}\,G_t/\sqrt t$. In phase 1 ($G=d_1t$) their slope is
   $-\sqrt{8\ln2}\,d_1/(2\sqrt t)$: the $\sqrt t$ halves the marginal rate. After the switch a fresh increment $d_2$ enters at the full
   rate, so at $t_1$ the slope becomes $-\sqrt{8\ln2}\,(d_2-d_1/2)/\sqrt{t_1}$. It jumps by the factor $(2d_2-d_1)/d_1\approx3.4$, not 2.
   From the table, the log-odds change per turn is $\approx-0.00035$ over games 19→29 and $\approx-0.0016$ over 29→39, a ratio of $\approx4.6$ (noisy).
4. *Reset learner.* Per-game slope $G/t$ ($\pm$ here is a 95% half-width, whereas the persistent slopes' $\pm$ is the sd over seeds): $0.0143\pm0.0023$ vs MCTS and $0.0354\pm0.0026$ vs the exploiter
   (turn-weighted 0.0115 and 0.0293). These are the same per-turn advantages as the persistent stream
   (0.0129 and 0.0281, turn-weighted). But each game restarts at $t=1$, so by (b) $w_F$ only falls to
   $\operatorname{lgs}(-0.03\cdot2.35\sqrt{35})\approx0.40$ by the end of a typical game (logged end $w_F$ 0.38, mean $p_F$ 0.41).
5. *The exploiter does not enlarge $d$ when $p_F$ is large.* A pooled regression (240 games per phase, both modes) gives:
   - exploiter phase: $d=0.0286+0.0002\,p_F$ (coefficient $\pm0.014$), no dependence;
   - MCTS phase: $d=0.0263-0.034\,p_F$ ($\pm0.019$), confounded with time, since $p_F$ is high early.

   So the hypothesis "the exploiter's surrogate edge grows with the MCTS share" is **not supported**.
   The extra exposure of the reset learner comes through $w_F$ alone.
6. *Explaining fact (iii) in surrogate units.* By Lemma R3a(c) the per-turn deficit to the equilibrium
   expert is $w_Fd$. It is estimated per game as mean $p_F\times$ game slope, which ignores the within-game
   covariance of $w_F$ and $d$.

   | phase | persistent | reset | ratio |
   |---|---|---|---|
   | vs exploiter | 0.0040 (1.6 engine units) | 0.0116 (4.6 engine units) | 2.9 |
   | vs MCTS | 0.0043 | 0.0051 | 1.2 |

   This matches the sign and the asymmetry of fact (iii): a large paired difference vs the exploiter
   (+0.146) and a small one vs MCTS (+0.06, n.s.). The realised deficit is far below the regret bound
   (0.0040 vs 0.0178, and 0.0116 vs 0.137 per turn; the persistent regret bound covers both phases, while the realised 0.0040 is exploiter-phase only). The bound does not discriminate the two modes;
   the identity does.
7. *Not supported at the game level.* Per game, the predicted deficit difference does not correlate with
   the paired score difference (corr $-0.17$, $n=120$, exploiter phase). The surrogate explanation holds
   on average and is not a per-game predictor. Surrogate units are not win-rate units (§R4).

**Proposition R3c (bracket on the loss to a white-box exploiter).** Let $b_t=\arg\min_bp_t\cdot\hat A_t[:,b]$
(`exploiter_reply`), let $y^*_t$ be a minimax strategy of $\hat A_t$, and define
$\gamma_{F,t}:=\operatorname{val}\hat A_t-q_{F,t}\cdot\hat A_ty^*_t\ge0$ and
$\mathrm{gap}_{F,t}:=\operatorname{val}\hat A_t-\min_bq_{F,t}\cdot\hat A_t[:,b]$. Then on every turn
$$w_{F,t}\,\gamma_{F,t}\;\le\;\operatorname{val}\hat A_t-p_t\cdot\hat A_t[:,b_t]\;\le\;w_{F,t}\,\mathrm{gap}_{F,t}+w_{N,t}\,\varepsilon_s.$$
For a learner reset at the start of each game, the weight at in-game turn $j$ satisfies
$w_{F,j}\ge\operatorname{lgs}\big(-\bar d_j\sqrt{8\ln2\,(j-1)}\big)$, where $\bar d_j=\max(0,G_{j-1}/(j-1))$ is the running mean advantage.

*Proof.*

1. *Lower inequality.* Since $b_t$ minimises, $p_t\cdot\hat A_t[:,b_t]\le p_t\cdot\hat A_ty^*_t=w_Fq_F\cdot\hat A_ty^*+w_Nq_N\cdot\hat A_ty^*$.
   Moreover $x\cdot\hat A_ty^*\le\operatorname{val}\hat A_t$ for every $x$, because $y^*$ is minimax. Rearranging gives $\ge w_F\gamma_F$.
2. *Upper inequality.* $\min_b$ of a convex combination is $\ge$ the convex combination of the minima, and (E) gives $\min_bq_N\cdot\hat A[:,b]\ge\operatorname{val}-\varepsilon_s$.
3. *Weight bound.* Lemma R3a(a) at in-game time $j-1$: the log-odds are $-\eta_{j-1}G_{j-1}=-\sqrt{8\ln2/(j-1)}\,G_{j-1}$. $\square$

*Reading.* The surrogate loss to the white-box exploiter is linear in the MCTS share, with slopes
$\gamma_F$ and $\mathrm{gap}_F$ fixed by the stage game. With the measured $\bar d\approx0.035$ and games of
$\le35$ turns, a reset learner ends a game at mean $w_F=0.38$ (min 0.20; median game 35 learner turns), while the persistent learner enters the exploiter phase at mean $w_F=0.278$
(seed range 0.232–0.362) and ends it at $0.07$. A **quantitative** lower bound needs $\gamma_F$, which is not
logged. Logging $\gamma_{F,t}$ (one LP per turn) would turn R3c into a measured lower bound. That is a
suggestion, not done here.

**New vs known.** The identities are elementary consequences of FTRL. Using them to back out the
surrogate advantage from logged weights, the phase-constant-slope finding, and R3c are new to this project.

---

## R4. Surrogate-to-true transfer

`hedge_experts.md` already contains the per-turn transfer: step (ii) of Thm E2, which uses Lemma 1(i)
and Lemma 2(a) of `full_game.md` (Lean `composition_step`). It is not re-proved here. The trajectory form needed for R3 follows.

**Corollary R4.** Let $\tilde A_t=a\hat A_t+c$ with $\lVert\tilde A_t-Q^*_t\rVert_\infty\le\varepsilon_v(t)$. Then for any
strategies $x_t,q_t$ and the realised columns $b_t$:

- (a) $\sum_tx_t\cdot Q^*_t[:,b_t]\ge\sum_tx_t\cdot\tilde A_t[:,b_t]-\sum_t\varepsilon_v(t)$;
- (b) $\sum_t(x_t-q_t)\cdot Q^*_t[:,b_t]\ge\sum_t(x_t-q_t)\cdot\tilde A_t[:,b_t]-2\sum_t\varepsilon_v(t)$;
- (c) $\sum_t\big(x_t\cdot Q^*_t[:,b_t]-\operatorname{val}Q^*_t\big)\ge\sum_t\big(x_t\cdot\tilde A_t[:,b_t]-\operatorname{val}\tilde A_t\big)-2\sum_t\varepsilon_v(t)$.

In particular, in game units, the true trajectory advantage of the equilibrium expert satisfies
$\sum_t(q_N-q_F)\cdot Q^*_t[:,b_t]\ge400a\,G_t-2\sum_t\varepsilon_v(t)$ (no clipping).

*Proof.* (a) and (b) follow from $|x\cdot(Q^*-\tilde A)[:,b]|\le\varepsilon_v$ (`full_game.md` Lemma 1(i)) applied to $x$ and to $q$.
(c) adds Lemma 2(a), $|\operatorname{val}Q^*-\operatorname{val}\tilde A|\le\varepsilon_v$. The last display is (b) with
$(q_N-q_F)\cdot\tilde A_t[:,b_t]=400a\,d_t$. $\square$

*Check:* `checks/rt4_transfer.py` (200 random trajectories, all three inequalities hold).

**Reading: the drift is a property of the surrogate.**

- With $G_t\approx dt$, the corollary certifies a true advantage only if $400a\,d>2\bar\varepsilon_v$ per turn.
- The global calibration has $a=0$ on every exact endgame (§R2, Remark 3), so nothing is certified. Even an
  optimistic per-state reading would not change this: mean $\varepsilon_v^{\rm cal}$ is 0.15–0.61 while $400a\,d$ is
  of order $10^{-2}$–$10^{-1}$ for any slope $a$ near the heuristic's scale ($\approx1/200$ per engine unit).
- So "the learner drifts to the equilibrium expert" is a statement about $\hat A$, which by construction is
  the matrix whose equilibrium $q_N$ is.
- The only true-unit evidence is the paired score difference (fact (iii)). That is an empirical fact, not a consequence of R3.

---

## R5. Exploitability bracket

Fix a target policy $\pi$, in score units, with $v^*=1/2$ (seat-swapped pairs make the game symmetric).
$\operatorname{expl}(\pi)=v^*-\inf_\sigma s(\pi,\sigma)$, where $s$ is $\pi$'s expected score.

**Proposition R5a (valid lower bounds; selection; fresh seeds).** Let $X_1,\dots,X_m$ be exploiters, each
a legal strategy in the same information model as $\pi$ and fixed before the evaluation seeds are drawn.
Exploiter $X_j$ is evaluated on $n$ independent seat-swapped pairs with pair scores
$Z_{j,i}\in[0,1]$ (exploiter's score). Let $\bar Z_j$ be the mean. Then:

1. *Single exploiter.* $\operatorname{expl}(\pi)\ge\mathbb E\bar Z_j-v^*$, and with probability $\ge1-\vartheta$,
   $\operatorname{expl}(\pi)\ge\bar Z_j-v^*-\sqrt{\ln(1/\vartheta)/(2n)}$.
2. *Maximum over the tried exploiters.* With probability $\ge1-\vartheta$,
   $\operatorname{expl}(\pi)\ge\max_j\bar Z_j-v^*-\sqrt{\ln(m/\vartheta)/(2n)}$. Without the $\ln m$ term, $\max_j\bar Z_j$ is biased upward (winner's curse).
3. *Fresh seeds.* If $j^*$ is chosen on one set of seeds and $X_{j^*}$ is re-evaluated on $n'$ fresh independent
   pairs, then (1) holds for the fresh mean with $n'$ in place of $n$ and no $\ln m$ term.

*Proof.*

1. $\mathbb E Z_{j,i}=1-s(\pi,X_j)\le1-\inf_\sigma s(\pi,\sigma)$, which is $\operatorname{expl}(\pi)+v^*$ when $v^*=1/2$. Hoeffding's inequality for $[0,1]$ variables gives the probability statement.
2. Union bound over $j$: $\Pr[\exists j:\bar Z_j-\mathbb E\bar Z_j\ge r]\le m\,e^{-2nr^2}$. Set this to $\vartheta$.
3. Conditional on the selection seeds, $j^*$ is fixed and the fresh pairs are independent of it. Apply (1) conditionally. $\square$

**Proposition R5b (what the exploitation gain is).** For a same-strength non-exploiting opponent $\beta$
define $\mathrm{gain}=s(\pi,\beta)-s(\pi,X)$ (RESULTS.md, "Correction: an LBR score mixes exploitation with
strength"). Then:

1. *Identity.* $v^*-s(\pi,X)=\big[v^*-s(\pi,\beta)\big]+\mathrm{gain}$.
2. *Bound.* $\mathrm{gain}\le\operatorname{expl}(\pi)+\operatorname{expl}(\beta)$.
3. *Consequences.*
   - If $s(\pi,\beta)\le v^*$, the gain is a valid lower bound on $\operatorname{expl}(\pi)$, but never larger than the plain bound $v^*-s(\pi,X)$.
   - If $s(\pi,\beta)>v^*$, the gain is not a lower bound on $\operatorname{expl}(\pi)$ by itself. It certifies $\operatorname{expl}(\pi)\ge\mathrm{gain}-\operatorname{expl}(\beta)$.

*Proof.*

1. Add and subtract $s(\pi,\beta)$.
2. $s(\pi,X)\ge v^*-\operatorname{expl}(\pi)$ by the definition of expl. Also $s(\pi,\beta)\le\sup_{\pi'}s(\pi',\beta)=v^*+\operatorname{expl}(\beta)$, by the definition of $\beta$'s exploitability in a symmetric game with value $1/2$. Subtract.
3. The first case follows from (1), since the bracketed term is $\ge0$. The second case is (2) rearranged. $\square$

*When it is meaningful.*

- The gain answers a design question: how much of the exploiter's success comes from modelling the target rather than from strength. It reads as an exploitability statement only when $\operatorname{expl}(\beta)\ll\mathrm{gain}$.
- For the equilibrium bots, $\beta=\pi$ and $s=1/2$ exactly, so gain $=$ the plain bound.
- For foul-play the baseline is equilibrium top move, whose exploitability is $\ge0.325$ (wb2, point estimate; certified at 95%: 0.051 for one exploiter, 0.005 after selecting among 3). At 2,000 it ($s=0.31\le\tfrac12$) the gain 0.035 is a valid lower bound, but weaker than the plain 0.225. At 75 ms ($s=0.65>\tfrac12$) it certifies nothing, since $\operatorname{expl}(\beta)\gtrsim0.325>$ gain.
- At 75 ms foul-play beats its baseline ($s=0.65>v^*$), so its gain is not a lower bound at all.

**Proposition R5c (bracket).** In the full-information model, under the assumptions of `full_game.md` Theorem A, with probability $\ge1-\vartheta$:
$$\max_j\bar Z_j-v^*-\sqrt{\tfrac{\ln(m/\vartheta)}{2n}}\;\le\;\operatorname{expl}(\pi)\;\le\;\min\Big\{\tfrac12,\ \tfrac12\sup_\sigma\mathbb E\big[\textstyle\sum_{t<\tau}\delta(t,s_t)\big]\Big\}\;\le\;\min\Big\{\tfrac12,\ \tfrac12\bar H(\varepsilon_s+2\varepsilon_v)\Big\}.$$
The upper side is (A.1)/(A.2) converted to score units. Under hidden information it gains the
$\eta_{\rm NL}+4\,\mathrm{TV}$ terms of `theory_section.md`.

**Numeric check** (`checks/rt5_bracket.py`). Hoeffding radius at $\vartheta=0.05$: $n=20$: $0.274$ ($m=1$),
$0.320$ ($m=3$); $n=30$: $0.223$; $n=100$: $0.122$.

| exploiter → target | point LB | LCB, $m=1$ | LCB with selection |
|---|---|---|---|
| wb2 → top move ($n=20$) | 0.325 | 0.051 | 0.005 ($m=3$) |
| wb2 → sampled ($n=10$) | 0.35 | $-0.037$ | $-0.079$ ($m=2$) |
| LBR → top move (E6q, $n=30$ pairs) | 0.183 | $-0.040$ | |
| LBR → foul-play 2000 (E6f, $n=30$) | 0.225 | 0.002 | |

Winner's curse with $m$ zero-edge exploiters ($n=20$): $\mathbb E\max-1/2=+0.067$ ($m=3$) and $+0.121$ ($m=10$) (computed for draw-free coin-flip games; with 50% draws the bias is $+0.047$ / $+0.086$).
The re-test on fresh seeds has bias $<10^{-3}$.

**Reading.**

- Point lower bounds (0.18–0.35) sit far below the vacuous upper side ($1/2$, since $\varepsilon_v^{\rm glob}\approx1$).
- Distribution-free (Hoeffding) intervals at the current pair counts certify, at 95%, wb2 → top move (LCB 0.051) and, barely, LBR → foul-play 2,000 it (LCB 0.002, single exploiter).
- The CLT intervals in RESULTS.md are narrower, because scores are far from worst-case variance, but they are not corrected for selection.
- The fresh-seed confirmation planned for wb2 (100 pairs) would give radius 0.122, a certified bound $\approx0.20$ if the point estimate holds.

**New vs known.** Hoeffding plus a union bound, and the fresh-sample remedy, are textbook. R5b (the gain
identity and the bound $\mathrm{gain}\le\operatorname{expl}(\pi)+\operatorname{expl}(\beta)$) is a short new observation for this project.

---

## R6. Leaf structure

**Lemma R6a (affine invariance; calibrated Theorem A).**

1. *RM+ invariance.* For $a>0$ and $c\in\mathbb R$, `nash.rm_plus(aM+c)` returns the same strategies as
   `rm_plus(M)` (in exact arithmetic; in floating point, outputs on matrices with exact ties can differ at round-off level). Hence $\varepsilon_s(aM+c)=a\,\varepsilon_s(M)$, and every equilibrium of $M$ is an equilibrium of $aM+c$.
2. *Calibrated Theorem A.* Theorem A ((A.0)–(A.2)) holds with the per-state budget
   $\delta(t,s)=a_{t,s}\varepsilon_s(t,s)+2\varepsilon_v^{\rm cal}(t,s)$, where $(a_{t,s},c_{t,s})$ is any per-state calibration
   ($a\ge0$) and $\varepsilon_v^{\rm cal}(t,s)=\lVert a_{t,s}\hat Q_t(s)+c_{t,s}-Q^*_t(s)\rVert_\infty$.
3. *The $a=0$ case.* With $a=0$ the bound is $\delta=2\lVert c-Q^*\rVert_\infty\ge\operatorname{range}(Q^*)$, which is the trivial per-turn loss.

*Proof.*

1. *Shift.* In `rm_plus` the regret update uses $u-x\cdot u$ and $w-y\cdot w$. A constant shift of $M$ adds the same constant to every entry of $u$ (resp. $w$) and cancels.
2. *Scale.* Scaling by $a>0$ scales $R_x,R_y$ by $a$ at every iteration (induction), so $R/\sum R$ and the "$\sum R>0$" test are unchanged. The averages are therefore identical. Finally, $\operatorname{val}(aM+c)=a\operatorname{val}M+c$ and $\min_yx^\top(aM+c)y=a\min_yx^\top My+c$.
3. *Calibrated Theorem A.* Apply `full_game.md` Lemma 1 with $B=a\hat Q+c$, in which $x$ is $a\varepsilon_s$-optimal by step 2. The induction of Theorem A uses Lemma 1 one state at a time, so the calibration may vary with $(t,s)$.
4. *$a=0$.* $B$ is constant, so every $x$ is $0$-optimal in it, and Lemma 1 gives $2\lVert c-Q^*\rVert_\infty$. Minimising over $c$ gives $\operatorname{range}(Q^*)$. $\square$

This is the quantity `endgame_eps.py` computes ("eps_s on the calibrated matrix"). Per-state calibration
is allowed for the per-turn agent but not for the Hedge learner (§R2, Remark 2).

**Lemma R6b (inversions and $a=0$; exact terminals).** Let $M,Q\in\mathbb R^{n\times m}$, indexed by cells $i$.

1. *Inversion lower bound.* For every $a\ge0$:
   $\min_c\lVert aM+c-Q\rVert_\infty=\tfrac12\max_{i,j}\big[(Q_i-Q_j)-a(M_i-M_j)\big]$. Hence
   $\varepsilon_v^{\rm cal}(M,Q)\ge\tfrac12\max\{Q_i-Q_j:M_i\le M_j\}$.
2. *Criterion for $a=0$.* $a=0$ is an optimal calibration, with $\varepsilon_v^{\rm cal}=\operatorname{range}(Q)/2$, **iff** some cell
   $i\in\arg\max Q$ and some $j\in\arg\min Q$ have $M_i\le M_j$.
3. *Exact terminals.* Suppose $\hat V=u$ on terminal children. Then
   $|\hat Q(a,\beta)-Q^*(a,\beta)|\le\pi_{\rm live}(a,\beta)\,\varepsilon_{\rm live}(s)$, where
   $\pi_{\rm live}(a,\beta)=P(\text{next state live}\mid s,a,\beta)$ and
   $\varepsilon_{\rm live}(s)=\max_{\text{live }s'}|\hat V(s')-V^*_{t+1}(s')|$.

   So cells whose children are all terminal are exact, and $\varepsilon_v(t,s)\le\max_{a,\beta}\pi_{\rm live}(a,\beta)\,\varepsilon_{\rm live}(s)$.

*Proof.*

1. For fixed $a$, the best $c$ centres the values $aM_i-Q_i$, giving half their range. That range is the stated maximum over pairs. Dropping $a(M_j-M_i)\ge0$ on pairs with $M_i\le M_j$ gives the lower bound.
2. $f(a)=\tfrac12\max_{i,j}[(Q_i-Q_j)-a(M_i-M_j)]$ is convex and piecewise linear, with $f(0)=\operatorname{range}(Q)/2$. The pairs active at $a=0$ are exactly $i\in\arg\max Q$, $j\in\arg\min Q$. So $f'(0^+)=\max_{\rm active}\tfrac12(M_j-M_i)$, and $a=0$ is optimal on $a\ge0$ iff $f'(0^+)\ge0$.
3. $\hat Q-Q^*=\sum_{s'}P(s'\mid s,a,\beta)(\hat V-V^*)(s')$. Terminal terms vanish, and live terms are each bounded by $\varepsilon_{\rm live}$. $\square$

**Why one mis-scored terminal forces $a=0$.**

- A won terminal child has $Q^*=+1$, the maximum. The heuristic scores it by material, $-50$ to $+185$ engine units, while live children span $\pm215$.
- If any cell with lower true value is scored at or above it, part (1) costs $\ge(1-Q^*_j)/2$ for every $a$. If that cell is the worst cell, part (2) forces $a=0$, and then $\varepsilon_v^{\rm cal}=\operatorname{range}/2$, close to 1 when the worst cell is near $-1$.
- The global calibration pools every cell of every state, so a single such pair anywhere forces $a=0$ globally. That is the data: $a=0$ and $\max\varepsilon_v^{\rm glob}=1.0$ on all six endgames.
- Per-state, $a=0$ occurs at 9–67% of live states (`a_zero_states`). This count also includes states with constant $Q^*$, where $a=0$ is harmless; the summary does not separate the two.

*The fix.*

- Exact $\pm1$ terminals remove the terminal-vs-live inversions entirely, and part (3) then charges only live mass.
- $\tanh(h/\tau)$ maps live values into $(-1,1)$, below every won terminal and above every lost one, so it restores the order that makes $a>0$ affordable.
- Unlike an affine map, $\tanh$ changes the equilibria. That is intended: it changes the strategy, which a calibration cannot.

**Numeric check** (`checks/rt6_leaf.py`).

- *(a) Invariance.* $\max|x(M)-x(aM+c)|=2\times10^{-12}$ over 200 random $(M,a,c)$ with $a\in[10^{-3},10^3]$; $\varepsilon_s(aM+c)/(a\varepsilon_s(M))=1.000000$.
- *(b) Inversion bound and criterion.* The LP value is never below the inversion bound (0/500). The criterion holds in 14/14 cases.
- *Synthetic stage.* A row with live cells $Q^*=\tanh(1.5M/215)$ has $\varepsilon_v^{\rm cal}=0.125$. Adding one won terminal scored $0$ or $-50$ raises it to $0.56$ or $0.68$. A won terminal scored 20, below a worst cell ($Q^*=-0.2$) scored 60, gives $a=0$ and $\varepsilon_v^{\rm cal}=0.600=\operatorname{range}/2$.
- *Exact-terminal cell bound.* Holds on 500/500 random cells.

**Proposition R6c (non-termination; loops).** Consider the undiscounted game in which endless play pays 0
(Everett's *recursive games*). There is one live state $s$; the agent chooses from $\{E,C\}$ and the opponent from $\{E,C\}$:

- $(E,\cdot)$ ends with payoff $0.3$;
- $(C,E)$ ends with $0.8$;
- $(C,C)$ returns to $s$.

1. *Fixed points.* The Shapley operator $V\mapsto\operatorname{val}\begin{pmatrix}0.3&0.3\\0.8&V\end{pmatrix}$ has every $V\in[0.3,0.8]$ as a fixed point.
2. *The value.* Value iteration from $0$, which equals the values of the $H$-turn capped game with $0$ at the cap, is $0.3$ for every $H\ge1$. This is the game's value: $E$ guarantees $0.3$, and the opponent's "always $C$" holds every agent to $\le0.3$.
3. *Stage-wise optimality is not enough.* At the true value $V^*=0.3$ the stage matrix is $\begin{pmatrix}0.3&0.3\\0.8&0.3\end{pmatrix}$. The stationary agent "always $C$" is a stage maximin, with $\min=0.3=\operatorname{val}$, so its stage budget is $\delta\equiv0$. Yet against "always $C$" it never terminates, scores $0$, and has exploitability $0.3$. Evaluating it with endless play valued at $V^*$ instead of $0$ reports exploitability $0$.
4. *The cap makes backward induction exact.* With a cap $H$ and $u_H=0$, the time-indexed stage equilibria are exact (Theorem A with $\delta\equiv0$; Prop. 0 of `full_game.md`). At $t=H-1$ the stage matrix is $\begin{pmatrix}0.3&0.3\\0.8&0\end{pmatrix}$ and forces $E$.

   For the stationary $C$-agent in the capped game, the time-indexed budget is $\delta_t=0$ for $t<H-1$ and $\delta_{H-1}=0.3$. So Theorem A charges the whole loss to the last turn, and the first inequality of (A.2), $\operatorname{expl}\le\mathbb E_{\sigma^*}[\sum_{t<\tau}\delta_t]$, holds with **equality** ($0.3=0.3$).
5. *Residual form.* For any fixed point $V$ of the Shapley operator, any stationary agent with stage budgets $\delta$ measured in $Q_V$, and any opponent,
   $$V(s_0)-W\le\mathbb E\Big[\sum_{t<\tau}\delta(s_t)\Big]+\limsup_{T\to\infty}\mathbb E\big[V(s_T)\,\mathbf 1\{\tau>T\}\big].$$
   The residual vanishes whenever $\tau<\infty$ a.s. under the opponent, in particular whenever $\bar H=\mathbb E_{\sigma^*}[\tau]<\infty$, for that opponent $\sigma^*$.

*Proof.*

1. For $V\ge0.3$ the row $C$ has minimum $\min(0.8,V)$ and row $E$ has $0.3$, so the value is $\max(0.3,\min(0.8,V))$. This equals $V$ iff $V\in[0.3,0.8]$.
2. Backward induction from $0$ gives $\operatorname{val}\begin{pmatrix}0.3&0.3\\0.8&0\end{pmatrix}=0.3$ (row $E$ is optimal; any weight on $C$ lowers the payoff against column $C$). Then $V\equiv0.3$ is a fixed point of the recursion. The game value follows from the two strategies named in the statement.
3. Direct computation.
4. The capped game is the finite-horizon model of `full_game.md` §1. The budgets follow from $V^*_{t+1}=0.3$ for $t+1<H$ and $V^*_H=0$.
5. Run the telescoping identity of Thm E2 (i) up to time $T$, with $V$ in place of $V^*$. The fixed-point property makes $V$ the stage value at each $s_t$, and the boundary term at $T$ is $\mathbb E[V(s_T)\mathbf1\{\tau>T\}]$. Since $W=\lim_T\mathbb E[u\,\mathbf1\{\tau\le T\}]$ (endless play pays 0, bounded convergence), letting $T\to\infty$ gives the display. If $\tau<\infty$ a.s., the residual is $\le\lVert V\rVert_\infty\Pr(\tau>T)\to0$. $\square$

**Relation to the $\bar H$ refinement and to fact (vi).**

- In the example $\delta\equiv0$, and the whole exploitability is the residual. The best response loops forever, so $\bar H=\infty$ and $\bar H(\varepsilon_s+2\varepsilon_v)$ is undefined or vacuous. (A.2) needs $\bar H<\infty$ (or the cap) for the telescoping to close.
- The engine's endgame has the same structure: `endgame_eps.py:110` iterates Shapley from $0$. Its exact-equilibrium agent has exploitability $0.029$ in payoff units (garchomp vs rotomwash @ 100%, seat 1, `eps.jsonl`).
  - Two looping states give the opponent an endless line, which the best response values as a draw.
  - The best-response evaluation started from $V^*$ instead reports $\approx10^{-16}$.
  - Both match (3).
- The 300-turn rule makes the real game capped, so $\bar H\le300$ and (A.2) holds with time-indexed $V^*_t$. In the capped model the 0.029 is charged to the turns just before the cap, as in (4). This is an interpretation: the engine's best response was evaluated uncapped.
- The agents in the code are stationary (no turn counter). So the cap does not remove this exploitability; it only makes the theory account for it.
- Removing it needs either turn-indexed values near the cap or a tie-break toward terminating actions among stage-optimal ones. That is a suggestion, not proved to suffice in general.

**Numeric check** (`checks/rt6_leaf.py`, part (d)).

- The fixed points on a grid fill $[0.300,0.800]$.
- Value iteration from 0 gives 0.300.
- Stationary $C$ gets $W=0.000$ (exploitability 0.300) with endless play $=0$, and $W=0.300$ when evaluated from $V^*$.
- Capped backward induction gives 0.300 for $H=1,2,5,300$, with $E$ forced at the last turn.

**New vs known.**

- Non-uniqueness of fixed points and the lack of optimal stationary strategies in undiscounted recursive games are classical (Everett 1957, *Recursive games*, Contributions to the Theory of Games III).
- That an optimal strategy must be both *thrifty* (stage-optimal for the value) and *equalizing* (residual $\to0$) is the Dubins–Savage characterisation (Dubins & Savage 1965; "conserving and equalizing" is the Maitra & Sudderth usage; Maitra & Sudderth 1996, *Discrete Gambling and Stochastic Games*; **check** exact theorem numbers).
- Part (5) is that characterisation, written in the form of Theorem A.
- The minimal example, the exact last-turn attribution under the cap, and the link to $\bar H$ are this note's.

---

## Verification status

R1–R6: independently verified (2026-11-14), with corrections.

| Statement | Status | Evidence |
|---|---|---|
| R1 incremental anytime MW, $\mathrm{Reg}_T/T\to\kappa(\alpha)=\min(2\alpha,1-2\alpha)^+$ | **proved**; numerically checked | `rt1`: 0.4000 at $\alpha=0.3$ (all $T$, $k=2,3,10$), max deviation from $\kappa$ 0.0015 at $T=10^6$; the shipped Learner's incremental branch gives 0.4001 |
| R2a anytime FTRL bound (R2.1) for the **code's indexing** ($\beta_t=\eta_{t-1}$) | **proved**; numerically checked | `rt2`: ratio $\le0.66$; indexing matches the Learner to $10^{-15}$ |
| CBL Thm 2.3 constant $2\sqrt{(S/2)\ln k}+\sqrt{\ln k/8}$ | **cited** (constant from memory, check); applies to CBL's indexing, not literally to the code | — |
| E2′ (carried weights, global calibration, clipping term) | **proved** from E2 + R2a + R6a; the clipping term is new; the global-calibration requirement is new | composition steps reused from `hedge_experts.md` |
| E2′ budget term on real leaves | **does not apply** ($a>0$ fails; budget vacuous as $a\to0^+$): global $a=0$, $\varepsilon_v^{\rm glob}\approx1$ on 6/6 endgames | `eps.jsonl` `global_a_c` |
| R3a identities | **proved** (elementary) | — |
| R3b phase-constant slopes $d_1=0.0133\pm0.0023$, $d_2=0.0291\pm0.0014$; trajectory predicted within 0.01 at the tabulated games (max error 0.017 over all 60 games, RMS 0.006; slopes fitted in-sample) | **numerically checked** (data) | `rt3` |
| R3b "exploiter edge grows with $p_F$" | **not supported** (coefficient $0.0002\pm0.014$) | `rt3` |
| R3b deficit ratio reset/persistent 2.9 (exploiter), 1.2 (MCTS) | **numerically checked**; approximation (ignores within-game covariance) | `rt3` |
| R3b per-game link to score | **not supported** (corr $-0.17$) | `rt3` |
| R3c bracket $w_F\gamma_F\le$ loss $\le w_F\mathrm{gap}_F+w_N\varepsilon_s$ | **proved**; quantitative LB needs $\gamma_F$ (not logged) | — |
| R4 transfer | **reference** to `hedge_experts.md` E2(ii) and `full_game.md` Lemmas 1–2 (Lean `composition_step`); trajectory corollary proved; numerically checked | `rt4`: (a)–(c) hold on 200/200 random trajectories (min slack 0.035) |
| R5a Hoeffding/union/fresh-seed LCBs | **proved** (textbook); numerically checked | `rt5`: wb2 → top move LCB 0.051 ($m=1$), 0.005 ($m=3$); winner's-curse bias $+0.067$ ($m=3$, $n=20$) |
| R5b gain identity and $\mathrm{gain}\le\operatorname{expl}(\pi)+\operatorname{expl}(\beta)$ | **proved** | — |
| R5c bracket | **proved** (combines R5a with Theorem A); upper side vacuous for current leaves | — |
| R6a RM+ affine invariance; calibrated Theorem A | **proved**; numerically checked | `rt6` (a): $2\times10^{-12}$ |
| R6b inversion bound, $a=0$ criterion, exact-terminal cell bound | **proved**; numerically checked | `rt6` (b): 0/500, 14/14, 500/500 |
| R6c loop example, cap exactness, residual form | **proved**; numerically checked | `rt6` (d) |
| Removing the 0.029 by turn-indexing or a terminating tie-break | **conjecture / suggestion** | — |

## Refuted / superseded (kept so they are not re-derived)

- "CBL Thm 2.3 applies verbatim to `exp3_live`": the code uses $\eta_{t-1}$, not $\eta_t$. It is superseded by (R2.1), which is smaller.
- "The exploiter's surrogate advantage grows with the learner's MCTS share": not supported by the data (R3b, item 5). The reset learner's extra exposure is through $w_F$ alone.
- "Any calibration, per turn, may be used in E2": false for the Hedge learner. Per-turn slopes re-create the per-turn normaliser (R2, Remark 2). Per-state calibration is fine for Theorem A.
- "Stage-wise equilibria of the correct $V^*$ are unexploitable in the uncapped game": false (R6c(3)), even at the least fixed point.
