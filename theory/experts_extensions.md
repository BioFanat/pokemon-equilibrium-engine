# Experts-problem extensions for the new components (style posterior, Bayes-safex, bandit roots, K worlds)

Companion to `theory_map.md` (notation §0, assumptions A1–A12, status vocabulary §0.4). Lectures: `../lecture6.pdf`
(experts problem, MWU, adversary model) and `../lecture7.pdf` (no-regret dynamics). Code: `engine/showdown/behavior.py`
(style posterior), `engine/showdown/run_ourbot.py` (`hedge_bayes`, `bayes_safex`, `bandit_*`, `rmplus_avg`,
`bandit_selfplay` :146–181), `engine/nash.py:244–262` (`safex_strategy`). Numeric checks:
`/private/tmp/claude-501/-Users-biofanat-Documents-classes-f26-cs535/2bbe670b-2f97-4d9a-bfad-e1139e0352c6/scratchpad/experts_ext/`
(`bayes_hedge_check.py`, `bayes_B_recheck.py`, `safex_lipschitz.py`, `bandit_bounds.py`, `parse_e4.py`, `worlds_K.py`,
`make_real_matrices.py`; outputs `out_*.txt`). Written 2026-11-15; **this text predates the BH, E4emv and E4umv runs** (results listed after the measured facts) and has since been corrected by `verifier` (evidence in `scratchpad/verify_ext/`; edits marked below). Code facts used in B6 were read directly:
`run_ourbot.py:393–403` (the $\hat q$-safex strategy replaces `Q[3]` before `p = wts @ Q` and before `pending` is
stored) and `exp3_live.py:60–70` (`t += 1`, then `logw = sqrt(8 ln k / t) * cum`, the R2a indexing). No research
ledger is enabled in this project (`.claude/ledger/config.json` absent); all numbers below are scratch probes.

**Measured facts used below (sources, not RESULTS.md, which has none of these numbers).**
`showdown/results/E4e/E4e_summary.json`: bandit_exp3 (shares) vs rmplus_avg pair score 0.0625 [0.043, 0.082], 296
pairs, mean 25.9 turns; logged stage gap (A01 units) mean 0.0861, **median 0.0585**, 52.8% > 0.05.
`E4u_summary.json`: bandit_ucb1 0.0502 [0.031, 0.069], 299 pairs, 26.0 turns; gap mean 0.0926, median 0.0578.
`E2_summary.json` / `E2fs_summary.json`: hedge 0.3108, hedge_fs 0.3033 vs stock foul-play, 399 pairs each; **16.4% /
16.8% of feedback turns skipped** (`not_in_worlds` 2155 / 2160 of ≈27,300, the rest fainted / no_action / cant).
`AF_summary.json`: agent form 0.3144 vs stock; `E1_summary.json`: 0.5587 vs PIMC. `results/belief/showdown.json`:
foul-play exact posterior mean P(truth) tera 0.610 / item 0.839 / move 0.815 / ability 0.969; K=4 worlds absent
fraction (`frac_p0`) 0.0956 / 0.0616 / 0.0306 / 0.0058; log loss exact 0.627 / 0.257 / 0.273 / 0.045 vs K=4 1.127 /
0.614 / 0.447 / 0.077. Decision logs (`*.decisions.jsonl`, parsed by `parse_e4.py`): rows $m$ median 7 (p90 13),
**columns $n$ median 18 (p90 27, max 44)**, 29.8 bandit decisions with a logged gap per game.
`LAM=1.0` is the default, so every shipped posterior uses no forgetting.
**New results (run after this document was written; the text of B1–B12 predates them unless marked).**
BH (`hedge_bayes` vs stock): 0.3329 [0.302, 0.364], 398 pairs; paired against E2 hedge: +0.024 [−0.018, +0.066]
(n.s.). E4emv (`bandit_exp3_mv` vs `rmplus_avg`): 0.4949 [0.455, 0.535], 295 pairs. E4umv (`bandit_ucb1_mv`): 0.5933
[0.555, 0.631], 300 pairs. No `bayes_safex` game has been run.

---

## 0. Setting and notation

Per turn $t$ (one decision node, A12) the bot builds $K=4$ worlds, aligned world matrices $M_{t,1..K}$ (rows = our
options, columns = opponent labels, own perspective, maximiser) and the world-averaged matrix $\bar A_t$
(`avg_matrix`, per-column renormalised over the worlds that contain the column). $\Delta_t=\operatorname{ptp}(\bar A_t)$
(engine units; 1 if constant). $v_t=\operatorname{val}(\bar A_t)$. The opponent's realised column is $b_t$; its law
given the public history is $\sigma(h_t)$ (A1). A **feedback turn** is a turn whose $b_t$ is mapped to a column of some
world; the others (16–17% on E2) are **skipped**: no Hedge update, no posterior update (`run_ourbot.py:323–326`).

**Styles.** $k_s=5$ styles $s\in\{\mathrm{eq},\mathrm{br\_unif},\mathrm{br\_us},\mathrm{safe},\mathrm{uniform}\}$; style $s$
at turn $t$ is a distribution $q_{s,t}\in\mathcal P(\text{columns of }\bar A_t)$ computed from $(\bar A_t,p^{\rm ref}_t)$
*before* $b_t$ is seen (`behavior.decide`, stored in `pend`), with $q_{s,t}\ge\mathrm{MIX}/n_t$ entrywise
($\mathrm{MIX}=0.05$, $n_t$ columns). Log loss $\ell_{s,t}=-\ln q_{s,t}(b_t)\in[0,\ln(n_t/\mathrm{MIX})]$. Posterior
$P_t$, prediction $\hat q_t=\sum_sP_{t-1}(s)q_{s,t}$, loss $\hat\ell_t=-\ln\hat q_t(b_t)$.

**Safex LP** (`nash.py:260`): $x^{\varepsilon}(\bar A,q)\in\arg\max\{x^\top\bar Aq:\ x\in\mathcal P(\text{rows}),\
(x^\top\bar A)_j\ge v-\varepsilon\Delta\ \forall j\}$, feasible set $F_\varepsilon(\bar A)$ (contains the maximin
strategy, so never empty); $g_{\bar A,\varepsilon}(q)=\max_{x\in F_\varepsilon}x^\top\bar Aq$. Shipped
$\varepsilon=0.003$ (`SAFEX_EPS`). Fallback to the LP equilibrium if the solver fails (0 of 2,856 calls in the check).

**Bandit roots.** $A^{01}_t=\tfrac12(1+\tanh(\bar A_t/160))\in[0,1]^{m\times n}$; `bandit_selfplay` runs $N=2000$
iterations of decoupled Exp3 (anytime $\eta_\tau=\sqrt{\ln k/(k\tau)}$, importance-weighted losses, **no forced
exploration**) or UCB1 ($\sqrt{2\ln\tau/c}$ bonus) with Bernoulli$(A^{01}_{ij})$ feedback; returns visit shares
$(\bar x_N,\bar y_N)$. `bandit_*` plays $a_t\sim\bar x_N$; `bandit_*_mv` plays $\arg\max\bar x_N$; `rmplus_avg` plays
$a_t\sim$ RM+(1000) average. Logged `gap` $=\max_i(A^{01}\bar y)_i-\min_j(\bar x^\top A^{01})_j$ (both sides, A01 units).

Units are kept apart throughout: **engine units** ($\bar A$, $\Delta_t\approx230$ median), **A01 units** (range 1),
**Hedge reward units** ($/400$, A9), **payoff units** ($[-1,1]$, $\Delta=2$), **score** ($[0,1]$). The map from any
engine quantity to payoff is unknown (A7); every whole-game statement below inherits $\varepsilon_v$.

---

## 1. The style posterior is Hedge with $\eta=1$ on log loss

### Statements

**Lemma B1 (Bayes = exponential weights, pathwise $\ln k_s$ regret).** [M0 (M3 inputs); L6; **cite** (Cesa-Bianchi &
Lugosi 2006 §3.3 exp-concave losses, §9 log loss / Bayes mixture) + adaptation **proved**; numeric]
*Assumptions.* (B1.1) At every feedback turn the style matrix $(q_{s,t})_s$ is a function of the public history
before $b_t$ (true: it is computed from $\bar A_t$ and $p^{\rm ref}_t$ at decision time); $b_t$ may be chosen by an
adversary who knows $(q_{s,t})_s$ and $P_{t-1}$ (A1 reading of Lecture 6 §2.1). (B1.2) $\lambda=1$ (shipped default).
(B1.3) Sums run over the $T$ feedback turns only; skipped turns change nothing (`update` is not called).
*Statement.* (i) $P_t(s)\propto\exp\big(-\sum_{u\le t}\ell_{s,u}\big)$: the posterior is Lecture 6's MWU in its
exponential form with learning rate $\eta=1$, the uniform prior being the initial weights. (ii) For every sequence
$(b_t)_{t\le T}$ and every style $s$:
$$\sum_{t\le T}\hat\ell_t-\sum_{t\le T}\ell_{s,t}\ \le\ \ln k_s=\ln5\approx1.609,$$
with equality in the limit when one style dominates (the bound is attained to 4 decimals in the check). The bound is
anytime (holds for every prefix) and needs no bound on the losses.
(iii) Lecture 6's MWU bound, applied after rescaling the losses to $[0,1]$ by $B_t=\ln(n_t/\mathrm{MIX})$ ($\le5.56$ at
$n_t=13$, the random-matrix check only; losses reach $\ln(44/0.05)=6.78$ at the logged maximum $n=44$), gives only $\hat L\le(1+\eta)L^*+B\ln k_s/\eta$; (ii) is sharper because log loss is 1-mixable (exp-concave):
the Hoeffding step of Lemma H is replaced by an identity.

*Proof.* (i) `Posterior.update` sets $\log P_t=\lambda\log P_{t-1}+\log q_{\cdot,t}(b_t)-\text{const}$; with
$\lambda=1$ and $\log P_0=-\ln k_s$ this unrolls to $\log P_t(s)=-\ln k_s-\sum_{u\le t}\ell_{s,u}-\sum_u\text{const}_u$,
and the constants cancel in the normalisation. (ii) Prequential chain rule: $\hat q_t(b_t)=\sum_sP_{t-1}(s)q_{s,t}(b_t)$
and $P_{t-1}(s)=\frac1{k_s}\prod_{u<t}q_{s,u}(b_u)\big/\sum_{s'}\frac1{k_s}\prod_{u<t}q_{s',u}(b_u)$, so
$\prod_{t\le T}\hat q_t(b_t)=\frac1{k_s}\sum_s\prod_{t\le T}q_{s,t}(b_t)\ge\frac1{k_s}\prod_{t\le T}q_{s^*,t}(b_t)$ for
every $s^*$. Take $-\ln$. Nothing about how $b_t$ was chosen is used, so the adversary of (B1.1) is covered; the
history dependence of $q_{s,t}$ (through $\bar A_t$ and $p^{\rm ref}_t$, which depends on the Hedge weights) is allowed
because $q_{s,t}$ is fixed before $b_t$. (iii) is Lecture 6 §4 with $l^t_i=\ell_{s,t}/B$. $\square$

**Lemma B2 (fixed share on log loss: tracking a switching style).** [M0; L6; **proved** (Lemma H with $\eta=1$ and
the mixability identity; cf. Herbster & Warmuth 1998)]
*Assumptions.* (B1.1), (B1.3); the posterior step is replaced by $v_s\propto P_{t-1}(s)e^{-\ell_{s,t}}$,
$P_t=(1-\alpha)v+\alpha/k_s$ (**not** the shipped `lam`; see B3).
*Statement.* For every comparator sequence $(s_t)$ with $m$ switches,
$$\sum_{t\le T}(\hat\ell_t-\ell_{s_t,t})\ \le\ \ln k_s+m\ln\frac{k_s}{\alpha}+(T-m)\ln\frac1{1-\alpha}.$$
With $\alpha=m/T$ this is $\approx(m+1)\ln k_s+m\ln(T/m)+m$. There is no $\eta T/8$ term (contrast Lemma H).
*Proof.* As Lemma H: $Z_t=\sum_sP_{t-1}(s)e^{-\ell_{s,t}}=e^{-\hat\ell_t}$ exactly (mixability), so
$\ln P_t(s_t)\ge\ln P_{t-1}(s_t)-\ell_{s_t,t}+\hat\ell_t+\ln(1-\alpha)$ when $s_{t}=s_{t-1}$, and
$\ln P_t(s_t)\ge\ln(\alpha/k_s)+\ln P_{t-1}(s_{t-1})-\ell_{s_{t-1},t}+\hat\ell_t$ at a switch (since
$v_{s_{t-1}}\le1$). Sum, $\ln P_0=-\ln k_s$, $\ln P_T\le0$. $\square$

**Proposition B3 (the shipped forgetting $\lambda<1$ is exponent discounting, not fixed share).** [M0; L6; (a) **proved**;
(b) **proved**; (c) **REFUTED for $\lambda<\tfrac12$, open for $\lambda\ge\tfrac12$**]
*Setting.* `update` with $\lambda<1$ gives $P_t\propto\exp(-D_t)$, $D_t=\lambda D_{t-1}+\ell_t$, i.e.
$D_t(s)=\sum_{u\le t}\lambda^{t-u}\ell_{s,u}$ (asserted against `behavior.Posterior(lam=0.8)` in the check). The code
predicts with $P_{t-1}\propto e^{-D_{t-1}}$ ("predict, then discount").
*Statement.* (a) The telescoping of B1 breaks ($D_{t-1}+\ell_t\ne D_t$), and no sublinear undiscounted regret bound
against a fixed style holds for $\lambda<1$: $D_t(s)\le\max_u\ell_u/(1-\lambda)$ is bounded, so every weight stays in a
fixed interval $[\underline w,\overline w]\subset(0,1)$; on a sequence where style $s^*$ beats the others by a fixed
margin at every turn the per-turn excess $\hat\ell_t-\ell_{s^*,t}$ is then bounded below by a positive constant, and
the undiscounted regret is linear in $T$ (the comparator is forgotten at rate $\lambda$).
(b) **Discount-then-predict variant.** If instead the prediction uses $P'_{t}\propto e^{-\lambda D_{t-1}}$ (one line:
raise the stored posterior to the power $\lambda$ before predicting), then for every $T$ and every style $s$ the
**discounted regret** satisfies
$$\sum_{t\le T}\lambda^{T-t}\big(\hat\ell'_t-\ell_{s,t}\big)\ \le\ \ln k_s .$$
(c) For the coded variant the same inequality held on all 360 adversarial and random paths (anytime, max 1.52 at
$\lambda\in\{0.9,0.7\}$, bound 1.609) but the per-step potential inequality needs
$\big(\tfrac1{k_s}\sum_sa_s\big)\big(\tfrac1{k_s}\sum_se^{-\ell_s}\big)\le\tfrac1{k_s}\sum_sa_se^{-\ell_s}$ (positive
correlation of weights and likelihoods), which an adversary can violate. **REFUTED for $\lambda<\tfrac12$** (verifier,
`verify_ext/b3c_counter.py`): the coded predict-then-discount variant has discounted regret 3.82 at $\lambda=0.3$ and
2.25 at $\lambda=0.4$, against $\ln5=1.609$. **Open for $\lambda\ge\tfrac12$.** Discount-then-predict (B3(b)) is
Chernov–Zhdanov's AAD, bound $\ln K/\eta$.
*Proof of (b).* $F_t:=-\ln\frac1{k_s}\sum_se^{-D_t(s)}$, $F_0=0$, $F_T\le D_T(s)+\ln k_s$ for every $s$. With
$a_s=e^{-D_{t-1}(s)}$: $\hat\ell'_t=\ln\frac1{k_s}\sum_sa_s^\lambda-\ln\frac1{k_s}\sum_sa_s^\lambda e^{-\ell_{s,t}}=\ln\frac1{k_s}\sum_sa_s^\lambda+F_t$,
and $\lambda F_{t-1}=-\ln\big(\frac1{k_s}\sum_sa_s\big)^\lambda$. Jensen for the concave $x\mapsto x^\lambda$ gives
$\frac1{k_s}\sum_sa_s^\lambda\le(\frac1{k_s}\sum_sa_s)^\lambda$, hence $\hat\ell'_t\le F_t-\lambda F_{t-1}$. Multiply
by $\lambda^{T-t}$ and sum: $\sum_t\lambda^{T-t}\hat\ell'_t\le F_T-\lambda^TF_0=F_T$. $\square$
*Reading.* (b) is the Lecture 6 guarantee on an exponentially weighted window of effective length $1/(1-\lambda)$
(cf. Chernov & Zhdanov 2010, discounted prediction with expert advice, **cited from memory**). For a tracking guarantee
over the whole run (best *switching* style) use fixed share (B2), which is Lemma H's mechanism with $\eta=1$.

**Corollary B4 (log loss and prediction error).** [M0; L6; (i) **proved**; (ii)–(iii) **proved** under realisability;
numeric]
(i) *Agnostic.* Average log loss $\frac1T\sum_t\hat\ell_t\le\min_s\frac1T\sum_t\ell_{s,t}+\frac{\ln k_s}T$ on every path;
with (A1) and the tower rule, $\mathbb E\sum_t\mathrm{KL}(\sigma_t\,\|\,\hat q_t)\le\min_s\mathbb E\sum_t\mathrm{KL}(\sigma_t\,\|\,q_{s,t})+\ln k_s$,
where $\sigma_t$ is the opponent's conditional law given $h_t$ **and the full feedback event** (verifier: not only
$b_t\in C_t$; censoring by fainted / cant / no_action, 8.5% of E2 feedback records, depends on $b_t$'s priority, our
$a_t$ and chance, so $\sigma_t$ must be conditioned on all of it, as in B6(c)) (the law
$\sigma(h_t)$ renormalised to the worlds' column set; the update only happens on that event): the posterior's
cumulative KL to the in-support truth is at most $\ln5$ worse than the best style's.
(ii) *Realisable* ($\sigma_t=q_{s^*,t}$ for one style $s^*$ at all feedback turns): $\mathbb E\sum_t\mathrm{KL}(q_{s^*,t}\|\hat q_t)\le\ln k_s$,
hence by Pinsker and Cauchy–Schwarz
$$\mathbb E\sum_{t\le T}\lVert\hat q_t-q_{s^*,t}\rVert_1\ \le\ \sqrt{2T\ln k_s},\qquad\frac1T\mathbb E\sum_t\lVert\hat q_t-q_{s^*,t}\rVert_1\le\sqrt{2\ln k_s/T}.$$
(iii) *Accuracy.* $\Pr_{b\sim q^*}(\arg\max\hat q=b)\ge\Pr_{b\sim q^*}(\arg\max q^*=b)-\lVert\hat q-q^*\rVert_1$, so under
(ii) the posterior's average top-1 accuracy is within $\sqrt{2\ln k_s/T}$ of the true style's (0.33 at $T=30$, 0.057
at $T=1000$). Log-loss regret gives **no** accuracy statement without realisability (a style can have the best log
loss and the wrong argmax).
*Proof.* (i) is B1(ii) divided by $T$; the KL form is $\mathbb E[\hat\ell_t-\ell_{s,t}\mid h_t]=\mathrm{KL}(\sigma_t\|\hat q_t)-\mathrm{KL}(\sigma_t\|q_{s,t})$.
(ii) the second KL vanishes; Pinsker $\lVert p-q\rVert_1^2\le2\,\mathrm{KL}$ and $\sum_t\sqrt{x_t}\le\sqrt{T\sum_tx_t}$.
(iii) $q^*(a^*)-q^*(\hat a)\le[q^*(a^*)-\hat q(a^*)]+[\hat q(\hat a)-q^*(\hat a)]\le\lVert\hat q-q^*\rVert_1$ with
$a^*=\arg\max q^*$, $\hat a=\arg\max\hat q$ (use $\hat q(\hat a)\ge\hat q(a^*)$). $\square$

### Numeric check (`bayes_hedge_check.py`, `bayes_B_recheck.py`; 4.6 s + 12.5 s)
Style matrices from `behavior.style_dists` on 400 random $N(0,150^2)$ matrices ($2$–$13$ rows/columns) and the 57 real
`nash.full_matrix` stage matrices (`make_real_matrices.py`), posterior = `behavior.Posterior`.
- A. Weights equal Hedge($\eta=1$) weights to $10^{-10}$ at every step (asserted). Regret vs the best style over 60
  turns: max **1.6094** on 200 paths each for the worst-column, random and anti-leader adversaries (bound $\ln5=1.6094$);
  anytime max also 1.6094. Largest single loss 5.56 $=\ln(13/0.05)$.
- B. Realisable, 3,000 paths: $\mathbb E\sum\mathrm{KL}=1.532\pm0.025$ ($T=30$), $1.624\pm0.031$ ($T=300$); expected
  regret vs $s^*$ $1.497$ / $1.609\pm0.000$ (the style is identified; the bound is tight). Mean $\sum_t\lVert\hat q_t-q_{s^*,t}\rVert_1$
  = 4.54 at $T=30$ (bound 9.83) and 5.27 at $T=300$ (bound 31.1): the Pinsker route is loose by 2–6× because the
  posterior concentrates exponentially, not at the $\sqrt T$ rate.
- C. Fixed share, comparator switching every 25/50 turns: max regret 16.3 / 28.1 (bounds 21.2 / 35.7 at $\alpha=0.01$);
  plain Bayes against the same switching comparator: 36 / 124 (linear in $T$: B1 bounds regret against a *fixed*
style only).
- D. Discounting: both variants satisfied the discounted bound on all paths (max 1.47 coded, 1.52 discount-then-predict).

*Verifier (B1).* Confirmed: the implementation matches; pathwise max regret 1.60944 = $\ln5$. Caveat for §8's test
"one violation refutes the implementation": first check that the update counts in the logs and state files match.
Streams 0 and 6 of BH had reruns that overwrote logs (2344 vs 2266, 2114 vs 2069 updates), and stream 6 reads 5.03 for
that reason. Chernov–Zhdanov 2010 and Bubeck–Cesa-Bianchi Thm 3.1 citations confirmed (B3, B7).

*Status.* B1 **cite + adaptation** (the only new content is the exact placement in our turn structure: history-dependent
styles, skipped turns, the 5% mix making losses finite, the comparison with Lecture 6's bound). B2 **known** (fixed
share; new only as the $\eta=1$ instance). B3(b) **known** in the discounted-experts literature; B3(c) **refuted for $\lambda<\tfrac12$, open for $\lambda\ge\tfrac12$**. B4 **standard**.

---

## 2. From prediction to payoff: the ε-safe expert and the whole game

### Statements

**Lemma B5 (safex: floor independent of the prediction, loss Lipschitz in the prediction).** [M1; L6; **proved**
with (d) corrected by the verifier; numeric, tight]
*Assumptions.* One stage; $\bar A\in\mathbb R^{m\times n}$ with range $\Delta$; $\varepsilon\ge0$; $\hat x=x^\varepsilon(\bar A,\hat q)$
(any maximiser), $q\in\mathcal P$ arbitrary (the opponent's true conditional law, or any column).
*Statement.* (a) *Floor.* $\min_j(\hat x^\top\bar A)_j\ge v-\varepsilon\Delta$ for every $\hat q$; so for every realised
column $b$, $\hat x^\top\bar A[:,b]\ge v-\varepsilon\Delta$ (within LP tolerance; measured slack $\ge-7.7\cdot10^{-16}\Delta$).
(b) *Lipschitz value.* $|g(q)-g(\hat q)|\le\tfrac\Delta2\lVert q-\hat q\rVert_1=\Delta\,\mathrm{TV}(q,\hat q)$.
(c) *Loss of predicting wrongly.* $g(q)-\hat x^\top\bar Aq\le\Delta\lVert q-\hat q\rVert_1=2\Delta\,\mathrm{TV}(q,\hat q)$,
and the constant is **sharp** (verifier): the $I_2$ witness below binds the floor at the shipped $\varepsilon=0.003$
and gives ratio 0.006, so it is sharp only at $\varepsilon=0$. A witness with a constant column,
$\bar A=\begin{pmatrix}1&0&0\\0&1&0\end{pmatrix}$, gives ratio 1.000 at $\varepsilon=0$ and 0.003. Original witness
(sharp at $\varepsilon=0$ only): $\sup=1$, approached by $\bar A=I_2$ ($\Delta=1$), $\hat q=(\tfrac12-\delta,\tfrac12+\delta)$
(so $\hat x=e_2$ for every $\delta>0$, no tie-breaking needed), $q=e_1$: loss $1$, $\lVert q-\hat q\rVert_1=1+2\delta$,
ratio $1/(1+2\delta)\to1$. What binds at small $\varepsilon$ is the sharper bound
$\text{loss}\le g(q)-v+\varepsilon\Delta$. (c) can fail on the LP-fallback path (solver failure); this was seen 0 times
in 5,700+ calls.
(d) *Gain.* $g(q)\ge v$ always ($F_\varepsilon$ contains the maximin strategy); $g$ is non-decreasing and **concave**
in $\varepsilon$ (the LP value is concave in its right-hand side; 35/40 real matrices show strictly concave kinks, 0
convex; the earlier "convex" was wrong), and convex in $q$ (the gain curve of `theory_map.md` §6.3).
*Proof.* (a) is the LP constraint. (b) $g$ is a maximum of linear functions $x^\top\bar Aq$ over a fixed set, so for
the maximiser $x_q$ at $q$: $g(q)-g(\hat q)\le x_q^\top\bar A(q-\hat q)=\sum_j c_j(q_j-\hat q_j)$ with $c_j\in[\min\bar A,\max\bar A]$;
since $\sum_j(q_j-\hat q_j)=0$ we may centre $c$, giving $\le\tfrac\Delta2\lVert q-\hat q\rVert_1$; symmetric. (c)
$g(q)-\hat x^\top\bar Aq=[g(q)-g(\hat q)]+[\hat x^\top\bar A\hat q-\hat x^\top\bar Aq]$, each bracket $\le\tfrac\Delta2\lVert q-\hat q\rVert_1$
by the centring argument. Sharpness: $g(e_1)=1$, $\hat x=e_2$ gives $0$; $\lVert e_1-\hat q\rVert_1=1+2\delta\to1=\Delta$ (at $\varepsilon=0$ only; see (c)). $\square$

**Theorem B6 (whole-game statements for `hedge_bayes`).** [M2–M3; L6; **proved with named hypotheses**; (c) proved
after the hypothesis fix below; the regret step is Lemma R2a, the composition is L2 / Theorem E2]
*Setting.* Experts $e\in\{\text{top},\text{sampled},\text{safest},\text{safex}(\varepsilon,\hat q_t)\}$, $k=4$; anytime
FTRL $w_t\propto\exp(\eta_t\sum_{u<t}r_u)$, $\eta_t=\sqrt{8\ln k/t}$ (Lemma R2a; note the code's step is R2a's $\beta_t=\sqrt{8\ln k/(t-1)}$, not $\eta_t=\sqrt{8\ln k/t}$; so pathwise
$\mathrm{Reg}_T\le R(T):=\tfrac32\sqrt{T\ln k/2}+\sqrt{\ln k/8}$ in reward units); rewards
$r_{t,e}=\operatorname{clip}\big((q_{e,t}\cdot M_{t,\kappa_t}[:,b_t]-\min M_{t,\kappa_t})/\bar\Delta,0,1\big)$,
$\bar\Delta=400$, scored on the single world $\kappa_t$ that contains $b_t$ with the highest weight
(`run_ourbot.py:321,332`); the safex expert is built on $\bar A_t$ with the prediction $\hat q_t$ computed from
$p^{\rm ref}_t=w_t\cdot Q_t^{\rm old}$ before $b_t$.
*Assumptions.* A1, A6, A9 (fixed normaliser), A12; (B6.1) **scoring error**: $\lVert M_{t,\kappa_t}[:,b_t]-Q^*_t[:,b_t]\rVert_\infty\le\varepsilon_v$
and $\lVert\bar A_t-Q^*_t\rVert_\infty\le\varepsilon_v$ on the payoff scale after one global affine calibration
(A7 + E2′'s (B2); the choice of $\kappa_t$ depends on $b_t$, which is harmless for the *pathwise* regret but is part
of $\varepsilon_v$); (B6.2) **no clipping** ($\operatorname{range}\le\bar\Delta$; violated on ≈0.5% of turns, RESULTS
`exp3_live` bullet; add the clipping loss $C_S$ of E2′ otherwise); (B6.3) skipped turns: $T_{\rm skip}$ of the $T_{\rm all}$
live turns give no feedback; on them the played mixture has no guarantee beyond the payoff range.
*Statement.* Let $u$ be the game payoff, $V^*$ the value, $\mathbb E$ over the match (sums over games; $T$ = feedback
turns, $T_{\rm all}$ = all live turns).
*Units.* The left side is on the payoff scale; the Hedge regret is in reward units ($400$ engine units each), so it
enters through the global calibration slope $a$ of E2′'s hypothesis (B2): $a\cdot400\cdot R(T)$ payoff units. The map
records $a=0$ on the measured exact endgames (E2′ "does not apply on real leaves"); with $a=0$ the statements below
are surrogate inequalities in engine units (Proposition E1's form with $\bar A_t$ in place of $Q^*$), not payoff
bounds. We write $\bar\Delta_a:=a\cdot400$.
(a) **Safety, route 1 (through the sampled expert; L2(b)).** Under (E) for the sampled expert, where
$\varepsilon_s$ is the agent-form strategy's loss in the **single scoring world** $M_{t,\kappa_t}$ (Theorem B's
decomposition / the blend-loss bound, numeric $4\delta_H$), not the $10^{-5}$ solver gap:
$$\mathbb E\big[V^*-u\big]\ \le\ (\varepsilon_s+2\varepsilon_v)\,\mathbb E[T]+\bar\Delta_a\,\mathbb E[R(T)]+\Delta\,\mathbb E[T_{\rm skip}].$$
The skipped-turn term alone is $\approx2\times0.164\times30\approx10$ payoff units at the E2 skip rate: vacuous; it is
the price of certifying nothing on turns without feedback.
(b) **Safety, route 2 (through the safex expert; §6.3 of the map).** With no hypothesis on the sampled expert, using
Lemma B5(a) in $\bar A_t$ as hypothesis (E) with $\varepsilon_s\to\varepsilon\Delta_t$ (converted to the payoff scale):
$$\mathbb E\big[V^*-u\big]\ \le\ \mathbb E\Big[\sum_{t\le T}(\varepsilon\Delta^{\rm pay}_t+\varepsilon_{s,t})\Big]+2\varepsilon_v\mathbb E[T]+\bar\Delta_a\,\mathbb E[R(T)]+\Delta\,\mathbb E[T_{\rm skip}],$$
where $\varepsilon_{s,t}$ is the LP's own solver gap (≈0). The **$\varepsilon\cdot$range term appears on route 2 only**; the
task's formula with both $(\varepsilon_s+2\varepsilon_v)$ and $\varepsilon\cdot$range double counts (a judgement about
two different certifying experts, not a theorem): take the minimum of (a), (b).
(c) **Gain (realisable).** Whether turn $t$ is a feedback turn is decided by $b_t$ (it must land in the column set
$C_t$ of the worlds), so conditioning on $h_t$ alone gives, with $\pi_t=\sigma_t(C_t^c)$ and $\sigma^{\rm in}_t$ the law
$\sigma_t$ renormalised to $C_t$: $\mathbb E[\mathbb 1\{b_t\in C_t\}\,\hat x_t^\top\bar A_t[:,b_t]\mid h_t]=(1-\pi_t)\,\hat x_t^\top\bar A_t\sigma^{\rm in}_t$.
Realisability is therefore a statement about $\sigma^{\rm in}_t$ (verifier fix: it must be stated for the law of $b_t$
given $h_t$ **and the full feedback event**, not only $b_t\in C_t$: censoring by fainted / cant / no_action, 8.5% of E2
feedback records, depends on $b_t$'s priority, our $a_t$ and chance; B4(i)'s $\sigma_t$ needs the same change): if
$\sigma^{\rm in}_t=q_{s^*,t}$ at every turn, then
with $g_t=g_{\bar A_t,\varepsilon}$:
$$\mathbb E\sum_{t\le T_{\rm all}}(1-\pi_t)\,p_t^\top\bar A_t\sigma^{\rm in}_t\ \ \ge\ \ \mathbb E\sum_{t\le T_{\rm all}}(1-\pi_t)\,g_t(q_{s^*,t})\;-\;\bar\Delta_{\max}\sqrt{2\,\mathbb E[T]\ln k_s}\;-\;\bar\Delta\,\mathbb E[R(T)]\;-\;2\varepsilon_v\mathbb E[T]$$
(engine units for the first three terms, $\bar\Delta_{\max}=\max_t\Delta_t$, $T$ the number of feedback turns; the last
term is the scoring-matrix error). The left side is the expected surrogate payoff earned on feedback turns; the first
term on the right is the oracle safe-exploitation gain against the true (in-support) style, $\ge\sum_t(1-\pi_t)v_t$ by
B5(d). This is the same restriction the map puts on applying P1 to logs ("$\bar A_t$ not depending on $b_t$, skipped
turns counted").
(d) **Agnostic gain.** Replace $q_{s^*,t}$ by $\sigma^{\rm in}_t$ and $\sqrt{2T\ln k_s}$ by
$\sqrt{2T(\min_s\mathbb E\sum_t\mathrm{KL}(\sigma^{\rm in}_t\|q_{s,t})+\ln k_s)}$: the posterior pays the best style's
misspecification plus $\ln5$.
*Proof.* (a) is L2(b) / Theorem E2 verbatim with R2a's constant; the extra term is (B6.3): on a skipped turn
$V^*_t-p_t^\top Q^*_t\sigma_t\le\Delta$, and L1(a) sums the stage deficits over **all** live turns. (b) Lemma B5(a) says
the safex expert is $\varepsilon\Delta_t$-optimal in $\bar A_t$ for every column, so hypothesis (E) holds for it with
that budget; then Lemma 1 (two matrices) and L0 as in E2 step (i). (c) Hedge's pathwise regret against the safex
expert: $\sum_tp_t\cdot r_t\ge\sum_tr_{t,\text{safex}}-R(T)$; in engine units
$\sum_tp_t^\top M_t[:,b_t]\ge\sum_t\hat x_t^\top M_t[:,b_t]-\bar\Delta R(T)$; pass from $M_{t,\kappa_t}$ to $\bar A_t$
at the cost $2\varepsilon_v$ per turn; the tower rule (A1: $b_t\sim\sigma_t$ given $h_t$, $\hat x_t$ and $C_t$ are
$h_t$-measurable) gives $\mathbb E[\mathbb 1\{b_t\in C_t\}\hat x_t^\top\bar A_t[:,b_t]\mid h_t]=(1-\pi_t)\hat x_t^\top\bar A_t\sigma^{\rm in}_t\ge(1-\pi_t)[g_t(\sigma^{\rm in}_t)-\Delta_t\lVert\sigma^{\rm in}_t-\hat q_t\rVert_1]$
by B5(c); **keep the weights $(1-\pi_t)$** (verifier correction: do not drop them) and apply weighted
Cauchy–Schwarz together with B4(ii) (whose KL identity holds conditional on the feedback event, with $\sigma^{\rm in}_t$
in place of $\sigma_t$) to $\mathbb E\sum_t(1-\pi_t)\lVert q_{s^*,t}-\hat q_t\rVert_1$; the bound $\sqrt{2\mathbb E[T]\ln5}$ is unchanged.
(d) is the same with B4(i). $\square$
*Remarks.* (1) **P1 does not apply across games**: the safex expert depends on the carried posterior, so (P1.3)
"experts fixed across games" fails; within one game with a reset posterior it is a history-dependent expert and P1
holds as stated. External regret (what Hedge bounds) is not policy regret (P1(iv)). P1 also fails across games through
the carried Hedge weights (via $p^{\rm ref}$ and `br_us`), not only through the posterior. (2) **Sizes.** Per game
($T\approx25$ feedback turns) the gain slack $\bar\Delta_{\max}\sqrt{2T\ln5}\approx9\,\Delta_t$: vacuous. Carried over the
E2 budget ($T\approx2.3\cdot10^4$ feedback turns): $\sqrt{2T\ln5}/T\approx0.012$ range per turn (0.018 at $T=10^4$), against an
oracle gain $g_t(q)-v_t$ of median 0.049 range at $\varepsilon=0.003$ (0.065 at 2%) (**SYNTHETIC**, from the
`safex_lipschitz.py` probe on styles and Dirichlet $q$, not measured from logs) — non-vacuous only amortised over
hundreds of games (same conclusion as R3d/AdaHedge). The regret term $\bar\Delta R(T)$ is $\approx\tfrac32\sqrt{T\ln4/2}$
$\approx190$ reward units at $T=2.3\cdot10^4$, i.e. 0.008 of the 400-unit normaliser per turn. (3) The floor (a)/(b) is
in $\bar A_t$ (blend of worlds); it transfers to the true stage matrix through $\varepsilon_v$, which already contains
Theorem C's sampling and belief terms (and the blend-loss bound, numeric $4\delta_H$). The 2% certificate did not
survive M1→M2 (SXW); at 0.3% the add-on is $0.3\%\times\Delta\times\bar H\approx0.1$ score (certificate arithmetic),
the only part of (b) that is non-vacuous. (4) The skipped-turn term is the Target 4 phenomenon seen from the learner:
`not_in_worlds` (7.9% of feedback turns on E2) is exactly "the true value is in none of the $K$ worlds".

### Numeric check (`safex_lipschitz.py`, 15.5 s)
57 real stage matrices + 300 random, $\varepsilon\in\{0.003,0.02\}$, $q$ a style or Dirichlet, $\hat q$ a perturbation or
a posterior mixture (2,856 LPs, 0 fallbacks). Floor slack $\ge-7.7\cdot10^{-16}$ (units of $\Delta$). Loss ratio
$(g(q)-\hat x^\top\bar Aq)/(\Delta\lVert q-\hat q\rVert_1)$: max 0.212 (bulk), 0.309 (adversarial search on $2$–$4$ square
matrices), **1.000** at the designed near-tie $\delta=10^{-9}$ (B5(c) sharp); 0 negative losses. Lipschitz ratio $|g(q)-g(\hat q)|/(\tfrac\Delta2\lVert\cdot\rVert_1)$
max 0.954 (bound 1). Oracle gain over $v$: median 0.049 / p90 0.178 range at $\varepsilon=0.003$; predicted-$\hat q$ gain
median 0.047, minimum $-0.0025$ ($\ge-\varepsilon$, the floor).

*Status.* B5 **proved with (d) fixed** (concave in $\varepsilon$; sharpness witness replaced), elementary (the floor is Ganzfried & Sandholm 2015 / the map's §6.3; the Lipschitz constant
and its tightness are new here, standard LP sensitivity). B6 **proved with named hypotheses** (B6(c) proved after the hypothesis fix): new is the assembly
(Bayes log-loss regret → Pinsker → LP Lipschitz → Hedge → L2), the two safety routes, and the skipped-turn term;
every ingredient is known. **Realisability is a strong assumption**: the five styles are softmaxes of $\bar A_t$ with
5% noise; a real opponent is at best approximately one of them (route (d)).

---

## 3. Bandit roots

### Statements

**Proposition B7 (visit-share play: identity, pseudo-regret constants, what they bound).** [M0–M1; L7; L4 **proved**;
Exp3 constant **cite** (Bubeck & Cesa-Bianchi 2012 Thm 3.1, **from memory**: $2\sqrt{Nk\ln k}$ for
$\eta_\tau=\sqrt{\ln k/(k\tau)}$; $\sqrt{2Nk\ln k}$ for the tuned fixed step); the gap between pseudo-regret and
$\mathbb E[\text{gap}]$ **cite** (Audibert & Bubeck 2010; Lecture 7 §3.2 last sentence); numeric]
(i) By L4, at every two-sided node $\mathrm{gap}_N:=\max_i(A^{01}\bar y_N)_i-\min_j(\bar x_N^\top A^{01})_j=(\mathrm{Reg}^1_N+\mathrm{Reg}^2_N)/N$
with **realised counterfactual** regrets $\mathrm{Reg}^1_N=\max_i\sum_\tau(A^{01}y_\tau)_i-\sum_\tau x_\tau^\top A^{01}y_\tau$
(the sampling distributions $x_\tau,y_\tau$; the Bernoulli noise is not in this identity). **Verifier:** the identity
needs the **realised one-hot actions** (visit shares $=\mathrm{cnt}/N$); with the sampling distributions it gives the
same result for UCB1 but not for Exp3. Since `bandit_*` plays
$a_t\sim\bar x_N$, its stage exploitability in $A^{01}_t$ is $\le\mathrm{gap}_N$ (the row part of it).
(ii) Exp3's guarantee is on the **pseudo-regret** $\max_i\mathbb E[\cdot]$. In self-play the opponent is adaptive, and
$\mathbb E[\max_i\cdot]\ge\max_i\mathbb E[\cdot]$, so the pseudo-regret constant does **not** bound $\mathbb E[\mathrm{gap}_N]$;
the quantities that do are the high-probability bounds (Exp3.P / Exp3-IX), which B3(b) already found $\ge1$ range
at $k=9$, $N=2000$. The pseudo-regret constants are therefore a *reference scale*, not a theorem about the logged gap:

| $k$ | 4 | 7 | 9 | 11 | 13 | 18 | 27 | 33 |
|---|---|---|---|---|---|---|---|---|
| $2\sqrt{k\ln k/N}$ (anytime, per side) | 0.105 | 0.165 | 0.199 | 0.230 | 0.258 | 0.323 | 0.422 | 0.480 |
| $\sqrt{2k\ln k/N}$ (tuned fixed step) | 0.074 | 0.117 | 0.141 | 0.162 | 0.183 | 0.228 | 0.298 | 0.340 |

With the **logged** shapes (rows median 7, columns median 18, p90 27) the two-sided reference is mean 0.444, median
0.474 A01 units per decision; the logged gap is 0.086 mean / 0.0585 median, i.e. **0.16 of the reference** (median
ratio), and exceeds it on 0.01% of decisions. The gap grows with the column count ($n<8$: mean 0.024; $n\ge27$: 0.185).
UCB1 has **no adversarial guarantee** (L4b lockstep; U1–U4 are large-$N$ statements; H4: at $N\le10^4$ the gap is
unequalised exploration); its logged gap (0.093 / 0.058) is indistinguishable from Exp3's.
(iii) **Whole game (Theorem A).** With $\varepsilon_s(t)=\mathrm{gap}_t$ (converted to payoff units through A7),
$\operatorname{expl}\le\mathbb E\sum_{t<\tau}(\mathrm{gap}_t+2\varepsilon_v)$. The logged sum is **2.56 (E4e) / 2.76 (E4u) A01
units per game** (29.8 two-sided decisions; the logged gap is two-sided, so this is an **upper bound** on Theorem A's
row-side sum): larger than the whole range in any plausible unit map, so the bound is
**vacuous** (a fortiori) and does not explain the 0.05–0.06 scores. The logs also give no support for the mechanism: the
bandit's score *rises* with its mean stage gap (per-game mean-gap quartiles Q1→Q4: 0.007 / 0.034 / 0.054 / 0.154 for
E4e, 0.000 / 0.007 / 0.040 / 0.153 for E4u). This is correlational and confounded: the gap grows with the column count
$n$ ($n<8$: 0.024; $n\ge27$: 0.185), and $n$ tracks the game phase and how many options the bandit still has
(survival). The vacuity conclusion does not rest on it. Conceptually, the gap is exploitability against a *best
response*; the RM+ root is not exploiting (B8).

**Proposition B8 (loss against an equilibrium opponent is the exploration deficit, and max-visit removes it).**
[M0–M1; L7/S; **proved** (identity (i)–(ii)); (iii) a **hypothesis**; **numeric** for the sizes]
*Assumptions.* One stage; the opponent plays an exact stage equilibrium $y^*$ of $A$ (the `rmplus_avg` root, up to its
$10^{-5}$ gap and up to its own worlds' matrix differing from ours — the latter is an $\varepsilon_v$-type term; see
the verifier note under (iii)).
*Statement.* (i) For any mixture $x$, the stage deficit is $v-x^\top Ay^*=\sum_ix_i\,[v-(Ay^*)_i]=\sum_{i\notin\mathrm{BR}(y^*)}x_i\,\mathrm{slack}_i$:
only the mass on non-best-response rows costs, and it costs their slack. It is $\le$ the row stage gap and can be
far smaller. (ii) A pure row $i\in\mathrm{BR}(y^*)$ has deficit 0 even when its stage gap $v-\min_jA_{ij}$ is large. (iii)
By L1(a) the whole-game loss against a stage-equilibrium opponent is $\mathbb E\sum_t(\text{deficit}_t)$ **exact only
if the opponent plays an equilibrium of the matrix in which the deficit is measured; `rmplus_avg` plays an RM+ average
of its OWN squashed depth-2 matrix, which is not such an opponent** (so this is a hypothesis, not a theorem, for the
real opponent). Evidence: E4umv scores 0.593 > 0.5, which a stage-equilibrium opponent would rule out. It is not
$\sum_t\mathrm{gap}_t$.
*Numeric (synthetic, `bandit_bounds.py` §3: 57 real squashed matrices + 40 random, $N=2000$, one seed each).*
Exp3 shares: row stage gap mean 0.047 (pure-saddle matrices, 23% of the real ones) / 0.068 (mixed); **deficit vs
$y^*$ 0.046 / 0.028**. Exp3 max-visit: row stage gap 0.011 / **0.167** (worse than shares on 67% of mixed matrices; artifact of the 40 random matrices, see the verifier note under B8);
**deficit vs $y^*$ 0.010 / 0.0034**. UCB1: shares deficit 0.056 / 0.034; max-visit 0.008 / 0.0023.
*Reading.* Against the RM+ root, shares lose ≈0.03–0.05 A01 units per two-sided decision through exploration mass on
dominated rows; 30 decisions give ≈1–1.5 A01 units, the right order for a 0.44-score loss if A01 ≈ score (A7 makes
this a consistency statement, not an explanation). **Prediction for the `bandit_*_mv` arms (E4 _mv vs
rmplus_avg), as originally written (before the runs):** the max-visit roots lose much less than the shares roots against the RM+ root (deficit 8–10× smaller),
while being more exploitable on mixed matrices (row stage gap 2.5× larger) — the opposite ordering against an
exploiter. Refuted if _mv scores $\le$ shares against rmplus_avg at $n=300$.
*Verifier (pre-registration, `verify_ext/prereg_B8.txt`, made before the runs).* Per-decision deficit 0.039 → 0.008
(Exp3) and 0.046 → 0.007 (UCB1), shares → max-visit, a ratio of 5–8× (the 8–10× above was too large). **Outcome:** the
ordering held: E4emv 0.495 and E4umv 0.593 vs 0.063 and 0.050 for the shares roots. The claim "max-visit 2.5× more
exploitable on mixed matrices" was an artifact of the 40 random Gaussian matrices; on the 57 real matrices the gaps
are 0.053 vs 0.058. It is **UNTESTED** against an exploiter (no exploiter run).

**Proposition B9 (most-visited play: what is provable).** [M0; L7; (i)–(ii) **proved** (elementary); (iii) **cite**;
(iv) **open**]
(i) *Mixed equilibria.* If $A$ has no pure optimal row (every optimal $x$ is strictly mixed), the max-visit row
$i_{\rm mv}$ has stage gap $v-\min_jA_{i_{\rm mv}j}\ge\gamma_A:=\min_i(v-\min_jA_{ij})>0$ for **every** $N$; the
shares' gap is $(\mathrm{Reg}^1+\mathrm{Reg}^2)/N$ and vanishes iff the regret sum is $o(N)$. So max-visit is never
consistent for the stage game, shares are consistent exactly when the selection rule is Hannan-consistent in
self-play (L4).
(ii) *Pure saddles.* If $(i^*,j^*)$ is a pure saddle and $\bar x_N$ puts its maximal mass on $i^*$, max-visit has
stage gap 0 while the shares' row stage gap is $\max_j\sum_i\bar x_i(v-A_{ij})\ge\sum_{i\ne i^*}\bar x_i\,(v-A_{ij^*})$
(the saddle column), which is positive iff exploration mass sits on a row with $A_{ij^*}<v$ (it can be 0 with
exploration mass: $A=\begin{pmatrix}1&1&1\\1&0&1\\1&1&0\end{pmatrix}$, $\bar x=(1-2\epsilon,\epsilon,\epsilon)$ has gap
$\epsilon$ from the off-saddle columns only). No theorem guarantees the premise: L4b (deterministic lockstep UCB1 keeps uniform
shares with a dominant row) is a counterexample; with Bernoulli feedback the premise held on all 13 pure-saddle real
matrices in the check (max-visit gap mean 0.011 vs shares 0.047).
(iii) *Known results* (corrected by the verifier). Kocsis & Szepesvári 2006 Thm 6 is for finite-horizon
**single-agent** MDPs and recommends the max-**mean** root action, not max-visit. Kovařík & Lisý (Machine Learning,
DOI 10.1007/s10994-019-05832-z) show that Hannan consistency alone is insufficient even for the **average** strategy
(Thm 4.1 counterexample); convergence needs UPO plus exploration, or SM-MCTS-A. There is no result about max-visit. Lecture 6
§2.3: a deterministic choice (max-visit) is exploitable at every node; Lecture 7 §2.2 bounds only the averages.
(iv) *Open.* A finite-$N$ statement of the form "max-visit stage gap $\le$ shares gap $+\,o(1)$ on pure-saddle matrices
under Bernoulli UCB1/Exp3 self-play" (the premise of (ii)). Conjecture U3 (plateaus) suggests it may fail for UCB1.

### Numeric check (`parse_e4.py`, `bandit_bounds.py`; 15 s + 5 s)
Logs: 17,820 (E4e) / 17,863 (E4u) two-sided bandit decisions; shapes and gaps as in the header; Theorem A sums and
quartile tables as stated; Exp3 reference constants per decision as in B7. Synthetic: `bandit_selfplay` copied verbatim
from `run_ourbot.py:146–181` (importing `run_ourbot` pulls Showdown dependencies), matrices squashed exactly as the
code ($\tfrac12(1+\tanh(\bar A/160))$); pure saddle tested as $\max_i\min_jA=\min_j\max_iA$.

*Status.* B7(i) is L4; (ii) the pseudo-/expected-regret caveat is known; (iii) the vacuity and the anti-correlation
are **new measurements** on the E4 logs. B8 is elementary but, to our knowledge, not stated in the map; the _mv
prediction is **new and testable**. B9(i)–(ii) elementary; (iii) cite; (iv) open.

---

## 4. Number of sampled worlds $K$ and calibration

### Statements

**Proposition B10 (absent-from-all-worlds probability; Jensen).** [M3; S; **proved under (B10.1); logged item data
contradict (B10.1) for item**; numeric]
*Assumptions.* (B10.1) The $K$ worlds are i.i.d. draws from the belief $b$ (foul-play's set sampler). Verifier:
**holds at the sampler level**: foul-play's `prepare_random_battles` draws each revealed mon's set independently with
`random.choices` on counts (`verify_ext/fp_iid_test.py`). **But** the logged item data are inconsistent with it: K=4
item absent 0.062 vs an upper bin bound of 0.051; mean $\hat p$ 0.816 vs exact 0.839; at k=4 revealed moves all 148
item events have exact $p=1.000$ vs K=4 mean $\hat p$ 0.863. The cause is downstream of sampling and **UNRESOLVED**. (B10.2) the attribute of interest
has posterior mass $p=b(\theta_{\rm true})$ on the truth.
*Statement.* (i) $\Pr(\theta_{\rm true}\notin\text{worlds}\mid p)=(1-p)^K$. (ii) Over a population of events with
$p\sim\mu$: $\Pr(\text{absent})=\mathbb E_\mu(1-p)^K\ \ge\ (1-\mathbb E_\mu p)^K$ (Jensen; $(1-p)^K$ is convex), with a
large gap when $\mu$ is spread. **Plugging the mean $P(\text{truth})$ into $(1-\bar p)^K$ underestimates the measured
absent fraction by 4× (tera) to 90× (item).** (iii) *Belief-term lower bound.* Read Theorem B with $\hat b:=\hat b_K$, the empirical $K$-world belief the LP
actually optimises against (then $S=0$ and the $K$-world absence sits in the TV term). On the absent event the
attribute marginal of $\hat b_K$ gives the truth mass 0 while $b$ gives $p$, so by data processing
$\mathrm{TV}(\hat b_K,b)\ge\mathrm{TV}(\text{marginals})\ge p$; hence $\mathbb E\,\mathrm{TV}(\hat b_K,b)\ge\mathbb E_\mu[p(1-p)^K]$ and
Theorem B's term $2\Delta\,\mathrm{TV}$ is at least $2\Delta\,\mathbb E_\mu[p(1-p)^K]$ (a lower bound on the *bound's* term,
not on the loss).
(iv) *Log loss of the $K$-world predictor.* With the empirical frequency $\hat p=\mathrm{Bin}(K,p)/K$ and the 1e-3 clip,
$\mathbb E[-\ln\max(\hat p,10^{-3})]$ exceeds $-\ln p$ by a term that is $O(1/K)$ for $p$ bounded away from 0 and by
$(1-p)^K\ln(10^3p)$ from the absent event.
*Proof.* (i) independence; (ii) Jensen; (iii) $\hat b_K(\theta_{\rm true})=0$ on the event, so the TV is at least
$b(\theta_{\rm true})=p$; (iv) Taylor of $-\ln$ at $p$ plus the clipped mass. $\square$

**Corollary B11 ($K$-dependence of the sampling terms in our bounds).** [M3; S; **proved** (arithmetic on B′, C.1, S1,
S2); numeric]
(verifier: Theorem B′ is stated for $K\ge9$, so applying it at $K=4$ in (i) is **outside its range**.)
(i) Corollary C.1's sampling term $4\sqrt{2n\log K/K}$ falls by the factor $\sqrt{(\ln16/16)/(\ln4/4)}=0.707$ from
$K=4$ to $16$ (29%, not 50%, because of the $\log K$); it is **vacuous at both** ($n=7$: 8.8 → 6.2 payoff units vs the
range 2; Theorem B′ with $\Delta=1$: 2.28 → 1.59). (ii) S2's lower bound says no uniform-over-priors rate beats
$K^{-1/2}$, so quadrupling $K$ halves the irreducible sampling loss at best. (iii) S1: for interior priors the loss
decays as $e^{-2K\gamma^2}$, so $K=4\to16$ multiplies it by $e^{-24\gamma^2}$. (iv) The measurable $K$-effects are
B10's: absent fraction and excess log loss (below); **no win-rate gain is derivable** from the bounds (in-engine
X2, 8 vs 4 worlds: 0.495, n.s.).

### Numeric check (`worlds_K.py`, 0.5 s)
$\mu$ is taken from foul-play's exact-posterior calibration bins (`showdown.json` `fp_exact/*/calibration`: per
candidate value, bin count $n_b$, mean predicted $p_b$, empirical frequency $f_b$, so $n_bf_b$ true values at $p\approx p_b$;
a 10-bin quadrature).

| attribute | mean $P$(truth) | naive $(1-\bar p)^4$ | $\mathbb E_\mu(1-p)^4$ | **measured K=4 absent** | pred. K=8 | pred. K=16 | $\mathbb E_\mu[p(1-p)^4]$ | log loss exact / K4 meas. / K4 sim / **K16 sim** |
|---|---|---|---|---|---|---|---|---|
| tera | 0.610 | 0.023 | 0.101 | 0.096 | 0.026 | 0.0040 | 0.032 | 0.627 / 1.127 / 1.169 / **0.679** |
| item | 0.839 | 0.0007 | 0.038 | 0.062 | 0.011 | 0.0024 | 0.011 | 0.257 / 0.614 / 0.453 / **0.275** |
| move | 0.815 | 0.0012 | 0.032 | 0.031 | 0.007 | 0.0012 | 0.012 | 0.273 / 0.447 / 0.462 / **0.290** |
| ability | 0.969 | 0.0000 | 0.005 | 0.006 | 0.0008 | 0.0001 | 0.002 | 0.045 / 0.077 / 0.075 / **0.047** |

The quadrature reproduces the measured K=4 absent fractions and log losses for tera, move and ability; item is
under-predicted (0.038 vs 0.062; log loss 0.45 vs 0.61), consistent with items being coupled to whole-set draws and
with (B10.1) being unverified for foul-play's sampler. A Beta fit for tera (mean 0.61, variance matched to the
measured 0.096) predicts 0.013 at $K=16$ (vs 0.004 from the bins): the K=16 absent fraction predictions (verifier, bin-edge bounds) are **tera $\le$1.1%, item $\le$0.78%, move
$\le$0.38%** ("others $\le$0.3%" is not supported); the exact `frac_p0` (tera 0.0011, item 0.0019, move 0.0005) is a
floor for every K. Also, the K=16 worlds' log loss is within 7–8% of the exact posterior's (K=4: 1.6–2.4×).
*Prediction for the learner:* the `not_in_worlds` skip rate (7.9% of E2 feedback turns) should fall by roughly the
ratio of absent fractions for move-driven misses, but **not** for switch-driven misses (bench species: 97% absent at
K=4; the bench belief is near-uninformative, so no feasible $K$ fixes it).

*Status.* B10 **proved under (B10.1); logged item data contradict (B10.1) for item**; elementary / **new as a diagnosis** (the Jensen gap is the content; the TV lower bound is two lines).
B11 arithmetic on known results (B11(i) at $K=4$ is out of range of Theorem B′, which needs $K\ge9$).

---

## 5. Two posterior learners against each other (dynamics; optional)

**Observation B12.** [M2 meta; L7; **cite** + **open**; citations (Kalai–Lehrer, Nachbar, Foster–Young, Berk) **not checked**]
(i) Lecture 7's CCE/minimax argument needs vanishing **payoff** regret for both players. A `bayes_safex` player has
log-loss regret $\le\ln5$ (B1) and a per-stage **floor** (B5(a)); it has no payoff-regret guarantee: it best-responds
(within $F_\varepsilon$) to a prediction, which is the fictitious-play mechanism Lecture 6 §5 notes is *not* no-regret.
So Lecture 7 gives nothing about the empirical joint play of two Bayes-safex bots beyond the one-sided statement of
PA4β: each side's payoff stays within $\varepsilon\Delta\bar H+(\varepsilon_s+2\varepsilon_v)\bar H$ of the value (route (b) of B6
with $T_{\rm skip}$), so the self-play value is bracketed near $\tfrac12$ and the dynamics are unobservable at the
budgets of PA4α (B1 in the map).
(ii) Bayesian learning in games: Kalai & Lehrer 1993 (if each prior has a *grain of truth* on the opponent's actual
strategy, play converges to approximate Nash of the repeated game); Nachbar 1997/2005 and Foster & Young 2001
(impossibility: in general the prior cannot contain the opponent's learning rule). Here the five styles do not
contain "the other posterior learner", so the grain of truth fails by construction; Berk 1966 (misspecified Bayes)
suggests each posterior concentrates on the styles KL-closest to the realised play, but that theorem is for i.i.d.
data and the opponent's play is history-dependent. **Open**: whether the pair (posterior$_1$, posterior$_2$) has a
limit, and whether the induced joint play is an $\varepsilon$-CCE for some $\varepsilon$ tied to the misspecification.
(iii) What *is* immediate: if one side runs any payoff no-regret learner over fixed experts (E2's HedgeBot) against a
Bayes-safex bot, L2(a) sandwiches the learner's payoff by its best expert's trajectory sums; nothing more.

---

## 6. Ranked summary

| # | Result | Status | Level / lens | New vs known | Check |
|---|---|---|---|---|---|
| 1 | B5 safex floor + loss $\le\Delta\lVert q-\hat q\rVert_1$, constant sharp (constant-column witness; (d) concave in $\varepsilon$) | proved with (d) fixed; numeric sharp | M1 / L6 | Lipschitz + tightness new (standard LP sensitivity) | ratio 1.000 at the tie; 0 fallbacks |
| 2 | B8 equilibrium-opponent deficit = exploration mass × slack; max-visit removes it; _mv prediction (ordering held: 0.495, 0.593 vs 0.063, 0.050) | proved (identity); (iii) hypothesis; + numeric | M0–M1 / L7 | new statement and testable prediction | deficit 0.028–0.056 (shares) vs 0.002–0.010 (_mv) |
| 3 | B1 Bayes = Hedge($\eta=1$), regret $\le\ln5$ pathwise, history-dependent styles, skipped turns | cite + adaptation | M0 / L6 | known (CBL §3, §9) | max regret 1.6094 = bound |
| 4 | B10 Jensen: absent fraction is $\mathbb E(1-p)^K$ not $(1-\bar p)^K$; K=16 predictions | proved under (B10.1) + numeric; logged item data contradict (B10.1) | M3 / S | new as diagnosis | tera 0.101 pred vs 0.096 meas.; item under 0.038 vs 0.062 |
| 5 | B6 whole-game for hedge_bayes: two safety routes (no double count), skipped-turn term, realisable gain via Pinsker (in-support law) | proved with named hypotheses (B6(c) after the hypothesis fix); payoff form needs $a>0$ | M2–M3 / L6 | assembly new; ingredients known | sizes: vacuous per game, 0.012 range/turn at $T=2.3\cdot10^4$ |
| 6 | B7 Exp3 constants with real $k$ (median two-sided 0.47 vs gap 0.058); Theorem A sum 2.6/game (upper bound on the row-side sum) vacuous; gap anti-correlated with outcome | cite + numeric | M0–M1 / L7 | measurements new | logs |
| 7 | B3(b) discount-then-predict: discounted regret $\le\ln k_s$ | proved | M0 / L6 | known (discounted experts) | max 1.52 |
| 8 | B2 fixed share on log loss, tracking bound | proved | M0 / L6 | known (Herbster–Warmuth) | 16.3 ≤ 21.2 |
| 9 | B9 max-visit: never consistent on strictly mixed matrices; pure-saddle premise unproved | proved (i)–(ii) / cite / open (iv) | M0 / L7 | (i)–(ii) elementary | mv gap 0.167 vs 0.068 on mixed |
| 10 | B11 $K$-dependence: C.1 term × 0.71, vacuous at both K; no win-rate prediction | arithmetic (B11(i) at K=4 outside Theorem B′'s range $K\ge9$) | M3 / S | known | — |
| 11 | B3(c) coded forgetting satisfies the discounted bound? | refuted for $\lambda<\tfrac12$ (3.82 at 0.3, 2.25 at 0.4 vs 1.609); open for $\lambda\ge\tfrac12$ | M0 / L6 | open | 360 paths, no violation (but `b3c_counter.py` finds one) |
| 12 | B12 two posterior learners | cite / open | M2 / L7 | open | — |

---

## 7. Refuted / not supported (keep so they are not re-derived)

- **"The Bayes posterior is Lecture 6 MWU with bounded losses."** The losses are $-\ln q_s(b)$, bounded only by
  $\ln(n_t/\mathrm{MIX})\approx5.6$; the clean $\ln k_s$ bound comes from mixability, not Hoeffding. MWU's bound after
  rescaling is $\hat L\le(1+\eta)L^*+5.6\ln5/\eta$, much weaker.
- **"The shipped $\lambda<1$ is fixed share / gives a tracking bound."** It is exponent discounting; B2's bound does not
  apply; B3(b) applies only to the discount-then-predict variant (B3(c) refuted for the code at $\lambda<\tfrac12$, open for $\lambda\ge\tfrac12$).
- **"Log-loss regret bounds prediction accuracy."** Only under realisability (B4(iii)); agnostically a style can win on
  log loss with the wrong argmax.
- **Whole-game formula with both $(\varepsilon_s+2\varepsilon_v)$ and $\varepsilon\cdot$range.** Double counts: the two terms belong
  to two different certifying experts (B6(a) vs (b)); take the minimum.
- **P1 for hedge_bayes across games.** Fails (P1.3): the safex expert depends on the carried posterior.
- **"Exp3's $O(\sqrt{k\ln k/N})$ bounds the expected visit-share gap."** It bounds pseudo-regret; the expected gap needs
  high-probability bounds, which are $\ge1$ range at these $k$, $N$ (B3(b) of the map). With the real column counts
  (median 18) even the pseudo-regret reference is 0.47 per decision.
- **"The per-turn stage gap summed over $\bar H$ explains the 0.05–0.06 bandit scores."** Theorem A's sum is 2.6–2.8 A01
  units per game (vacuous), and on the logs the score *increases* with the mean gap. The loss against the RM+ root is
  the exploration deficit (B8), not the exploitability.
- **"The true value is absent with probability $(1-\bar p)^K$" with $\bar p$ the mean $P$(truth).** Jensen: 0.023 vs
  measured 0.096 (tera), 0.0007 vs 0.062 (item). Use $\mathbb E_\mu(1-p)^K$.
- **"$K=4\to16$ halves the sampling term."** Factor 0.71 for the $\sqrt{\log K/K}$ terms (and both are vacuous);
  halving is the S2 lower-bound rate.
- **"Lecture 7 gives CCE convergence for two posterior learners."** They have no payoff-regret guarantee (B12).
- **Verifier-refuted items.** (1) B3(c) for $\lambda<\tfrac12$: discounted regret 3.82 at $\lambda=0.3$, 2.25 at 0.4 vs
  $\ln5$ (`b3c_counter.py`). (2) B5(d) "$g$ convex in $\varepsilon$": it is concave. (3) B5(c) "sharp via $I_2$" at the
  shipped $\varepsilon$: ratio 0.006 there. (4) B8(iii) "exactly in $Q^*$ units" for `rmplus_avg`: E4umv 0.593 > 0.5.
  (5) B8 "max-visit 2.5× more exploitable on mixed matrices": an artifact of 40 random matrices (0.053 vs 0.058 on 57
  real); and "deficit 8–10× smaller" (pre-registered 5–8×). (6) B1(iii) "losses $\le5.56$": 6.78 at $n=44$. (7) B9(iii)
  citations: Kocsis–Szepesvári Thm 6 is single-agent and max-mean; Kovařík–Lisý says nothing on max-visit. (8) B10
  "others $\le$0.3% at K=16": not supported (item $\le$0.78%, move $\le$0.38%). (9) B7(i) identity with sampling
  distributions for Exp3. (10) B11(i) at $K=4$ with Theorem B′ ($K\ge9$).
- Earlier numeric slip in this sprint: a 300-path estimate of $\mathbb E\sum\mathrm{KL}$ at $T=300$ read 1.73 $>\ln5$; at
  3,000 paths it is $1.624\pm0.031$ (Monte Carlo noise; the identity $\mathbb E\,\mathrm{regret}=1.609$ holds).

---

## 8. What to run next (tests the new content)

1. **E4 _mv vs rmplus_avg** (B8 prediction; **done**: E4emv 0.4949, E4umv 0.5933; only the exploiter arm remains): expect `bandit_exp3_mv`, `bandit_ucb1_mv` $\gg$ 0.06 against the RM+
   root; then _mv vs an exploiter, expecting the ordering to flip.
2. **hedge_bayes / bayes_safex vs stock and vs the styled opponents** with the posterior and per-turn $(\hat q_t,b_t)$
   logged: check B1(ii) pathwise on every log ($\le1.609$; one violation refutes the implementation **only after** checking that the update counts in the logs and state files match, see the B1 verifier note), B4(i)'s KL
   comparison with the best style, and the per-turn gain $\hat x_t^\top\bar A_t[:,b_t]-v_t$ against B5.
3. **K=16 belief run** (B10): absent fractions tera $\le1.1\%$, item $\le0.78\%$, move $\le0.38\%$; `not_in_worlds` skip rate falls
   for move misses only.
4. If forgetting is wanted: ship fixed share (B2) or discount-then-predict (B3(b)), not the current exponent discount.
