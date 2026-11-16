Verified 2026-11-16 (theory/tightening/verify/); corrections applied.

# Bayesian opponent model and learning dynamics: settled items, new statements, pre-registered predictions

Companion to `../experts_extensions.md` (B1–B12, §6 ranking, §7 refuted list) and `../synthesis.md` (§3.2 N1/N2,
§3.3 N3). Notation as there: $k_s=5$ styles, $q_{s,t}$ the style distributions (every entry $\ge\mathrm{MIX}/n_t$,
$\mathrm{MIX}=0.05$), $\ell_{s,t}=-\ln q_{s,t}(b_t)\in[\,\ell_{\min},B_t]$ with $B_t=\ln(n_t/\mathrm{MIX})$ (5.56 at
$n_t=13$, 6.78 at the logged maximum $n_t=44$), posterior $P_t$, mixture $\hat q_t=\sum_sP_{t-1}(s)q_{s,t}$,
$\hat\ell_t=-\ln\hat q_t(b_t)$. Code: `engine/showdown/behavior.py` (`Posterior.update`: `logp <- lam*logp + log lik`,
prediction with the stored `logp`), `engine/showdown/run_ourbot.py:447–468` (`hedge_bayes`, `bayes_safex`),
`engine/nash.py:244–262` (`safex_strategy`). Written 2026-11-16. Status words as in `theory_map.md` §0.4:
**proved** / **sketch** / **conjecture** / **refuted** / **measured**. Scripts and outputs:
`theory/tightening/scripts/` (`b3c_all_lambda.py`, `b3c_realisable.py`, `osc_bh.py`, `b12_selfplay.py`,
`b4_concentration.py`, outputs `out_*.txt|json`). All experiments are scratch probes on CPU (no ledger in this project).
The RQ runs (`hedge_bayes` vs `style_eq` / `style_safe` / `style_uniform`) had **no results directory** when this was
written (`engine/showdown/results/` has no `RQ*`), so §4's predictions are pre-registered.

Real matrices: the 57 stage matrices of `scratchpad/experts_ext/real_matrices.npz` (`nash.full_matrix`, 2–13 rows,
2–13 columns, median 7 columns); they are smaller than the logged mid-game matrices (columns median 18), so every
"real-matrix" number below is a lower-bound proxy for loss ranges and separations.

---

## 0. Summary of what changed

| item | before | now |
|---|---|---|
| B3(c) coded forgetting, discounted regret $\le\ln k$ | refuted for $\lambda<\tfrac12$, open for $\lambda\ge\tfrac12$ | **refuted for every $\lambda<1$ with unbounded losses; with the shipped cap $B=\ln(n/\mathrm{MIX})$ refuted for all $\lambda\le\lambda^*_2(n)$ by a 2-turn example ($\lambda^*=0.951$ at $n=13$, $0.985$ at $n=44$, on the attainable loss range), further by longer paths; realisable violations on the real style matrices at $\lambda\in\{0.5,0.7,0.9\}$; a correct data-dependent bound is proved (T1)** |
| posterior oscillation | described (synthesis §3.2) | characterised as FTL leader switching on a random walk with drift; $\sigma^2/(2\mu^2)$ agrees in order of magnitude with the number of MAP-wrong turns (563 vs 649 measured per stream mean; per-stream ratio 0.03–2.0; autocorrelation-corrected estimate $\approx740$); FTL regret 5–18 nats vs the mixture's 1.609 (T2) |
| B4 realisability rate | Pinsker, $\sqrt{2T\ln k}$ | Bhattacharyya: $\mathbb E\sum_t(1-P_{t-1}(s^*))\le\sum_{s\ne s^*}1/(1-\rho_s)$, **constant in $T$**; replaces B6(c)'s $\sqrt{2T\ln5}$ by $2\sum_s1/(1-\rho_s)$ (33–82 on real matrices: better than Pinsker beyond $T\approx350$–2,100, i.e. at BH/E2 lengths, not within one RQ stream for safe/br_unif); misspecified version with a Chernoff-½ coefficient (T3) |
| RQ outcome | not predicted | pre-registered: ordering uniform ≻ safe ≻ eq in identification speed/robustness, with mismatch thresholds (§4) |
| B12 two posterior learners | cite + open | every last iterate of two $\varepsilon$-safe players is a $2\varepsilon\Delta$-NE (no regret argument); Hedge vs safex sandwiched; self-play posteriors measured (T4) |
| tracking after a style switch | — | flip time $\approx t_0\mu_a/\mu_b$ for $\lambda=1$ (N1), $\lesssim(\ln(k/\alpha)+B)/\min_s\mu_{b\to s}$ for fixed share, $\ln((\mu_a+\mu_b)/\mu_b)/\ln(1/\lambda)$ for discount-then-predict; checked (T5) |

---

## 1. B3(c) settled: the coded forgetting has no $\ln k$ discounted bound

### Setting
$k$ styles, losses $\ell_t\in[0,B]^k$ (the shipped class has $0<\ell_{\min}\le\ell\le B_t=\ln(n_t/\mathrm{MIX})$; the
sequences below use losses $0$ and $B$, and §1.1 gives realisable ones). Coded variant (`behavior.Posterior` with
`lam`$=\lambda<1$): $D_t=\lambda D_{t-1}+\ell_t$, $D_0=0$, prediction with $P_{t-1}\propto e^{-D_{t-1}}$ ("predict, then
discount"). Discounted regret $R^\lambda_T(s)=\sum_{t\le T}\lambda^{T-t}(\hat\ell_t-\ell_{s,t})$. The claim under test
was $R^\lambda_T(s)\le\ln k$ for all sequences (true for the discount-then-predict variant, B3(b)).

**Theorem T1.** [M0; L6; (a)–(c) **proved**, (d) **measured**, (e) **conjecture**]

(a) *Unbounded losses.* For every $\lambda\in(0,1)$, $k\ge2$ and $B_0>0$ there is a 2-turn sequence with
$R^\lambda_2(s)\ \to\ \lambda\ln k+(1-\lambda)B_0$ as the other styles' second-turn loss $C\to\infty$; this exceeds
$\ln k$ as soon as $B_0>\ln k$, and is unbounded in $B_0$. So no bound $R^\lambda_T\le f(k,\lambda)$ exists for the coded
variant (contrast B3(b)).

(b) *Capped losses, exact 2-turn value.* Turn 1: style 1 pays 0, every other style pays $B_0\le B$. Turn 2: style 0
pays 0, every other style pays $B$. With $a=\ln(1+(k-1)e^{-B_0})$,
$$R^\lambda_2(0)=\lambda\big(\ln k-a-B_0\big)+a-\ln\!\big(e^{-B_0}+(1+(k-2)e^{-B_0})e^{-B}\big).$$
Dropping the $O(ke^{-B_0})$ corrections, $R^\lambda_2(0)\approx\lambda\ln k-\lambda B_0-\ln(e^{-B_0}+e^{-B})$, maximised
at $B_0^*=B-\ln\frac{\lambda}{1-\lambda}$ when $\lambda\ge\tfrac12$ (else at the cap $B_0^*=B$), with value
$$R^{\lambda,*}_2\approx\lambda\ln k+(1-\lambda)B-H(\lambda),\qquad H(\lambda)=-\lambda\ln\lambda-(1-\lambda)\ln(1-\lambda).$$
Hence the 2-turn example exceeds $\ln k$ iff (up to the dropped corrections)
$$B\ >\ \ln k+\frac{H(\lambda)}{1-\lambda}.$$
For $\lambda<\tfrac12$ the condition is $B>\ln k+\ln2/(1-\lambda)$, which is why the verifier's examples at
$\lambda\in\{0.3,0.4\}$ worked and the one at $0.5$ (a 30-turn build-up, not the optimal $B_0$) did not.

(c) *A correct data-dependent bound.* For every sequence, every $T$, every style $s$,
$$R^\lambda_T(s)\ \le\ \ln k+(1-\lambda)\sum_{t\le T}\lambda^{T-t}\operatorname{range}_s D_{t-1}(s),\qquad
\operatorname{range}_sD=\max_sD(s)-\min_sD(s).$$
Since $\operatorname{range}D_{t-1}\le B(1-\lambda^{t-1})/(1-\lambda)$ this is at most $\ln k+B/(1-\lambda)$, i.e. no
better than the trivial bound $B/(1-\lambda)$ in the worst case — the content is the data-dependent form (tight to
0.66–0.98 depending on the example set) and the mechanism it exposes.

(d) *Thresholds for the shipped loss class* (measured, `b3c_all_lambda.py`, exact grid over $B_0$, $k=5$):

| $n_{\max}$ | $B=\ln(n/\mathrm{MIX})$ | $\lambda^*_2$ (attainable range) | $\lambda^*_2$ from $B=\ln k+H/(1-\lambda)$ | $R^{\lambda,*}_2$ at $\lambda=0.9$ (attainable range) |
|---|---|---|---|---|
| 13 | 5.561 | 0.9506 | 0.949 | 1.678 |
| 18 | 5.886 | 0.9640 | 0.963 | 1.710 |
| 27 | 6.292 | 0.9759 | 0.975 | 1.749 |
| 44 | 6.780 | 0.9851 | 0.985 | 1.797 |

($\ln5=1.609$.) Losses $0$ and $B=\ln(n/\mathrm{MIX})$ cannot occur on the same turn: the attainable range is $\ln(n(1-\mathrm{MIX})/\mathrm{MIX}+1)$ (5.513 at $n=13$, 6.730 at $n=44$), which gives $\lambda^*=0.951$ ($n=13$) $\ldots$ $0.985$ ($n=44$); the exact-value columns above use that range. The shipped class `behavior.Posterior(lam)` reproduces the closed form to $10^{-4}$ on each path
(violations at $\lambda=0.3,0.5,0.7,0.9,0.95$ for both $n=13$ and $n=44$). Longer paths violate more: a log-space
hill-climb over $[0,B]^{T\times5}$ at $B=6.78$ finds $R^\lambda_T$ of 2.16 ($T=3$), 2.38 ($T=4$), 2.77 ($T=6$) at
$\lambda=0.9$ and 1.85 / 1.99 / 2.22 at $\lambda=0.95$ (2-turn optimum 1.80 / 1.67). The old `b3c_search.py` "inf" at
$T=2$ was exp underflow with 500-nat losses, not a finding.

(e) *Conjecture.* With the cap, the coded variant violates $\ln k$ for every $\lambda<1$ once $T$ is allowed to grow
(the $T$-step maxima in (d) increase monotonically in $T$ at every $\lambda$ tried). The open band is
$\lambda\in(\lambda^*_T(n),1)$, shrinking with $T$; it has no practical content.

*Proofs.* (a) At $t=1$, $P_0$ uniform: $\hat\ell_1=-\ln\frac1k(1+(k-1)e^{-B_0})\to\ln k$ for large $B_0$;
$D_1=(B_0,0,B_0,\dots)$, so $P_1(0)=e^{-B_0}/(1+(k-1)e^{-B_0})$. At $t=2$ with the other styles' loss $C$:
$\hat\ell_2=-\ln\big(P_1(0)+(1-P_1(0))e^{-C}\big)\to-\ln P_1(0)=B_0+\ln(1+(k-1)e^{-B_0})$ as $C\to\infty$. Then
$R^\lambda_2(0)=\lambda(\hat\ell_1-B_0)+\hat\ell_2\to\lambda\ln k+(1-\lambda)B_0+(1-\lambda)\ln(1+(k-1)e^{-B_0})$.
(b) The same computation with $C=B$. The stationarity condition of
$-\lambda B_0-\ln(e^{-B_0}+e^{-B})$ is $e^{-B_0}/(e^{-B_0}+e^{-B})=\lambda$, i.e. $B_0=B-\ln\frac\lambda{1-\lambda}$, feasible
($\le B$) iff $\lambda\ge\frac12$; substituting gives $(1-\lambda)B+\lambda\ln\lambda+(1-\lambda)\ln(1-\lambda)$.
(c) Potential $F_t=-\ln\frac1k\sum_se^{-D_t(s)}$, $F_0=0$, $F_T\le D_T(s)+\ln k$. With $a_s=e^{-D_{t-1}(s)}$ and
$\mu\propto a_se^{-\ell_{s,t}}$ (the undiscounted Bayes posterior after the turn):
$\hat\ell_t-(F_t-\lambda F_{t-1})=(1-\lambda)\ln\tfrac1k\sum_sa_s+\ln\mathbb E_\mu\big[a_s^{\lambda-1}\big]
\le-(1-\lambda)\min_sD_{t-1}+(1-\lambda)\max_sD_{t-1}$. Multiply by $\lambda^{T-t}$, sum, telescope. The
mixability identity $\sum_sP_{t-1}(s)e^{-\ell_{s,t}}=e^{-\hat\ell_t}$ is used verbatim; the extra term is exactly where
B3(b)'s Jensen step ($\frac1k\sum a_s^\lambda\le(\frac1k\sum a_s)^\lambda$) is replaced by an inequality that goes the
wrong way when the posterior mass $\mu$ sits on styles with large $D_{t-1}$ — the "surprise" turn. $\square$

*Reading.* The coded variant is the discount-then-predict algorithm run at learning rate $\eta=1/\lambda>1$:
$P_{t-1}\propto\exp(-\tfrac1\lambda\cdot\lambda D_{t-1})$. Log loss is $\eta$-mixable only for $\eta\le1$ (Vovk), so the
aggregating-algorithm bound $\ln k/\eta$ is unavailable and the excess $(1-\lambda)B=(\eta-1)\lambda B$ is the price of
over-aggressive weights. **Fix (one line):** `z = self.lam * (self.logp + np.log(lik))` in `Posterior.update` stores
$-\lambda D_t$, so the prediction uses $e^{-\lambda D_{t-1}}$ (B3(b), bound $\ln k$ on every path). Default `LAM=1.0`
is unaffected.

### 1.1 Realisable violations (real style matrices)
`b3c_realisable.py`: the 569 (matrix, column) loss vectors of the 57 real matrices (p_ref uniform, max loss 5.56),
hill-climbed over sequences of real turns, scored with `behavior.Posterior(lam)`:

| $\lambda$ | $T=2$ | $T=3$ | $T=4$ | $T=6$ | $T=8$ |
|---|---|---|---|---|---|
| 0.5 | **1.614** | **1.729** | **1.813** | **1.836** | **1.838** |
| 0.7 | 1.434 | 1.570 | **1.636** | **1.714** | **1.753** |
| 0.9 | 1.463 | 1.528 | 1.588 | **1.633** | 1.608 |
| 0.95 | 1.526 | 1.543 | 1.576 | 1.599 | 1.602 |

(bold $>\ln5=1.609$). The $\lambda=0.9$, $T=4$ witness has the predicted shape: a turn where `eq` is right and the
others pay 3–5 nats, then a turn where `br_unif`/`br_us` are right and `eq` pays 5.1. On these small matrices the loss
range is 5.56 and the violation margins are small; at the logged $n=44$ ($B=6.78$) they are larger (Table (d)).

---

## 2. Posterior oscillation: FTL on a random walk with drift (T2)

### Setting and measured facts (`osc_bh.py`, BH logs, 8 clean streams, 1,969–2,450 feedback turns each)
Final leaders $s_1$ (best style) and $s_2$ (runner-up) are always {safe, br_unif}. Gap $G_t=\sum_{u\le t}(\ell_{s_2,u}-\ell_{s_1,u})$,
increments $g_t$: mean $\mu\in[0.011,0.028]$ nats/turn, sd $\sigma\in[0.58,0.62]$, lag-1 autocorrelation
$0.09$–$0.17$ (weakly dependent, not i.i.d.; 15% of turns have $g_t=0$, forced turns). Final gaps 28–67 nats.

| stream | $T$ | $\mu$ | $\sigma$ | MAP switches | turns $G_t<0$ (measured) | $\sigma^2/(2\mu^2)$ | Hoeffding $\sum_te^{-2\mu^2t/B^2}$ (range $B$; with range $2B$ larger still) | FTL regret | mixture regret |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 2159 | 0.0244 | 0.616 | 8 | 10 | 318 | 2098 | 8.0 | 1.609 |
| 2 | 2376 | 0.0268 | 0.576 | 7 | 461 | 231 | 2286 | 5.3 | 1.609 |
| 3 | 2173 | 0.0155 | 0.582 | 31 | 899 | 702 | 2148 | 17.8 | 1.609 |
| 4 | 1969 | 0.0202 | 0.591 | 20 | 796 | 427 | 1933 | 10.0 | 1.609 |
| 5 | 2404 | 0.0279 | 0.598 | 4 | 461 | 230 | 2305 | 5.8 | 1.609 |
| 7 | 2234 | 0.0205 | 0.622 | 10 | 89 | 460 | 2187 | 9.8 | 1.609 |
| 8 | 2396 | 0.0166 | 0.615 | 13 | 965 | 686 | 2360 | 5.0 | 1.609 |
| 9 | 2450 | 0.0114 | 0.616 | 32 | 1512 | 1453 | 2432 | 16.2 | 1.609 |
| mean | 2270 | | | 15.6 | **649** | **563** | 2219 | 9.7 | 1.609 |

Mean log loss per turn: mixture 1.5078, FTL/MAP 1.5114, best style 1.5071, runner-up 1.5275; argmax accuracy
0.516 / 0.5125 / 0.5145.

**Proposition T2.** [M0 meta; **L6 object, not L7**; (a) **proved** (Lecture 6 §2.3 adapted); (b) **sketch** +
measured; (c) **proved**]
(a) *MAP = Follow-The-Leader.* With $\lambda=1$ the MAP style is $\arg\min_s\sum_{u<t}\ell_{s,u}$, i.e. FTL on log
loss; Lecture 6 §2.3's alternating sequence ($\ell_1=(B/2,0)$, then $(0,B),(B,0),\dots$) gives FTL regret
$\ge(T-1)B/2$, linear in $T$, while the mixture's regret is $\le\ln k$ on the same sequence (B1). On the BH streams the
FTL regret is 5.0–17.8 nats (mean 9.7, mean 15.6 leader switches) against the mixture's 1.609 on every stream;
per turn the excess is tiny (0.0036 nats) because the two leaders are near-identical on most turns.
(b) *Time on the wrong side.* Model $g_t$ as i.i.d. with mean $\mu>0$ and variance $\sigma^2$ (Brownian approximation).
Then $\mathbb E\,\#\{t\le T:\ G_t<0\}=\sum_t\Pr(G_t<0)\approx\int_0^\infty\Phi(-\mu\sqrt t/\sigma)\,dt=\frac{\sigma^2}{\mu^2}\int_0^\infty\Phi(-\sqrt u)\,du=\frac{\sigma^2}{2\mu^2}$
(substitute $u=z^2$; $\int_0^\infty2z\Phi(-z)dz=\int_0^\infty z^2\varphi(z)dz=\tfrac12$). Measured per stream the ratio
measured/predicted ranges over 0.03–2.0 (the time on the wrong side of a drifted walk has a heavy-tailed law, so
single paths scatter), with means 649 vs 563 over the 8 streams; the autocorrelation-corrected estimate is $\approx740$, which brackets the measured 649 together with 563. The estimate therefore agrees in order of magnitude, not per stream. The discrete Hoeffding bound
($g_t$ has range $2B$, so $\sum_t\exp(-\mu^2t/(2B^2))$, even closer to $T$ than the 2,219 of the table, which used range $B$) is $\approx T$: vacuous at $\mu\approx0.02$, $B=5.6$ — the honest *bound* says
oscillation can last the whole run; the Brownian *estimate* says a quarter of it, and that is what is seen. The
number of sign changes has no robust prediction (Brownian motion crosses 0 infinitely often; the discrete count
depends on the increment law); the duration is the quantity to report. Consequence for synthesis §3.2's observation:
the last posterior is a vertex because $G_T\in[28,67]\gg1$ gives $P_T(s_2)=1/(1+e^{G_T})\approx0$, while the time
average $\frac1T\sum_t1/(1+e^{G_t})$ is mixed exactly because $G_t$ spends $\approx\sigma^2/(2\mu^2)$ turns below 0
and many more within a few nats of it.
(c) *Mixture over MAP.* (i) Log loss: regret $\le\ln k$ pathwise for the mixture (B1); for the MAP only
$\mathrm{Reg}_T^{\rm MAP}\le\sum_t\mathbb 1\{\text{leader}_{t-1}\ne s^*\}(\ell_{\text{leader},t}-\ell_{s^*,t})\le B\cdot\#\{\text{wrong turns}\}$,
which with (b) is $\approx B\sigma^2/(2\mu^2)$ — up to 3,000 nats here (measured 5–18: the leaders rarely disagree
by much). (ii) Prediction error (feeds B5(c)): $\lVert\sigma-\hat q\rVert_1\le\sum_sP(s)\lVert\sigma-q_s\rVert_1$ (convexity),
so the mixture never does worse than the posterior-weighted average of the styles' errors, while the MAP pays a single
style's error and can pick the wrong one for $\sigma^2/(2\mu^2)$ turns. No pathwise payoff statement beyond this: the
safex *argmax* is not Lipschitz in $q$ (only the value $g$ is, B5(b)). *Proof of (a),(c)*: as stated; (b) is a
heuristic, labelled sketch. $\square$

*Lecture framing.* The posterior's time average has no payoff meaning (there is no game between the styles), so
Lecture 7's CCE/minimax reading does not apply to it; the right lens is Lecture 6: MAP is the deterministic FTL that
§2.3 shows is exploitable, the mixture is MWU with $\eta=1$ on an exp-concave loss. Lecture 7's "daily play may
oscillate; the averages converge" is the analogy (the last iterate is a vertex; the cumulative gap, not the posterior,
is what converges — to a line of slope $\mu$).

---

## 3. Realisability: geometric concentration via the Bhattacharyya coefficient (T3)

**Lemma T3.** [M0 (M3 inputs); L6; (i)–(iv) **proved**; (v) **proved** under the stated independence; numeric]
*Assumptions.* (T3.1) Realisable: at every feedback turn, conditional on the history $h_t$ (and the feedback event,
B6(c)'s $\sigma^{\rm in}_t$), $b_t\sim q_{s^*,t}$ for one style $s^*$. (T3.2) $\lambda=1$, uniform prior. Define the
conditional Bhattacharyya coefficient $\mathrm{BC}_t(s)=\sum_b\sqrt{q_{s,t}(b)\,q_{s^*,t}(b)}\in(0,1]$ ($h_t$-measurable)
and the likelihood ratio $\mathrm{LR}_{s,t}=P_t(s)/P_t(s^*)=\prod_{u\le t}q_{s,u}(b_u)/q_{s^*,u}(b_u)$.
(i) $\mathbb E[\sqrt{\mathrm{LR}_{s,t}}\mid h_t]=\sqrt{\mathrm{LR}_{s,t-1}}\;\mathrm{BC}_t(s)$; hence
$M_t=\sqrt{\mathrm{LR}_{s,t}}/\prod_{u\le t}\mathrm{BC}_u(s)$ is a mean-one martingale, and if $\mathrm{BC}_u(s)\le\rho_s<1$
a.s. for all $u$ then $\mathbb E\sqrt{\mathrm{LR}_{s,t}}\le\rho_s^{\,t}$.
(ii) $1-P_t(s^*)\le\sum_{s\ne s^*}\min(1,\mathrm{LR}_{s,t})\le\sum_{s\ne s^*}\sqrt{\mathrm{LR}_{s,t}}$, so
$\mathbb E[1-P_t(s^*)]\le\sum_{s\ne s^*}\rho_s^{\,t}$ and
$$\mathbb E\sum_{t\le T}\big(1-P_{t-1}(s^*)\big)\ \le\ \sum_{s\ne s^*}\frac1{1-\rho_s}\qquad\text{(independent of $T$).}$$
(iii) $\Pr(\text{MAP}_t\ne s^*)\le\sum_{s\ne s^*}\rho_s^{\,t}$ (Markov on $\sqrt{\mathrm{LR}}$).
(iv) $\lVert\hat q_t-q_{s^*,t}\rVert_1\le2\,(1-P_{t-1}(s^*))$, hence
$\mathbb E\sum_{t\le T}\lVert\hat q_t-q_{s^*,t}\rVert_1\le2\sum_{s\ne s^*}1/(1-\rho_s)$ — a constant, in place of B4(ii)'s
$\sqrt{2T\ln k}$. **B6(c) upgraded:** the realisable gain slack $\bar\Delta_{\max}\sqrt{2\mathbb E[T]\ln k_s}$ becomes
$2\bar\Delta_{\max}\sum_{s\ne s^*}1/(1-\rho_s)$ (same proof, weights $(1-\pi_t)\le1$ kept).
(v) *i.i.d.-matrix form.* If the turn matrices are i.i.d. (so $\mathrm{BC}_t(s)$ is independent of $h_t$), then
$\mathbb E\sqrt{\mathrm{LR}_{s,t}}=(\mathbb E\,\mathrm{BC}(s))^t$ exactly and (ii)–(iv) hold with $\rho_s$ replaced by the
*mean* coefficient $\bar\rho_s=\mathbb E\,\mathrm{BC}(s)$. The real game is in between (history-dependent matrices):
(i) with the a.s. bound is what holds there; forced turns ($n_t=1$, $\mathrm{BC}=1$) make $\rho_s=1$ over the whole run,
so the usable statement counts non-forced feedback turns only (on them $\mathrm{BC}_t(s)<1$ whenever $q_{s,t}\ne q_{s^*,t}$).
(vi) *Misspecified version (new).* Let $\sigma_t$ be the true conditional law and $s^\circ$ a reference style; define the
Chernoff-½ coefficient $c_t(s)=\mathbb E_{b\sim\sigma_t}\sqrt{q_{s,t}(b)/q_{s^\circ,t}(b)}$. The same martingale gives
$\mathbb E\sqrt{P_t(s)/P_t(s^\circ)}\le\prod_{u\le t}\sup c_u(s)$; if $c_u(s)\le\rho^\circ_s<1$ for all $s\ne s^\circ$ the
posterior concentrates on $s^\circ$ geometrically, and with B5(c) the `bayes_safex` surrogate payoff satisfies
$$\mathbb E\sum_t\hat x_t^\top\bar A_t\sigma_t\ \ge\ \mathbb E\sum_tg_t(\sigma_t)\;-\;\Delta\,\mathbb E\sum_t\lVert\sigma_t-q_{s^\circ,t}\rVert_1\;-\;2\Delta\sum_{s\ne s^\circ}\frac1{1-\rho^\circ_s},$$
i.e. the oracle safe-exploitation gain against the truth, minus the (linear) misspecification of the reference style
in total variation, minus a constant. ($c_t(s)<1$ is weaker than "$s^\circ$ is KL-closest": by Cauchy–Schwarz
$c_t(s)\le\sqrt{\mathbb E_\sigma[q_s/q_{s^\circ}]}$, and $c_t(s)<1$ holds e.g. whenever $\mathrm{KL}(\sigma\|q_{s^\circ})<\mathrm{KL}(\sigma\|q_s)$
and the log-ratio is concentrated; it fails when $q_s$ and $q_{s^\circ}$ tie.)
*Proof.* (i) tower rule with $\mathbb E[\sqrt{q_s(b)/q_{s^*}(b)}\mid h_t]=\sum_bq_{s^*}(b)\sqrt{q_s(b)/q_{s^*}(b)}=\mathrm{BC}_t(s)$.
(ii) $P_t(s)\le\mathrm{LR}_{s,t}$ because $P_t(s^*)\le1$; $\min(1,x)\le\sqrt x$. (iii) $\{\text{MAP}\ne s^*\}\subset\bigcup_s\{\mathrm{LR}_{s,t}\ge1\}$,
$\Pr(\sqrt{\mathrm{LR}}\ge1)\le\mathbb E\sqrt{\mathrm{LR}}$. (iv) $\hat q_t-q_{s^*,t}=\sum_{s\ne s^*}P_{t-1}(s)(q_{s,t}-q_{s^*,t})$, each
term of $\ell_1$-norm $\le2$. (v) independence turns the a.s. bound into an equality of expectations. (vi) replace
$q_{s^*}$ by $\sigma_t$ in (i); the payoff line is B5(c) with the triangle inequality $\lVert\sigma_t-\hat q_t\rVert_1\le\lVert\sigma_t-q_{s^\circ,t}\rVert_1+\lVert q_{s^\circ,t}-\hat q_t\rVert_1$ and (iv) for $s^\circ$. $\square$

*What is new.* The route (Bhattacharyya/Hellinger supermartingale → posterior mass → $\ell_1$ prediction error → LP
Lipschitz) replaces the Pinsker–Cauchy–Schwarz route of B4(ii)/B6(c) and removes the $\sqrt T$; the earlier numeric
observation "Pinsker is loose by 2–6× because the posterior concentrates exponentially" (experts_extensions §1,
check B) is this lemma. The ingredients are classical (Hellinger affinity of product measures; Barron's and
Walker's consistency arguments); the statement for a history-dependent style class with the exact per-turn constant and
its use as a payoff bound for safe exploitation is specific to this setting. Numeric checks: §3.1 (filled after
`b4_concentration.py`).

### 3.1 Numbers on the 57 real matrices (`b4_concentration.py`, `b4_chernoff_half.py`; 11.5 min + 0.6 min)
p_ref = the one-hot safex strategy (as `bayes_safex`); turns i.i.d. over the 57 matrices (so (v) applies); 40 paths.

*Separation (per turn, nats; mean over matrices).* Bhattacharyya distance $-\ln\mathrm{BC}$ and KL:

| $s^*\backslash s$ | eq | br_unif | br_us | safe | uniform |
|---|---|---|---|---|---|
| eq | — | 0.277 / 0.842 | 0.257 / 0.746 | 0.211 / 0.614 | 0.505 / 1.509 |
| safe | 0.211 / 1.107 | 0.074 / 0.266 | 0.070 / 0.243 | — | 0.180 / 0.637 |
| uniform | 0.505 / 2.064 | 0.150 / 0.632 | 0.107 / 0.474 | 0.180 / 0.749 | — |
| br_unif | 0.277 / 1.336 | — | 0.071 / 0.267 | 0.074 / 0.315 | 0.150 / 0.533 |

(entries: $-\ln\mathrm{BC}$ / KL). `eq` and `uniform` are the best separated; `safe`, `br_unif`, `br_us` form a cluster
(0.07 nats Bhattacharyya per turn, 11–12% of matrices with $\mathrm{BC}>0.99$ to a neighbour). The a.s. constants
$\rho_s=\max_{\rm matrices}\mathrm{BC}$ are 0.81–1.00 (safe vs br_us: 1.000), so the **sup-version of (ii) is vacuous
on real matrices** ($\sum_s1/(1-\rho_s)$ = 132 for `eq`, $\infty$ for `safe`); the mean-coefficient version (v) is the usable one.

*Concentration under exact realisability ($\sigma_m=0$), $T=600$.*

| truth $s^*$ | $P_{25}(s^*)$ | $P_{50}(s^*)$ | $P_{100}(s^*)$ | MAP correct at $T=25$ | final MAP | realised $\sum_t(1-P_{t-1}(s^*))$ mean (max) | bound (v) $\sum_s1/(1-\bar\rho_s)$ | realised-BC-product bound, mean | $\ell_1$ bound (iv) $2\sum_s1/(1-\bar\rho_s)$ vs Pinsker $\sqrt{2T\ln5}$ at $T=600$ / crossover $T$ |
|---|---|---|---|---|---|---|---|---|---|
| eq | 0.981 | 1.000 | 1.000 | 0.97 | 40/40 | 3.85 (21.3) | 16.7 | 16.3 | 33 vs 43.9 / 346 |
| safe | 0.962 | 0.991 | 1.000 | 1.00 | 40/40 | 7.98 (17.1) | 41.1 | 39.8 | 82 vs 43.9 / 2,100 |
| uniform | 0.986 | 1.000 | 1.000 | 1.00 | 40/40 | 4.86 (21.0) | 26.1 | 26.3 | 52 vs 43.9 / 850 |
| br_unif | 0.943 | 0.999 | 1.000 | 0.97 | 40/40 | 8.08 (35.5) | 41.0 | 41.1 | 82 vs 43.9 / 2,090 |

The bound holds (expectation form, T3 is an expectation bound) and is 2–5× loose. Compared on the
quantity B4(ii) bounds, $\sum_t\lVert\hat q_t-q_{s^*,t}\rVert_1$ (realised $\le2\times$ the column above, i.e. 8–16), the
new constant $2\sum_s1/(1-\bar\rho_s)$ beats Pinsker's $\sqrt{2T\ln5}$ only beyond the crossover
$T^\times=(2\sum_s1/(1-\bar\rho_s))^2/(2\ln5)$: at the RQ stream length $T=600$ it wins for `eq` only (33 vs 44);
at the BH length $T\approx2{,}300$ (Pinsker 86) it wins for all four styles, and at the E2 budget $T=2.3\cdot10^4$
(Pinsker 272) by 3–8×. The point is the $T$-independence, not the constant. The i.i.d. identity (v) is confirmed: $\sum_s1/(1-\bar\rho_s)$ and the mean realised product agree
to 1%. Identification needs $\approx25$–50 turns, i.e. one or two games.

*Misspecified case (vi) under the mismatch proxy.* The Chernoff-½ coefficients $c(s)=\mathbb E_\sigma\sqrt{q_s/q_{s^\circ}}$
with $s^\circ$ the KL-closest style exceed 1 for one or more $s\ne s^\circ$ at every $\sigma_m>0$ tried (e.g. truth `safe`,
$\sigma_m=0.1$: $c=1.007$ for br_unif and br_us although their KL gap to `safe` is 0.11–0.13 nats/turn): the
log-likelihood-ratio increments are heavy-tailed under misspecification, so the ½-moment contraction of (vi) **fails
on the proxy** and only the first-moment (LLN) argument remains — concentration on $s^\circ$ still happens (table in §4)
but without a geometric rate from this lemma. (vi) is therefore proved but, as measured, not applicable at these
mismatch levels; it is kept because its hypothesis is checkable on logs ($c_t(s)$ is a function of the logged $q_{s,t}$
and the realised $b_t$'s empirical law).


---

## 4. Pre-registered predictions for the RQ runs (`hedge_bayes` vs `style_eq` / `style_safe` / `style_uniform`)

Pre-registered: committed in fb3cc8e (2026-11-16 01:13 −05:00) before any RQ run was launched; predictions below unchanged since.

*What the runs are* (from `run_ourbot.py:220–231`): the style bot samples the style distribution over **its own**
rows computed on **its own** $\bar A$ (its worlds for our hidden sets); our posterior scores $q_s$ computed on **our**
$\bar A$ (our worlds for its hidden sets). Realisability is exact only for `uniform` (no matrix dependence). The
mismatch proxy below is additive noise $A'=A+\sigma_m\Delta\,N(0,1)$ on the opponent's copy; the real inter-bot
$\bar A$ distance is **not logged** (synthesis §7 item 1), so predictions are given as a function of $\sigma_m$ with the
decision rule that would settle which regime the run is in.

*Model-based numbers* (`b4_concentration.py` §2–3; $T=600$ feedback turns per carried stream, which is what 100
seeds × 2 games / 10 streams × ≈28 feedback turns per game gives — if RQ uses a different stream count, rescale $T$):

| opponent | $\sigma_m$ | KL-closest style (margin, nats/turn) | $P_T(s^*)$ at $T=25$ / 100 / 600 | final MAP (40 paths) | $\sum_t(1-P_{t-1}(s^*))$ |
|---|---|---|---|---|---|
| style_uniform | any | uniform (0.474) | 0.99 / 1.00 / 1.00 | uniform 40/40 | 5.5 |
| style_safe | 0 | safe | 0.96 / 1.00 / 1.00 | safe 40/40 | 8.0 |
| style_safe | 0.05 | safe (0.194) | 0.86 / 1.00 / 1.00 | safe 40/40 | 12.1 |
| style_safe | 0.10 | safe (0.111) | 0.73 / 0.94 / 1.00 | safe 40/40 | 29.4 |
| style_safe | 0.20 | br_unif (0.001) | 0.31 / 0.15 / 0.13 | br_unif 20, br_us 15, safe 5 | 512 |
| style_safe | 0.40 | uniform (0.068) | 0.06 / 0.01 / 0.00 | uniform 38 | 596 |
| style_eq | 0 | eq | 0.98 / 1.00 / 1.00 | eq 40/40 | 3.9 |
| style_eq | 0.05 | **safe** (0.074) | 0.30 / 0.14 / 0.00 | safe 40/40 | 549 |
| style_eq | 0.10 | safe (0.040) | 0.00 / 0.00 / 0.00 | safe 37, br_us 3 | 598 |
| style_eq | 0.20 | br_us (0.025) | 0 | br_unif 21, br_us 16 | 599 |

**Predictions (pre-registered, 2026-11-16, no RQ results existed).**
P1. *Universal (N1, any opponent):* every clean stream ends with regret vs the best style $=\ln5+\ln P_T(s_{\rm best})$,
i.e. $1.609$ to 3 decimals wherever $P_T\ge0.999$; anytime max $\le1.609+$ rounding. Checked by the same script as
synthesis §3.2. A violation after the update-count check would refute the implementation, not the theory.
P2. *style_uniform:* $P_T(\text{uniform})\ge0.99$ by the 50th feedback turn of every stream and $=1.000$ at the end;
final MAP = uniform in 10/10 streams; the posterior never switches after turn ≈50 (MAP-wrong turns $\le10$ per
stream). Best style's log loss $=\mathbb E\ln n_t$ (the uniform's own loss, ≈ 2.3–2.9 nats). **Score**: `hedge_bayes`
beats a uniform opponent heavily; speculative point estimate $>0.8$ (no unit map, A7).
P3. *style_safe:* if the real mismatch is at or below the $0.1\Delta$ proxy, `safe` is identified in every stream
within ≈100 feedback turns (≈4 games), $P_T(\text{safe})=1.000$ at the end, MAP-wrong turns 10–50 per stream (T2(b)
with $\mu\approx0.1$–$0.2$, $\sigma\approx0.6$–1: $\sigma^2/(2\mu^2)\approx10$–50), final gap to the runner-up
(br_unif or br_us) $\ge60$ nats. If the mismatch is at the $0.2\Delta$ level the MAP ends on br_unif/br_us with
`safe` within a few nats (near-tie, §2 oscillation). **Decision rule**: the per-turn mean log-loss gap between `safe`
and `br_unif` on the logs ($\ge0.1$ nats/turn ⇒ low-mismatch regime). Point prediction: identified, 8–10 of 10 streams
end with MAP = safe. Score: `safe` is an exploitable near-deterministic column choice (softmax of column maxima at
$\tau=0.1\Delta$); speculative 0.6–0.75.
P4. *style_eq:* the sharp one. Under exact realisability `eq` would be identified by turn 50 (0.98 at $T=25$). Under
any mismatch $\ge0.05\Delta$ the opponent's LP equilibrium on its matrix looks, from ours, like `safe`
(KL 0.66 vs 0.76 for `eq` at $\sigma_m=0.05$; the LP solution's support is not stable under perturbation while the
softmax of column maxima is). **Prediction: `eq` is not identified** — final MAP $\in\{$safe, br_us, br_unif$\}$
(safe most likely), $P_T(\text{eq})<0.1$ in $\ge7/10$ streams, and `eq`'s mean log loss exceeds `safe`'s. If instead
$P_T(\text{eq})\ge0.9$ in most streams, the world-sampling mismatch is far smaller than $0.05\Delta$-equivalent for
the equilibrium functional, which would be worth knowing (it would also mean $\varepsilon_v$ is small). Score: an
equilibrium opponent on a nearly-equal matrix is the hardest of the three; speculative 0.50–0.60 (the 5% uniform mix
and the mismatch are the only exploitable parts).
P5. *Ordering* (robust to the unit map): identification speed and final $P_T(s^*)$: uniform ≻ safe ≻ eq; score:
uniform ≻ safe ≻ eq. The B4(iii) accuracy bound $\sqrt{2\ln5/T}$ is irrelevant at these $T$ — accuracy is decided by
whether the right style is identified at all (P3/P4), not by the rate.

*Caveats.* (i) The proxy replaces the real mismatch (different world samples on **both** sides, different column
sets, `not_in_worlds` skips) by i.i.d. Gaussian noise on one copy; the thresholds (0.05 / 0.1 / 0.2) are proxy
numbers, the *ordering* and the *fragility of `eq`* are the robust content. (ii) The 57 matrices are early-game
sized (median 9 columns here vs 18 logged); more columns make the softmax styles peakier and the LP support less
stable, which strengthens P4. (iii) p_ref in `hedge_bayes` is the Hedge mixture, not the one-hot safex; only `br_us`
depends on it.

---

## 5. B12: two posterior learners, and a posterior learner against Hedge (T4)

**Theorem T4.** [M0 (shared matrix) / M1 (mismatch); L7 + S; (a)–(c) **proved** (elementary); (d) **measured**;
(e) **conjecture**] *Setting.* One fixed matrix $A$ ($m\times n$, range $\Delta$, value $v$), played repeatedly; a
**safex player** on side $i$ plays $x_t\in F_\varepsilon(A)$ (any point; `bayes_safex` picks the LP maximiser of
$x^\top A\hat q_t$), a **Hedge player** runs anytime Hedge over its pure actions with average regret $\varepsilon_R(T)$.
(a) *Two safex players, shared matrix: every last iterate is an equilibrium.* For every $t$: $x_t$ is $\varepsilon\Delta$-exploitable
($v-\min_j(x_t^\top A)_j\le\varepsilon\Delta$), $y_t$ likewise, so $(x_t,y_t)$ is a $2\varepsilon\Delta$-Nash equilibrium of $A$
and $x_t^\top Ay_t\in[v-\varepsilon\Delta,\,v+\varepsilon\Delta]$; the time averages $(\bar x_T,\bar y_T)\in F_\varepsilon\times F^{\rm col}_\varepsilon$
(convexity) satisfy the same, and the time-average joint distribution $\frac1T\sum_tx_t\otimes y_t$ is a
$2\varepsilon\Delta$-CCE. No regret argument and no averaging is needed: the posteriors only choose *which* point of
$F_\varepsilon$ is played. This is the exact opposite of Lecture 7 §2.2 (MWU: the last iterate oscillates, only the
averages converge, with error $\varepsilon_R+\varepsilon_C\to0$): here the last iterate is always within a fixed,
non-vanishing band $\varepsilon\Delta=0.003\,\Delta$, and nothing improves with $T$.
(b) *Mismatch.* If side 2 computes $F_\varepsilon$ on $\tilde A$ with $\delta=\lVert\tilde A-A\rVert_\infty$, then its iterates
are $(\varepsilon\tilde\Delta+2\delta)$-exploitable in $A$ (an $\varepsilon$-optimal strategy of $\tilde A$ guarantees
$\tilde v-\varepsilon\tilde\Delta$ in $\tilde A$, $\ge v-\delta-\varepsilon\tilde\Delta$ in $A$ since $|\tilde v-v|\le\delta$ and
payoffs move by $\le\delta$) — the $\varepsilon_v$ term of the map, now on the opponent's side; the band in (a) widens to
$\varepsilon\Delta+2\delta$ and side 1 (correct matrix) can gain up to that much per round.
(c) *Hedge (rows) vs safex (columns).* Hedge's average payoff $\bar g$ satisfies $v-\varepsilon_R(T)\le\bar g\le v+\varepsilon\Delta$
(Lecture 7 eq. (1) for the left side with $\max_x x^\top A\bar y\ge v$; the floor of $y_t$ for the right side); $\bar y_T$
is $\varepsilon\Delta$-exploitable (convexity). **Nothing bounds $\bar x_T$'s exploitability**: Lecture 7's two-sided
argument needs the column player to be no-regret, and the safex player is a (constrained) best responder, which Lecture
6 §5 notes is not no-regret. Hedge's last iterate has no guarantee at all.
*Proofs.* (a) the LP constraint of `safex_strategy` is $(x^\top A)_j\ge v-\varepsilon\Delta$ for all $j$; the column
side symmetric; the payoff band is the two floors; convex combinations of points of $F_\varepsilon$ stay in
$F_\varepsilon$; CCE: a row deviation $i$ earns $\frac1T\sum_t(Ay_t)_i\le v+\varepsilon\Delta$ while the joint play earns
$\ge v-\varepsilon\Delta$. (b) two-matrix sandwich as stated. (c) as stated. $\square$

(d) *Measured* (`b12_selfplay.py`, 57 real matrices, $T=400$ rounds, sampled actions, posteriors updated on the realised
action, $\varepsilon=0.003$, both sides 5 styles with p_ref = the one-hot safex as in `run_ourbot.py:463`):

| pairing | payoff band $(x_t^\top Ay_t-v)/\Delta$: mean / min / max | last-iterate exploitability, max over $t$ and matrices (row / col) | time-average exploitability, mean (row / col) | posterior final MAP (row ; col) |
|---|---|---|---|---|
| safex vs safex, shared $A$ | $-0.0001$ / $-0.0030$ / $+0.0030$ | 0.0030 / 0.0030 | 0.0023 / 0.0022 | eq 52/57 ; eq 54/57; $P_T(\mathrm{MAP})\ge0.99$ on 57/57 both; median 0 MAP switches |
| safex vs safex, col sees $A+0.1\Delta\,N(0,1)$ | $+0.0174$ / $-0.0030$ / $+0.106$ | 0.0030 / **0.155** (median per-matrix max 0.033) | 0.0024 / 0.044 | eq 25, br_unif 11, br_us 10, safe 11 ; eq 27, br_unif 14, safe 9 |
| same, noise $0.3\Delta$ | $+0.0549$ / $-0.0019$ / $+0.274$ | 0.0030 / **0.488** | 0.0024 / 0.113 | br_unif 22, br_us 16, safe 10, eq 5 ; br_unif 21, safe 14, eq 11 |
| Hedge (rows) vs safex (cols) | $-0.0056$ / $-0.342$ / $+0.0029$ | **0.463** (median 0.210) / 0.0030 | **0.047** (max 0.263) / 0.0023 | — ; eq 36, br_us 12, safe 9; time-avg posterior eq 0.38 / br_us 0.39; median 5 switches |
| safex (rows) vs Hedge (cols) | $+0.0052$ / $-0.0030$ / $+0.508$ | 0.0030 / 0.508 | 0.0022 / 0.054 | eq 27, br_unif 9, br_us 10, safe 11 ; — |

Hedge's per-round regret was 0.0098 (bound $1.5\sqrt{\ln k/(2T)}\approx0.074$ at $k=7$). Readings. (i) The band in
(a) is attained exactly ($\pm0.0030$); self-play of two `bayes_safex` bots is a $0.006\Delta$-equilibrium at every round
on every matrix — B12(i)'s "bracketed near ½" is a theorem on a shared matrix, and (b) says that in the real game the
bracket is set by the world-sampling mismatch $\delta$, not by the dynamics: with $0.1\Delta$ noise the mismatched side is
exploited for $0.017\Delta$ per round on average (0.11 max), with $0.3\Delta$ for $0.055\Delta$. (ii) Against Hedge the
contrast of (c) is visible: the safex side's last iterate never leaves the $0.003$ band while Hedge's last iterate is
$0.21\Delta$-exploitable at its median worst round and $0.46\Delta$ at the worst, and even its time average is
$0.047\Delta$-exploitable — Hedge converges in payoff ($-0.006\Delta$, inside its regret bound) but not in strategy,
because its opponent is not no-regret. (iii) Posteriors: in exact self-play both concentrate on `eq` (the realised play is
a vertex of $F_\varepsilon$, which at $\varepsilon=0.003$ is $\ell_1$-close to $x^*$; `eq` is the KL-closest style), fully
($P_T\ge0.99$) and without switching. Against Hedge the KL-closest style changes as Hedge drifts, and the posterior
switches (median 5 times; time-averaged posterior split eq/br_us) — the §2 phenomenon in a controlled setting.
Kalai–Lehrer's grain of truth fails formally (no style equals the safex best response), but in self-play the play is
within $\varepsilon$ of the `eq` style's support, so the Berk-type conclusion (concentration on the KL-closest style)
is what is observed. (iv) The 5% uniform mix is what makes the 5 non-`eq` outcomes possible: the `eq` style of a
matrix with a non-unique or degenerate equilibrium differs from the vertex actually played.
(e) *Conjecture.* For a matrix with a unique equilibrium $(x^*,y^*)$ of full support on its optimal faces, as
$\varepsilon\to0$ and $\mathrm{MIX}\to0$ the self-play posteriors converge to `eq` a.s. and the play to $(x^*,y^*)$.
Not needed for anything above; stated because the measurement suggests it.

*Status.* (a)–(c) elementary, **new as statements in this project** (B12 had only the citation list); the Lecture 7
contrast (last iterate of an $\varepsilon$-safe player vs time average of a no-regret player) is the point to use in the
report. (d) new measurements.

---

## 6. Tracking after a style switch (T5)

**Proposition T5.** [M0; L6; (a) **proved** for the mean log-odds crossing, not the expected first-passage time; (b)–(c) **sketch**; **measured**]
*Setting.* The opponent plays style $s_a$ for $t_0$ feedback turns, then $s_b$; turn matrices i.i.d.; per-turn mean
log-likelihood drifts $\mu_{x\to y}:=\mathbb E\,\mathrm{KL}(q_x\|q_y)$ (the expected increment of $\ln q_x(b)/q_y(b)$ when
$b\sim q_x$).
(a) *$\lambda=1$ (shipped).* By N1 the log-odds of $s$ against $s_b$ at time $t_0+\tau$ have mean
$t_0(\mu_{a\to b}-\mu_{a\to s})-\tau\,\mu_{b\to s}$ (with $\mu_{a\to a}=0$, $\mu_{b\to b}=0$). The MAP becomes $s_b$ when
all of them are negative, so the expected flip delay is
$$\tau^*\ \approx\ \max_{s\ne b}\ \frac{t_0\,(\mu_{a\to b}-\mu_{a\to s})^+}{\mu_{b\to s}},$$
which is $t_0\mu_{a\to b}/\mu_{b\to a}$ (the $s=a$ term) unless a third style that was *less wrong* than $s_b$ during the
first phase and is *close* to $s_b$ afterwards dominates. The delay is **linear in $t_0$**: a posterior carried over
80 games takes of the order of 80 games to notice a switch (B3(a)'s mechanism, quantified).
(b) *Fixed share $\alpha$ (B2, not shipped).* $P_t(s)\ge\alpha/k$ for every $t$, so the log-odds against $s_b$ are at
most $\ln(k/\alpha)+B$ at the switch and $\tau^*\lesssim(\ln(k/\alpha)+B)/\min_{s\ne b}\mu_{b\to s}$, independent of $t_0$.
(c) *Discount-then-predict $\lambda$ (B3(b), one-line change).* The discounted log-odds against $s_b$ are in steady
state $\mu_{a\to b}/(1-\lambda)$ and decay as $\lambda^\tau\mu_{a\to b}/(1-\lambda)-(1-\lambda^\tau)\mu_{b\to a}/(1-\lambda)$,
giving $\tau^*\approx\ln\!\big((\mu_{a\to b}+\mu_{b\to a})/\mu_{b\to a}\big)/\ln(1/\lambda)$, independent of $t_0$.

*Measured* (`b4_concentration.py` §5; $t_0=300$, run to 1,200, 40 paths, "flip" = first turn after which the MAP
is $s_b$ for 20 consecutive turns):

| switch | $\mu_{a\to b}$ / $\mu_{b\to a}$ | (a) prediction, $s=a$ term only | (a) prediction with all $s$ | $\lambda=1$ measured mean (median, max) | fixed share $\alpha=0.01$ | reference $\ln(k/\alpha)/\mu_{b\to a}$ | DTP $\lambda=0.98$ | (c) prediction |
|---|---|---|---|---|---|---|---|---|
| safe → br_unif | 0.266 / 0.315 | 254 | 254 | 252 (243, 452) | 16 (14, 93) | 20 | 34 (32, 96) | 30 |
| eq → safe | 0.614 / 1.107 | 167 | 167 | 169 (167, 233) | 11 (9, 37) | 5.6 | 25 (24, 38) | 22 |
| uniform → eq | 2.064 / 1.509 | 410 | **642** (via safe/br_us) | 667 (670, 763) | 6 (4, 36) | 4.1 | 61 (61, 82) | 43 |
| br_unif → safe | 0.315 / 0.266 | 355 | 355 | 336 (335, 481) | 17 (16, 54) | 23 | 41 (39, 92) | 39 |

The $\lambda=1$ delays match (a) within 6% once the third-style term is included (uniform → eq: `safe` and `br_us`
were only 0.5–0.75 nats/turn behind `uniform` during the first phase while `eq` was 2.06 behind, so `eq` has to
overtake *them*). Fixed share and DTP react in 6–60 turns regardless of $t_0$, as (b)–(c) say. The reference column $\ln(k/\alpha)/\mu_{b\to a}$ is not the (b) bound and is exceeded by the measurements (eq → safe: 11 vs 5.6; uniform → eq: 6 vs 4.1); the actual bound $(\ln(k/\alpha)+B)/\min_s\mu_{b\to s}$ ($\approx48$ for eq → safe) holds. **For E3-type switch tests of a posterior bot** (E3a/E3f run
`hedge`, which has no posterior, so they do not test this): a carried $\lambda=1$ posterior will not track a mid-run
switch; ship fixed share or DTP if the switch is to be detected within a game or two. The per-turn drifts here are
for 9-column matrices; on 18-column matrices the softmax styles are peakier and the drifts larger, so the delays in
turns shrink but the ratio $t_0\mu_{a\to b}/\mu_{b\to a}$ does not change its order.

---

## 7. Ranked options, experiments, rejected ideas, recommendation

### 7.1 Ranked (impact × confidence ÷ cost; "new" = not in experts_extensions/synthesis)

| # | result | status | new vs known | evidence | what to do with it |
|---|---|---|---|---|---|
| 1 | **T3 Bhattacharyya concentration**: $\mathbb E\sum_t(1-P_{t-1}(s^*))\le\sum_s1/(1-\rho_s)$, constant in $T$; B6(c)'s $\sqrt{2T\ln5}$ → $2\sum_s1/(1-\rho_s)$ | proved (i)–(v); (vi) proved but hypothesis fails on the proxy | **new in this setting** (classical ingredients) | 57 matrices: realised $\sum(1-P)$ 4–8 vs bound 16–41; on the $\ell_1$ sum 33–82 vs Pinsker 44 ($T=600$), 86 (BH), 272 (E2): wins beyond $T\approx350$–2,100 | replace B4(ii)/B6(c)'s rate in the report; the only gain statement that is non-vacuous within one match |
| 2 | **T1 B3(c) settled**: coded forgetting violates $\ln k$ for all $\lambda<1$ (unbounded losses) and for $\lambda\le0.985$ at the logged $n=44$; closed form, threshold $B=\ln k+H(\lambda)/(1-\lambda)$; data-dependent bound | proved; thresholds measured; realisable at 0.5/0.7/0.9 | **new** (closes the open item; explains the ½ in the verifier's finding) | `b3c_all_lambda.py`, `b3c_realisable.py` | one-line fix to DTP if forgetting is ever turned on; keep `LAM=1` otherwise |
| 3 | **T2 oscillation = FTL on a drifted walk**; $\sigma^2/(2\mu^2)$ agrees in order of magnitude with MAP-wrong turns; FTL regret 5–18 vs 1.609 | (a),(c) proved; (b) sketch + measured | new measurement + framing (Lecture 6 object) | BH logs: 649 vs 563 | panel D caption; justifies the mixture over the MAP with a number |
| 4 | **§4 RQ pre-registration**: uniform ≻ safe ≻ eq; `eq` misidentified as `safe` at ≥0.05Δ mismatch | model-based prediction | new, falsifiable | `b4_concentration.py` §2–3 | compare when RQ lands; the `eq` outcome measures the equilibrium-functional mismatch |
| 5 | **T4 B12**: two $\varepsilon$-safe players are a $2\varepsilon\Delta$-NE at every last iterate (no averaging); Hedge vs safex sandwich; self-play posteriors → eq; mismatch sets the band | proved (elementary) + measured | new as statements; the Lecture 7 contrast | 57 matrices × 400 rounds | one paragraph in the dynamics section; replaces B12's "open" |
| 6 | **T5 tracking**: flip delay $t_0\mu_{a\to b}/\mu_{b\to a}$ with the third-style correction; fixed share / DTP constant | proved (mean log-odds crossing) + measured | new (quantifies B3(a)) | 4 switches × 3 learners | decide before any posterior switch test (E3 does not test it) |

### 7.2 Experiments run (all scratch, CPU, `engine/.venv/bin/python`; total ≈ 17 min)
- `theory/tightening/scripts/b3c_all_lambda.py` (15 s) → `out_b3c_all_lambda.txt`: closed form, thresholds, shipped-class check, log-space search, bound check.
- `theory/tightening/scripts/b3c_realisable.py` (12 s) → `out_b3c_realisable.txt`: real-matrix sequences.
- `theory/tightening/scripts/osc_bh.py` (1 s) → `out_osc_bh.txt|json`: BH streams.
- `theory/tightening/scripts/b12_selfplay.py 400` (2.6 min) → `out_b12_selfplay.txt|json`.
- `theory/tightening/scripts/b4_concentration.py 40` (11.5 min) → `out_b4_concentration.txt|json`; `b4_concentration_lib.py` shared.
- `theory/tightening/scripts/b4_chernoff_half.py` (36 s) → `out_b4_chernoff_half.txt`.

### 7.3 Rejected / not supported (add to experts_extensions §7)
- "B3(c) holds for $\lambda\ge\tfrac12$": refuted (T1); the ½ was an artefact of the verifier's build-up construction (cap-bound $B_0=B$ is optimal only for $\lambda<\tfrac12$).
- "The coded forgetting has *some* finite discounted bound $f(k,\lambda)$": refuted for unbounded losses (T1(a)); with the cap only the trivial $B/(1-\lambda)$-type bound and the data-dependent T1(c).
- "Bound the number of leader switches": no robust statement (Brownian crossings are discretisation-dependent); the duration $\sigma^2/(2\mu^2)$ is the quantity. Hoeffding's duration bound is vacuous at $\mu\approx0.02$.
- "The posterior's time average is a Lecture 7 object": no payoff, no game; it is FTL vs MWU (Lecture 6).
- "Sup-Bhattacharyya constant $\rho_s$ on real matrices": 0.81–1.00, vacuous; use the mean coefficient (i.i.d. form) or the realised product.
- "Chernoff-½ geometric rate under misspecification" (T3(vi)): hypothesis $c<1$ fails on the mismatch proxy at every level tried; concentration on the KL-closest style still occurs, by the LLN only.
- "Pathwise payoff regret of bayes_safex vs the best style in hindsight": not available — the safex argmax is not Lipschitz in $q$; only the expectation route (T3(iv)+B5(c)) goes through.
- "Both self-play posteriors might lock on `safe`/`br_us`" (concern raised in planning): measured 52–54/57 on `eq` at $\varepsilon=0.003$; the 3–5 exceptions have degenerate equilibria.
- "E3 tests posterior tracking": E3a/E3f run `hedge` (no posterior).
- "Kalai–Lehrer merging applies": grain of truth fails by construction (B12(ii)); what holds is T4(a) (trivially) and the Berk-type concentration (measured).

### 7.4 Recommendation
1. Take **T3** into the report as the replacement for B4(ii)/B6(c)'s rate, with the §3.1 table (realisable only; say so).
2. Record **T1** as the closure of B3(c) in experts_extensions (§1 B3, §6 row 11, §7), with the one-line DTP fix as the
   recommendation if forgetting is ever enabled; nothing changes for the shipped `LAM=1`.
3. Put **T2** behind panel D (the 649-vs-563 number and the FTL regret column) and **§4** on file before RQ is read;
   the `eq`-misidentification prediction is the one that can fail informatively.
What would change the ranking: if RQ shows `eq` identified ($P_T\ge0.9$), the mismatch proxy overstates the real
inter-bot distance and item 4's thresholds must be redone from the logged $q_{s,t}$ (and $\varepsilon_v$ is smaller than
feared); if the BH-style oscillation reappears against `style_safe`, the styles are closer on real mid-game matrices
than on the 57 early ones and T2's $\mu$ should be re-measured there.
