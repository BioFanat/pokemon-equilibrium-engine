# CHECK: `full_game.md` (adversarial check, 2026-11-14)

What was checked: `theory/full_game.md`, read in full, against `full_game_targets.md`, `engine/nash.py:47-61` (`rm_plus`), `engine/hidden_agents.py:36` (`agent_form_lp`), `engine/RESULTS.md:328-336`, and `lit/game_theory_noregret.md`.
Tools: Lean 4.34.1 with Mathlib. The project is `theory/lean/`; its `.lake/packages` is a symlink to an existing Mathlib build in `~/.claude/lean/proofs`. Python checks were run with `engine/.venv/bin/python`; the scripts are in `theory/checks/`.
Labels: **verified-Lean** / **verified-numeric** / **verified-by-reading** / **wrong** / **unclear**.

## Highest-risk findings first

1. **Example C.1 does not refute the targets' four-term form (wrong).** The note says $\eta_{\rm NL}=0$ ("$V_2$ is linear"), but that only covers stage 1. At stage 0 the continuation $V_1(s_L,\cdot)=\max(b_1-b_2,b_2/2)$ is not linear. The note's own definition gives $\eta_{\rm NL}(0)=\sup_{b'}[b'_1+\tfrac12 b'_2-\max(b'_1-b'_2,\tfrac12 b'_2)]=3/5$, attained at $b'=(3/5,2/5)$. With $\varepsilon_s=\varepsilon_v=0$ and $K\to\infty$, the targets' bound is therefore $\ge 3/5$ in the sum form, or $H\cdot 3/5=6/5$ in the $H\max$ form. Both exceed $\operatorname{expl}=1/5$, so the example is consistent with the four-term form.
   - **The definition cannot be narrowed to rescue the example.** One might restrict $\eta_{\rm NL}(t)$ to stages where the agent has a choice; stage 0 here has $|A|=1$. That breaks Theorem C's own proof. At stage 0 the revealed stage objective is $f_{b_0}=\tfrac12$ while $V_0=\tfrac15$, and this $0.3$ gap is absorbed only by the $\eta_{\rm NL}(0)$ term. The values $V_0=1/5$ and $\operatorname{expl}=1/5$ themselves are correct. Script: `checks/check_example_c1.py`.
   - **The conclusion survives with a different example (verified-numeric).** Take $\Theta=\{1,2,3\}$ with $b_0$ uniform. At stage 0 the agent has one action and the opponent picks $\beta\in\{1,2,3\}$, which leads to public state $s_\beta$. At stage 1 (the last) the agent picks risky or safe at $s_\beta$. Risky pays $-1$ against type $\beta$ and $+c$ against the other two types; safe pays $0$; $c\in(\tfrac12,1]$.
     - The hard-evidence agent plays risky everywhere. Its margin is $(2c-1)/3>0$, so the choice is unique and has $\varepsilon_s=0$.
     - Each type routes to its own state, so the agent gets $-1$. The value is $V_0=0$ (play safe), so $\operatorname{expl}=1$.
     - $\eta_{\rm NL}(0)=c/(1+c)$ and $\eta_{\rm NL}(1)=0$. With $c=3/4$, $\sum_t\eta_{\rm NL}(t)=3/7$ and $H\max_t\eta_{\rm NL}=6/7$, both below $1$.
     - It separates the four-term form from the five-term form (C.1); it does not contradict Theorem C. The (C.1) right-hand side is $\eta_{\rm NL}(0)=3/7$ at stage 0 (TV $=0$) plus $2\Delta\cdot\mathrm{TV}=2\Delta\cdot\tfrac23$ at stage 1 (true posterior $\delta_\beta$ vs. uniform $b^{\rm hard}$; $\Delta=7/4$ gives $7/3$). The total is $\ge1=\operatorname{expl}$.
     - Script: `checks/example_c1_repaired.py`.
   - **Two types are not enough in this two-stage family (verified-by-reading and numeric).** Let $\ell_{\beta\theta}=V_1(s_\beta,\delta_\theta)-g_{\beta\theta}(x_\beta)\ge0$ be type $\theta$'s loss at $s_\beta$ under the agent's choice. Since $\sum_\theta b_0(\theta)\ell_{\beta\theta}=\eta(0,s_\beta,b_0)$, we get $\operatorname{expl}\le\sum_\theta b_0(\theta)\max_\beta\ell_{\beta\theta}\le|\Theta|\,\eta_{\rm NL}(0)$. So with $|\Theta|=2$ and $H=2$, the $H\max$ form cannot be violated in this family.
     - Random search (`check_example_c1.py`): the largest ratio $\operatorname{expl}/\eta_{\rm NL}(0)$ was 1.50, below 2.
     - Exhaustive symmetric search (`search_symmetric_c1.py`): there are instances with $\operatorname{expl}>\eta_{\rm NL}(0)$, for example $u_L=[[1,-1],[-\tfrac12,0]]$ with $\operatorname{expl}=0.8$ and $\eta_{\rm NL}=0.6$. These violate only the sum form.
2. **§4 "Cor. A′ for the code's alternating scheme: not proved" is out of date (wrong).** The cross term in (4.1) is not "not sign-definite": for RM+ it is always $\le0$. This is Burch, Moravčík & Schmid 2019, Theorem 5 ($\sigma^{t+1}\cdot v^t\ge\sigma^t\cdot v^t$ for RM and RM+). It is proved in Lean here as `rmplus_improvement`.
   - The same argument covers the code's exact indexing. Write $r^x_{t+1}$ for the row regret of $(x_t,y_t)$, and add one virtual row update. Then
     $W[u(a,\bar y)-u(\bar x,\beta)]=\sum_t t\,r^x_{t+1}(a)+\sum_t t\,r^y_t(\beta)+\sum_t t\,[u(x_t,y_t)-u(x_t,y_{t-1})]$.
     The last sum is $\le0$ term by term, because the column player's RM+ update does not decrease its own utility against $x_t$.
   - Lemma 3's weighted bound then gives $\text{gap}\le 2\Delta(\sqrt{n(T+1)}+\sqrt{mT})/(T+1)\le 2\Delta(\sqrt n+\sqrt m)/\sqrt T$. So Proposition 4's bound, and Corollary A′ with $c=4$, **hold for `nash.rm_plus` as coded**.
   - BMS 2019 Theorem 8 (per the arXiv text as extracted by WebFetch, not read in full) states the same rate for CFR+ with alternating updates and weights $2i/(t^2+t)$, with player 2's average shifted by one index. For one information set per player its constant is $2|\mathcal I|\,l\sqrt{k/t}=4\Delta\sqrt{\max(n,m)/T}$.
   - Numeric checks: the identity holds to $3\cdot10^{-10}$; every cross step is $\le0$; the largest gap/bound ratio is 0.20 (`checks/check_rmplus_code_bound.py`). Against the note's bound over 1,920 matrix/T cases (240 matrices × 8 values of T, plus 3,000 random 2×2/3×3 matrices at T≤16), the code's ratio is at most 0.123 and the simultaneous variant's at most 0.172 (`checks/check_rmplus.py`).
3. **§6.1 author list (wrong).** Kovařík et al. 2023 is by Kovařík, Seitz, Lisý, Rudolf, Sun & Ha (arXiv 1906.06412; *Artificial Intelligence* 314:103805, 2023). Bošanský is not an author.
4. **§6.1 ReBeL Theorem 3 is misquoted (wrong in detail).** Per the arXiv text as extracted by WebFetch (not read in full), it reads $\varepsilon=\delta C_1+\delta C_2/\sqrt T$, not $C_1\delta+C_2/\sqrt T$. It also assumes no off-policy exploration at test time and a value network trained as in Theorem 2. "Constants linear in depth" is not in the statement (unclear).
5. **Theorem B′ is vacuous at the engine's likely $K$ (correct but weak).** $S\le\Delta\,\mathrm{TV}(\hat b,\text{empirical})\le\Delta$ holds trivially. The bound (ii), $\Delta\sqrt{2(n\log K+\log2)/K}$, only drops below $\Delta$ once $K/\log K\gtrsim 2n$. At $n=3$, $K=25$ it is $0.91\Delta$, while the measured $\mathbb E S/\Delta$ is 0.054 (`checks/check_theorem_b.py`).

## Part 1: Lean (`theory/lean/FullGame/Basic.lean`, builds with `lake build`; no `sorry`; `#print axioms` lists only propext / Classical.choice / Quot.sound, see `theory/lean/Axioms.lean`)

| Item | Lean name | Statement formalized | Non-vacuity |
|---|---|---|---|
| (a) Lemma 1 | `lemma1_one_step`, `lemma1_mixed` | `SupDist A B ε` ∧ `IsVal A vA` ∧ `IsVal B vB` ∧ `x` mixed ∧ `Guar B x (vB-εs)` ⇒ `Guar A x (vA-εs-2ε)`; the mixed form gives `vA-εs-2ε ≤ Σ_j y_j (xᵀA)_j` for every mixed `y`. `IsVal A v` means $v=\max_{x\in\Delta}\min_j(x^\top A)_j$ (attained, and no mixed $x$ guarantees more). | `example`: the note's $2\times1$ tightness instance ($\varepsilon=1$). All hypotheses are proved, and the conclusion holds with equality. |
| (b) Lemma 2(a) | `val_lipschitz` | `IsVal A vA` ∧ `IsVal B vB` ∧ `SupDist A B ε` ⇒ `|vA - vB| ≤ ε`, with the value taken over the simplex. | `exists_val`: `IsVal A v` exists for every $A$ with $n,m\ge1$ (compactness of `stdSimplex`). `example` on the instance above. |
| (c) Theorem A induction | `backward_induction`, `deficit_le_sum`, `theoremA_abstract` | Finite node type, history-dependent kernel `K t h h'` (which encodes $x\cdot\sigma(h)\cdot P$), live/terminal predicate. From $W=\sum K W'$, $D=\delta+\sum K D'$ and the stage hypothesis $V_t-\delta_t\le\sum K V_{t+1}$ (supplied by `lemma1_mixed`), it derives $W_t\ge V_t-D_t$ for all $t\le H$ and $D_t\le\sum_{t\le s<H}d_s$, hence $W_0\ge V_0-\sum_{t<H}d_t$. These are (A.0) and the middle inequality of (A.1). | `example`: one node, $H=2$, $\delta=1$; all hypotheses hold and the conclusion is tight. |
| (d) bonus | `rmplus_improvement` | $Q\ge0$ and $\langle Q,r\rangle=0$ ⇒ $\langle (Q+r)^+,r\rangle\ge0$ (BMS19 Thm 5). This closes finding 2. | `example` with $Q=(1,0)$, $r=(0,1)$. |

Limits of the formalization: `rmplus_improvement` gives $\langle(Q+r)^+,r\rangle\ge0$, which says "new strategy $\ge$ old" only when $(Q+r)^+\ne0$. The code's uniform-fallback case is covered by reading, not mechanized: $(Q+r)^+=0$ with $Q\ge0$ and $\langle Q,r\rangle=0$ forces $Q=0$, and then $\sum r=0$ forces $r=0$. The Lean statement of (c) is abstract. The step from the game model to the kernel (definition (1.1)) and from `lemma1_mixed` to `hstage` is by reading, not mechanized. The existence of the minimax value (von Neumann) is not used, because only the maximin form is needed. Reproducibility: `.lake/packages` is a symlink to `~/.claude/lean/proofs/.lake/packages`. If that directory moves, delete the symlink and run `lake exe cache get` (Mathlib rev `v4.34.1`).

## Part 2: claim-by-claim

| Claim (note §) | Verdict | Evidence |
|---|---|---|
| Lemma 1, factor 2 tight (§2) | **verified-Lean** | `lemma1_one_step`; the tightness `example` |
| Lemma 2(a) | **verified-Lean** | `val_lipschitz` |
| Lemma 2(b),(c) | verified-by-reading | Convex combination, then induction. Correct. |
| Lemma 2(d) "attained for every $d$" | verified-by-reading, minor caveat | A constant shift passes through $\mathcal P$ only if no transition reaches a terminal state, because terminal $\hat V^{(k)}=u$ is exact. Read it as "there exist games where the bound is attained". |
| Prop. 0 | verified-by-reading | |
| Theorem A, (A.0), (A.1), A1 (§3) | **verified-Lean** (abstract induction) + by reading | `theoremA_abstract`. Conditioning on the agent's past realized actions is handled correctly, because the induction runs over histories. |
| A2 for a best-response length | verified-by-reading | (A.0) holds for every $\sigma$, including the argmin, which exists for fixed Markov $\pi$ in a finite game. |
| A2 counterexample (end/cont) | verified-by-reading; wording caveat | $V^*_t(s_c)=-c\delta$, $\operatorname{expl}=H\delta$, "end at once" is minimax-optimal: all check. **Caveat:** "cont always" is *also* minimax-optimal (it holds every agent to $-(\#b)\delta\le0$) and gives $\tau=H$. So "the expected length under an equilibrium opponent" is ill-defined. The correct claim is "under *some* equilibrium opponent (e.g. end at once)". |
| Lemma 3 (RM+ regret) | verified-by-reading + numeric | Proof correct. On the code's accumulators, $\max\|R_t\|_2/(\Delta\sqrt{\dim\cdot t})=0.53\le1$ (`check_rmplus.py`). |
| Prop. 4 (simultaneous, linear averaging): $2\Delta(\sqrt n+\sqrt m)/\sqrt T$; uniform averaging gives $\Delta(\ldots)$ | verified-by-reading + numeric | Max gap/bound 0.172 on the simultaneous variant (`check_rmplus.py`). |
| Cor. A′, $c=4$ (linear) / $c=2$ (uniform) | verified-by-reading | Arithmetic correct. |
| §4 "(4.1) cross term ... not sign-definite" | **wrong** | It is $\le0$ for RM+ (Lean `rmplus_improvement`; BMS19 Thm 5; 0/240 positive in `check_rmplus.py`). |
| §4 "Cor. A′ for the code's alternating scheme: not proved" (also §8 row) | **wrong / now closable** | See finding 2. The bound holds for the code as written (`check_rmplus_code_bound.py`). |
| BMS 2019 characterization ("naive folk theorem can fail for alternating updates; counterexample + corrected theorem") | verified (arXiv 1810.11542) | Observation 1 is the counterexample (non-RM strategies $t \bmod 2$). Theorem 8 is the corrected CFR+ bound. JAIR 64:429–443. |
| RESULTS numbers quoted in §4 ($6.3\cdot10^{-4}\Delta$ at $T=100$, slope $-1.46$, quartiles $-1.68/-1.46/-1.25$) | verified (attribution) | `engine/RESULTS.md:333-336` |
| Lemma 5(a)–(d) | verified-by-reading | The midpoint trick is correct. It needs a *common* entry interval across $\theta$, which the note assumes. |
| Theorem B: $\varepsilon_s+2\Delta\mathrm{TV}+2\varepsilon_v+2S$ | **verified-numeric** + by reading | 1,500 random instances, $S$ computed *exactly* by LP over pure-column profiles, suboptimal $\hat x$ allowed: 0 violations, max lhs/rhs 0.869 (`check_theorem_b.py`). The seven-step chain checks. |
| Remark 1: $2\Delta\mathrm{TV}$ tight | **verified-numeric** | loss$/(2\Delta\mathrm{TV})=1.0000$ at $t=0.1,0.3,0.5$; $f_{\hat b}\equiv0$ to $10^{-17}$. |
| Remark 2: targets' $2R\cdot\mathrm{TV}$ with payoffs in $[-R,R]$ should be $4R\,\mathrm{TV}=2\Delta\,\mathrm{TV}$ | verified | Remark 1 gives loss $2\Delta\mathrm{TV}>\Delta\mathrm{TV}=2R\,\mathrm{TV}$ (half-range reading). The targets' constant is right only if $R$ means the range. Random instances do not separate the two readings (max ratio to $\Delta\mathrm{TV}$ is 0.955), consistent with the note's comment on the 0.48 check. |
| Lemma 6 (covering, $(3/\varepsilon)^{n}$) | verified-by-reading | |
| Theorem B′(i),(ii), $C=\sqrt2$, $K\ge9$ | verified-by-reading + numeric | $3\sqrt K\le K\iff K\ge9$. Absorbing the $+1$ needs $n\log K+\log(2/\delta)\ge2$, and $\log9>2$ gives it. The maximal inequality with $2N$ variables and proxy $\Delta^2/(4K)$ gives $\Delta\sqrt{\log(2N)/(2K)}$. Simulated: $\mathbb E S$ and q90 of $S$ are far below the bounds at every config (`check_theorem_b.py`). Exact $S$ was cross-checked against dense sampling (4,000 points), with no case where sampling exceeded the exact value. Weakness: vacuous for small $K$ (finding 5). |
| Lemma 7(a) convex, 2-Lipschitz in TV | verified-by-reading | For fixed $\pi$ the types decouple, so $V$ is a max of linear functions with coefficients in $[-1,1]$. |
| Lemma 7(b) inequality, perspective convexity, Sion step | verified-by-reading | The posterior after $(a,\beta,s')$ does not depend on $x$, as claimed. The perspective of a bounded convex $V$ is convex and continuous at $0$ ($|\mathrm{persp}(m)|\le\|m\|_1\max|V|$). $\Phi_b$ is linear in $x$ and convex in $y$, both domains are compact and convex, so Sion applies. |
| Theorem C (five-term decomposition) and proof | verified-by-reading | Stage guarantee: Theorem B with $\hat b=b$, then Jensen via concavity, then Lemma 5(c) twice. $\Phi\in[\mathrm{Rev}-\eta_{\rm NL},\mathrm{Rev}]$ and $\min_y\mathrm{Rev}=f_p$ are correct. The induction mirrors Theorem A. The fresh-sample argument (the opponent faces $\tilde x$) is correct. I found no gap. |
| Cor. C.1 (non-revealing ⇒ $p_h=b^{\rm hard}_h$) and its constants | verified-by-reading | $b\le4\Delta$: at $K=9$, $2\sqrt{2(1+\log2/\log9)}=3.24$. |
| Example C.1: $V_0=1/5$, $\operatorname{expl}=1/5$, agent plays $a_2$ uniquely | **verified-numeric** | LP gives $V_0=0.2$ at $x_L=x_R=(0.2,0.8)$; agent payoff 0 (`check_example_c1.py`). |
| Example C.1: "$\eta_{\rm NL}=0$", "all terms but TV equal zero", "the targets' form is false" | **wrong** | $\eta_{\rm NL}(0)=3/5$. Also $2\mathbb ES_1>0$ for finite $K$, so state the example with an exact-belief solve ($K\to\infty$). See finding 1 for a valid replacement. |
| "the TV term charges 2 at stage 1" | correct, but the (C.1) bound for the example is $2+3/5$ | |
| Two-sided remark | verified-by-reading | |
| §6.1 CFR-D (AAAI 2014, gadget, safe re-solving) | verified (paper page + abstract) | |
| §6.1 Bro17 quote "Resolving guarantees ..." | consistent with the lit review (`lit/game_theory_noregret.md:34`, PDF-read) | |
| §6.1 DeepStack Theorem 1 ($k_1\epsilon+k_2/\sqrt T$) | verified (arXiv 1701.01724) | The supplement's Theorem 2 gives $(d+1)k/\sqrt T+(2d+1)j\epsilon$ after $d$ re-solves, so "$k_1$ grows with re-solves" is right. |
| §6.1 ReBeL Theorem 3 | **wrong in detail** | Finding 4. |
| §6.1 Kovařík et al. authors | **wrong** | Finding 3. |
| Tammelin 2014 (arXiv 1407.5042) / Tammelin, Burch, Johanson & Bowling IJCAI 2015 | consistent | |
| §7 fit form | not checked numerically. Its $d$ inherits finding 1: the floor is real, but the claim that the TV term is *needed* should rest on the repaired example. | |

## Must-fix list (exact corrections)

1. **§6.3 Example C.1.** Replace "$\eta_{\rm NL}=0$ ($V_2$ is linear in $b$)" with "$\eta_{\rm NL}(1)=0$ ($V_2$ is linear in $b$) but $\eta_{\rm NL}(0)=3/5$ (at $s_L$, $b'=(3/5,2/5)$)". Then either:
   - (a) replace the example with the three-type example: risky $=(-1\text{ on type }\beta,\ +c\text{ on the others})$, safe $=0$, $c=3/4$. It has $V_0=0$, $\operatorname{expl}=1$, $\eta_{\rm NL}(0)=c/(1+c)=3/7$, and $H\max_t\eta_{\rm NL}=6/7<1$; or
   - (b) keep it, and change the conclusion to "consistent with the four-term form, which gives $\ge3/5$".

   Also say the agent solves $f_{b^{\rm hard}}$ exactly ($K\to\infty$, or `agent_form_lp` on the exact belief), so that the sampling term is zero. Update §8's row "Targets' four-term form against arbitrary opponents: false (Example C.1 ...)" to cite the new example. Optionally add the remark that in two-stage games of this shape $\operatorname{expl}\le|\Theta|\,\eta_{\rm NL}(0)$, so at least 3 types are needed to break the $H\max$ form at $H=2$.
2. **§4 "What the code does differently".** Replace "it is typically negative (alternation helps) but is not sign-definite" with "it is $\le0$ term by term, because an RM+ update never lowers the updating player's utility against the vector that produced it (BMS19, Thm 5)". Add the code-indexed derivation from finding 2, ending in "so $\varepsilon_s\le\text{gap}\le 2\Delta(\sqrt n+\sqrt m)/\sqrt T$ for `rm_plus` as coded (cf. BMS19 Thm 8)". Change §8's row "Cor. A′ for the code's alternating scheme: not proved" to "proved (cross term $\le0$; BMS19 Thms 5, 8)", and drop "further differences I did not analyse".
3. **§6.1 Kovařík.** Change "Kovařík, Seitz, Lisý, Bošanský et al." to "Kovařík, Seitz, Lisý, Rudolf, Sun & Ha, *Artificial Intelligence* 314:103805 (2023)", here and in the references list.
4. **§6.1 ReBeL.** Change it to "Theorem 3: with no off-policy exploration at test time, a value network of error $\le\delta$ trained as in Theorem 2, and $T$ CFR iterations per subgame, the algorithm plays a $(\delta C_1+\delta C_2/\sqrt T)$-Nash equilibrium ($C_1,C_2$ game-specific)". Drop "linear in depth" unless it is sourced.
5. **§3 A2.** Change "It cannot be replaced by the expected length under an equilibrium opponent" to "... under *an* equilibrium opponent (equilibrium opponents need not be unique: here 'end at once' and 'cont always' are both minimax-optimal, with lengths 1 and $H$)".
6. **§5, after Theorem B′ (recommended).** Add: "Since $S\le\Delta$ trivially, (ii) is informative only when $K/\log K\gtrsim2|A|$ (e.g. $|A|=3$ needs $K\gtrsim 30$); at the engine's $K$ the measured $S$ should be used instead."
7. **§2 Lemma 2(d) (minor).** Change "attained for every $d$" to "attained for every $d$ in games where no transition before stage $t+d$ reaches a terminal state".
8. **§8 status table.** Mark Lemma 1, Lemma 2(a) and the Theorem A induction as "proved here; Lean-checked (`theory/lean/FullGame/Basic.lean`)".
