# Consistency, grammar and terminology fix list (reviewer pass, 2026-11-16)

## Consistency, grammar and terminology check: fix list (read-only, nothing edited)

**Verdict:** none of the six documents is clean. I found the replacement damage by diffing each file against HEAD at the word level (`git show HEAD:<file>` vs the working tree). I then read final5.tex and final.tex in full, read about 70% of RESULTS.md, and checked current_report.tex and theory_map.md through the diff plus targeted reads. Every line number below was confirmed in the current file.

---

### 1. Meaning-changing errors (fix before committing)

**1a. The offline joint-posterior table now names the wrong predictor.** The automated pass rewrote "membership-aware" as "per-world style likelihood". In the offline table, though, "membership" is a different predictor: research/joint_posterior.md:108–118 defines it as "same q̂, restricted to each world's columns". The per-world style likelihood is a separate row there. Row 1 now says it subtracts the per-world style likelihood, which is false, and row 2 is circular.
- engine/RESULTS.md:757 → "(current minus per-world style likelihood)" → "(current minus the averaged predictor restricted to each world's legal actions)"
- engine/RESULTS.md:758 → "per-world style likelihood (membership minus per-world)" → "per-world style likelihood (restricted averaged predictor minus per-world style likelihood)"
- engine/RESULTS.md:761 → "current 1.623 vs per-world style likelihood 1.299 nats" → "current 1.623 vs the restricted averaged predictor 1.299 nats"
- report/current_report.tex:1268 → same fix as RESULTS:757.
- report/current_report.tex:1269 → "per-world style likelihood (per-world style likelihood minus per-world)" → same fix as RESULTS:758.

**1b. ε_v^aff: final.tex still uses min over a ≥ 0, and several "a = 0" statements assume a = 0 is allowed.**
- report/final.tex:186 → "$\epsv^{\rm aff}:=\min_{a\ge0,c}\lVert a\Qhat+c-\Qstar\rVert_\infty$. At $a=0$ this is at least $\range\Qstar$, the trivial per-turn loss." → "$\epsv^{\rm aff}:=\inf_{a>0,c}\lVert a\Qhat+c-\Qstar\rVert_\infty$. As $a\to0$ it tends to the error of the best constant fit, the trivial per-turn loss."
  - **Send to `verifier`:** "at least range Q*" contradicts final.tex:419, where the best c at a = 0 gives ½·max(Q_i − Q_j), i.e. half the range. The same claim appears in theory/theory_map.md:315 ("with $a=0$ the budget is the trivial range(Q*)").
- report/final.tex:446 → "and $a=0$ is an optimal affine fit if and only if" → "and the best affine fit is constant (the infimum is approached as $a\to0$) if and only if". final.tex:419 ("$a=0$ is optimal iff") needs the same change.
- engine/RESULTS.md:386 → "The best fit is a constant (a = 0) at 9–67% of live states" → "The best fit is constant (a → 0) at 9–67% of live states"
- theory/theory_map.md:182–183 → "A per-state affine calibration $a_{t,s}\hat Q+c_{t,s}$ ($a\ge0$) … needs one global calibration (E2′)" → "A per-state affine fit $a_{t,s}\hat Q+c_{t,s}$ ($a>0$) … needs one global affine fit (E2′)"
- theory/theory_map.md:315 → "$2\varepsilon_v^{\rm cal}(t,s)$; with $a=0$ the budget is …" → "$2\varepsilon_v^{\rm aff}(t,s)$; as $a\to0$ the per-turn error tends to …"
- theory/theory_map.md:318–319 → "$\varepsilon_v^{\rm cal}$" → "$\varepsilon_v^{\rm aff}$"; "$a=0$ is an optimal calibration iff" → "the best affine fit is constant iff"
- theory/theory_map.md:216 and :632 → "best global affine fit $a=0$" / "calibration gives $a=0$" → "the best global affine fit is constant"
- final5.tex:149 and current_report.tex:824 already use inf over a > 0 (correct).

**1c. In-engine "Foul Play" vs the real Foul Play.** final5 calls the in-engine bot "MCTS (Foul Play's final-move rule)" and keeps "Foul Play" for the real client, timed in ms per sampled battle. The other documents use "Foul Play" for both.
- report/final5.tex:212 → "did not change the score against Foul Play at 75\,ms" → "did not change the score against MCTS with Foul Play's final-move rule at 75\,ms per move". This is the in-engine LF run (RESULTS.md:584), so final5 contradicts its own convention here.
- report/final5.tex:208 → "A PPO best response to Foul Play's MCTS (not our agent)" → "A PPO best response to MCTS with Foul Play's final-move rule (not our agent)". The run is `foulplay_it:2000` (RESULTS.md:478).
- report/final5.tex:187 (Table 1 caption) → "MCTS uses Foul Play's final-move rule" → "MCTS in rows 1–2 uses Foul Play's final-move rule; MCTS-2000 elsewhere plays its most-visited action". The LBR row and the Level 2 MCTS opponent are plain MCTS (RESULTS.md:459, :519).
- report/final.tex:192 → "\emph{Foul Play} applies Foul Play's selection rule to MCTS on one world under full information." → "\emph{MCTS (Foul Play's rule)} applies Foul Play's final-move rule to MCTS on one world under full information." Then rename the in-engine uses at final.tex:202 (×3), :206, :221, :249, :268, :276 (×4).
- report/final.tex:61, :75, :123 → "Foul Play's move rule" → "Foul Play's final-move rule"
- report/current_report.tex:764, :768, :771, :818 → "Foul Play 75 ms" / "Foul Play-2000" / "{Foul Play MCTS, …}" → "MCTS with Foul Play's final-move rule, 75 ms per move" / "…, 2,000 iterations"
- **Wrong units:**
  - engine/RESULTS.md:599 → "at 100 ms per move" → "at 100 ms per sampled battle"
  - engine/RESULTS.md:668–669 → "with N ms per state" → "with N ms per sampled battle"
  - engine/figures/FIGURE_VOCAB.md:35 → "N ms of search per state" → "N ms of search per sampled battle"

**1d. RESULTS.md:172 table cell.** A parenthetical was inserted into the middle of TV(b, b̂), and a one-sided quantity is labelled "duality gap".
- current: "| duality gap ≤ solver + belief + value error | L2: v(b) − f_b(x̂) ≤ ε_T + 2Δ·TV (Δ = payoff range; = 4·TV on [−1,1])(b, b̂) + 2‖M − M̂‖_∞"
- proposed: "| stage exploitability ≤ solver + belief + value error | L2: v(b) − f_b(x̂) ≤ ε_T + 2Δ·TV(b, b̂) + 2‖M − M̂‖_∞ (Δ = payoff range; 2Δ·TV = 4·TV on [−1,1])"

**1e. theory/theory_map.md:372.** Changing "certified" to "bounded" reversed the sentence's sense.
- "With $a=0$ on every exact endgame nothing is bounded." → "With the best fit constant on every exact endgame, the transfer bound is vacuous."

**1f. Stale PDFs.** report/final.pdf and report/current_report.pdf (18:01) are older than their .tex files (18:06), and both are staged as modified. Rebuild them before committing. final5.pdf (18:15) is current.

---

### 2. Grammar damage from the automated pass

- engine/RESULTS.md:1 → "Bayesian Bayesian-LP solving" → "Bayesian-LP solving"
- engine/RESULTS.md:661 → "test of the per-world style likelihood style posterior (`hedge_bayes_mw`)" → "test of the style posterior with the per-world style likelihood (`hedge_bayes_mw`)"
- engine/RESULTS.md:729 → "per-world style likelihood style posterior (per-world style likelihood restricted to each world's own columns)." → "Style posterior with the per-world style likelihood (each style's likelihood computed on each world's own columns)."
- engine/RESULTS.md:761, :763, :835 and theory/theory_map.md:1057, :1059–1060 → "The/the restricting the likelihood to each world's legal actions" → "Restricting the likelihood to each world's legal actions" (drop "the"). At theory_map:1059–1060, also fix "(§11) (the loss …)" → "(§11; the loss …)" and "Better prediction (the restricting …, `hedge_bayes_mw`)" → "Better prediction (restricting …; `hedge_bayes_mw`)".
- engine/RESULTS.md:721 → "detection delay time" → "detection delay"
- report/current_report.tex:1170 → "posterior's detection delay delay is linear" → "posterior's detection delay is linear"
- report/current_report.tex:1211 → "This is the first live test of T5, and and no pre-registered prediction" → "…of T5, and no pre-registered prediction". On the same line, "The detection delay time" → "The detection delay".
- report/current_report.tex:1219 → "\subsection{per-world style likelihood style posterior}" → "\subsection{Style posterior with the per-world style likelihood}"
- report/current_report.tex:1223 → "(the \emph{restricting the likelihood to each world's legal actions} of …), and its exploit expert solves the per-world safe-exploit LP" → "(restricting the likelihood to each world's legal actions, \cref{sec:bench-joint}), and its $\varepsilon$-safe expert solves the $\varepsilon$-safe best-response LP per world"
- report/current_report.tex:1273, :1275 → "The restricting …" / "we keep the restricting …" → "Restricting …" / "we keep the restriction of the likelihood to each world's legal actions"
- report/current_report.tex:1133 and report/final.tex:322 → "in a round-robin round-robin tournament" → "in a round-robin tournament"
- report/current_report.tex:107 → the parenthetical now reads "…(E1)) in a nine-bot round-robin tournament, passes every pre-registered refutation test of a live test of…". Proposed: "A final round, a nine-bot round-robin tournament (\cref{sec:bench}), places the agent below Foul Play at its shortest search time and not separable from PIMC averaging and the top move at the tournament's sample size (0.5375 against PIMC over 40 pairs, consistent with the larger Showdown run); the live test of the posterior's detection delay (T5) refuted none of its pre-registered predictions; and the per-world style likelihood's prediction gain replicates while its win-rate gain does not."
- report/current_report.tex:220 → "Bayesian-LP solves" → "the Bayesian-LP agent solves"
- report/current_report.tex:1105 vs :1108 → the paragraph repeats the caption sentence word for word ("The played mixture put 65% on Shiftry (a switch), which the agent played; the opponent replied with its own Shadow Ball."). Delete it from :1108.
- report/current_report.tex:1097 → "the belief moves to Best response to uniform" → "the belief moves to \emph{best response to uniform}"
- report/final.tex:61 (abstract) → "beats PIMC averaging (the Showdown Bayesian-LP vs PIMC run, E1) (0.5587 [0.5209, 0.5966])" → "beats PIMC averaging (0.5587 [0.5209, 0.5966])". The run ID should not be in the abstract either.
- report/final.tex:123 → "\emph{top} (the Bayesian LP's argmax row), … and \emph{$\varepsilon$-safe} ($\varepsilon$-safe, $\varepsilon=0.3\%$: the best reply…" → "\emph{top move} (…), … and the \emph{$\varepsilon$-safe best response} ($\varepsilon=0.3\%$: the best reply…"
- report/current_report.tex:692 → same pattern: "\emph{top} … \emph{$\varepsilon$-safe} ($\varepsilon$-safe best response, $\varepsilon=0.3\%$)" → "\emph{top move} … \emph{$\varepsilon$-safe best response} ($\varepsilon=0.3\%$)"
- report/final.tex:272 → "the Bayesian-LP solve beats the PIMC averaging 0.585" → "the Bayesian-LP agent beats PIMC averaging, 0.585"
- report/final.tex:313 → "consistent with the larger Bayesian-LP vs PIMC run (E1))." → "consistent with the larger Showdown Bayesian-LP vs PIMC run)."
- report/final.tex:317 → "\textbf{per-world style likelihood.}" → "\textbf{Per-world style likelihood.}"
- report/final.tex:437 → "Without the non-revealing assumption the TV term is necessary (a three-type, two-turn example): in a three-type, two-turn example where" → "Without the non-revealing assumption the TV term is necessary: in a three-team, two-turn example where"
- engine/figures/FIGURE_VOCAB.md:20 → "best reply (Ganzfried and Sandholm 2015): best reply to the predicted opponent strategy" → "best reply (Ganzfried and Sandholm 2015) to the predicted opponent strategy"
- theory/theory_map.md:60 → "gives, at the point estimate, the lower bound on exploitability $\ge 2(Z-\tfrac12)$" → "gives, at the point estimate, the lower bound $\operatorname{expl}\ge 2(Z-\tfrac12)$"
- theory/theory_map.md:740–741 → "so the bound is informative only in that the add-on is non-vacuous" → "so the ε-safe bound says only that its add-on is non-vacuous"
- theory/theory_map.md:445 → the bold heading now holds a 70-word parenthetical. Proposed: "**PIMC-averaging loss bound (PIMC averaging vs Bayesian LP, one stage).**", then move the Ponssard–Sorin and PIMC-averaging definitions into an ordinary sentence after it.

---

### 3. Leftover old terms in prose

- **"safe-exploit" / "safe exploit"**
  - report/final.tex:77 → "a safe-exploit sweep and model-based exploiters are measured" → "an $\varepsilon$-safe best-response sweep and exploiters that know a bot's algorithm are measured"
  - engine/RESULTS.md:573 "**Safe-exploit bot**" and :583 "vs the safe-exploit bot" → "ε-safe best-response bot"
  - report/current_report.tex:590 "safe-exploit expert" → "$\varepsilon$-safe expert"
  - report/current_report.tex:1367 "a safe-exploit sweep" → "an $\varepsilon$-safe best-response sweep"
- **"white-box"**
  - report/final.tex:332 → "(up to 40\% for LBR in the white-box runs)" → "(up to 40\% for LBR in the informed-best-response runs)"
  - engine/RESULTS.md:548 → "**White-box search exploiter (`wb2`).**" → "**Depth-2 informed best response (`wb2`).**"
- **"surrogate":** report/final.tex:89 → "Our surrogate-to-true transfer term" → "Our transfer term from the heuristic matrix to the true one"
- **"buys robustness but not safety" / "resists the exploiter"**
  - report/final.tex:204 → section title "Sampling the equilibrium appears to buy robustness, but not safety" → "Sampling the equilibrium lowers exploitability but does not remove it"
  - report/final.tex:241 → "Carrying Hedge weights across games resists an exploiter at no detectable cost" → "Carried Hedge weights are exploited less, at no detectable cost"
- **"tracking" / "re-concentrates", plus code-style style labels:** report/final.tex:315 → "\textbf{Tracking after a style switch.} When the opponent switches from the \emph{safe} to the \emph{uniform} style, the delay before the $\lambda=1$ posterior re-concentrates is predicted" → "\textbf{Detection delay after a style switch.} When the opponent switches from the cautious (maximin) to the uniform style, the detection delay of the $\lambda=1$ posterior is predicted". On the same line, change "30 streams", "per-stream" and "20 streams" to "matches".
- **"stream" vs "match".** final5 uses "match", defined at final5:110. final.tex defines "match" at :123/:125 but then says "stream":
  - :280 "each of 10 streams" → "each of 10 matches"; "all eight clean streams of BH" → "all eight matches with complete logs"
  - :282 ×2, :284 ("by tag and stream" → "by run and match"; "mid-stream" → "mid-match"), :317 "by stream" → "by match"
  - current_report.tex (about 60 uses), RESULTS.md (about 17) and theory_map.md (about 22) never define "stream". Either replace it globally or add the definition once (audit row 37).
- **Code labels as style names** (audit row 28):
  - report/final.tex:139 → "$\mathcal S=\{$eq, br\_unif, br\_us, safe, uniform$\}$: the opponent plays the stage equilibrium, …" → "five styles: stage equilibrium, best response to uniform, logit best response to our mixture, cautious (maximin) and uniform"
  - report/final.tex:282 → "\emph{eq}", "\emph{safe}", "\emph{br\_unif}/\emph{br\_us}", "\emph{uniform}" → "stage equilibrium", "cautious", "best response to uniform / logit best response", "uniform"
  - report/final.tex:305 (table) → "style eq / safe / uniform" → "stage equilibrium / cautious / uniform"
  - report/current_report.tex:391, :394 → "(eq)", "(safe)" → "(stage equilibrium)", "(cautious)"
  - report/current_report.tex:988 → "MAP $\in\{$safe, br\_us, br\_unif$\}$" → "MAP is cautious, logit best response or best response to uniform"
  - The "safe" style label next to the "ε-safe" expert is exactly the clash the audit removed.
- **"three-type" / "own type" → team**
  - report/final.tex:156 → "(a three-type, two-turn example;" → "(a three-team, two-turn example;"
  - report/final.tex:434 → "that knows its own type" → "that knows its own team"
  - report/current_report.tex:604, engine/RESULTS.md:367 ("the 3-type counterexample") and theory/theory_map.md:440–441 ("three-type Example C.1"; "At least 3 types") → team
- **"budget" for the per-turn error δ**
  - report/final.tex:171 → "the conditional expected sum of remaining budgets" → "of remaining per-turn errors"
  - report/final.tex:187 and :448 → "zero stage budget" → "zero per-turn error"
  - report/final.tex:430 → "to the per-turn budget" → "to the per-turn error"
  - theory/theory_map.md:46 → "per-stage budget $\varepsilon_s+2\varepsilon_v$" → "per-turn error $\varepsilon_s+2\varepsilon_v$"
  - theory/theory_map.md:314–315, :354, :366 → "budget" → "per-turn error"
- **"budget" for Foul Play's search time.** final5 uses "search time".
  - report/final.tex:313 → "sits below Foul Play's smallest budget" → "sits below Foul Play at its shortest search time"
  - report/final.tex:322 → "at each search budget … per budget" → "at each search time … per search time"
  - report/current_report.tex:107, :1117–1165 and engine/RESULTS.md:683–702 (for example "Foul Play budget (ms per sampled battle)" → "Foul Play search time (ms per sampled battle)")
- **"calibration" in the affine sense** (probabilistic calibration of beliefs, ECE etc. is correct and stays)
  - report/final.tex:268 → "rather than affine miscalibration" → "rather than an affine distortion"
  - report/final.tex:411 → "The calibration must be global" → "The affine map must be global"
  - engine/RESULTS.md:544 → "lacks information, not calibration" → "lacks information, not an affine rescaling"
  - theory/theory_map.md:631–632 (E2′ "global calibration $a>0$") and :839 ("calibrated Thm A") → "global affine fit" / "affine-invariant Thm A"
- **"Lens":** theory/theory_map.md:85 ("Lens tags"), :832 ("Lens:"), :834 (table header "Lens") → "View"
- **"genuinely"** (audit row 34): report/current_report.tex:1274 and engine/RESULTS.md:762 → "a genuinely joint posterior" → "a joint posterior"
- **Stale cross-reference:** report/current_report.tex:1413 cites the RESULTS section "``Safe exploitation''". That section is now titled "ε-safe best response and leaf rescaling" (RESULTS.md:572).
- **Code names in RESULTS prose and tables** (audit row 17 and the code-identifier rule): engine/RESULTS.md:604–614, :668, :677–681, :683–692, :824 use "agentform", "pimc", "top", "maxpower", "random" without backticks → "Bayesian LP", "PIMC averaging", "Bayesian LP, top move", "Max damage", "Uniform random". At :701, "Foul Play@25 against Foul Play@1000" → "Foul Play at 25 ms against Foul Play at 1000 ms".
- **"information-set planner":** engine/RESULTS.md:662 and :788 → "an information-set planner (S1)" → "a planner that updates the posterior inside the lookahead (S1)"
- **Run IDs in final.tex prose:** :286 "1 in 40,000 in AFdet; LSn counts both bots" → "1 in 40,000 in the deterministic-evaluation control; one run counts both bots". The Tag column in final.tex Table 3 is acceptable as provenance.

"Bayesian matrix game" is never attributed to Ponssard & Sorin in any of the six documents (verified). The PIMC-averaging and strategy-fusion definitions match the audit wording in final5:97, final.tex:87/:118, current_report:121/:789 and RESULTS:125. The only deviation is that the world-weight symbol is still b in final.tex, current_report, RESULTS and theory_map (see section 4).

---

### 4. Inconsistencies across documents

- **World weights b_k → β_k** (audit row 18; b is the opponent's action in every document):
  - report/final.tex:118 "$\max_x\sum_k b_k\min_j$", "the $b$-weighted average"
  - report/current_report.tex:121, :219–221, :704, :1305–1313
  - engine/RESULTS.md:67
  - theory/theory_map.md:445
  - In final.tex, β is already taken by the Hedge step sizes (:130, :409) and the baseline β (:417, :465). Rename those (e.g. step γ_t, baseline B) if b_k becomes β_k.
- **Duality gap vs stage exploitability**
  - report/final5.tex:134 → "whose measured gap falls about as $T^{-1.3}$" → "whose measured duality gap falls about as $T^{-1.3}$". The measured quantity is the duality gap (RESULTS.md:329, current_report:574).
  - report/final.tex:441 → "the measured RM$^+$ gap" → "the measured RM$^+$ duality gap"
  - report/final.tex:403 → "This gives $\mathrm{gap}\le$" → "This gives the duality-gap bound $\mathrm{gap}\le$ (which bounds $\epss$)"
  - engine/RESULTS.md:630–631 → "median stage gap 0.0585 / 0.0578", "positive stage gap", "stage gaps are similar" → "duality gap". final5:265 already calls the same numbers "median duality gaps".
  - report/current_report.tex:59 → `\DeclareMathOperator{\NashGap}{DualGap}` prints "DualGap" → `{gap}` (as in final5:56) or `{DualityGap}` (as in theory_map:475).
- **Turn units.** final.tex has "non-forced turns" (:123, :125) and current_report:400 has "per non-forced turn". final5 defines "scored turn" (:183), and final5:166 says "feedback turns". Use one term:
  - final5:166 → "for $t_0$ feedback turns" → "for $t_0$ scored turns"
  - final.tex:123 → "We let $t$ count non-forced turns." → "We let $t$ count scored turns (turns on which the opponent had at least two legal actions and its action was observed)." First check this is the same set the code counts.
- **Horizon unit:** final5:86/:183 and current_report:731 say "decision nodes"; final.tex:107/:192 and RESULTS.md:70 say "300-turn cap". Pick one.
- **Max damage definition:** final5:219 "base power × same-type bonus × type effectiveness" matches the code (engine/showdown/simple_bots.py:5, :62–68 include STAB = 1.5). engine/figures/FIGURE_VOCAB.md:33 "highest base power times type effectiveness" → "highest base power times same-type bonus (STAB) times type effectiveness"
- **MWU vs Hedge:** report/final.tex:96 → "MWU (randomised weighted majority~\cite{LW94}, Hedge~\cite{FS97}) … updates $w^{t+1}_i=w^t_i(1-\eta l^t_i)$" → "MWU (randomised weighted majority~\cite{LW94}) …". final5:88 and final.tex:125 both separate linear MWU from exponential Hedge.
- **Hedge citation:** report/final.tex:69 → "Hedge~\cite{FS97,Gan15}" → "Hedge~\cite{FS97}"; cite Gan15 at the safe-exploitation clause instead.
- **Expert and style names.** These match the audit everywhere I checked (top move, sampled, pure maximin, ε-safe best response; stage equilibrium, best response to uniform, logit best response to our mixture, cautious (maximin), uniform), except final.tex:123/current_report:692 ("top", "ε-safe") and the code labels in section 3.
- **Bayesian LP vs the Bayesian-LP agent**
  - report/final5.tex:154 → "and \cref{thm:C} Bayesian LP;" → "and \cref{thm:C} the Bayesian-LP agent;"
  - report/final.tex:77 → "The Bayesian LP advantage transfers" → "The Bayesian-LP agent's advantage over PIMC averaging transfers"
  - engine/RESULTS.md:124 → "The Bayesian LP advantage comes from" → "The Bayesian-LP agent's advantage comes from"
  - engine/RESULTS.md:138 → "the Bayesian matrix game beats PIMC averaging" → "the Bayesian-LP agent beats PIMC averaging"
- **Detection-delay symbol:** final5 uses d, d* and d̂. theory_map.md:1004 uses "flip delay τ*", and FIGURE_VOCAB:37–38 uses tau_map/tau_hat as code fields, which is fine. Change theory_map:1004 → "detection delay d*".
- **Forgetting vs fixed-share citation** (audit row 35): final5:170 "Fixed share and forgetting~\cite{HW98,CBL06}" and :273 "without forgetting or fixed share~\cite{HW98}". Cite forgetting separately (Kulhavý & Zarrop 1993); this needs a new bib entry.
- **Figure vocabulary:** the final5 Fig. 1 axis labels "top move" and "PIMC" vs FIGURE_VOCAB's "Bayesian LP, top move" and "PIMC averaging". Its caption also never states that the "N ms" are per sampled battle → add "Foul Play's search time in ms per sampled battle" to final5:219.
- **Tautological entries in FIGURE_VOCAB:**
  - engine/figures/FIGURE_VOCAB.md:5 → "pure maximin on the turn's payoff matrix" → "plays the pure maximin (security) row of the turn's payoff matrix"
  - engine/figures/FIGURE_VOCAB.md:24 → "plays the maximin (pure maximin) move" → "plays its pure maximin (security) move"

---

### 5. final5.tex: undefined terms, terms used before their definition, symbol collisions

**Undefined or used before definition**
- "scored turn" appears in the abstract (:67), the intro (:78) and the Table 2 caption before it is defined at :183. Move the sentence "A \emph{scored turn} is one where the opponent had two or more legal actions and its action was observed." from :183 to the end of "Bayes as Hedge" (:103).
- "one-turn LBR gain" (:195) and "gains significantly" (:208): gain is never defined. Add to the Table 1 caption: "LBR gain: the target's score against a same-strength non-exploiting opponent minus its score against LBR". This matches final.tex Prop. R5(b) and RESULTS.md:451.
- "measured gap" (:134): see section 4.
- "informed BR" is used in Table 1 (:196, :198), which floats above its definition at :208. Add to the caption: "informed BR: depth-2 informed best response that knows the target's policy".
- "Bayesian-LP agent" (abstract :67) and "Bayesian LP" (:154, tables) are never introduced. At :108 add: "We call this agent the \emph{Bayesian-LP agent} (Bayesian LP in tables)."
- "deterministic leaf evaluation" (:183) is used before its definition in Limitations (:280) → ":$\pm0.049$ for the deterministic-evaluation control (\cref{sec:discussion}; 299 pairs)"
- PIMC (:73/:97), PPO (:208), VGC and CFR (:73) are never expanded. Proposed: "Perfect Information Monte Carlo (PIMC)" and "a reinforcement-learning (PPO) best response". Optional at 5 pages.
- :261 → "Its score against Foul Play trends down" (the grammatical subject is Foul Play) → "The Bayesian-LP agent's score against Foul Play trends down"
- :261 and abstract :67 → "the agent sits below Foul Play's shortest search time" / "it rates below Foul Play's shortest search time" → "below Foul Play even at its shortest search time"
- :263 → "Hedge scores like its one expert alone (0.311 vs 0.314)" → "Hedge scores like the Bayesian-LP agent alone (0.311 vs 0.314)"
- :108 → "and \emph{pure maximin} the pure maximin row" → "and the \emph{pure-maximin} policy its pure maximin (security) row"
- :108 → "with the Rust engine poke-engine, whose cells average" ("whose" attaches to the engine) → "with the Rust engine poke-engine; its cells average"
- :178 → "leaves the identity entirely" → "is not covered by the identity"
- :208 → "is bounded below by any fixed opponent's score, by a Hoeffding margin" → "is bounded below by any fixed opponent's score, up to a Hoeffding margin"
- :275 → "play does not improve, so other terms ($\epsv$) dominate" → "play does not improve, consistent with other terms ($\epsv$) dominating" (keeps the "consistent with" rule)
- :110 → "(stage equilibrium, …, cautious (maximin), uniform; …)" has nested parentheses → "cautious (the maximin column)" or switch to a dash.
- :73 → "Our view is no-regret learning" → "We view the agent through no-regret learning"

**Symbol collisions** (the abstract and §2 rely on one meaning per symbol)
- **d:** "exact depth-$d$ lookahead … for every $d\ge1$ … before stage $t+d$" (:156) vs detection delay d (:166–167, :245–247) → depth $D$ (or $\ell$) at :156.
- **d* vs d̂:** Prop. 7 names the prediction d* (:167); Table 2 and the text call it d̂ (:245–246). Pick one.
- **s:** state (:86, :137, Thm 3) vs style (:103, :160, :166) vs time index (:88, "$\sum_{s<t}$"). Rename the sum index to $\tau'$ or $j$; consider σ̃ or $\theta$ for styles. Note that σ is the opponent's policy.
- **k:** number of experts (:88, :110, "$k=2$", "$k=4$") vs world index (:97, :99, :119; $\beta_k$, $M_k$, $\omega^k$). Use $m_E$ or $J$ for the expert count.
- **β:** world weights $\beta_k$ (:97) vs hard-evidence belief $\beta_h$ (:152). These are related; either say "$\beta_h$, the world weights at history h" or rename.
- **P:** transition kernel (:86) vs posterior $P_t$ (:103).
- **p:** Hedge mixture $p_t$ (:88, :110) vs true posterior $p_h$ (:152).
- **a:** our action (:86) vs affine scale $a$ (:149).
- **L:** total loss L and L* (:88) vs Hedge reward sums $L_e$ (Algorithm 1, :118, :124, :126) → $R_e$ in Algorithm 1.
- **b vs β_k:** done correctly in final5. **τ vs d:** done correctly apart from the depth clash above. **σ:** opponent policy only (correct).

**Layout (from the rendered final5.pdf)**
- 5 pages of text plus references on page 6. Check whether the 5-page limit excludes references.
- The code URL breaks across columns at the bottom of p.3 ("Code: https:" / "//github.com/…" on p.4) → add `\allowbreak` or move the URL to a footnote.
- The §5.1 heading has stretched inter-word spacing ("In-engine:  strength,").
- Tables and figures otherwise render cleanly and are readable.

---

### 6. Minor / optional
- final5:97: the audit says to present eq. (1) as "the one-sided case of Ponssard & Sorin (1980)". final5 never cites PS80, which is not wrong; add it optionally (needs a bib entry).
- final.tex:89 → "No external regret goes back to Hannan" → "No-external-regret learning goes back to Hannan"
- final.tex:107 → "scores $s=(u+1)/2$" collides with state s → "scores $(u+1)/2$"
- final.tex:337 → "a clean whole-battle exploitability bound" ("clean" is the old modifier) → "a whole-battle exploitability bound"
- engine/RESULTS.md:29, :131, :476 → "MCTS (200 iterations) (actually 1,000; …)" has back-to-back parentheses → "MCTS (200 iterations requested, 1,000 run: poke-engine rounds up to multiples of 1,000)"
- engine/RESULTS.md:385 → a full sentence sits inside a parenthetical ("… It is 5.1 if …) where exploitability > 0.") → split it out.
- engine/RESULTS.md:398, :845 → "the leaf evaluation error dominates the bound" stated without "consistent with" → "consistent with the leaf evaluation error dominating the bound"

### Per-document status
- **report/final5.tex:** not clean. Mostly sections 1c and 5, plus :134 and :154. The terminology itself (Bayesian matrix game, PIMC averaging, strategy fusion, experts, styles, Foul Play unit, ε_v^aff) is otherwise correct.
- **report/final.tex:** not clean. Sections 1b, 1c, 2 (doubled phrases, abstract run ID), 3 (stream, safe style labels, budget, white-box, surrogate, section titles), 4.
- **report/current_report.tex:** not clean. Sections 1a, 2 (delay delay, and and, round-robin round-robin, "the restricting", lowercase subsection), 3, 4. Its PDF is stale.
- **engine/RESULTS.md:** not clean. Sections 1a, 1b, 1c units, 1d, 2 (title, :729, "the restricting"), 3 (agentform, safe-exploit, white-box).
- **theory/theory_map.md:** not clean. Sections 1b (calibration, ε_v^cal, a ≥ 0), 1e, 2, 3 (Lens, per-stage budget, three-type, τ*).
- **engine/figures/FIGURE_VOCAB.md:** not clean, but minor. :35 unit, :33 STAB, :20 doubled "best reply", :5/:24 tautologies.

### Supporting files (scratchpad)
- Word-level diffs vs HEAD:
  - /private/tmp/claude-501/-Users-biofanat-Documents-classes-f26-cs535/2bbe670b-2f97-4d9a-bfad-e1139e0352c6/scratchpad/wd_current_report.tex.txt
  - /private/tmp/claude-501/-Users-biofanat-Documents-classes-f26-cs535/2bbe670b-2f97-4d9a-bfad-e1139e0352c6/scratchpad/wd_final.tex.txt
  - /private/tmp/claude-501/-Users-biofanat-Documents-classes-f26-cs535/2bbe670b-2f97-4d9a-bfad-e1139e0352c6/scratchpad/wd_RESULTS.md.txt
  - /private/tmp/claude-501/-Users-biofanat-Documents-classes-f26-cs535/2bbe670b-2f97-4d9a-bfad-e1139e0352c6/scratchpad/wd_theory_map.md.txt
  - /private/tmp/claude-501/-Users-biofanat-Documents-classes-f26-cs535/2bbe670b-2f97-4d9a-bfad-e1139e0352c6/scratchpad/wd_FIGURE_VOCAB.md.txt
- Leftover-term grep hits: /private/tmp/claude-501/-Users-biofanat-Documents-classes-f26-cs535/2bbe670b-2f97-4d9a-bfad-e1139e0352c6/scratchpad/leftover.txt
- Source behind item 1a: /Users/biofanat/Documents/classes/f26/cs535/final-integration/research/joint_posterior.md:108–118
- Source behind the Max damage fix: /Users/biofanat/Documents/classes/f26/cs535/final-integration/engine/showdown/simple_bots.py:5, :62–68