# Related work for the "tightening" theory results (CS535 Pokemon agent)

Scope: results in `theory/tightening/{bounds_epsv,bayes_dynamics,voi}.md`. Lit notes already in this directory (Lon10, Fra98, Gan15, Lis13, Tim20, Bro18, Mor17, Per22, Das18, Sou05, Joh09, Her-style tracking in `next_directions.md`) are not repeated except where they are the closest citation.

Verification legend: **[PDF]** = full text read this session (via Undermind `read_pdfs` or local PDF text); **[abs]** = abstract/search snippet only; **[memory]** = cited from background knowledge, not checked (DOI/venue should be verified before use). Classification of our result: **new / adaptation / known**.

Honest caveat: this is a targeted search (about 25 Undermind queries plus a few web searches), not a systematic review. "I found no prior statement of X" means exactly that.

---

## 1. Lemma 1' (profile-restricted one-step lemma) and the signed pathwise residual identity

**What is known.**
- Error propagation for approximate DP in zero-sum Markov games is classical. Perolat, Scherrer, Piot, Pietquin (Per15) generalise the Scherrer et al. L_p error-propagation analysis to AVI/API/AGPI in zero-sum stochastic games. [PDF, local text] Theorem 1: `||l_k||_{p,rho} <= 2(gamma-gamma^k) C^{1}_{q,k,0}^{1/p} / (1-gamma)^2 * sup_{j<=k-1} ||eps_j||_{pq',sigma} + (1-gamma^k)... sup_j ||eps'_j|| + 2 gamma^k/(1-gamma) ...`, with eps_j the evaluation (value) error and eps'_j the greedy-step error; the concentrability coefficients are sup over all strategy sequences of Radon-Nikodym norms. Remark 5 there: with no second player the bound reduces to the MDP bound of Scherrer et al. 2012. The leading factor 2 and the sup over all profile sequences are exactly the "sup over cells" slack that our Lemma 1' removes.
- Earlier: Munos 2005/2007 (Mun05, Mun07, L_p-norm AVI bounds), Farahmand-Szepesvari-Munos 2010 (Far10, error propagation for API/AVI), Lagoudakis & Parr 2002 (Lag02, value-function approximation in zero-sum Markov games, [abs]). Singh & Yee 1994 is the usual source of the 2*eps/(1-gamma) greedy-policy loss (Sin94, [memory]).
- Depth-limited solving with learned leaves: Brown, Sandholm, Amos 2018 (Bro18; in earlier notes), Moravcik et al. 2017 DeepStack (Mor17). Kovarik & Lisy (Kov19b) formalise depth-limited solving with value functions. [PDF] They state: the paper "does not give a quantitative theorem bounding exploitability by a value-function approximation error". Their exact result is Prop. 4.20 (counterfactually optimal value functions: DL-CFR exploitability -> 0; "In general, the same is not true for reachably-optimal value functions") and Theorem 3 ("Reachably-optimal value functions preserve Nash equilibria"). Empirically (Sec. 6, Fig. 3) "validation Huber loss around 0.001 was sufficient for exploitability below 0.01" in goofspiel, Leduc, oshi-zumo.
- Simulation lemma / performance-difference: Kakade & Langford 2002 (Kak02, [memory]). Lobel & Parr 2024 (Lob24) give the tight simulation-lemma bound `|V - V^| <= 1/(1-gamma) - (1-eps_R)/(1-gamma(1-eps_T/2))`. [PDF] Their proof does use a signed decomposition `V - V^ = sum_t gamma^t [<P_t - M_t, R> + <M_t, R-R^> + <M_t - P^_t, R^>]` at the state-distribution level (p. 6) but then drops the last (nonpositive) term and symmetrises to an absolute value. No pathwise (per-trajectory) version.

**Closest citations.** Per15 (error propagation in games; the structure we refine), Kov19b (the depth-limited-leaf-error question in imperfect information; explicit gap), Lob24 (signed decomposition, then discarded), Lag02.

**How ours differs.**
- Lemma 1' (`val Q* - min_y x'Q*y <= eps_s + x*'(Q*-B)y^ + max_y x'(B-Q*)y`, tightening `eps_s + 2||B-Q*||_inf`): **adaptation/refinement** of the standard "greedy w.r.t. approximate Q loses 2*sup-error" step, restricted to the supports of the two equilibrium profiles. I found no prior statement of the profile-restricted form for matrix-game stage solvers; the proof (three-term split) is elementary, so the contribution is the form and the measurement that it cuts the budget 3-6x on exact endgames, not depth. Treat as "new but small".
- The pathwise telescoping identity `z - phi(v_1) = sum_t (-R_t - S_t + D_t)` is a trajectory-level performance-difference identity (**known in structure**; telescoping a value estimate along the realised path); the new part is (i) the three-way split into value-side, guarantee-side and realised-opponent residuals, (ii) using the signed sum, measured on logged games, to explain exploitability. Per15/Lob24 only bound absolute or sup quantities.

**Open questions these papers raise that our data can address.**
- Kov19b: no quantitative error-to-exploitability bound exists; their empirical Huber-loss vs exploitability curve (Fig. 3) is the analogue of our measured (eps_v ~ 1, exploitability small) mismatch. Our exact endgame chains (signed value-side +4.86 vs guarantee-side -4.13) are direct evidence for why sup-norm leaf error is a poor predictor, and could be offered as a counter-point to their "validation loss predicts exploitability" observation.
- Per15 states bounds via concentrability coefficients; none of these papers measures signed residuals along real play. Our 284 logged games (signed drift ~ -0.012/decision vs absolute 0.04-0.056) are such a measurement.

---

## 2. Exponentially discounted Bayesian / exponential-weights predictors on log loss

**What is known.**
- Chernov & Zhdanov, "Prediction with Expert Advice under Discounted Loss" (Che10, ALT 2010). [PDF] Protocol 1 (Sec. 1): the accountant announces alpha_{t-1} before predictions and `L_t = alpha_{t-1} L_{t-1} + lambda(gamma_t, omega_t)`: i.e. previous accumulated loss is discounted before the current loss is added and before the next prediction. Theorem 1 (Sec. 2, p. 4): for games satisfying their condition (3), the Aggregating Algorithm with Discounting (AAD, Alg. 2) guarantees `L_T <= c L_T^k + c ln K / eta`. For binary log loss with eta >= 1 the reading gives `c = eta` and thus `L_T <= eta L_T^k + ln K` (a multiplicative factor on the expert's loss, not an additive ln K regret; the paper does not separately spell this out, per the PDF reader). Theorem 2 (Sec. 3): bounded convex losses `L_T <= L_T^k + sqrt(B_T ln K / beta_T)`, and for alpha_t = alpha, `L_T <= L_T^k + sqrt(ln K/(1-alpha))`.
- Reid, Frongillo, Williamson (Rei14): constant regret `ln|Theta|/eta` iff the loss is eta-mixable (Theorems 1-2). [PDF] Log loss is eta-mixable only for eta <= 1; eta > 1 is not mixable (the paper does not state it separately; the PDF reader derived it from the mixability inequality: two experts putting all mass on different outcomes require probabilities >= 2^{-1/eta} each, summing > 1). With eta != 1 the update is a power/tempered likelihood `prod p_s(x_s)^eta`: eta < 1 tempers, eta > 1 sharpens.
- Tracking / forgetting: Herbster & Warmuth 1998 (Her95, "Tracking the best expert", fixed share); Kulhavy & Zarrop 1993 (Kul93, general concept of forgetting in Bayesian estimation); Bhattacharya, Pati, Yang (Bha16, fractional/tempered posteriors); Grunwald & van Ommen (Gru14, SafeBayes: tempering to repair misspecification). Cesa-Bianchi & Lugosi book (Ces06) for exponentially weighted forecasters.
- Search also returned Mhammedi & Williamson 2018 (Mha18) and Soft-Bayes (Ors17), neither used for the discounted question.

**Closest citation.** Che10: AAD is "discount-then-predict" (discount applied to the old accumulated loss before the next prediction). This matches our B3(b) variant for which a ln k discounted bound holds; the match is my reading of Protocol 1, not something the paper says in those words.

**How ours differs.** Theorem T1 concerns the *coded* "predict, then discount" variant (`D_t = lambda D_{t-1} + l_t`, predict from `e^{-D_{t-1}}`): (a) unbounded-loss counterexample with `R^lambda_2 -> lambda ln k + (1-lambda) B_0`; (b) exact two-turn value and threshold `B > ln k + H(lambda)/(1-lambda)`; (c) a data-dependent bound `ln k + (1-lambda) sum lambda^{T-t} range D_{t-1}`. I found no prior paper analysing the ordering predict-then-discount versus discount-then-predict for exponential weights; Che10 only treats the discount-first order. Classification: **new (small but not found elsewhere)**; the "discount-first gives ln k" part is **known** (Che10 Thm 1/2, c = eta = 1 case for log loss). The eta > 1 case is **known as non-mixable** (Rei14), with Che10's `eta L + ln K` as the best generic bound available; ours should be stated against that, not against "no bound".

**Open questions that our data can address.** Che10 gives no data on realised losses; Rei14 leaves the practical consequence of tempering with eta > 1 open. Our measured 57 real style matrices and loss ranges `B_t = ln(n_t/MIX)` give a realisable (not worst-case) check of when the 2-turn counterexample threshold `B > ln k + H/(1-lambda)` is crossed in practice (§1.1 of bayes_dynamics.md).

---

## 3. Two epsilon-safe exploiters: approximate NE at every iterate; last-iterate vs average

**What is known.**
- Safe exploitation: Ganzfried & Sandholm 2015 (Gan15, in earlier notes; the original "safe" notion relative to the game value). Johanson, Zinkevich, Bowling 2007 (Joh07, restricted Nash response). Jeary & Turrini 2023 (Jea23, "Safe opponent exploitation for epsilon equilibrium strategies"). [PDF] They define `sigma_i` eps-safe if `u_i(sigma_i, s_-i) >= v_i - eps` for all s_-i (Sec. 2); their new notion is prime-safety relative to the initial strategy's worst-case value; the paper "does not claim that the exploiter's changing strategy is itself an approximate Nash equilibrium", and does not study iterates or last-iterate behaviour.
- Muller et al. 2025 (Mul25, "Best of both worlds: regret minimization versus minimax play"). [PDF] Theorems 3.1 and 4.1: `R(mu^c) <= 1` and worst-case regret `~O(delta^{-1} sqrt T)`; bounds are cumulative in expectation, "no last-iterate or per-iterate guarantee is stated"; adaptive opponents are an open question (Appendix A).
- DiGiovanni & Tewari 2021 (Dig21, LAFF). [PDF] Guarantees are cumulative/high-probability regret and non-exploitability, not per round.
- Safe subgame exploitation: Ge et al. 2024 (Ge24c, [abs], PDF not accessible), Liu et al. 2022 (Liu22b, [abs]).
- Last-iterate vs average for no-regret dynamics: Daskalakis & Panageas 2018 (Das18), Mertikopoulos, Papadimitriou, Piliouras 2018 (Mer17, FTRL cycles/Poincare recurrence, [abs]), Lee, Kroer, Luo 2021 (Lee21, last-iterate in EFGs), Perolat et al. 2020 (Per20, regularisation to converge).

**Closest citation.** Gan15 for ε-safety; Mer17/Das18 for the contrast with MWU/FTRL last iterate. Jea23 and Mul25 are the closest *recent* safe-exploitation papers; both explicitly guarantee only cumulative quantities.

**How ours differs.** T4(a) (two safex players on a shared matrix: every iterate is a `2*eps*Delta`-Nash equilibrium; time-average joint play is a CCE) is **elementary / known by definition** of eps-safety (each iterate sits in the convex set `F_eps`; no learning argument is needed). The statement's value is as a contrast to MWU last-iterate oscillation, and the finding that the band does not shrink with T. T4(b) (mismatch widens band to `eps*Delta + 2 delta`) is the standard two-matrix sandwich. T4(c) (safex vs Hedge: nothing bounds `x_bar_T`'s exploitability) is an observation consistent with Lecture 6's "best response is not no-regret". I would not claim novelty beyond being a clean per-iterate statement for the *constrained-best-responder* (safex) class.

**Open questions that our data can address.** Mul25 leaves adaptive (non-oblivious) opponents open; Jea23 leaves multiplayer; neither measures what happens when two safe exploiters (or a safe exploiter and a no-regret learner) face each other. Our B12 two-learner self-play (57 matrices, T = 400) is such a measurement.

---

## 4. Posterior concentration via Bhattacharyya / Hellinger affinity martingales

**What is known.**
- Lattimore, Hutter, Sunehag 2013 (Lat13, "Concentration and confidence for discrete Bayesian sequence predictors"). [PDF] Theorem 1 (Sec. 3): `E_mu H_inf <= E_mu D_inf <= ln(1/w_mu)` (cumulative squared Hellinger and KL prediction error). Theorem 2 (credited to Hutter & Muchnik 2007): w.p. >= 1-delta, `H_inf <= ln(1/w_mu) + 2 ln(1/delta)`. Theorem 3: `D_inf <= e (ln 6/delta)(ln 2/delta + ln 1/w_mu)`. Theorem 6: finite uniform class `E C_inf <= 6 ln^2 K + 14 ln K + 8`. Lemma 14 (App. A): the likelihood ratio `xi_{<t}/mu_{<t}` is a supermartingale (Ville); Lemma 13 uses a Bhattacharyya-affinity inequality. They do not state a bound directly on posterior mass of wrong hypotheses (PDF reader).
- Merging of opinions: Kalai & Lehrer 1993 (Kal93, "Rational learning leads to Nash equilibrium", Econometrica), Sorin 1999 (Sor99), Pomatto-Al-Najjar-Sandroni 2014 (Pom14).
- Posterior contraction rates: Walker 2004 (Wal04, Hellinger-affinity martingale route to Bayesian consistency), Zhang 2006 (Zha06, Renyi/Bhattacharyya-type analysis, KL-entropy), Bhattacharya-Pati-Yang 2019 (Bha16, fractional posterior contraction via Renyi divergence, no testing conditions); Ghosal-Ghosh-van der Vaart 2000 [memory].
- Bayesian opponent modelling in games: Southey et al. 2005 (Sou05, "Bayes' bluff", [abs] + earlier notes), Baker & Cowling 2007 (Bak07, [abs]).

**Closest citation.** Lat13 (finite/countable class, Hellinger affinity, `ln(1/w)` and `K` dependence) and Wal04/Zha06 (affinity-supermartingale route).

**How ours differs.** Lemma T3 applies the standard Bhattacharyya-affinity supermartingale argument to a *finite hypothesis class of opponent style matrices* and evaluates the resulting geometric rates numerically on 57 real matrices. The mathematical device is **known** (Walker; Zhang; Lat13 Lemma 13/14); the instantiation and the numeric calibration are **adaptation**. Do not claim the bound is new. Lat13's Thm 2 is a cumulative Hellinger-error bound; a *geometric rate for the wrong-hypothesis posterior mass* under realisability is in the Walker/Zhang line, and I did not find a game-specific statement.

**Open questions that our data can address.** Lat13 Sec. 6 (p. 10): they expect the 1/delta dependence of the KL confidence bound can be improved to logarithmic, and flag a decreasing prior weight `w = 1/t` for countable classes as future work. Neither is tested on real opponent data; our 57-matrix Chernoff-half numbers (§3.1 of bayes_dynamics.md) are an applied check of the rate.

---

## 5. MAP = follow-the-leader on log loss; oscillation between near-tied hypotheses

**What is known.**
- FTL fails for adversarial/linear losses and is stabilised by regularisation or perturbation: Kalai & Vempala (Kal02 / JCSS 2005 [memory]), Hutter & Poland 2005 (Hut05, FPL, [abs]), Cesa-Bianchi & Lugosi Ch. 4 (Ces06). FTL enjoys fast rates for curved/regular sets: Huang, Lattimore, Gyorgy, Szepesvari (Hua16/Hua17, [abs]); for strongly convex or exp-concave losses FTL gets logarithmic regret (Hazan et al. 2007, [memory]).
- MAP on log loss is FTL on cumulative log loss with a prior regulariser; for a *stationary* well-specified class it is consistent (Kalai-Lehrer, Walker). I found no paper that specifically analyses MAP oscillation between near-tied hypotheses on a drifting stream.

**Closest citation.** Ces06 (FTL failure, Lecture 6 style) and Hut05/Kal02 (perturbation).

**How ours differs.** Proposition T2 (FTL on a random walk with drift; (a) proved by adapting Lecture 6 Sec. 2.3) is **known in kind** (FTL instability on near-tied leaders); the novelty is the measured oscillation rate in BH logs (8 streams, 1,969-2,450 feedback turns each) and the link to the posterior used by the agent. The log-loss structure is *not* the one that makes FTL fail (log loss is exp-concave for bounded probabilities); the near-tie flipping is a hypothesis-switching cost, which is why tracking regret (Her95) rather than static regret is the right comparator. Classification: **known / adaptation, with measured data new**.

**Open questions that our data can address.** Hua16 and the fast-rate FTL line ask when FTL is stable on curved sets; the discrete hypothesis simplex has no curvature, so near-ties are the generic bad case. Our logs give the empirical switching frequency.

---

## 6. Value of information / concealment in imperfect-information games, exact in small subgames

**What is known.**
- Computation: sequence-form LP, Koller, Megiddo, von Stengel 1994 (KMS94, [memory], STOC); von Stengel 1996 (vS96, [memory], GEB); Ponssard 1975 (Pon75, note on the LP formulation for incomplete-information sequential zero-sum games, [abs]).
- Value of information in zero-sum games: Sun & Ho 1976 (Sun76), Gossner & Mertens 2001 (Gos01, "The value of information in zero-sum games", unpublished manuscript, [abs]), Lehrer & Rosenberg (Leh03, [abs]), Mertens-Sorin-Zamir / repeated games with incomplete information (Zam92, Mer82, [abs]). Non-zero-sum: Hirshleifer 1971 (Hir71, negative private/social value of information), Levine & Ponssard 1977 (Lev77), Bassan, Gossner, Scarsini, Zamir 2003 (Bas03, "Positive value of information in games", [abs]), Bertschinger et al. 2013 (Ber13b, [abs]).
- PIMC and imperfect information: Frank & Basin 1998 (Fra98: strategy fusion), Long et al. 2010 (Lon10: strategy fusion and non-locality; "irrespective of the number of hypothetical worlds"), Arjonilla et al. 2024 (Arj24, EPIMC; Props. 1-3: depth never increases strategy fusion; the paper "does not give a bound on PIMC's error", per earlier notes). Furtak & Buro 2013 (Fur13, recursive MC search).

**Closest citations.** Lon10 (non-locality = what our eta_NL measures) and Sun76/Gos01 for zero-sum VOI; KMS94 for the LP.

**How ours differs.** The one-sided exact tera subgame (§2.1 of voi.md) computes `VOI_total = sum_theta b_0 V_0(delta_theta) - V_0(b_0)` and the non-locality term `eta_NL(0)` exactly for 30 Pokemon matchups, and checks the five-term Theorem C bound with the other terms zero. The sequence-form LP is **known**; applying it to measure non-locality / VOI on a real game's tera subgame, and showing that the A3 leaf's loss is bounded by `eta_NL` and actually small, is **new empirical application**. I did not locate a prior measurement of Lon10's non-locality in a real game's exact subgames (Lon10 used synthetic trees). In zero-sum games additional private information to a player cannot hurt him in the one-shot case (Blackwell-style), so negative VOI (Hir71) is a non-zero-sum phenomenon and should not be cited as expected in our setting; in our setting "VOI" is the oracle gap of a PIMC-style leaf.

**Open questions that our data can address.** Lon10 and Arj24 give no real-game measurement of how often strategy fusion/non-locality matters; our result (median exploitability 0, max 3.1%, >0.1% of range in 2/30 matchups) is such a measurement. Gos01-type results concern repeated/long games, not single-stage subgames with a hidden tera type.

---

## 7. Model-based exploiters that know the opponent's algorithm but not its private samples

**What is known.**
- Exploiting known learners: Deng, Schneider, Sivan 2019 (Deng19, "Strategizing against no-regret learners": a mean-based learner can be exploited above the Stackelberg value), Mansour, Mohri, Schneider, Sivan 2022 (Man22, Bayesian games), Brown, Schneider, Vodrahalli 2023 (Bro23, "Is learning in games good for the learners?"). Not read; abstracts only. These assume the optimiser knows the learner's *deterministic* update given observed actions.
- Opponent-model search: Li, Zanuttini, Ventos 2024 (Li24b). [PDF] "each OM describes a behaviour strategy of MIN"; models given as input; with probability `p_inf` MIN deviates adversarially; pure maxmin over multiple models NP-complete even with perfect information and 2 models (Prop. 6). So the model is a behaviour strategy, not an algorithm with private randomness.
- Exploitability of MCTS: Lisy et al. 2013 (Lis13), Kovarik & Lisy (Kov15b). [PDF for Kov15b] Theorem 4.1: there is a game and an HC selection rule where the average strategy converges outside the 1/5-Nash set; Theorem 5.12: with eps-HC, guaranteed exploration and eps-unbiased payoff observations, the average strategy is close to a `C'eps`-NE with `C' = 12(2^D-1) - 8D`; "The paper does not state a theorem or counterexample specifically about the max-visit output"; Sec. 6 "warns that using the last-iteration strategy can be exploitable."
- Best-response / exploitability estimation: Lisy & Bowling 2017 (Lis16, local best response, [abs]), Timbers et al. 2020 (Tim20, ISMCTS-BR / learned BR; in earlier notes), Milec et al. (Mil20, [abs]; Mil24).

**Closest citations.** Kov15b (what is and is not proved about MCTS output), Lis16/Tim20 (approximate best responses as lower bounds on exploitability), Deng19 (knowing the learner's algorithm).

**How ours differs.** The empirical finding (the white-box exploiter that knows the algorithm but not its private samples does not beat MCTS-2000 or RM+ play: difference -0.0014 +/- 0.0040 at n = 60/100) is a **new data point**: it is a *weak* approximate-best-response lower bound on exploitability, in the same role as LBR/ISMCTS-BR, with the particular property that the exploiter's model is exact except for the opponent's sampling. It is consistent with Kov15b's positive results for averaged strategies but does not contradict their counterexample (which concerns worst-case games). Not a theorem. The Deng19 line suggests the failure is due to the *private samples*: if the learner's randomness is unobservable, the exploiter's information advantage is only the algorithm, and for max-visit play that is a nearly deterministic map of the board, so the exploiter needs more than a model, it needs a *better evaluation* (consistent with Lemma 1').

**Open questions that our data can address.** Kov15b / Mul25: adaptive-opponent guarantees are open; our white-box exploiter is an adaptive opponent that fails to profit. Kov15b's max-visit gap: no theorem about max-visit output; our measurement (white-box exploiter vs max-visit play) is an empirical probe of that gap.

---

## 8. Time-budgeted engine evaluation: non-deterministic leaves under CPU load as a reproducibility threat

**What is known.**
- RL/benchmark reproducibility: Henderson et al. 2018 "Deep RL that matters" (Hen18, [memory], arXiv:1709.06560); Agarwal et al. 2021 (Aga21, [abs], statistical precipice); Clary et al. 2019 (Cla19b, [abs], variability of agents in Atari); Colas et al. 2019 (Col19, [abs]); Jordan et al. 2024 (Jor24, [abs]); Volz & Naujoks 2020 (Vol20, [abs], "Towards game-playing AI benchmarks via performance reporting standards"). None of these, from their abstracts, studies wall-clock-time-limited *search* agents.
- Computer chess practice (practitioner, not peer reviewed): TalkChess threads "nondeterministic testing" (https://talkchess.com/viewtopic.php?p=138632, p=138080), "fixed nodes testing" (p=416063), "Hardware oblivious testing" (p=484899). Snippets: time-based testing gives inconsistent results; "timing jitter alone can change the chosen move in time-per-move games, even with just 1 CPU searching"; fixed-node search removes the noise but "ignores the time-management logic"; Komodo's "Nodes Is Time" converts node counts to a pseudo-clock. [web snippets only, not fetched in full]
- Also related in MCTS: Huang, Coulom, Lin 2010 (Hua10, time management in Go MCTS, [abs]); Sironi & Winands 2019/2021 (Sir19, Sir21, [abs], randomised search-control parameters); Heinz 2000 (Hei00, self-play experiments revisited, [abs]).

**Closest citation.** The chess-engine testing folklore (TalkChess) and Agarwal 2021 / Cla19b for the general point that agent evaluation variance must be reported.

**How ours differs.** `scripts/leaf_nondet.py` (voi.md §2.5) measures the load dependence of the depth-limited leaf in our agent. I found **no peer-reviewed paper** that documents CPU-load-induced leaf non-determinism as a threat to reproducibility of game-AI benchmarks; the issue is well known in computer-chess practice, so the claim to make is "known folklore, quantified here for a Showdown agent", not new in principle. Classification: **known (practice) / new (quantified for this agent)**.

**Open questions that our data can address.** Vol20 asks for reporting standards for game-playing AI; our leaf-nondet measurement gives a concrete item (report the engine's budget in nodes or iterations, not only seconds, and the CPU load) for such a standard.

---

## Gaps

- Per15 PDF (not in Undermind): the bound was read from a locally fetched copy of the ICML PDF text; the constants in Theorem 1 above were taken from that text. Munos 2005/2007, Farahmand 2010, Singh-Yee 1994, Kakade-Langford 2002 not read.
- Gan15, Ge24c, Liu22b, Gos01, Sun76, Bas03, Hir71, KMS94, Sor99, Deng19, Man22, Bro23, Lis16, Hen18: abstract-level or memory only.
- Not searched in depth: Bernasconi et al. on safe exploitation in EFGs; Hazan-Kale on FTL log regret for exp-concave losses; Kamien-Tauman-Zamir 1990 (VOI in strategic conflict).
- No paper found on CPU-load nondeterminism in game-AI evaluation (result 8); searching computer-chess forums is not peer-reviewed.
- No paper found proving a profile-restricted per-node bound for depth-limited search with approximate leaves (result 1); a deeper search should target "restricted error", "policy-dependent" / "on-policy" value-error bounds, and "support-restricted" or "occupancy-measure-weighted" error in zero-sum Markov games.

---

## BibTeX

DOIs taken from Undermind metadata unless noted. Entries marked `% unverified` come from memory.

```bibtex
@inproceedings{Per15,
  author = {P{\'e}rolat, Julien and Scherrer, Bruno and Piot, Bilal and Pietquin, Olivier},
  title = {Approximate Dynamic Programming for Two-Player Zero-Sum Markov Games},
  booktitle = {ICML}, year = {2015}, pages = {1321--1329},
  url = {https://proceedings.mlr.press/v37/perolat15.html}}
@article{Mun07,
  author = {Munos, R{\'e}mi}, title = {Performance Bounds in L_p-norm for Approximate Value Iteration},
  journal = {SIAM J. Control Optim.}, year = {2007}, doi = {10.1137/040614384}}
@inproceedings{Far10,
  author = {Farahmand, Amir-massoud and Szepesv{\'a}ri, Csaba and Munos, R{\'e}mi},
  title = {Error Propagation for Approximate Policy and Value Iteration},
  booktitle = {NeurIPS}, year = {2010}, pages = {568--576}}
@inproceedings{Lag02,
  author = {Lagoudakis, Michail G. and Parr, Ronald},
  title = {Value Function Approximation in Zero-Sum Markov Games},
  booktitle = {UAI}, year = {2002}, doi = {10.48550/arXiv.1301.0580}}
@article{Kov19b,
  author = {Kov{\'a}rik, Vojt{\v e}ch and Seitz, Dominik and Lis{\'y}, Viliam and Rudolf, Jan and Sun, Shuo and Ha, Karel},
  title = {Value Functions for Depth-Limited Solving in Zero-Sum Imperfect-Information Games},
  journal = {Artificial Intelligence}, year = {2023}, volume = {314}, pages = {103805},
  doi = {10.1016/j.artint.2022.103805}}
@inproceedings{Lob24,
  author = {Lobel, Sam and Parr, Ronald},
  title = {An Optimal Tightness Bound for the Simulation Lemma},
  booktitle = {RLC (RLJ)}, year = {2024}, pages = {785--797}, doi = {10.48550/arXiv.2406.16249}}
@inproceedings{Kak02,  % unverified
  author = {Kakade, Sham and Langford, John},
  title = {Approximately Optimal Approximate Reinforcement Learning},
  booktitle = {ICML}, year = {2002}}
@techreport{Sin94,  % unverified
  author = {Singh, Satinder and Yee, Richard},
  title = {An Upper Bound on the Loss from Approximate Optimal-Value Functions},
  journal = {Machine Learning}, year = {1994}, volume = {16}, pages = {227--233}}
@inproceedings{Che10,
  author = {Chernov, Alexey and Zhdanov, Fedor},
  title = {Prediction with Expert Advice under Discounted Loss},
  booktitle = {ALT}, year = {2010}, doi = {10.1007/978-3-642-16108-7_22}}
@misc{Rei14,
  author = {Reid, Mark D. and Frongillo, Rafael M. and Williamson, Robert C.},
  title = {Generalised Mixability, Constant Regret, and Bayesian Updating},
  year = {2014}, doi = {10.48550/arXiv.1403.2433}}
@article{Her95,
  author = {Herbster, Mark and Warmuth, Manfred K.}, title = {Tracking the Best Expert},
  journal = {Machine Learning}, year = {1998}, doi = {10.1023/A:1007424614876}}
@article{Kul93,
  author = {Kulhav{\'y}, Rudolf and Zarrop, Martin B.}, title = {On a General Concept of Forgetting},
  journal = {Int. J. Control}, year = {1993}, doi = {10.1080/00207179308923034}}
@article{Bha16,
  author = {Bhattacharya, Anirban and Pati, Debdeep and Yang, Yun},
  title = {Bayesian Fractional Posteriors}, journal = {Ann. Statist.}, year = {2019}, doi = {10.1214/18-AOS1712}}
@article{Gru14,
  author = {Gr{\"u}nwald, Peter and van Ommen, Thijs},
  title = {Inconsistency of Bayesian Inference for Misspecified Linear Models, and a Proposal for Repairing It},
  journal = {Bayesian Analysis}, year = {2017}, doi = {10.1214/17-BA1085}}
@book{Ces06,
  author = {Cesa-Bianchi, Nicol{\`o} and Lugosi, G{\'a}bor}, title = {Prediction, Learning, and Games},
  publisher = {Cambridge University Press}, year = {2006}, doi = {10.1017/CBO9780511546921}}
@article{Jea23,
  author = {Jeary, Linus and Turrini, Paolo}, title = {Safe Opponent Exploitation For Epsilon Equilibrium Strategies},
  journal = {arXiv}, year = {2023}}  % arXiv id not checked
@misc{Mul25,
  author = {M{\"u}ller, Adrian and others}, title = {Best of Both Worlds: Regret Minimization versus Minimax Play},
  year = {2025}, note = {arXiv, 2025-02-17}}
@inproceedings{Dig21,
  author = {DiGiovanni, Anthony and Tewari, Ambuj}, title = {Balancing Adaptability and Non-exploitability in Repeated Games},
  booktitle = {UAI}, year = {2021}, pages = {559--568}}
@article{Gan15,
  author = {Ganzfried, Sam and Sandholm, Tuomas}, title = {Safe Opponent Exploitation},
  journal = {ACM TEAC}, year = {2015}, doi = {10.1145/2716322}}
@inproceedings{Ge24c,
  author = {Ge, Zhenxing and others}, title = {Safe and Robust Subgame Exploitation in Imperfect Information Games},
  booktitle = {ICML}, year = {2024}, pages = {15255--15270}}
@misc{Mer17,
  author = {Mertikopoulos, Panayotis and Papadimitriou, Christos and Piliouras, Georgios},
  title = {Cycles in Adversarial Regularized Learning}, year = {2018}, note = {SODA 2018; arXiv}}
@misc{Das18,
  author = {Daskalakis, Constantinos and Panageas, Ioannis}, title = {Last-Iterate Convergence: Zero-Sum Games and Constrained Min-Max Optimization},
  year = {2019}, doi = {10.48550/arXiv.1807.04252}}
@misc{Lee21,
  author = {Lee, Chung-Wei and Kroer, Christian and Luo, Haipeng}, title = {Last-iterate Convergence in Extensive-Form Games},
  year = {2021}, doi = {10.48550/arXiv.2106.14326}}
@misc{Lat13,
  author = {Lattimore, Tor and Hutter, Marcus and Sunehag, Peter},
  title = {Concentration and Confidence for Discrete Bayesian Sequence Predictors}, year = {2013}, note = {arXiv}}
@article{Kal93,
  author = {Kalai, Ehud and Lehrer, Ehud}, title = {Rational Learning Leads to Nash Equilibrium},
  journal = {Econometrica}, year = {1993}, doi = {10.2307/2951492}}
@article{Wal04,
  author = {Walker, Stephen}, title = {New Approaches to Bayesian Consistency},
  journal = {Ann. Statist.}, year = {2004}, pages = {2028--2043}}
@article{Zha06,
  author = {Zhang, Tong}, title = {From $\epsilon$-entropy to KL-entropy: Analysis of Minimum Information Complexity Density Estimation},
  journal = {Ann. Statist.}, year = {2006}, doi = {10.1214/009053606000000704}}
@article{Pom14,
  author = {Pomatto, Luciano and Al-Najjar, Nabil and Sandroni, Alvaro}, title = {Merging and Testing Opinions},
  journal = {Ann. Statist.}, year = {2014}, doi = {10.1214/14-AOS1212}}
@article{Sor99,
  author = {Sorin, Sylvain}, title = {Merging, Reputation, and Repeated Games with Incomplete Information},
  journal = {Games Econ. Behav.}, year = {1999}, doi = {10.1006/GAME.1999.0722}}
@misc{Sou05,
  author = {Southey, Finnegan and others}, title = {Bayes' Bluff: Opponent Modelling in Poker}, year = {2005}, doi = {10.48550/arXiv.1207.1411}}
@article{Hut05,
  author = {Hutter, Marcus and Poland, Jan}, title = {Adaptive Online Prediction by Following the Perturbed Leader},
  journal = {JMLR}, year = {2005}, doi = {10.48550/arXiv.cs/0504078}}
@article{Hua17,
  author = {Huang, Ruitong and Lattimore, Tor and Gy{\"o}rgy, Andr{\'a}s and Szepesv{\'a}ri, Csaba},
  title = {Following the Leader and Fast Rates in Online Linear Prediction: Curved Constraint Sets and Other Regularities},
  journal = {JMLR}, year = {2017}}
@article{Kal05,  % unverified
  author = {Kalai, Adam and Vempala, Santosh}, title = {Efficient Algorithms for Online Decision Problems},
  journal = {J. Comput. Syst. Sci.}, year = {2005}}
@inproceedings{KMS94,  % unverified
  author = {Koller, Daphne and Megiddo, Nimrod and von Stengel, Bernhard},
  title = {Fast Algorithms for Finding Randomized Strategies in Game Trees}, booktitle = {STOC}, year = {1994}}
@article{vS96,  % unverified
  author = {von Stengel, Bernhard}, title = {Efficient Computation of Behavior Strategies},
  journal = {Games Econ. Behav.}, year = {1996}}
@incollection{Sun76,
  author = {Sun, F.-K. and Ho, Y.}, title = {Value of Information in Zero-Sum Games}, year = {1976}, doi = {10.1007/BFb0120774}}
@unpublished{Gos01,
  author = {Gossner, Olivier and Mertens, Jean-Fran{\c c}ois}, title = {The Value of Information in Zero-Sum Games}, year = {2001}}
@incollection{Hir71,
  author = {Hirshleifer, Jack}, title = {The Private and Social Value of Information and the Reward to Inventive Activity},
  journal = {American Economic Review}, year = {1971}, doi = {10.1016/B978-0-12-214850-7.50038-3}}
@article{Bas03,
  author = {Bassan, Bruno and Gossner, Olivier and Scarsini, Marco and Zamir, Shmuel},
  title = {Positive Value of Information in Games}, journal = {Int. J. Game Theory}, year = {2003}, pages = {17--31}}
@article{Lev77,
  author = {Levine, Pierre and Ponssard, Jean-Pierre}, title = {The Values of Information in Some Nonzero Sum Games},
  journal = {Int. J. Game Theory}, year = {1977}, pages = {221--229}}
@article{Pon75,
  author = {Ponssard, Jean-Pierre}, title = {A Note on the L-P Formulation of Zero-Sum Sequential Games with Incomplete Information},
  journal = {Int. J. Game Theory}, year = {1975}, pages = {1--5}}
@inproceedings{Deng19,
  author = {Deng, Yuan and Schneider, Jon and Sivan, Balasubramanian}, title = {Strategizing against No-regret Learners},
  booktitle = {NeurIPS}, year = {2019}, pages = {1577--1585}}
@misc{Man22,
  author = {Mansour, Yishay and Mohri, Mehryar and Schneider, Jon and Sivan, Balasubramanian},
  title = {Strategizing against Learners in Bayesian Games}, year = {2022}, note = {arXiv}}
@misc{Bro23,
  author = {Brown, William and Schneider, Jon and Vodrahalli, Kiran}, title = {Is Learning in Games Good for the Learners?}, year = {2023}, note = {arXiv}}
@inproceedings{Li24b,
  author = {Li, Junkang and Zanuttini, Bruno and Ventos, V{\'e}ronique}, title = {Opponent-Model Search in Games with Incomplete Information},
  booktitle = {AAAI}, year = {2024}, doi = {10.1609/aaai.v38i9.28844}}
@article{Kov15b,
  author = {Kov{\'a}{\v r}ik, Vojt{\v e}ch and Lis{\'y}, Viliam}, title = {Analysis of Hannan Consistent Selection for Monte Carlo Tree Search in Simultaneous Move Games},
  journal = {Machine Learning}, year = {2019}, doi = {10.1007/s10994-019-05832-z}}
@inproceedings{Lis13,
  author = {Lis{\'y}, Viliam and Kovar{\'i}k, Vojt{\v e}ch and Lanctot, Marc and Bo{\v s}ansk{\'y}, Branislav},
  title = {Convergence of Monte Carlo Tree Search in Simultaneous Move Games}, booktitle = {NeurIPS}, year = {2013}, doi = {10.48550/arXiv.1310.8613}}
@misc{Lis16,
  author = {Lis{\'y}, Viliam and Bowling, Michael}, title = {Equilibrium Approximation Quality of Current No-Limit Poker Bots},
  year = {2016}, note = {AAAI-17 workshop; arXiv:1612.07547 (unverified id)}}
@misc{Tim20,
  author = {Timbers, Finbarr and others}, title = {Approximate Exploitability: Learning a Best Response in Large Games},
  year = {2020}, doi = {10.48550/arXiv.2004.09677}}
@misc{Hen18,  % unverified
  author = {Henderson, Peter and others}, title = {Deep Reinforcement Learning that Matters}, year = {2018}, note = {arXiv:1709.06560}}
@inproceedings{Aga21,
  author = {Agarwal, Rishabh and others}, title = {Deep Reinforcement Learning at the Edge of the Statistical Precipice},
  booktitle = {NeurIPS}, year = {2021}, doi = {10.48550/arXiv.2108.13264}}
@misc{Cla19b,
  author = {Clary, Kaleigh and others}, title = {Let's Play Again: Variability of Deep Reinforcement Learning Agents in Atari Environments},
  year = {2019}, doi = {10.48550/arXiv.1904.06312}}
@inproceedings{Vol20,
  author = {Volz, Vanessa and Naujoks, Boris}, title = {Towards Game-Playing AI Benchmarks via Performance Reporting Standards},
  booktitle = {IEEE CoG}, year = {2020}, doi = {10.1109/CoG47356.2020.9231705}}
```
