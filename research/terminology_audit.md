# Terminology audit and decisions (2026-11-16)

Audit of report/final5.tex, report/final.tex, report/current_report.tex, theory/theory_map.md, engine/RESULTS.md and
engine/figures/FIGURE_VOCAB.md against the literature. Evidence: [primary] = read in the source's own text
(PokaiTrainer arXiv 2608.29197v2; Ponssard & Sorin 1980; Long et al. 2010; Zinkevich et al. 2007); [lit] = quoted in
lit/*.md; [secondary] = descriptions of the source; [memory] = standard usage not re-checked. Code identifiers
(`agentform`, `hedge_bayes_mw`, ...) stay in code and in `\texttt{}` only; prose uses the terms below.

## Decisions (apply in every document)

| old term | decision | basis |
|---|---|---|
| agent form, agent-form game / LP / solver, "AF" | **Bayesian matrix game** (PokaiTrainer's term) for the formulation; eq. (1) is **its maximin LP, the one-sided case of Ponssard & Sorin (1980)** (incomplete information on one side); the bot is the **Bayesian-LP agent** ("Bayesian LP" in tables). Never attribute "agent form" to PokaiTrainer. | PokaiTrainer never uses "agent form" (0 hits, full text) and says "We solve every decision ... as a Bayesian matrix game" (§4.2) [primary]. In game theory "agent (normal) form" means one agent per information set (Kuhn/Selten) or per type (Myerson) [secondary]. Ponssard & Sorin 1980 Lemma 1 gives exactly eq. (1) for an uninformed player 1 [primary]. |
| PIMC, "PIMC blend", "PIMC averaging", "the blend" | **PIMC averaging**, defined once: "solve each sampled world as if it were known and play the b-weighted average of the per-world equilibria (a PIMC-style variant)". Drop "blend". | Long et al. 2010: worlds are "sampled and solved ... also called Perfect Information Monte Carlo (PIMC)" [primary]. |
| strategy fusion definition | "PIMC assumes it can use a different strategy in each world, although the worlds belong to one information set" [Frank & Basin 1998; Long et al. 2010]. Keep **non-locality** as a separate error. | Long et al. 2010 p.134: "Strategy fusion arises because PIMC search (incorrectly) believes it can use a different strategy in each world" [primary]. |
| η_NL "the non-locality gap behind strategy fusion" | "η_NL: the value of revealing the opponent's team to both sides (ours; it bounds the error Long et al. call non-locality)". Kept as our defined term. | Long et al.: non-locality = "a node's value may depend on other regions of the game tree" [primary]. |
| run IDs in prose (E1, E2, E4, E4u, E4umv, E5, AF, BH, G1, LOG, RQ, SW2/SW4, MW, L1/L2/L3 as experiments, W1, T5 P2/P3/P6...) | **Descriptive labels** in prose and tables; IDs only in an appendix / provenance column. Algorithm levels: "Level 1/2/3". | IDs collide across documents (E1, E4, W1, MW, L1 mean different things). |
| "Notation (one meaning per symbol)" | Remove the claim or fix collisions: world weights β_k (not b_k; b is the opponent action), detection delay d (not τ, the stopping turn), "search time" (not "budget"), meta-game mass not σ. | b, σ, k, τ, "budget" each had two meanings. |
| certified / certifies | "gives a 95% lower confidence bound on exploitability" | Exploitability = performance against the worst-case opponent (Zinkevich 2007 §5) [primary]; any fixed exploiter gives a lower bound. |
| white-box exploiter; model-based exploiter; exploitation gain; one-turn LBR | "a depth-2 **informed best response** that knows the target's policy (cf. PokaiTrainer's informed LBR)"; "**local best response (LBR)** [Lisý & Bowling 2017]"; define "gain" or drop it. | PokaiTrainer §5.2: "An informed local best response (Lisý & Bowling, 2017)" [primary]. |
| error budget; "Leaf-Limited Error Budget"; per-turn budget δ; clean error budget | "**whole-battle exploitability bound**"; "per-turn error δ". | The result is an exploitability bound (error propagation in zero-sum Markov games, Perolat et al. 2015) [lit]. |
| leaf, leaf error ε_v, matrix (leaf) error, ε_leaf, "the leaf is the bottleneck" | ε_v: **stage-matrix error**; ε_leaf: **leaf evaluation error**; "the leaf" → "the heuristic evaluation function"; "the leaf is the bottleneck" → "the leaf evaluation error dominates the bound" (keep "consistent with" wording). | Depth-limited search with a heuristic evaluation function at the leaves (Kovařík et al. 2023) [lit]. |
| calibrated ε_v | define once as **ε_v^aff := inf_{a>0,c} ‖aQ̂+c−Q*‖∞** ("affine-invariant stage-matrix error"; RM⁺ and the LP are invariant to positive affine maps); never "calibrated" elsewhere. | Not standard; "calibrated" means probabilistic calibration. |
| regret-minimising self-play (RM⁺, or an exact LP); "Self-Play per Turn" | "solves the stage game exactly (LP) or by regret-matching⁺ self-play"; drop "Self-Play per Turn" from titles. | An LP is not self-play. |
| solver gap / Nash gap | ε_s: **stage exploitability** of x (one-sided); Fact 1's two-sided quantity: **duality gap**. | Duality gap (e.g. Cai et al. 2023) [lit]; exploitability (Zinkevich 2007) [primary]. |
| foul-play / Foul-play / stock / "MCTS with foul-play's move rule" | **Foul Play** in prose; "Foul Play (decoupled UCT, N ms per sampled battle)"; in-engine "MCTS with Foul Play's final-move rule". Drop "stock". | Repository name "Foul Play"; it uses decoupled UCT [lit]. |
| decoupled MCTS | **decoupled UCT (DUCT)** [Shafiei et al. 2009; Lisý et al. 2013] | PokaiTrainer: "Decoupled UCT (Shafiei et al., 2009)" [primary]. |
| styles {eq, br_unif, br_us, safe, uniform} | keep "**style**", defined once as "a behavioural type in the sense of type-based reasoning [Albrecht & Stone 2018]"; labels: **stage equilibrium, best response to uniform, logit best response to our mixture, cautious (maximin), uniform**. | Albrecht & Stone: "each type is a complete specification of the agent's behaviour" [lit]. |
| safe exploit, q_safex | **ε-safe best response** to q̂ (ε = 0.3% of the payoff range) [Ganzfried & Sandholm 2015] | "Safe opponent exploitation" (Ganzfried & Sandholm 2015) [lit]. |
| safest; top move | **pure maximin** (security strategy); "top move" → **argmax play** (or keep "top move" defined as argmax of x) | standard [memory]; PokaiTrainer "play argmax of σ̄" [primary]. |
| "credits each expert counterfactually" | "**full-information feedback**: each expert is credited with the payoff its strategy would have earned against b_t" | "Counterfactual regret" is CFR's technical term (Zinkevich 2007) [primary]. |
| surrogate rewards / surrogate Ā | "**normalised stage payoffs** r_t (computed on Ā_t)" | "Surrogate loss" means a convex upper bound in learning theory [memory]. |
| Hedge cited to FS99 | Hedge is **Freund & Schapire 1997**; "exponentially weighted average forecaster" (CBL06 §2.1); MWU with (1−ηℓ) is Littlestone–Warmuth / Arora–Hazan–Kale. | [memory] |
| membership-aware; support fix; genuinely joint | "**per-world style likelihood**: each style's likelihood is computed on the actions legal in that sampled world"; "support fix" → "restricting the likelihood to each world's legal actions"; delete "genuinely". | coinages |
| re-concentration delay; tracking; false alarm | "**detection delay** after a switch"; keep "false alarm"; cite forgetting separately (Kulhavý & Zarrop 1993) from fixed share (Herbster & Warmuth 1998). | quickest change detection vocabulary [memory]; [lit] |
| gauntlet | **round-robin tournament** | 36 pairings = all pairs; a gauntlet is one engine vs the rest [secondary]. |
| stream; clean streams; str. | define: "**match**: a sequence of games over which Hedge weights and the posterior are carried"; state the criterion behind "clean". | — |
| scored turn; non-forced turn | define once: "a turn on which the opponent had at least two legal actions and its action was observed". | — |
| anchor claim | "Prediction tested". | — |
| one principle, three levels; lens; honest; buys robustness but not safety; resists the exploiter | "We apply no-regret learning at three places"; "lens" → "view"; "honest" → "valid"; "lowers exploitability but does not remove it"; "is exploited less". | slop |
| soft-best-responds; soft maximin | **logit best response** (temperature ...) [McKelvey & Palfrey 1995] | [memory] |
| world vs type θ | keep "**world**", defined once as "a sampled opponent team (a determinization)"; "team" (not type θ). | Long et al. "perfect information worlds" [primary]. |
| carried vs reset | "kept across the games of a match" vs "restarted each game" (keep the short words once defined). | — |
| stage experts | per-turn experts | — |
| most-visited arm | **most-visited action** ("robust child", Browne et al. 2012) | [secondary] |
| information-set planner (scouting) | "a planner that updates the posterior inside the lookahead" | — |
| [ext] = file path | "Extended report (supplementary material)" | — |

Confirmed standard (no change): exploitability; Bayes = exponential weights with η = 1 on log loss; non-revealing;
stage game; Nash averaging / meta-game; H̄ (ours, defined); seat-swapped pairs (poker: "duplicate").

Full audit with line references: verifier report of 2026-11-16 (session); sources at arXiv 2608.29197, Ponssard &
Sorin 1980, Long et al. 2010 (AAAI), Zinkevich et al. 2007 (NeurIPS).
