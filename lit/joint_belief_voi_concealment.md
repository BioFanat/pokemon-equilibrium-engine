# Joint type x style beliefs, value of information, concealment: literature notes (CS535 Pokemon agent)

Date: 2026-11-16. Method: Undermind `search_papers` / `lookup_papers_by_metadata` in workspace "Untitled Research Project" (search only, nothing starred or written), then `read_pdfs` sub-agents for quotes, then my own `pdftotext` spot-check of the PDFs I downloaded (Alb15b, Yu26d).
Cite keys follow the Undermind keys used in `/Users/biofanat/Documents/classes/f26/cs535/final/lit/` (Sou05, Gan16, Joh07, Joh09, Lon10, Fra98, Bro20, Sch21, Mor17, Iha18, Whi11, Pow14, Gos01, Gri25, Kar25b, Yu26d, Li24e). New keys: Alb15b, Alb17, Gmy05, Dos05, Dos05b, Bar07, Bak07, Zha24h, Cow15, Gua26b, Cho87, Sor96, Oku90, Shm06, Lau16, Vei23.

Evidence tags:
- **[spot-checked]**: quote re-found by me in a local PDF text dump.
- **[PDF-read]**: quote returned by an Undermind `read_pdfs` sub-agent from full text; NOT re-verified by me. Re-check before it goes in the paper.
- **[abstract]**: abstract only (no PDF). **[memory]**: my background knowledge, no source retrieved.
Status: **known** = published result we can cite as-is; **adaptable** = nearest result needs a translation to our setting; **gap** = not found.

Already covered elsewhere (not repeated here): PokaiTrainer main results, PokeChamp/Metamon/Foul Play numbers, Gan15/Gan25b, Li24e, Dig21, Pon80/Gil08, DeepStack/Libratus details (see `game_theory_noregret.md`, `pokemon_ai.md`, `scout_exploitation_pokemon.md`, `tightening_related_work.md`).

## 0. Answer in brief

1. **Joint vs independent beliefs.** The only explicit theorem comparing them is Albrecht-Crandall-Ramamoorthy (Alb15b): a product (independent-factor) posterior is "only guaranteed to learn independent type distributions"; a separate "correlated posterior" is needed for correlated type distributions (Thm 3). Southey and Ganzfried-Sun (Sou05, Gan16) put the posterior on the opponent *strategy* (a map from private card to action) and marginalise the unobserved card in the likelihood, so card x action correlation is built into the strategy parameter. Nobody in this set tests joint type x style posteriors on a real game.
2. **VOI.** Gmy05 states that with horizon 1 "listening does not provide any useful information"; Lon10/Fra98 give the PIMC error taxonomy (strategy fusion, non-locality) but no VOI quantity; Bro20/Sch21 say beliefs must enter the value function. No Pokemon paper models scouting.
3. **Concealment.** Sch21: "to avoid being exploited, players must act in a way that does not reveal their own private information." Cow15 and Gua26b give algorithmic/numeric concealment examples outside Pokemon. PokaiTrainer (Yu26d) explicitly *concedes* the agent's private information to the opponent (conservative; "never exploits what the opponent does not know"). Aumann-Maschler cav u is the theory anchor but I could not retrieve a PDF.

## 1. Joint posteriors over opponent private type and strategy/style

### Alb15b: Albrecht, Crandall, Ramamoorthy, "Belief and truth in hypothesised behaviours", Artificial Intelligence 235 (2016), doi:10.1016/j.artint.2016.02.004, arXiv:1507.07688 -- ADAPTABLE (HBA)
- Model: stochastic Bayesian game; "the types theta_1..theta_n are sampled from Theta+ with probability Upsilon(theta_1..theta_n), and each player i is informed only about its own type" (Sec 3.1, p.6) [PDF-read].
- HBA posterior (Sec 3.2, p.7, Eq. 1-2): under "independence of types" Pr(theta_{-i}|H^t) = prod_{j != i} Pr_j(theta_j|H^t), with Bayes update per player [PDF-read]. Action choice Eq. 3: expected payoff averaged over posterior and the hypothesised types' action probabilities, then argmax (Alg. 1, p.8) [PDF-read].
- Key quote (Sec 4.3, p.14), **[spot-checked]**: "since the sum posterior is in the form of (1), it is in fact only guaranteed to learn independent type distributions. This is opposed to correlated type distributions, which cannot be represented as a product of n independent factors." Correlated posterior defined Eq. 14; Thm 3 (p.14): with a correlated type distribution and the correlated posterior, if AO_j = AS_j = 0 (zero average overlap/stochasticity) then Pr(theta_{-i}|H^t) = Upsilon(theta_{-i}) [PDF-read].
- Thm 1 (Sec 4.1, p.11): product posterior + positive priors gives correct *future-play predictions*, but "HBA is guaranteed to make correct future predictions after some time, it is not guaranteed to learn the type distribution of the game" [PDF-read]. Assumption 1: true types are in the hypothesised set.
- Domain: Rapoport-Guyer 78 repeated 2x2 matrix games, fully observable states ("A central assumption in SBGs is that the states and chosen actions are fully observable", Sec 3.3, p.9) [PDF-read]. Not a hidden-state game.
- What it shows about joint vs marginal: product posteriors suffice for *prediction* but not for *identifying* a correlated type distribution.
- Analogue for us: our "5 styles" are HBA's hypothesised types; per-turn expected payoff averaged over the style posterior and sampled worlds is HBA Eq. 3. If our style posterior is updated independently of the sampled opponent team (product form), Alb15b's caveat applies exactly when style and team are correlated (e.g. stall teams and passive style). Cite Alb15b to justify either (a) a joint (team, style) posterior or (b) the claim that product form is enough for prediction (Thm 1).

### Alb17: Albrecht & Stone, "Autonomous agents modelling other agents: a comprehensive survey and open problems", Artificial Intelligence 258 (2018), doi:10.1016/j.artint.2018.01.002, arXiv:1709.08071 -- KNOWN (framing)
- Sec 4.2, pp.14-15 [PDF-read]: "Type-based reasoning methods assume that the modelled agent has one of several known types. Each type is a complete specification (a model) of the agent's behaviour..."; "each time a new action is observed, the belief is updated according to the probability with which the types predicted the observed action."
- Sec 4.5, p.24: "I-POMDPs modify POMDPs by adding model spaces to the environment state, such that an agent has beliefs about the environment state and the models of other agents." Particle-filter I-POMDP solvers are listed (Doshi & Gmytrasiewicz 2009).
- Sec 5.3 (p.33), the only nearby open problem: "Agents that model other agents can consider the possibility of taking actions so as to explore certain aspects of the other agents' behaviours ... such actions may carry a risk in that they may modify the behaviour of the modelled agents in unintended ways." The survey states no open problem on correlated type x strategy beliefs or on concealment (sub-agent found none; I did not full-text search it myself).
- Analogue: citation for the whole "type = style" design; Sec 5.3 for active information gathering.

### Gmy05: Gmytrasiewicz & Doshi, "A framework for sequential planning in multi-agent settings" (I-POMDP), JAIR 24 (2005) 49-79, doi:10.1613/jair.1579 -- ADAPTABLE
- Interactive state IS_i = S x M_j (Sec 5, p.57); belief b_i in Delta(S x M_j); update Prop. 2 / Eq. 6 (p.59) [PDF-read]. This is the canonical *joint* belief over physical state and the other agent's model, not a product of marginals.
- "The notion of a type we use here coincides with the notion of type in game theory, where it is defined as consisting of all of the agent i's private information relevant to its decision making" (Sec 5, p.57) [PDF-read]. So an I-POMDP type bundles private information and behaviour in one object, which is the joint object we want (team + style).
- Exact I-POMDP solving is nested and intractable; particle filters approximate it: Dos05 ("interactive PF ... samples and propagates beliefs at each level", AAAI 2005, [abstract]); Dos05b (AAMAS 2005, doi:10.1145/1082473.1082522, [abstract-level]).
- Analogue: our K sampled worlds x 5 styles is a particle approximation of an I-POMDP belief over (opponent team, style), with level-0 (non-nested) opponent models. Nesting is not needed because opponent styles are fixed policies (subintentional models in their terminology).

### Sou05: Southey et al., "Bayes' Bluff: Opponent modelling in poker", UAI 2005, doi:10.48550/arXiv.1207.1411 -- KNOWN
- Posterior is over the opponent *strategy* beta, with private cards marginalised in the likelihood of fold hands: "we do not observe P2's private cards, D. We must marginalize away this hidden variable by summing over all possible sets of cards P2 could hold" (Sec 3.3, p.4) [PDF-read]. No separate joint posterior over (D, beta) is maintained.
- Independence assumptions (Sec 3.1, p.3) [PDF-read]: "the players' strategies are independent. More formally, P(alpha, beta) = P(alpha)P(beta)" and "all hands are i.i.d. given the strategies". Leduc prior: independent Dirichlet(2,2,2) per information set (Sec 5, p.6). For Texas hold'em, "priors with independent distributions for each information set are both intractable and ineffective" (Sec 5, p.6); an expert-defined 5-parameter informed prior is used instead (Table 1).
- Responses: Bayesian Best Response, MAP, Thompson (sample one strategy from the posterior and best-respond); BBR "is erratic and takes longer to converge" than MAP/Thompson in Leduc (Sec 7.1, p.7); results only against opponents drawn from the prior, plus Opti and a frequentist baseline (figures, no numeric tables).
- What it shows: the *strategy* is the right place to hold a correlated type x action belief, because a strategy is a map card -> action. Product-over-information-sets priors are explicitly flagged as the weak assumption in a big game, which motivates a low-dimensional style parameterisation like our 5 styles.
- Analogue: Thompson response = our "sample a style, then solve", BBR = our posterior-weighted mixture. Exact match on the design choice.

### Gan16: Ganzfried & Sun, "Bayesian opponent exploitation in imperfect-information games", arXiv:1603.03491 (IEEE CIG 2018), doi:10.48550/arXiv.1603.03491 -- KNOWN
- Setting (Sec 4, p.6) [PDF-read]: "Player 1 is given private information state x_i ... Then he takes a publicly observable action ... Player 2 ... (but not his private information)." Hidden-type posterior when the card is not observed is a *mixture over cards* (Sec 4.1, p.7): p(q|O) = (q_{O|K} p(q) + q_{O|J} p(q)) / (2 p(O)); general form is a sum over types (Sec 4.3, p.10). For multiple observations "the posterior is not Dirichlet" and terms grow exponentially in the number of private states (Sec 4.3, p.11).
- Thm 2.1 / Cor 2.2 (Sec 2, pp.3-4): payoff against a strategy distribution equals payoff against its mean, so best-responding to the posterior mean is Bayes-optimal.
- Quote (Sec 4.1, p.7): "There are no known algorithms even for the simplified case of fully unobservable opponent's private information."
- Numbers (Table 3, p.16, 95% CI): EBBR -0.0004 +- 0.0009 vs BBR -0.0004 +- 0.0009 at 10 iterations, diverging to 0.0002 vs -0.0065 at 25; Nash -0.375; MAP -0.266, Thompson -0.300 at 25 (Kuhn-style K/J game).
- Analogue: Cor 2.2 gives an exactness argument for averaging over the style posterior in a single LP (our posterior-weighted objective) only if the opponent's strategy is a fixed stationary draw from the prior; our opponent is a per-turn sampled style, so the "mean strategy" argument holds per decision, not across time. Note the failure of sampled responses (MAP/Thompson) in this paper versus Sou05's good results: both are for prior-drawn opponents.

### Joh07 / Joh09: Johanson, Zinkevich, Bowling, "Computing robust counter-strategies", NIPS 2007, doi:10.7939/R35N5V; Johanson & Bowling, "Data biased robust counter strategies", AISTATS 2009 -- ADAPTABLE
- RNR (Joh07 Sec 5, p.5) [PDF-read]: opponent model is a *single fixed strategy* sigma_fix; the opponent is restricted to mix_p(sigma_fix, sigma'); "p provides a balance between exploitation and exploitability". Result: "the curves are highly concave meaning that dramatic reductions in exploitability can be achieved with only a small sacrifice in the ability to exploit the model" (Sec 5, p.6). Heads-up limit Hold'em; RNR table averages 43-78 mb/h (Table 2).
- Joh09 (abstract only): data-biased response, same family, uses limited data.
- Closest analogue: a mixture weight p between "style model" and "worst case" is the same trade-off as blending the agent-form LP with a style-conditioned best response (cf. Gan15 notes). Neither paper has a distribution over opponent types; both are single-model. Gap-ish for our joint posterior.

### Zha24h: Zhang, Imani, Lan, "Modeling other players with Bayesian beliefs for games with incomplete information", arXiv:2405.14122 -- ADAPTABLE
- Belief is over "types" theta (payoffs, strategies, preferences), "Pr(theta|O) proportional to P(theta) prod P(h_i|theta)" (Sec 3.1, Eq. 5, p.4); strategies are type-indexed (Sec 2.1, p.3); no separate joint over type and strategy [PDF-read]. Texas hold'em, types normal/conservative/aggressive. Table 1 average exploitability: BCFR+ 0.02, BCFR 0.17, CFR 0.28, CFR+ 0.07; ablation (no posterior update) 0.27 vs 0.19 with update vs 0.16 complete-information reference [PDF-read].
- Closest published analogue to "posterior over a small number of playing styles, then a CFR solve against the posterior-weighted game". Nothing about concealment or what the opponent infers about us.

### Particle filters for opponent modelling
- Bar07: Bard & Bowling, "Particle filtering for dynamic agent modelling in simplified poker", AAAI 2007 [abstract]: Kuhn poker, "deliberately hidden information", Rao-Blackwellised PF for "inferring both the opponent's state as well as a model of its dynamics" = joint (state, model) estimation. ADAPTABLE. No DOI (semantic scholar link only).
- Bak07: Baker & Cowling, "Bayesian opponent modeling in a simple poker environment", IEEE CIG 2007, doi:10.1109/CIG.2007.368088 [abstract]: four opponent *styles*, identify the style from past actions and counter it. Closest to our "5 opponent decision styles" in the poker literature. KNOWN.
- Doshi/Gmytrasiewicz PF: see Gmy05 above.

### Section 1 verdict
- Known: Bayesian posterior over fixed opponent types (Alb15b, Alb17, Bak07, Zha24h); posterior over strategy with hidden cards marginalised (Sou05, Gan16); joint beliefs over (state, model) as definition (Gmy05).
- Adaptable: Alb15b Thm 3 (correlated posterior) to a (team, style) joint; Sou05's Thompson/BBR comparison to K-sampled-worlds.
- **Gap**: no paper measures the loss from a *product* posterior vs a *joint* posterior of (private type, style) in a real imperfect-information game, nor a regret/exploitability bound for a sampled-world (PIMC-like) approximation of the type x style belief. Our K-world sampling with 5 styles plus an ablation (joint vs product) would be new.

## 2. Value of information and active information gathering

### Gmy05 (again) -- KNOWN (it states our depth-1 observation)
- Sec 3.2, p.53 [PDF-read]: "when the time horizon is equal to 1, listening does not provide any useful information since the game does not continue to allow for the use of this information. For longer time horizons the benefits of results of listening results in policies which are better in some ranges of initial belief."
- Sec 7.2, p.66: "The more informative these estimates are the more value agent i can expect out of the interaction."
- Analogue: direct citation for "VOI is zero at depth 1 with a scalar leaf" (cf. `game_theory_noregret.md` item 6, which found no source). This is the tiger-problem remark, single-agent.

### Lau16: Lauri et al., "Myopic policy bounds for information acquisition POMDPs", arXiv:1601.07279 -- ADAPTABLE
- Belief-dependent reward rho(b,a) = sum_s r(s,a) b(s) - w_a f(b) (Eq. 4, p.3) [PDF-read]; myopic policies bound the optimal infinite-horizon policy: mu*_1 <= mu* <= mu-bar*_1 under TP2/copositivity/observation-ordering assumptions (Sec IV, p.4); the bound does not hold for finite horizon ("the preceding inequalities do not hold anymore for finite horizon optimal policies"). Not submodularity.
- Analogue: justification (single-agent) for when one-step lookahead on information is enough; our problem is a two-player game so the structure does not transfer directly.
- Vei23 Veiga & Renoux, ACM Comput. Surv. 2023, doi:10.1145/3583068 [abstract-level, no PDF]: survey of active sensing / information gathering in decision-theoretic planning. Cite as general VOI-in-POMDP survey. Not read.

### Lon10 and Fra98 (PIMC error theory) -- KNOWN
- Lon10 (AAAI 2010, doi:10.1609/aaai.v24i1.7562, pp.134-140) [PDF-read]. Strategy fusion (p.135): "PIMC search (incorrectly) believes it can use a different strategy in each world, whereas in reality there are situations (or information sets) which consist of multiple perfect information scenarios." Non-locality (p.135): "a node's value may depend on other regions of the game tree not contained within its subtree, primarily due to the opponent's ability to direct the play towards regions of the tree that he knows ... using private information that he possesses but we do not."
- Synthetic-tree properties (p.137): leaf correlation lc, bias b, disambiguation factor df; "PIMC search is at its worst when leaf node correlation is low" (p.138); Skat/Hearts measure lc 0.8-1.0, df ~0.6 (p.139); Kuhn poker (df, lc, b) = (0, 0.5, 0.5), predicted poor (p.139). Limitation (p.140): "the performance of PIMC search could be substantially worse against a player that attempts to exploit its mistakes"; "If an opponent is able to build a best-response against a PIMC player, then the PIMC player is vulnerable to significant exploitation as player 2". No value-of-information quantity is defined; df = 1 is the perfect-information limit.
- Fra98 (Artif. Intell. 100, 1998, doi:10.1016/S0004-3702(97)00082-9) [abstract]: formalises the best-defence model, shows equilibrium strategies exist for it, and "precisely formalise[s] and distinguish[es] the problems that lead to sub-optimality" of analysing sampled complete-information sub-games.
- Analogue: our PIMC baseline is exactly this; agent-form LP removes strategy fusion on the agent side. Lon10's real-game parameter measurements are for card games only; none for Pokemon (see also Iha18).

### ISMCTS and information-set search -- KNOWN / ADAPTABLE
- Cowling, Powley, Whitehouse, "Information Set Monte Carlo Tree Search", IEEE TCIAIG 4(2):120-143, 2012, doi:10.1109/TCIAIG.2012.2200894 **[memory; not found in Undermind]**. Whi11 (Dou Di Zhu, doi:10.1109/CIG.2011.6031993) and Pow14 (Artif. Intell. 2014, "Information capture and reuse strategies...") are the located follow-ups [abstract-level].
- Iha18 (Ihara et al., IEEE SMC 2018, ISMCTS for Pokemon) [PDF-read]: "ISMCTS gathered multiple states in one information set regardless of the result of determinization in the first player node" (Sec VI.B, p.7); Table I (200 games per cell): ISMCTS vs DMCTS 57.5%; but perfect-information variants CMCTS/CEMCTS beat ISMCTS (ISMCTS wins 25.0/26.5%); Table II: ISMCTS average rises 36.3 -> 40.7 as the opponent pool shrinks "3 from 6" -> "3 from 3". Opponent parameters from Pokemon Global Link top-10 statistics. The paper discusses no scouting and no concealment (Sec VII only lists team selection as future work).

### Public belief states -- KNOWN
- Bro20 (ReBeL), arXiv:2007.13544 [PDF-read]. "the value of an action may depend on the probability it is chosen. Thus, a state defined only by the sequence of actions and observations does not have a unique value" (Sec 1, p.2). PBS = "a joint probability distribution over the agents' possible infostates" (Sec 4, p.4); "every PBS beta has a unique value V_i(beta) ... defined by both players playing a Nash equilibrium in the subgame rooted at the PBS" (Sec 4, p.5). No text on strategy fusion or "value of information" per se (sub-agent found none).
- Sch21 (Student of Games), Science Advances 9, eadg3256 (2023), doi:10.1126/sciadv.adg3256 [PDF-read]. PBS beta = (s_pub, r) (p.2); counterfactual value network input is the PBS (p.4); on determinization: "these methods are not guaranteed to converge to an optimal strategy over time" (Related work, p.5). Poker: +7 +- 3 mbb/hand vs Slumbot; 434 +- 9 vs a fold/call LBR; Scotland Yard 55% vs PimBot at 10M simulations (Table 2 p.8, Fig. 5 p.9).
- Mor17 (DeepStack, Science 2017, doi:10.1126/science.aam6960) [located, not re-read].
- Yu26d (PokaiTrainer) uses a PBS over the opponent's bring-subset worlds with a *per-world* value vector from a network: "A public belief state beta = (s, b) pairs s with the solver's posterior b in Delta(W(s))" (Sec 4.1, p.5); node value V = <b, v_theta> (Eq. 1, Sec 4.2) **[spot-checked, p.~5-6]**. Hidden stat spreads are held "not as extra worlds but inside every world as a mixture over a candidate set" and "assumed independent across Pokemon, so a world's spread belief is the product prod_j q_j" (App. C, p.17) **[spot-checked, line 1081 of the text dump]**. This is exactly an *independent-marginals* choice inside a joint world distribution. The +8-12 pp quoted for the hidden-stat belief summary is an encoder ablation (phase-only vs v7, against round-matched D1SW, 512-game cells, App. E.2, p.21) **[spot-checked, text lines 523 and 1389]**, not a measurement of VOI.

### Pokemon scouting / information gathering -- GAP
- No Pokemon paper treats scouting as an action. Yu26d (sub-agent): "I found no discussion of dedicated scouting or information-gathering actions" and inference is passive, "inference from observed battle outcomes and move order" (Sec 4.4, p.6). Kar25b: hidden info is "gradually revealed through gameplay"; opponent modelling = usage-statistics likelihood for A/D stats plus LLM action prediction, "does not describe this as Bayesian inference"; no scouting (Sec 3 p.6, Sec 4.1 pp.8-9) [PDF-read]. Gri25: inference is implicit in the sequence model, "relying entirely on memory to infer the opponent's team" (Sec 4, p.6) [PDF-read]. Iha18: none.

### Section 2 verdict
- Known: PIMC failure modes (Lon10, Fra98), PBS beliefs in values (Bro20, Sch21, Mor17), ISMCTS (Cowling 2012 [memory], Whi11, Pow14, Iha18), horizon-1 VOI remark (Gmy05).
- Adaptable: Lau16 / Vei23 for single-agent VOI; Alb17 Sec 5.3 for exploration risk.
- **Gap**: (i) no measurement of VOI (oracle-vs-belief gap decomposed into strategy fusion, non-locality and belief error) in a real game; (ii) no game-theoretic (two-player) VOI in Pokemon; (iii) no scouting. Our oracle gap (~0.1) and eta_NL measurements are the new facts.

## 3. Concealment, signalling, deception

### Sch21 (statement of the principle) -- KNOWN
- "Hence, to avoid being exploited, players must act in a way that does not reveal their own private information." (Game-theoretic reasoning and CFR, p.3) and "past actions can leak otherwise private information, agents must reason about which information states players could be in to act soundly." (Background, p.2) [PDF-read]. Does not use the phrase "keep ranges balanced".

### Yu26d: what PokaiTrainer does about concealment -- KNOWN (important for novelty claims)
- **[spot-checked]**, Sec 4.1: "Information is one-sided: seat 1's configuration is treated as known to both players, conceding the opponent our hidden spreads and reserves. Conceding information makes the solved game harder for us, never easier, so sigma_1 is conservative, though it never exploits what the opponent does not know." So the agent does not model its own concealment; it plays a worst-case-informed-opponent game. Sec 2.1 (p.2): "OTS still leaves three sources of hidden information: the stat spreads ...; which two of the four non-leads were brought; and the opponent's live commitment on every simultaneous turn" and "Deterministic play is exploitable, so optimal play requires mixing" [PDF-read]. Tera is not modelled (only "a once-per-battle gimmick (Mega Evolution in Reg-MB)").
- Consequence for us: the one-sided hidden-info structure (agent's private info is public to the opponent) removes the concealment term entirely; a version where the opponent holds a belief about *our* team and updates from our actions would be new.

### Cow15: Cowling, Whitehouse, Powley, "Emergent bluffing and inference with Monte Carlo Tree Search", IEEE CIG 2015, doi:10.1109/CIG.2015.7317927 -- ADAPTABLE [abstract]
- "in its vanilla form [MCTS] is unable to infer or bluff"; bluffing is obtained "by self-determinization (introducing 'impossible' worlds into the agent's pool of sampled states)"; inference via the search tree as an approximate opponent model. Tested on The Resistance. Closest published *sampling-based* concealment mechanism: our K sampled worlds could include self-determinized "what the opponent believes our hidden set is" worlds.

### Gua26b: Guan, Maity, Tsiotras, "Strategic concealment of environment representations in competitive games", ACC 2026, arXiv:2509.05503 -- ADAPTABLE (no Pokemon)
- Attacker has a private type (environment representation); the Defender "seeks to infer it based on the Attacker's behavior"; "The Attacker faces a strategic trade-off between path optimality and the concealment of its representation type." (Sec 2 p.2, Sec 4 p.6); Perfect Bayesian Stackelberg equilibrium, solved as a bilinear program (Eq. 17, Sec 4.5, p.11) [PDF-read].
- Numbers (Sec 5, pp.12-13): full-information expected performance -0.978 vs -0.841 with strategic concealment, "14% improvement"; 30.5% excluding the constant Phase I payoff. Single hand-built example, no general theorem on the value of concealment ("There is no general theorem ... giving a numerical value or cost of concealment" per sub-agent). Posterior [0.264, 0.736] on the red path.
- Analogue: a quantified value-of-concealment in a small game with a Bayesian-updating opponent; the structure (private type, public trajectory, belief update) matches the tera/hidden-set problem.

### Classical theory -- KNOWN (theory), not retrievable here
- Sor96: Sorin's review of Aumann & Maschler, *Repeated Games with Incomplete Information* (MIT Press, 1995, with Stearns), Games Econ. Behav. 1996, doi:10.1006/GAME.1996.0090 [located; no PDF]. **[memory, unverified]**: one-sided incomplete information; informed player's non-revealing (stage-game) strategies; value of the infinitely repeated game = cav u(p), the concavification of the value u(p) of the non-revealing average game, so the informed player gains from revealing only where u is not already concave, and the uninformed player's gain from information is zero in the zero-sum case only on concave regions. Nothing in this file verifies the statement; cite the book directly after checking.
- Shm06 (Shmaya, Int. J. Game Theory 2006, doi:10.1007/s00182-006-0018-5) and Gos01 (Gossner & Mertens, "The value of information in zero-sum games", unpublished, [abstract]): value of information structures in zero-sum games. Gos01 abstract: negative value of information (Hirshleifer 1971) arises in non-zero-sum games; for one agent "more information is always better" (Blackwell); the zero-sum class is the one that has Blackwell-type results.
- Cho87 (Cho & Kreps, Quart. J. Econ. 102:179-221, 1987, doi:10.2307/1885060) [abstract]: signalling games, "the second party may entertain a wealth of beliefs in response to out-of-equilibrium messages"; relevant if a non-revealing action is off-path. Oku90 (Okuno-Fujiwara, Postlewaite, Suzumura, Rev. Econ. Stud. 57:25-47, 1990, doi:10.2307/2297541) [abstract]: voluntary disclosure of private information with certifiable statements.
- Poker-specific concealment: no paper in this search *quantifies* the value of concealment in Kuhn/Leduc equilibria (Lon10 treats Kuhn only as a PIMC test and states best-response exploitation of PIMC as player 2). **Gap** (I did not run a dedicated Kuhn/Leduc query beyond the two semantic searches).

### Pokemon concealment/deception -- mostly GAP
- Gri25 (Metamon), Sec 2, p.3 [PDF-read]: "We may try to mislead our opponent by revealing information that suggests one team design only to surprise them later in the battle." No modelling of it; the agent is a history-conditioned policy.
- Kar25b (PokeChamp), Sec 6, p.15 [PDF-read]: "our work can be taken advantage adversarially due to static opponent modeling." Not a concealment result.
- Foul Play (Mariglia blog/GitHub) and Tera discussion: see `pokemon_ai.md` rows (set filtering, DUCT sampling) -- not re-read here; no concealment in any of the notes there.
- PokaiTrainer: see above (concedes information).
- Cow15 and Gua26b are the nearest non-Pokemon algorithmic concealment results.

### Section 3 verdict
- Known: principle (Sch21), theory (Aumann-Maschler [memory]), signalling formalism (Cho87), one numerical concealment example (Gua26b), MCTS bluffing hack (Cow15).
- Adaptable: Cow15 self-determinization; Gua26b PBSE.
- **Gap**: no Pokemon agent models what the opponent infers about *our* hidden set/tera type, and no measured value of concealment in a real game's exact subgame. Our one-sided exact tera subgame (eta_NL, VOI_total in `tightening_related_work.md` sec 6) would be the first measurement.

## 4. Table: papers

| Key | Citation | Topic | Status | Why |
|---|---|---|---|---|
| Alb15b | Albrecht, Crandall, Ramamoorthy, AIJ 2016, 10.1016/j.artint.2016.02.004 | 1 | adaptable | HBA; product vs correlated posterior (Thm 1 vs Thm 3) |
| Alb17 | Albrecht & Stone, AIJ 2018, 10.1016/j.artint.2018.01.002 | 1,2 | known | type-based reasoning survey; model exploration risk |
| Gmy05 | Gmytrasiewicz & Doshi, JAIR 2005, 10.1613/jair.1579 | 1,2 | adaptable | joint state x model belief; horizon-1 VOI remark |
| Dos05, Dos05b | Doshi & Gmytrasiewicz, AAAI 2005; AAMAS 2005, 10.1145/1082473.1082522 | 1 | adaptable | interactive particle filter |
| Sou05 | Southey et al., UAI 2005, 10.48550/arXiv.1207.1411 | 1 | known | Bayesian strategy posterior, hidden-card marginalisation, BBR/MAP/Thompson |
| Gan16 | Ganzfried & Sun, arXiv 1603.03491 | 1 | known | posterior mixture over hidden types; posterior-mean best response |
| Joh07, Joh09 | Johanson et al. NIPS 2007 (10.7939/R35N5V); AISTATS 2009 | 1 | adaptable | RNR, data-biased response (single model) |
| Zha24h | Zhang, Imani, Lan, arXiv 2405.14122 | 1 | adaptable | Bayesian-CFR over style types in Hold'em |
| Bar07 | Bard & Bowling, AAAI 2007 | 1 | adaptable | PF joint state/model in Kuhn |
| Bak07 | Baker & Cowling, CIG 2007, 10.1109/CIG.2007.368088 | 1 | known | four opponent styles in simple poker |
| Lon10 | Long et al., AAAI 2010, 10.1609/aaai.v24i1.7562 | 2 | known | strategy fusion, non-locality, lc/b/df |
| Fra98 | Frank & Basin, AIJ 1998, 10.1016/S0004-3702(97)00082-9 | 2 | known | origin of PIMC error analysis |
| Whi11, Pow14, Iha18 | ISMCTS lineage; Iha18 = ISMCTS for Pokemon | 2 | known | information-set search |
| Cow12 | Cowling, Powley, Whitehouse, TCIAIG 2012, 10.1109/TCIAIG.2012.2200894 [memory] | 2 | known | ISMCTS |
| Bro20, Sch21, Mor17 | ReBeL; Student of Games; DeepStack | 2,3 | known | PBS value functions; Sch21 concealment principle |
| Lau16, Vei23 | Lauri et al. 1601.07279; Veiga & Renoux 2023 | 2 | adaptable | VOI/myopic bounds, active sensing survey |
| Yu26d | Yu, PokaiTrainer, arXiv:2608.29197 | 2,3 | known | PBS over worlds; concedes our info; no scouting |
| Gri25, Kar25b | Metamon; PokeChamp | 2,3 | known | implicit memory; static opponent model; mislead remark |
| Cow15 | Cowling et al., CIG 2015, 10.1109/CIG.2015.7317927 | 3 | adaptable | MCTS bluffing via self-determinization |
| Gua26b | Guan et al., ACC 2026, arXiv 2509.05503 | 3 | adaptable | quantified concealment, PBSE |
| Sor96, Gos01, Shm06, Cho87, Oku90 | Aumann-Maschler review; value of info; signalling | 3 | known (theory), unverified | cav u, signalling |

## 5. Gaps and what a deeper search would target
- Aumann-Maschler cav u and the one-sided repeated game statements: no PDF; cite the book/Mertens-Sorin-Zamir and verify the statement. Search target: a modern survey with a clear statement (e.g. Zamir chapter in the Handbook of Game Theory).
- Cowling et al. 2012 ISMCTS: not found in Undermind; DOI is from memory.
- Kuhn/Leduc "value of concealment" and "value of private information" in poker equilibria: not located; target Hirshleifer-type / "information value in extensive-form games" papers and the Ponssard-Sorin LP line (Pon80, already in notes).
- Particle-filter opponent models in Pokemon or similar simultaneous-move games with hidden teams: none found.
- Foul-play posts and Tera concealment: not re-read in this pass.
- Not spot-checked by me: all quotes from Alb17, Sou05, Gan16, Bro20, Lon10, Joh07, Gmy05, Zha24h, Gri25, Kar25b, Sch21, Gua26b, Lau16, Iha18 (these are sub-agent PDF reads). Spot-checked: Alb15b Sec 4.3 p.14 quote; Yu26d Sec 4.1 info-concession quote, App. C independence quote, +8-12 pp ablation.

## 6. BibTeX (fields from Undermind metadata; fill author lists where marked "and others")

```bibtex
@article{Alb15b, author={Albrecht, Stefano V. and Crandall, Jacob W. and Ramamoorthy, Subramanian}, title={Belief and Truth in Hypothesised Behaviours}, journal={Artificial Intelligence}, volume={235}, year={2016}, doi={10.1016/j.artint.2016.02.004}, note={arXiv:1507.07688}}
@article{Alb17, author={Albrecht, Stefano V. and Stone, Peter}, title={Autonomous Agents Modelling Other Agents: A Comprehensive Survey and Open Problems}, journal={Artificial Intelligence}, volume={258}, year={2018}, doi={10.1016/j.artint.2018.01.002}}
@article{Gmy05, author={Gmytrasiewicz, Piotr J. and Doshi, Prashant}, title={A Framework for Sequential Planning in Multi-Agent Settings}, journal={Journal of Artificial Intelligence Research}, volume={24}, pages={49--79}, year={2005}, doi={10.1613/jair.1579}}
@inproceedings{Dos05, author={Doshi, Prashant and Gmytrasiewicz, Piotr J.}, title={A Particle Filtering Based Approach to Approximating Interactive POMDPs}, booktitle={AAAI}, pages={969--974}, year={2005}}
@inproceedings{Dos05b, author={Doshi, Prashant and Gmytrasiewicz, Piotr J.}, title={Approximating State Estimation in Multiagent Settings Using Particle Filters}, booktitle={AAMAS}, pages={320--327}, year={2005}, doi={10.1145/1082473.1082522}}
@inproceedings{Sou05, author={Southey, Finnegan and others}, title={Bayes' Bluff: Opponent Modelling in Poker}, booktitle={UAI}, year={2005}, doi={10.48550/arXiv.1207.1411}}
@article{Gan16, author={Ganzfried, Sam and Sun, Qingyun}, title={Bayesian Opponent Exploitation in Imperfect-Information Games}, journal={arXiv:1603.03491}, year={2016}, doi={10.48550/arXiv.1603.03491}}
@inproceedings{Joh07, author={Johanson, Michael and Zinkevich, Martin and Bowling, Michael}, title={Computing Robust Counter-Strategies}, booktitle={NeurIPS}, pages={721--728}, year={2007}, doi={10.7939/R35N5V}}
@inproceedings{Joh09, author={Johanson, Michael and Bowling, Michael}, title={Data Biased Robust Counter Strategies}, booktitle={AISTATS}, pages={264--271}, year={2009}}
@article{Zha24h, author={Zhang, Zuyuan and Imani, Mahdi and Lan, Tian}, title={Modeling Other Players with Bayesian Beliefs for Games with Incomplete Information}, journal={arXiv:2405.14122}, year={2024}, doi={10.48550/arXiv.2405.14122}}
@inproceedings{Bar07, author={Bard, Nolan and Bowling, Michael}, title={Particle Filtering for Dynamic Agent Modelling in Simplified Poker}, booktitle={AAAI}, pages={515--521}, year={2007}}
@inproceedings{Bak07, author={Baker, Roderick J. S. and Cowling, Peter I.}, title={Bayesian Opponent Modeling in a Simple Poker Environment}, booktitle={IEEE Symposium on Computational Intelligence and Games}, pages={125--131}, year={2007}, doi={10.1109/CIG.2007.368088}}
@inproceedings{Lon10, author={Long, Jeffrey and Sturtevant, Nathan R. and Buro, Michael and Furtak, Timothy}, title={Understanding the Success of Perfect Information Monte Carlo Sampling in Game Tree Search}, booktitle={AAAI}, pages={134--140}, year={2010}, doi={10.1609/aaai.v24i1.7562}}
@article{Fra98, author={Frank, Ian and Basin, David}, title={Search in Games with Incomplete Information: A Case Study Using Bridge Card Play}, journal={Artificial Intelligence}, volume={100}, year={1998}, doi={10.1016/S0004-3702(97)00082-9}}
@article{Cow12, author={Cowling, Peter I. and Powley, Edward J. and Whitehouse, Daniel}, title={Information Set Monte Carlo Tree Search}, journal={IEEE Transactions on Computational Intelligence and AI in Games}, volume={4}, number={2}, pages={120--143}, year={2012}, doi={10.1109/TCIAIG.2012.2200894}, note={unverified: not located in Undermind}}
@inproceedings{Whi11, author={Whitehouse, Daniel and Powley, Edward J. and Cowling, Peter I.}, title={Determinization and Information Set Monte Carlo Tree Search for the Card Game Dou Di Zhu}, booktitle={IEEE CIG}, pages={87--94}, year={2011}, doi={10.1109/CIG.2011.6031993}}
@inproceedings{Iha18, author={Ihara, Hiroyuki and Imai, Shunsuke and Oyama, Satoshi and Kurihara, Masahito}, title={Implementation and Evaluation of Information Set Monte Carlo Tree Search for Pok{\'e}mon}, booktitle={IEEE SMC}, pages={2182--2187}, year={2018}}
@inproceedings{Bro20, author={Brown, Noam and Bakhtin, Anton and Lerer, Adam and Gong, Qucheng}, title={Combining Deep Reinforcement Learning and Search for Imperfect-Information Games}, booktitle={NeurIPS}, year={2020}, doi={10.48550/arXiv.2007.13544}}
@article{Sch21, author={Schmid, Martin and others}, title={Student of Games: A Unified Learning Algorithm for Both Perfect and Imperfect Information Games}, journal={Science Advances}, volume={9}, pages={eadg3256}, year={2023}, doi={10.1126/sciadv.adg3256}}
@article{Mor17, author={Morav{\v c}{\'\i}k, Matej and others}, title={DeepStack: Expert-level Artificial Intelligence in Heads-up No-limit Poker}, journal={Science}, volume={356}, pages={508--513}, year={2017}, doi={10.1126/science.aam6960}}
@misc{Lau16, author={Lauri, Mikko and others}, title={Myopic Policy Bounds for Information Acquisition POMDPs}, year={2016}, eprint={1601.07279}, archivePrefix={arXiv}}
@article{Vei23, author={Veiga, Tiago and Renoux, Jennifer}, title={From Reactive to Active Sensing: A Survey on Information Gathering in Decision-theoretic Planning}, journal={ACM Computing Surveys}, year={2023}, doi={10.1145/3583068}}
@misc{Yu26d, author={Yu, Max}, title={PokaiTrainer: Scaling Equilibrium Search to Competitive Pok{\'e}mon VGC}, year={2026}, eprint={2608.29197}, archivePrefix={arXiv}}
@misc{Gri25, author={Grigsby, Jake and others}, title={Human-Level Competitive Pok{\'e}mon via Scalable Offline Reinforcement Learning with Transformers}, year={2025}, eprint={2504.04395}, archivePrefix={arXiv}, doi={10.48550/arXiv.2504.04395}}
@misc{Kar25b, author={Karten, Seth and Nguyen, Andy and Jin, Chi}, title={Pok{\'e}Champ: an Expert-level Minimax Language Agent}, year={2025}, eprint={2503.04094}, archivePrefix={arXiv}, doi={10.48550/arXiv.2503.04094}}
@inproceedings{Cow15, author={Cowling, Peter I. and Whitehouse, Daniel and Powley, Edward J.}, title={Emergent Bluffing and Inference with Monte Carlo Tree Search}, booktitle={IEEE CIG}, pages={114--121}, year={2015}, doi={10.1109/CIG.2015.7317927}}
@inproceedings{Gua26b, author={Guan, Yue and Maity, Dipankar and Tsiotras, Panagiotis}, title={Strategic Concealment of Environment Representations in Competitive Games}, booktitle={American Control Conference}, year={2026}, note={arXiv:2509.05503}}
@article{Cho87, author={Cho, In-Koo and Kreps, David M.}, title={Signaling Games and Stable Equilibria}, journal={Quarterly Journal of Economics}, volume={102}, pages={179--221}, year={1987}, doi={10.2307/1885060}}
@article{Oku90, author={Okuno-Fujiwara, Masahiro and Postlewaite, Andrew and Suzumura, Kotaro}, title={Strategic Information Revelation}, journal={Review of Economic Studies}, volume={57}, pages={25--47}, year={1990}, doi={10.2307/2297541}}
@article{Shm06, author={Shmaya, Eran}, title={The Value of Information Structures in Zero-sum Games with Lack of Information on One Side}, journal={International Journal of Game Theory}, volume={34}, pages={155--165}, year={2006}, doi={10.1007/s00182-006-0018-5}}
@article{Sor96, author={Sorin, Sylvain}, title={Review of Aumann and Maschler, Repeated Games with Incomplete Information}, journal={Games and Economic Behavior}, pages={347--352}, year={1996}, doi={10.1006/GAME.1996.0090}}
@book{AM95, author={Aumann, Robert J. and Maschler, Michael B.}, title={Repeated Games with Incomplete Information}, publisher={MIT Press}, year={1995}, note={with the collaboration of Richard E. Stearns; unverified: not retrieved}}
@unpublished{Gos01, author={Gossner, Olivier and Mertens, Jean-Fran{\c c}ois}, title={The Value of Information in Zero-Sum Games}, year={2001}, note={manuscript}}
```
