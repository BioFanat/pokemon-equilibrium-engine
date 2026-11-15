# No-regret / online learning vs RL, and as a tool for RL: literature scout for the CS535 Pokemon report

Date: 2026-11-15. Scope: complements (does not repeat) `lit/game_theory_noregret.md`, `notes/novelty_and_readings.md`, `notes/prior_art_pokemon.md`. Items already covered there (CFR+, DCFR, Lee21 last-iterate, Cai24, MMD quote, Arora12/18 as "abstract, not read", PokaiTrainer, Metamon, PokeChamp, foul-play, Wang 2024 thesis, LBR "not located") are only cross-referenced; here I add verified IDs and the RL side.

Verification method and tags:
- **[arXiv-meta]**: title, authors, date and abstract fetched from the arxiv.org/abs page on 2026-11-15 (meta tags). Quotes tagged this way are verbatim abstract text. Not full-text.
- **[PDF-spot]**: I downloaded the PDF and grepped it myself (only Rudolph25 and nothing else).
- **[Crossref]**: DOI confirmed by a Crossref bibliographic query (title + first author + year match).
- **[web]**: confirmed by web search result snippets only (POLITEX, Perolat15, Lisy13).
- **[background]**: my own knowledge, NOT checked this session. Treat as lead. Venue/year for papers without a [Crossref]/[web] tag are background unless stated.
No Explore agents were used: the load-bearing facts are abstracts/IDs, which I fetched directly. Undermind was loaded (orientation done) but arXiv/Crossref lookups were faster and more exact for known titles; no workspace was written to.

Correction to the brief: the paper you remember as "PPO is a strong baseline in two-player zero-sum games" is, I believe, **Rudolph et al., "Reevaluating Policy Gradient Methods for Imperfect-Information Games", arXiv 2502.08938 (Feb 2025)**. I found no paper with the first title; it is about imperfect-information (not perfect-information) zero-sum games. Venue not verified.

---

## 0. One-paragraph map of the relationship (for orientation)

No-regret learning and RL are two views of sequential decision-making: in RL, a Markov environment is fixed and the goal is to maximise return (regret vs the best policy, sample complexity); in online learning, losses are adversarial and the goal is low regret vs a comparator class. The bridges are (i) policy optimisation = FTRL/mirror descent per state (NPG = Hedge per state), (ii) self-play in zero-sum games, where average no-regret play gives Nash, (iii) imitation/exploration/model-selection as online-learning reductions. Our project sits at (ii) per turn (RM+ on a stage matrix), at (i)-style error propagation across turns (the H-bar bound), and at "experts over policies" (Hedge over MCTS/equilibrium across games). RL appears only as (a) the source of leaf values and (b) the best-response oracle used to measure exploitability.

---

## A. Games setting

### A1. No-regret dynamics vs RL for solving games (CFR/CFR+/DCFR/RM+ vs PG/PPO self-play; last vs average iterate)

| key | paper | ID | what it shows |
|---|---|---|---|
| Rudolph25 | Rudolph, Lichtle, Mohammadpour, Bayen, Kolter, Mei(?). Reevaluating Policy Gradient Methods for Imperfect-Information Games. arXiv 2025 | arXiv 2502.08938 [arXiv-meta]. DOI not checked | PPO-type PG matches or beats FP/DO/CFR-based deep RL on exact exploitability |
| Sokota23 | Sokota, D'Orazio, Kolter, Loizou, Lanctot, Mitliagkas, Brown, Kroer. A Unified Approach to RL, Quantal Response Equilibria, and Two-Player Zero-Sum Games (magnetic mirror descent, MMD). ICLR 2023 [background venue] | arXiv 2206.05825 [arXiv-meta] | last-iterate convergent regularised mirror-descent; competitive with CFR in tabular games |
| Hennes20 | Hennes, Morrill, Omidshafiei, Munos, Perolat, et al. Neural Replicator Dynamics (NeuRD). AAMAS 2020 [background venue] | arXiv 1906.00190 [arXiv-meta] | one-line change to PG so that it reduces to Hedge; equivalence to softmax-CFR |
| Perolat21 | Perolat, Munos, Lespiau, Omidshafiei, Rowland, et al. From Poincare Recurrence to Convergence in Imperfect Information Games: Finding Equilibrium via Regularization. ICML 2021 [background venue] | arXiv 2002.08456 [arXiv-meta] | FTRL cycles (Poincare recurrence); reward regularisation gives last-iterate convergence (R-NaD) |
| Perolat22 | Perolat, de Vylder, Hennes, et al. Mastering the game of Stratego with model-free multiagent RL (DeepNash). Science 378, 2022 | arXiv 2206.15378 [arXiv-meta]; DOI 10.1126/science.add4679 [Crossref] | R-NaD scaled with deep RL, no search |
| Bro19 | Brown, Sandholm. Solving Imperfect-Information Games via Discounted Regret Minimization (DCFR). AAAI 2019 [background venue] | arXiv 1809.04040 [arXiv-meta] | CFR variants that discount old regrets; "optimistic regret matching" |
| Lee21 | Lee, Kroer, Luo. Last-iterate Convergence in Extensive-Form Games. NeurIPS 2021 [background venue] | arXiv 2106.14326 [arXiv-meta] | "most regret-based algorithms, including CFR and its variants, rely on iterate averaging to achieve convergence" |
| Cai24 | Cai, Farina, Grand-Clement, Kroer, Lee. Fast Last-Iterate Convergence of Learning in Games Requires Forgetful Algorithms. NeurIPS 2024 [background venue] | arXiv 2406.10631 [arXiv-meta] | OMWU/optimistic FTRL can have constant gap after 1/delta rounds |
| Fari21 | Farina, Kroer, Sandholm. Faster Game Solving via Predictive Blackwell Approachability: Connecting Regret Matching and Mirror Descent. AAAI 2021 [background venue] | arXiv 2007.14358 [arXiv-meta] | RM+ is a Blackwell-approachability/online-learning method; predictive versions |
| Dask21 | Daskalakis, Foster, Golowich. Independent Policy Gradient Methods for Competitive RL. NeurIPS 2020 [background venue] | arXiv 2101.04233 [arXiv-meta] | PG with two-timescale converges to Nash in zero-sum stochastic games (tabular) |

Quoted evidence:
- Rudolph25 abstract [arXiv-meta]: "we hypothesize that simpler generic policy gradient methods like PPO are competitive with or superior to these FP-, DO-, and CFR-based DRL approaches ... Over 7000 training runs, we find that FP-, DO-, and CFR-based approaches fail to outperform generic policy gradient methods."
- Rudolph25 intro [PDF-spot, 2502.08938 text]: "none of FP, DO and CFR enjoy last-iterate convergence, creating a layer of indirection to extract the desired policy." Also: "FP and DO require expensive best response computations at each iteration (i.e., each iteration requires solving an entire reinforcement learning problem)". Their comparison: "NFSP, PSRO, ESCHER, and R-NaD fail to outperform the generic PG methods (MMD, PPO, PPG)"; exact exploitability computed "under 2 minutes on commodity hardware". Important: the comparison is of deep-RL variants (function approximation, model-free, sampled trajectories), not of tabular CFR+/RM+ with a known model.
- Sokota23 abstract [arXiv-meta]: "the first standard reinforcement learning algorithm to achieve empirically competitive results with CFR in tabular settings" and "first quantal response equilibria solver to achieve linear convergence for extensive-form games with first order feedback". (Your existing note quotes the PDF: "While CFR+ yields the strongest performance ..." Sec 4.)
- Hennes20 abstract [arXiv-meta]: "NeuRD reduces to the exponential weights/Hedge algorithm in the single-state all-actions case" and has "formal equivalence to softmax counterfactual regret minimization".
- Perolat21 abstract [arXiv-meta]: "We generalize existing results of Poincare recurrence from normal-form games to zero-sum two-player imperfect information games ... adapting the reward (by adding a regularization term) of the game can give strong convergence guarantees".
- Lee21 [arXiv-meta] as above. Cai24 [arXiv-meta]: "for any arbitrarily small delta>0, there exists a 2x2 matrix game such that the algorithm admits a constant duality gap even after 1/delta rounds."
- Existing notes already record (Cai23, Far23): RM+ last-iterate works only with strict NE; in practice RM+ average iterate is what is used.

How we position: Our stage solver is the classical no-regret route (RM+ on a small matrix, average iterate, model known via simulator) which is the regime where Rudolph25's negative result for CFR-style deep RL does not apply (they test model-free function-approximation versions; with exact tiny matrices CFR+/RM+ remain the stronger tool, consistent with the Sokota23 note that CFR+ is strongest tabularly). Our "RL BR fails at 100k games" is consistent with Rudolph25's remark that BR-oracle methods need "an entire reinforcement learning problem" per iteration; but note PPO is also the self-play baseline recommended there, so frame our PPO result as a BR (single-agent vs fixed policy), not as self-play. Last-iterate vs average-iterate: the per-turn solver outputs the average (or must justify using it); Lee21/Cai24 say last-iterate of regret-based methods is the weak point, so mention the avg-iterate choice explicitly.

### A2. Search + no-regret + learned values; the leaf-value bottleneck

| key | paper | ID |
|---|---|---|
| Mor17 | Moravcik et al. DeepStack: Expert-level AI in heads-up no-limit poker. Science 356, 2017 | arXiv 1701.01724 [arXiv-meta]; DOI 10.1126/science.aam6960 [Crossref] |
| Bro17 | Brown, Sandholm. Safe and Nested Subgame Solving for Imperfect-Information Games. NeurIPS 2017 | arXiv 1705.02955 [arXiv-meta] |
| Bro18 | Brown, Sandholm. Superhuman AI for heads-up no-limit poker: Libratus. Science 359, 2018 | DOI 10.1126/science.aao1733 [Crossref] |
| Bro18d | Brown, Sandholm, Amos. Depth-Limited Solving for Imperfect-Information Games. NeurIPS 2018 | arXiv 1805.08195 [arXiv-meta] |
| Bro19b | Brown, Sandholm. Superhuman AI for multiplayer poker (Pluribus). Science 365, 2019 | DOI 10.1126/science.aay2400 [Crossref] |
| Kov23 | Kovarik, Seitz, Lisy, Rudolf, Sun, Ha. Value Functions for Depth-Limited Solving in Zero-Sum Imperfect-Information Games. AIJ 2023 [background venue] | arXiv 1906.06412 [arXiv-meta] |
| Bro20 | Brown, Bakhtin, Lerer, Gong. Combining Deep RL and Search for Imperfect-Information Games (ReBeL). NeurIPS 2020 | arXiv 2007.13544 [arXiv-meta] |
| Sch23 | Schmid et al. Student of Games. Science Advances 2023 [background venue] | arXiv 2112.03178 [arXiv-meta] |
| Sil17 | Silver et al. Mastering Chess and Shogi by Self-Play (AlphaZero). | arXiv 1712.01815 [arXiv-meta]; Go version DOI 10.1038/nature24270 [Crossref] |
| Sch20 | Schrittwieser et al. MuZero. Nature 2020 [background venue] | arXiv 1911.08265 [arXiv-meta] |
| Lis13 | Lisy, Kovarik, Lanctot, Bosansky. Convergence of MCTS in Simultaneous Move Games. NeurIPS 2013, pp 2112-2120 | arXiv 1310.8613 [web] |
| Lan13 | Lanctot, Lisy, Winands. MCTS in Simultaneous Move Games with Applications to Goofspiel. CGW 2013 (CCIS 408) | DOI 10.1007/978-3-319-05428-5_3 (from your existing notes; not re-verified) |

Quoted evidence:
- Bro18d [arXiv-meta]: "A fundamental challenge in imperfect-information games is that states do not have well-defined values. As a result, depth-limited search algorithms used in single-agent settings and perfect-information games do not apply." Fix: "allowing the opponent to choose among a number of strategies for the remainder of the game at the depth limit."
- Kov23 [arXiv-meta]: formal definition of depth-limited games; "A key ingredient of these depth-limited games are value functions" (abstract truncated by me at 1100 chars; the full text of the sentence was not read).
- Mor17 [arXiv-meta]: "a form of intuition that is automatically learned from self-play using deep learning ... The approach is theoretically sound and is shown to produce more difficult to exploit strategies than prior approaches."
- Bro20 [arXiv-meta]: "prior algorithms of this form cannot cope with imperfect-information games. This paper presents ReBeL, a general framework for self-play reinforcement learning and search".
- Sch23 [arXiv-meta]: "combining guided search, self-play learning, and game-theoretic reasoning."
- Lis13 [web snippet]: "if the selection method is epsilon-Hannan consistent in matrix games and explores enough, MCTS converges to an approximate Nash equilibrium." (This is exactly "per-node no-regret selector gives SM-MCTS convergence"; it is a no-regret-in-search result.)
- Bro17 [arXiv-meta]: "a subgame cannot be solved in isolation and must instead consider the strategy for the entire game as a whole" (already used in prior notes).

Leaf-value bottleneck by system [background, my summary of what each does; verify before citing details]:
- DeepStack: depth-limited re-solving with a counterfactual-value network at the leaf; the value is a function of the public state AND both players' ranges (a vector, not a scalar). Error in this net directly enters the re-solved strategy; the paper's soundness claim is conditional on that error.
- Libratus: no learned leaf; blueprint plus safe/nested endgame solving to the end of the game (Bro17) in late rounds, so the bottleneck is abstraction, not values.
- Pluribus / Bro18d: depth-limited with the leaf as a choice among k continuation strategies for the opponent (multi-valued leaves) to make the leaf robust: the leaf value is itself a game.
- ReBeL / Student of Games: value network over public belief states, trained by self-play on solved subgames; error in v enters every re-solve. Kov23 formalises "value function for depth-limited game" as the object whose quality limits exploitability.
- AlphaZero / MuZero: scalar value (valid because perfect information); no re-solving of imperfect information.
- SM-MCTS (Lis13, Lan13): per-node matrix game with a no-regret / Hannan-consistent selector (EXP3, RM); leaf = rollout or heuristic; convergence claims assume exact leaves in the limit.
Ours: stage matrix + heuristic leaf, solved with RM+; the bound has 2*eps_v at each of H-bar levels. This is the same position as depth-limited solving with an approximate value function; the Pokemon-specific novelty is a finite-horizon, chance-heavy, simultaneous-move version with an explicit additive bound.

How we position: say plainly that our agent is a depth-1 re-solver in the DeepStack/ReBeL family but with a heuristic (not learned) leaf and a perfect simulator (no belief-state value net); RL is the source of leaf values if one wanted to upgrade it (Metamon's value head or PokaiTrainer-style value net would slot in as eps_v). Cite Lis13 for "no-regret selector inside simultaneous-move search converges", and Bro18d/Kov23 for "leaf value error is the bottleneck".

### A3. Measuring exploitability with RL best responses; why RL BRs are weak lower bounds

| key | paper | ID |
|---|---|---|
| Tim20 | Timbers, Bard, Lockhart, Lanctot, Schmid, Burch, Schrittwieser, Hubert, Bowling. Approximate exploitability: Learning a best response (ABR). arXiv 2020 (also IJCAI 2022 [background]) | arXiv 2004.09677 [arXiv-meta] |
| Lan17 | Lanctot et al. A Unified Game-Theoretic Approach to Multiagent RL (PSRO). NeurIPS 2017 | arXiv 1711.00832 [arXiv-meta] |
| Lis17 | Lisy, Bowling. Equilibrium Approximation Quality of Current No-Limit Poker Bots (Local Best Response). AAAI-17 workshop [background] | arXiv 1612.07547 [arXiv-meta]. (Your existing notes said LBR was not located; it is located.) |
| Glea20 | Gleave et al. Adversarial Policies: Attacking Deep RL. ICLR 2020 [background] | arXiv 1905.10615 [arXiv-meta] |
| Wang23 | Wang, Gleave, Tseng, Pelrine, Belrose, Miller, et al. Adversarial Policies Beat Superhuman Go AIs. ICML 2023 [background] | arXiv 2211.00241 [arXiv-meta] |
| Jin21b | Jin, Liu, Yu. The Power of Exploiter: Provable Multi-Agent RL in Large State Spaces. ICML 2022 [background] | arXiv 2106.03352 [arXiv-meta] |
| Rudolph25 | see A1 | exact exploitability for five games |

Quoted evidence:
- Lis17 [arXiv-meta]: "for the simplest variants of the game, it is possible to evaluate the quality of the approximation by computing the value of the best response strategy, this is currently not computationally feasible for larger variants ... we present a simple and computationally inexpensive Local Best Response method for computing an approximate lower bound on the value of the best response strategy."
- Tim20 [arXiv-meta]: "exact computation is unfortunately infeasible in larger domains" (abstract continues, only partly read) and they train an RL best response to get an "approximate exploitability"; it is a lower bound, as any exploiter's value is.
- Lan17 [arXiv-meta]: describes joint-policy correlation metric; PSRO's inner loop is an RL best response to the meta-Nash mixture (inner oracle = "an entire reinforcement learning problem", per Rudolph25 [PDF-spot]).
- Wang23 [arXiv-meta]: adversaries reach ">97% win rate against KataGo" and "do not win by playing Go well. Instead, they trick KataGo into making serious blunders." This is the best evidence that RL BR exploiters CAN succeed against a strong agent and that the exploiter is trained against a FROZEN victim.
- Glea20 [arXiv-meta]: adversarial policies against self-play-trained victims in zero-sum games.
- Jin21b [arXiv-meta, title only read in this session]: "exploiter" as an algorithmic component in multi-agent RL theory.

Why RL BRs are weak lower bounds [partly my reasoning, partly Rudolph25]: (1) an RL BR is a stochastic learner; its measured win rate lower-bounds the true BR value, and failure to find an exploit is not evidence of safety; (2) cost: a BR for a stochastic, long-horizon, simultaneous-move game with hidden sets needs many episodes and credit assignment through chance; (3) it is solving the single-agent MDP induced by freezing our agent, whose reward is sparse (win/loss at end of up to ~100 stochastic turns); (4) imperfect information plus chance makes the signal very noisy. Wang23 shows success needs targeted training budgets and a known victim; ours uses 100k games.

How we position: report exploitability as a bracket, not a number. White-box search exploiter = certified lower bound that knows our policy (Lis17-style; Lis17 is the closest analogue and is "an approximate lower bound" by construction). PPO BR = Tim20/Wang23-style RL BR; its failure at ~100k games is a statement about RL BR cost (Rudolph25 reports the same "each BR is an entire RL problem" bottleneck), not evidence of low exploitability. Cite Wang23 to pre-empt "why not just train an adversary" (it works at massive scale with a known victim) and Lis17/Tim20 as the method family. Never claim an upper bound from either.

### A4. Stochastic / Markov games: no-regret learning, hardness, policy regret (our Prop E3)

| key | paper | ID |
|---|---|---|
| Bai20 | Bai, Jin. Provable Self-Play Algorithms for Competitive RL. ICML 2020 [background venue] | arXiv 2002.04017 [arXiv-meta] |
| Bai20b | Bai, Jin, Yu. Near-Optimal RL with Self-Play. NeurIPS 2020 [background venue] | arXiv 2006.12007 [arXiv-meta] |
| Liu21 | Liu, Yu, Bai, Jin. A Sharp Analysis of Model-based RL with Self-Play. ICML 2021 [background venue] | arXiv 2010.01604 [arXiv-meta] |
| Jin22 | Jin, Liu, Wang, Yu. V-Learning. | arXiv 2110.14555 [arXiv-meta] |
| Liu22 | Liu, Wang, Jin. Learning Markov Games with Adversarial Opponents: Efficient Algorithms and Fundamental Limits. ICML 2022 [background venue] | arXiv 2203.06803 [arXiv-meta] |
| Tian21 | Tian, Wang, Yu, Sra. Online Learning in Unknown Markov Games. ICML 2021 [background venue] | arXiv 2010.15020 [arXiv-meta] |
| Aro12 | Arora, Dekel, Tewari. Online Bandit Learning against an Adaptive Adversary: from Regret to Policy Regret. ICML 2012 | arXiv 1206.6400 [arXiv-meta] |
| Aro18 | Arora, Dinitz, Marinov, Mohri. Policy Regret in Repeated Games. NeurIPS 2018 [background venue] | arXiv 1811.04127 [arXiv-meta] |
| Per15 | Perolat, Scherrer, Piot, Pietquin. Approximate Dynamic Programming for Two-Player Zero-Sum Markov Games. ICML 2015, PMLR 37:1321-1329 | proceedings.mlr.press/v37/perolat15.html [web] |
| Zhao22 | Zhao, Tian, Lee, Du. Provably Efficient Policy Optimization for Two-Player Zero-Sum Markov Games. AISTATS 2022 [background] | arXiv 2102.08903 [arXiv-meta] |
| Xie20 | Xie, Chen, Wang, Yang. Learning Zero-Sum Simultaneous-Move Markov Games Using Function Approximation and Correlated Equilibrium. COLT 2020 [background] | arXiv 2002.07066 [arXiv-meta] |
| Sha53 | Shapley, Stochastic games (already in report bib) | PNAS 1953 |

Quoted evidence (all [arXiv-meta]):
- Bai20: regret "measured by the agent's performance against a fully adversarial opponent who can exploit the agent's strategy at any step" and "O~(sqrt(T)) after playing T steps". This is the one-sided guarantee closest in spirit to "exploitability of the deployed policy".
- Liu22: "no-regret learning in Markov games with adversarial opponents when competing against the best fixed policy in hindsight ... When the policies of the opponents are revealed at the end of each episode, we propose new efficient algorithms achieving sqrt(K)-regret bounds when either (1) the baseline policy class is small or (2) the opponent's policy class is small. This is complemented with an exponential lower bound when neither conditions are true. When the policies of the opponents are not revealed, we prove a statistical hardness result even in the most favorable scenario".
- Tian21: "achieving sublinear regret against the best response in hindsight is statistically hard. We then consider a weaker notion of regret by competing with the minimax value of the game, and present an algorithm that achieves a sublinear O~(K^{2/3}) regret after K episodes."
- Aro12: "no bandit algorithm can guarantee a sublinear policy regret against an adaptive adversary with unbounded memory. On the other hand, if the adversary's memory is bounded ... converts any bandit algorithm with a sublinear regret bound into an algorithm with a sublinear policy regret bound."
- Aro18: "there are online learning settings in which policy regret and external regret are incompatible ... In [the game-theoretic] setting, we show that external regret and policy regret are not in conflict ... so long as the adversary is also using such an algorithm."
- Per15: "unified error-propagation analysis in L_p-norm for three value-based methods adapted to stochastic games: approximate value iteration, approximate policy iteration, and approximate generalized policy iteration" [web paraphrase of the abstract, not verbatim]. This is the closest published analogue of "ADP error propagation with an adversarial opponent" that I found. Discounted infinite-horizon; ours is finite-horizon.

How we position (Prop E3): Liu22 and Tian21 show that sublinear regret against the best response in hindsight (even best FIXED policy against adaptive/unobserved opponent) is statistically hard in Markov games; hence a Hedge layer can only promise external regret vs fixed experts along the realised trajectory (and Aro12 shows policy regret against an adaptive adversary with unbounded memory is impossible in general). The Pokemon opponent is not adaptive across games (bots, or at most a ladder), so the external-regret comparator is meaningful only if the game-to-game opponent drift is not exploited by switching; Aro18 gives the clean statement: external and policy regret coexist when the adversary also runs a no-regret algorithm. We should state this as a limit, not a deficiency, and cite Tian21 for "competing with the minimax value is the achievable benchmark", which is exactly our "pathwise safety" target. Per15 is the right citation for the stage-wise value-error propagation under an adversary.

### A5. Pokemon: RL vs search (existing coverage + pointers)

Already covered in `notes/prior_art_pokemon.md` (A1-A13). IDs re-verified here [arXiv-meta]:
- Metamon, Grigsby, Xie, Sasek, Zheng, Zhu. "Human-Level Competitive Pokemon via Scalable Offline RL with Transformers", arXiv 2504.04395. Abstract: "AI research in CPS has been led by heuristic tree search and online self-play, but the game may also create a platform to study adaptive policies trained offline on large datasets."
- PokeChamp, Karten, Nguyen, Jin. arXiv 2503.04094. Abstract: "a minimax agent powered by LLMs ... LLMs replace three key modules: (1) player action sampling, (2) opponent modeling, and (3) value function estimation".
- PokaiTrainer, Yu, arXiv 2608.29197 (v2 date in report bib; arXiv page lists first submission 2026-08-29). Abstract: "No prior Pokemon agent performs equilibrium search".
- Foul Play (MCTS/DUCT, code), Wang 2024 MIT MEng thesis (PPO+MCTS, hdl 1721.1/153888), Lee & Togelius CIG 2017 DOI 10.1109/CIG.2017.8080435 [Crossref]: see existing notes.
How we position: the RL-vs-search axis in Pokemon is (offline RL policy | LLM+minimax | MCTS/DUCT | PPO+MCTS | equilibrium search). We are the only one in the list that (i) solves a stage matrix with a no-regret method AND (ii) states a full-game exploitability bound; RL is the opponent family for evaluation (Metamon-style policies as exploiters/benchmarks), not our method. (The novelty claim relative to PokaiTrainer is in your existing notes; I did not re-assess it.)

---

## B. Broad (non-game) setting

### B6. Online learning <-> RL reductions

| key | paper | ID |
|---|---|---|
| EKM09 | Even-Dar, Kakade, Mansour. Online Markov Decision Processes. Math. Oper. Res. 34(3), 2009 | DOI 10.1287/moor.1090.0396 [Crossref] |
| Neu14 | Neu, Gyorgy, Szepesvari, Antos. Online Markov Decision Processes under Bandit Feedback. IEEE TAC 59(3), 2014 (NIPS 2010 version) | DOI 10.1109/TAC.2013.2292137 [Crossref] |
| Tomar22 | Tomar, Shani, Efroni, Ghavamzadeh. Mirror Descent Policy Optimization. ICLR 2022 [background] | arXiv 2005.09814 [arXiv-meta] |
| Gei19 | Geist, Scherrer, Pietquin. A Theory of Regularized Markov Decision Processes. ICML 2019 [background] | arXiv 1901.11275 [arXiv-meta] |
| Sha20 | Shani, Efroni, Mannor. Adaptive Trust Region Policy Optimization: Global Convergence and Faster Rates for Regularized MDPs. AAAI 2020 [background] | arXiv 1909.02769 [arXiv-meta] |
| Aga21 | Agarwal, Kakade, Lee, Mahajan. On the Theory of Policy Gradient Methods: Optimality, Approximation, and Distribution Shift. JMLR 22, 2021 [background] | arXiv 1908.00261 [arXiv-meta] |
| Cai20 | Cai, Yang, Jin, Wang. Provably Efficient Exploration in Policy Optimization (OPPO). ICML 2020 [background] | arXiv 1912.05830 [arXiv-meta] |
| Efr20 | Efroni, Shani, Rosenberg, Mannor. Optimistic Policy Optimization with Bandit Feedback. ICML 2020 [background] | arXiv 2002.08243 [arXiv-meta] |
| Pol19 | Abbasi-Yadkori, Bartlett, Bhatia, Lazic, Szepesvari, Weisz. POLITEX: Regret Bounds for Policy Iteration using Expert Prediction. ICML 2019, PMLR 97:3692-3702 | proceedings.mlr.press/v97/lazic19a.html [web]; no arXiv found |
| KL02 | Kakade, Langford. Approximately Optimal Approximate RL (CPI). ICML 2002 | [background, not checked]. Kakade, A Natural Policy Gradient, NeurIPS 2001 [background] |
| Sri18 | Srinivasan, Lanctot, Zambaldi, Perolat, Tuyls, Munos, Bowling. Actor-Critic Policy Optimization in Partially Observable Multiagent Environments. NeurIPS 2018 [background] | arXiv 1810.09026 [arXiv-meta] (bridges policy gradient and regret minimisation) |
| Schu15,17 | TRPO (Schulman et al. ICML 2015); PPO (Schulman et al. arXiv 1707.06347) [background, IDs not fetched this session; arXiv API was rate-limited] |

Quoted evidence (all [arXiv-meta] unless stated):
- Tomar22: "We highlight the connections between on-policy MDPO and two popular trust-region RL algorithms: TRPO and PPO" (abstract truncated by me; MDPO is "mirror descent policy optimization", with "a linearization of the standard RL objective and a proximity term").
- Sha20: "We show that the adaptive scaling mechanism used in TRPO is in fact the natural 'RL version' of traditional mirror descent updates" (the clause I read ends "RL ve..."; text beyond the truncation was not read, so treat the "RL version" wording as approximate).
- Gei19: "a general theory of regularized Markov Decision Processes that generalizes these approaches ... entropy or Kullback-Leibler divergence".
- Aga21: "provable characterization of the computational, approximation, and sample size properties of policy gradient methods"; the NPG-multiplicative-weights relation appears in the body, not the abstract (not checked by me). It is also stated in your brief as a known fact; I mark it [background].
- Cai20: "Optimistic variant of the Proximal Policy Optimization algorithm (OPPO)" with exploration via an "optimism in the face of uncertainty" bonus.
- Pol19 [web]: "a soft variant of policy iteration. Each policy corresponds to a Boltzmann distribution over the sum of previous action-value functions" with regret ~O~(sqrt(d) T^{3/4} + eps*T) in uniformly mixing MDPs, where eps is the worst-case value-approximation error [web paraphrase]. Note the "+ eps*T" term: approximation error enters regret linearly, the same shape as our "H-bar * eps_v" and the same shape as the linear-regret bug.
- EKM09 [Crossref only; not read]: online MDP regret with adversarial rewards and known dynamics. Neu14 [Crossref only]: bandit-feedback version.

How we position: policy optimisation per state is Hedge/FTRL on Q-values (NPG = multiplicative weights per state; POLITEX = FTRL on cumulative Q; NeuRD = Hedge with function approximation); our Hedge layer is the same algorithm at a coarser granularity (experts = whole policies), with comparator = best fixed expert, which is precisely the CMDP/adversarial-MDP comparator of EKM09/Neu14 but with a model-free surrogate loss. The state-wise decomposition is also where our full-game bound sits: it resembles the performance-difference lemma + per-state regret (as in EKM09/CPI), but with a game, not an MDP, at each state.

### B7. No-regret as an RL tool

| key | paper | ID |
|---|---|---|
| Ross11 | Ross, Gordon, Bagnell. A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning (DAgger). AISTATS 2011 | arXiv 1011.0686 [arXiv-meta] |
| Jak10 | Jaksch, Ortner, Auer. Near-optimal Regret Bounds for RL (UCRL2). JMLR 11, 2010 [background, not checked] |
| Aue02 | Auer, Cesa-Bianchi, Freund, Schapire. The Nonstochastic Multiarmed Bandit Problem (EXP3, EXP4). SIAM J. Comput. 32(1), 2002 | DOI 10.1137/S0097539701398375 [Crossref] |
| Aga17 | Agarwal, Luo, Neyshabur, Schapire. Corralling a Band of Bandit Algorithms. COLT 2017 [background] | arXiv 1612.06246 [arXiv-meta] |
| Pac20 | Pacchiano, Dann, Gentile, Bartlett. Regret Bound Balancing and Elimination for Model Selection in Bandits and RL. arXiv 2020 | arXiv 2012.13045 [arXiv-meta] |
| Fos19 | Foster, Krishnamurthy, Luo. Model Selection for Contextual Bandits. NeurIPS 2019 [background] | arXiv 1906.00531 [arXiv-meta] |
| Bro19c | Brown, Lerer, Gross, Sandholm. Deep Counterfactual Regret Minimization. ICML 2019 [background] | arXiv 1811.00164 [arXiv-meta] |

Quoted evidence:
- Ross11 [arXiv-meta]: "Sequential prediction problems such as imitation learning, where future observations depend on previous predictions (actions), violate the common i.i.d. assumptions". Reduction to no-regret online learning (abstract continues; remainder not quoted).
- Aga17 [arXiv-meta]: "combining multiple bandit algorithms ... with the goal of creating a master algorithm that performs almost as well as the best base algorithm if it were to be run on its own. The main challenge is that when run with a master, base algorithms unavoidably receive much less feedback".
- Pac20 [arXiv-meta]: "we only require that each base algorithm comes with a candidate regret bound that may or may not hold during all rounds"; model selection "in stochastic bandit and reinforcement learning problems".
- Aue02 [Crossref only]: EXP3/EXP4 are Hedge with importance-weighted losses; EXP4 is experts-over-policies (my background knowledge).

How we position: DAgger is the cleanest example of "no-regret = tool for RL": the guarantee is a regret bound on a sequence of losses chosen adaptively by the learner's own state distribution, which is the same structure as our across-game Hedge with surrogate losses; note DAgger's reduction requires the comparator to be a fixed policy against the on-policy state distribution. Corral/RBB are the model-selection analogues of Hedge over {MCTS, equilibrium}: their difficulty (base algorithms get starved of feedback) is absent in our case because both experts can be evaluated on every game via the simulator (full-information Hedge, not EXP4), which should be said explicitly: it is why plain Hedge/FTRL suffices.

### B8. Surrogate/proxy rewards; ADP error propagation; simulation lemma

| key | paper | ID |
|---|---|---|
| SY94 | Singh, Yee. An Upper Bound on the Loss from Approximate Optimal-Value Functions. Machine Learning 16, 1994 [background, not checked] |
| KS02 | Kearns, Singh. Near-Optimal RL in Polynomial Time (simulation lemma). Machine Learning 49, 2002 | DOI 10.1023/A:1017984413808 [Crossref] |
| Mun07 | Munos. Performance Bounds in L_p-norm for Approximate Value Iteration. SIAM J. Control Optim. 46(2), 2007 | DOI 10.1137/040614384 [Crossref] (Munos, Error bounds for approximate value iteration, AAAI 2005 [background]) |
| Sch15 | Scherrer, Ghavamzadeh, Gabillon, Lesner, Geist. Approximate Modified Policy Iteration and its Application to the Game of Tetris. JMLR 16, 2015 | arXiv 1205.3054 [arXiv-meta] (arXiv title is "Approximate Modified Policy Iteration"; JMLR title includes Tetris [background]) |
| Per15 | see A4 | zero-sum Markov game analogue |
| NHR99 | Ng, Harada, Russell. Policy Invariance under Reward Transformations: Theory and Application to Reward Shaping. ICML 1999 [background, not checked] |

Quoted evidence:
- Sch15 [arXiv-meta]: "Modified policy iteration (MPI) is a dynamic programming algorithm that contains the two celebrated policy and value iteration methods ... we propose three implementations of approximate MPI (AMPI) that are extensions of well-known approximate DP algorithms" (error-propagation bounds are the paper's main result; the abstract excerpt I read ends at the implementation list).
- Per15 [web]: L_p error propagation for AVI/API/AGPI in stochastic games.
- Rest [background]: Singh-Yee: greedy policy wrt V-hat with sup-error eps loses at most 2*eps/(1-gamma) (finite-horizon analogue: 2*H*eps_v). KS02 simulation lemma: model/reward error alpha gives value error O(H^2 alpha)-type bounds depending on form; if reward-only error, H*alpha. I did not check exact constants here; verify before using.

How we position (my inference, not a paper claim): H-bar*(eps_s + 2*eps_v) is the finite-horizon, adversarial-opponent version of the Singh-Yee/Munos/Scherrer error-propagation template: a per-step value error eps_v enters twice (once for comparing own action against the reference, once for the opponent's best response) and accumulates linearly over H-bar steps; eps_s is the per-step optimisation (solver) error, the analogue of the "approximation of greedy step" term in AMPI. Per15 is the closest published match (adds the min-max operator). The "R4: surrogate-to-true costs 2*eps_v" claim is the same as the standard "greedy w.r.t. wrong value loses 2*eps" argument: cite SY94/Mun07. For reward-model misspecification I did not find a standard result that is exactly "regret under an estimated reward vs true reward"; the simulation lemma is the closest, plus NHR99 for the (non-)effect of potential-based reward shaping. Gap flagged below.

### B9. Anytime / adaptive step sizes; doubling trick; scale-free

| key | paper | ID |
|---|---|---|
| Orab19 | Orabona. A Modern Introduction to Online Learning. arXiv 2019 (rev.) | arXiv 1912.13213 [arXiv-meta]; abstract title there "Online Learning: A Modern Introduction Using Convex Optimization" |
| OP18 | Orabona, Pal. Scale-Free Online Learning. TCS 716, 2018 | arXiv 1601.01974 [arXiv-meta]; DOI 10.1016/j.tcs.2017.11.021 [Crossref] |
| OP16 | Orabona, Pal. Coin Betting and Parameter-Free Online Learning. NeurIPS 2016 [background] | arXiv 1602.04128 [arXiv-meta] |
| BK18 | Besson, Kaufmann. What Doubling Tricks Can and Can't Do for Multi-Armed Bandits. arXiv 2018 | arXiv 1803.06971 [arXiv-meta] |
| ACG02 | Auer, Cesa-Bianchi, Gentile. Adaptive and Self-Confident On-Line Learning Algorithms. JCSS 64(1), 2002 | DOI 10.1006/jcss.2001.1795 [Crossref] |
| dRvEGK14 | de Rooij, van Erven, Grunwald, Koolen. Follow the Leader If You Can, Hedge If You Must. JMLR 15, 2014 [background] | arXiv 1301.0534 [arXiv-meta] |
| Ors21 | Orseau, Hutter. Isotuning with Applications to Scale-Free Online Learning. arXiv 2021 | arXiv 2112.14586 [arXiv-meta] |

Quoted evidence:
- OP18 [arXiv-meta]: "algorithms for online linear optimization that have optimal regret and at the same time do not need to know any upper or lower bounds on the norm of the loss vectors ... instances of FTRL and Mirror Descent ... scale invariance, i.e., our algorithms make exactly the same ..." (truncated).
- Orab19 [arXiv-meta]: "I attempted to unify all the literature as instantiations of Online Mirror Descent and Follow-the-Regularized-Leader ... I paid particular attention to the issue of tuning the parameters of the algorithms, through adaptive ...".
- BK18 [arXiv-meta]: "An online reinforcement learning algorithm is anytime if it does not need to know in advance the horizon T ... A well-known technique to obtain an anytime algorithm from any non-anytime algorithm is the 'Doubling Trick'."
- OP16 [arXiv-meta]: "parameter-free algorithms ... achieve optimal regret bounds that depend on the unknown competitors, without having to tune the learning rates with oracle choices".
- Ors21 [arXiv-meta]: "fast, adaptive, anytime and scale-free online learning algorithms".
- dRvEGK14 [arXiv-meta]: "FTL ... guarantees constant regret in the stochastic setting, but has terrible performance for worst-case data" (relevant to the "incremental anytime update" and to Hedge-vs-FTL on fixed opponents; existing notes section 2.6).

How we position: the anytime-update bug (linear regret) belongs to the same family as the known pitfalls in OP18's scale-free analysis (updates that use the running maximum/accumulated scale incorrectly); BK18 is the right reference for "doubling vs truly anytime" and Orab19 for the textbook FTRL-with-time-varying-regulariser analysis. I did not re-read OP18 to confirm the exact construction you cite; your own derivation of the bug is not something I verified.

---

## Framing paragraph draft (for intro / related work; about one page, edit to taste)

Reinforcement learning and no-regret online learning are two answers to the same question, how an agent should improve from experience, but they make different assumptions and offer different guarantees. In RL the environment is a fixed Markov decision process and the objective is return of the learned policy; the central theory is dynamic-programming error propagation (Singh and Yee 1994; Munos 2007; Scherrer et al. 2015) and regret against the optimal policy under exploration (Jaksch et al. 2010; Cai et al. 2020). In online learning the data are adversarial and the objective is regret against a comparator class, with algorithms such as Hedge and FTRL (Cesa-Bianchi and Lugosi 2006; Orabona 2019). The two meet in three places. First, policy optimisation is online learning on each state: natural policy gradient is multiplicative weights on Q-values (Agarwal et al. 2021), TRPO and PPO are mirror-descent steps (Tomar et al. 2022; Shani et al. 2020), and POLITEX is FTRL over cumulative action-values (Abbasi-Yadkori et al. 2019). Second, in zero-sum games, average no-regret play converges to equilibrium, which is why regret-matching variants (CFR, CFR+, RM+) underlie superhuman poker (Moravcik et al. 2017; Brown and Sandholm 2018, 2019), while policy-gradient self-play only converges in the last iterate after regularisation (Perolat et al. 2021, 2022; Sokota et al. 2023). Recent large-scale evidence suggests generic PPO is competitive with CFR-, fictitious-play- and double-oracle-based deep RL on exact exploitability (Rudolph et al. 2025), but that comparison concerns model-free learners, not search with a known simulator. Third, no-regret learning is a tool for RL: DAgger reduces imitation to online learning (Ross et al. 2011), and Corral and related model-selection algorithms run a master over base learners (Agarwal et al. 2017).

Our setting, Gen 9 Pokemon Random Battles, is a finite-horizon zero-sum stochastic game with simultaneous moves, chance and hidden information, and our contribution lives at this interface. The per-turn agent solves a stage matrix, one simulated turn followed by a heuristic leaf, with RM+; this is depth-limited re-solving as in DeepStack, Libratus, Pluribus and ReBeL (Moravcik et al. 2017; Brown et al. 2018; Brown et al. 2020), with the leaf value, not the solver, as the bottleneck (Kovarik et al. 2023), and with a no-regret per-node selector as in simultaneous-move MCTS (Lisy et al. 2013). Our full-game bound has the form of the standard error-propagation template, a per-stage solver error plus twice the per-stage value error, accumulated over the horizon, extended to a min-max step as in approximate DP for zero-sum Markov games (Perolat et al. 2015). Across games we run Hedge or FTRL over whole policies (MCTS and an equilibrium solver), which is experts-over-policies; because both experts can be evaluated on every game, the corralling difficulties of Agarwal et al. (2017) do not arise. What this layer can guarantee is limited by known hardness: against an adaptive opponent in a Markov game, sublinear regret versus the best response in hindsight is statistically hard (Tian et al. 2021; Liu et al. 2022), and policy regret cannot be controlled against an adversary with unbounded memory (Arora et al. 2012), so we state external regret along the realised trajectory only. RL enters in two roles: as a source of leaf values (a Metamon- or PokaiTrainer-style value network would plug into the same bound as eps_v), and as a best-response oracle for measuring exploitability (Timbers et al. 2022; Lanctot et al. 2017). The latter is a weak instrument: an RL best response gives only a lower bound whose tightness depends on training cost, and our PPO adversary failed to exploit even at about 100k games, while a white-box search exploiter (in the spirit of local best response, Lisy and Bowling 2017) certifies a stronger lower bound. We therefore report exploitability as a bracket, not as a point estimate.

---

## Top 12 citations with bibitems (bib keys chosen to avoid clashes with existing report keys: Tim20, Aro12b, OP18, CBL06, Mor17, Bro20, Tam14 already exist)

Metadata comes from arXiv meta / Crossref as above; page numbers only where verified ([web] or Crossref), otherwise omitted. Check authors truncated in this list: I abbreviated long author lists with "et~al." Please expand if the venue requires.

```
\bibitem{Rud25} M.~Rudolph, N.~Lichtle, S.~Mohammadpour, A.~Bayen, J.~Z.~Kolter, et~al. Reevaluating policy gradient methods for imperfect-information games. arXiv:2502.08938, 2025.

\bibitem{Sok23} S.~Sokota, R.~D'Orazio, J.~Z.~Kolter, N.~Loizou, M.~Lanctot, I.~Mitliagkas, N.~Brown, C.~Kroer. A unified approach to reinforcement learning, quantal response equilibria, and two-player zero-sum games. ICLR, 2023; arXiv:2206.05825.

\bibitem{Per22} J.~Perolat, B.~de~Vylder, D.~Hennes, et~al. Mastering the game of Stratego with model-free multiagent reinforcement learning. Science 378(6623):990--996, 2022; doi:10.1126/science.add4679; arXiv:2206.15378.

\bibitem{Bro18d} N.~Brown, T.~Sandholm, B.~Amos. Depth-limited solving for imperfect-information games. NeurIPS, 2018; arXiv:1805.08195.

\bibitem{Lis13} V.~Lis\'y, V.~Kova\v{r}\'{\i}k, M.~Lanctot, B.~Bo\v{s}ansk\'y. Convergence of Monte Carlo tree search in simultaneous move games. NeurIPS, pp.~2112--2120, 2013; arXiv:1310.8613.

\bibitem{Lis17} V.~Lis\'y, M.~Bowling. Equilibrium approximation quality of current no-limit poker bots. AAAI-17 Workshop on Computer Poker and Imperfect Information Games; arXiv:1612.07547, 2016.

\bibitem{Lan17} M.~Lanctot, V.~Zambaldi, A.~Gruslys, et~al. A unified game-theoretic approach to multiagent reinforcement learning. NeurIPS, 2017; arXiv:1711.00832.

\bibitem{Tian21} Y.~Tian, Y.~Wang, T.~Yu, S.~Sra. Online learning in unknown Markov games. ICML, 2021; arXiv:2010.15020.

\bibitem{Liu22} Q.~Liu, Y.~Wang, C.~Jin. Learning Markov games with adversarial opponents: efficient algorithms and fundamental limits. ICML, 2022; arXiv:2203.06803.

\bibitem{Per15} J.~P\'erolat, B.~Scherrer, B.~Piot, O.~Pietquin. Approximate dynamic programming for two-player zero-sum Markov games. ICML, PMLR 37:1321--1329, 2015.

\bibitem{Aga21} A.~Agarwal, S.~M.~Kakade, J.~D.~Lee, G.~Mahajan. On the theory of policy gradient methods: optimality, approximation, and distribution shift. JMLR 22(98), 2021; arXiv:1908.00261.

\bibitem{Tom22} M.~Tomar, L.~Shani, Y.~Efroni, M.~Ghavamzadeh. Mirror descent policy optimization. ICLR, 2022; arXiv:2005.09814.

\bibitem{Pol19} Y.~Abbasi-Yadkori, P.~Bartlett, K.~Bhatia, N.~Lazic, C.~Szepesv\'ari, G.~Weisz. POLITEX: regret bounds for policy iteration using expert prediction. ICML, PMLR 97:3692--3702, 2019.

\bibitem{Ross11} S.~Ross, G.~J.~Gordon, J.~A.~Bagnell. A reduction of imitation learning and structured prediction to no-regret online learning. AISTATS, 2011; arXiv:1011.0686.

\bibitem{Aga17} A.~Agarwal, H.~Luo, B.~Neyshabur, R.~E.~Schapire. Corralling a band of bandit algorithms. COLT, 2017; arXiv:1612.06246.

\bibitem{Mun07} R.~Munos. Performance bounds in $L_p$-norm for approximate value iteration. SIAM J.\ Control Optim.\ 46(2):541--561, 2007; doi:10.1137/040614384.

\bibitem{Orab19} F.~Orabona. A modern introduction to online learning. arXiv:1912.13213, 2019.
```
(Selection: 16 entries because the brief's "8-12 most important" had to cover A and B; if you must cut to 12 drop Aga17, Orab19, Tom22, Lis17 first. Rud25, Sok23, Per22, Bro18d, Lis13, Lan17, Tian21, Liu22, Per15, Aga21, Pol19, Ross11 are the minimal 12. Existing report bibitems already cover Aro12b, OP18, Tim20, Mor17, Bro20, CBL06.) Page numbers for Per22 (990--996) and Mun07 (541--561) are from background knowledge, not verified here; remove if not rechecked.

## What I could not verify

- Venue/year for: Rudolph25 (arXiv only), Sokota23 (ICLR 2023), Hennes20, Perolat21 (ICML 2021?), Bro19, Lee21, Cai24, Fari21, Dask21, Bai20, Bai20b, Liu21, Liu22, Tian21 (ICML 2021), Aro18, Tomar22, Gei19, Sha20, Aga21, Cai20, Efr20, Aga17 (COLT 2017), Ross11 pages, Sch23, Sch20, Tim20 IJCAI-22: all [background]; DOIs for these not checked (no Crossref query run for them).
- Full-text content: apart from Rudolph25 (grep of intro/related work), every quote is from abstracts. No PDF of Liu22, Tian21, Aro12, Aro18, Per15, Pol19 was read, so "hardness" claims are abstract-level; read the theorem statements (Liu22 Thm on non-revealed opponent policies; Tian21 hardness result; Aro12 Thm on unbounded memory) before turning Prop E3's positioning into a formal statement.
- NPG = Hedge per state: stated in your brief and widely known; I did not locate the exact sentence in Agarwal et al. (abstract does not contain it) nor Kakade 2001/Kakade-Langford 2002 (no ID fetched).
- Singh-Yee 2eps/(1-gamma) and simulation-lemma constants: background, unchecked; no paper found that is exactly "regret under estimated reward vs true reward", so R4's surrogate-to-true transfer likely has to be self-derived (cite SY94/Mun07 as analogues).
- Perolat15 abstract only via search snippets; its theorem forms (L_p, discounted) not checked and ours is finite-horizon with simultaneous moves and chance.
- DeepStack/ReBeL/SoG/Pluribus leaf-value descriptions are my summary from background knowledge; the specific claims (value net input = ranges for DeepStack; k=4 continuation strategies for Pluribus) are unchecked.
- Lanctot-Lisy-Winands CGW 2013 DOI is carried over from your existing note; Lisy-Kovarik NeurIPS result confirmed only via web snippets.
- Metamon/PokeChamp/Wang-thesis numeric claims were not re-checked (existing notes cover them).
- Whether Rudolph25 uses "Mei" as last author was not checked (author list in the arXiv meta tag is truncated by my script at 80 chars); expand authors before citing.
- Not searched: LLM/RLHF direction (a single hit, InstructGPT 2203.02155, appeared incidentally in a title query; irrelevant here), CORRAL-style results for adversarial MDPs specifically, Safe RL / "robust MDP" literature as another angle on adversarial comparators, and the Undermind deep-search sweep (not run; targeted arXiv/Crossref lookups were used instead). A deeper search should target: (i) policy-regret in Markov games (Arora-type results with state), (ii) sample complexity of best-response/exploiter learning in zero-sum Markov games with simultaneous moves (Jin21b follow-ups), (iii) error propagation for finite-horizon zero-sum Markov games with approximate value functions at depth limit (Kov23-style bounds with chance nodes).
