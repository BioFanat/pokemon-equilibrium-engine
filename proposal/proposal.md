# Bayesian Memory and No-Regret Learning for Strategic Decision-Making in Pokémon Showdown

*CS535 project proposal: [Name 1], [Name 2]*

## 1. Introduction and motivation

A competitive Pokémon battle is a two-player, winner-take-all game played under uncertainty:
- **Hidden information.** Players see the opponent's Pokémon, their health and the moves already used, but not each Pokémon's full moveset, item, ability or Tera type. A single unseen Choice Scarf or coverage move can decide a battle.
- **Simultaneous choices.** Both players choose *at the same time*.
- **Randomness.** Damage, accuracy and critical hits are random.

We focus on Gen 9 Random Battles, where teams are drawn from a known random generator.

**How current bots play.** The strongest bots pair search with guesses about what the opponent is holding:
- **Foul Play** (open source; peak ≈ 2341 Elo) samples a handful of complete guesses about the opponent's team. It plans separately for each one as if that guess were true and visible to both sides, then blends the plans.
- **Jaxcalibur** (closed source; self-reported ≈ 2557 Elo) adds a network trained by self-play.
- **Chronoquazicalar** (open source) does the same blending, with regret-based search.
- **Other bots** learn from replays (Metamon) or use language models (PokéChamp).

**What they leave on the table:**
1. **Blending per-guess plans is a known mistake.** We can't play differently in each guessed world, because we don't know which one we're in. The opponent *does* know its own team, so a blended plan can be exploited.
2. **Their memory is shallow.** They rule out impossible sets using hard evidence, but never ask what the opponent's *choices* say about what it holds or how it plays.
3. **They can't say how good they are** beyond ladder Elo: how exploitable they are, how they fare against predictable opponents, or where they stand among bots of equal strength.

We confirmed the first two in the open bots' code.

## 2. Proposed idea

A strong player keeps two questions in mind at once:
- *What does the opponent have?*
- *How does this opponent play?*

Take an opponent that threatens a knockout. Switching to a counter invites a setup move, while staying in risks the knockout. The right answer depends on both questions, and no single action is always right, so good play *mixes* its choices.

Inspired by poker AI, we propose an agent with two parts:
- a **Bayesian memory** that answers both questions together;
- a **no-regret learner** that turns those answers into a mixed strategy.

**One memory for "what they have" and "how they play."** The agent keeps a single probability distribution over two unknowns.

**The set S.** The hidden details of the opponent's team: each Pokémon's moves, item, ability and Tera type, including Pokémon not yet revealed. At the start of a battle we draw S from Showdown's own team generator, so our starting belief already knows which moves, items and abilities tend to appear together.

**The style σ.** A short menu of ways an opponent might pick actions. Each style is a rule that gives, for any position and any set, the probability of each legal action:
- **Near-optimal:** plays the equilibrium strategy that our own solver computes for that set.
- **Aggressive:** like near-optimal, but attacks more often than it should.
- **Switch-happy:** switches out of bad matchups more often than it should.
- **Greedy:** always uses its highest-damage move. Predictable, and easy to exploit if detected.

Starting weights on the styles come from how often each style fits human replays.

**What we observe.** Each turn t we see:
- the opponent's **choice** c_t: which move it used, or which Pokémon it switched to;
- the **evidence** e_t: everything the game shows about the result, such as damage dealt, who moved first, items or abilities activating, and Tera.

The **history** H_t is everything seen so far in this battle, (c_1, e_1, …, c_t, e_t), together with our own actions.

**The update.** After each turn, Bayes' rule gives

  P(S, σ | H_t) ∝ P(e_t | S) · P(c_t | S, σ) · P(S, σ | H_{t−1}).

- **P(e_t | S): how likely this result was if they hold set S** (given the actions both sides took). Usually 0 or 1, because a set either fits the evidence or doesn't: a Pokémon that moved first must be fast enough, and a move it used must be in its set. Damage contributes a probability, because each of Showdown's 16 damage rolls is equally likely. This term only sharpens beliefs about the set.
- **P(c_t | S, σ): how likely this choice was for that set and style.** This term updates both unknowns at once. A near-optimal player who refuses to stay in probably holds something that can't afford to, while a switch-happy player doing the same tells us little.
- **P(S, σ | H_{t−1}):** our belief going into the turn.

**Long-term memory.** The set belief resets each battle, because teams are random. The style belief carries over between battles against the same opponent.

**How the agent picks a move.** Each turn has four steps. Both learners in it are multiplicative weights (MWU).

1. **Guess what they might have.** Sample several possible sets from the memory. For each set, simulate every pair (our action, their action) to get a table of our estimated win probability.
2. **Plan the safe strategy (MWU self-play, in our head).**
   - *Players:* our experts are our legal moves and switches. The adversary is one imaginary opponent per sampled set; its experts are that set's actions.
   - *Each iteration:* every player scores its actions against the others' current mixes and reweights them by MWU.
   - *Result:* our **average** mix converges to a strategy that is safe against every possible set at once (the turn's equilibrium). This replaces "one plan per guess, blended".
   - *By-product:* each imaginary opponent's average mix tells us how a near-optimal player with that set would act. That feeds the memory's update.
3. **Choose the move (MWU over policies, against the real opponent).** The experts are now policies:
   - "play safe" (from step 2);
   - one "exploit σ" per style, which best-responds to what that style predicts.

   We mix their recommendations by their current weights and **sample** our move.
4. **Learn from what happened.** After seeing the opponent's real choice, we score every expert against it and reweight by MWU, and the memory updates by Bayes' rule. These weights persist across turns and across battles:
   - against an unpredictable opponent, the exploit experts lose weight and we stay safe;
   - against a predictable opponent, the matching exploit expert takes over;
   - because "play safe" is always one of the experts, we never fall far below its value.

In a quick experiment on random small games, a blended plan gave up value in about three quarters of cases.

**Gathering and hiding information.**
- **Gathering:** some actions are worth taking for what they reveal (Protect, a scouting switch), so the agent values information explicitly.
- **Hiding:** our own team is hidden too, so the agent tracks what the opponent can infer about *us* and avoids revealing more than necessary. This is a principled form of bluffing.

## 3. Implementation steps and evaluation

**Steps:**
1. A fast open-source battle engine and the poke-env interface, on a local Showdown server.
2. The joint set-and-style memory and its updates.
3. A small value network trained on public replays, with a hand-written evaluation until it is ready.
4. The move-picking loop (the planner and the policy learner), plus the "plan per guess and blend" rule on the same inputs for comparison.
5. Information gathering and concealment.
6. An opponent pool:
   - fixed-style bots: always attack, switch-happy, max-damage, random;
   - Foul Play;
   - an *exploiter* that sees our strategy and best-responds to it;
   - copies of our own bot.

**Evaluation: a "status card"** that says how good the bot is against each kind of opponent:
- **Worst case:** how much the exploiter can beat us, i.e., how exploitable we are.
- **Predictable opponents:** learning curves showing how fast we learn to beat each fixed-style bot, and how much we give up while learning.
- **Equally strong bots:**
  - mirror matches against our own copies (a fair benchmark is 50%) and against weakened versions;
  - a round-robin table over all bots, analyzed as a game in its own right, to see whether our bot is one a rational player would pick.

**Diagnostics.** We also check *why* the card looks the way it does:
- how often blending changes the decision in real positions;
- how quickly the solver settles;
- how accurate and well-calibrated the set and style predictions are;
- how often gathering or hiding information changes the move.

Each component is switched on and off in turn.

**Timeline:**
- **Weeks 1–2:** engine, memory, solver, fixed bots.
- **Weeks 3–4:** value network, information, exploitation, the first two rows of the card.
- **Weeks 5–6:** concealment, the round-robin, write-up.

**Stretch goals:** self-play training, deeper search, and testing whether human players mix their choices the way theory predicts.

## 4. Theoretical explorations and goals

- **Bayesian games and Bayes–Nash equilibrium:** formulating each turn as a Bayesian game (as in auctions, with the hidden set as private information), and why per-set learners recover its equilibrium.
- **No-regret dynamics and the minimax theorem (step 2):**
  - why averaged play converges, and why the most recent play can cycle;
  - whether "optimistic" variants and warm-starting from the last turn speed things up.
- **The cost of blending:** when planning per guess and blending is harmless, and how much it can lose. This is our main theorem target.
- **An error budget:**
  - how mistakes in beliefs, in position evaluation and in the solver add up to total exploitability;
  - how much is lost by remembering each attribute separately instead of the joint set.
- **Value of information:** when information is worth paying for (never, if it wouldn't change our plan).
- **Revealing vs. hiding:** when hiding a move is optimal. This connects to repeated games with incomplete information, and to buyers who hide their values from a learning seller in repeated auctions.
- **Beliefs and play feeding each other:** we infer sets and styles from choices, and choose based on those inferences. Does that loop settle down, or oscillate?
- **Experts and safe exploitation (steps 3–4):**
  - regret bounds against fixed-style opponents, while keeping a guaranteed floor;
  - Bayes' rule as multiplicative weights;
  - across battles, what happens when our learner meets other learners, including itself.

Guarantees concern each turn's game and the round-robin game. Full-battle guarantees are a known hard problem we will discuss, not solve.

## 5. Desired takeaways and further reading

**We want:**
1. **An explainable bot.** Every turn, it can show its beliefs about the opponent's set and style, the mixed strategy it chose, and why.
2. **A clear answer** to whether fixing the blending mistake and reading the opponent's choices actually matter.
3. **A status card** that says how strong the bot is against worst-case, predictable and equally strong opponents, with each row tied to a concept from the course.

Ultimately: an agent that knows when to take risks, randomize, gather or hide information, and exploit a predictable opponent, and that can show how well it does each.

**Readings we plan to do, by course topic:**
- **No-regret learning and minimax:**
  - Freund–Schapire;
  - Hart–Mas-Colell (regret matching);
  - Hartline–Syrgkanis–Tardos (no-regret learning in Bayesian games);
  - optimistic learning (Daskalakis et al.; Syrgkanis et al.);
  - counterfactual regret minimization (Zinkevich et al.).
- **Equilibria in games with hidden information:**
  - sequence-form LPs (Koller–Megiddo–von Stengel);
  - the blending problem (Frank–Basin; Long et al.);
  - safe subgame solving (Brown–Sandholm);
  - DeepStack and ReBeL.
- **Bayesian games and information:**
  - Aumann–Maschler;
  - Bayesian persuasion (Kamenica–Gentzkow);
  - strategic buyers in repeated auctions (Amin et al.);
  - value of information (Gossner–Mertens).
- **Learning against opponents, and evaluating agents:**
  - safe exploitation (Ganzfried–Sandholm);
  - restricted Nash responses (Johanson et al.);
  - Bayesian opponent modeling in poker (Southey et al.);
  - population training (Lanctot et al.);
  - Nash-averaged ratings (Balduzzi et al.).
