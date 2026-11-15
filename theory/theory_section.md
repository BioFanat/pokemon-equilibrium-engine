# Theory: how fast the per-turn agent approaches the full-game equilibrium

Simplified statement for the write-up. Full proofs, the general case and counterexamples are in
`full_game.md`; the checker's report is `CHECK.md`; Lean 4 proofs are in `lean/FullGame/Basic.lean`.

## Setting

Pokémon battles are a finite-horizon, two-player, zero-sum stochastic game with simultaneous
moves and chance. Payoffs are $+1$ (win), $0$ (draw or 300-turn cap) and $-1$ (loss), so every
stage matrix has payoff range at most $\Delta = 2$.

Each turn the agent builds a stage matrix $\hat Q$ (one simulated turn plus a leaf value) and runs
$T$ iterations of RM+ on it. Under hidden information it samples $K$ opponent worlds and solves
the agent-form game. It then plays the resulting strategy. Notation:

| Symbol | Meaning |
|---|---|
| $n$ | most legal actions for either side in a turn (≤ 13 here) |
| $\varepsilon_v$ | leaf error: $\lVert \hat Q - Q^\star \rVert_\infty$, the estimated vs true stage matrix |
| $\bar H$ | expected game length against a best-responding opponent |
| $\mathrm{expl}(\pi)$ | game value minus the agent's worst-case expected payoff |

## Assumptions

- **(A1) Non-revealing opponent.** The opponent's moves don't depend on its hidden sets, so
  evidence from its behaviour doesn't move our belief away from the truth. This holds trivially
  under full information.
- **(A2) World-only leaves.** The value after the lookahead depends on the true position, not
  on what each side believes. Concealment then has no value.
- **(A3) Exact prior.** Opponent sets are drawn from the Random Battle generator, which we sample
  from directly.

## Results

**Lemma 1 (one turn).** If the agent's strategy is within $\varepsilon_s$ of optimal in $\hat Q$,
it is within $\varepsilon_s + 2\varepsilon_v$ of optimal in the true stage game. The factor 2 is
tight.

**Theorem 1 (whole game).** Per-turn losses add up over the turns the game lasts:
$$\mathrm{expl}(\pi) \;\le\; \bar H\,(\varepsilon_s + 2\varepsilon_v).$$
$\bar H$ is the expected length when the opponent is a best response. It can't be replaced by
the length of typical or self-play games: an opponent can choose to prolong the game, and a
counterexample has a minimax-optimal opponent end at once while the exploitability is still
$H(\varepsilon_s + 2\varepsilon_v)$.

**Proposition 2 (solver).** RM+ with alternating updates and linear averaging, exactly as in
`nash.rm_plus`, gives $\varepsilon_s \le 2\Delta(\sqrt n + \sqrt m)/\sqrt T \le 8\sqrt{n/T}$.

**Proposition 3 (sampling worlds).** Solving $K$ sampled worlds instead of the exact belief adds,
in expectation, at most $2\Delta\sqrt{2(n\log K + \log 2)/K} \approx 4\sqrt{2n\log K/K}$ per turn,
for $K \ge 9$.

**Headline bound.** Under (A1)–(A3):
$$\boxed{\;\mathrm{expl}(\pi)\;\le\;\bar H\left(\underbrace{8\sqrt{\tfrac{n}{T}}}_{\text{solver}}
\;+\;\underbrace{2\,\varepsilon_v}_{\text{leaf value}}
\;+\;\underbrace{4\sqrt{\tfrac{2n\log K}{K}}}_{\text{sampled worlds}}\right)\;}$$
Under full information the last term is 0.

**What it says:**
- **Solver:** exploitability falls like $1/\sqrt T$ in the worst case.
- **Sampled worlds:** it falls like $\sqrt{\log K/K}$ in the number of worlds.
- **Leaf value:** this term doesn't shrink with compute at all. Only a better leaf, which is
  where RL comes in, removes it. Deeper exact search keeps $\varepsilon_v$ at the leaf error
  (no contraction without discounting), so depth alone doesn't help either.

## The general case (drop A1 and A2)

Without the assumptions, each turn adds two terms that more computation never removes:
$$\mathrm{expl}(\pi)\;\le\;\bar H\left(8\sqrt{\tfrac{n}{T}} + 2\varepsilon_v
+ 4\sqrt{\tfrac{2n\log K}{K}} + \eta_{\mathrm{NL}} + 4\,\mathrm{TV}\right).$$
- $\eta_{\mathrm{NL}}$ is the **non-locality gap**: the belief-averaged per-world value minus
  the belief-dependent value. It is 0 under (A2) and positive when concealing or revealing
  information matters.
- $\mathrm{TV}$ is the **belief-model error**: the total-variation distance between the true
  posterior under the opponent's actual strategy and our hard-evidence belief. It is 0 under (A1).

Both terms are needed. A 3-type, two-turn example has $\mathrm{expl} = 1$ while
$\sum_t \eta_{\mathrm{NL}}(t) = 3/7$, while the solver, leaf and sampling terms are all 0 (the
agent solves the exact belief game). So the formula without TV fails against a revealing
opponent; the full five-term bound still holds. Removing these terms is what public-belief-state methods do: values
that take beliefs as input (CFR-D, DeepStack, ReBeL) remove $\eta_{\mathrm{NL}}$.

## Tightness and what we measured

| Quantity | Bound | Measured |
|---|---|---|
| RM+ solver gap | $\propto T^{-1/2}$ | $\approx T^{-1.3}$ on 35 real stage matrices (median slope −1.26 at seed 0; −1.24 to −1.39 across seeds 1–3; `report/figures/data/rmplus_gap.json`); on synthetic random matrices 0.12–0.20 of the bound; on the 35 real stage matrices the gap is at most 0.04 of the proved bound |
| Sampling term ($n=3$, $K=25$) | $0.91\Delta$ | $0.054\Delta$ |

The bound is loose in constants. It only becomes informative once $K/\log K \gtrsim 2n$.
- **Leaf value:** time-matched MCTS beats depth-2 matrix play (X1), consistent with
  $\varepsilon_v$ being the dominant term.
- **Floor:** the 0.1 oracle gap is a candidate for the size of $\eta_{\mathrm{NL}} + 4\,\mathrm{TV}$
  summed over a game (conjecture).

## Fit form for the Modal runs

$$\mathrm{expl}(T, K, \text{leaf}) \approx \bar H\,\bigl(a\,T^{-\alpha} + b\sqrt{\log K/K}
+ c\,\varepsilon_v\bigr) + d.$$
- **Measurement:** exploitability is measured by RL best response (a lower bound) over a grid of
  T, K and leaf variants (heuristic vs value net, depth 1 vs 2). $\bar H$ is the mean game length
  against the exploiter.
- **Predictions:** the theory predicts $\alpha \ge 1/2$ and $c \le 2$. $d > 0$ is the floor from
  $\eta_{\mathrm{NL}}$ and TV.

## Verification status

| Claim | Status |
|---|---|
| Lemma 1 | Lean (`lemma1_one_step`), with the factor-2 tightness instance |
| Value is 1-Lipschitz (used for depth) | Lean (`val_lipschitz`, `exists_val`) |
| Theorem 1, backward induction | Lean in abstract form (`theoremA_abstract`); the link to the game model checked by reading |
| $\bar H$ = best-response length; counterexample for typical length | Checked by reading and brute force |
| Proposition 2 for the code's RM+ | Key step in Lean (`rmplus_improvement`, Burch–Moravčík–Schmid 2019 Thm 5); constant derived for the code's indexing; numerically 0.12–0.20 of the bound |
| Proposition 3 | Checked by reading and simulation |
| General case (five terms) | Checked by reading; 3-type example brute-forced |
| $\sum_t (\eta_{\mathrm{NL}} + 4\,\mathrm{TV}) \approx$ oracle gap | Conjecture |
