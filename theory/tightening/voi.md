Verified 2026-11-16 (theory/tightening/verify/voi/, deterministic leaf); corrections applied.
# Making the value of information measurable: constructions, costs, predictions, results

Status: brainstorm with prototypes, 2026-11-16. Everything marked *measured* was computed in this session with
`engine/.venv/bin/python` on the scripts in `theory/tightening/scripts/` (raw outputs and the summary in
`scripts/out/`); *estimate* / *speculative* are labelled. No project file outside `theory/tightening/` was changed.

**Two headline findings, in order of importance.**
1. *Measured, affects every production result:* the agent's leaf `nash._leaf` → `pe.id(child, 0)` is
   **time-budgeted and sometimes returns a deeper search under CPU contention** (§2.5). Unloaded: 0/480 children change value
   across 5 repeats. With 12 busy processes on 14 cores: 16/240 (1v1) and 17/240 (6v6) children change, with spreads
   up to 8550 and 48 engine units (typical root payoff range: 270). The first parallel run of the exact subgames
   produced payoff ranges up to 17,179 units; the guarded reruns give 229–655. Under load `pe.id(child, 0)` sometimes returns a complete
   DEEPER search (`depth_searched` $\ge2$, or a wrapped i8 value); `depth_searched` does flag it, and terminal
   children then score 100·depth·`battle_is_over`, so values blow up. `engine/nash.py` now has an opt-in
   deterministic leaf (`OURBOT_DET_LEAF=1`, retry until `depth_searched == 1`). Production ran under contention: the Showdown game records show
   foul-play's MCTS threads consuming ≈ 9.6 CPU-s per game on the same host as our agent's `pe.id` calls
   (`stock_mcts_workers_cpu_s` in `AF-s1-g1.json`); Modal worker counts per node are not recorded here.
2. *Measured:* in exact 1v1 two-stage tera-concealment subgames, VOI is computable, real in about a third of
   matchups, but small (median 0, max 5.2% of the payoff range), and the world-only (A3) agent loses almost none of
   it (exploitability median 0, max 3.1%, and > 0.1% of the range in only 2/30 matchups). All of Theorem C's
   identities hold exactly (30/30). Where A3 is wrong at all (3/30 matchups with sandwich gap > 0.1% Δ) it is a
   near-pure mis-*valuation* in one (Mimikyu: gap 14.1, loss 0.08), a TV-driven mis-*decision* in one (Floatzel: gap
   7.0, loss 16.1) and mixed in one (Tauros: 1.6 / 0.7).

## 0. The question and the obstruction

Theorem C (`theory/full_game.md` §6.3): $\delta_C=\varepsilon_s+2\varepsilon_v+2\mathbb ES_t+\eta_{\rm NL}(t)+2\Delta\,\mathrm{TV}$ with
$\eta(t,s',b')=\sum_\theta b'(\theta)V_{t+1}(s',\delta_\theta)-V_{t+1}(s',b')\ge0$ the value to the agent of a perfect
reveal at $(s',b')$; N4 identifies VOI with it. A3 replaces $V_{t+1}(s',b')$ by $\sum_\theta b'(\theta)V_{t+1}(s',\delta_\theta)$,
so inside the agent $\eta\equiv0$ and no probing or concealment can be seen.

**Reframing.** $\eta$ is a property of the *game*, not of the agent: $V_{t+1}(s',\cdot)$ is convex whatever the agent
does (Lemma 7a). The cheapest belief-dependent leaf is therefore not a new value function but *computing
$V_{t+1}(s',b')$ at a child shared by several types*; at the last live stage that is exactly
`hidden_agents.agent_form_lp` over the per-type matrices at the child (`engine/hidden_agents.py:34`). Two conditions
make this cheap:

1. The hidden variable must have **no transition footprint before it is used**, so several types reach the same
   public child. Among the opponent's hidden attributes only the **tera type** qualifies: moves are revealed when used,
   items/abilities change damage (children separate by HP), but the tera type does nothing until the opponent
   terastallizes, and then it is public. *Measured:* 370/509 species have a role with $\ge2$ tera types (402/509 counting the union over roles); 61% of roles (533/873)
   have $\ge2$ (`randbats.sets()`); the subgame assumes the opponent's role is known. So the construction covers most of the population. The solver asserts
   type-invariance of every non-tera child (probabilities and public keys) and it held in every matchup.
2. The horizon must be short enough that stage-1 values are exact: $H=2$ with the engine's own leaf as terminal
   utility, so $\varepsilon_v=0$ *by construction* and the information question is separated from the leaf question.

Direction 1 (belief-dependent leaf) and direction 2 (exact small subgame) are thus the **same construction**:
AF-LP-at-the-child, exact at $H=2$. PBS value nets and reveal penalties approximate it.

## 1. Ranked constructions

| # | Construction | What it measures | Theory tested | Prediction written before the run | Cost | Status |
|---|---|---|---|---|---|---|
| 1 | **Exact one-sided 1v1 tera subgame** (`scripts/tera_subgame.py`): opponent's tera type hidden among its role's `teraTypes`, uniform prior (= the generator's `rng.choice`, so A4 holds); everything else known (one-sided, our team conceded); stage 0 at $s_0$, stage 1 at every public child, terminal = `nash._leaf`; sequence-form LP for $V_0(b_0)$ and $V_0(\delta_\theta)$; $\eta$ at every shared child; the A3 agent's exact exploitability against a type-informed best response | VOI$_{\rm total}=\sum_\theta b_0V_0(\delta_\theta)-V_0(b_0)$ (the subgame's oracle gap); $\eta_{\rm NL}(0)$ (sup over children and a belief grid); sandwich $v_{A3}-V_0\in[0,\eta_{\rm NL}(0)]$; expl(A3); the TV term under the best response; equilibrium tera timing hidden vs known | Theorem C (C.1) with $\varepsilon_s=\varepsilon_v=S=0$, $\eta_{\rm NL}(1)=0$; Lemma 7(a); the proof's sandwich $f_b-\eta_{\rm NL}\le G_b\le f_b$; the $\sum\eta_{\rm NL}$-vs-oracle-gap conjecture at $H=2$ | $\eta\ge0$ everywhere, mostly 0 with a positive tail where our best stage-1 move depends on the opponent's tera; sandwich and (C.1) hold exactly; expl(A3) $\ge0$ and well below $\eta_{\rm NL}(0)$ (a sup); no predicted direction for tera timing | 15–110 s per matchup on one core (300–1800 public children, 2–4 types) | **run**, §2.1 |
| 2 | **Two-sided 1v1 tera subgame** (`scripts/tera_two_sided.py`): both tera types hidden, agent's plan indexed by its own type | $V^{\rm real}_0-\sum_\phi q(\phi)V^{\rm conc}_0(\phi)$: the value of concealing *our* tera; the opponent's VOI in the two-sided game | The two-sided Remark (`full_game.md` §6.3): $\operatorname{expl}^{\rm real}\le\operatorname{expl}^{\rm conc}+(V^{\rm real}_0-V^{\rm conc}_0)$; the concealment value is $\ge0$ exactly | $\ge0$ always; same order as VOI$_{\rm total}$ of option 1 | 35–130 s per matchup ($\le9$ joint types) | **run**, §2.2 |
| 3 | **Per-decision $\hat\eta$ in the live 6v6 engine, no new leaf** (`scripts/eta_hat_engine.py`): at every `agentform:4` decision, $\hat\eta_{\rm set}=\tfrac1K\sum_k{\rm val}(M_k)-v_{AF}$ from the $K$ matrices the agent already builds; $\hat\eta_{\rm tera}$ with worlds = MAP sets varying only the active opponent's tera type | A lower bound in expectation (concavity of $\eta$ in $b$; exact up to $\varepsilon_v$) on $\eta(t-1,s_t,b_t)$ along real play, per decision and summed per game | The conjecture $\sum_t\eta_{\rm NL}\approx$ oracle gap (lower side); how much of the strategy-fusion value is tera | a few % of the payoff range per decision; tera part $\approx0$ | one extra LP per world per decision; 12 games $\approx$ 7 min on 2 cores | **run**, §2.3 |
| 4 | **Belief-dependent leaf inside the live agent** ("public-child AF leaf"): group the $K$ worlds' depth-2 children by public view (`hidden.make_view`) and replace each child's per-world leaf by the AF-LP value over the worlds consistent with that public child ($b'$ from legality and chance likelihoods; A2 for the stage-0 opponent action) | Turns $\eta$ at depth 1 on for the live agent | Corollary C.1 with $\eta_{\rm NL}(0)$ pushed one level down | Gain bounded by the children's $\eta$, which options 1 and 3 put at $\lesssim$ 1–2% of the range in expectation: below the 0.04-score resolution of a 300-pair run (*estimate*) | one LP per public child, 300–1000 per decision at 2–5 ms: 1–3 s per decision in Python, 3–6$\times$ today's 0.5 s. For full-set worlds almost no children are shared (damage differs), so in practice it is the tera leaf of option 1 | not built; §3 says it is not worth it on these numbers |
| 5 | **Tera-reveal statistics from Showdown logs** (`scripts/tera_logs.py`) | P(tera in game), turn of first tera, ours vs foul-play | none: no equilibrium reference, and both bots are A3-type | descriptive only | 0.5 s | **run**, §2.4 |
| 6 | **Oracle-gap decomposition by revealed attribute** (not built): oracle vs agentform with the oracle told only tera types / only items | The game-level value of each hidden attribute; the conjecture's right-hand side split by attribute | tera share small (option 3: $\hat\eta_{\rm tera}\approx0$ at the root) | 100 paired engine games $\approx$ 30 min, must run with $\le3$ workers (§2.5) | *estimate* only |

Ranking rationale: 1 is the only construction in which VOI, $\eta_{\rm NL}$, expl and the TV term are all exactly
computable in the real engine, so Theorem C's hidden-information terms get a numeric check instead of a proxy; 2 is
the only one that measures *our* concealment; 3 is free and runs on real play; 4 is what one would ship if 1 and 3
showed a large effect; 5 and 6 are descriptive.

## 2. Results (*measured*)

Only guarded runs produce numbers (leaf re-evaluated until three evaluations agree exactly; §2.5). The first,
contaminated run (`scratchpad voi/batch_v1_noisy.jsonl`, 33 matchups, 8 workers) is cited once, as evidence of the
artefact. Full summary: `scripts/out/summary_final.txt`.

### 2.1 Exact one-sided tera subgames (30 matchups; 21 with 2 types, 5 with 3, 4 with 4)

Units: engine leaf units; $\Delta$ = payoff range of the subgame's terminal values, quantiles 229 / 403 / **456** / 500 / 655
(min / p25 / median / p75 / max). Selection: 16 further matchups were dropped because a leaf did not stabilise under
load (first guard, 3 workers alongside a stale 8-worker process); the 16 matchups of the final run (2 workers, majority
guard) all succeeded. The 16 dropped matchups all succeed with the deterministic leaf (they were dropped for load, not
for their content). Across all 46: VOI > 0.1%Δ 14/46; VOI max 5.2%; $\eta_{\rm NL}(0)$ median 0.6%, max 6.7%; expl
median 0, max 3.1%, > 0.1%Δ in 3/46 (new: Calyrex-Ice 0.43%); sandwich, (C.1), VOI $\ge0$ and expl $\ge0$ hold 46/46;
sup-form looseness $\ge18.5\times$; $\mathbb E[\mathrm{TV}_1]>0$ in 12/46. The reported 30 were recomputed
bit-identically with the deterministic leaf.

| quantity | median | p75 | p90 | max | /Δ median | /Δ p90 | /Δ max | frac > 0.1% Δ | frac > 1% Δ |
|---|---|---|---|---|---|---|---|---|---|
| VOI$_{\rm total}$ (subgame oracle gap) | 0.00 | 0.73 | 6.01 | 25.92 | 0 | 0.016 | 0.052 | 0.30 | 0.13 |
| $\eta_{\rm NL}(0)$ (sup over children, belief grid step 0.1) | 1.76 | 13.84 | 31.82 | 38.36 | 0.004 | 0.057 | 0.067 | 0.57 | 0.40 |
| $\max_I\eta(0,I,b_0)$ | 1.66 | 12.15 | 27.18 | 31.97 | 0.004 | 0.053 | 0.061 | 0.53 | 0.40 |
| mean over pre-tera children of $\eta(0,I,b_0)$ | 0.02 | 0.88 | 2.82 | 5.73 | 0.0001 | 0.006 | 0.010 | 0.30 | 0.00 |
| $\mathbb E_{\rm BR}[\eta$ along play$]$ | 0.00 | 0.01 | 0.89 | 14.18 | 0 | 0.002 | 0.028 | 0.13 | 0.07 |
| sandwich gap $v_{A3}-V_0$ | 0.00 | 0.00 | 0.36 | 14.10 | 0 | 0.0007 | 0.028 | 0.10 | 0.07 |
| expl(A3) | 0.00 | 0.00 | 0.11 | 16.14 | 0 | 0.0002 | 0.031 | 0.07 | 0.03 |

Final 16-matchup run alone (majority guard, 2 workers, nothing else running): VOI$_{\rm total}$ median 0 / max 25.9
(5.2% Δ), $\eta_{\rm NL}(0)$ median 0.24 / max 32.4 (6.5% Δ), expl(A3) median 0 / max 16.1 (3.1% Δ), > 0.1% Δ in 1/16;
only the VOI and expl maxima come from this run; the $\eta_{\rm NL}$ max of 38.36 is seed 2012 from the 3-worker
run. The combined-30 picture is unchanged.

Identity checks, 30/30 each (the Jensen step is not logged and holds by construction of the LP: min $\eta$ on the
belief grid $\ge-1.1\times10^{-12}$ in 12 matchups; $\eta_{\rm NL}(0)$ is a lower bound on the sup over a 0.1-step belief
grid, so "max 6.7%" is a lower bound): $\eta\ge0$ at every child; VOI$_{\rm total}\ge0$; $0\le v_{A3}-V_0\le\eta_{\rm NL}(0)$
(the proof's sandwich); expl(A3) $\ge0$; Theorem C's (C.1) under the best response,
expl $\le\eta_{\rm NL}(0)+2\Delta\,\mathbb E_{\rm BR}[\mathrm{TV}_1]$; and the sharper along-play form
expl $\le\mathbb E_{\rm BR}[\eta(0,I,b_0)]+2\Delta\,\mathbb E_{\rm BR}[\mathrm{TV}_1]$ (not a theorem; it held
30/30, but it fails without TV in 2/30 matchups at $b_0$ and 3/30 at the true posterior, so "held 30/30" is informative
only in the 23 matchups with TV $=0$; when TV $>0$, $2\Delta\,\mathrm{TV}\ge\sim90$ units). The provable per-stage term from the proof's identity $\Phi_p=\mathrm{Rev}_p-\sum x_an_p\,\eta(t,s',b')$ is
$\mathbb E_\sigma[\eta(t,s_{t+1},p_{h_{t+1}})]$ at the *true posterior*; the check here used $\eta(0,I,b_0)$ at the hard
belief, which coincides with it in the 23 matchups with TV $=0$ and differs in the 7 with TV $>0$. The sequence-form value with a point prior equals backward induction to
$10^{-6}\Delta$ in every matchup (built-in assert). Over matchups with expl $>0$, expl/(bound) = 0.000, 0.001, 0.001,
0.003, 0.029, 0.054: the sup-form bound is loose by $\ge18\times$.

Three objects, not one (pairwise correlations across matchups 0.23–0.35): VOI$_{\rm total}>0.1\%\Delta$ with
$\eta_{\rm NL}(0)=0$ in 4 matchups (Venomoth: VOI 10.7, $\eta_{\rm NL}=0$); $\eta_{\rm NL}(0)>0.1\%\Delta$ with VOI$_{\rm total}=0$
in 10 (Arceus-Ground: VOI 0, $\eta_{\rm NL}$ 38). VOI$_{\rm total}$ is the root's own decoupling (the value of the
information, which Theorem C's benchmark $V_0(b_0)$ already forgoes); $\eta_{\rm NL}(0)$ is the stage-1 continuation
nonlinearity (the $\delta_C$ term), positive in most matchups but reached along play in few
($\mathbb E_{\rm BR}[\eta]$ median 0); expl(A3) is the loss, $\approx0$ almost everywhere.

The TV term is live: in 7/30 matchups (under the script's best-response tie-break; 5/30 under the tie-break that
minimises TV; 4 of the 7 have expl = 0) the type-informed best response to the A3 agent is type-dependent at stage 0
($\mathbb E[\mathrm{TV}_1]$ up to 0.5), and the largest loss is such a case (Floatzel(2 types) vs Braviary-Hisui:
expl 16.1 = 3.1% Δ, $\eta_{\rm NL}(0)$ 31.8, sandwich 7.0, $\mathbb E[\mathrm{TV}_1]=0.5$): a two-type illustration in a real
position, at ≈1/18 of Example C.1's magnitude (C.1 expl/Δ = 4/7 vs Floatzel 0.031).

Concealment behaviour: equilibrium P(tera at stage 0 | type) with the type hidden minus with it known, over 73
(matchup, type) pairs: mean 0.001, $|\text{diff}|>0.1$ in 14% (7% each way). Equilibria are not unique (LP duals), so
the stage-0 argmax comparison eq vs A3 (differs in 1/30) is weak evidence.

Contaminated first run, for contrast only: $\Delta$ quantiles 349 / 3871 / 8904 / 10702 / 17179; the same identity
checks held 33/33 (they are internal to the stored matrices), which is why identity checks alone could not have
caught the artefact.

### 2.2 Two-sided tera subgames (4 matchups, 1–2 workers, majority guard)

| own (types) vs opp (types) | Δ | $V^{\rm real}$ | concealment value | /Δ | opp VOI two-sided | opp VOI one-sided avg |
|---|---|---|---|---|---|---|
| Deoxys(3) vs Talonflame(2) | 396 | 19.71 | 0.00 | 0 | 0.00 | 0.00 |
| Garganacl(2) vs Dugtrio-Alola(2) | 411 | 25.27 | 0.00 | 0 | 0.00 | 0.00 |
| Chesnaught(2) vs Amoonguss(2) | 430 | −48.00 | 0.00 | 0 | 0.00 | 0.00 |
| Drifblim(3) vs Snorlax(2) | 455 | −34.90 | **1.78** | 0.39% | 2.23 | 2.25 |

The value of concealing our tera is $\ge0$ in all four (an exact inequality: the agent can play each conceded-optimal
plan against a type-blind opponent) and 0 in three. A first version of `seqform2` indexed the stage-0 plan with one
shared block for all our types, over-constraining the agent and giving −1.14% Δ on Drifblim; that was a bug, found
because the inequality is exact, and is fixed (per-type stage-0 blocks). The fix moved $V^{\rm real}$ from −41.88 to
−34.90 with $V^{\rm conc}$ unchanged.

### 2.3 $\hat\eta$ along real 6v6 play (12 engine games, `agentform:4:200` self-play, 1003 decisions, 2 workers, guard)

- $\hat\eta_{\rm set}$/range per decision: mean 0.0134, median 0.0043, p90 0.037, max 0.218; exactly 0 in 37% of
  decisions; > 5% of the range in 6%. Per game per side, $\sum_t\hat\eta_{\rm set}/\text{range}$: mean 0.56, median
  0.55, max 1.09 (median 39 turns). These are *range units*, not score; incomparable with the oracle gap ≈ 0.1 score
  without a squash, so no conversion is attempted.
- $\hat\eta_{\rm tera}$/range (592 decisions with an un-tera'd active opponent with $\ge2$ tera types): mean 0.0013,
  median 0, 0 to solver tolerance in 87% (literally 0 in 70%), > 1% in 2.7%, max 0.156. (Slightly inflated at U-turn nodes, where each determinization
  resamples the pending move.)
- Root payoff range: median 268, p99 640, max 652 units (the contaminated 30-game run: max 10,595). The contaminated
  run's statistics were otherwise close (mean 0.0146 / 0.0015), so load noise did not drive these means; ≈ 48
  matrix re-evaluations were needed over ≈ 5,500 matrix evaluations even at 4–5 concurrent processes.

### 2.4 Tera-reveal statistics from Showdown logs (150 games per run, `input_log` `terastallize` commands)

| run:method | P(tera in game) | first tera, median turn | as fraction of game |
|---|---|---|---|
| AF:agentform / AF:stock (foul-play) | 0.55 / 0.90 | 16.5 / 16.0 | 0.51 / 0.48 |
| AF16:agentform / AF16:stock | 0.50 / 0.93 | 11.0 / 13.0 | 0.36 / 0.41 |
| E2:hedge / E2:stock | 0.53 / 0.95 | 13.0 / 14.0 | 0.51 / 0.44 |
| BH:hedge_bayes / BH:stock | 0.55 / 0.91 | 12.0 / 16.0 | 0.41 / 0.53 |

Command formats are identical on both sides (checked), so our bots genuinely tera in about half the games and
foul-play in nearly all. **This is not concealment evidence and points the other way:** in the one-sided model our
tera is conceded, an A3 agent has zero concealment cost, so a concealment story would predict it teras *more* than an
equilibrium player, not less. Uncontrolled confounds: our bots switch more (685 vs 523 commands in 60 games), game
length, which side is losing. Descriptive only.

### 2.5 The leaf is load-dependent (`scripts/leaf_nondet.py`)

`nash._leaf(child)` calls `pe.id(child, 0)` twice (`engine/nash.py:96–119`), i.e. a time-budgeted iterative-deepening
search with a 0 ms budget. *Measured* on 240 children of six 1v1 states and 240 children of six 6v6 states (5
evaluations each): unloaded, 0/480 children change value. With 12 busy processes on the 14-core machine: 16/240 (1v1)
and 17/240 (6v6) change, spreads among the changed ones median 6.3 / 8.0 and max 8550 / 48 units (typical root payoff
range 270). The same child object gave `full_matrix` results differing by 450 units on two consecutive calls while the
8-worker first batch was active. Verified mechanism: under load `pe.id(child, 0)` sometimes returns a complete DEEPER search (`depth_searched` $\ge2$,
or a wrapped i8 value); `depth_searched` does flag it, and terminal children then score 100·depth·`battle_is_over`, so
values blow up. `engine/nash.py` now has an opt-in deterministic leaf (`OURBOT_DET_LEAF=1`, retry until
`depth_searched == 1`).
Consequences: (i) any result produced with several workers per node (the 8-worker engine runs here; Modal runs if
they shared nodes, which is not recorded) has an unmeasured leaf-noise component, and the Showdown runs demonstrably
shared the host with foul-play's MCTS threads; (ii) the guarded reruns here are the first with a certified-deterministic leaf;
(iii) a cheap follow-up for the orchestrator (not run): check whether the NaN count of `pe.id(child, 0).matrix` rises
under load; if so it is a free production guard. Scripts here re-evaluate until three evaluations agree exactly
(`tera_subgame.stable_full_matrix`); `eta_hat_engine.stable_engine_matrix` uses the weaker two-consecutive-agree rule.

## 3. What the results say about the theory

- **N4 / proof goal (iii), delivered as an experiment.** VOI is exactly computable in the real engine at $H=2$ and is
  > 0.1%Δ in 9/30 tera subgames (strictly positive in 11/30; max 5.2% Δ). The identity VOI $=\eta$ holds at the root by definition; the operative
  statement is the sandwich $V_0\le v_{A3}\le V_0+\eta_{\rm NL}(0)$, verified 30/30.
- **Theorem C's five-term bound is confirmed and very loose.** (C.1) held in every matchup, and the realised loss is
  $\le5.4\%$ of the bound. The along-play form with $\mathbb E_{\rm BR}[\eta(0,I,b_0)]$ in place of the sup also held
  30/30 and is tighter by an order of magnitude; proving it (the proof already has $0\le\eta\le\eta_{\rm NL}$ per term)
  is a one-line tightening worth doing.
- The four-term form (no TV) held in 30/30 (46/46). TV is needed only in the along-play form: in Floatzel and
  Calyrex-Ice the loss sits entirely on TV, because $\mathbb E_{\rm BR}[\eta]$ at the true posterior is 0.
- **The $\eta_{\rm NL}$-vs-oracle-gap conjecture, tested at $H=2$ in 1v1.** The *sup* form does not track VOI$_{\rm total}$
  (correlation 0.35; 4 matchups with VOI $>0$, $\eta_{\rm NL}=0$ and 10 the other way). The *along-play* form as written
  in `theory_map.md` (typical play, TV included) is mixed: $\mathbb E_{\rm BR}[\eta(0,I,b_0)]$ alone correlates 0.82
  with VOI$_{\rm total}$ (the earlier 18 / 6 / 3+3 contingency is unverifiable; reproducible counts at 0.1%Δ: 20 both below / 3 both above / 6 VOI only / 1 $\mathbb E_{\rm BR}[\eta]$ only; at $10^{-6}$: 18 / 7 / 4 / 1); adding $2\Delta\,\mathbb E[\mathrm{TV}_1]$
  drops the correlation to 0.17 because the TV term is large exactly where VOI is not. Register the along-play
  $\eta$ part as supported at $H=2$ and the TV part as a separate object.
- **A3's error, where it exists (3/30 with sandwich gap $>0.1\%\Delta$):** one near-pure mis-valuation (Mimikyu), one
  TV-driven mis-decision (Floatzel), one mixed (Tauros); in the other 27 both are $\approx0$. "Mis-valuation rather
  than mis-decision" describes one matchup, not a rule.
- **The two-sided Remark's quantity is small**: concealing our tera is worth 0 in 3/4 matchups and 0.4% Δ in the fourth.
- **In 6v6 play the strategy-fusion value is ≈ 1.3% of the range per decision and is almost entirely about sets,
  not tera** ($\hat\eta_{\rm tera}$ is 0 to solver tolerance in 87% of decisions (literally 0 in 70%)).

## 4. Rejected or deferred ideas

- **PBS / range-conditioned value network (DeepStack/ReBeL leaf).** The right object, but training data would need the
  exact values of option 1 at scale; option 1 is its exact special case at $H=2$ and says the target is small.
- **Reveal-penalty heuristic** (leaf $-\lambda\cdot\mathbf 1[\text{we revealed}]$). No theorem links $\lambda$ to a
  Theorem C term; its effect size is $\lambda$. The principled quantity (option 2) is 0–0.4% Δ.
- **K-world opponent-view leaf** ($x^*(b_{\rm us})^\top M\,y^*(b_{\rm them})$). Level-1 reasoning, not a value; not
  convex in either belief; $K\times$ the leaf cost.
- **Full-set belief-dependent leaf in 6v6.** Children of different set-worlds almost never share a public state, so
  there is nothing to group without a public-state abstraction, which is the expensive part.
- **Equilibrium tera-timing reference from logs.** Both bots are A3-type; their difference measures search style.
- **Horizon $>2$ exact subgames.** $\approx300\times$ per extra stage; $H=3$ is 1–2 h per matchup, and VOI is already
  resolved at $H=2$.
- **Option 4 (live belief-dependent leaf).** Deferred on evidence: the children's $\eta$ along play is median 0 and
  the A3 loss is $\le0.1\%$ Δ in 28/30 matchups; a 3–6× slower agent for an effect below run resolution.

## 5. Recommendation

1. **First, the leaf.** Before any further Modal run, decide how to make `pe.id(child, 0)` deterministic (NaN-count
   guard, re-evaluation until agreement, or a fixed-depth evaluator) and re-measure the oracle gap and E1 with
   $\le3$ workers per node. This changes the ranking of everything else: the $\varepsilon_v$ term now has a measured
   load-dependent component.
2. **Report option 1 as the VOI experiment** the proposal asked for: exact VOI in 30 real positions, sign and size,
   Theorem C verified with all other terms zero, A3's loss quantified. Register the along-play tightening of (C.1)
   (true-posterior form) as a small theorem. The `report/current_report.tex` line "no experiment possible without belief-dependent leaves"
   can be replaced by "exact at $H=2$ for the tera type; §…".
3. **Do not build option 4.** What would change this: a matchup family with $\mathbb E_{\rm BR}[\eta]$ above ~5% Δ
   (none in 30), or a 2v2/3v3 version of option 1 showing growth with the number of live mons (cost ≈ 10× per extra
   mon; not run).

## 6. Reproduction

```
cd theory/tightening/scripts
P=../../../engine/.venv/bin/python
$P tera_subgame.py   --n 16 --seed 3 --workers 2 --step 0.1 --out batch3.jsonl      # 16 matchups, ~8 min
$P tera_two_sided.py --n 4  --seed 5 --workers 2 --out two_sided3.jsonl              # 4 matchups, ~3 min
$P eta_hat_engine.py --games 12 --workers 2 --out eta_hat2.jsonl                     # ~7 min
$P tera_logs.py --runs AF AF16 E2 BH --max 150
PYTHONPATH=../../../engine $P leaf_nondet.py 0                                       # run once idle, once with 12 busy procs
$P summarize_voi.py --subgame out/guarded_all.jsonl --two out/two_sided3.jsonl --eta out/eta_hat2.jsonl
```
Keep `--workers` $\le2$–3 on a 14-core machine and nothing else running; every record carries `leaf_retries`.
`out/guarded_all.jsonl` = the 14 successes of the 3-worker run (seed 2) + the 16 of the 2-worker run (seed 3).
