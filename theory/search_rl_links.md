# Links: no-regret per-turn play, MCTS, and RL best responses (L0–L6)

Draft 2026-11-15. Status of every claim is in the table at the end; nothing here is "verified" until
that table says so. Model and notation follow `full_game.md` §1 and `regret_theory.md` §0; the per-turn
theorem is `full_game.md` Theorem A (called Theorem 1 in `theory_section.md`).

## 0. Setting, and the facts about the code that the statements use

**Game.** $G$ is a finite-horizon two-player zero-sum stochastic game with simultaneous moves and chance. It has stages $t=0,\dots,H-1$, finite states, and action sets $A(s)$ (agent, maximiser) and $B(s)$ (opponent). The terminal payoff is $u\in[-1,1]$, and reaching stage $H$ without a terminal state pays $0$ (the cap). $Q^*_t(s,a,b)=\mathbb E[V^*_{t+1}(s')\mid s,a,b]$ and $V^*_t(s)=\operatorname{val}Q^*_t(s)$, with $V^*:=u$ on terminal states and $V^*_H:=0$. $\tau\le H$ is the first stage at which the game is over (terminal or cap). An agent $\pi$ maps histories $h_t$ to $x_t\in\Delta(A(s_t))$, an opponent $\sigma$ maps them to $y_t\in\Delta(B(s_t))$, and $V(\pi,\sigma)=\mathbb E_{\pi,\sigma}[u(s_\tau)]$ (cap payoff $0$). Exploitability is $\operatorname{expl}(\pi)=V^*_0(s_0)-\inf_\sigma V(\pi,\sigma)$. Here $\Delta=2$ is the payoff range on the $[-1,1]$ scale.

**Notation.**

| Symbol | Meaning |
|---|---|
| $e_t$ | stage exploitability of the agent's stage strategy **in the true** $Q^*$: $e_t=V^*_t(s_t)-\min_b x_t^\top Q^*_t(s_t)e_b\ \ (\ge0)$ |
| $\ell_t(x,y)$ | stage loss relative to the value: $V^*_t(s_t)-x^\top Q^*_t(s_t)\,y$ |
| $\sigma_{\rm BR}$ | a best response to $\pi$ (attains $\inf_\sigma V(\pi,\sigma)$; exists by backward induction over the finite history tree) |
| $\sigma_g$ | the **myopic $Q^*$-exploiter**: plays $b_t\in\arg\min_b x_t^\top Q^*_t(s_t)e_b$ (any tie rule) |
| $\bar H(\sigma)$ | $\mathbb E_{\pi,\sigma}[\tau]$, the expected game length against $\sigma$; $\bar H=\bar H(\sigma_{\rm BR})$ as in Theorem A |
| $\varepsilon_s,\ \varepsilon_v$ | solver gap and leaf (stage-matrix) error, as in Theorem A |

**Code facts used below** (read from source on 2026-11-15).

| Component | What the code does | Where |
|---|---|---|
| Our agent | each turn: `nash.full_matrix` builds the one-turn matrix with the `'mid'` leaf; `rm_plus(A, 1000)` (alternating, linear averaging); play the average, its argmax, or the Hedge mixture of experts | `engine/nash.py:122`, `engine/exp3_live.py:50` |
| poke-engine MCTS | **decoupled UCB1**: at each node each side independently picks $\arg\max_i\ \bar r_i+\sqrt{2\ln N_{\rm node}/n_i}$. Unvisited options get $+\infty$. Rewards are in $[0,1]$: side one is credited `score` and side two `1 − score` | poke-engine 0.0.48 (uv sdist cache; copy at `scratchpad/leaf/src/mcts.rs`) `maximize_ucb_for_side`, `ucb1`, `backpropagate` |
| poke-engine leaf | no rollout. A newly expanded node gets `sigmoid(0.0125·(eval(s) − eval(root)))`; terminals get exactly $1$ or $0$ | `mcts.rs` `rollout`, `sigmoid` |
| foul-play move rule | sums visit shares over the sampled worlds, weighted by sample chance; keeps the moves with share $\ge0.75\times$ the top share; samples among them proportionally to share | foul-play source checked out at `/private/tmp/fpsrc` (not in the repo): `fp/search/main.py:16` `select_move_from_mcts_results` |

Leaf-curve consistency (an identity, not a theorem): the project's fixed leaf $\tanh(h/160)\in[-1,1]$ maps to $[0,1]$ as $\tfrac12(1+\tanh(h/160))=\operatorname{sigmoid}(h/80)=\operatorname{sigmoid}(0.0125\,h)$. This is poke-engine's curve exactly, except that poke-engine subtracts the **root's** evaluation, so its leaf is a function of (root, $s$), not of $s$ alone. Every per-turn statement below is applied per root, which Theorem A allows (budgets may vary with $(t,s)$; `full_game.md` A1).

---

## L0. One telescoping identity behind all three formulations

**Lemma L0 (potential telescoping).** Let $\Phi_t(h_t)$ be any real function of histories for $t\le H$. Require $\Phi=u$ when $s_t$ is terminal and $\Phi_H=0$ for live histories at the cap. Write $Q[\Phi]_t(h,a,b)=\mathbb E[\Phi_{t+1}(h')\mid h,a,b]$. Then for every (history-dependent) $\pi,\sigma$:
$$\Phi_0(h_0)-V(\pi,\sigma)=\mathbb E_{\pi,\sigma}\Big[\sum_{t<\tau}\big(\Phi_t(h_t)-x_t^\top Q[\Phi]_t(h_t)\,y_t\big)\Big].$$

*Proof.*
1. On each path, $\Phi_0(h_0)-\Phi_\tau(h_\tau)=\sum_{t<\tau}(\Phi_t(h_t)-\Phi_{t+1}(h_{t+1}))$, and $\Phi_\tau(h_\tau)$ equals the payoff (terminal $u$ or cap $0$) by the boundary conditions.
2. For $t<\tau$, $a_t\sim x_t$ and $b_t\sim y_t$ are independent given $h_t$ (simultaneous moves: no shared randomness, and neither side sees the other's realised action this turn; this is the only assumption beyond the boundary conditions), so $\mathbb E[\Phi_{t+1}(h_{t+1})\mid h_t]=x_t^\top Q[\Phi]_t(h_t)y_t$.
3. Take expectations of step 1 and apply step 2 term by term (tower rule; $\tau\le H$, so the sum has at most $H$ terms). $\square$

Each formulation is L0 with a different potential.

| Potential $\Phi$ | Reading | Result below |
|---|---|---|
| $V^*$ (game value) | exploitability = expected sum of per-turn losses relative to the value; Theorem A is its upper half | L1, L2 |
| $V^{\pi,\sigma_k}$ (value of the current opponent policy in the MDP $M_\pi$) | Kakade–Langford performance-difference lemma, the engine of policy-gradient / NPG analyses | L3 |
| $\hat V$ (a learned or heuristic leaf) | exploitability bounded by the minimax TD (Bellman) residual of the leaf | L6 |

---

## L1. Exploitability is a sum of per-turn stage exploitabilities, sandwiched

**Theorem L1.**
(a) *Identity.* For all $\pi,\sigma$: $V^*_0(s_0)-V(\pi,\sigma)=\mathbb E_{\pi,\sigma}\big[\sum_{t<\tau}\ell_t(x_t,y_t)\big]$. Hence $\operatorname{expl}(\pi)=\sup_\sigma\mathbb E_{\pi,\sigma}[\sum_{t<\tau}\ell_t]$.
(b) *Sandwich.* $\displaystyle\mathbb E_{\pi,\sigma_g}\Big[\sum_{t<\tau}e_t\Big]\ \le\ \operatorname{expl}(\pi)\ \le\ \mathbb E_{\pi,\sigma_{\rm BR}}\Big[\sum_{t<\tau}e_t\Big].$
(c) *Theorem A as a corollary.* If $e_t\le\varepsilon_s+2\varepsilon_v$ at every live state (Lemma 1 of `full_game.md`), the upper side is $\le\bar H(\varepsilon_s+2\varepsilon_v)$.

*Proof.*
(a) Apply L0 with $\Phi=V^*$; $Q[V^*]=Q^*$ by definition.
(b) *Lower.* Under $\sigma_g$, $y_t=e_{b_t}$ with $b_t$ minimising $x_t^\top Q^*_t e_b$, so $\ell_t=e_t$. Also $\operatorname{expl}(\pi)\ge V^*_0-V(\pi,\sigma_g)$. *Upper.* $\operatorname{expl}(\pi)=V^*_0-V(\pi,\sigma_{\rm BR})=\mathbb E_{\pi,\sigma_{\rm BR}}[\sum\ell_t]$ by (a), and $\ell_t(x,y)\le e_t$ pointwise because $x^\top Q^*y\ge\min_bx^\top Q^*e_b$. *Sign.* $e_t\ge0$ because $\min_bx^\top Q^*e_b\le\operatorname{val}Q^*=V^*_t$.
(c) $\mathbb E_{\pi,\sigma_{\rm BR}}[\sum_{t<\tau}e_t]\le(\varepsilon_s+2\varepsilon_v)\,\mathbb E[\tau]$. $\square$

**Remarks.**
1. Both sides of (b) sum **the same per-turn quantity**; only the trajectory distribution differs (myopic vs farsighted opponent). The upper side is Theorem A, and the lower side is what a one-turn white-box exploiter with exact $Q^*$ certifies. Our `wb2` exploiter is a depth-2 version of $\sigma_g$ with $\hat Q$ in place of $Q^*$, so it is not literally $\sigma_g$; its certificate comes from R5, not from L1.
2. **The gap can approach the whole payoff range $\Delta=2$** (Example L1b, $H\ge2$). At $s_0$ the agent is forced and the opponent has two actions. $b_1$ ends the game with payoff $1-\delta$ ($0<\delta<1$). $b_2$ moves deterministically to $s_1$, where the agent has a safe row paying $+1$ against every column and a bad row paying $+1$ / $-1$, and the agent plays the bad row. Then $V^*(s_1)=1$, $Q^*_0(s_0)=(1-\delta,1)$, $V^*_0=1-\delta$ and $e_0=0$. $\sigma_g$ picks $b_1$, so the lower side is $0$. $\sigma_{\rm BR}$ plays $b_2$ and then the $-1$ column, so $\operatorname{expl}=2-\delta$. The upper side is $e_0+e_1=2$. (With matching pennies at $s_1$ and $b_1$ paying $-\delta$, the same pattern gives a gap of $1-\delta$, half the range.)
3. *RL reading and prior art.* With a trivial opponent, (a) is Kakade & Langford's (2002) performance-difference lemma (their Lemma 6.1, applied with the optimal policy as the reference). With an opponent, a $\sup_\sigma$ appears, and (b) brackets it. The upper side has discounted precedents: Wei, Lee, Zhang & Luo (COLT 2021, Lemma 32) bound the duality gap by $\tfrac{2}{1-\gamma}\max_s$ of the $Q^*$-stage gap, and Daskalakis, Foster & Golowich (2020, Lemma 3) give a Markov-game performance-difference lemma. Ours is the finite-horizon, trajectory-weighted refinement. The exact identity (a) and the myopic lower side with Example L1b were not found in the papers read (`lit/links_attribution.md`); that is not proof of novelty.

---

## L2. Hedge with an equilibrium expert is safe in the full game (and RL is the same theorem with a bigger policy class)

**Setting (idealised `exp3_live.Learner`).**
- $G$ games are played in sequence against an opponent that may adapt within and across games and may see the learner's mixture $p_i$ before it moves, **but not the realised action $a_i$** (L0 needs $a_i\perp b_i$ given the history; an opponent that sees $a_i$ breaks (b) in simulation).
- Turns are numbered globally $i=1,\dots,T$, where $T$ is random and $T\le GH$.
- At turn $i$ the learner sees a matrix $\hat A_i\in[-1,1]^{n\times m}$ (clipped to the payoff range, which keeps $\lVert\hat A_i-Q^*_{t_i}(s_i)\rVert_\infty\le\varepsilon_v$), and $k$ expert strategies $q_{e,i}$.
- It plays $p_i=\sum_ew_{e,i}q_{e,i}$, where $w_i$ is anytime-FTRL Hedge (the code's indexing) on the **normalised** past rewards $(q_{e,j}^\top\hat A_je_{b_j}+1)/2\in[0,1]$.

**Assumption (H).** Pathwise regret: for every $e$, $\sum_{i\le T}(q_{e,i}-p_i)^\top\hat A_ie_{b_i}\le R(T)$, with $R$ nondecreasing. With the normalised rewards above, R2a rescaled gives $R(T)=2\big(1.5\sqrt{T\ln k/2}+\sqrt{\ln k/8}\big)$. Applying the code's step $\eta_t$ to raw $[-1,1]$ rewards does **not** satisfy this $R$ (simulated regret/$R$ reaches 1.32); its provable bound is $\ln k/\beta_T+\tfrac12\sum_t\beta_t\approx3.18\sqrt{T\ln k}$. With an unclipped $\hat A$, rewards leave the range and $R$ fails as well (ratio 1.30 at $\varepsilon_v=0.4$). The code's fixed normaliser $\bar\Delta=400$ and clipping are handled by E2′ and are not modelled here.

**Theorem L2.** Let $\hat\ell_i(q,b)=\operatorname{val}(\hat A_i)-q^\top\hat A_ie_b$.
(a) $\displaystyle\mathbb E\Big[\sum_{g=1}^G\big(V^*_0(s^g_0)-u_g\big)\Big]\le\mathbb E\Big[\min_e\sum_{i\le T}\hat\ell_i(q_{e,i},b_i)\Big]+\mathbb E[R(T)]+2\varepsilon_v\,\mathbb E[T].$
(b) *Safety.* If some expert $e^\star$ is $\varepsilon_s$-optimal in every $\hat A_i$ ("Equilibrium, sampled"), then $\mathbb E[\sum_g(V^*_0-u_g)]\le(\varepsilon_s+2\varepsilon_v)\mathbb E[T]+\mathbb E[R(T)]$. The average per-game loss to the value is $\le\bar L(\varepsilon_s+2\varepsilon_v)+\mathbb E[R(T)]/G$, where $\bar L=\mathbb E[T]/G$.
(c) *Exploitation* (a restatement of (a), read from the gain side). $\mathbb E[\sum_g(u_g-V^*_0)]\ge\mathbb E[\max_e\sum_i(-\hat\ell_i(q_{e,i},b_i))]-\mathbb E[R(T)]-2\varepsilon_v\mathbb E[T]$. The learner banks the best expert's per-turn gains along its own trajectory, minus regret and leaf error.
(d) *Persistent vs reset (comparison of bounds).* Resetting the weights at every game start replaces $R(T)$ by $\sum_gR(T_g)$, where $T_g$ is game $g$'s number of non-forced turns. With $R(T)=c_1\sqrt T+c_0$, subadditivity of $\sqrt\cdot$ gives $\sum_g\sqrt{T_g}\ge\sqrt{\sum_gT_g}$. (Reset compares against the best expert per game, a stronger comparator, so (d) is about bounds, not realised outcomes.) For $G$ games of length $L$, the per-game regret overhead is $c_1\sqrt L+c_0$ (reset) vs $(c_1\sqrt{L}+c_0/\sqrt G)/\sqrt G$ (persistent).

*Proof.*
1. *Value side.* Apply L1(a) to each game with $x=p_i$. Then $\sum_g(V^*_0-\mathbb E u_g)=\mathbb E\sum_i\ell_i(p_i,y_i)=\mathbb E\sum_i\big(V^*(s_i)-p_i^\top Q^*_ie_{b_i}\big)$ (tower rule, $b_i\sim y_i$ given the history).
2. *Leaf.* $\big|\big(V^*(s_i)-p^\top Q^*_ie_b\big)-\hat\ell_i(p,b)\big|\le|V^*(s_i)-\operatorname{val}\hat A_i|+|p^\top(\hat A_i-Q^*_i)e_b|\le2\varepsilon_v$, because val is 1-Lipschitz in sup-norm (`full_game.md` Lemma 2(a)).
3. *Regret.* By (H), $\sum_i\hat\ell_i(p_i,b_i)\le\sum_i\hat\ell_i(q_{e,i},b_i)+R(T)$ for every $e$, pathwise. Take $\min_e$, then expectations.
4. (b) $\hat\ell_i(q_{e^\star,i},b)\le\varepsilon_s$ for all $b$, by $\varepsilon_s$-optimality. (c) Negate steps 1–3. (d) Apply (H) per game.
5. *Forced turns.* With one agent option, $\ell_i=V^*-Q^*(a,b)\le0$, so leaving forced turns out of $T$ (as the code does) keeps the bounds valid. $\square$

**Remarks (the RL link).**
1. *Experts in every state.* Run L2 with the pure actions as experts and **separate weights per state** $(t,s)$. The proof of (a) and (c) is unchanged ((b) needs an $\varepsilon_s$-optimal expert, which no pure action is in general), with regret $\sum_sR(T_s)\le c_1\sqrt{|S_T|\,T}+c_0|S_T|$ over the $|S_T|$ distinct states visited (Cauchy–Schwarz).
   - With one shared fixed $\eta$ the product weights factorise, so this is exactly Hedge over the class of all Markov policies: $\ln|\Pi|=|S|\ln n$ vs $\ln k$ for our tied 3-expert mixture. With per-state anytime steps it is a different algorithm with the same leading-order bound.
   - This per-state Hedge is Even-Dar, Kakade & Mansour (2009), and the NPG update is the same per-state Hedge on advantages (Agarwal et al. 2021).
   - In Pokémon almost every visited state is new ($|S_T|\approx T$), so per-state Hedge's regret **bound** is linear without generalisation. That is the formal reason RL needs function approximation, and why our 3-expert learner learns within tens of games.
2. L2 uses the potential $V^*$ and gives safety relative to the game value. NPG's convergence to a best response uses the potential $V^{\pi,\sigma_k}$ (L3). Same update, different potential.
3. *Prior art.* The idea is known. Ganzfried & Sandholm (2015, §6.4) note that no-regret selection between an equilibrium and an exploiter is "safe in the limit" but not in finitely repeated games (their safety definition requires zero slack per period). McCracken & Bowling (2004) call it Safe Policy Selection (abstract only), and the matrix-game core is Hannan consistency. What L2 adds is the quantified finite-sample slack $R(T)+2\varepsilon_v\mathbb E[T]$, the extension to stochastic games with a leaf through L1, and (d).

---

## L3. An RL best response is Hedge in every state of the MDP $M_\pi$

**Setting.**
- Fix a Markov agent $\pi$ (stage strategy $x_t(s)$). The opponent then faces the finite-horizon MDP $M_\pi$ with state $(t,s)$, action $b$, transition $\sum_ax_t(s)(a)P(\cdot\mid s,a,b)$ and loss equal to the agent's payoff.
- For an opponent policy $\sigma$, write $Q^{\sigma}_t(s,b)=x_t(s)^\top Q^{\pi,\sigma}_t(s)e_b\in[-1,1]$ and $V^\sigma_t(s)=\langle\sigma_t(s),Q^\sigma_t(s,\cdot)\rangle$.
- The exploiter runs per-state Hedge on losses: $\sigma_{k+1}(b\mid t,s)\propto\exp(-\eta\sum_{j\le k}\hat Q^{\sigma_j}_t(s,b))$, with a critic error $\varepsilon_k(t,s)=\max_b|\hat Q^{\sigma_k}_t(s,b)-Q^{\sigma_k}_t(s,b)|$ and $\hat Q$ clipped to $[-1,1]$.
- $d^\circ_t(s)=P_{\pi,\sigma^\circ}(s_t=s,\ t<\tau)$ for a comparator opponent $\sigma^\circ$, so $\sum_{t,s}d^\circ_t(s)=\bar H(\sigma^\circ)$.

**Theorem L3.** For every Markov comparator $\sigma^\circ$, in particular a Markov best response:
$$\frac1K\sum_{k=1}^KV(\pi,\sigma_k)-V(\pi,\sigma^\circ)\ \le\ \sum_{t,s}d^\circ_t(s)\,\frac{\mathrm{Reg}_K(t,s)}{K}+\frac2K\sum_{k}\sum_{t,s}d^\circ_t(s)\varepsilon_k(t,s).$$
Here $\mathrm{Reg}_K(t,s)\le\frac{\ln|B(s)|}{\eta}+\frac{\eta K}{2}$. With $m=\max_s|B(s)|$ and $\eta=\sqrt{2\ln m/K}$ the first term is $\le\bar H(\sigma^\circ)\sqrt{2\ln m/K}$.

*Proof.*
1. *PDL.* L0 in $M_\pi$ with $\Phi=V^{\sigma_k}$ and opponent $\sigma^\circ$ gives $V(\pi,\sigma_k)-V(\pi,\sigma^\circ)=\sum_{t,s}d^\circ_t(s)\,\langle\sigma_k(s)-\sigma^\circ(s),Q^{\sigma_k}_t(s,\cdot)\rangle$, using $x^\top Q[V^{\sigma_k}]y^\circ=\langle\sigma^\circ(s),Q^{\sigma_k}_t(s,\cdot)\rangle$.
2. *Fixed weighting.* $d^\circ$ does not depend on $k$. Averaging over $k$ gives, at each $(t,s)$, the regret of the per-state learner against the fixed comparator $\sigma^\circ(s)$ on the loss sequence $Q^{\sigma_k}_t(s,\cdot)$.
3. *Critic.* $\langle\sigma_k-\sigma^\circ,Q\rangle\le\langle\sigma_k-\sigma^\circ,\hat Q\rangle+2\varepsilon_k$. Hedge on the losses $\hat Q\in[-1,1]$ (range 2) has regret $\le\ln N/\eta+\eta K\cdot 2^2/8$. $\square$

This is the finite-horizon form of Even-Dar et al. (2009) and Agarwal et al. (2021). What is ours is the adaptation: the horizon factor is the comparator's expected game length, the same $\bar H$ that appears in Theorem A.

**Corollary L3a (RL gives a lower bound with a computable slack).** Let $\bar\sigma$ play a uniformly random iterate per game. Then $\operatorname{expl}(\pi)-\mathrm{slack}\le V^*_0-V(\pi,\bar\sigma)\le\operatorname{expl}(\pi)$, where slack is the right side of L3 with $\sigma^\circ=\sigma_{\rm BR}$. Only the regret term is computable in advance (using $\bar H\le H$); the critic term needs the BR's occupancy $d^{\rm BR}$.

**Corollary L3b (distribution mismatch).** Suppose the critic is fit on-policy, so that at every stage $\sum_sd^{\sigma_k}_t(s)\,\varepsilon_k(t,s)\le\varepsilon_{\rm crit}$. Let $C=\max_{k,t,s}d^{\rm BR}_t(s)/d^{\sigma_k}_t(s)$. Then the critic term in L3 with $\sigma^\circ=\sigma_{\rm BR}$ is $\le2HC\,\varepsilon_{\rm crit}$, because $\sum_sd^{\rm BR}_t\varepsilon_k\le C\sum_sd^{\sigma_k}_t\varepsilon_k$ at each of at most $H$ stages. $C$ is large exactly when the BR's exploiting line is one the learner rarely visits (L3c).

**Proposition L3c (undirected exploration can need $|B|^D$ games; white-box planning needs none).**

*Construction.* A chain $s_0\to\dots\to s_D$. The agent is forced on the chain. At $s_j$ ($j<D$), opponent action $b^*_j$ moves to $s_{j+1}$, and every other action ends the game with payoff $0$. $s_D$ is matching pennies ($\pm1$), and $\pi$ plays pure heads there. Then $V^*_0=0$ and $\operatorname{expl}(\pi)=1$.

*Claim.* Consider any exploiter whose per-state policy stays uniform as long as all of its observed returns are equal, such as per-state Hedge / NPG / PPO with a Monte-Carlo critic and no exploration bonus. With probability $\ge1-N|B|^{-D}$ its first $N$ games never reach $s_D$, so its gain is $0$ in each. By contrast, backward induction in $M_\pi$ (known model plus white-box $\pi$, which is what `wb2` uses) finds the gain-1 response with $O(D|B|)$ work and no games.

*Proof.* Under uniform play a game reaches $s_D$ with probability $|B|^{-D}$. Until that happens every observed return is $0$, so the policy stays uniform. A union bound over $N$ games finishes it. Each game that misses $s_D$ pays $0=V^*_0$. $\square$

*Prior art.* This is an instance of the classical undirected-exploration lower bounds: Whitehead (1991, Thm 8 / Cor 9), Kakade's thesis (2003, §4.3) and Osband et al. (2016, Ex. 1). In the same spirit, Wang et al. (2023) attacked KataGo with gray-box victim access inside search (victim-play), and Timbers et al. (2022) query the victim's policy in ISMCTS-BR.

*Scope.* Optimistic tabular RL needs only poly$(D,|B|)$ games here because the chain states are observable. The exact-critic Hedge of L3 does not satisfy the hypothesis (exact $Q$ carries signal). PPO satisfies it only as a tabular softmax at uniform initialisation; a randomly initialised network reaches $s_D$ with probability $\prod_j\sigma_{\rm init}(b^*_j)$ instead of $|B|^{-D}$. The exact miss probability is $(1-|B|^{-D})^N$ (simulation matches it).

---

## L4. One MCTS node is self-play no-regret; the visit shares are the average strategy

Work on the engine's scale. At a node, side one maximises $\hat A\in[0,1]^{n\times m}$ and side two maximises $1-\hat A$. Simulation $i$ picks $(a_i,b_i)$ and returns $r_i\in[0,1]$ with $\mathbb E[r_i\mid\text{past},a_i,b_i]=\hat A(a_i,b_i)$. This holds at depth 1 with a fixed leaf. Deeper, the cell means drift as subtrees are refined. Write $\bar x_N(a)=N_a/N$ and $\bar y_N(b)=N_b/N$ (visit shares). The mean-reward external regrets are
$\mathrm{Reg}^1_N=\max_a\sum_i\hat A(a,b_i)-\sum_i\hat A(a_i,b_i)$ and $\mathrm{Reg}^2_N=\max_b\sum_i(1-\hat A(a_i,b))-\sum_i(1-\hat A(a_i,b_i))$.

**Proposition L4 (exact folk identity for visit shares).**
$$\operatorname{NashGap}_{\hat A}(\bar x_N,\bar y_N):=\max_a(\hat A\bar y_N)_a-\min_b(\bar x_N^\top\hat A)_b=\frac{\mathrm{Reg}^1_N+\mathrm{Reg}^2_N}{N}.$$

*Proof.*
1. $\sum_i\hat A(a,b_i)=N(\hat A\bar y_N)_a$ and $\sum_i\hat A(a_i,b)=N(\bar x_N^\top\hat A)_b$.
2. Add the two regrets. The realised joint terms $\pm\sum_i\hat A(a_i,b_i)$ cancel, leaving $N\max_a(\hat A\bar y)_a-N\min_b(\bar x^\top\hat A)_b$. $\square$

**Consequences.**
1. *Same object as ours.* Our RM+ average is also the average strategy of a self-play no-regret dynamic on one stage matrix (Prop. 2 / Corollary A′ of `full_game.md`). MCTS's visit shares are the same object with **bandit** feedback (one sampled cell per simulation) instead of full information. Both feed Theorem A through the stage gap: by Lemma 1, a root strategy $\varepsilon$-optimal in the tree's matrix is $\varepsilon+2\varepsilon_v^{\rm tree}$-optimal in $Q^*$.
2. *Where they differ.* L4 converts regret into a gap exactly, so the visit shares converge iff $\mathrm{Reg}^1_N+\mathrm{Reg}^2_N=o(N)$. Sublinear regret on each side is sufficient but not necessary, because one side's regret can be negative: decoupled UCB1 on matching pennies has gap $0$ while $\mathrm{Reg}^2_N/N=+\tfrac12$.
   - RM+ and Exp3 / regret-matching selection have adversarial (Hannan-consistent) guarantees. For deeper trees, Lisý et al. (2013, Thm 4.11) prove almost-sure convergence to a subgame-perfect $O(D^2\varepsilon)$-equilibrium (quadratic in depth; no rate). It holds for ε-Hannan-consistent selection in the **mean-propagating** variant of SM-MCTS on a complete tree with exact terminals. Kovařík & Lisý (Machine Learning, Thm 4.1) show plain SM-MCTS with Hannan-consistent selection can converge outside the 1/5-Nash set; averaging (their Thm 5.4) or an unbiased-payoff property (Thm 5.12) restores convergence. None of this covers poke-engine (raw returns, UCB1, heuristic leaf). Shafiei, Sturtevant & Schaeffer (2009) report a biased-RPS case where UCT does not converge (as described by Bošanský et al. 2016; PDF not obtained). No formal DUCT non-convergence proof was found, so Proposition L4b may be the first explicit one (not established).
   - UCB1's guarantee assumes i.i.d. rewards per arm. In decoupled self-play each arm's mean drifts with the other side's choices, so no regret guarantee applies, and **numerically the visit shares do not converge** (`checks/lk4_duct.py`).
     - *Model.* One node, depth 1, fixed leaf (stationary cell means). Self-play with exactly poke-engine's rule (`mcts.rs` `maximize_ucb_for_side`/`ucb1`/`backpropagate`). Matrices: 200 random 3×3, 5×5 and 10×10, plus biased RPS (exact and 1%-perturbed) and matching pennies. $N\le10^6$.
     - *Deterministic or low-noise rewards* (Gaussian $\sigma\le0.1$): on games with a mixed equilibrium, the median NashGap at $N=10^6$ is 0.10–0.25, and it **grows** with $N$ (rand10: 0.09 at $10^4$, 0.25 at $10^6$). One 3×3 instance reaches 0.65 at $N=10^7$.
     - *Bernoulli rewards:* the gap shrinks slowly. Medians at $10^6$ are 0.008 / 0.024 / 0.043 for 3×3 / 5×5 / 10×10, with 45–100% of games still above 0.01. One 3×3 instance stays near 0.095 from $10^3$ to $10^7$.
     - *Pure saddles* converge under every noise model.
     - *Symmetric games* with deterministic rewards: the visit shares provably stay uniform (Proposition L4b), even when a row is dominant. Any reward noise breaks this.
     - *Hannan-consistent selection* on the same matrices: Exp3 and bandit regret matching reach median gaps of 0.002–0.013 at $10^6$, and full-information RM reaches 0.0004.
     - *Scope.* This is a model result. It does not include deeper-tree drift, foul-play's world sampling, or the engine's real iteration counts (default 100 ms per world; iterations per second not measured). It also does not say which noise regime the engine's sigmoid leaf is in. At smaller $N$, deterministic DUCT is less bad.
3. *The leaf term is shared.* poke-engine has no rollout: each expansion evaluates the heuristic leaf, so the frontier carries the same leaf error that Theorem A charges as $2\varepsilon_v$. MCTS differs in where the frontier lies (adaptive, deeper on promising lines) and in having exact terminals inside the tree. L5 quantifies what depth buys.
4. *foul-play's final rule.* By threshold, keeping the moves with share $\ge0.75\cdot$max and sampling proportionally lies between "Equilibrium, top move" (threshold 1) and "Equilibrium, sampled" (threshold 0). By exploitability it does **not** always lie between them.
   - *The rule's own floor.* Applied to an exact Nash $x^*$, the 0.75 rule has median exploitability 0.143 on biased RPS, 0 on rand3, 0.083 on rand5 and 0.198 on rand10 (engine scale $[0,1]$). This cost remains even if search converged perfectly.
   - *Applied to DUCT visit shares* at $N=10^6$ (deterministic rewards, side one, median exploitability rand3 / rand5 / rand10):
     - visit shares: .011 / .089 / .121;
     - most-visited move: .056 / .214 / .368;
     - 0.75 rule: .038 / .181 / .327.
   - *Support-restricted gap.* Any rule whose **marginal** play (averaged over its own randomness: MCTS RNG, sampled worlds) is supported on a set $S$ has stage exploitability $\ge\operatorname{val}\hat A-\max_{x\in\Delta(S)}\min_bx^\top\hat Ae_b$. For a single move this is the pure maximin gap $g(\hat A)=\operatorname{val}\hat A-\max_a\min_b\hat A(a,b)$. A per-run support does not bound a randomised pipeline: a 50/50 mix of two top-move outcomes can beat both.
   - With $Q^*$ in place of $\hat A$, L1(b) turns these per-turn gaps into a full-game lower bound along $\sigma_g$'s trajectories. This is the formal version of the W1 finding (top move is more exploitable than sampled). It is a bound with a computable diagnostic, not a measured decomposition of W1's $+0.16$.
   - *Scope of L4's identity.* $N=\sum_aN_a$. This holds at the root. At a non-root node where `expand` returns the node itself, `times_visited` increments without crediting the node's own options.


**Proposition L4b (decoupled UCB1 can stay uniform on a game with a dominant move).** Let $\hat A\in[0,1]^{n\times n}$ be symmetric in the sense $\hat A(a,b)+\hat A(b,a)=1$ for all $a,b$, with both sides listing their options in the same order. Run the code's rule with deterministic rewards $r_i=\hat A(a_i,b_i)$: UCB $\bar r+\sqrt{2\ln N/n}$, unvisited options at $+\infty$, ties to the first index (strict `>`). Then $a_i=b_i$ for every $i$, every option's mean stays $\tfrac12$, and play is round-robin, so $\bar x_N=\bar y_N$ is exactly uniform whenever $n\mid N$. In particular, on $\hat A=[[.5,.9],[.1,.5]]$, where row 0 is dominant, $\operatorname{NashGap}(\bar x_N,\bar y_N)=0.4$ for every even $N$.

*Proof.* Induction on $i$, with hypothesis: both sides have identical (count, total) statistics per option.
1. *Base.* At $i=0$ all counts are $0$.
2. *Same pick.* Identical statistics give identical UCB vectors, and the same tie rule then gives $a_i=b_i=:c$.
3. *Same reward.* Symmetry gives $\hat A(c,c)=\tfrac12$, so side one is credited $\tfrac12$ and side two $1-\tfrac12=\tfrac12$. The statistics stay identical, and every visited option's mean is $\tfrac12$.
4. *Round-robin.* With all means equal, UCB is decreasing in $n_i$, and unvisited options are at $+\infty$. So the pick is the least-visited option (first index on ties), which is round-robin.
5. *Gap.* For the 2×2 example, $\hat A\bar y=(.7,.3)$ and $\bar x^\top\hat A=(.3,.7)$, so the gap is $.7-.3=.4$. $\square$

*Scope.* The proposition is knife-edge: reward noise breaks the lockstep, and with Bernoulli or $\sigma=0.02$ Gaussian rewards this 2×2 converges (gap about $10^{-4}$ at $10^6$). The non-convergence on **generic** matrices with deterministic or low-noise rewards is not knife-edge, but it is numerical only (`checks/lk4_duct.py`).

---

## L5. What depth buys with exact terminals: the leaf error scales with the chance of still playing

**Lemma L5.0 (value difference).** For matrices $A,B$ of the same shape, with $x_A$ maximin-optimal in $A$ and $y_B$ minimax-optimal in $B$: $\operatorname{val}A-\operatorname{val}B\le x_A^\top(A-B)y_B$.
*Proof.* $\operatorname{val}A=\min_yx_A^\top Ay\le x_A^\top Ay_B$ and $\operatorname{val}B=\max_xx^\top By_B\ge x_A^\top By_B$; subtract. $\square$

**Setting.**
- The depth-$D$ lookahead values $V^D$ use exact val at inner nodes, the exact payoff at terminals and at the cap, and a leaf $\hat V$ at live frontier states, with $|\hat V-V^*|\le\varepsilon_{\rm leaf}$ there.
- Recursively, $V^D_t(s)=\operatorname{val}Q^{(D)}_t(s)$ with $Q^{(D)}_t(s,a,b)=\mathbb E[V^{D-1}_{t+1}(s')]$, and $V^0=\hat V$ on live states.
- Define the **cooperative survival probability** $\rho_0(t,s)=\mathbb 1[s\text{ live},t<H]$. For live $s$ and $t<H$, $\rho_D(t,s)=\max_{a,b}\mathbb E[\rho_{D-1}(t+1,s')\mid s,a,b]$; $\rho_D=0$ otherwise. This is the largest probability, over joint strategies, that $s_{t+D}$ is live **and** $t+D<H$ (reaching the cap counts as not live), so $\rho_D(t,s)=0$ whenever $t+D\ge H$.

**Theorem L5.** $|V^D_t(s)-V^*_t(s)|\le\varepsilon_{\rm leaf}\,\rho_D(t,s)$ and $\lVert Q^{(D)}_t(s)-Q^*_t(s)\rVert_\infty\le\varepsilon_{\rm leaf}\,\rho_D(t,s)$. Hence Theorem A holds with the per-state budget $\delta(t,s)=\varepsilon_s+2\varepsilon_{\rm leaf}\rho_D(t,s)$.

*Proof.* Induction on $D$.
1. *Base.* At $D=0$ the error is $\le\varepsilon_{\rm leaf}$ on live states and $0$ on terminal and cap states.
2. *Cells.* For $D\ge1$, each cell satisfies $|Q^{(D)}-Q^*|\le\mathbb E|V^{D-1}_{t+1}(s')-V^*_{t+1}(s')|\le\varepsilon_{\rm leaf}\mathbb E[\rho_{D-1}(t+1,s')]$, which is $\le\varepsilon_{\rm leaf}\rho_D(t,s)$ after the max over cells.
3. *Value.* val is 1-Lipschitz in sup-norm (equivalently, L5.0 applied both ways). $\square$

**Remarks.**
1. Step 2 at $D=1$, with $\varepsilon_{\rm leaf}$ replaced by the error on $s$'s own children and the cap treated as exact, is R6b's per-cell exact-terminal bound. The theorem as stated (max over cells, global $\varepsilon_{\rm leaf}$) is coarser. R6b counts cap children as live, while L5 treats them as exact.
2. In the middle game both players can usually prolong the game, so $\rho_D\approx1$ and depth alone does not shrink the leaf term. This repeats `theory_section.md`, "no contraction without discounting", now with the exact factor.
3. Near the end, $\rho_D$ can be small and deep search becomes nearly exact. That is consistent with the endgame result (exact terminals give $\approx0$ exploitability on 2 of 3 endgames) and with MCTS's strength coming from terminal-rich lines.
4. *Sharper version.* Use two one-sided recursions: $\rho^+$ follows the profile $(x^{(D)},y^*)$ and gives $V^D-V^*\le\varepsilon\rho^+$, and $\rho^-$ follows $(x^*,y^{(D)})$ and gives $V^*-V^D\le\varepsilon\rho^-$. Then $|V^D-V^*|\le\varepsilon\max(\rho^+,\rho^-)$, and the cell bound uses $\mathbb E[\max(\rho^+,\rho^-)(s')]$. This needs $x^*$ and $y^*$ and depends on tie-breaking, so it is an analysis device, not a computable quantity. In brute force it was strictly smaller than $\rho_D$ at 24% of nodes.
5. Undiscounted termination thus plays the role that $\gamma$ plays in the ADP error-propagation bounds (Munos 2007, Scherrer et al., Perolat et al. 2015).

---

## L6. The leaf's minimax TD residual bounds exploitability, linearly in game length

**Setting.**
- $\hat V_t(s)$ is a leaf on the payoff scale with exact terminals and $\hat V_H=0$ at the cap.
- The agent plays $x_t(s)$, $\varepsilon_s$-optimal in $Q[\hat V]_t(s)$ (one exact turn, then the leaf, which is our agent's design).
- The **signed minimax TD residual** is $\delta_t(s)=\hat V_t(s)-\operatorname{val}Q[\hat V]_t(s)$. It is the error of the minimax Bellman operator used by minimax-Q (Littman 1994) and by self-play value learning that solves the stage game.
- $\pi^*$ plays maximin-optimal stage strategies of $Q^*$. $\hat\sigma$ plays minimax-optimal stage strategies of $Q[\hat V]$ (the opponent that is greedy with respect to $\hat V$).

**Theorem L6.**
$$\operatorname{expl}(\pi)\ \le\ \mathbb E_{\pi,\sigma_{\rm BR}}\Big[\sum_{t<\tau}\big(\delta_t(s_t)+\varepsilon_s\big)\Big]-\mathbb E_{\pi^*,\hat\sigma}\Big[\sum_{t<\tau}\delta_t(s_t)\Big]\ \le\ \sum_{t<H}\big(2\sup_s|\delta_t(s)|+\varepsilon_s\big).$$

*Proof.* L0 with $\Phi=\hat V$, used twice.
1. *Agent side.* L0 gives $\hat V_0-V(\pi,\sigma_{\rm BR})=\mathbb E_{\pi,\sigma_{\rm BR}}[\sum(\hat V_t-x_t^\top Q[\hat V]_ty_t)]$. Also $x^\top Q[\hat V]y\ge\min_{y'}x^\top Q[\hat V]y'\ge\operatorname{val}Q[\hat V]-\varepsilon_s=\hat V_t-\delta_t-\varepsilon_s$. So $\hat V_0-V(\pi,\sigma_{\rm BR})\le\mathbb E_{\pi,\sigma_{\rm BR}}[\sum(\delta_t+\varepsilon_s)]$.
2. *Value side.* $\pi^*$ guarantees $V^*_0$, so $V^*_0\le V(\pi^*,\hat\sigma)$. L0 gives $V(\pi^*,\hat\sigma)-\hat V_0=-\mathbb E_{\pi^*,\hat\sigma}[\sum(\hat V_t-x^{*\top}Q[\hat V]_t\hat y_t)]$. Since $\hat y$ is minimax-optimal in $Q[\hat V]$, $x^{*\top}Q[\hat V]\hat y\le\operatorname{val}Q[\hat V]=\hat V_t-\delta_t$, so the summand is $\ge\delta_t$. Hence $V^*_0-\hat V_0\le-\mathbb E_{\pi^*,\hat\sigma}[\sum\delta_t]$.
3. *Combine.* Add steps 1 and 2: $\operatorname{expl}=(V^*_0-\hat V_0)+(\hat V_0-V(\pi,\sigma_{\rm BR}))$. The sup-norm form follows because $\tau\le H$. $\square$

**Remarks.**
1. *Incomparable with Theorem A.* Theorem A with $\varepsilon_v\le\lVert\hat V_{t+1}-V^*_{t+1}\rVert_\infty\le\sum_{u>t}\sup|\delta_u|$ can be quadratic in $H$, while L6 is linear in the residual. Conversely, if $\lVert\hat V-V^*\rVert_\infty=\varepsilon$, then $|\delta|\le2\varepsilon$, and L6 gives $4H\varepsilon$ against Theorem A's $2H\varepsilon$. Use the smaller of the two.
   - *Drift example.* Take a deterministic chain with **both** sides forced and $\hat V_t=V^*_t+c(H-t)/H$. The residual is $c/H$ per stage. The trajectory form of L6 is $0$ (the two trajectory sums coincide), the sup form is $2c$, and Theorem A's budget is exactly $c(H-1)$.
   - *Reverse example (trajectory level).* `reverse_example` in `checks/lk6_residual.py` has $H=3$ and $\eta=0.01$. $\pi^*$ goes to a state the agent never reaches, where $\hat V$ makes $\hat\sigma$ play badly. Result: $\operatorname{expl}=0.010$, the L6 trajectory form is $0.970$, and Theorem A's trajectory form is $0.040$. In 1000 random games, by contrast, L6's trajectory form was never worse than Theorem A's.
   - *A family of bounds.* Step 2 never uses the optimality of $x^*$, only that $\pi^*$ guarantees $V^*_0$. So any best response, any minimax $\hat\sigma$ and any such $\pi^*$ give a valid bound, and so does the minimum over these choices. A per-state $\varepsilon_s(t,s)$ may replace the global $\varepsilon_s$.
2. *RL link and prior art.* $\delta$ is the TD error of the minimax Bellman operator, so L6 bounds exploitability by a training metric of value learning. It weights the residual only on the states reached by (agent, its best response) and by (equilibrium, $\hat V$-greedy opponent), which is where a value net trained on self-play data may be weakest; this is the same mismatch as L3b. The closest published statements are discounted. Singh & Yee (1994): greedy loss $\le2\gamma\varepsilon/(1-\gamma)$ with $\varepsilon\le\delta/(1-\gamma)$ (MDPs). Pérolat et al. (ICML 2016, Thm 1): $\lVert Q^*-Q_\mu\rVert_2\le\tfrac{2}{1-\gamma}\sqrt{(C_2+C_2')/2}\,\lVert T^*Q-Q\rVert_2$ for zero-sum Markov games. Williams & Baird (1993): MDP version, abstract only. L6 is the finite-horizon, undiscounted, signed, exact-trajectory-weight analogue, with termination playing the role of $\gamma$.
3. *Measurable.* $\delta_t(s)$ needs only the leaf at $s$ and the value of $s$'s own stage matrix, and our agent computes that matrix every turn.
4. *Mis-scored terminals.* Let $m=\hat V-u$ on terminal states, with $Q[\hat V]$, $\delta$, the agent and $\hat\sigma$ all built from the mis-scored $\hat V$. Then
   $$\operatorname{expl}(\pi)\le\mathbb E_{\pi,\sigma_{\rm BR}}\Big[\sum_{t<\tau}(\delta_t+\varepsilon_s)\Big]-\mathbb E_{\pi^*,\hat\sigma}\Big[\sum_{t<\tau}\delta_t\Big]+\mathbb E_{\pi,\sigma_{\rm BR}}[m(s_\tau)]-\mathbb E_{\pi^*,\hat\sigma}[m(s_\tau)].$$
   Note the **minus** on the second terminal term. In the sup form this adds at most $\sup m-\inf m\le2\sup|m|$. Brute force: the corrected form has min slack $-3\times10^{-16}$ over 1000 games, while adding both terms with a plus sign fails in 419 of them. A cap with $\hat V_H\ne0$ adds the same two terms with $m=\hat V_H$ on live cap states (derived, not tested).

5. *Measured on real games (diagnostic, not a report number).* 20 games: our agent ("Equilibrium, sampled") vs `mcts_policy(2000)`, leaf $\tanh(\text{mid}/k)$ with exact terminals, 651 non-forced states from the 17 games that did not hit the cap. Mean $\delta^2$ is lowest at $k\approx320$–$400$ (game bootstrap). At our $k=160$ it is 1.7× that minimum (0.0073 vs 0.0043). The bound is vacuous: $2\sum_t|\delta_t|$ per game has median 4.2, against a payoff range of 2, because the residual is about 0.04 per turn and games last about 35 turns. Exact terminals do not lower the residual (terminal children already have $\tanh\approx0.99$); they remove the terminal-mismatch term of Remark 4 instead. The value net's residual is higher than the tanh leaf's for every $k\ge80$. Scripts and data: `checks/residual/` (probe run, not re-verified).

---

## Summary: the three formulations side by side

| | Our per-turn agent | MCTS (poke-engine, foul-play) | RL best response (PPO) |
|---|---|---|---|
| Where regret is minimised | the current turn's stage matrix | every tree node, each side separately | every state (shared through the policy net) |
| Learner and feedback | RM+, full information, adversarial guarantee | UCB1 per side, bandit feedback, stochastic-bandit guarantee only (visit shares need not converge: L4) | Hedge on advantages (NPG; PPO approximates it) |
| Output | linear average of RM+ iterates | visit shares (uniform average), then the 0.75-threshold rule | current policy |
| Regret → gap at the decision point | folk theorem | exact identity L4 | PDL, L3 |
| Potential in L0 | $V^*$ (L1) | $V^*$ (L1) | $V^{\pi,\sigma_k}$ (L3) |
| Error more iterations do not remove | leaf: $2\varepsilon_v$ per turn | frontier leaf $2\varepsilon_{\rm leaf}\rho_D$ (L5) | critic error under the BR's distribution, $\times C$ (L3b) |
| Horizon factor | $\bar H$ = expected length vs BR | same | $\bar H(\sigma_{\rm BR})$ |

## Verification status

| Claim | Status | Evidence |
|---|---|---|
| L0, L1(a)–(c), Example L1b (corrected to gap $2-\delta$) | **proved**; checked by brute force, min slack $-3\times10^{-16}$ (1000 Markov and 300 history-dependent games) | `checks/lk1_telescoping.py` |
| L2(a)–(d) (with normalised rewards, clipped $\hat A$, opponent not seeing $a_i$) | **proved**; checked by simulation (pathwise regret/R ≤ 0.23 in games, ≤ 0.66 on alternation; expectation cells hold with wide margin) | `checks/lk2_safe_hedge.py` |
| L3, L3a, L3b, L3c | **proved** (adapted from EKM09 / Aga21); PDL identity error $2\times10^{-16}$; bound max LHS/RHS 0.42, tight with realised regret; lock probabilities match | `checks/lk3_rl_hedge.py` |
| L4 identity | **proved**; checked (300 random sequences, error $7\times10^{-14}$) | `checks/lk4_duct.py` |
| L4 consequence 2: decoupled-UCB1 visit shares do not converge on mixed-equilibrium games (one-node model) | **numerically checked** (verifier run, plus an independent minimal replica: 0.40 constant on the symmetric 2×2; rand3_95 gap grows 0.11 → 0.40 from $10^3$ to $10^5$); lockstep on symmetric games **proved** by induction | `checks/lk4_duct.py`, `checks/lk4_duct_worst.json` |
| L5, L5.0, Remark 4 | **proved**; checked by brute force (600 games, all $D$; bound attained, ratio 1) | `checks/lk5_depth.py` |
| L6 (trajectory and sup forms), drift and reverse examples, Remark 4 (sign corrected) | **proved**; checked by brute force (1000 games, min slack $-3\times10^{-16}$) | `checks/lk6_residual.py` |
| Leaf-curve identity $\tfrac12(1+\tanh(h/160))=\operatorname{sigmoid}(0.0125h)$ | algebra: $\tfrac12(1+\tanh z)=\operatorname{sigmoid}(2z)$ | — |
| Attributions (KL02 L6.1, EKM Thm 5, Aga21 L5.1/Thm 5.3/Cor 6.1, Lis13 Thm 4.11, KL-ML Thm 4.1, Wei21 L32, DFG20 L3, Per16 Thm 1, SY94, GS15 §6.4, Whitehead91) | **cited**, statements read from PDFs except where `lit/links_attribution.md` tags [abstract]/[snippet] (SSS09, WB93, EKM 2009 journal, Munos 2007) | `lit/links_attribution.md` |

## Refuted / superseded

- L4 Consequence 2 "converges iff both sides have sublinear regret": false (matching pennies under decoupled UCB1). Correct: iff the sum is $o(N)$.
- L4 Consequence 4 "the 0.75 rule lies between the two experts": true for the threshold, not for exploitability.
- L2 Assumption (H) as first drafted (code step on raw $[-1,1]$ rewards, unclipped $\hat A$, $R=2\cdot$R2a): false; needs normalised rewards and clipping.
- L2(d) attributed to Cauchy–Schwarz: it is subadditivity of $\sqrt\cdot$.
- L6 Remark 4 as first drafted ("add these two terms", both with $+$): false. The second terminal term enters with a minus sign (fails in 419 of 1000 random games).
- L1 Remark 2 as first drafted ("the whole payoff range" from the matching-pennies example): that example gives only $\Delta/2$; replaced by the $2-\delta$ example.
