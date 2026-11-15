import FullGame.Basic

set_option linter.deprecated false

/-!
# Lean checks for `theory/hedge_experts.md` (Hedge + fixed-share experts learner)

* `learner_reward_linear`: the learner's normalised reward `Σ_e w_e r_e` equals the normalised
  payoff of the mixture `p = Σ_e w_e q_e` (needs `Σ w = 1`).
* `claim1_weighted`: Claim 1 in the units the code uses (per-turn normaliser `Δ_t`), with the Hedge
  regret bound as a hypothesis.
* `claim1_fixed`: Claim 1 in raw units when the normaliser is a fixed `Δbar`.
* `hedge_telescope`: the summation step of the fixed-share Hedge regret bound, with the per-step
  potential inequality (Hoeffding's lemma) as a hypothesis.
* `composition_step`: per-turn link from the surrogate `Â` to the true stage game `Q*`
  (`V* - pᵀQ*e_b ≤ val Â - pᵀÂe_b + 2ε_v`), from `Basic.lean`'s `colPay_close` and `val_ge`.

Turns are indexed by `t ∈ range S`; expert payoffs are abstract reals `u t e = q_{e,t} · Â_t[:, b_t]`,
so the number of actions may change from turn to turn.
-/

open Finset

namespace FullGame

variable {k : ℕ}

/-- Linearity: if `Σ_e w_e = 1`, the weighted normalised expert reward is the normalised payoff of
the mixture. -/
lemma learner_reward_linear (w u : Fin k → ℝ) (lo Δ : ℝ) (hw : ∑ e, w e = 1) :
    ∑ e, w e * ((u e - lo) / Δ) = ((∑ e, w e * u e) - lo) / Δ := by
  have h : ∀ e, w e * ((u e - lo) / Δ) = (w e * u e) / Δ - (w e * lo) / Δ := fun e => by ring
  rw [Finset.sum_congr rfl (fun e _ => h e), Finset.sum_sub_distrib, ← Finset.sum_div,
    ← Finset.sum_div, ← Finset.sum_mul, hw, one_mul, sub_div]

/-- **Claim 1, normalised units (as coded).** Per turn `t < S`: weights `w t` on `k` experts summing
to 1, expert payoffs `u t e` against the realised column, normaliser `lo t`, `Δ t > 0`.
`hreg` is the Hedge (+ fixed share) regret bound against expert `estar` in normalised rewards;
`hsafe` says `estar` guarantees `v t - ε t` (an `ε_s`-equilibrium expert of `Â_t`, any column).
Conclusion: the learner's normalised payoff is at least the normalised `val - ε_s`, minus `R`. -/
theorem claim1_weighted (S : ℕ) (w u : ℕ → Fin k → ℝ) (lo Δ v ε : ℕ → ℝ) (estar : Fin k) (R : ℝ)
    (hw : ∀ t, ∑ e, w t e = 1) (hΔ : ∀ t, 0 < Δ t)
    (hreg : ∑ t ∈ range S, (u t estar - lo t) / Δ t
        ≤ ∑ t ∈ range S, ∑ e, w t e * ((u t e - lo t) / Δ t) + R)
    (hsafe : ∀ t, v t - ε t ≤ u t estar) :
    ∑ t ∈ range S, (v t - ε t - lo t) / Δ t - R
        ≤ ∑ t ∈ range S, ((∑ e, w t e * u t e) - lo t) / Δ t := by
  have h1 : ∑ t ∈ range S, (v t - ε t - lo t) / Δ t ≤ ∑ t ∈ range S, (u t estar - lo t) / Δ t :=
    Finset.sum_le_sum fun t _ =>
      div_le_div_of_nonneg_right (by linarith [hsafe t]) (hΔ t).le
  have h2 : ∑ t ∈ range S, ∑ e, w t e * ((u t e - lo t) / Δ t)
      = ∑ t ∈ range S, ((∑ e, w t e * u t e) - lo t) / Δ t :=
    Finset.sum_congr rfl fun t _ => learner_reward_linear (w t) (u t) (lo t) (Δ t) (hw t)
  linarith

/-- **Claim 1, raw units, fixed normaliser `Δbar > 0`.** The regret bound `R` is in normalised
units; in raw units it is multiplied by `Δbar`. -/
theorem claim1_fixed (S : ℕ) (w u : ℕ → Fin k → ℝ) (lo : ℝ) (Δbar : ℝ) (v ε : ℕ → ℝ)
    (estar : Fin k) (R : ℝ) (hw : ∀ t, ∑ e, w t e = 1) (hΔ : 0 < Δbar)
    (hreg : ∑ t ∈ range S, (u t estar - lo) / Δbar
        ≤ ∑ t ∈ range S, ∑ e, w t e * ((u t e - lo) / Δbar) + R)
    (hsafe : ∀ t, v t - ε t ≤ u t estar) :
    ∑ t ∈ range S, (v t - ε t) - Δbar * R ≤ ∑ t ∈ range S, ∑ e, w t e * u t e := by
  have h := claim1_weighted S w u (fun _ => lo) (fun _ => Δbar) v ε estar R hw (fun _ => hΔ)
    hreg hsafe
  simp only [← Finset.sum_div] at h
  have h' := mul_le_mul_of_nonneg_right h hΔ.le
  rw [sub_mul, div_mul_cancel₀ _ hΔ.ne', div_mul_cancel₀ _ hΔ.ne'] at h'
  simp only [Finset.sum_sub_distrib, Finset.sum_const, Finset.card_range, nsmul_eq_mul] at h'
  rw [Finset.sum_sub_distrib]
  linarith

/-- **Summation step of the Hedge + fixed-share bound.** `L t` = log-weight of the comparator before
turn `t`. Per step (Hoeffding's lemma + the share step, not formalised here):
`L (t+1) ≥ L t + η (r t - m t) - c`, with `c = η²/8 + log(1/(1-α))`. With `L 0 = -log k` and
`L S ≤ 0` (a weight is `≤ 1`): `Σ_t (r t - m t) ≤ (log k + S c)/η`. -/
theorem hedge_telescope (S : ℕ) (L r m : ℕ → ℝ) (η c logk : ℝ) (hη : 0 < η)
    (hstep : ∀ t, L t + η * (r t - m t) - c ≤ L (t + 1))
    (h0 : L 0 = -logk) (hS : L S ≤ 0) :
    ∑ t ∈ range S, (r t - m t) ≤ (logk + S * c) / η := by
  have key : ∀ N, L 0 + η * ∑ t ∈ range N, (r t - m t) - N * c ≤ L N := by
    intro N
    induction N with
    | zero => simp
    | succ N ih =>
      rw [Finset.sum_range_succ, Nat.cast_succ]
      have := hstep N
      nlinarith
  have := key S
  rw [le_div_iff₀ hη]
  nlinarith

/-- **Composition, one turn.** `Q` (true stage matrix, value `V`) and `Ahat` (value `vh`) are
`ε_v`-close; for any mixed `p` and any column `j` (the opponent's realised move),
`V - pᵀQ e_j ≤ (vh - pᵀ Ahat e_j) + 2 ε_v`. -/
theorem composition_step {n m : ℕ} {Q Ahat : Fin n → Fin m → ℝ} {εv V vh : ℝ}
    (hd : SupDist Q Ahat εv) (hV : IsVal Q V) (hvh : IsVal Ahat vh)
    {p : Fin n → ℝ} (hp : IsMixed p) (j : Fin m) :
    V - colPay Q p j ≤ (vh - colPay Ahat p j) + 2 * εv := by
  have h1 := (abs_le.mp (colPay_close hd hp j)).1
  have h2 := val_ge hd hV.1 hvh.2
  linarith

end FullGame

#print axioms FullGame.claim1_weighted
#print axioms FullGame.claim1_fixed
#print axioms FullGame.hedge_telescope
#print axioms FullGame.composition_step
