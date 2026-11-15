import Mathlib

set_option linter.deprecated false

/-!
# Lean checks for `theory/full_game.md`

* (a) Lemma 1 (one step): `lemma1_one_step`
* (b) Lemma 2(a): the maximin value is 1-Lipschitz in sup norm: `val_lipschitz`
  (value = max over the simplex of the min over pure columns; existence: `exists_val`)
* (c) Theorem A's backward-induction step, abstractly: `backward_induction`,
  `deficit_le_sum`, `theoremA_abstract`
* (d) bonus, used to close the §4 "alternating updates" gap: the RM+ one-step improvement
  inequality `rmplus_improvement`.

Matrices are `Fin n → Fin m → ℝ`, row player (agent) = maximiser.
-/

open Finset

namespace FullGame

variable {n m : ℕ}

/-- `x` is a mixed strategy (a point of the probability simplex). -/
def IsMixed {k : ℕ} (x : Fin k → ℝ) : Prop := (∀ i, 0 ≤ x i) ∧ ∑ i, x i = 1

/-- Payoff of mixed row `x` against pure column `j`: `(xᵀA)_j`. -/
def colPay (A : Fin n → Fin m → ℝ) (x : Fin n → ℝ) (j : Fin m) : ℝ := ∑ i, x i * A i j

/-- `x` guarantees `c` in `A`: every pure column pays the row player at least `c`
(equivalently `min_j (xᵀA)_j ≥ c`; see `Guar.mixed` for mixed columns). -/
def Guar (A : Fin n → Fin m → ℝ) (x : Fin n → ℝ) (c : ℝ) : Prop := ∀ j, c ≤ colPay A x j

/-- `v = max_{x ∈ Δ} min_j (xᵀA)_j`: attained by some mixed `x`, and no mixed `x` guarantees more. -/
def IsVal (A : Fin n → Fin m → ℝ) (v : ℝ) : Prop :=
  (∃ x, IsMixed x ∧ Guar A x v) ∧ ∀ x, IsMixed x → ∀ c, Guar A x c → c ≤ v

/-- `‖A - B‖_∞ ≤ ε` (max entry norm). -/
def SupDist (A B : Fin n → Fin m → ℝ) (ε : ℝ) : Prop := ∀ i j, |A i j - B i j| ≤ ε

lemma SupDist.symm {A B : Fin n → Fin m → ℝ} {ε : ℝ} (h : SupDist A B ε) : SupDist B A ε :=
  fun i j => by rw [abs_sub_comm]; exact h i j

lemma Guar.mono {A : Fin n → Fin m → ℝ} {x : Fin n → ℝ} {c c' : ℝ} (h : Guar A x c) (hc : c' ≤ c) :
    Guar A x c' := fun j => hc.trans (h j)

/-- A guarantee against pure columns is a guarantee against mixed columns:
`c ≤ xᵀ A y` for every mixed `y`. -/
lemma Guar.mixed {A : Fin n → Fin m → ℝ} {x : Fin n → ℝ} {c : ℝ} (h : Guar A x c)
    {y : Fin m → ℝ} (hy : IsMixed y) : c ≤ ∑ j, y j * colPay A x j := by
  calc c = ∑ j, y j * c := by rw [← Finset.sum_mul, hy.2, one_mul]
    _ ≤ ∑ j, y j * colPay A x j :=
        Finset.sum_le_sum fun j _ => mul_le_mul_of_nonneg_left (h j) (hy.1 j)

/-- Step (i) of the note's proof of Lemma 1, at a pure column. -/
lemma colPay_close {A B : Fin n → Fin m → ℝ} {ε : ℝ} (hAB : SupDist A B ε)
    {x : Fin n → ℝ} (hx : IsMixed x) (j : Fin m) : |colPay A x j - colPay B x j| ≤ ε := by
  unfold colPay
  rw [← Finset.sum_sub_distrib]
  calc |∑ i, (x i * A i j - x i * B i j)| ≤ ∑ i, |x i * A i j - x i * B i j| :=
        Finset.abs_sum_le_sum_abs _ _
    _ ≤ ∑ i, x i * ε := Finset.sum_le_sum fun i _ => by
        rw [← mul_sub, abs_mul, abs_of_nonneg (hx.1 i)]
        exact mul_le_mul_of_nonneg_left (hAB i j) (hx.1 i)
    _ = ε := by rw [← Finset.sum_mul, hx.2, one_mul]

/-- A guarantee moves by at most `ε` between `ε`-close matrices. -/
lemma Guar.transfer {A B : Fin n → Fin m → ℝ} {ε : ℝ} (hAB : SupDist A B ε)
    {x : Fin n → ℝ} (hx : IsMixed x) {c : ℝ} (h : Guar B x c) : Guar A x (c - ε) := fun j => by
  have := (abs_le.mp (colPay_close hAB hx j)).1
  linarith [h j]

/-- Step (ii): `val(B) ≥ val(A) - ε`, using only that `val A` is attained and `val B` is an upper
bound on guarantees in `B`. -/
lemma val_ge {A B : Fin n → Fin m → ℝ} {ε vA vB : ℝ} (hAB : SupDist A B ε)
    (hA : ∃ x, IsMixed x ∧ Guar A x vA) (hB : ∀ x, IsMixed x → ∀ c, Guar B x c → c ≤ vB) :
    vA - ε ≤ vB := by
  obtain ⟨xA, hxA, hgA⟩ := hA
  exact hB xA hxA _ (hgA.transfer hAB.symm hxA)

/-- **(b) Lemma 2(a)**: the matrix-game value is 1-Lipschitz in the max-entry norm. -/
theorem val_lipschitz {A B : Fin n → Fin m → ℝ} {ε vA vB : ℝ} (hAB : SupDist A B ε)
    (hvA : IsVal A vA) (hvB : IsVal B vB) : |vA - vB| ≤ ε := by
  have h1 := val_ge hAB hvA.1 hvB.2
  have h2 := val_ge hAB.symm hvB.1 hvA.2
  rw [abs_sub_le_iff]; constructor <;> linarith

/-- **(a) Lemma 1 (one step)**: `‖A-B‖_∞ ≤ ε` and `x` `εs`-optimal in `B` (guarantees
`val B - εs`) ⇒ `x` guarantees `val A - εs - 2ε` in `A`. -/
theorem lemma1_one_step {A B : Fin n → Fin m → ℝ} {ε εs vA vB : ℝ} (hAB : SupDist A B ε)
    (hvA : IsVal A vA) (hvB : IsVal B vB) {x : Fin n → ℝ} (hx : IsMixed x)
    (hopt : Guar B x (vB - εs)) : Guar A x (vA - εs - 2 * ε) := by
  have hv := val_ge hAB hvA.1 hvB.2
  exact (hopt.transfer hAB hx).mono (by linarith)

/-- Same, stated against an arbitrary mixed opponent `y` (the form used in Theorem A). -/
theorem lemma1_mixed {A B : Fin n → Fin m → ℝ} {ε εs vA vB : ℝ} (hAB : SupDist A B ε)
    (hvA : IsVal A vA) (hvB : IsVal B vB) {x : Fin n → ℝ} (hx : IsMixed x)
    (hopt : Guar B x (vB - εs)) {y : Fin m → ℝ} (hy : IsMixed y) :
    vA - εs - 2 * ε ≤ ∑ j, y j * colPay A x j :=
  (lemma1_one_step hAB hvA hvB hx hopt).mixed hy

/-- The value exists (max over the compact simplex of a continuous min), so `IsVal` is never
vacuous for `n, m ≥ 1`. -/
theorem exists_val [NeZero n] (A : Fin n → Fin m → ℝ) (hm : (univ : Finset (Fin m)).Nonempty) :
    ∃ v, IsVal A v := by
  set F : (Fin n → ℝ) → ℝ := fun x => univ.inf' hm (fun j => colPay A x j) with hF
  have hcont : Continuous F := by
    apply Continuous.finset_inf'_apply
    intro j _
    unfold colPay
    fun_prop
  have hne : (stdSimplex ℝ (Fin n)).Nonempty := ⟨fun _ => 1 / n, by
    refine ⟨fun _ => by positivity, ?_⟩
    simp [Finset.sum_const, Finset.card_univ, Fintype.card_fin]⟩
  obtain ⟨x0, hx0, hmax⟩ := (isCompact_stdSimplex ℝ (Fin n)).exists_isMaxOn hne hcont.continuousOn
  refine ⟨F x0, ⟨x0, hx0, fun j => Finset.inf'_le _ (mem_univ j)⟩, ?_⟩
  intro x hx c hc
  have : c ≤ F x := Finset.le_inf' _ _ fun j _ => hc j
  exact this.trans (hmax (show x ∈ stdSimplex ℝ (Fin n) from hx))

/-! ### Non-vacuity: the note's tightness example for the factor 2 (ε = 1)
`A = (0, -2)ᵀ`, `B = (-1, -1)ᵀ` (n = 2, m = 1), `val A = 0`, `val B = -1`, `x = e₂`. -/

def Aex : Fin 2 → Fin 1 → ℝ := ![![0], ![-2]]
def Bex : Fin 2 → Fin 1 → ℝ := ![![-1], ![-1]]
def e1 : Fin 2 → ℝ := ![1, 0]
def e2 : Fin 2 → ℝ := ![0, 1]

lemma mixed_e1 : IsMixed e1 := ⟨by intro i; fin_cases i <;> simp [e1], by simp [e1, Fin.sum_univ_two]⟩
lemma mixed_e2 : IsMixed e2 := ⟨by intro i; fin_cases i <;> simp [e2], by simp [e2, Fin.sum_univ_two]⟩

lemma supDist_ex : SupDist Aex Bex 1 := by
  intro i j; fin_cases i <;> fin_cases j <;> norm_num [Aex, Bex]

lemma val_Aex : IsVal Aex 0 := by
  refine ⟨⟨e1, mixed_e1, fun j => by fin_cases j; simp [colPay, Aex, e1, Fin.sum_univ_two]⟩, ?_⟩
  intro x hx c hc
  have h := hc 0
  simp only [colPay, Aex, Fin.sum_univ_two] at h
  simp at h
  linarith [hx.1 1]

lemma val_Bex : IsVal Bex (-1) := by
  refine ⟨⟨e2, mixed_e2, fun j => by fin_cases j; simp [colPay, Bex, e2, Fin.sum_univ_two]⟩, ?_⟩
  intro x hx c hc
  have h := hc 0
  have hs := hx.2
  simp only [colPay, Bex, Fin.sum_univ_two] at h hs
  simp at h
  linarith

/-- Hypotheses of Lemma 1 are jointly satisfiable, and the conclusion is attained with equality
(`e₂` pays exactly `-2 = val A - 0 - 2·1` in `A`): the factor 2 is tight. -/
example : Guar Aex e2 (0 - 0 - 2 * 1) ∧ colPay Aex e2 0 = 0 - 0 - 2 * 1 := by
  refine ⟨lemma1_one_step supDist_ex val_Aex val_Bex mixed_e2 ?_, ?_⟩
  · intro j; fin_cases j; simp [colPay, Bex, e2, Fin.sum_univ_two]
  · simp [colPay, Aex, e2, Fin.sum_univ_two]

example : |(0 : ℝ) - (-1)| ≤ 1 := val_lipschitz supDist_ex val_Aex val_Bex

/-! ### (c) Theorem A's backward induction, abstract form

Nodes (histories) live in a finite type `α`; `K t h h'` is the one-step probability of moving from
`h` to `h'` at stage `t` (it encodes `x_t(s)(a)·σ(h)(β)·P(s'|s,a,β)` and may depend on the whole
history, i.e. on a history-dependent opponent). `live t h` marks non-terminal nodes.
`W` = payoff of (π, σ), `V` = benchmark value (e.g. `V*_t(s_t)`), `δ` = per-stage loss,
`D` = expected accumulated loss. The stage hypothesis `hstage` is exactly what Lemma 1 supplies
(`lemma1_mixed`): against the opponent's actual reply the agent's one-step expectation of `V_{t+1}`
is at least `V_t - δ_t`. -/

section Induction

variable {α : Type*} [Fintype α]

/-- `W_t(h) ≥ V_t(h) - D_t(h)` for all `t ≤ H` (the induction in the proof of Theorem A). -/
theorem backward_induction (H : ℕ) (K : ℕ → α → α → ℝ) (live : ℕ → α → Prop)
    (W V D δ : ℕ → α → ℝ)
    (hK0 : ∀ t h h', 0 ≤ K t h h')
    (hbase : ∀ h, V H h - D H h ≤ W H h)
    (hdead : ∀ t < H, ∀ h, ¬ live t h → V t h - D t h ≤ W t h)
    (hW : ∀ t < H, ∀ h, live t h → W t h = ∑ h', K t h h' * W (t + 1) h')
    (hD : ∀ t < H, ∀ h, live t h → D t h = δ t h + ∑ h', K t h h' * D (t + 1) h')
    (hstage : ∀ t < H, ∀ h, live t h → V t h - δ t h ≤ ∑ h', K t h h' * V (t + 1) h') :
    ∀ t ≤ H, ∀ h, V t h - D t h ≤ W t h := by
  intro t ht
  obtain ⟨k, rfl⟩ : ∃ k, t = H - k := ⟨H - t, by omega⟩
  induction k with
  | zero => simpa using hbase
  | succ k ih =>
    intro h
    by_cases hk : k < H
    · have hlt : H - (k + 1) < H := by omega
      have hs : H - (k + 1) + 1 = H - k := by omega
      have ih' := ih (by omega)
      by_cases hl : live (H - (k + 1)) h
      · rw [hW _ hlt h hl, hD _ hlt h hl]
        have hst := hstage _ hlt h hl
        rw [hs] at hst ⊢
        have : ∑ h', K (H - (k + 1)) h h' * V (H - k) h'
            - ∑ h', K (H - (k + 1)) h h' * D (H - k) h'
            ≤ ∑ h', K (H - (k + 1)) h h' * W (H - k) h' := by
          rw [← Finset.sum_sub_distrib]
          exact Finset.sum_le_sum fun h' _ => by
            rw [← mul_sub]; exact mul_le_mul_of_nonneg_left (ih' h') (hK0 _ h h')
        linarith
      · exact hdead _ hlt h hl
    · have : H - (k + 1) = H - k := by omega
      rw [this]; exact ih (by omega) h

/-- The accumulated loss is at most the sum of per-stage sups: `D_t(h) ≤ Σ_{t ≤ s < H} d_s`
(the middle inequality of (A.1); `K t h ·` is a probability vector on live nodes). -/
theorem deficit_le_sum (H : ℕ) (K : ℕ → α → α → ℝ) (live : ℕ → α → Prop)
    (D δ : ℕ → α → ℝ) (d : ℕ → ℝ)
    (hK0 : ∀ t h h', 0 ≤ K t h h') (hK1 : ∀ t < H, ∀ h, live t h → ∑ h', K t h h' = 1)
    (hd0 : ∀ t, 0 ≤ d t)
    (hbase : ∀ h, D H h ≤ 0)
    (hdead : ∀ t < H, ∀ h, ¬ live t h → D t h ≤ 0)
    (hD : ∀ t < H, ∀ h, live t h → D t h = δ t h + ∑ h', K t h h' * D (t + 1) h')
    (hδ : ∀ t < H, ∀ h, live t h → δ t h ≤ d t) :
    ∀ t ≤ H, ∀ h, D t h ≤ ∑ s ∈ Finset.Ico t H, d s := by
  intro t ht
  obtain ⟨k, rfl⟩ : ∃ k, t = H - k := ⟨H - t, by omega⟩
  induction k with
  | zero => intro h; simpa using hbase h
  | succ k ih =>
    intro h
    have hnn : 0 ≤ ∑ s ∈ Finset.Ico (H - (k + 1)) H, d s :=
      Finset.sum_nonneg fun s _ => hd0 s
    by_cases hk : k < H
    · have hlt : H - (k + 1) < H := by omega
      have hs : H - (k + 1) + 1 = H - k := by omega
      have ih' := ih (by omega)
      by_cases hl : live (H - (k + 1)) h
      · rw [hD _ hlt h hl, hs]
        have hsplit : ∑ s ∈ Finset.Ico (H - (k + 1)) H, d s
            = d (H - (k + 1)) + ∑ s ∈ Finset.Ico (H - k) H, d s := by
          rw [← hs, Finset.sum_eq_sum_Ico_succ_bot hlt]
        have hsum : ∑ h', K (H - (k + 1)) h h' * D (H - k) h'
            ≤ ∑ s ∈ Finset.Ico (H - k) H, d s := by
          calc ∑ h', K (H - (k + 1)) h h' * D (H - k) h'
              ≤ ∑ h', K (H - (k + 1)) h h' * ∑ s ∈ Finset.Ico (H - k) H, d s :=
                Finset.sum_le_sum fun h' _ => mul_le_mul_of_nonneg_left (ih' h') (hK0 _ h h')
            _ = ∑ s ∈ Finset.Ico (H - k) H, d s := by
                rw [← Finset.sum_mul, hK1 _ hlt h hl, one_mul]
        linarith [hδ _ hlt h hl]
      · exact (hdead _ hlt h hl).trans hnn
    · have : H - (k + 1) = H - k := by omega
      rw [this]; exact ih (by omega) h

/-- **Theorem A, abstract (A.0)+(A.1)**: with `D = 0` at terminal nodes / at the horizon and
`W = V` there, `W_0(h) ≥ V_0(h) - Σ_{t<H} d_t`. -/
theorem theoremA_abstract (H : ℕ) (K : ℕ → α → α → ℝ) (live : ℕ → α → Prop)
    (W V D δ : ℕ → α → ℝ) (d : ℕ → ℝ)
    (hK0 : ∀ t h h', 0 ≤ K t h h') (hK1 : ∀ t < H, ∀ h, live t h → ∑ h', K t h h' = 1)
    (hd0 : ∀ t, 0 ≤ d t)
    (hWH : ∀ h, W H h = V H h) (hDH : ∀ h, D H h = 0)
    (hWdead : ∀ t < H, ∀ h, ¬ live t h → W t h = V t h)
    (hDdead : ∀ t < H, ∀ h, ¬ live t h → D t h = 0)
    (hW : ∀ t < H, ∀ h, live t h → W t h = ∑ h', K t h h' * W (t + 1) h')
    (hD : ∀ t < H, ∀ h, live t h → D t h = δ t h + ∑ h', K t h h' * D (t + 1) h')
    (hstage : ∀ t < H, ∀ h, live t h → V t h - δ t h ≤ ∑ h', K t h h' * V (t + 1) h')
    (hδ : ∀ t < H, ∀ h, live t h → δ t h ≤ d t) (h0 : α) :
    V 0 h0 - ∑ s ∈ Finset.range H, d s ≤ W 0 h0 := by
  have h1 := backward_induction H K live W V D δ hK0 (fun h => by rw [hWH, hDH]; simp)
    (fun t ht h hl => by rw [hWdead t ht h hl, hDdead t ht h hl]; simp) hW hD hstage 0 (by omega) h0
  have h2 := deficit_le_sum H K live D δ d hK0 hK1 hd0 (fun h => (hDH h).le)
    (fun t ht h hl => (hDdead t ht h hl).le) hD hδ 0 (by omega) h0
  rw [Finset.range_eq_Ico]
  linarith

end Induction

/-- Non-vacuity of `theoremA_abstract`: one node, horizon 2, every stage live, loss `δ = d = 1`,
`V_t = D_t = 2 - t`, `W ≡ 0`. All hypotheses hold and the conclusion `V_0 - 2 ≤ W_0` is an equality. -/
example : (2 : ℝ) - ((0 : ℕ) : ℝ) - ∑ s ∈ Finset.range 2, (fun _ => (1 : ℝ)) s ≤ 0 :=
  theoremA_abstract (α := Unit) 2 (fun _ _ _ => 1) (fun _ _ => True)
    (fun _ _ => 0) (fun t _ => (2 : ℝ) - t) (fun t _ => (2 : ℝ) - t)
    (fun _ _ => 1) (fun _ => 1)
    (by intros; norm_num) (by intros; simp) (by intros; norm_num)
    (by intro h; norm_num) (by intro h; norm_num)
    (fun t _ h hl => absurd trivial hl) (fun t _ h hl => absurd trivial hl)
    (by intros; simp)
    (by intro t _ h _; simp; ring)
    (by intro t _ h _; simp; ring_nf; rfl)
    (by intros; exact le_refl _) ()

/-! ### (d) RM+ one-step improvement (used in CHECK.md to close the §4 gap)
If `Q ≥ 0` and `⟨Q, r⟩ = 0` (true for `r = g - ⟨x,g⟩𝟙` when `x ∝ Q`), then the updated accumulator
`(Q + r)⁺` has `⟨(Q + r)⁺, r⟩ ≥ 0`: the new RM+ strategy does at least as well against the reward
vector that produced it as the old one. -/

theorem rmplus_improvement {k : ℕ} (Q r : Fin k → ℝ) (hQ : ∀ i, 0 ≤ Q i)
    (horth : ∑ i, Q i * r i = 0) : 0 ≤ ∑ i, max (Q i + r i) 0 * r i := by
  set p : Fin k → ℝ := fun i => max (Q i + r i) 0 with hp
  -- ⟨p, r⟩ = ‖p‖² - ⟨p, Q⟩  since p·(Q+r) = p²
  have hpz : ∀ i, p i * (Q i + r i) = p i ^ 2 := by
    intro i; simp only [hp]
    rcases le_total (Q i + r i) 0 with h | h
    · rw [max_eq_right h]; ring
    · rw [max_eq_left h]; ring
  have hsplit : ∑ i, p i * r i = ∑ i, p i ^ 2 - ∑ i, Q i * p i := by
    rw [eq_sub_iff_add_eq, ← Finset.sum_add_distrib]
    exact Finset.sum_congr rfl fun i _ => by rw [← hpz i]; ring
  -- ‖Q‖² = ⟨Q, Q + r⟩ ≤ ⟨Q, p⟩
  have hQQ : ∑ i, Q i ^ 2 ≤ ∑ i, Q i * p i := by
    calc ∑ i, Q i ^ 2 = ∑ i, Q i * (Q i + r i) := by
          rw [show (∑ i, Q i * (Q i + r i)) = ∑ i, Q i ^ 2 + ∑ i, Q i * r i by
            rw [← Finset.sum_add_distrib]; exact Finset.sum_congr rfl fun i _ => by ring, horth,
            add_zero]
      _ ≤ ∑ i, Q i * p i := Finset.sum_le_sum fun i _ =>
          mul_le_mul_of_nonneg_left (le_max_left _ _) (hQ i)
  -- Cauchy–Schwarz: ⟨Q,p⟩² ≤ ‖Q‖²‖p‖² ≤ ⟨Q,p⟩‖p‖²  ⇒  ⟨Q,p⟩ ≤ ‖p‖²
  have hcs := Finset.sum_mul_sq_le_sq_mul_sq Finset.univ Q p
  have hpp : 0 ≤ ∑ i, p i ^ 2 := Finset.sum_nonneg fun i _ => sq_nonneg _
  have key : ∑ i, Q i * p i ≤ ∑ i, p i ^ 2 := by
    rcases le_or_gt (∑ i, Q i * p i) 0 with h | h
    · linarith
    · have : (∑ i, Q i * p i) ^ 2 ≤ (∑ i, Q i * p i) * ∑ i, p i ^ 2 :=
        hcs.trans (mul_le_mul_of_nonneg_right hQQ hpp)
      nlinarith
  show 0 ≤ ∑ i, p i * r i
  linarith [hsplit]

/-- Non-vacuity: `Q = (1,0)`, `r = (0,1)` (orthogonal), update `(1,1)`, improvement `1 ≥ 0`. -/
example : 0 ≤ ∑ i, max ((![1, 0] : Fin 2 → ℝ) i + (![0, 1] : Fin 2 → ℝ) i) 0 * (![0, 1] : Fin 2 → ℝ) i :=
  rmplus_improvement _ _ (by intro i; fin_cases i <;> simp) (by simp [Fin.sum_univ_two])

end FullGame
