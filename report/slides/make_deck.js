// Builds deck.pptx: 8-slide, 4-minute talk for the CS535 final project.
// Every number is taken from ../final5.tex.
// Run: NODE_PATH=<dir with pptxgenjs> node make_deck.js
const path = require("path");
const pptxgen = require("pptxgenjs");
const { applyTheme } = require(
  "/Users/biofanat/.claude/skills/synced/b4cc7400-752a-4f56-bf97-0158b6471348_0967a200-5d21-4c9e-8841-ea4bd100999d/pptx/scripts/apply_theme.js"
);

const OUT = path.join(__dirname, "deck.pptx");
const FIG = path.join(__dirname, "..", "figures");

// ---------- theme: pine green, sage, terracotta accent ----------
const THEME = {
  name: "Pine Academic",
  headFontFace: "Cambria",
  bodyFontFace: "Calibri",
  colors: {
    dk1: "222222", lt1: "FFFFFF",
    dk2: "24392F", lt2: "EDF2EE",
    accent1: "3F6B55", accent2: "B5653A", accent3: "8FA597",
    accent4: "C9A227", accent5: "5B4B6B", accent6: "6E7F73",
    hlink: "3F6B55", folHlink: "5B4B6B",
  },
};

const pres = new pptxgen();
pres.layout = "LAYOUT_16x9"; // 10 x 5.625 in
pres.title = "No-Regret Learning at Three Levels in Pokemon Random Battles";
pres.author = "[Names]";
pres.theme = { headFontFace: THEME.headFontFace, bodyFontFace: THEME.bodyFontFace };
const C = pres.SchemeColor;
const HEAD = "+mj-lt"; // theme heading font (Cambria) for equations and big numbers

// ---------- layouts ----------
pres.defineSlideMaster({
  title: "TITLE_DARK",
  background: { color: C.text2 },
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: 0.6, y: 1.05, w: 8.8, h: 1.5,
        fontSize: 36, bold: true, color: C.background1, align: "left", valign: "bottom" }, text: "" } },
    { placeholder: { options: { name: "body", type: "body", x: 0.6, y: 2.75, w: 8.4, h: 0.95,
        fontSize: 18, italic: true, color: C.accent3, align: "left", valign: "top" }, text: "" } },
  ],
});
pres.defineSlideMaster({
  title: "CONTENT",
  background: { color: C.background1 },
  margin: [0.5, 0.5, 0.5, 0.5],
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: 0.5, y: 0.3, w: 9.0, h: 0.75,
        fontSize: 32, bold: true, color: C.text2, align: "left", valign: "middle" }, text: "" } },
    { text: { text: "No-regret learning in Random Battles · CS535",
        options: { x: 0.5, y: 5.2, w: 6.0, h: 0.3, fontSize: 10, color: C.accent6, margin: 0 } } },
  ],
  slideNumber: { x: 9.0, y: 5.2, w: 0.5, h: 0.3, fontSize: 10, color: C.accent6, align: "right" },
});
pres.defineSlideMaster({
  title: "CLOSING_DARK",
  background: { color: C.text2 },
  margin: [0.5, 0.5, 0.5, 0.5],
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: 0.5, y: 0.3, w: 9.0, h: 0.75,
        fontSize: 32, bold: true, color: C.background1, align: "left", valign: "middle" }, text: "" } },
  ],
  slideNumber: { x: 9.0, y: 5.2, w: 0.5, h: 0.3, fontSize: 10, color: C.accent3, align: "right" },
});

// ---------- helpers ----------
// Parse "M_kᵀ", "β_{k}", "Reg^{1}" into runs with sub/superscripts.
function math(src, base = {}) {
  const runs = [];
  let buf = "";
  const flush = () => { if (buf) { runs.push({ text: buf, options: { ...base } }); buf = ""; } };
  for (let i = 0; i < src.length; i++) {
    const ch = src[i];
    if ((ch === "_" || ch === "^") && i + 1 < src.length) {
      flush();
      let tok;
      if (src[i + 1] === "{") { const j = src.indexOf("}", i); tok = src.slice(i + 2, j); i = j; }
      else { tok = src[i + 1]; i += 1; }
      runs.push({ text: tok, options: { ...base, [ch === "_" ? "subscript" : "superscript"]: true } });
    } else buf += ch;
  }
  flush();
  return runs;
}

// Numbered circle: the deck's motif.
function badge(slide, label, x, y, d, fill, name) {
  slide.addShape(pres.shapes.OVAL, { x, y, w: d, h: d, fill: { color: fill }, line: { color: fill },
    objectName: name });
  slide.addText(label, { x, y, w: d, h: d, align: "center", valign: "middle", fontSize: d > 0.5 ? 20 : 14,
    bold: true, color: C.background1, fontFace: HEAD, margin: 0, isTextBox: true, objectName: name + " label" });
}

function card(slide, x, y, w, h, fill, name) {
  slide.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h, rectRadius: 0.08,
    fill: { color: fill }, line: { color: fill }, objectName: name });
}

function content(section, title) {
  const s = pres.addSlide({ masterName: "CONTENT", sectionTitle: section });
  s.addText(title, { placeholder: "title" });
  return s;
}

// =====================================================================
// 1. Title
pres.addSection({ title: "Opening" });
{
  const s = pres.addSlide({ masterName: "TITLE_DARK", sectionTitle: "Opening" });
  s.addText("No-Regret Learning at Three Levels in Pokémon Random Battles", { placeholder: "title" });
  s.addText("Can per-turn equilibrium play, built from no-regret learning, play Random Battles — and where does it break?",
    { placeholder: "body" });
  [["1", C.accent1], ["2", C.accent2], ["3", C.accent5]].forEach(([n, col], i) =>
    badge(s, n, 0.6 + i * 0.6, 4.35, 0.42, col, `Level ${n} badge`));
  s.addText("[Names] · CS535", { x: 2.6, y: 4.35, w: 5.0, h: 0.42, fontSize: 16, color: C.background1,
    valign: "middle", margin: 0, isTextBox: true, objectName: "Authors" });
  s.addNotes("Hi, we are [Names]. PokaiTrainer showed that per-turn equilibrium play over sampled opponent " +
    "worlds is competitive in Pokémon VGC. We ask whether the same idea, built entirely from no-regret learning, " +
    "can play Gen 9 Random Battles against Foul Play, the strongest public bot. Short answer: the pieces compose " +
    "cleanly and come with guarantees, but the agent stays below Foul Play, and our theory and tests point to where it breaks.");
}

// =====================================================================
// 2. The game
pres.addSection({ title: "Setting" });
{
  const s = content("Setting", "The game: simultaneous moves, chance, hidden teams");
  s.addText("Each turn: a stage matrix in each of K = 4 sampled worlds", { x: 0.5, y: 1.15, w: 9.0, h: 0.4,
    fontSize: 18, color: C.text1, margin: 0, isTextBox: true, objectName: "Turn summary" });

  card(s, 0.5, 1.75, 4.3, 1.55, C.background2, "PIMC card");
  s.addText([
    { text: "PIMC averaging", options: { bold: true, fontSize: 18, color: C.accent2, breakLine: true } },
    { text: "Solve each world as if known, then average the equilibria", options: { fontSize: 15, color: C.text1 } },
  ], { x: 0.7, y: 1.85, w: 3.9, h: 1.35, valign: "top", margin: 0, isTextBox: true, objectName: "PIMC text" });

  card(s, 5.2, 1.75, 4.3, 1.55, C.background2, "Bayesian card");
  s.addText([
    { text: "Bayesian matrix game", options: { bold: true, fontSize: 18, color: C.accent1, breakLine: true } },
    { text: "One strategy x; the opponent best-responds in each world (PokaiTrainer)", options: { fontSize: 15, color: C.text1 } },
  ], { x: 5.4, y: 1.85, w: 3.9, h: 1.35, valign: "top", margin: 0, isTextBox: true, objectName: "Bayesian text" });

  s.addText(math("max_x  Σ_k  β_k  min_j  (M_kᵀ x)_j", { fontFace: HEAD, fontSize: 28, color: C.text2 }),
    { x: 0.5, y: 3.6, w: 9.0, h: 0.8, align: "center", valign: "middle", margin: 0, isTextBox: true,
      objectName: "Bayesian LP equation" });
  s.addText("one LP per turn", { x: 0.5, y: 4.4, w: 9.0, h: 0.35, align: "center", fontSize: 14, italic: true,
    color: C.accent6, margin: 0, isTextBox: true, objectName: "Equation caption" });
  s.addNotes("A battle is a zero-sum stochastic game with simultaneous moves, chance, and hidden opponent teams. " +
    "Each turn we sample K equals 4 opponent worlds and build a depth-2 stage matrix in each. PIMC averaging solves " +
    "each world as if it were known and averages, so it suffers strategy fusion. The Bayesian matrix game, from " +
    "PokaiTrainer, picks one strategy x while the opponent best-responds in every world. It is a single LP.");
}

// =====================================================================
// 3. From lecture to our forms
{
  const s = content("Setting", "From lecture to our forms");
  card(s, 0.5, 1.2, 4.3, 1.45, C.background2, "MWU card");
  s.addText([
    { text: "MWU (lecture)", options: { bold: true, fontSize: 18, color: C.accent6, breakLine: true } },
    { text: "w ← w (1 − ηℓ)", options: { fontFace: HEAD, fontSize: 24, color: C.text1 } },
  ], { x: 0.7, y: 1.3, w: 3.9, h: 1.25, valign: "top", margin: 0, isTextBox: true, objectName: "MWU text" });

  card(s, 5.2, 1.2, 4.3, 1.45, C.background2, "Hedge card");
  s.addText([
    { text: "Hedge (ours)", options: { bold: true, fontSize: 18, color: C.accent1, breakLine: true } },
    { text: "w ← w · e", options: { fontFace: HEAD, fontSize: 24, color: C.text1 } },
    { text: "ηr", options: { fontFace: HEAD, fontSize: 24, color: C.text1, superscript: true, breakLine: true } },
    ...math("p_t(e) ∝ exp(η Σ_{s<t} r_{s,e})", { fontFace: HEAD, fontSize: 16, color: C.text1 }),
  ], { x: 5.4, y: 1.3, w: 3.9, h: 1.25, valign: "top", margin: 0, isTextBox: true, objectName: "Hedge text" });

  s.addText([
    { text: "Up- and down-weighting coincide after normalising; the difference is exponential vs linear",
      options: { bullet: true, breakLine: true } },
    { text: "Bayes = Hedge with η = 1 on log loss", options: { bullet: true } },
  ], { x: 0.5, y: 2.8, w: 9.0, h: 0.85, fontSize: 15, color: C.text1, paraSpaceAfter: 4, valign: "top",
    margin: 0, isTextBox: true, objectName: "Notes bullets" });

  card(s, 0.5, 3.8, 9.0, 1.15, C.text2, "Bottom line card");
  s.addText([
    { text: "No-regret self-play: time averages form an ε-equilibrium,  ", options: { color: C.background1 } },
    ...math("ε = (Reg_1 + Reg_2) / T", { fontFace: HEAD, bold: true, color: C.background1 }),
    { text: "", options: { breakLine: true } },
    { text: "(in zero-sum games the time-average CCE is minimax)", options: { fontSize: 13, italic: true, color: C.accent3 } },
  ], { x: 0.7, y: 3.85, w: 8.6, h: 1.05, fontSize: 17, valign: "middle", margin: 0, isTextBox: true,
    objectName: "Bottom line text" });
  s.addNotes("Lecture gave us multiplicative weights, a linear update on losses. Our code uses Hedge, an exponential " +
    "update on rewards, so the strategy is a softmax of cumulative reward. Up-weighting and down-weighting coincide " +
    "after normalising; the real difference is exponential versus linear. Bayesian updating is Hedge with step one on " +
    "log loss. And the classic link: in self-play, the summed regret over T bounds the duality gap of the time averages.");
}

// =====================================================================
// 4. Three levels
pres.addSection({ title: "Method and theory" });
{
  const s = content("Method and theory", "No-regret at three places");
  const boxes = [
    ["1", "Inside a turn", "Bayesian LP / RM⁺", "→ CFR lineage", C.accent1, null],
    ["2", "Across games", "Hedge over 4 experts + style posterior", "→ fixed share / Bayes", C.accent2, null],
    ["3", "Search root", "Bandits UCB1 / Exp3", "→ visit-share identity", C.accent5,
      "gap = (Reg^1 + Reg^2) / N"],
  ];
  const w = 2.8, gap = 0.3, y = 1.3, h = 3.45;
  boxes.forEach(([n, where, what, lin, col, eq], i) => {
    const x = 0.5 + i * (w + gap);
    card(s, x, y, w, h, C.background2, `Level ${n} card`);
    badge(s, n, x + 0.2, y + 0.2, 0.55, col, `Level ${n} badge`);
    s.addText(`Level ${n}`, { x: x + 0.9, y: y + 0.2, w: w - 1.1, h: 0.55, fontSize: 14, bold: true,
      color: col, valign: "middle", margin: 0, isTextBox: true, objectName: `Level ${n} label` });
    const runs = [
      { text: where, options: { bold: true, fontSize: 18, color: C.text2, breakLine: true } },
      { text: what, options: { fontSize: 15, color: C.text1, breakLine: true } },
      { text: lin, options: { fontSize: 14, italic: true, color: C.accent6 } },
    ];
    s.addText(runs, { x: x + 0.2, y: y + 0.95, w: w - 0.4, h: 1.65, valign: "top", paraSpaceAfter: 6,
      margin: 0, isTextBox: true, objectName: `Level ${n} text` });
    if (eq) s.addText(math(eq, { fontFace: HEAD, fontSize: 16, color: C.text2 }),
      { x: x + 0.2, y: y + 2.7, w: w - 0.4, h: 0.5, valign: "middle", margin: 0, isTextBox: true,
        objectName: `Level ${n} equation` });
  });
  s.addNotes("We use no-regret learning in three places. Level one, inside a turn: the Bayesian LP, or RM-plus self-play, " +
    "from the CFR lineage. Level two, across turns and games: Hedge over four per-turn experts, plus a Bayesian style " +
    "posterior that predicts the opponent, related to fixed share. Level three, the search root: MCTS is bandit " +
    "self-play, and the duality gap of the visit shares equals the two sides' summed regret divided by N, exactly.");
}

// =====================================================================
// 5. Headline theory
{
  const s = content("Method and theory", "Headline theory: a whole-battle bound");
  card(s, 0.5, 1.2, 9.0, 1.1, C.text2, "Bound card");
  s.addText(math("expl(π)  ≤  H̄ · sup (ε_s + 2ε_v)", { fontFace: HEAD, fontSize: 32, color: C.background1 }),
    { x: 0.5, y: 1.2, w: 9.0, h: 1.1, align: "center", valign: "middle", margin: 0, isTextBox: true,
      objectName: "Bound equation" });
  const rows = [
    ["ε_s", "solver error: shrinks with compute", C.accent1],
    ["ε_v", "stage-matrix error from the heuristic evaluation: does not", C.accent2],
    ["H̄", "expected length against a best response (tight)", C.accent5],
  ];
  rows.forEach(([sym, txt, col], i) => {
    const y = 2.6 + i * 0.68;
    s.addShape(pres.shapes.OVAL, { x: 0.8, y, w: 0.5, h: 0.5, fill: { color: col }, line: { color: col },
      objectName: `Term ${i + 1} badge` });
    s.addText(math(sym, { fontFace: HEAD, fontSize: 16, bold: true, color: C.background1 }),
      { x: 0.8, y, w: 0.5, h: 0.5, align: "center", valign: "middle", margin: 0, isTextBox: true,
        objectName: `Term ${i + 1} symbol` });
    s.addText(txt, { x: 1.55, y, w: 7.9, h: 0.5, fontSize: 18, color: C.text1, valign: "middle", margin: 0,
      isTextBox: true, objectName: `Term ${i + 1} text` });
  });
  s.addText("Plus a hidden-information version; the core step is checked in Lean 4", { x: 0.5, y: 4.7, w: 9.0,
    h: 0.35, fontSize: 13, italic: true, color: C.accent6, margin: 0, isTextBox: true, objectName: "Theory footer" });
  s.addNotes("The headline theorem charges whole-battle exploitability to per-turn errors: exploitability is at most " +
    "H-bar times the sup of epsilon-s plus two epsilon-v. The solver error epsilon-s shrinks with compute. The " +
    "stage-matrix error epsilon-v comes from the heuristic leaf evaluation, and deeper search does not shrink it. " +
    "H-bar is the expected game length against a best response, and that factor is tight. A hidden-information version " +
    "adds sampling and belief terms.");
}

// =====================================================================
// 6. Engine experiments
pres.addSection({ title: "Experiments" });
{
  const s = content("Experiments", "Experiments in the engine");
  const stats = [
    ["0.760 / 0.650", "MCTS at matched time beats equilibrium play (vs sampled / top move; 100 games)", C.accent2],
    ["0.250 vs 0.090", "95% lower bounds on exploitability: top move vs sampled play", C.accent1],
    ["0.585", "Bayesian LP beats PIMC averaging, sampled play [0.540, 0.630]; 400 games", C.accent1],
    ["6.9–155×", "Exact endgames: the bound holds; error consistent with the leaf evaluation dominating", C.accent5],
  ];
  const w = 4.3, h = 1.7;
  stats.forEach(([big, label, col], i) => {
    const x = 0.5 + (i % 2) * (w + 0.4), y = 1.2 + Math.floor(i / 2) * (h + 0.25);
    card(s, x, y, w, h, C.background2, `Stat ${i + 1} card`);
    s.addText(big, { x: x + 0.25, y: y + 0.15, w: w - 0.5, h: 0.75, fontFace: HEAD, fontSize: 36, bold: true,
      color: col, valign: "middle", margin: 0, isTextBox: true, objectName: `Stat ${i + 1} number` });
    s.addText(label, { x: x + 0.25, y: y + 0.92, w: w - 0.5, h: 0.68, fontSize: 14, color: C.text1,
      valign: "top", margin: 0, isTextBox: true, objectName: `Stat ${i + 1} label` });
  });
  s.addNotes("In-engine results. At matched time, MCTS beats equilibrium play: 0.760 against sampled and 0.650 against " +
    "top move. An informed best response gives 95 percent lower bounds on exploitability of 0.250 for top move and " +
    "0.090 for sampled play, so randomising helps. The Bayesian LP beats PIMC averaging, 0.585 over 400 games. On exact " +
    "endgames the bound holds but is loose by 6.9 to 155 times, consistent with the leaf evaluation dominating.");
}

// =====================================================================
// 7. Real Showdown
{
  const s = content("Experiments", "Real Showdown and pre-registered tests");
  // 680 x 450 px figure
  const fw = 4.5, fh = fw * 450 / 680;
  s.addImage({ path: path.join(FIG, "fig_gauntlet_col.png"), x: 0.5, y: 1.25, w: fw, h: fh,
    altText: "Round-robin Bradley-Terry ratings: Foul Play at all search times above Bayesian LP; top move and PIMC near 0",
    objectName: "Round-robin ratings figure" });
  s.addText("Round robin, 1,800 games; ratings on an Elo scale, 95% CIs", { x: 0.5, y: 1.3 + fh, w: fw, h: 0.3,
    fontSize: 11, italic: true, color: C.accent6, margin: 0, isTextBox: true, objectName: "Figure caption" });
  s.addText([
    { text: "Below Foul Play even at its smallest budget (25 ms per sampled battle)", options: { bullet: true, breakLine: true } },
    { text: "Level with the other one-turn solvers", options: { bullet: true, breakLine: true } },
    { text: "Pre-registered: detection delay matches its prediction per match (ρ = 0.86); no prediction refuted",
      options: { bullet: true, breakLine: true } },
    { text: "Better opponent prediction (≈0.3 nats per scored turn), but no detectable win-rate change (replicated)",
      options: { bullet: true } },
  ], { x: 5.3, y: 1.25, w: 4.2, h: 3.6, fontSize: 15, color: C.text1, paraSpaceAfter: 10, valign: "top",
    margin: 0, isTextBox: true, objectName: "Showdown bullets" });
  s.addNotes("On real Showdown we ran a round robin of 1,800 games. Foul Play at every search time, even 25 milliseconds " +
    "per sampled battle, rates above our Bayesian LP; we are level with the other one-turn solvers. Our pre-registered " +
    "detection-delay test refuted none of its predictions: each match's delay tracks its own prediction, rho 0.86. " +
    "Better opponent prediction, about 0.3 nats per scored turn, gave no detectable win-rate change, and the null replicated.");
}

// =====================================================================
// 8. Takeaways
pres.addSection({ title: "Closing" });
{
  const s = pres.addSlide({ masterName: "CLOSING_DARK", sectionTitle: "Closing" });
  s.addText("Takeaways", { placeholder: "title" });
  const items = [
    ["1", "No-regret learning composes cleanly across the three levels", C.accent1],
    ["2", "The bound is consistent with the stage-matrix (leaf) error dominating; better beliefs don't move win rate", C.accent2],
    ["3", "Next: an evaluation function trained on exact terminal values", C.accent5],
  ];
  items.forEach(([n, txt, col], i) => {
    const y = 1.35 + i * 1.0;
    badge(s, n, 0.6, y, 0.6, col, `Takeaway ${n} badge`);
    s.addText(txt, { x: 1.5, y: y - 0.1, w: 7.9, h: 0.8, fontSize: 20, color: C.background1, valign: "middle",
      margin: 0, isTextBox: true, objectName: `Takeaway ${n} text` });
  });
  s.addText("Also: joint set×style belief and lookahead-posterior nulls; case study in the paper", { x: 0.5, y: 4.65, w: 9.0,
    h: 0.35, fontSize: 13, italic: true, color: C.accent3, margin: 0, isTextBox: true, objectName: "Also line" });
  s.addNotes("Three takeaways. First, no-regret learning composes cleanly across a turn, a match and a search root. " +
    "Second, the whole-battle bound points at the stage-matrix error from the leaf evaluation, and our evidence is " +
    "consistent with it dominating: better beliefs improve prediction, not win rate. So, third, the next step is " +
    "an evaluation function trained on exact terminal values. The paper also has the joint belief, the nulls and a case study. Thank you.");
}

(async () => {
  await pres.writeFile({ fileName: OUT });
  await applyTheme(OUT, THEME);
  console.log("wrote", OUT);
})();
