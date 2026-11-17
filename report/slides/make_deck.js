// Builds deck.pptx: 10-slide, ~4-minute talk for the CS535 final project, in five sections.
// Every number is taken from ../final5.tex.
// Run (any working directory; paths are relative to this file): NODE_PATH=<dir with pptxgenjs> node make_deck.js
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
pres.title = "No-Regret Learning and Bayesian Opponent Memory in Pokemon Random Battles";
pres.author = "[Names]";
pres.theme = { headFontFace: THEME.headFontFace, bodyFontFace: THEME.bodyFontFace };
const C = pres.SchemeColor;
const HEAD = "+mj-lt"; // theme heading font (Cambria) for equations and big numbers

const SECTIONS = ["Preliminaries", "Method", "Experiments", "Theory", "Conclusions"];

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

// Section tracker: a thin rule and the five section names along the bottom of the layout.
// The current section is bold in the accent colour; the others are muted.
function trackerObjects(current, dark) {
  const on = dark ? C.accent4 : C.accent2, off = dark ? C.accent3 : C.accent6;
  const cellW = 1.6, x0 = 0.5, y = 5.17, h = 0.3;
  const objs = [{ line: { x: 0.5, y: 5.1, w: 9.0, h: 0, line: { color: off, width: 0.75 } } }];
  SECTIONS.forEach((name, i) => {
    const cur = i === current;
    objs.push({ text: { text: name, options: { x: x0 + i * cellW, y, w: cellW, h, fontSize: 10, bold: cur,
      color: cur ? on : off, align: "center", valign: "middle", margin: 0 } } });
    if (i > 0) objs.push({ text: { text: "·", options: { x: x0 + i * cellW - 0.1, y, w: 0.2, h, fontSize: 10,
      color: off, align: "center", valign: "middle", margin: 0 } } });
  });
  return objs;
}

// One layout per section: SEC1–SEC4 light content frames, SEC5 the dark closing frame.
SECTIONS.forEach((name, i) => {
  const dark = i === SECTIONS.length - 1;
  pres.defineSlideMaster({
    title: `SEC${i + 1}`,
    background: { color: dark ? C.text2 : C.background1 },
    margin: [0.5, 0.5, 0.5, 0.5],
    objects: [
      { placeholder: { options: { name: "title", type: "title", x: 0.5, y: 0.3, w: 9.0, h: 0.75,
          fontSize: 28, bold: true, color: dark ? C.background1 : C.text2, align: "left", valign: "middle" },
          text: "" } },
      ...trackerObjects(i, dark),
    ],
    slideNumber: { x: 9.0, y: 5.17, w: 0.5, h: 0.3, fontSize: 10, color: dark ? C.accent3 : C.accent6,
      align: "right" },
  });
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

// A slide on its section's layout (SEC1–SEC5), which carries the tracker.
function content(sec, title) {
  const s = pres.addSlide({ masterName: `SEC${sec}`, sectionTitle: SECTIONS[sec - 1] });
  s.addText(title, { placeholder: "title" });
  return s;
}

// =====================================================================
// Section 1: Preliminaries
pres.addSection({ title: SECTIONS[0] });

// 1. Title (no tracker)
{
  const s = pres.addSlide({ masterName: "TITLE_DARK", sectionTitle: SECTIONS[0] });
  s.addText("No-Regret Learning and Bayesian Opponent Memory in Pokémon Random Battles", { placeholder: "title" });
  s.addText("Can per-turn equilibrium play, built from no-regret learning, play Random Battles — and where does it break?",
    { placeholder: "body" });
  [["1", C.accent1], ["2", C.accent2], ["3", C.accent5]].forEach(([n, col], i) =>
    badge(s, n, 0.6 + i * 0.6, 4.35, 0.42, col, `Level ${n} badge`));
  s.addText("[Names] · CS535", { x: 2.6, y: 4.35, w: 5.0, h: 0.42, fontSize: 16, color: C.background1,
    valign: "middle", margin: 0, isTextBox: true, objectName: "Authors" });
  s.addNotes("Hi, we are [Names]. PokaiTrainer showed that per-turn equilibrium play over sampled opponent " +
    "worlds is competitive in Pokémon VGC. We ask whether the same idea, built entirely from no-regret learning, " +
    "can play Gen 9 Random Battles against Foul Play, the strongest public bot. Short answer: the pieces compose " +
    "cleanly, but the agent stays below Foul Play, and our theory points to where it breaks.");
}

// 2. The game and the yardsticks
{
  const s = content(1, "The game and the yardsticks");
  s.addText("Zero-sum stochastic game: simultaneous moves, chance, hidden teams → a stage matrix each turn",
    { x: 0.5, y: 1.15, w: 9.0, h: 0.5, fontSize: 16, color: C.text1, margin: 0, isTextBox: true,
      objectName: "Game summary" });
  const rows = [
    ["Regret", "Reg_T = max_e Σ_t r_{t,e} − Σ_t p_t · r_t", "best expert in hindsight minus what we earned", C.accent1],
    ["Exploitability", "expl(x) = val(A) − min_y xᵀAy",
      "loss against a best-responding opponent, relative to the game value", C.accent2],
    ["Duality gap", "gap(x, y) = max_i (Ay)_i − min_j (xᵀA)_j", "the exploitability of both players together", C.accent5],
  ];
  rows.forEach(([name, eq, gloss, col], i) => {
    const y = 1.8 + i * 1.07;
    card(s, 0.5, y, 9.0, 0.92, C.background2, `${name} card`);
    s.addText(name, { x: 0.7, y, w: 2.2, h: 0.92, fontSize: 18, bold: true, color: col, valign: "middle",
      margin: 0, isTextBox: true, objectName: `${name} label` });
    s.addText([
      ...math(eq, { fontFace: HEAD, fontSize: 20, color: C.text2 }),
      { text: "", options: { breakLine: true } },
      { text: gloss, options: { fontSize: 13, italic: true, color: C.accent6 } },
    ], { x: 2.9, y: y + 0.08, w: 6.4, h: 0.76, valign: "middle", margin: 0, isTextBox: true,
      objectName: `${name} definition` });
  });
  s.addNotes("A battle is a zero-sum stochastic game with simultaneous moves, chance and hidden teams, so each turn " +
    "we face a stage matrix. Three yardsticks. Regret: the best expert in hindsight minus what we earned. " +
    "Exploitability: how much a best-responding opponent takes from us, relative to the game value. The duality gap " +
    "is the exploitability of both players together; it is zero exactly at equilibrium.");
}

// 3. From lecture: MWU → Hedge, and regret ⇒ equilibrium
{
  // Long title: 24 pt so it stays on one line.
  const s = pres.addSlide({ masterName: "SEC1", sectionTitle: SECTIONS[0] });
  s.addText([{ text: "From lecture: MWU → Hedge, and regret ⇒ equilibrium", options: { fontSize: 24 } }],
    { placeholder: "title" });
  card(s, 0.5, 1.2, 4.3, 1.4, C.background2, "MWU card");
  s.addText([
    { text: "MWU (lecture)", options: { bold: true, fontSize: 18, color: C.accent6, breakLine: true } },
    { text: "w ← w (1 − ηℓ)", options: { fontFace: HEAD, fontSize: 24, color: C.text1 } },
  ], { x: 0.7, y: 1.3, w: 3.9, h: 1.2, valign: "top", margin: 0, isTextBox: true, objectName: "MWU text" });

  card(s, 5.2, 1.2, 4.3, 1.4, C.background2, "Hedge card");
  s.addText([
    { text: "Hedge (ours)", options: { bold: true, fontSize: 18, color: C.accent1, breakLine: true } },
    { text: "w ← w · e", options: { fontFace: HEAD, fontSize: 24, color: C.text1 } },
    { text: "ηr", options: { fontFace: HEAD, fontSize: 24, color: C.text1, superscript: true, breakLine: true } },
    ...math("p_t ∝ exp(η · cumulative reward)", { fontFace: HEAD, fontSize: 16, color: C.text1 }),
  ], { x: 5.4, y: 1.3, w: 3.9, h: 1.2, valign: "top", margin: 0, isTextBox: true, objectName: "Hedge text" });

  s.addText([
    { text: "Up- and down-weighting coincide after normalising", options: { bullet: true, breakLine: true } },
    { text: "Bayes = Hedge with η = 1 on log loss", options: { bullet: true } },
  ], { x: 0.5, y: 2.75, w: 9.0, h: 0.75, fontSize: 15, color: C.text1, paraSpaceAfter: 4, valign: "top",
    margin: 0, isTextBox: true, objectName: "Hedge bullets" });

  card(s, 0.5, 3.6, 9.0, 1.35, C.text2, "Fact 1 card");
  s.addText([
    { text: "Fact 1  ", options: { bold: true, color: C.accent4 } },
    { text: "both players no-regret ⇒ time averages are an ε-equilibrium,  ", options: { color: C.background1 } },
    ...math("ε = (Reg_1 + Reg_2) / T", { fontFace: HEAD, bold: true, color: C.background1 }),
    { text: "", options: { breakLine: true } },
    { text: "averaged joint play is an ε-CCE, which in zero-sum games is minimax",
      options: { fontSize: 13, italic: true, color: C.accent3, breakLine: true } },
    { text: "CFR: regret bounds the exploitability of the average strategy",
      options: { fontSize: 13, italic: true, color: C.accent3 } },
  ], { x: 0.7, y: 3.65, w: 8.6, h: 1.25, fontSize: 16, valign: "middle", margin: 0, isTextBox: true,
    objectName: "Fact 1 text" });
  s.addNotes("Lecture gave us multiplicative weights, a linear update on losses. Our code uses Hedge, an exponential " +
    "update on rewards, so the strategy is a softmax of cumulative reward; up- and down-weighting coincide after " +
    "normalising. Bayesian updating is Hedge with step one on log loss. Fact 1: if both players have no regret, the " +
    "time averages are an epsilon-equilibrium, and CFR uses regret to bound exploitability.");
}

// =====================================================================
// Section 2: Method
pres.addSection({ title: SECTIONS[1] });

// 4. Three levels
{
  const s = content(2, "No-regret at three places");
  const boxes = [
    ["1", "Inside a turn", "Bayesian LP / RM⁺ per turn", "→ CFR lineage", C.accent1, null],
    ["2", "Across games", "Hedge over experts + style posterior", "→ experts problem", C.accent2, null],
    ["3", "Search root", "Bandits UCB1 / Exp3", "→ visit-share identity", C.accent5,
      "gap = (Reg^1 + Reg^2) / N"],
  ];
  const w = 2.8, gap = 0.3, y = 1.15, h = 2.55;
  boxes.forEach(([n, where, what, lin, col, eq], i) => {
    const x = 0.5 + i * (w + gap);
    card(s, x, y, w, h, C.background2, `Level ${n} card`);
    badge(s, n, x + 0.2, y + 0.15, 0.55, col, `Level ${n} badge`);
    s.addText(`Level ${n}`, { x: x + 0.9, y: y + 0.15, w: w - 1.1, h: 0.55, fontSize: 14, bold: true,
      color: col, valign: "middle", margin: 0, isTextBox: true, objectName: `Level ${n} label` });
    const runs = [
      { text: where, options: { bold: true, fontSize: 18, color: C.text2, breakLine: true } },
      { text: what, options: { fontSize: 15, color: C.text1, breakLine: true } },
      { text: lin, options: { fontSize: 14, italic: true, color: C.accent6 } },
    ];
    s.addText(runs, { x: x + 0.2, y: y + 0.8, w: w - 0.4, h: 1.3, valign: "top", paraSpaceAfter: 4,
      margin: 0, isTextBox: true, objectName: `Level ${n} text` });
    if (eq) s.addText(math(eq, { fontFace: HEAD, fontSize: 16, color: C.text2 }),
      { x: x + 0.2, y: y + 2.05, w: w - 0.4, h: 0.4, valign: "middle", margin: 0, isTextBox: true,
        objectName: `Level ${n} equation` });
  });
  s.addText([
    { text: "Bayesian matrix game:   ", options: { fontSize: 16, bold: true, color: C.accent1 } },
    ...math("max_x  Σ_k  β_k  min_j  (M_kᵀ x)_j", { fontFace: HEAD, fontSize: 24, color: C.text2 }),
  ], { x: 0.5, y: 3.85, w: 9.0, h: 0.6, align: "center", valign: "middle", margin: 0, isTextBox: true,
    objectName: "Bayesian LP equation" });
  s.addText("one LP per turn, vs PIMC averaging (solve each world as if known, then average → strategy fusion)",
    { x: 0.5, y: 4.48, w: 9.0, h: 0.35, align: "center", fontSize: 13, italic: true, color: C.accent6, margin: 0,
      isTextBox: true, objectName: "Equation caption" });
  s.addNotes("We use no-regret learning in three places. Level one, inside a turn: the Bayesian LP, or RM-plus " +
    "self-play. Its object is PokaiTrainer's Bayesian matrix game: one strategy x against an opponent that " +
    "best-responds in each sampled world, unlike PIMC averaging, which suffers strategy fusion. Level two: Hedge over " +
    "experts plus a style posterior. Level three: at a search root, the visit shares' duality gap equals summed regret over N.");
}

// 5. Level 2 is the experts problem
{
  const s = content(2, "Level 2 is the experts problem");
  const experts = [
    ["Top move", "argmax of x", C.accent1],
    ["Sampled", "play x", C.accent1],
    ["Pure maximin", "the security row", C.accent5],
    ["ε-safe best response", "to the predicted opponent q̂ (style posterior)", C.accent2],
  ];
  const w = 4.3, h = 0.8;
  experts.forEach(([name, rule, col], i) => {
    const x = 0.5 + (i % 2) * (w + 0.4), y = 1.15 + Math.floor(i / 2) * (h + 0.2);
    card(s, x, y, w, h, C.background2, `Expert ${i + 1} card`);
    s.addText([
      { text: name, options: { bold: true, fontSize: 16, color: col, breakLine: true } },
      { text: rule, options: { fontSize: 14, color: C.text1 } },
    ], { x: x + 0.2, y: y + 0.08, w: w - 0.4, h: h - 0.16, valign: "middle", margin: 0, isTextBox: true,
      objectName: `Expert ${i + 1} text` });
  });
  card(s, 0.5, 3.1, 9.0, 1.85, C.text2, "Hedge card");
  s.addText([
    { text: "Hedge:  ", options: { fontSize: 18, bold: true, color: C.accent4 } },
    ...math("p = Σ_e w_e q_e", { fontFace: HEAD, fontSize: 22, color: C.background1 }),
    { text: "", options: { breakLine: true } },
    ...math("full-information feedback r_e from the observed opponent action", { fontSize: 15, color: C.background1 }),
    { text: "", options: { breakLine: true } },
    { text: "weights kept across the games of a match", options: { fontSize: 15, color: C.background1, breakLine: true } },
    { text: "As in the lecture's experts problem, with each fixed action replaced by a fixed rule; " +
      "Hedge competes with the best rule in hindsight", options: { fontSize: 13, italic: true, color: C.accent3 } },
  ], { x: 0.7, y: 3.17, w: 8.6, h: 1.71, valign: "middle", margin: 0, isTextBox: true,
    objectName: "Hedge text" });
  s.addNotes("Level two is the lecture's experts problem. Each expert is a fixed rule applied to the turn's solved " +
    "game: top move, sampled, pure maximin, and an epsilon-safe best response to the predicted opponent. Hedge mixes " +
    "them with full-information feedback: each rule earns what it would have earned against the " +
    "opponent's observed action. Weights carry across the games of a match, so Hedge competes with the best rule in hindsight.");
}

// 6. Refinements: the Bayesian opponent model
{
  const s = content(2, "Refinements: the Bayesian opponent model");
  const rows = [
    ["Style posterior", [
      { text: "5 behavioural types: stage equilibrium, best response to uniform, logit best response to our " +
        "mixture, cautious (maximin), uniform", options: { fontSize: 13, color: C.text1, breakLine: true } },
      { text: "Bayes = Hedge with η = 1 on log loss  ⇒  regret ≤ ln 5",
        options: { fontFace: HEAD, fontSize: 15, color: C.text2 } },
    ], 1.0, C.accent1],
    ["q̂ → ε-safe best response", [
      { text: "exploit the predicted opponent, keeping the stage value minus 0.3% of the payoff range",
        options: { fontSize: 14, color: C.text1 } },
    ], 0.66, C.accent2],
    ["Per-world style likelihood", [
      { text: "score each style on each sampled world's own legal actions, not on the world-averaged matrix",
        options: { fontSize: 14, color: C.text1 } },
    ], 0.66, C.accent5],
    ["Forgetting (λ < 1) / fixed share", [
      { text: "track an opponent that switches style → sets up the detection-delay result",
        options: { fontSize: 14, color: C.text1 } },
    ], 0.66, C.accent6],
  ];
  let y = 1.15;
  rows.forEach(([name, runs, h, col], i) => {
    card(s, 0.5, y, 9.0, h, C.background2, `Refinement ${i + 1} card`);
    s.addText(name, { x: 0.7, y, w: 2.5, h, fontSize: 15, bold: true, color: col, valign: "middle", margin: 0,
      isTextBox: true, objectName: `Refinement ${i + 1} label` });
    s.addText(runs, { x: 3.3, y: y + 0.05, w: 6.0, h: h - 0.1, valign: "middle", margin: 0, isTextBox: true,
      objectName: `Refinement ${i + 1} text` });
    y += h + 0.12;
  });
  s.addNotes("The opponent model is Bayesian. A posterior over five behavioural types predicts the opponent; Bayes is " +
    "Hedge with step one on log loss, so its regret is at most log five. That prediction feeds the epsilon-safe best " +
    "response, which exploits while keeping the stage value minus 0.3 percent of the range. The per-world likelihood " +
    "scores styles on each world's legal actions. Forgetting or fixed share tracks a switching opponent.");
}

// =====================================================================
// Section 3: Experiments
pres.addSection({ title: SECTIONS[2] });

// 7. Engine experiments
{
  const s = content(3, "In the engine");
  const stats = [
    ["0.760 / 0.650", "MCTS at matched time beats equilibrium play (vs sampled / top move)", C.accent2],
    ["0.250 vs 0.090", "95% lower confidence bounds on exploitability: top move vs sampled", C.accent1],
    ["0.585", "Bayesian LP beats PIMC averaging (sampled play, 400 games)", C.accent1],
    ["6.9–155×", "Exact endgames: the bound holds, loose by this factor; consistent with the leaf evaluation error dominating",
      C.accent5],
  ];
  const w = 4.3, h = 1.7;
  stats.forEach(([big, label, col], i) => {
    const x = 0.5 + (i % 2) * (w + 0.4), y = 1.15 + Math.floor(i / 2) * (h + 0.2);
    card(s, x, y, w, h, C.background2, `Stat ${i + 1} card`);
    s.addText(big, { x: x + 0.25, y: y + 0.12, w: w - 0.5, h: 0.65, fontFace: HEAD, fontSize: 34, bold: true,
      color: col, valign: "middle", margin: 0, isTextBox: true, objectName: `Stat ${i + 1} number` });
    s.addText(label, { x: x + 0.25, y: y + 0.82, w: w - 0.5, h: 0.8, fontSize: 14, color: C.text1,
      valign: "top", margin: 0, isTextBox: true, objectName: `Stat ${i + 1} label` });
  });
  s.addNotes("In-engine results. At matched time, MCTS beats equilibrium play: 0.760 against sampled and 0.650 against " +
    "top move. An informed best response gives 95 percent lower confidence bounds on exploitability of 0.250 for top " +
    "move and 0.090 for sampled play, so randomising helps. The Bayesian LP beats PIMC averaging, 0.585 over 400 games. " +
    "On exact endgames the bound holds but is loose by 6.9 to 155 times.");
}

// 8. Real Showdown
{
  const s = content(3, "Real Showdown and pre-registered tests");
  // 680 x 450 px figure
  const fw = 4.5, fh = fw * 450 / 680;
  s.addImage({ path: path.join(FIG, "fig_gauntlet_col.png"), x: 0.5, y: 1.2, w: fw, h: fh,
    altText: "Round-robin Bradley-Terry ratings: Foul Play at all search times above Bayesian LP; top move and PIMC near 0",
    objectName: "Round-robin ratings figure" });
  s.addText("Round-robin tournament, 1,800 games; Elo-scale ratings, 95% CIs", { x: 0.5, y: 1.25 + fh, w: fw,
    h: 0.3, fontSize: 11, italic: true, color: C.accent6, margin: 0, isTextBox: true, objectName: "Figure caption" });
  s.addText([
    { text: "Below Foul Play, even at its shortest search time", options: { bullet: true, breakLine: true } },
    { text: "Level with the other one-turn solvers", options: { bullet: true, breakLine: true } },
    { text: "Hedge regret 0.0044 per scored turn vs its best expert", options: { bullet: true, breakLine: true } },
    { text: "Detection delay matches its prediction per match (ρ = 0.86); no prediction refuted",
      options: { bullet: true, breakLine: true } },
    { text: "Better opponent prediction (≈0.3 nats per scored turn), no detectable win-rate change (replicated)",
      options: { bullet: true } },
  ], { x: 5.3, y: 1.2, w: 4.2, h: 3.75, fontSize: 15, color: C.text1, paraSpaceAfter: 8, valign: "top",
    margin: 0, isTextBox: true, objectName: "Showdown bullets" });
  s.addNotes("On real Showdown, a round-robin tournament of 1,800 games: Foul Play at every search time rates above " +
    "our Bayesian LP, which is level with the other one-turn solvers. Hedge's regret against its best expert is 0.0044 " +
    "per scored turn. The pre-registered detection-delay test refuted no prediction, rho 0.86. Better opponent " +
    "prediction, about 0.3 nats per scored turn, gave no detectable win-rate change, and the null replicated.");
}

// =====================================================================
// Section 4: Theory
pres.addSection({ title: SECTIONS[3] });

// 9. The bound and the anchors
{
  const s = content(4, "The bound, and how it builds on the anchors");
  card(s, 0.5, 1.15, 9.0, 0.85, C.text2, "Bound card");
  s.addText(math("expl(π)  ≤  H̄ · sup (ε_s + 2ε_v)", { fontFace: HEAD, fontSize: 30, color: C.background1 }),
    { x: 0.5, y: 1.15, w: 9.0, h: 0.85, align: "center", valign: "middle", margin: 0, isTextBox: true,
      objectName: "Bound equation" });
  const rows = [
    ["ε_s", "stage exploitability: shrinks with compute (regret ⇒ equilibrium; CFR)", C.accent1],
    ["ε_v", "stage-matrix error from the heuristic evaluation: does not", C.accent2],
    ["H̄", "expected length against a best response (tight)", C.accent5],
  ];
  rows.forEach(([sym, txt, col], i) => {
    const y = 2.12 + i * 0.5;
    s.addShape(pres.shapes.OVAL, { x: 0.8, y, w: 0.42, h: 0.42, fill: { color: col }, line: { color: col },
      objectName: `Term ${i + 1} badge` });
    s.addText(math(sym, { fontFace: HEAD, fontSize: 14, bold: true, color: C.background1 }),
      { x: 0.8, y, w: 0.42, h: 0.42, align: "center", valign: "middle", margin: 0, isTextBox: true,
        objectName: `Term ${i + 1} symbol` });
    s.addText(txt, { x: 1.4, y, w: 8.1, h: 0.42, fontSize: 16, color: C.text1, valign: "middle", margin: 0,
      isTextBox: true, objectName: `Term ${i + 1} text` });
  });
  const anchors = [
    ["Freund & Schapire", "→ Fact 1 / ε_s"],
    ["Zinkevich et al. (CFR)", "→ exploitability metric; regret ⇒ exploitability"],
    ["Cesa-Bianchi & Lugosi", "→ Hedge, Bayes = Hedge, tracking (detection delay)"],
    ["PokaiTrainer", "→ the Bayesian matrix game"],
  ];
  const aw = 2.1, ag = 0.2;
  anchors.forEach(([who, what], i) => {
    const x = 0.5 + i * (aw + ag);
    card(s, x, 3.68, aw, 0.95, C.background2, `Anchor ${i + 1} card`);
    s.addText([
      { text: who, options: { bold: true, fontSize: 12, color: C.text2, breakLine: true } },
      ...math(what, { fontSize: 11, color: C.text1 }),
    ], { x: x + 0.1, y: 3.73, w: aw - 0.2, h: 0.85, valign: "top", margin: 0, isTextBox: true,
      objectName: `Anchor ${i + 1} text` });
  });
  s.addText([
    { text: "plus: hidden-information version; bandit-root identity ", options: {} },
    ...math("gap = (Reg^1 + Reg^2)/N", { fontFace: HEAD }),
    { text: "; core step Lean-checked", options: {} },
  ], { x: 0.5, y: 4.72, w: 9.0, h: 0.3, fontSize: 12, italic: true, color: C.accent6, margin: 0, isTextBox: true,
    objectName: "Theory footer" });
  s.addNotes("The headline theorem: exploitability is at most H-bar times the sup of epsilon-s plus two epsilon-v. " +
    "Epsilon-s, the stage exploitability, shrinks with compute, by Freund and Schapire's regret-to-equilibrium fact and " +
    "CFR. Epsilon-v, the stage-matrix error from the heuristic evaluation, does not. H-bar, the length against a best " +
    "response, is tight. Hedge and tracking come from Cesa-Bianchi and Lugosi, the Bayesian matrix game from PokaiTrainer.");
}

// =====================================================================
// Section 5: Conclusions
pres.addSection({ title: SECTIONS[4] });

// 10. Takeaways and next steps
{
  const s = content(5, "Takeaways and next steps");
  const items = [
    ["1", "No-regret learning composes cleanly across three levels; the measured regrets are small", C.accent1],
    ["2", "Bound and experiments are consistent with the stage-matrix (leaf) error dominating; better beliefs don't move win rate", C.accent2],
    ["3", "Next: an evaluation function trained on exact terminal values; log the opponent's side to measure two-sided regret more fully", C.accent5],
  ];
  items.forEach(([n, txt, col], i) => {
    const y = 1.3 + i * 1.0;
    badge(s, n, 0.6, y, 0.6, col, `Takeaway ${n} badge`);
    s.addText(txt, { x: 1.5, y: y - 0.1, w: 7.9, h: 0.8, fontSize: 18, color: C.background1, valign: "middle",
      margin: 0, isTextBox: true, objectName: `Takeaway ${n} text` });
  });
  s.addText("Also in the paper: joint set×style belief and lookahead-posterior nulls; case study", { x: 0.5, y: 4.6,
    w: 9.0, h: 0.35, fontSize: 13, italic: true, color: C.accent3, margin: 0, isTextBox: true, objectName: "Also line" });
  s.addNotes("Three takeaways. First, no-regret learning composes cleanly across a turn, a match and a search root, " +
    "and the measured regrets are small. Second, the bound and the experiments are consistent with the stage-matrix " +
    "error from the leaf evaluation dominating: better beliefs improve prediction, not win rate. Third, next steps: an " +
    "evaluation function trained on exact terminal values, and logging the opponent's side to measure two-sided regret. Thank you.");
}

(async () => {
  await pres.writeFile({ fileName: OUT });
  await applyTheme(OUT, THEME);
  console.log("wrote", OUT);
})();
