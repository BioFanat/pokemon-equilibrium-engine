"""Build slides.pptx (10 slides, ~4 min). Native, editable text boxes, shapes, tables and charts.

  uv run --with python-pptx python make_slides.py      # -> slides.pptx
Numbers come from ../engine/RESULTS.md; GRAY is the gray-box RL curve (results/psro/run_rlbr_gray.log).
"""
from pptx import Presentation
from pptx.chart.data import CategoryChartData
from pptx.dml.color import RGBColor
from pptx.enum.chart import XL_CHART_TYPE, XL_LEGEND_POSITION, XL_LABEL_POSITION, XL_MARKER_STYLE
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.util import Inches, Pt

DARK, WHITE, TINT = RGBColor(0x1B, 0x24, 0x33), RGBColor(0xFF, 0xFF, 0xFF), RGBColor(0xEE, 0xF1, 0xF5)
ACCENT, BLUE, BODY, MUTED = RGBColor(0xB5, 0x46, 0x2F), RGBColor(0x2F, 0x66, 0x90), RGBColor(0x33, 0x3D, 0x4D), RGBColor(0x5F, 0x68, 0x78)
ON_DARK, ON_DARK_MUTED = RGBColor(0xE6, 0xEA, 0xF0), RGBColor(0xA9, 0xB4, 0xC4)
HEAD, TEXT = "Cambria", "Calibri"

# gray-box RL curve so far (decisions in M, win rate); None = not reached yet
BLACK = [0.060, 0.065, 0.090, 0.100, 0.1225, 0.125]          # at 0, 1.00, 3.01, 4.94, 6.00, 8.01 M
GRAY = [0.055, 0.0925, 0.0975, 0.0825, 0.0825, 0.075]       # at 0, 1.01, 3.01, 5.00, 6.00, 8.01 M

prs = Presentation()
prs.slide_width, prs.slide_height = Inches(13.333), Inches(7.5)
BLANK = prs.slide_layouts[6]
L, TOP, W = Inches(0.7), Inches(0.55), Inches(11.93)


def bg(slide, color):
    f = slide.background.fill
    f.solid(); f.fore_color.rgb = color


def text(slide, x, y, w, h, runs, size=16, color=BODY, font=TEXT, bold=False, align=PP_ALIGN.LEFT,
         anchor=MSO_ANCHOR.TOP, name=None, spacing=None):
    """runs: str, or list of paragraphs; a paragraph is str or list of (text, {bold, color, italic, size})."""
    tb = slide.shapes.add_textbox(x, y, w, h)
    if name:
        tb.name = name
    tf = tb.text_frame
    tf.word_wrap, tf.vertical_anchor = True, anchor
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    paras = [runs] if isinstance(runs, str) else runs
    for i, para in enumerate(paras):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        if spacing:
            p.space_after = Pt(spacing)
        for t, o in ([(para, {})] if isinstance(para, str) else para):
            r = p.add_run(); r.text = t
            r.font.size, r.font.name = Pt(o.get("size", size)), o.get("font", font)
            r.font.bold, r.font.italic = o.get("bold", bold), o.get("italic", False)
            r.font.color.rgb = o.get("color", color)
            if o.get("sub") or o.get("sup"):
                r.font._element.set("baseline", "-25000" if o.get("sub") else "30000")
    return tb


def box(slide, x, y, w, h, fill, name=None, shape=MSO_SHAPE.RECTANGLE):
    s = slide.shapes.add_shape(shape, x, y, w, h)
    s.fill.solid(); s.fill.fore_color.rgb = fill
    s.line.fill.background(); s.shadow.inherit = False
    if name:
        s.name = name
    return s


def header(slide, eyebrow, title, dark=False):
    text(slide, L, TOP, W, Inches(0.35), eyebrow.upper(), size=13, bold=True, color=ACCENT, name="Eyebrow")
    text(slide, L, TOP + Inches(0.38), W, Inches(0.9), title, size=36, bold=True, font=HEAD,
         color=WHITE if dark else DARK, name="Title")


def notes(slide, s):
    slide.notes_slide.notes_text_frame.text = s


def card(slide, x, y, w, h, label, head, body, hot=False):
    box(slide, x, y, w, h, TINT, name=f"Card {label}")
    pad = Inches(0.3)
    text(slide, x + pad, y + pad, w - 2 * pad, Inches(0.3), label.upper(), size=12, bold=True,
         color=ACCENT if hot else BLUE)
    text(slide, x + pad, y + pad + Inches(0.4), w - 2 * pad, Inches(0.9), head, size=22, bold=True, font=HEAD, color=DARK)
    text(slide, x + pad, y + pad + Inches(1.5), w - 2 * pad, h - Inches(2.0), body, size=16, color=BODY)


def style_chart(ch, size=14):
    ch.font.size, ch.font.name, ch.font.color.rgb = Pt(size), TEXT, BODY


# 1 ---------------------------------------------------------------- title
s = prs.slides.add_slide(BLANK); bg(s, DARK)
text(s, L, Inches(2.0), W, Inches(0.4), "CS535 · FINAL PROJECT", size=14, bold=True, color=RGBColor(0xE9, 0x96, 0x7F))
text(s, L, Inches(2.5), W, Inches(1.8), ["Equilibrium play in", "Pokémon Random Battles"], size=48, bold=True,
     font=HEAD, color=WHITE, name="Title")
text(s, L, Inches(4.35), W, Inches(0.6), "Per-turn no-regret solving, with whole-game exploitability bounds",
     size=22, color=ON_DARK)
text(s, L, Inches(6.4), W, Inches(0.4), "[Author names]", size=14, color=ON_DARK_MUTED)
notes(s, "Pokémon battles: two-player zero-sum, simultaneous moves, chance, hidden teams. Can an agent that "
         "solves a small game every turn, with guarantees, compete with the search bots people use?")

# 2 ---------------------------------------------------------------- question
s = prs.slides.add_slide(BLANK); bg(s, WHITE)
header(s, "The question", "Search bots are strong. Are they hard to exploit?")
cw, gap, y = Inches(3.77), Inches(0.31), Inches(2.1)
card(s, L, y, cw, Inches(3.9), "The game", "Simultaneous, random, hidden",
     "Both players move at once; damage rolls and crits add chance; the opponent’s team is unknown.")
card(s, L + cw + gap, y, cw, Inches(3.9), "Today’s bots", "MCTS search",
     "foul-play-style bots choose from search visit counts. Strong, but no guarantee against an opponent that adapts.")
card(s, L + 2 * (cw + gap), y, cw, Inches(3.9), "Our approach", "Solve a small game every turn",
     "Play a near-equilibrium strategy each turn, and bound how much the whole game can be exploited.", hot=True)
notes(s, "Two axes throughout: strength against fixed bots, and exploitability against an opponent that adapts. "
         "Search optimises the first; equilibrium play is built for the second.")

# 3 ---------------------------------------------------------------- agent
s = prs.slides.add_slide(BLANK); bg(s, WHITE)
header(s, "The agent", "Each turn: build a matrix, solve it, play")
steps = [("1", "Payoff matrix", "Every pair of moves, two turns of lookahead, then a heuristic position value."),
         ("2", "Solve with RM+", "Regret matching+, a no-regret method, finds a near-equilibrium mixed strategy."),
         ("3", "Hidden teams", "Sample K opponent teams and solve them as one game (agent form), not one by one."),
         ("4", "Hedge on top", "Online learner over {top move, sampled, safest}, updated from the opponent’s moves.")]
sw, sg, y = Inches(2.62), Inches(0.48), Inches(2.2)
for i, (n, h, b) in enumerate(steps):
    x = L + i * (sw + sg)
    c = box(s, x, y, Inches(0.75), Inches(0.75), ACCENT if i == 3 else BLUE, name=f"Step {n} number", shape=MSO_SHAPE.OVAL)
    c.text_frame.text = n
    p = c.text_frame.paragraphs[0]; p.alignment = PP_ALIGN.CENTER
    p.runs[0].font.size, p.runs[0].font.bold, p.runs[0].font.color.rgb, p.runs[0].font.name = Pt(24), True, WHITE, HEAD
    text(s, x, y + Inches(1.0), sw, Inches(0.5), h, size=22, bold=True, font=HEAD, color=DARK)
    text(s, x, y + Inches(1.55), sw, Inches(1.8), b, size=16)
    if i < 3:
        a = box(s, x + sw + Inches(0.08), y + Inches(0.22), Inches(0.32), Inches(0.3), MUTED, shape=MSO_SHAPE.RIGHT_ARROW)
        a.name = f"Arrow {i + 1}"
text(s, L, Inches(6.3), W, Inches(0.5),
     "Built on poke-engine 0.0.48. Its own search matrix is 55–80% pruned, so we recompute every cell.", size=14, color=MUTED)
notes(s, "The pruning point matters: treating poke-engine's search matrix as a payoff matrix silently breaks every "
         "equilibrium method, and it invalidated our first round of experiments.")

# 4 ---------------------------------------------------------------- theory
s = prs.slides.add_slide(BLANK); bg(s, WHITE)
header(s, "Theory", "Per-turn errors add up over the game")
box(s, L, Inches(1.95), W, Inches(1.25), TINT, name="Formula panel")
text(s, L, Inches(1.95), W, Inches(1.25),
     [[("exploitability  ≤  H̄ · ( ", {}), ("8√(n/T)", {"color": BLUE}), ("  +  ", {}), ("2ε", {"color": ACCENT}),
       ("v", {"color": ACCENT, "sub": True}), ("  +  ", {}), ("4√(2n log K / K)", {"color": BLUE}), (" )", {})]],
     size=32, font=HEAD, color=DARK, align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE, name="Formula")
terms = [("Solver", [[("Shrinks with RM+ iterations T. Measured on real turns: about T", {}), ("−1.3", {"sup": True}),
                      (", faster than the bound.", {})]], BLUE),
         ("Sampled teams", "Shrinks with the number of sampled worlds K. Zero under full information.", BLUE),
         ("Leaf value", "Does not shrink with compute or depth. Only a better value estimate removes it.", ACCENT)]
tw = Inches(3.77)
for i, (h, b, col) in enumerate(terms):
    x = L + i * (tw + Inches(0.31))
    text(s, x, Inches(3.6), tw, Inches(0.5), h, size=22, bold=True, font=HEAD, color=col)
    text(s, x, Inches(4.15), tw, Inches(1.5), b, size=16)
text(s, L, Inches(6.3), W, Inches(0.5),
     "H̄ = expected game length against a best-responding opponent; n = moves per side. One-turn lemma checked in Lean.",
     size=14, color=MUTED)
notes(s, "Key lemma: a strategy that is epsilon-optimal in a matrix that is off by epsilon-v is within epsilon plus "
         "two epsilon-v in the true matrix. The whole-game bound sums that over turns. Punchline: the leaf term "
         "does not go away with more search time.")

# 5 ---------------------------------------------------------------- hidden info
s = prs.slides.add_slide(BLANK); bg(s, WHITE)
header(s, "Result 1 · hidden information", "Joint solving beats averaging the worlds")
text(s, L, Inches(2.1), Inches(5.5), Inches(1.6), "0.585", size=96, bold=True, font=HEAD, color=ACCENT, name="Big number")
text(s, L, Inches(3.75), Inches(5.5), Inches(1.4),
     ["win rate of agent form vs PIMC", "95% CI [0.540, 0.630] · 400 games · equal compute (4 worlds each)"],
     size=16, color=BODY, spacing=4)
x2 = Inches(7.0)
box(s, x2, Inches(2.1), Inches(5.63), Inches(3.5), TINT, name="Bound panel")
text(s, x2 + Inches(0.35), Inches(2.4), Inches(4.9), Inches(0.4), "THE BOUND HOLDS ON REAL PLAY", size=12, bold=True, color=BLUE)
text(s, x2 + Inches(0.35), Inches(2.85), Inches(4.9), Inches(0.9), "46,890 decisions", size=36, bold=True, font=HEAD, color=DARK)
text(s, x2 + Inches(0.35), Inches(3.8), Inches(4.9), Inches(1.6),
     [[("PIMC’s loss never exceeded our 4δ", {}), ("H", {"sub": True}), (" bound.", {})], "Largest loss: 0.098 of the bound.",
      "Median loss: 0.19% of the payoff range."], size=16, spacing=6)
notes(s, "This is the central claim. PIMC solves each sampled world separately and averages; agent form solves one "
         "game over all worlds. Same compute, 0.585. The formulation is from PokaiTrainer; the head-to-head is ours.")

# 6 ---------------------------------------------------------------- strength
s = prs.slides.add_slide(BLANK); bg(s, WHITE)
header(s, "Result 2 · strength", "Matrix play beats fast MCTS, loses at equal time")
cd = CategoryChartData()
cd.categories = ["vs MCTS, 2,000 iterations (2.7 ms/move)", "vs MCTS, time-matched (75 ms/move)"]
cd.add_series("Equilibrium, top move", (0.58, 0.345))
cd.add_series("Safest move", (0.56, 0.263))
gf = s.shapes.add_chart(XL_CHART_TYPE.COLUMN_CLUSTERED, L, Inches(1.85), Inches(7.6), Inches(4.7), cd)
gf.name = "Strength chart"
ch = gf.chart; style_chart(ch)
ch.has_legend, ch.legend.position, ch.legend.include_in_layout = True, XL_LEGEND_POSITION.BOTTOM, False
va = ch.value_axis; va.minimum_scale, va.maximum_scale, va.major_unit = 0, 0.7, 0.1
va.has_major_gridlines = True; va.major_gridlines.format.line.color.rgb = RGBColor(0xDD, 0xE1, 0xE6)
va.tick_labels.number_format, va.tick_labels.number_format_is_linked = "0.0", False
va.has_title = True; va.axis_title.text_frame.text = "Win rate"
for ser, col in zip(ch.plots[0].series, (BLUE, MUTED)):
    ser.format.fill.solid(); ser.format.fill.fore_color.rgb = col
pl = ch.plots[0]; pl.has_data_labels = True; pl.gap_width = 80
pl.data_labels.number_format, pl.data_labels.number_format_is_linked = "0.00#", False
pl.data_labels.position = XL_LABEL_POSITION.OUTSIDE_END
text(s, Inches(8.8), Inches(2.1), Inches(3.83), Inches(4.2),
     [[("The leaf value is the bottleneck.", {"bold": True, "color": DARK, "size": 20, "font": HEAD})],
      "Given the same time per move, MCTS wins clearly.",
      "Sampling the equilibrium scores only 0.44 vs fast MCTS: against a fixed opponent, randomising costs value.",
      "A value net fit to game outcomes did not help (0.235 vs time-matched MCTS)."], size=16, spacing=10)
notes(s, "Against a non-adapting opponent, randomising only costs value. This matches the theory: the leaf term is "
         "the one compute can't shrink. CIs: 0.58 [0.54, 0.62], 0.345 [0.289, 0.401].")

# 7 ---------------------------------------------------------------- robustness
s = prs.slides.add_slide(BLANK); bg(s, WHITE)
header(s, "Result 3 · exploitability", "Randomising and Hedge resist exploitation")
cd = CategoryChartData()
cd.categories = ["Equilibrium, top move", "Equilibrium, sampled"]
cd.add_series("Exploiter win rate", (0.873, 0.713))
gf = s.shapes.add_chart(XL_CHART_TYPE.BAR_CLUSTERED, L, Inches(2.25), Inches(6.2), Inches(2.8), cd)
gf.name = "Exploiter chart"
ch = gf.chart; style_chart(ch); ch.has_legend = False; ch.has_title = False
va = ch.value_axis; va.minimum_scale, va.maximum_scale, va.major_unit = 0, 1.0, 0.25
va.has_major_gridlines = True; va.major_gridlines.format.line.color.rgb = RGBColor(0xDD, 0xE1, 0xE6)
va.tick_labels.number_format, va.tick_labels.number_format_is_linked = "0.00", False
ser = ch.plots[0].series[0]; ser.format.fill.solid(); ser.format.fill.fore_color.rgb = BLUE
pt = ser.points[0]; pt.format.fill.solid(); pt.format.fill.fore_color.rgb = ACCENT
pl = ch.plots[0]; pl.has_data_labels = True; pl.gap_width = 60
pl.data_labels.number_format, pl.data_labels.number_format_is_linked = "0.00#", False
ch.category_axis.reverse_order = True
text(s, L, Inches(1.85), Inches(6.2), Inches(0.4), "White-box exploiter’s win rate (lower = harder to exploit)",
     size=14, bold=True, color=MUTED)
text(s, L, Inches(5.25), Inches(6.2), Inches(0.8), "100 pairs; difference +0.160 [+0.091, +0.229].", size=14, color=MUTED)
x2 = Inches(7.6)
box(s, x2, Inches(1.85), Inches(5.03), Inches(4.2), TINT, name="Hedge panel")
text(s, x2 + Inches(0.35), Inches(2.15), Inches(4.3), Inches(0.4), "HEDGE, WEIGHTS CARRIED ACROSS GAMES", size=12, bold=True, color=ACCENT)
text(s, x2 + Inches(0.35), Inches(2.6), Inches(4.3), Inches(1.0), "+0.146", size=60, bold=True, font=HEAD, color=DARK)
text(s, x2 + Inches(0.35), Inches(3.75), Inches(4.3), Inches(2.1),
     ["score vs the exploiter, compared with resetting weights each game", "95% CI [+0.039, +0.253] · 120 games",
      "No cost against MCTS."], size=16, spacing=6)
notes(s, "The exploiter knows the bot's algorithm and best-responds one turn deep. Sampling is less exploitable "
         "than the top move. Hedge over experts with weights carried across games is significantly harder to "
         "exploit than resetting each game, which is what the regret theory covers.")

# 8 ---------------------------------------------------------------- links
s = prs.slides.add_slide(BLANK); bg(s, WHITE)
header(s, "Links to MCTS and RL", "Agent, MCTS and RL: three per-state regret learners")
rows = [("", "Learner at each state", "Guarantee vs an adaptive opponent"),
        ("Our agent", "RM+ / Hedge on the turn’s matrix", "Yes: regret + 2 × leaf error per turn"),
        ("MCTS (poke-engine)", "UCB1 per side at each node", "No: visit shares fail to converge on mixed games"),
        ("RL best response", "Policy gradient ≈ Hedge in every state", "Yes, but undirected exploration can need |B|^D games")]
tbl = s.shapes.add_table(4, 3, L, Inches(1.95), W, Inches(2.9)); tbl.name = "Comparison table"
t = tbl.table
for j, wd in enumerate((2.8, 4.3, 4.83)):
    t.columns[j].width = Inches(wd)
for i, r in enumerate(rows):
    for j, v in enumerate(r):
        c = t.cell(i, j); c.text = v
        c.fill.solid(); c.fill.fore_color.rgb = DARK if i == 0 else (TINT if i % 2 else WHITE)
        p = c.text_frame.paragraphs[0]
        for run in p.runs:
            run.font.size, run.font.name = Pt(16), TEXT
            run.font.bold = i == 0 or j == 0
            run.font.color.rgb = WHITE if i == 0 else (DARK if j == 0 else BODY)
        c.margin_left = c.margin_right = Inches(0.15)
        c.vertical_anchor = MSO_ANCHOR.MIDDLE
text(s, L, Inches(5.25), W, Inches(1.2),
     [[("One identity behind all three: ", {"bold": True, "color": DARK}),
       ("exploitability = expected sum of per-turn losses against a potential value function. With the game value it "
        "gives our bound; with the policy’s own value it is RL’s performance-difference lemma.", {})]], size=16)
notes(s, "Results L0 to L6 in the report. The search node result is exact: the Nash gap of the visit shares equals "
         "the two sides' summed average regret. poke-engine's UCB1 has no adversarial guarantee; in our tests its "
         "visit shares don't converge.")

# 9 ---------------------------------------------------------------- RL curve
s = prs.slides.add_slide(BLANK); bg(s, WHITE)
header(s, "RL as an exploitability probe", "8M decisions of RL barely dent foul-play")
cd = CategoryChartData()
cd.categories = ["0", "1M", "3M", "5M", "6M", "8M"]
cd.add_series("RL, black box", BLACK)
cd.add_series("RL, sees foul-play’s move distribution", GRAY)
gf = s.shapes.add_chart(XL_CHART_TYPE.LINE_MARKERS, L, Inches(1.85), Inches(7.6), Inches(4.7), cd)
gf.name = "RL curve chart"
ch = gf.chart; style_chart(ch)
ch.has_legend, ch.legend.position, ch.legend.include_in_layout = True, XL_LEGEND_POSITION.BOTTOM, False
va = ch.value_axis; va.minimum_scale, va.maximum_scale, va.major_unit = 0, 0.2, 0.05
va.has_major_gridlines = True; va.major_gridlines.format.line.color.rgb = RGBColor(0xDD, 0xE1, 0xE6)
va.tick_labels.number_format, va.tick_labels.number_format_is_linked = "0.00", False
va.has_title = True; va.axis_title.text_frame.text = "Win rate vs foul-play"
ch.category_axis.has_title = True; ch.category_axis.axis_title.text_frame.text = "Training decisions"
for ser, col in zip(ch.plots[0].series, (MUTED, ACCENT)):
    ser.format.line.color.rgb = col; ser.format.line.width = Pt(2.5); ser.smooth = False
    ser.marker.style, ser.marker.size = XL_MARKER_STYLE.CIRCLE, 8
    ser.marker.format.fill.solid(); ser.marker.format.fill.fore_color.rgb = col
    ser.marker.format.line.color.rgb = col
text(s, Inches(8.8), Inches(2.1), Inches(3.83), Inches(4.2),
     [[("A useful lower bound needs > 0.5.", {"bold": True, "color": DARK, "size": 20, "font": HEAD})],
      "147k training games move RL from 0.06 to 0.125.",
      "Giving RL foul-play’s exact move distribution doesn’t help: 0.075 at 8M.",
      "A one-turn search exploiter wins 0.725 with no training."], size=16, spacing=10)
notes(s, "Exploitability is usually measured with an RL best response. Here RL stays far below 0.5, so it certifies "
         "nothing. The gray-box run tests whether knowing the opponent's policy is the missing piece; it only gets the "
         "distribution, not the payoff matrix the search exploiter uses. The 0.725 search exploiter mixes "
         "exploitation with being a stronger bot.")

# 10 --------------------------------------------------------------- takeaways
s = prs.slides.add_slide(BLANK); bg(s, DARK)
header(s, "Takeaways", "What we learned", dark=True)
pts = [("Hidden information:", " solving sampled worlds as one game beats PIMC at equal compute (0.585)."),
       ("Theory:", " per-turn errors add up; solver and sampling terms shrink, the leaf term doesn’t."),
       ("Trade-off:", " MCTS is stronger at equal time; equilibrium play and Hedge are harder to exploit."),
       ("Measurement:", " white-box search exploiters beat RL best responses as an exploitability probe.")]
text(s, L, Inches(2.0), W, Inches(4.0), [[(a, {"bold": True, "color": WHITE}), (b, {})] for a, b in pts],
     size=22, color=ON_DARK, spacing=18)
text(s, L, Inches(6.4), W, Inches(0.4), "Next: a better leaf value, and belief-aware values for hidden information.",
     size=16, color=ON_DARK_MUTED)
notes(s, "One-line summary: the solver is not the bottleneck; the value estimate and the beliefs are.")

prs.save("slides.pptx")
print("wrote slides.pptx")
