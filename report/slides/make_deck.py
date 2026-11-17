"""Build deck.pptx: the 10-slide, ~4-minute CS535 final-project talk, in five sections.

Built on the user's template ("random template.pptx"): its master, theme, 16:9 size,
title-slide style and its five-box section tracker, which lives in the template's
"1_Title and Content" ... "5_Title and Content" layouts. Only the tracker labels change.
Content boxes copy the template's slide-3 shape formats (accent5 header boxes, bg2 boxes,
tx1 outlines and text, 2 pt straight connectors). Every number is taken from ../final5.tex.

Run (any working directory): <python with python-pptx> make_deck.py
"""
import copy
import os
import re

from lxml import etree
from pptx import Presentation
from pptx.oxml import parse_xml
from pptx.util import Emu, Inches

HERE = os.path.dirname(os.path.abspath(__file__))
TEMPLATE = os.path.join(HERE, "random template.pptx")
OUT = os.path.join(HERE, "deck.pptx")
FIG = os.path.join(HERE, "..", "figures", "fig_gauntlet_col.png")

SECTIONS = ["Preliminaries", "Method", "Experiments", "Theory", "Conclusions"]
# Template layout that highlights each section's tracker box (box 1 ... box 5).
SECTION_LAYOUT = ["1_Title and Content", "2_Title and Content", "3_Title and Content",
                  "4_Title and Content", "5_Title and Content"]
TRACKER_LAYOUTS = ["Title and Content"] + SECTION_LAYOUT

# Title slide: the template's slide 1 has a dark accent5 background with white text.
TITLE_DARK = True

# Size scale (pt).
TITLE, BODY, SECONDARY, CAPTION, EQ = 36, 20, 16, 12, 22
BOX = 18  # the template's own text size inside its boxes (inherited there; written explicitly here)

NS = ('xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" '
      'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" '
      'xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main"')
HN = ('<a:latin typeface="Helvetica Neue" panose="02000503000000020004" pitchFamily="2" charset="0"/>'
      '<a:ea typeface="Helvetica Neue" panose="02000503000000020004" pitchFamily="2" charset="0"/>'
      '<a:cs typeface="Helvetica Neue" panose="02000503000000020004" pitchFamily="2" charset="0"/>')
# Shape formats copied verbatim from the template's slide 3.
FILL_HEADER = '<a:solidFill><a:schemeClr val="accent5"><a:lumMod val="60000"/><a:lumOff val="40000"/></a:schemeClr></a:solidFill>'
FILL_BOX = '<a:solidFill><a:schemeClr val="bg2"/></a:solidFill>'
RECT_STYLE = ('<p:style><a:lnRef idx="2"><a:schemeClr val="accent1"><a:shade val="50000"/></a:schemeClr></a:lnRef>'
              '<a:fillRef idx="1"><a:schemeClr val="accent1"/></a:fillRef><a:effectRef idx="0"><a:schemeClr val="accent1"/>'
              '</a:effectRef><a:fontRef idx="minor"><a:schemeClr val="lt1"/></a:fontRef></p:style>')
LINE_STYLE = ('<p:style><a:lnRef idx="1"><a:schemeClr val="accent1"/></a:lnRef><a:fillRef idx="0"><a:schemeClr val="accent1"/>'
              '</a:fillRef><a:effectRef idx="0"><a:schemeClr val="accent1"/></a:effectRef><a:fontRef idx="minor">'
              '<a:schemeClr val="tx1"/></a:fontRef></p:style>')

# Content frame (EMU): the template's title / tracker left and right edges.
LEFT, RIGHT = 516924, 11675075
WIDTH = RIGHT - LEFT
GAP = Inches(0.4)


def esc(s):
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


# ---------- text: paragraphs and runs ----------
def math_runs(src):
    """Split 'M_k^T', 'β_{k}', 'Reg^{1}' into (text, baseline) pieces: sub -25 %, super +30 %."""
    out, buf, i = [], "", 0
    while i < len(src):
        ch = src[i]
        if ch in "_^" and i + 1 < len(src):
            if buf:
                out.append((buf, 0))
                buf = ""
            if src[i + 1] == "{":
                j = src.index("}", i)
                tok, i = src[i + 2:j], j + 1
            else:
                tok, i = src[i + 1], i + 2
            out.append((tok, -25000 if ch == "_" else 30000))
        else:
            buf += ch
            i += 1
    if buf:
        out.append((buf, 0))
    return out


def run(text, size=None, b=False, i=False, color="tx1", baseline=0):
    attrs = f'lang="en-US" sz="{(size or BOX) * 100}"'
    if b:
        attrs += ' b="1"'
    if i:
        attrs += ' i="1"'
    if baseline:
        attrs += f' baseline="{baseline}"'
    fill = f'<a:solidFill><a:schemeClr val="{color}"/></a:solidFill>'
    sp = ' xml:space="preserve"' if text != text.strip() else ""
    return f'<a:r><a:rPr {attrs} dirty="0">{fill}{HN}</a:rPr><a:t{sp}>{esc(text)}</a:t></a:r>'


def P(*pieces, size=None, b=False, i=False, color="tx1", align="ctr", bullet=False, before=0):
    """One paragraph. Each piece is a str (plain), ('$', src) (math) or (text, {overrides})."""
    runs = []
    for pc in pieces:
        if isinstance(pc, tuple) and pc[0] == "$":
            for t, bl in math_runs(pc[1]):
                runs.append(run(t, EQ, False, False, color, bl))
        elif isinstance(pc, tuple):
            o = {"size": size, "b": b, "i": i, "color": color, **pc[1]}
            runs.append(run(pc[0], **o))
        else:
            runs.append(run(pc, size, b, i, color))
    ppr = f'algn="{align}"'
    inner = ""
    if bullet:
        ppr += ' marL="285750" indent="-285750"'
        inner = '<a:buFont typeface="Arial" panose="020B0604020202020204" pitchFamily="34" charset="0"/><a:buChar char="&#8226;"/>'
    if before:
        inner = f'<a:spcBef><a:spcPts val="{before * 100}"/></a:spcBef>' + inner
    return f'<a:p><a:pPr {ppr}>{inner}</a:pPr>{"".join(runs)}</a:p>'


def EQP(src, align="ctr", before=0):
    """An equation on its own line: Helvetica Neue, 22 pt, Unicode math."""
    return P(("$", src), align=align, before=before)


# ---------- shapes ----------
class Shapes:
    """Appends template-style shapes to one slide's spTree with unique ids."""

    def __init__(self, slide):
        self.tree = slide.shapes._spTree
        self.next_id = 100

    def _id(self):
        self.next_id += 1
        return self.next_id

    def _add(self, xml):
        el = parse_xml(xml)
        self.tree.append(el)
        return el

    def rect(self, x, y, w, h, paras, fill, name, anchor="ctr"):
        sid = self._id()
        return self._add(
            f'<p:sp {NS}><p:nvSpPr><p:cNvPr id="{sid}" name="{esc(name)}"/><p:cNvSpPr/><p:nvPr/></p:nvSpPr>'
            f'<p:spPr><a:xfrm><a:off x="{int(x)}" y="{int(y)}"/><a:ext cx="{int(w)}" cy="{int(h)}"/></a:xfrm>'
            f'<a:prstGeom prst="rect"><a:avLst/></a:prstGeom>{fill}'
            f'<a:ln><a:solidFill><a:schemeClr val="tx1"/></a:solidFill></a:ln></p:spPr>{RECT_STYLE}'
            f'<p:txBody><a:bodyPr rtlCol="0" anchor="{anchor}"/><a:lstStyle/>{"".join(paras)}</p:txBody></p:sp>')

    def header(self, x, y, w, h, paras, name):
        """Template slide-3 header box: light accent5 fill, tx1 outline, bold tx1 text."""
        return self.rect(x, y, w, h, paras, FILL_HEADER, name)

    def box(self, x, y, w, h, paras, name):
        """Template slide-3 content box: bg2 fill, tx1 outline, tx1 text."""
        return self.rect(x, y, w, h, paras, FILL_BOX, name)

    def text(self, x, y, w, h, paras, name, anchor="t"):
        sid = self._id()
        return self._add(
            f'<p:sp {NS}><p:nvSpPr><p:cNvPr id="{sid}" name="{esc(name)}"/><p:cNvSpPr txBox="1"/><p:nvPr/></p:nvSpPr>'
            f'<p:spPr><a:xfrm><a:off x="{int(x)}" y="{int(y)}"/><a:ext cx="{int(w)}" cy="{int(h)}"/></a:xfrm>'
            f'<a:prstGeom prst="rect"><a:avLst/></a:prstGeom><a:noFill/></p:spPr>'
            f'<p:txBody><a:bodyPr wrap="square" rtlCol="0" anchor="{anchor}"><a:noAutofit/></a:bodyPr><a:lstStyle/>'
            f'{"".join(paras)}</p:txBody></p:sp>')

    def line(self, x1, y1, x2, y2, name, colour="tx1"):
        """Template slide-3 straight connector: 2 pt tx1 line."""
        sid = self._id()
        flip = ' flipV="1"' if y2 < y1 else ""
        x, y = min(x1, x2), min(y1, y2)
        return self._add(
            f'<p:cxnSp {NS}><p:nvCxnSpPr><p:cNvPr id="{sid}" name="{esc(name)}"/><p:cNvCxnSpPr/><p:nvPr/></p:nvCxnSpPr>'
            f'<p:spPr><a:xfrm{flip}><a:off x="{int(x)}" y="{int(y)}"/><a:ext cx="{int(abs(x2 - x1))}" cy="{int(abs(y2 - y1))}"/>'
            f'</a:xfrm><a:prstGeom prst="line"><a:avLst/></a:prstGeom><a:ln w="25400"><a:solidFill>'
            f'<a:schemeClr val="{colour}"/></a:solidFill></a:ln></p:spPr>{LINE_STYLE}</p:cxnSp>')

    def slide_number(self, n):
        """Slide-number placeholder; position, size and font come from the layout."""
        sid = self._id()
        self._add(
            f'<p:sp {NS}><p:nvSpPr><p:cNvPr id="{sid}" name="Slide Number Placeholder"/><p:cNvSpPr>'
            f'<a:spLocks noGrp="1"/></p:cNvSpPr><p:nvPr><p:ph type="sldNum" sz="quarter" idx="4"/></p:nvPr></p:nvSpPr>'
            f'<p:spPr/><p:txBody><a:bodyPr/><a:lstStyle/><a:p><a:fld id="{{7E75810F-9510-A844-820C-E3462544137D}}" '
            f'type="slidenum"><a:rPr lang="en-US"/><a:t>{n}</a:t></a:fld><a:endParaRPr lang="en-US" dirty="0"/></a:p>'
            f'</p:txBody></p:sp>')


def set_text(shape, paras):
    """Replace a placeholder's paragraphs with our own."""
    txBody = shape.text_frame._txBody
    for p in txBody.findall("{http://schemas.openxmlformats.org/drawingml/2006/main}p"):
        txBody.remove(p)
    for p in paras:
        txBody.append(parse_xml(p.replace("<a:p>", f"<a:p {NS}>", 1)))


# ---------- template preparation ----------
def set_theme_fonts(theme_part):
    xml = theme_part.blob.decode("utf-8")
    hn = '<a:latin typeface="Helvetica Neue" panose="02000503000000020004"/>'
    xml = re.sub(r'(<a:majorFont>)<a:latin [^>]*/>', r"\1" + hn, xml)
    xml = re.sub(r'(<a:minorFont>)<a:latin [^>]*/>', r"\1" + hn, xml)
    theme_part._blob = xml.encode("utf-8")


def relabel_tracker(layout):
    """Set the five tracker boxes' labels; shapes, fills and run formatting stay the template's."""
    boxes = [sp for sp in layout.shapes if sp.name.startswith("Rectangle") and sp.has_text_frame]
    boxes.sort(key=lambda sp: sp.left)
    assert len(boxes) == 5, layout.name
    for sp, label in zip(boxes, SECTIONS):
        runs = sp.text_frame.paragraphs[0].runs
        runs[0].text = label
        for r in runs[1:]:
            r._r.getparent().remove(r._r)


def delete_all_slides(prs):
    sldIdLst = prs.slides._sldIdLst
    for sldId in list(sldIdLst):
        rId = sldId.rId
        sldIdLst.remove(sldId)
        prs.part.drop_rel(rId)


def add_sections(prs, groups):
    """Write PowerPoint sections (p14:sectionLst) from [(name, [slide, ...]), ...]."""
    p14 = "http://schemas.microsoft.com/office/powerpoint/2010/main"
    pres = prs.part._element
    ext_lst = pres.find("{http://schemas.openxmlformats.org/presentationml/2006/main}extLst")
    secs = "".join(
        f'<p14:section name="{esc(name)}" id="{{8C7F1A2B-0000-4000-8000-00000000000{k}}}"><p14:sldIdLst>'
        + "".join(f'<p14:sldId id="{s.slide_id}"/>' for s in slides)
        + "</p14:sldIdLst></p14:section>"
        for k, (name, slides) in enumerate(groups, 1))
    ext = parse_xml(f'<p:ext {NS} uri="{{521415D9-36F7-43E2-AB2F-B90AF26B5E84}}"><p14:sectionLst xmlns:p14="{p14}">'
                    f'{secs}</p14:sectionLst></p:ext>')
    ext_lst.insert(0, ext)


# ---------- slides ----------
prs = Presentation(TEMPLATE)
master = prs.slide_master
set_theme_fonts(master.part.part_related_by(
    "http://schemas.openxmlformats.org/officeDocument/2006/relationships/theme"))
set_theme_fonts(prs.notes_master.part.part_related_by(
    "http://schemas.openxmlformats.org/officeDocument/2006/relationships/theme"))
layouts = {l.name: l for l in master.slide_layouts}
for name in TRACKER_LAYOUTS:
    relabel_tracker(layouts[name])
delete_all_slides(prs)

groups = [(s, []) for s in SECTIONS]


def content(sec, title):
    """A slide on its section's tracker layout, with the title and slide number filled in."""
    s = prs.slides.add_slide(layouts[SECTION_LAYOUT[sec - 1]])
    set_text(s.shapes.title, [P(title, size=TITLE, align="l")])
    S = Shapes(s)
    S.slide_number(len(prs.slides))
    groups[sec - 1][1].append(s)
    return s, S


def notes(slide, text):
    slide.notes_slide.notes_text_frame.text = text


TOP = Inches(1.7)  # first content row, below the template's title rule (1.48 in)
HDR_H = Inches(0.42)  # template slide-3 header box (0.38 in), 0.04 in taller so 18 pt fits
COL2 = (WIDTH - GAP) // 2
COL3 = (WIDTH - 2 * GAP) // 3
COL4 = (WIDTH - 3 * GAP) // 4

# ===================== Section 1: Preliminaries =====================
# 1. Title slide (template slide-1 style, no tracker)
s = prs.slides.add_slide(layouts["Title Slide"])
groups[0][1].append(s)
fg = "bg1" if TITLE_DARK else "tx1"
if TITLE_DARK:
    bg = parse_xml(f'<p:bg {NS}><p:bgPr><a:solidFill><a:schemeClr val="accent5"><a:lumMod val="50000"/>'
                   f'</a:schemeClr></a:solidFill><a:effectLst/></p:bgPr></p:bg>')
    s._element.cSld.insert(0, bg)
title_ph, sub_ph = s.placeholders[0], s.placeholders[1]
title_ph.left, title_ph.top, title_ph.width, title_ph.height = Emu(1524000), Emu(1470025), Emu(9144000), Emu(2387600)
set_text(title_ph, [P("No-Regret Learning and Bayesian Opponent Memory in Pokémon Random Battles",
                      size=TITLE, b=True, color=fg)])
sub_ph.left, sub_ph.top, sub_ph.width, sub_ph.height = Emu(1524000), Emu(3949700), Emu(9144000), Emu(760000)
set_text(sub_ph, [P("Can per-turn equilibrium play, built from no-regret learning, play Random Battles "
                    "— and where does it break?", size=BODY, i=True, color=fg)])
S = Shapes(s)
S.line(1524000, 4777581, 1524000 + 9144000, 4777581, "Title rule", colour=fg)
S.text(1524000, 4880000, 9144000, 460000, [P("[Names] · CS535", size=BODY, color=fg)], "Authors")
notes(s, "Hi, we are [Names]. PokaiTrainer showed that per-turn equilibrium play over sampled opponent "
      "worlds is competitive in Pokémon VGC. We ask whether the same idea, built entirely from no-regret learning, "
      "can play Gen 9 Random Battles against Foul Play, the strongest public bot. Short answer: the pieces compose "
      "cleanly, but the agent stays below Foul Play, and our theory points to where it breaks.")

# 2. The game and the yardsticks
s, S = content(1, "The game and the yardsticks")
S.text(LEFT, TOP, WIDTH, Inches(0.8), [P("Zero-sum stochastic game: simultaneous moves, chance, hidden teams "
                                         "→ a stage matrix each turn", size=BODY, align="l")], "Game summary")
rows = [
    ("Regret", "Reg_T = max_e Σ_t r_{t,e} − Σ_t p_t · r_t", "best expert in hindsight minus what we earned"),
    ("Exploitability", "expl(x) = val(A) − min_y x^TAy",
     "loss against a best-responding opponent, relative to the game value"),
    ("Duality gap", "gap(x, y) = max_i (Ay)_i − min_j (x^TA)_j", "the exploitability of both players together"),
]
lw, rh = Inches(2.6), Inches(1.05)
for k, (name, eq, gloss) in enumerate(rows):
    y = TOP + Inches(0.9) + k * (rh + Inches(0.3))
    S.header(LEFT, y, lw, rh, [P(name, b=True)], f"{name} label")
    S.line(LEFT + lw, y + rh // 2, LEFT + lw + GAP, y + rh // 2, f"{name} connector")
    S.box(LEFT + lw + GAP, y, WIDTH - lw - GAP, rh, [EQP(eq), P(gloss, size=SECONDARY)], f"{name} definition")
notes(s, "A battle is a zero-sum stochastic game with simultaneous moves, chance and hidden teams, so each turn "
      "we face a stage matrix. Three yardsticks. Regret: the best expert in hindsight minus what we earned. "
      "Exploitability: how much a best-responding opponent takes from us, relative to the game value. The duality gap "
      "is the exploitability of both players together; it is zero exactly at equilibrium.")

# 3. From lecture: MWU -> Hedge, and regret => equilibrium
s, S = content(1, "MWU → Hedge, and regret ⇒ equilibrium")
bh = Inches(1.25)
S.header(LEFT, TOP, COL2, HDR_H, [P("MWU (lecture)", b=True)], "MWU header")
S.header(LEFT + COL2 + GAP, TOP, COL2, HDR_H, [P("Hedge (ours)", b=True)], "Hedge header")
by = TOP + HDR_H + Inches(0.15)
S.box(LEFT, by, COL2, bh, [EQP("w ← w (1 − ηℓ)")], "MWU update")
S.box(LEFT + COL2 + GAP, by, COL2, bh, [EQP("w ← w · e^{ηr}"), EQP("p_t ∝ exp(η · cumulative reward)")],
      "Hedge update")
S.line(LEFT + COL2, by + bh // 2, LEFT + COL2 + GAP, by + bh // 2, "MWU to Hedge connector")
S.text(LEFT, by + bh + Inches(0.15), WIDTH, Inches(0.85), [
    P("Up- and down-weighting coincide after normalising", size=BODY, align="l", bullet=True),
    P("Bayes = Hedge with η = 1 on log loss", size=BODY, align="l", bullet=True, before=4),
], "Hedge bullets")
fy = by + bh + Inches(1.15)
S.header(LEFT, fy, WIDTH, HDR_H, [P("Fact 1: both players no-regret ⇒ time averages are an ε-equilibrium", b=True)],
         "Fact 1 header")
S.box(LEFT, fy + HDR_H + Inches(0.15), WIDTH, Inches(1.25), [
    EQP("ε = (Reg_1 + Reg_2) / T"),
    P("averaged joint play is an ε-CCE, which in zero-sum games is minimax", size=SECONDARY),
    P("CFR: regret bounds the exploitability of the average strategy", size=SECONDARY),
], "Fact 1 box")
notes(s, "Lecture gave us multiplicative weights, a linear update on losses. Our code uses Hedge, an exponential "
      "update on rewards, so the strategy is a softmax of cumulative reward; up- and down-weighting coincide after "
      "normalising. Bayesian updating is Hedge with step one on log loss. Fact 1: if both players have no regret, the "
      "time averages are an epsilon-equilibrium, and CFR uses regret to bound exploitability.")

# ===================== Section 2: Method =====================
# 4. Three levels
s, S = content(2, "Applying No-Regret Learning")
levels = [
    ("Level 1 · Inside a turn", "Bayesian LP / RM⁺ per turn", "→ CFR lineage", None),
    ("Level 2 · Across games", "Hedge over experts + style posterior", "→ experts problem", None),
    ("Level 3 · Search root", "Bandits UCB1 / Exp3", "→ visit-share identity", "gap = (Reg^1 + Reg^2) / N"),
]
bh = Inches(1.7)
by = TOP + HDR_H + Inches(0.15)
for k, (head, what, lin, eq) in enumerate(levels):
    x = LEFT + k * (COL3 + GAP)
    S.header(x, TOP, COL3, HDR_H, [P(head, b=True)], f"Level {k + 1} header")
    paras = [P(what), P(lin, size=SECONDARY)] + ([EQP(eq)] if eq else [])
    S.box(x, by, COL3, bh, paras, f"Level {k + 1} box")
    if k:
        S.line(x - GAP, by + bh // 2, x, by + bh // 2, f"Level {k} to {k + 1} connector")
gy = by + bh + Inches(0.35)
S.header(LEFT, gy, WIDTH, HDR_H, [P("Bayesian matrix game", b=True)], "Bayesian matrix game header")
S.box(LEFT, gy + HDR_H + Inches(0.15), WIDTH, Inches(1.2), [
    EQP("max_x  Σ_k  β_k  min_j  (M_k^T x)_j"),
    P("one LP per turn, vs PIMC averaging (solve each world as if known, then average → strategy fusion)",
      size=SECONDARY),
], "Bayesian LP equation")
notes(s, "We apply no-regret learning inside a turn, across games, and at the search root. Level one, inside a turn: the Bayesian LP, or RM-plus "
      "self-play. Its object is PokaiTrainer's Bayesian matrix game: one strategy x against an opponent that "
      "best-responds in each sampled world, unlike PIMC averaging, which suffers strategy fusion. Level two: Hedge over "
      "experts plus a style posterior. Level three: at a search root, the visit shares' duality gap equals summed regret over N.")

# 5. Level 2 is the experts problem
s, S = content(2, "Level 2 is the experts problem")
experts = [("Top move", "argmax of x"), ("Sampled", "play x"), ("Pure maximin", "the security row"),
           ("ε-safe best response", "to the predicted opponent q̂ (style posterior)")]
S.header(LEFT, TOP, WIDTH, HDR_H, [P("Experts: fixed rules applied to the turn's solved game", b=True)],
         "Experts header")
by, bh = TOP + HDR_H + Inches(0.15), Inches(1.25)
for k, (name, rule) in enumerate(experts):
    x = LEFT + k * (COL4 + GAP)
    S.box(x, by, COL4, bh, [P(name, b=True), P(rule, size=SECONDARY)], f"Expert {k + 1} box")
hy = by + bh + Inches(0.35)
S.header(LEFT, hy, WIDTH, HDR_H, [P("Hedge mixes the experts", b=True)], "Hedge header")
S.box(LEFT, hy + HDR_H + Inches(0.15), WIDTH, Inches(1.95), [
    EQP("p = Σ_e w_e q_e"),
    P("full-information feedback r", ("e", {"baseline": -25000}), " from the observed opponent action"),
    P("weights kept across the games of a match"),
    P("As in the lecture's experts problem, with each fixed action replaced by a fixed rule; "
      "Hedge competes with the best rule in hindsight", size=SECONDARY, before=4),
], "Hedge box")
notes(s, "Level two is the lecture's experts problem. Each expert is a fixed rule applied to the turn's solved "
      "game: top move, sampled, pure maximin, and an epsilon-safe best response to the predicted opponent. Hedge mixes "
      "them with full-information feedback: each rule earns what it would have earned against the "
      "opponent's observed action. Weights carry across the games of a match, so Hedge competes with the best rule in hindsight.")

# 6. Refinements: the Bayesian opponent model
s, S = content(2, "Refinements: the Bayesian opponent model")
refs = [
    ("Style posterior", [
        P("5 behavioural types: stage equilibrium, best response to uniform, logit best response to our "
          "mixture, cautious (maximin), uniform"),
        EQP("Bayes = Hedge with η = 1 on log loss  ⇒  regret ≤ ln 5")], Inches(1.35)),
    ("q̂ → ε-safe best response", [
        P("exploit the predicted opponent, keeping the stage value minus 0.3% of the payoff range")], Inches(0.8)),
    ("Per-world style likelihood", [
        P("score each style on each sampled world's own legal actions, not on the world-averaged matrix")],
     Inches(0.8)),
    ("Forgetting (λ < 1) / fixed share", [
        P("track an opponent that switches style → sets up the detection-delay result")], Inches(0.8)),
]
lw, y = Inches(3.3), TOP
for k, (name, paras, h) in enumerate(refs):
    S.header(LEFT, y, lw, h, [P(name, b=True)], f"Refinement {k + 1} label")
    S.line(LEFT + lw, y + h // 2, LEFT + lw + GAP, y + h // 2, f"Refinement {k + 1} connector")
    S.box(LEFT + lw + GAP, y, WIDTH - lw - GAP, h, paras, f"Refinement {k + 1} text")
    y += h + Inches(0.25)
notes(s, "The opponent model is Bayesian. A posterior over five behavioural types predicts the opponent; Bayes is "
      "Hedge with step one on log loss, so its regret is at most log five. That prediction feeds the epsilon-safe best "
      "response, which exploits while keeping the stage value minus 0.3 percent of the range. The per-world likelihood "
      "scores styles on each world's legal actions. Forgetting or fixed share tracks a switching opponent.")

# ===================== Section 3: Experiments =====================
# 7. Engine experiments
s, S = content(3, "In the engine")
stats = [
    ("0.760 / 0.650", "MCTS at matched time beats equilibrium play (vs sampled / top move)"),
    ("0.250 vs 0.090", "95% lower confidence bounds on exploitability: top move vs sampled"),
    ("0.585", "Bayesian LP beats PIMC averaging (sampled play, 400 games)"),
    ("6.9–155×", "Exact endgames: the bound holds, loose by this factor; consistent with the leaf evaluation "
                 "error dominating"),
]
nh, bh = Inches(0.55), Inches(1.35)
for k, (big, label) in enumerate(stats):
    x = LEFT + (k % 2) * (COL2 + GAP)
    y = TOP + (k // 2) * (nh + bh + Inches(0.35))
    S.header(x, y, COL2, nh, [P(big, size=EQ, b=True)], f"Stat {k + 1} number")
    S.box(x, y + nh, COL2, bh, [P(label)], f"Stat {k + 1} label")
notes(s, "In-engine results. At matched time, MCTS beats equilibrium play: 0.760 against sampled and 0.650 against "
      "top move. An informed best response gives 95 percent lower confidence bounds on exploitability of 0.250 for top "
      "move and 0.090 for sampled play, so randomising helps. The Bayesian LP beats PIMC averaging, 0.585 over 400 games. "
      "On exact endgames the bound holds but is loose by 6.9 to 155 times.")

# 8. Real Showdown
s, S = content(3, "Real Showdown and pre-registered tests")
fw = Inches(6.0)
fh = int(fw * 450 / 680)  # 680 x 450 px figure
pic = s.shapes.add_picture(FIG, LEFT, TOP, fw, fh)
pic.name = "Round-robin ratings figure"
pic._element.nvPicPr.cNvPr.set(
    "descr", "Round-robin Bradley-Terry ratings: Foul Play at all search times above Bayesian LP; "
             "top move and PIMC near 0")
S.text(LEFT, TOP + fh + Inches(0.05), fw, Inches(0.35),
       [P("Round-robin tournament, 1,800 games; Elo-scale ratings, 95% CIs", size=CAPTION, i=True, align="l")],
       "Figure caption")
rx = LEFT + fw + GAP
S.header(rx, TOP, RIGHT - rx, HDR_H, [P("Showdown results", b=True)], "Showdown header")
S.box(rx, TOP + HDR_H + Inches(0.15), RIGHT - rx, Inches(4.25), [
    P("Below Foul Play, even at its shortest search time", align="l", bullet=True),
    P("Level with the other one-turn solvers", align="l", bullet=True, before=6),
    P("Hedge regret 0.0044 per scored turn vs its best expert", align="l", bullet=True, before=6),
    P("Detection delay matches its prediction per match (ρ = 0.86); no prediction refuted", align="l",
      bullet=True, before=6),
    P("Better opponent prediction (≈0.3 nats per scored turn), no detectable win-rate change (replicated)",
      align="l", bullet=True, before=6),
], "Showdown bullets")
notes(s, "On real Showdown, a round-robin tournament of 1,800 games: Foul Play at every search time rates above "
      "our Bayesian LP, which is level with the other one-turn solvers. Hedge's regret against its best expert is 0.0044 "
      "per scored turn. The pre-registered detection-delay test refuted no prediction, rho 0.86. Better opponent "
      "prediction, about 0.3 nats per scored turn, gave no detectable win-rate change, and the null replicated.")

# ===================== Section 4: Theory =====================
# 9. The bound and the anchors
s, S = content(4, "The bound, and how it builds on the anchors")
S.box(LEFT, TOP, WIDTH, Inches(0.65), [EQP("expl(π)  ≤  H̄ · sup (ε_s + 2ε_v)")], "Bound equation")
terms = [("ε_s", "stage exploitability: shrinks with compute (regret ⇒ equilibrium; CFR)"),
         ("ε_v", "stage-matrix error from the heuristic evaluation: does not"),
         ("H̄", "expected length against a best response (tight)")]
sw, th = Inches(1.2), Inches(0.5)
for k, (sym, txt) in enumerate(terms):
    y = TOP + Inches(0.85) + k * (th + Inches(0.15))
    S.header(LEFT, y, sw, th, [EQP(sym)], f"Term {k + 1} symbol")
    S.line(LEFT + sw, y + th // 2, LEFT + sw + GAP, y + th // 2, f"Term {k + 1} connector")
    S.text(LEFT + sw + GAP, y, WIDTH - sw - GAP, th, [P(txt, size=BODY, align="l")], f"Term {k + 1} text",
           anchor="ctr")
anchors = [("Freund & Schapire", "→ Fact 1 / ε", ("s", -25000), ""),
           ("Zinkevich et al. (CFR)", "→ exploitability metric; regret ⇒ exploitability", None, ""),
           ("Cesa-Bianchi & Lugosi", "→ Hedge, Bayes = Hedge, tracking (detection delay)", None, ""),
           ("PokaiTrainer", "→ the Bayesian matrix game", None, "")]
ay, ah = TOP + Inches(2.9), Inches(1.3)
for k, (who, what, sub, _) in enumerate(anchors):
    x = LEFT + k * (COL4 + GAP)
    second = [what] + ([(sub[0], {"baseline": sub[1]})] if sub else [])
    S.box(x, ay, COL4, ah, [P(who, b=True), P(*second, size=SECONDARY)], f"Anchor {k + 1} box")
S.text(LEFT, ay + ah + Inches(0.1), WIDTH, Inches(0.75), [
    P("Plus: hidden-information version; core step Lean-checked; bandit-root identity", size=SECONDARY),
    EQP("gap = (Reg^1 + Reg^2) / N"),
], "Theory footer")
notes(s, "The headline theorem: exploitability is at most H-bar times the sup of epsilon-s plus two epsilon-v. "
      "Epsilon-s, the stage exploitability, shrinks with compute, by Freund and Schapire's regret-to-equilibrium fact and "
      "CFR. Epsilon-v, the stage-matrix error from the heuristic evaluation, does not. H-bar, the length against a best "
      "response, is tight. Hedge and tracking come from Cesa-Bianchi and Lugosi, the Bayesian matrix game from PokaiTrainer.")

# ===================== Section 5: Conclusions =====================
# 10. Takeaways and next steps
s, S = content(5, "Takeaways and next steps")
items = ["No-regret learning composes cleanly from the turn to the search root; the measured regrets are small",
         "Bound and experiments are consistent with the stage-matrix (leaf) error dominating; "
         "better beliefs don't move win rate",
         "Next: an evaluation function trained on exact terminal values; log the opponent's side to measure "
         "two-sided regret more fully"]
nw, ih = Inches(0.8), Inches(0.95)
for k, txt in enumerate(items):
    y = TOP + k * (ih + Inches(0.3))
    S.header(LEFT, y, nw, ih, [P(str(k + 1), b=True)], f"Takeaway {k + 1} number")
    S.line(LEFT + nw, y + ih // 2, LEFT + nw + GAP, y + ih // 2, f"Takeaway {k + 1} connector")
    S.box(LEFT + nw + GAP, y, WIDTH - nw - GAP, ih, [P(txt)], f"Takeaway {k + 1} text")
S.text(LEFT, TOP + 3 * (ih + Inches(0.3)), WIDTH, Inches(0.45),
       [P("Also in the paper: joint set×style belief and lookahead-posterior nulls; case study",
          size=SECONDARY, i=True, align="l")], "Also line")
notes(s, "Three takeaways. First, no-regret learning composes cleanly across a turn, a match and a search root, "
      "and the measured regrets are small. Second, the bound and the experiments are consistent with the stage-matrix "
      "error from the leaf evaluation dominating: better beliefs improve prediction, not win rate. Third, next steps: an "
      "evaluation function trained on exact terminal values, and logging the opponent's side to measure two-sided regret. Thank you.")

add_sections(prs, groups)
prs.core_properties.title = "No-Regret Learning and Bayesian Opponent Memory in Pokémon Random Battles"
prs.core_properties.author = "[Names]"
prs.save(OUT)
print("wrote", OUT)
