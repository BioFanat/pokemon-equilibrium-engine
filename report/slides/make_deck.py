"""Build deck.pptx: the 11-slide, ~4-minute CS535 final-project talk, in five sections.

Built on the user's template ("random template.pptx"): its master, theme, 16:9 size,
title-slide style and its five-box section tracker, which lives in the template's
"1_Title and Content" ... "5_Title and Content" layouts. Only the tracker labels change.
Content is plain text (Helvetica Neue, 20 pt, secondary 16 pt) with equations as native
PowerPoint equations (OMML, Cambria Math, 22 pt), wrapped like the template's slide 6:
mc:AlternateContent with an a14:m choice and a plain Unicode text fallback. Two slides keep
a small group of the template's slide-3 boxes (accent5 header, bg2 box, tx1 outline).
Every number is taken from ../final5.tex or the verified dynamics and theory results.

Run (any working directory): <python with python-pptx> make_deck.py
"""
import os
import re

from pptx import Presentation
from pptx.oxml import parse_xml
from pptx.util import Emu, Inches

HERE = os.path.dirname(os.path.abspath(__file__))
TEMPLATE = os.path.join(HERE, "random template.pptx")
OUT = os.path.join(HERE, "deck.pptx")
FIG_MAIN = os.path.join(HERE, "..", "figures", "fig_main.png")  # 1395 x 416 px, 4 panels
FIG_WINS = os.path.join(HERE, "..", "figures", "fig_wins.png")  # 655 x 525 px
GAME_PNG = os.path.join(HERE, "media", "game_intro.png")  # 493 x 290 px, battle view from the case-study game
GIF = os.path.join(HERE, "media", "showdown_combined_slide.gif")  # 800 x 542 px case-study animation

SECTIONS = ["Preliminaries", "Method", "Theory", "Experiments", "Conclusions"]
# Template layout that highlights each section's tracker box (box 1 ... box 5).
SECTION_LAYOUT = ["1_Title and Content", "2_Title and Content", "3_Title and Content",
                  "4_Title and Content", "5_Title and Content"]
TRACKER_LAYOUTS = ["Title and Content"] + SECTION_LAYOUT

# Title slide: the template's slide 1 has a dark accent5 background with white text.
TITLE_DARK = True

# Size scale (pt).
TITLE, BODY, SECONDARY, CAPTION, EQ = 36, 20, 16, 12, 22

NS = ('xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" '
      'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" '
      'xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main"')
# Namespaces of the template's slide-6 equation wrapper.
NS_MATH = ('xmlns:mc="http://schemas.openxmlformats.org/markup-compatibility/2006" '
           'xmlns:a14="http://schemas.microsoft.com/office/drawing/2010/main" '
           'xmlns:m="http://schemas.openxmlformats.org/officeDocument/2006/math"')
HN = ('<a:latin typeface="Helvetica Neue" panose="02000503000000020004" pitchFamily="2" charset="0"/>'
      '<a:ea typeface="Helvetica Neue" panose="02000503000000020004" pitchFamily="2" charset="0"/>'
      '<a:cs typeface="Helvetica Neue" panose="02000503000000020004" pitchFamily="2" charset="0"/>')
CM = '<a:latin typeface="Cambria Math" panose="02040503050406030204" pitchFamily="18" charset="0"/>'
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


# ---------- equations: a compact LaTeX-like spec -> OMML (+ a Unicode fallback) ----------
# Spec syntax: x_i, x^2, x_{t,e}^{1}; \sum_{lo}^{hi}{body}; \frac{a}{b}; \sqrt{a};
# \max_{x}{body}, \min_{y}{body} (name with the limit under it); \sup{body}, \ln{a}, \exp{a};
# \bar{H}, \hat{q}; {...} groups. Spaces only separate tokens (OMML spaces itself).
UPRIGHT = {"Regret", "MWU", "expl", "gap", "val", "E", "arg"}  # names set upright (E: expectation)
FUNCS = {"sup", "ln", "exp"}
_TOKEN = re.compile(r"\\[A-Za-z]+|[A-Za-z]+|\d+|\s+|.")


def _italic(text):
    """Math-italic code points, as PowerPoint stores them (template slide 6: 𝑃, 𝑋, 𝜆)."""
    out = []
    for ch in text:
        o = ord(ch)
        if ch == "h":
            out.append("\u210e")
        elif "a" <= ch <= "z":
            out.append(chr(0x1D44E + o - ord("a")))
        elif "A" <= ch <= "Z":
            out.append(chr(0x1D434 + o - ord("A")))
        elif ch == "\u03f5":  # lunate epsilon (the paper's \epsilon)
            out.append("\U0001D716")
        elif 0x3B1 <= o <= 0x3C9:
            out.append(chr(0x1D6FC + o - 0x3B1))
        else:
            out.append(ch)
    return "".join(out)


class _Parser:
    def __init__(self, src):
        self.t, self.i = _TOKEN.findall(src), 0

    def peek(self):
        return self.t[self.i] if self.i < len(self.t) else None

    def take(self):
        self.i += 1
        return self.t[self.i - 1]

    def seq(self):
        out = []
        while self.peek() not in (None, "}"):
            if self.peek().isspace():
                out.append(("sp", self.take()))
            else:
                out.append(self.item())
        return out

    def group(self):
        while self.peek() is not None and self.peek().isspace():
            self.take()
        return self.atom()

    def scripts(self):
        sub = sup = None
        while self.peek() in ("_", "^"):
            if self.take() == "_":
                sub = self.atom()
            else:
                sup = self.atom()
        return sub, sup

    def item(self):
        base = self.atom()
        sub, sup = self.scripts()
        return ("ss", base, sub, sup) if (sub or sup) else base

    def atom(self):
        tok = self.take()
        if tok == "{":
            body = self.seq()
            self.take()  # "}"
            return ("grp", body)
        if tok == "\\sum":
            sub, sup = self.scripts()
            return ("nary", "∑", sub, sup, self.group())
        if tok == "\\frac":
            return ("frac", self.group(), self.group())
        if tok == "\\sqrt":
            return ("rad", self.group())
        if tok in ("\\max", "\\min") or (tok == "\\sup" and self.peek() == "_"):
            sub, _ = self.scripts()
            return ("lim", tok[1:], sub, self.group())
        if tok[1:] in FUNCS:
            return ("func", tok[1:], self.group())
        if tok == "\\bar":
            return ("bar", self.group())
        if tok == "\\hat":
            return ("acc", "\u0302", self.group())
        if tok.isalpha():  # Latin words and Greek letters
            return ("w", tok)
        if tok.isdigit():
            return ("n", tok)
        return ("o", tok)


def _rpr(sz, ital):
    return (f'<a:rPr lang="en-US" sz="{sz * 100}" b="0" i="{1 if ital else 0}" dirty="0">'
            f'<a:solidFill><a:schemeClr val="tx1"/></a:solidFill>{CM}</a:rPr>')


def _ctrl(sz):
    return f"<m:ctrlPr>{_rpr(sz, True)}</m:ctrlPr>"


def _mr(text, sz, kind):
    """One math run: kind 'i' italic letters, 'p' upright name, 'o' operator or digit."""
    if kind == "i":
        return f"<m:r>{_rpr(sz, True)}<m:t>{esc(_italic(text))}</m:t></m:r>"
    sty = '<m:rPr><m:sty m:val="p"/></m:rPr>' if kind == "p" else ""
    return f"<m:r>{sty}{_rpr(sz, False)}<m:t>{esc(text)}</m:t></m:r>"


def _omml(node, sz):
    if node is None:
        return ""
    k = node[0]
    if k == "grp":
        return "".join(_omml(n, sz) for n in node[1])
    if k == "sp":
        return ""
    if k == "w":
        return _mr(node[1], sz, "p" if node[1] in UPRIGHT else "i")
    if k in ("n", "o"):
        return _mr(node[1], sz, "o")
    if k == "ss":
        _, base, sub, sup = node
        e = f"<m:e>{_omml(base, sz)}</m:e>"
        if sub and sup:
            return (f"<m:sSubSup><m:sSubSupPr>{_ctrl(sz)}</m:sSubSupPr>{e}<m:sub>{_omml(sub, sz)}</m:sub>"
                    f"<m:sup>{_omml(sup, sz)}</m:sup></m:sSubSup>")
        if sub:
            return f"<m:sSub><m:sSubPr>{_ctrl(sz)}</m:sSubPr>{e}<m:sub>{_omml(sub, sz)}</m:sub></m:sSub>"
        return f"<m:sSup><m:sSupPr>{_ctrl(sz)}</m:sSupPr>{e}<m:sup>{_omml(sup, sz)}</m:sup></m:sSup>"
    if k == "nary":
        _, chr_, sub, sup, body = node
        hide = ('<m:subHide m:val="1"/>' if sub is None else "") + ('<m:supHide m:val="1"/>' if sup is None else "")
        return (f'<m:nary><m:naryPr><m:chr m:val="{chr_}"/><m:limLoc m:val="undOvr"/>{hide}{_ctrl(sz)}</m:naryPr>'
                f"<m:sub>{_omml(sub, sz)}</m:sub><m:sup>{_omml(sup, sz)}</m:sup><m:e>{_omml(body, sz)}</m:e></m:nary>")
    if k == "frac":
        return (f"<m:f><m:fPr>{_ctrl(sz)}</m:fPr><m:num>{_omml(node[1], sz)}</m:num>"
                f"<m:den>{_omml(node[2], sz)}</m:den></m:f>")
    if k == "rad":
        return (f'<m:rad><m:radPr><m:degHide m:val="1"/>{_ctrl(sz)}</m:radPr><m:deg/>'
                f"<m:e>{_omml(node[1], sz)}</m:e></m:rad>")
    if k == "lim":
        _, name, sub, body = node
        return (f"<m:func><m:funcPr>{_ctrl(sz)}</m:funcPr><m:fName><m:limLow><m:limLowPr>{_ctrl(sz)}</m:limLowPr>"
                f"<m:e>{_mr(name, sz, 'p')}</m:e><m:lim>{_omml(sub, sz)}</m:lim></m:limLow></m:fName>"
                f"<m:e>{_omml(body, sz)}</m:e></m:func>")
    if k == "func":
        return (f"<m:func><m:funcPr>{_ctrl(sz)}</m:funcPr><m:fName>{_mr(node[1], sz, 'p')}</m:fName>"
                f"<m:e>{_omml(node[2], sz)}</m:e></m:func>")
    if k == "bar":
        return (f'<m:bar><m:barPr><m:pos m:val="top"/>{_ctrl(sz)}</m:barPr>'
                f"<m:e>{_omml(node[1], sz)}</m:e></m:bar>")
    if k == "acc":
        return (f'<m:acc><m:accPr><m:chr m:val="{node[1]}"/>{_ctrl(sz)}</m:accPr>'
                f"<m:e>{_omml(node[2], sz)}</m:e></m:acc>")
    raise ValueError(node)


SUB, SUP = -25000, 30000


def _flat(node, bl=0):
    """Unicode fallback: (text, baseline) pieces; sub -25 %, super +30 %."""
    if node is None:
        return []
    k = node[0]
    if k == "grp":
        return [pc for n in node[1] for pc in _flat(n, bl)]
    if k in ("sp", "w", "n", "o"):
        return [(node[1], bl)]
    if k == "ss":
        return _flat(node[1], bl) + _flat(node[3], bl or SUP) + _flat(node[2], bl or SUB)
    if k == "nary":
        return [("Σ", bl)] + _flat(node[2], bl or SUB) + _flat(node[3], bl or SUP) + [(" ", bl)] + _flat(node[4], bl)
    if k == "frac":
        def wrap(n):
            pcs = _flat(n, bl)
            simple = len("".join(t for t, _ in pcs).strip()) <= 1
            return pcs if simple else [("(", bl)] + pcs + [(")", bl)]
        return wrap(node[1]) + [("/", bl)] + wrap(node[2])
    if k == "rad":
        return [("√(", bl)] + _flat(node[1], bl) + [(")", bl)]
    if k == "lim":
        return [(node[1], bl)] + _flat(node[2], bl or SUB) + [(" ", bl)] + _flat(node[3], bl)
    if k == "func":
        return [(node[1] + " ", bl)] + _flat(node[2], bl)
    if k in ("bar", "acc"):
        mark = "\u0304" if k == "bar" else node[1]
        pcs = _flat(node[-1], bl)
        return pcs[:-1] + [(pcs[-1][0] + mark, pcs[-1][1])]
    raise ValueError(node)


def omath(src, sz):
    """OMML <m:oMath> for a spec string."""
    return "<m:oMath>" + "".join(_omml(n, sz) for n in _Parser(src).seq()) + "</m:oMath>"


def fallback_runs(src, sz):
    """Plain runs (Cambria Math, baseline-shifted scripts) for apps that cannot render OMML."""
    merged = []
    for t, bl in (pc for n in _Parser(src).seq() for pc in _flat(n)):
        if merged and merged[-1][1] == bl:
            merged[-1][0] += t
        else:
            merged.append([t, bl])
    out = []
    for t, bl in merged:
        attrs = f'lang="en-US" sz="{sz * 100}" dirty="0"' + (f' baseline="{bl}"' if bl else "")
        sp = ' xml:space="preserve"' if t != t.strip() else ""
        out.append(f'<a:r><a:rPr {attrs}><a:solidFill><a:schemeClr val="tx1"/></a:solidFill>{CM}</a:rPr>'
                   f"<a:t{sp}>{esc(t)}</a:t></a:r>")
    return "".join(out)


# ---------- text: paragraphs and runs ----------
class Para:
    """A paragraph as rendered by OMML-aware apps (choice) and by others (fallback)."""

    def __init__(self, choice, fallback=None):
        self.choice, self.fallback = choice, fallback or choice
        self.math = fallback is not None


def run(text, size=BODY, b=False, i=False, color="tx1"):
    attrs = f'lang="en-US" sz="{size * 100}"' + (' b="1"' if b else "") + (' i="1"' if i else "")
    fill = f'<a:solidFill><a:schemeClr val="{color}"/></a:solidFill>'
    sp = ' xml:space="preserve"' if text != text.strip() else ""
    return f'<a:r><a:rPr {attrs} dirty="0">{fill}{HN}</a:rPr><a:t{sp}>{esc(text)}</a:t></a:r>'


def _ppr(align, bullet, before, indent=False, number=False):
    ppr = f'algn="{align}"' + (' marL="342900"' if indent else "")
    inner = f'<a:spcBef><a:spcPts val="{before * 100}"/></a:spcBef>' if before else ""
    if number:
        ppr += ' marL="457200" indent="-457200"'
        inner += '<a:buFont typeface="+mj-lt"/><a:buAutoNum type="arabicPeriod"/>'
    elif bullet:
        ppr += ' marL="342900" indent="-342900"'
        inner += ('<a:buFont typeface="Arial" panose="020B0604020202020204" pitchFamily="34" charset="0"/>'
                  '<a:buChar char="&#8226;"/>')
    return f"<a:pPr {ppr}>{inner}</a:pPr>"


def P(*pieces, size=BODY, b=False, i=False, color="tx1", align="l", bullet=False, before=0, indent=False,
      number=False):
    """One paragraph. Each piece is a str (plain), ('$', spec) (inline equation) or (text, {overrides}).
    indent=True aligns a sub-line with the bullet text above it; number=True numbers the paragraph."""
    choice, fall, math = [], [], False
    for pc in pieces:
        if isinstance(pc, tuple) and pc[0] == "$":
            math = True
            choice.append(f"<a14:m>{omath(pc[1], size)}</a14:m>")
            fall.append(fallback_runs(pc[1], size))
        else:
            r = run(pc[0], **{"size": size, "b": b, "i": i, "color": color, **pc[1]}) if isinstance(pc, tuple) \
                else run(pc, size, b, i, color)
            choice.append(r)
            fall.append(r)
    head = _ppr(align, bullet, before, indent, number)
    end = f'<a:endParaRPr lang="en-US" sz="{size * 100}" dirty="0"/>'
    c = f"<a:p>{head}{''.join(choice)}{end}</a:p>"
    return Para(c, f"<a:p>{head}{''.join(fall)}{end}</a:p>") if math else Para(c)


def EQ_(src, before=0, size=EQ):
    """A displayed equation, centred on its own line (template slide-6 structure)."""
    head = _ppr("ctr", False, before)
    end = f'<a:endParaRPr lang="en-US" sz="{size * 100}" dirty="0"/>'
    c = (f'<a:p>{head}<a14:m><m:oMathPara><m:oMathParaPr><m:jc m:val="centerGroup"/></m:oMathParaPr>'
         f"{omath(src, size)}</m:oMathPara></a14:m>{end}</a:p>")
    return Para(c, f"<a:p>{head}{fallback_runs(src, size)}{end}</a:p>")


# ---------- the equations, one line each ----------
E_REG = r"Regret_T = \sum_t{p^t·l^t} − \min_i{\sum_t{l_i^t}}"
E_EXPL = r"expl(x) = \max_{x′}{\min_y{x′^⊤Ay}} − \min_y{x^⊤Ay}"
E_MWU = r"w_i^{t+1} = w_i^t(1 − ηl_i^t)"
E_HEDGE = r"w_i^{t+1} = w_i^t e^{−ηl_i^t}"
E_SOFTMAX = r"p_i^t ∝ \exp{(η\sum_{j<t}{r_i^j})}"
E_BLP = r"\max_x{\sum_k{β_k \min_j{{(M_k^⊤x)}_j}}}"
E_MIX = r"p_t = \sum_e{w_e q_e}"
E_MINIMAX = r"\max_x{x^⊤A\bar{y}} − \min_y{\bar{x}^⊤Ay} ≤ ε_R + ε_C"
E_BOUND = r"expl(π) ≤ \bar{H} \bar{δ},   δ = ε_s + 2ε_v"
E_LOSSES = r"expl(π) = \sup_σ{E[\sum_{t<τ}{ℓ_t}]}"
E_MWUB = r"L_{MWU} ≤ (1 + η)L^* + \frac{\ln{n}}{η}"


# ---------- shapes ----------
class Shapes:
    """Appends shapes to one slide's spTree with unique ids; shapes holding math get the
    template's mc:AlternateContent wrapper (a14 choice, plain-text fallback)."""

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

    def _sp(self, make, paras):
        """make(sid, paragraph_xml) -> <p:sp>; wrapped in AlternateContent when any paragraph has math."""
        sid = self._id()
        if not any(p.math for p in paras):
            return self._add(make(sid, "".join(p.choice for p in paras)))
        return self._add(
            f'<mc:AlternateContent {NS_MATH} {NS}><mc:Choice Requires="a14">'
            f'{make(sid, "".join(p.choice for p in paras))}</mc:Choice>'
            f'<mc:Fallback>{make(sid, "".join(p.fallback for p in paras))}</mc:Fallback></mc:AlternateContent>')

    def rect(self, x, y, w, h, paras, fill, name, anchor="ctr"):
        return self._sp(lambda sid, body: (
            f'<p:sp {NS}><p:nvSpPr><p:cNvPr id="{sid}" name="{esc(name)}"/><p:cNvSpPr/><p:nvPr/></p:nvSpPr>'
            f'<p:spPr><a:xfrm><a:off x="{int(x)}" y="{int(y)}"/><a:ext cx="{int(w)}" cy="{int(h)}"/></a:xfrm>'
            f'<a:prstGeom prst="rect"><a:avLst/></a:prstGeom>{fill}'
            f'<a:ln><a:solidFill><a:schemeClr val="tx1"/></a:solidFill></a:ln></p:spPr>{RECT_STYLE}'
            f'<p:txBody><a:bodyPr rtlCol="0" anchor="{anchor}"/><a:lstStyle/>{body}</p:txBody></p:sp>'), paras)

    def header(self, x, y, w, h, paras, name):
        """Template slide-3 header box: light accent5 fill, tx1 outline, bold tx1 text."""
        return self.rect(x, y, w, h, paras, FILL_HEADER, name)

    def box(self, x, y, w, h, paras, name, anchor="ctr"):
        """Template slide-3 content box: bg2 fill, tx1 outline, tx1 text."""
        return self.rect(x, y, w, h, paras, FILL_BOX, name, anchor)

    def text(self, x, y, w, h, paras, name, anchor="t"):
        return self._sp(lambda sid, body: (
            f'<p:sp {NS}><p:nvSpPr><p:cNvPr id="{sid}" name="{esc(name)}"/><p:cNvSpPr txBox="1"/><p:nvPr/></p:nvSpPr>'
            f'<p:spPr><a:xfrm><a:off x="{int(x)}" y="{int(y)}"/><a:ext cx="{int(w)}" cy="{int(h)}"/></a:xfrm>'
            f'<a:prstGeom prst="rect"><a:avLst/></a:prstGeom><a:noFill/></p:spPr>'
            f'<p:txBody><a:bodyPr wrap="square" rtlCol="0" anchor="{anchor}"><a:noAutofit/></a:bodyPr><a:lstStyle/>'
            f'{body}</p:txBody></p:sp>'), paras)

    def line(self, x1, y1, x2, y2, name, colour="tx1", w=25400, arrow=False):
        """Template slide-3 straight connector: 2 pt tx1 line (w in EMU)."""
        sid = self._id()
        flip = ' flipV="1"' if y2 < y1 else ""
        x, y = min(x1, x2), min(y1, y2)
        return self._add(
            f'<p:cxnSp {NS}><p:nvCxnSpPr><p:cNvPr id="{sid}" name="{esc(name)}"/><p:cNvCxnSpPr/><p:nvPr/></p:nvCxnSpPr>'
            f'<p:spPr><a:xfrm{flip}><a:off x="{int(x)}" y="{int(y)}"/><a:ext cx="{int(abs(x2 - x1))}" cy="{int(abs(y2 - y1))}"/>'
            f'</a:xfrm><a:prstGeom prst="line"><a:avLst/></a:prstGeom><a:ln w="{w}"><a:solidFill>'
            f'<a:schemeClr val="{colour}"/></a:solidFill>{"<a:tailEnd type=\"triangle\"/>" if arrow else ""}</a:ln></p:spPr>{LINE_STYLE}</p:cxnSp>')

    def slide_number(self, n):
        """Slide-number placeholder; position, size and font come from the layout."""
        sid = self._id()
        self._add(
            f'<p:sp {NS}><p:nvSpPr><p:cNvPr id="{sid}" name="Slide Number Placeholder"/><p:cNvSpPr>'
            f'<a:spLocks noGrp="1"/></p:cNvSpPr><p:nvPr><p:ph type="sldNum" sz="quarter" idx="4"/></p:nvPr></p:nvSpPr>'
            f'<p:spPr/><p:txBody><a:bodyPr wrap="none"/><a:lstStyle/><a:p><a:fld id="{{7E75810F-9510-A844-820C-E3462544137D}}" '
            f'type="slidenum"><a:rPr lang="en-US"/><a:t>{n}</a:t></a:fld><a:endParaRPr lang="en-US" dirty="0"/></a:p>'
            f'</p:txBody></p:sp>')


def set_text(shape, paras):
    """Replace a placeholder's paragraphs with our own (plain text only)."""
    txBody = shape.text_frame._txBody
    for p in txBody.findall("{http://schemas.openxmlformats.org/drawingml/2006/main}p"):
        txBody.remove(p)
    for p in paras:
        txBody.append(parse_xml(p.choice.replace("<a:p>", f"<a:p {NS}>", 1)))


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


def crop_png(path, box):
    """Crop a PNG (left, top, right, bottom in px) into an in-memory stream for add_picture."""
    import io
    from PIL import Image
    buf = io.BytesIO()
    Image.open(path).crop(box).save(buf, format="PNG")
    buf.seek(0)
    return buf, box[2] - box[0], box[3] - box[1]


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
    set_text(s.shapes.title, [P(title, size=TITLE)])
    S = Shapes(s)
    S.slide_number(len(prs.slides))
    groups[sec - 1][1].append(s)
    return s, S


def notes(slide, text):
    slide.notes_slide.notes_text_frame.text = text


TOP = Inches(1.7)  # first content row, below the template's title rule (1.48 in)
BOTTOM = Inches(7.0)  # above the tracker (7.15 in)
HDR_H = Inches(0.42)  # template slide-3 header box (0.38 in), 0.04 in taller so the text fits
COL2 = (WIDTH - GAP) // 2
COL3 = (WIDTH - 2 * GAP) // 3
COL4 = (WIDTH - 3 * GAP) // 4
SP = 10  # space before a new item (pt)

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
                      size=TITLE, b=True, color=fg, align="ctr")])
sub_ph.left, sub_ph.top, sub_ph.width, sub_ph.height = Emu(1524000), Emu(3949700), Emu(9144000), Emu(760000)
set_text(sub_ph, [P("Can per-turn equilibrium play, built from no-regret learning, play Random Battles "
                    "— and where does it break?", i=True, color=fg, align="ctr")])
S = Shapes(s)
S.line(1524000, 4777581, 1524000 + 9144000, 4777581, "Title rule", colour=fg)
S.text(1524000, 4880000, 9144000, 460000, [P("Nikhil Pesaladinne, Bo Chi · CS535", color=fg, align="ctr")], "Authors")
notes(s, "Hi, we are Nikhil and Bo. PokaiTrainer showed that per-turn equilibrium play over sampled opponent "
      "worlds is competitive in Pokémon VGC. We ask whether the same idea, built around no-regret learning, "
      "can play Gen 9 Random Battles against Foul Play, the strongest public bot. Short answer: the pieces compose "
      "cleanly, but the agent stays below Foul Play, and our theory points to where it breaks.")

# 1b. The game itself
s, S = content(1, "Pokémon Showdown: the game")
iw = Inches(6.2)
ih = int(iw * 290 / 493)
pic = s.shapes.add_picture(GAME_PNG, LEFT, TOP, iw, ih)
pic.name = "Battle view"
tx = LEFT + iw + Inches(0.3)
S.text(tx, TOP, RIGHT - tx, BOTTOM - TOP, [
    P("Each side has ", ("6 random Pokémon", {"b": True}), "; the opponent's team stays hidden until revealed",
      bullet=True),
    P("Each turn both players choose ", ("at the same time", {"b": True}), ": one of 4 moves or a switch "
      "(up to 13 options with Terastallization)", bullet=True, before=SP),
    P("Our Hippowdon here: Earthquake, Stone Edge, Slack Off or Curse, or switch to one of 5 teammates",
      bullet=True, before=SP),
    P("Damage is random; a Pokémon faints at 0 HP, and you lose when all six faint", bullet=True, before=SP),
], "Game bullets")
notes(s, "A quick picture of the game. Each player has six randomly generated Pokemon, and you only see the "
      "opponent's as they are revealed. Every turn both players choose at the same time: one of the active "
      "Pokemon's four moves, or a switch, plus a once-per-battle type change called Terastallization. In this real "
      "game our Hippowdon could use Earthquake, Stone Edge, Slack Off or Curse, or switch out; the opponent, Foul "
      "Play, has Sunflora in. Damage is random, a Pokemon faints at zero hit points, and you lose when all six have "
      "fainted. So it is a two-player zero-sum game with simultaneous moves, chance and hidden information.")

# 2. The objective: hard to exploit, not just strong
s, S = content(1, "Goal: hard to exploit, not just strong")
S.text(LEFT, TOP, WIDTH, Inches(2.45), [
    P("Strong bots are built to win on average: Foul Play searches a tree, RL agents learn a policy from "
      "self-play. Neither is built to be safe.", size=18, bullet=True),
    P("A fixed way of playing has habits, and an opponent who knows them can beat it, however strong it looks "
      "against everyone else.", size=18, bullet=True, before=SP),
    P(("Our goal: ", {"b": True}), "an agent built from no-regret learning that is ", ("hard to exploit", {"b": True}),
      ", with a bound on how much a best-responding opponent can take from it, and to measure what that costs in "
      "strength.", size=18, bullet=True, before=SP),
], "Objective")
by = TOP + Inches(2.85)
BHT = Inches(1.55)
yard = [("Exploitability: what we minimise", [
            EQ_(E_EXPL, size=20),
            P("best possible guarantee − our guarantee against a best response; 0 at equilibrium",
              size=SECONDARY, align="ctr", before=8)]),
        ("Regret: how our learners are judged", [
            EQ_(E_REG, size=20),
            P("our loss − the best fixed choice's, in hindsight; no-regret: ", ("$", "Regret_T/T → 0"),
              size=SECONDARY, align="ctr", before=8)])]
for k, (name, paras) in enumerate(yard):
    x = LEFT + k * (COL2 + GAP)
    S.header(x, by, COL2, HDR_H, [P(name, size=SECONDARY, b=True, align="ctr")], f"Yardstick {k + 1} header")
    S.box(x, by + HDR_H, COL2, BHT, paras, f"Yardstick {k + 1} box")
notes(s, "Why not just build another strong bot? The strongest bots are built to win on average: Foul Play runs "
      "tree search over sampled teams, and RL agents learn a policy that maximises win rate. Neither is built to be safe. "
      "Any fixed way of playing has habits, and an opponent who knows them can beat it, no matter how well it does "
      "against everyone else. So our goal is different: an agent built from no-regret learning that is hard to "
      "exploit, with a bound on how much a best-responding opponent can take from it, and then to measure what that "
      "safety costs in strength. Two yardsticks. Exploitability is what we try to make small: the best guarantee "
      "any strategy can secure, minus what ours secures against a best response; it is zero at an equilibrium. "
      "Regret is how our learners are judged: our total loss minus the best fixed choice's in hindsight, and "
      "no-regret means the average goes to zero.")

# ===================== Section 2: Method =====================
# 4. Preliminary model: worlds and matrices, one strategy x, then experts
s, S = content(2, "Preliminary model: an experts problem")


def world_table(x0, y0, title, cols, rows, vals, name):
    """A small payoff table: our moves (rows) x the opponent's options (columns) in one sampled world."""
    cw, ch, lw = Inches(1.12), Inches(0.3), Inches(1.05)
    S.text(x0, y0, lw + 3 * cw, Inches(0.3), [P(title, size=CAPTION, b=True, align="ctr")], f"{name} title")
    y = y0 + Inches(0.3)
    for j, c in enumerate(cols):
        S.header(x0 + lw + j * cw, y, cw, ch, [P(c, size=10, b=True, align="ctr")], f"{name} col {j}")
    for i, (r, row) in enumerate(zip(rows, vals)):
        yy = y + ch * (i + 1)
        S.header(x0, yy, lw, ch, [P(r, size=10, b=True, align="ctr")], f"{name} row {i}")
        for j, v in enumerate(row):
            S.box(x0 + lw + j * cw, yy, cw, ch, [P(v, size=10, align="ctr")], f"{name} cell {i}{j}")
    return y + ch * (len(rows) + 1)


tw = Inches(1.05) + 3 * Inches(1.12)
tx = RIGHT - tw
S.text(LEFT, TOP, tx - LEFT - Inches(0.3), Inches(1.6), [
    P(("1. Worlds and matrices. ", {"b": True}), "A world is one full guess of the opponent's hidden team and sets; "
      "we sample 4 per turn. Each world gives a payoff matrix: our moves × their options, each cell a heuristic "
      "score after a 2-turn lookahead.", size=16),
], "Step 1")
end = world_table(tx, TOP - Inches(0.05), "One world: Sunflora has Earth Power", ["Sludge Bomb", "Earth Power", "Switch"],
                  ["Earthquake", "Stone Edge", "Slack Off"],
                  [["62%", "58%", "55%"], ["60%", "57%", "52%"], ["66%", "61%", "54%"]], "World")
S.text(tx, end, tw, Inches(0.28), [P("illustrative values", size=10, i=True, align="ctr")], "Table caption")
y2 = end + Inches(0.2)
S.text(LEFT, y2, WIDTH, Inches(0.75), [
    P(("2. One strategy x. ", {"b": True}), "We cannot tell which world we are in, so we pick one mix ",
      ("$", "x"), " that does best on average when the opponent best-responds in each world (one LP).", size=16),
], "Step 2")
y3 = y2 + Inches(0.75)
S.text(LEFT, y3, WIDTH, Inches(0.45), [
    P(("3. Experts. ", {"b": True}), "Each proposes a way to play ", ("$", "x"), "; Hedge mixes them across games.", size=16),
], "Step 3")
experts = [("Top move", "the most likely action of x"),
           ("Sampled", "draw an action from x"),
           ("Pure maximin", "the single action with the best worst case"),
           ("ε-safe best response", "best reply to a prediction of the opponent, giving up at most ε of x's worst case")]
gy = y3 + Inches(0.5)
EBH = Inches(0.9)
for k, (name, rule) in enumerate(experts):
    x = LEFT + k * (COL4 + GAP)
    S.header(x, gy, COL4, HDR_H, [P(name, size=SECONDARY, b=True, align="ctr")], f"Expert {k + 1} header")
    S.box(x, gy + HDR_H, COL4, EBH, [P(rule, size=14, align="ctr")], f"Expert {k + 1} box")
by = gy + HDR_H + EBH + Inches(0.1)
S.text(LEFT, by, WIDTH, BOTTOM - by, [
    P("Each expert is credited with what it would have earned against the opponent's actual move; weights carry "
      "across games, so Hedge's regret bound covers the whole match", size=14, bullet=True),
], "Experts notes")
notes(s, "Our preliminary model, step by step. First, worlds and matrices: a world is one full guess of the "
      "opponent's hidden team and sets, and we sample four each turn. Each world gives a payoff matrix, like this "
      "one where Sunflora has Earth Power: our moves down the side, the opponent's options across the top, and our "
      "a heuristic score after a two-turn lookahead in each cell; these values are illustrative. Second, because "
      "we cannot tell which world we are in, we pick one mixed strategy x that does best on average when the opponent "
      "best-responds in each world, which is one linear program. Third, experts: each proposes a way to play x, its "
      "top move, a sample from it, the single safest action, or an epsilon-safe best response to a prediction of "
      "the opponent's move, and Hedge mixes them. After the opponent moves, each expert is credited with what it "
      "would have earned, and because the weights carry across games, Hedge's regret bound covers the whole match.")

# 6. Case study: one turn of a real Showdown game
s, S = content(2, "A turn, step by step")
gh = BOTTOM - TOP - Inches(0.5)  # room for a two-line caption
gw = int(gh * 800 / 542)  # 800 x 542 px GIF
gx = LEFT + (WIDTH - gw) // 2
pic = s.shapes.add_picture(GIF, gx, TOP, gw, gh)
pic.name = "Case-study animation"
pic._element.nvPicPr.cNvPr.set(
    "descr", "Showdown replay against Foul Play above our agent's decision panel: a slice of one sampled "
             "world's matrix, our mixed strategy as bars, estimated win chance in each cell")
S.text(gx, TOP + gh + Inches(0.03), gw, Inches(0.47),
       [P("A real Showdown game: each turn the agent picks one mixed strategy against 4 sampled opponent teams; "
          "bars show our mixed strategy, cells one world's heuristic score, shown as a win chance.", size=CAPTION, i=True)],
       "Animation caption")
notes(s, "This is a real Showdown game against Foul Play; the names at the top are the bots' account names. Each "
      "turn the agent samples 4 possible opponent teams and picks one mixed strategy that does best against all of them. The panel shows "
      "a slice of one sampled world's matrix: rows are our options, columns the opponent's, cells its heuristic score shown as a win chance. The bars are our mixed "
      "strategy, and the outlined cell is what we and Foul Play actually played.")

# ===================== Section 3: Theory =====================
# 8. The whole-battle bound
s, S = content(3, "Our two bounds: exploitability and regret")
S.text(LEFT, TOP, WIDTH, Inches(0.5), [
    P("Two guarantees: how much a perfect opponent could take from us in one battle, and how close Hedge gets to "
      "its best expert across a match.", size=16),
], "Bounds intro")
cw2 = (WIDTH - GAP) // 2
by = TOP + Inches(0.75)
bh = BOTTOM - by - HDR_H - Inches(0.15)
cols = [
    ("Exploitability bound (one battle)", [
        P("What an opponent who knows our play can win", size=19, align="ctr"),
        EQ_(r"expl ≤ \bar{H} · \bar{δ}", before=16),
        P("Each turn we can fall short by ", ("$", "δ = ε_s + 2ε_v"), size=18, align="ctr", before=16),
        P(("$", "ε_s"), " solver error: shrinks with more compute", size=18, align="ctr", before=10),
        P(("$", "ε_v"), " matrix error: wrong payoffs, not fixed by compute", size=18, align="ctr", before=8),
        P("Over a battle of ", ("$", r"\bar{H}"), " turns, these shortfalls add up", size=18, align="ctr", before=16),
    ]),
    ("Hedge regret bound (across a match)", [
        P("How much better the best single expert did", size=19, align="ctr"),
        EQ_(r"Regret(S) ≤ \frac{3}{2}\sqrt{\frac{S}{2}\ln{n}} + \sqrt{\frac{\ln{n}}{8}}", before=16),
        P("S turns played, n experts", size=18, align="ctr", before=16),
        P("Regret grows like ", ("$", r"\sqrt{S}"), ", so per turn it goes to 0", size=18, align="ctr",
          before=10),
        P("Over a long match, Hedge matches its best expert", size=18, align="ctr", before=16),
    ]),
]
for c, (title, paras) in enumerate(cols):
    x = LEFT + c * (cw2 + GAP)
    S.header(x, by, cw2, HDR_H, [P(title, size=SECONDARY, b=True, align="ctr")], f"Bound {c + 1} header")
    S.box(x, by + HDR_H, cw2, bh, paras, f"Bound {c + 1} box")
notes(s, "We have two guarantees. On the left, exploitability: how much an opponent who knows our strategy could take "
      "from us in one battle. Each turn we play the best mix for our estimated matrix, and we can fall short in two "
      "ways: the solver did not fully finish, which more compute fixes, or the matrix numbers are wrong, which it "
      "does not. Add those per-turn shortfalls over the battle and you get the bound: battle length times the average "
      "shortfall. On the right, Hedge across a match: its regret, how much better its best single expert would have "
      "done, grows only like the square root of the number of turns, so per turn it goes to zero, and over a long "
      "match Hedge does as well as its best expert.")

# ===================== Section 4: Experiments =====================
# 8. Exploitability and regret: theory vs measurement, one slide with plain explanations under each panel
s, S = content(4, "Our two bounds: theory vs measurement")
S.text(LEFT, TOP - Inches(0.05), WIDTH, Inches(0.45), [
    P(("Dashed line", {"b": True}), " = what our theory guarantees;  ", ("solid line or dots", {"b": True}),
      " = what we measured; every measurement stays on the safe side", size=14, align="ctr"),
], "ER legend")
from PIL import Image as _Im
img_path = os.path.join(HERE, "..", "figures", "fig_main_deck.png")
_w, _h = _Im.open(img_path).size
iy = TOP + Inches(0.45)
fh = Inches(3.7)
fw = int(fh * _w / _h)
fx = LEFT + (WIDTH - fw) // 2
pic = s.shapes.add_picture(img_path, fx, iy, fw, fh)
pic.name = "Exploitability and regret figure"
pic._element.nvPicPr.cNvPr.set("descr", "Four panels, each with the theoretical guarantee dashed and the measurement "
                                        "solid: exploitability bound vs actual exploitability on solved endgames; RM+ "
                                        "distance from equilibrium; Hedge regret per turn; opponent-model regret")
notes_txt = [
    "Each dot: one bot on an endgame we solved exactly. Our bound is never below the actual exploitability, but it "
    "is often far larger.",
    "Regret: how much better the best single expert would have done, per turn. Hedge's falls toward zero, under its "
    "guarantee.",
]
cw = fw // 2
ty = iy + fh + Inches(0.1)
for k, t in enumerate(notes_txt):
    S.text(fx + k * cw, ty, cw - Inches(0.2), BOTTOM - ty, [P(t, size=14)], f"ER note {k + 1}")
notes(s, "Here are the two bounds against measurements: the dashed line is what the theory guarantees, the solid "
      "line or dots what we measured. On the left, exploitability: we solved a few small endgames exactly, so we know "
      "each bot's actual exploitability there, and our bound is never below it, though it is often much larger. On "
      "the right, Hedge across games: its regret per turn, how much better its best single expert would have done, "
      "falls toward zero and stays well under the guarantee.")

# 4b. Limitations of the Hedge (preliminary) model: results, exploitability, then where the bound puts the error
from PIL import Image as _Im
s, S = content(4, "Limitations of the Hedge model")
lw = (WIDTH - GAP) // 2
S.header(LEFT, TOP, lw, HDR_H, [P("1. How it does", size=SECONDARY, b=True, align="ctr")], "Results header")
rows = [("Public ladder", "64–36, Elo 1704"),
        ("vs Foul Play", "0.33 (0.5 = even)")]
c0, rh = Inches(2.0), Inches(0.42)
ry = TOP + HDR_H + Inches(0.15)
for r, (a, b) in enumerate(rows):
    y = ry + r * rh
    S.header(LEFT, y, c0, rh, [P(a, size=14, b=True, align="ctr")], f"Results row {r}")
    S.box(LEFT + c0, y, lw - c0, rh, [P(b, size=14, align="ctr")], f"Results cell {r}")
ey = ry + len(rows) * rh + Inches(0.3)
S.header(LEFT, ey, lw, HDR_H, [P("Exploitability", size=SECONDARY, b=True, align="ctr")], "Exploit header")
S.box(LEFT, ey + HDR_H, lw, Inches(1.05), [
    P("How much more an opponent wins if it knows exactly how our bot decides. A simple opponent plays its best "
      "answer to the move it expects from us: once guessing blindly, once running our rule to predict it.", size=13),
], "Exploit box", anchor="t")
_hi = os.path.join(HERE, "..", "figures", "deck_exploit_hedge.png")
_hw, _hh = _Im.open(_hi).size
gy = ey + HDR_H + Inches(1.1)
gh = BOTTOM - gy - Inches(0.05)
gw = int(gh * _hw / _hh)
gp = s.shapes.add_picture(_hi, LEFT + (lw - gw) // 2, gy, gw, gh)
gp.name = "Hedge exploitability"
gp._element.nvPicPr.cNvPr.set("descr", "Opponent's score against the Hedge model: 0.37 guessing blindly, 0.56 knowing "
                                       "our rule, a gain of 0.19")

rx = LEFT + lw + GAP
S.header(rx, TOP, lw, HDR_H, [P("2. Why: the bound points at the payoffs", size=SECONDARY, b=True, align="ctr")],
         "Why header")
img = os.path.join(HERE, "..", "figures", "deck_delta.png")
_w, _h = _Im.open(img).size
fh = Inches(2.5)
fw = int(fh * _w / _h)
pic = s.shapes.add_picture(img, rx + (lw - fw) // 2, TOP + HDR_H + Inches(0.15), fw, fh)
pic.name = "Delta contributions"
pic._element.nvPicPr.cNvPr.set("descr", "Contributions to each turn's shortfall: solver error at most 0.16, matrix "
                                        "error 1.4 to 2.0")
ty = TOP + HDR_H + Inches(0.25) + fh
S.text(rx, ty, lw, BOTTOM - ty, [
    P("The true game: what each pair of moves is really worth if the battle is played out perfectly from there. "
      "Our matrix only estimates it with a 2-turn lookahead score.", size=14),
    P("The bound shows the main source of error is the matrix: how far those estimates are from the true game, "
      "not how well we solve them.", size=14, before=8),
], "Why text")
notes(s, "So why move on from the Hedge model? First, how it does: 64 and 36 on the public ladder at Elo 1704, and "
      "0.33 against Foul Play. Then exploitability: how much more an opponent wins if it knows exactly how our bot "
      "decides. We test it with a simple opponent that each turn plays its best answer to the move it expects from "
      "us, once guessing blindly and once running our rule to predict it. Against the Hedge model that knowledge is "
      "worth only 0.19, from 0.37 to 0.56, because it plays a mix. So it is hard to exploit, but too weak to win. "
      "Second, why. The true game is what each pair of moves is really worth if the battle is played out "
      "perfectly from there; our matrix only estimates it with a two-turn lookahead score. The bound splits each "
      "turn's shortfall into solver error and matrix error, and the matrix error, how far our estimates are from "
      "the true game, is by far the larger part.")

# 5. The Bayesian model: from the preliminary model's weakness to deeper search per world
s, S = content(4, "The Bayesian model: fix the payoffs")


def grid(x0, y0, cols, rows, widths, name, hh=Inches(0.45), rh=Inches(0.32), bold_cells=()):
    """Header row + label column table; cols = [(title, sub)], rows = [(label, [cells])]."""
    x = x0 + widths[0]
    for j, (c, sub) in enumerate(cols):
        S.header(x, y0, widths[j + 1], hh, [P(c, size=12, b=True, align="ctr")] +
                 ([P(sub, size=10, align="ctr")] if sub else []), f"{name} col {j}")
        x += widths[j + 1]
    for i, (r, vals) in enumerate(rows):
        yy = y0 + hh + i * rh
        S.header(x0, yy, widths[0], rh, [P(r, size=12, b=True, align="ctr")], f"{name} row {i}")
        x = x0 + widths[0]
        for j, v in enumerate(vals):
            S.box(x, yy, widths[j + 1], rh, [P(v, size=12, b=(i, j) in bold_cells, align="ctr")],
                  f"{name} cell {i}{j}")
            x += widths[j + 1]
    return y0 + hh + len(rows) * rh


# Row 1: how the old model did -> its weakness -> the fix
story = [("Keep: the belief over hidden sets", [
              P("Bayes over the opponent's possible sets, as before", size=14, align="ctr")]),
         ("Replace: 2-turn matrices with deep search", [
              P("Each world gets a full MCTS search, then a weighted vote", size=14, align="ctr")])]
BH = Inches(0.75)
for k, (title, paras) in enumerate(story):
    x = LEFT + k * (COL2 + GAP)
    S.header(x, TOP, COL2, HDR_H, [P(title, size=SECONDARY, b=True, align="ctr")], f"Story {k + 1} header")
    S.box(x, TOP + HDR_H, COL2, BH, paras, f"Story {k + 1} box")

# Row 2: worked example (belief -> per-world search -> weighted vote)
ey = TOP + HDR_H + BH + Inches(0.6)
S.text(LEFT, ey, Inches(5.6), Inches(0.35), [P("1. Belief: Bayes over Sunflora's possible sets", size=15, b=True)],
       "Step 1 title")
end1 = grid(LEFT, ey + Inches(0.4),
            [("Prior", "how common"), ("Saw Sludge Bomb?", ""), ("Belief", "")],
            [("Set A: Earth Power", ["0.45", "✓", "0.6"]), ("Set B: Leaf Storm", ["0.30", "✓", "0.4"]),
             ("Set C: no Sludge Bomb", ["0.25", "✗", "0"])],
            [Inches(2.15), Inches(1.1), Inches(1.3), Inches(1.1)], "Bayes",
            bold_cells={(0, 2), (1, 2), (2, 2)})
ax = LEFT + Inches(5.75)
S.line(ax, ey + Inches(1.0), ax + Inches(0.45), ey + Inches(1.0), "Belief arrow", arrow=True)
vx = ax + Inches(0.6)
S.text(vx, ey, RIGHT - vx, Inches(0.35), [P("2. Deep search in each world, then a weighted vote", size=15, b=True)],
       "Step 2 title")
end2 = grid(vx, ey + Inches(0.4), [("World A", "belief 0.6"), ("World B", "belief 0.4"), ("Weighted", "")],
            [("Earthquake", ["0.50", "0.25", "0.40"]), ("Stone Edge", ["0.20", "0.15", "0.18"]),
             ("Slack Off", ["0.30", "0.60", "0.42 ✓"])],
            [Inches(1.2), Inches(1.0), Inches(1.0), Inches(1.05)], "Vote", bold_cells={(2, 2)})
cy = max(end1, end2) + Inches(0.08)
S.text(LEFT, cy, WIDTH, Inches(0.6), [
    P("Cells on the right: share of each world's search spent on each move. We play the move with the highest "
      "belief-weighted share (Slack Off; World A alone would say Earthquake). Illustrative numbers.", size=12, i=True,
      align="ctr"),
], "Example caption")
notes(s, "So the Bayesian model keeps what worked, the belief over the opponent's hidden sets, and fixes the "
      "payoffs: instead of a shallow two-turn matrix, each world gets a full Monte Carlo tree search. Here is an "
      "example. The belief: Sunflora used Sludge Bomb, so set C, which has no Sludge Bomb, drops to zero, leaving "
      "0.6 on set A and 0.4 on set B. Then we search each world deeply, giving each of our moves a share of the "
      "search, and play the move with the highest belief-weighted share. World A alone would pick Earthquake, but "
      "weighting in world B makes Slack Off the choice. The numbers are illustrative.")

# 12. The Bayesian model: results (exploitability explained on top; paired-games table left; exploitability plot right)
from PIL import Image as _Im
s, S = content(4, "The Bayesian model: results")
S.text(LEFT, TOP - Inches(0.05), WIDTH, Inches(1.05), [
    P(("Stronger, but more exploitable. ", {"b": True}), "It is on par with Foul Play and reached the top 50, but "
      "because it always plays its top move, the same exploiter as for the Hedge model gains much more.", size=15),
], "Exploitability explained")
ty = TOP + Inches(1.15)
S.text(LEFT, ty, Inches(5.6), Inches(0.75), [
    P(("Public ladder: ", {"b": True}), "313–213, Elo 2308, reportedly top 50",
      size=14),
], "Ladder line")
rows = [("Bayesian model vs", "Score [95% CI]"),
        ("Foul Play, its default settings", "0.60 [0.48, 0.72]"),
        ("Foul Play, equal search time", "0.48 [0.35, 0.62]"),
        ("Our preliminary model", "0.83 [0.75, 0.92]")]
tbl_w, row_h = Inches(5.6), Inches(0.48)
c0 = int(tbl_w * 0.62)
for r, (a, b) in enumerate(rows):
    yy = ty + Inches(0.8) + r * row_h
    S.header(LEFT, yy, c0, row_h, [P(a, size=15, b=True, align="ctr" if r == 0 else "l")], f"Games row {r} label")
    if r == 0:
        S.header(LEFT + c0, yy, tbl_w - c0, row_h, [P(b, size=15, b=True, align="ctr")], "Games header score")
    else:
        S.box(LEFT + c0, yy, tbl_w - c0, row_h, [P(b, size=15, align="ctr")], f"Games row {r} score")
S.text(LEFT, ty + Inches(0.85) + row_h * len(rows), tbl_w, Inches(0.5), [
    P("Local paired games, 30 seat-swapped pairs each; 0.5 = even", size=12, i=True),
], "Table caption")
img_path = os.path.join(HERE, "..", "figures", "deck_exploit.png")
_w, _h = _Im.open(img_path).size
rx = LEFT + tbl_w + GAP
fw = RIGHT - rx
fh = int(fw * _h / _w)
ty = ty - Inches(0.35)
pic = s.shapes.add_picture(img_path, rx, ty, fw, fh)
pic.name = "Exploitability figure"
pic._element.nvPicPr.cNvPr.set("descr", "Opponent's score guessing blindly vs knowing our rule: Hedge model 0.37 to "
                                        "0.56, Bayesian model 0.21 to 0.70")
S.text(rx, ty + fh + Inches(0.05), fw, BOTTOM - ty - fh, [
    P("Knowing our rule lifts the opponent from 0.21 to 0.70: the Bayesian model always plays its top move, so it "
      "can be predicted. The Hedge model's mix gives away only 0.19.", size=14),
], "Exploit text")
notes(s, "Here are the Bayesian model's results. Strength first. On the public ladder it went 313 and 213 and reached "
      "2308 Elo, reportedly in the top 50. In local paired games "
      "it is on par with Foul Play, one of the strongest public bots: 0.60 at Foul Play's default settings and 0.48 at "
      "equal search time, and it beats our preliminary model 0.83. Then exploitability: how much more an opponent "
      "wins if it knows exactly how our bot decides. We play our bot against a simple opponent that each turn picks "
      "its best answer to the move it expects from us, twice. Guessing our move blindly, it scores 0.21. Running our "
      "rule to predict our move, it scores 0.70, because the Bayesian model always plays its top move. Randomising "
      "its choice removes this gap, but it then plays much worse. That trade-off between strength and predictability is what our theory says to expect.")

# ===================== Section 5: Conclusions =====================
# 12. Takeaways and next steps
s, S = content(5, "Takeaways and next steps")
items = ["Per-turn equilibrium play and Hedge across games compose into a bound on how exploitable the agent is over "
         "a whole battle, and the regrets we measured sit well inside it. The bound is most useful as a diagnosis: it "
         "shows that the error of our evaluation formula, not the solver, is what limits the agent.",
         "Spending compute where the bound points, on deeper search over the same Bayesian belief about the hidden "
         "team, gave our strongest agent, which reportedly reached the top 50 of the public ladder (Elo 2308). Its deterministic "
         "choice is the price: in-engine, an opponent who predicts it gains almost half a game.",
         "Hedge does what its bound promises, but it learns from the same shallow payoffs, so the guarantee holds in "
         "our estimate of the game rather than in the game being played; better evaluation is what turns it into wins."]
S.text(LEFT, TOP, WIDTH, Inches(4.0), [
    P(txt, number=True, size=18, before=0 if k == 0 else 14) for k, txt in enumerate(items)
], "Takeaways")
S.header(LEFT, TOP + Inches(4.05), WIDTH, Inches(0.38), [P("Next steps", size=SECONDARY, b=True, align="ctr")],
         "Next steps header")
S.box(LEFT, TOP + Inches(4.43), WIDTH, BOTTOM - TOP - Inches(4.43), [
    P("With more compute: learn a value function with RL and solve more precisely, so our no-regret learners see "
      "payoffs that match real play, the guarantees carry over to live games, and Hedge can join the Bayesian model "
      "without losing strength", size=15, align="ctr"),
], "Next steps box")
notes(s, "Three takeaways. First, per-turn equilibrium play and Hedge across games compose into a bound on how "
      "exploitable the agent is over a whole battle, and the regrets we measured sit well inside it. The bound turned "
      "out to be most useful as a diagnosis: it shows that the error of our evaluation formula, not the solver, is "
      "what limits the agent. Second, spending compute where the bound points, on deeper search over the same "
      "Bayesian belief about the hidden team, gave our strongest agent, which reportedly reached the top 50 of the "
      "public ladder; the price is that its deterministic choice can be predicted, and an opponent who does so gains "
      "almost half a game. Third, Hedge does what its bound promises, but it learns from the "
      "same shallow payoffs, so the guarantee holds in our estimate of the game rather than in the game being "
      "played, and better evaluation is what turns it into wins. Next, with more "
      "compute, we would learn a value function with reinforcement learning and solve positions more precisely, so "
      "that the payoffs our no-regret learners and bounds are measured on match real play. Then the guarantees would "
      "carry over to live games, Hedge could be merged into the Bayesian model without losing strength, and the agent "
      "would get stronger. Thank you.")

add_sections(prs, groups)
prs.core_properties.title = "No-Regret Learning and Bayesian Opponent Memory in Pokémon Random Battles"
prs.core_properties.author = "Nikhil Pesaladinne, Bo Chi"
prs.save(OUT)
print("wrote", OUT)
