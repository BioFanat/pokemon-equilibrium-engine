"""Build deck.pptx: the 13-slide, ~4-minute CS535 final-project talk, in five sections.

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
S.text(1524000, 4880000, 9144000, 460000, [P("[Names] · CS535", color=fg, align="ctr")], "Authors")
notes(s, "Hi, we are [Names]. PokaiTrainer showed that per-turn equilibrium play over sampled opponent "
      "worlds is competitive in Pokémon VGC. We ask whether the same idea, built entirely from no-regret learning, "
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
      "tree search over sampled teams, and RL agents learn a policy from self-play. Neither is built to be safe. "
      "Any fixed way of playing has habits, and an opponent who knows them can beat it, no matter how well it does "
      "against everyone else. So our goal is different: an agent built from no-regret learning that is hard to "
      "exploit, with a bound on how much a best-responding opponent can take from it, and then to measure what that "
      "safety costs in strength. Two yardsticks. Exploitability is what we try to make small: the best guarantee "
      "any strategy can secure, minus what ours secures against a best response; it is zero at an equilibrium. "
      "Regret is how our learners are judged: our total loss minus the best fixed choice's in hindsight, and "
      "no-regret means the average goes to zero.")

# 3. Background theory and our goals
s, S = content(1, "Background: no-regret theory")
S.text(LEFT, TOP, WIDTH, Inches(3.3), [
    P(("MWU", {"b": True}), " (Littlestone & Warmuth): shrink each expert's weight after a loss, ",
      ("$", E_MWU), ";  ", ("$", r"Regret_T ≤ 2\sqrt{T \ln{n}}"), size=18, bullet=True),
    P(("Hedge", {"b": True}), " (Freund & Schapire), the exponential form we use, ", ("$", E_HEDGE),
      ";  with an anytime step, ", ("$", r"Regret_T ≤ \frac{3}{2}\sqrt{\frac{T}{2}\ln{n}} + \sqrt{\frac{\ln{n}}{8}}"),
      size=18, bullet=True, before=12),
    P(("No-regret dynamics ⇒ minimax", {"b": True}), " (Freund & Schapire): if both players of a matrix game ",
      ("$", "A"), " are no-regret, ", ("$", E_MINIMAX), size=18, bullet=True, before=12),
    P(("No-regret dynamics ⇒ CCE:", {"b": True}), " in any finite game, time-average play is an ε-coarse "
      "correlated equilibrium", size=18, bullet=True, before=12),
], "Background results")
gy = TOP + Inches(3.45)
S.text(LEFT, gy, WIDTH, BOTTOM - gy, [
    P("Our goals", b=True),
    P("Use these bounds to characterise how exploitable our agent is over a whole battle", size=18, bullet=True,
      before=4),
    P("Check in practice: measure exploitability and regret", size=18, bullet=True, before=4),
    P("See whether the agent can still win", size=18, bullet=True, before=4),
], "Goals")
notes(s, "A few known results we build on. Multiplicative weights shrinks each expert's weight after a loss, and "
      "its regret grows like root T log n. Hedge is the exponential form, which we use; with an anytime step it "
      "needs no horizon and has the same rate. When both players of a matrix game are no-regret, their average "
      "strategies are within the sum of their average regrets of an equilibrium, which is the minimax theorem, and "
      "in any finite game time-average play is a coarse correlated equilibrium. Our goals: use these bounds to "
      "characterise how exploitable our agent is over a whole battle, check that in practice by measuring "
      "exploitability and regret, and see whether the agent can still win.")

# ===================== Section 2: Method =====================
# 4. Preliminary model: worlds, per-turn equilibrium, experts
s, S = content(2, "Preliminary model: worlds, equilibrium, experts")
steps = [("1. Sample 4 worlds", [P("each a full guess of the hidden team and sets, consistent with what we saw; "
                                   "weight ", ("$", "β_k"), size=SECONDARY, align="ctr")]),
         ("2. A matrix per world", [P("rows: our options; columns: theirs; cell: our value after a 2-turn "
                                      "lookahead → ", ("$", "M_k"), size=SECONDARY, align="ctr")]),
         ("3. One strategy x", [P("best on average when the opponent best-responds in each world (one LP)",
                                  size=SECONDARY, align="ctr")]),
         ("4. Hedge over experts", [P("top move, sampled, pure maximin, ε-safe best response; weights carried "
                                      "across games", size=SECONDARY, align="ctr")])]
BH = Inches(1.6)
for k, (name, paras) in enumerate(steps):
    x = LEFT + k * (COL4 + GAP)
    S.header(x, TOP, COL4, HDR_H, [P(name, size=SECONDARY, b=True, align="ctr")], f"Prelim step {k + 1} header")
    S.box(x, TOP + HDR_H, COL4, BH, paras, f"Prelim step {k + 1} box")
    if k < 3:
        ym = TOP + HDR_H + BH // 2
        S.line(x + COL4 + Inches(0.05), ym, x + COL4 + GAP - Inches(0.05), ym, f"Prelim arrow {k + 1}", arrow=True)
ty = TOP + HDR_H + BH + Inches(0.2)
S.text(LEFT, ty, WIDTH, BOTTOM - ty, [
    EQ_(E_BLP, size=20),
    P("One ", ("$", "x"), " for all worlds, since we cannot tell which world we are in", size=18, bullet=True,
      before=8),
    P("Each expert is credited with what it would have earned against the opponent's actual move", size=18,
      bullet=True, before=6),
    P("Opponent memory: a posterior over 5 play styles, ", ("$", r"Π_t(θ) ∝ Π_{t−1}(θ)\sum_k{β_k q_θ(b_t | M_k)}"),
      "; it is Hedge with ", ("$", "η = 1"), " and feeds the ε-safe expert", size=18, bullet=True, before=6),
], "Preliminary notes")
notes(s, "Our starting point. Each turn we sample four worlds, each a full guess of the opponent's hidden team and "
      "sets consistent with what we have seen, with a weight. For each world the engine builds a payoff matrix: "
      "our options against theirs, each cell our value after a two-turn lookahead. We pick one mixed strategy x "
      "that does best on average when the opponent best-responds in each world, a single linear program, because "
      "we cannot tell which world we are in. Across games, Hedge mixes four experts, four ways of playing x: its "
      "top move, a sample from it, the single safest action, and an epsilon-safe best response to the opponent "
      "memory's prediction. Each expert is credited with what it would have earned against the opponent's actual "
      "move. The opponent memory is a posterior over five play styles, updated with each observed move; with no "
      "forgetting it is exactly Hedge with step one on log loss, which is where its regret bound comes from. This "
      "is the model our theorems cover.")

# 5. The Bayesian model: the model we play, built on the preliminary formulation
s, S = content(2, "The Bayesian model: what we play")
S.text(LEFT, TOP, WIDTH, Inches(0.5), [
    P("Same belief over worlds, used more carefully; search in each world replaces the matrix and Hedge layer",
      size=18),
], "Bayesian model intro")
steps = [("1. Belief", [P("prior: how often each set occurs, conditioned on everything revealed", size=SECONDARY,
                              align="ctr")]),
         ("2. Stratified worlds", [P("a world for each item/ability group of the active opponent; weights ",
                                     ("$", "β_k"), " keep each group's probability", size=SECONDARY, align="ctr")]),
         ("3. Search each world", [P("MCTS (DUCT, as in Foul Play), 250 ms → share of root visits", size=SECONDARY,
                                     align="ctr")]),
         ("4. Weighted choice", [P("play the largest ", ("$", "β"), "-weighted visit share", size=SECONDARY,
                                         align="ctr")])]
sy = TOP + Inches(0.6)
BH = Inches(1.5)
for k, (name, paras) in enumerate(steps):
    x = LEFT + k * (COL4 + GAP)
    S.header(x, sy, COL4, HDR_H, [P(name, size=SECONDARY, b=True, align="ctr")], f"Bayes step {k + 1} header")
    S.box(x, sy + HDR_H, COL4, BH, paras, f"Bayes step {k + 1} box")
    if k < 3:
        ym = sy + HDR_H + BH // 2
        S.line(x + COL4 + Inches(0.05), ym, x + COL4 + GAP - Inches(0.05), ym, f"Bayes arrow {k + 1}", arrow=True)
ty = sy + HDR_H + BH + Inches(0.15)
S.text(LEFT, ty, WIDTH, BOTTOM - ty, [
    EQ_(r"a = arg\max_a{\sum_k{β_k \frac{n_k(a)}{\sum_b{n_k(b)}}}}", size=20),
    P(("Stratified worlds: ", {"b": True}), "still unbiased, every group covered, no between-group sampling error "
      "(Choice Specs missed 34% of the time → never)", size=16, bullet=True, before=6),
    P(("Search: ", {"b": True}), "goes deeper, at the evaluation error our bound points to, but gives up the solver "
      "guarantee (each world searched as if known)", size=16, bullet=True, before=6),
    P(("Argmax: ", {"b": True}), "near-deterministic, so it pays the price of determinism (theory)", size=16,
      bullet=True, before=6),
], "Bayesian model notes")
notes(s, "This is the model we actually play, built on the same formulation. It keeps the belief over the hidden "
      "team: the prior is how often each set occurs in Random Battles, conditioned on everything revealed. It uses "
      "that belief more carefully: instead of four independent draws, it gives each item and ability group of the "
      "active opponent its own world, with weights that keep each group's probability. Averages over worlds stay "
      "unbiased, every group is covered, and the part of the sampling error that comes from which group we drew "
      "disappears; with plain draws, one test missed Choice Specs a third of the time. Then, instead of a two-turn "
      "matrix and Hedge, it runs poke-engine's MCTS, the same DUCT search Foul Play uses, in each world for 250 "
      "milliseconds, turns each world's root visits into shares, weights them by the worlds' probabilities and "
      "plays the top action. Deeper search goes after the evaluation error that our bound says dominates, but it "
      "gives up the solver guarantee, and it is nearly deterministic, so it pays the price of determinism we prove "
      "in the theory section.")

# 6. Case study: one turn of a real Showdown game
s, S = content(2, "A turn, step by step")
gh = BOTTOM - TOP - Inches(0.5)  # room for a two-line caption
gw = int(gh * 800 / 542)  # 800 x 542 px GIF
gx = LEFT + (WIDTH - gw) // 2
pic = s.shapes.add_picture(GIF, gx, TOP, gw, gh)
pic.name = "Case-study animation"
pic._element.nvPicPr.cNvPr.set(
    "descr", "Showdown replay against Foul Play above our agent's decision panel: a slice of the Bayesian "
             "matrix game, our mixed strategy as bars, estimated win chance in each cell")
S.text(gx, TOP + gh + Inches(0.03), gw, Inches(0.47),
       [P("A real Showdown game: each turn the agent picks one mixed strategy against 4 sampled opponent teams; "
          "bars show our mixed strategy, cells our estimated win chance.", size=CAPTION, i=True)],
       "Animation caption")
notes(s, "This is a real Showdown game against Foul Play; the names at the top are the bots' account names. Each "
      "turn the agent samples 4 possible opponent teams and picks one mixed strategy that does best against all of them. The panel shows "
      "a slice: rows are our options, columns the opponent's, cells our estimated win chance. The bars are our mixed "
      "strategy, and the outlined cell is what we and Foul Play actually played.")

# ===================== Section 3: Theory =====================
# 8. The whole-battle bound
s, S = content(3, "The whole-battle bound")
S.text(LEFT, TOP, WIDTH, Inches(1.6), [
    P("Each turn our strategy can lose at most ", ("$", "δ = ε_s + 2ε_v"), " against the true game, and "
      "exploitability is the sum of these per-turn losses over the battle, so", size=18),
    EQ_(E_BOUND.split(",")[0], before=6),
], "Bound statement")
by = TOP + Inches(1.7)
BH3 = Inches(1.3)
terms = [([("$", "ε_s"), (": solver error", {})], [P("how far we are from our matrix's equilibrium", size=SECONDARY, align="ctr"),
                                P("shrinks with compute (≤ ", ("$", "ε_R + ε_C"), ")", size=SECONDARY, align="ctr",
                                  before=4)]),
         ([("$", "ε_v"), (": matrix error", {})], [P("how wrong our matrix is, mostly the evaluation formula", size=SECONDARY,
                                  align="ctr"),
                                P("more compute does not shrink it", size=SECONDARY, align="ctr", before=4)]),
         ([("$", r"\bar{H}"), (": battle length", {})], [P("expected turns an exploiter plays", size=SECONDARY, align="ctr"),
                               P("(δ̄: average δ along its games)", size=SECONDARY, align="ctr", before=4)])]
for k, (name, paras) in enumerate(terms):
    x = LEFT + k * (COL3 + GAP)
    S.header(x, by, COL3, HDR_H, [P(*name, size=SECONDARY, b=True, align="ctr")], f"Term {k + 1} header")
    S.box(x, by + HDR_H, COL3, BH3, paras, f"Term {k + 1} box")
gy = by + HDR_H + BH3 + Inches(0.3)
S.header(LEFT, gy, WIDTH, HDR_H, [P("Goal: use the bound to characterise exploitability", size=SECONDARY, b=True,
                                    align="ctr")], "Goal header")
S.box(LEFT, gy + HDR_H, WIDTH, Inches(1.05), [
    P("Measure each term on real positions and see which one limits the agent", size=18, bullet=True),
    P("Deterministic play pays more: ", ("$", r"expl ≥ E[\sum_t{g_t}]"), ", ", ("$", "g_t"),
      " = game value − best pure guarantee", size=18, bullet=True, before=6),
], "Goal box", anchor="t")
notes(s, "Our main result in one line. On each turn, our strategy can lose at most delta against the true game: the "
      "solver error plus twice the matrix error. Exploitability is exactly the sum of per-turn losses against a best "
      "response, so it is at most H-bar, the number of turns that exploiter plays, times the average delta along "
      "its games. The solver error shrinks with compute, and for no-regret self-play it is at most the summed "
      "regrets. The matrix error comes mostly from the evaluation formula, and more compute does not shrink it. Our "
      "goal is to use this to characterise exploitability: measure each term on real positions and see which one "
      "limits the agent. And a new lower bound: a deterministic policy loses at least the gap between each turn's "
      "game value and its best pure guarantee, on every turn where no pure strategy is optimal, which is 42 percent "
      "of our stage matrices. That is the price our Bayesian model's argmax pays. Hedge across games adds its own "
      "regret on top.")

# ===================== Section 4: Experiments =====================
# 8. Exploitability
s, S = content(4, "Exploitability and regret")
fw = WIDTH
fh = int(fw * 416 / 1395)  # 1395 x 416 px figure
pic = s.shapes.add_picture(FIG_MAIN, LEFT, TOP, fw, fh)
pic.name = "Exploitability and regret figure"
pic._element.nvPicPr.cNvPr.set(
    "descr", "Four panels: exact exploitability against the whole-battle bound; RM+ distance from equilibrium over iterations; "
             "Hedge regret per counted turn over 10 matches; memory regret against the bound ln 5")
by = TOP + fh + Inches(0.1)
S.text(LEFT, by, WIDTH, BOTTOM - by, [
    P("Exploitability lower bounds: ", ("0.25", {"b": True}), " top move, ", ("0.09", {"b": True}),
      " sampled play", size=18, bullet=True),
    P("Exact endgames: the bound holds, loose by ", ("7–155×", {"b": True}), "; evaluation error ", ("$", "ε_v"),
      " dominates", size=18, bullet=True, before=6),
    P("Regrets (b–d) stay under their bounds", size=18, bullet=True, before=6),
], "Exploitability bullets")
notes(s, "An informed best response that knows our policy gives exploitability lower bounds of 0.25 for top move and "
      "0.09 for sampled play, so randomising helps. On exact endgames the whole-battle bound holds on every row, loose "
      "by 7 to 155 times, and the evaluation error, not the solver, carries it. Panels b to d show the regrets under their "
      "bounds; the next slide gives the numbers.")

# 9. No-regret dynamics and the opponent memory
s, S = content(4, "No-regret dynamics and the opponent memory")
S.text(LEFT, TOP, WIDTH, BOTTOM - TOP, [
    P("Theory: if both players have average regrets ", ("$", "ε_R, ε_C"), ", their average strategies are within ",
      ("$", "ε_R + ε_C"), " of minimax"),
    EQ_(E_MINIMAX, before=4),
    P("Inside a turn: RM⁺ is within ", ("7×10⁻⁵", {"b": True}), " of the payoff range of minimax (bound 0.38); "
      "Hedge's averages stay within the bound on all 120 matrices", bullet=True, before=20),
    P("Across games: Hedge regret ", ("≈ 0.004", {"b": True}), " per turn vs Foul Play (13–16% of the bound); "
      "carried beat reset weights by ", ("+0.15", {"b": True}), bullet=True, before=SP),
    P("Opponent memory (Bayes = Hedge): regret ends at its bound ", ("$", r"\ln{5}"), "; per-world scoring lowers "
      "log loss by ", ("≈ 0.3", {"b": True}), " nats per turn, but win rate does not move", bullet=True, before=SP),
], "Dynamics bullets")
notes(s, "Theory says no-regret players' averages are within the sum of their regrets of minimax. Measured: RM-plus "
      "inside a turn reaches 7 times ten to the minus five of the payoff range, against a bound of 0.38, and Hedge's "
      "averages stay within the bound on all 120 stage matrices we tried. Across games, Hedge's regret against Foul "
      "Play is 13 to 16 percent of its bound, and carried weights beat reset weights, as the bounds order them. The "
      "opponent memory is Bayes, which is Hedge with step one, and its regret ends exactly at its bound, log five. "
      "Scoring styles on each sampled world's legal actions lowers log loss by about 0.3 nats per turn, but better "
      "prediction does not move win rate, which fits the theory: none of our bounds rewards better prediction.")

# 11. Win rates
s, S = content(4, "Win rates")
fh = BOTTOM - TOP - Inches(0.4)
fw = int(fh * 655 / 525)  # 655 x 525 px figure
pic = s.shapes.add_picture(FIG_WINS, LEFT, TOP, fw, fh)
pic.name = "Win-rate figure"
pic._element.nvPicPr.cNvPr.set(
    "descr", "Score of the first-named bot with 95% CIs on real Showdown, in the round-robin tournament and "
             "in-engine: Bayesian LP below Foul Play at every search time, above PIMC averaging, below MCTS")
S.text(LEFT, TOP + fh + Inches(0.03), fw, Inches(0.35),
       [P("Score of the first-named bot, 95% CIs (0.5 = even)", size=CAPTION, i=True)], "Figure caption")
rx = LEFT + fw + GAP
S.text(rx, TOP, RIGHT - rx, BOTTOM - TOP, [
    P("MCTS beats equilibrium play at matched time", bullet=True),
    P("The Bayesian LP beats PIMC averaging with sampled play", bullet=True, before=18),
    P("It loses to Foul Play at every search time: ", ("≈140–190", {"b": True}), " Elo below",
      bullet=True, before=18),
], "Win-rate bullets")
notes(s, "Do guarantees that hold only in the estimated matrix translate into wins? Not against search. At matched "
      "time, about 70 milliseconds per move, MCTS beats both equilibrium bots. Solving one Bayesian game over the "
      "sampled worlds does beat PIMC averaging with sampled play. But the agent loses to Foul Play at every search "
      "time we tried, and in the round-robin tournament Foul Play rates about 140 to 190 Elo above it.")

# 12. Out in the world: the public ladder
s, S = content(4, "Out in the world: the public ladder")
S.text(LEFT, TOP, WIDTH, Inches(0.8), [
    P("Gen 9 Random Battles against human players: sequential rated batches on one account, not paired tests",
      size=18),
], "Ladder intro")
stats = [("Preliminary model", "77–73", ["150 games · Elo 1577 → 1470"]),
         ("Bayesian model", "99–51", ["150 games · Elo 1470 → 2090", "peak 2141, top 500"])]
sy = TOP + Inches(0.85)
SBH = Inches(1.9)
for k, (name, big, smalls) in enumerate(stats):
    x = LEFT + k * (COL2 + GAP)
    S.header(x, sy, COL2, HDR_H, [P(name, size=SECONDARY, b=True, align="ctr")], f"Ladder stat {k + 1} header")
    S.box(x, sy + HDR_H, COL2, SBH, [P(big, size=44, b=True, align="ctr")] +
          [P(t, size=18, b=(j == 1), align="ctr", before=6) for j, t in enumerate(smalls)],
          f"Ladder stat {k + 1} box")
    if k == 0:
        ym = sy + HDR_H + SBH // 2
        S.line(x + COL2 + Inches(0.05), ym, x + COL2 + GAP - Inches(0.05), ym, "Ladder arrow", arrow=True)
by = sy + HDR_H + SBH + Inches(0.3)
S.text(LEFT, by, WIDTH, BOTTOM - by, [
    P("Strongest bot we built, as MCTS beat equilibrium play in-engine", size=18, bullet=True),
    P("Near-deterministic, so it pays the price of determinism; its exploitability is unmeasured", size=18,
      bullet=True, before=10),
], "Ladder notes")
notes(s, "Every other test uses fixed bots, so we also played on the public Showdown ladder against human players, in "
      "sequential rated batches on one account; these are not paired tests, and opponents and ratings change as we "
      "go. The preliminary model, equilibrium play with Hedge and the opponent memory, went 77 and 73 over 150 games "
      "and drifted from 1577 to 1470 Elo. The Bayesian model then went 99 and 51 over 150 games, climbing from 1470 "
      "to 2090 and peaking at 2141, which the operator saw as the top 500; its stratified worlds were in place from "
      "game 51. That is the strongest bot we built, in line with MCTS beating equilibrium play in-engine. But it is "
      "nearly deterministic, so our lower bound says it can be exploited, and we have not measured by how much: "
      "strength and guarantees pull apart.")

# ===================== Section 5: Conclusions =====================
# 12. Takeaways and next steps
s, S = content(5, "Takeaways and next steps")
items = ["Equilibrium play and Hedge compose into a whole-battle bound; measured regrets sit well under their "
         "bounds",
         "The guarantees hold in our estimated matrices; the evaluation formula's error, which more compute does not "
         "reduce, is what our experiments point to, and better beliefs do not move win rate",
         "On the public ladder, the Bayesian model reached the top 500 while the preliminary model stayed near 1500: "
         "strength and guarantees pull apart",
         "Next: train the evaluation on exact terminal values"]
S.text(LEFT, TOP, WIDTH, BOTTOM - TOP, [
    P(txt, number=True, before=0 if k == 0 else 22) for k, txt in enumerate(items)
], "Takeaways")
notes(s, "Four takeaways. First, per-turn equilibrium play and Hedge across games compose into a whole-battle bound, "
      "and the regrets we measured sit well under their bounds. Second, those guarantees hold in the "
      "estimated matrices. The error of the evaluation formula that scores positions two moves ahead is not reduced "
      "by more solver iterations or more sampled teams, and our experiments point to it as the limiting error, "
      "consistent with better beliefs improving prediction but not win rate. Third, on the public ladder, "
      "the Bayesian model climbed to the top 500 while the preliminary model stayed near 1500, so strength and guarantees pull "
      "apart. Fourth, the next step is to train the evaluation on exact terminal values. Thank you.")

add_sections(prs, groups)
prs.core_properties.title = "No-Regret Learning and Bayesian Opponent Memory in Pokémon Random Battles"
prs.core_properties.author = "[Names]"
prs.save(OUT)
print("wrote", OUT)
