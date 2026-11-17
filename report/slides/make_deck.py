"""Build deck.pptx: the 13-slide, ~4-minute CS535 final-project talk, in five sections.

Built on the user's template ("random template.pptx"): its master, theme, 16:9 size,
title-slide style and its five-box section tracker, which lives in the template's
"1_Title and Content" ... "5_Title and Content" layouts. Only the tracker labels change.
Content is plain text (Helvetica Neue, 20 pt, secondary 16 pt) with equations as native
PowerPoint equations (OMML, Cambria Math, 22 pt), wrapped like the template's slide 6:
mc:AlternateContent with an a14:m choice and a plain Unicode text fallback. One slide keeps
one small group of the template's slide-3 boxes (accent5 header, bg2 box, tx1 outline).
Every number is taken from ../final5.tex.

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
UPRIGHT = {"Regret", "MWU", "expl", "gap", "val"}  # multi-letter names set upright
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
        if tok in ("\\max", "\\min"):
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
E_EXPL = r"expl(x) = val(A) − \min_y{x^⊤Ay}"
E_GAP = r"gap(x, y) = \max_i{{(Ay)}_i} − \min_j{{(x^⊤A)}_j}"
E_MWU = r"w_i^{t+1} = w_i^t(1 − ηl_i^t)"
E_HEDGE = r"w_i^{t+1} = w_i^t e^{−ηl_i^t}"
E_SOFTMAX = r"p_i^t ∝ \exp{(η\sum_{j<t}{r_i^j})}"
E_BLP = r"\max_x{\sum_k{β_k \min_j{{(M_k^⊤x)}_j}}}"
E_MIX = r"p_t = \sum_e{w_e q_e}"
E_MWU_BOUND = r"L_{MWU} ≤ (1 + η)L^* + \frac{\ln{n}}{η}"
E_HEDGE_BOUND = r"Regret_T ≤ \sqrt{\frac{T}{2}\ln{n}}"
E_MINIMAX = r"\max_x{x^⊤A\bar{y}} − \min_y{\bar{x}^⊤Ay} ≤ ϵ_R + ϵ_C"
E_BOUND = r"expl(π) ≤ \bar{H}\sup{(ε_s + 2ε_v)}"
E_COR = r"expl(π) ≤ \bar{H}(ϵ_R + ϵ_C + 2ε_v)"


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

    def box(self, x, y, w, h, paras, name):
        """Template slide-3 content box: bg2 fill, tx1 outline, tx1 text."""
        return self.rect(x, y, w, h, paras, FILL_BOX, name)

    def text(self, x, y, w, h, paras, name, anchor="t"):
        return self._sp(lambda sid, body: (
            f'<p:sp {NS}><p:nvSpPr><p:cNvPr id="{sid}" name="{esc(name)}"/><p:cNvSpPr txBox="1"/><p:nvPr/></p:nvSpPr>'
            f'<p:spPr><a:xfrm><a:off x="{int(x)}" y="{int(y)}"/><a:ext cx="{int(w)}" cy="{int(h)}"/></a:xfrm>'
            f'<a:prstGeom prst="rect"><a:avLst/></a:prstGeom><a:noFill/></p:spPr>'
            f'<p:txBody><a:bodyPr wrap="square" rtlCol="0" anchor="{anchor}"><a:noAutofit/></a:bodyPr><a:lstStyle/>'
            f'{body}</p:txBody></p:sp>'), paras)

    def line(self, x1, y1, x2, y2, name, colour="tx1", w=25400):
        """Template slide-3 straight connector: 2 pt tx1 line (w in EMU)."""
        sid = self._id()
        flip = ' flipV="1"' if y2 < y1 else ""
        x, y = min(x1, x2), min(y1, y2)
        return self._add(
            f'<p:cxnSp {NS}><p:nvCxnSpPr><p:cNvPr id="{sid}" name="{esc(name)}"/><p:cNvCxnSpPr/><p:nvPr/></p:nvCxnSpPr>'
            f'<p:spPr><a:xfrm{flip}><a:off x="{int(x)}" y="{int(y)}"/><a:ext cx="{int(abs(x2 - x1))}" cy="{int(abs(y2 - y1))}"/>'
            f'</a:xfrm><a:prstGeom prst="line"><a:avLst/></a:prstGeom><a:ln w="{w}"><a:solidFill>'
            f'<a:schemeClr val="{colour}"/></a:solidFill></a:ln></p:spPr>{LINE_STYLE}</p:cxnSp>')

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

# 2. The game and the yardsticks
s, S = content(1, "The game and the yardsticks")
S.text(LEFT, TOP, WIDTH, BOTTOM - TOP, [
    P("A battle is a zero-sum stochastic game: simultaneous moves, chance, hidden teams → a stage matrix each turn"),
    P(("Regret", {"b": True}), "  our total loss minus the best expert's total loss in hindsight", before=22),
    EQ_(E_REG, before=4),
    P(("Exploitability", {"b": True}), "  loss against a best-responding opponent, relative to the game value",
      before=22),
    EQ_(E_EXPL, before=4),
    P(("Duality gap", {"b": True}), "  the exploitability of both players together; zero exactly at equilibrium",
      before=22),
    EQ_(E_GAP, before=4),
], "Definitions")
notes(s, "A battle is a zero-sum stochastic game with simultaneous moves, chance and hidden teams, so each turn "
      "we face a stage matrix. Three yardsticks. Regret: in the experts problem, our total loss minus the total loss "
      "of the best expert in hindsight; an algorithm is no-regret if this grows slower than the number of rounds. "
      "Exploitability: how much less than the game value we earn against a best-responding opponent. The duality gap "
      "is the exploitability of both players together; it is zero exactly at equilibrium.")

# 3. MWU -> Hedge, and regret => equilibrium
s, S = content(1, "MWU → Hedge, and regret ⇒ equilibrium")
ch = Inches(1.75)
S.text(LEFT, TOP, COL2, ch, [
    P("MWU", b=True),
    EQ_(E_MWU, before=8),
    P("linear in the loss", size=SECONDARY, align="ctr", before=6),
], "MWU column")
S.text(LEFT + COL2 + GAP, TOP, COL2, ch, [
    P("Hedge (ours)", b=True),
    EQ_(E_HEDGE, before=8),
    EQ_(E_SOFTMAX, before=4),
], "Hedge column")
S.text(LEFT, TOP + ch + Inches(0.1), WIDTH, BOTTOM - TOP - ch - Inches(0.1), [
    P("Exponential in the loss = a softmax of cumulative reward (the common factor cancels)", bullet=True),
    P("Bayes = Hedge with ", ("$", "η = 1"), " on log loss", bullet=True, before=6),
    P(("No-regret dynamics ⇒ minimax", {"b": True}), ("  (Freund & Schapire 1999)", {"size": SECONDARY}),
      before=24),
    P("both players no-regret, with average regrets ", ("$", "ϵ_R"), " and ", ("$", "ϵ_C"), ":",
      size=SECONDARY, before=4),
    EQ_(E_MINIMAX, before=6),
    P("and the time-average play is an ε-coarse correlated equilibrium (CCE)", size=SECONDARY, align="ctr",
      before=6),
], "Hedge and no-regret dynamics")
notes(s, "Multiplicative weights updates linearly in the loss. We use Hedge, the exponential form of multiplicative "
      "weights. With small steps it behaves like MWU, but it is the form the no-regret-in-games literature analyses, "
      "its anytime version suits matches of unknown length, and Bayesian updating is exactly Hedge with step one, so "
      "our opponent memory inherits the same no-regret guarantee. Inside a turn we use regret matching (RM-plus), as "
      "CFR and PokaiTrainer do. The key result: if both players are no-regret, the average strategies are an "
      "approximate minimax equilibrium, off by at most the sum of their average regrets, and time-average play is an "
      "approximate coarse correlated equilibrium.")

# ===================== Section 2: Method =====================
# 4. Where no-regret learning enters
s, S = content(2, "Equilibrium Play and Hedge")
cols = [
    ("Each turn: play an equilibrium", [
        P("Solve the Bayesian matrix game over the sampled worlds: an LP, or RM⁺ self-play (no-regret dynamics ⇒ minimax)", before=8),
        EQ_(E_BLP, before=10),
        P("PIMC averaging instead solves each world as if known, then averages → strategy fusion",
          size=SECONDARY, before=10)]),
    ("Across games: learn with Hedge", [
        P("Hedge over per-turn experts, learning from the opponent's observed actions", before=8),
        P("A Bayesian opponent memory, whose update is Hedge with η = 1", before=8)]),
]
for k, (head, paras) in enumerate(cols):
    S.text(LEFT + k * (COL2 + GAP), TOP, COL2, BOTTOM - TOP, [P(head, b=True)] + paras, f"{head} column")
S.text(LEFT, BOTTOM - Inches(0.7), WIDTH, Inches(0.6),
       [P("Foul Play's tree search is also self-play with a bandit at every node, but its visit counts need not "
          "converge to an equilibrium (Lisý et al. 2013), so we solve each turn's game directly",
          size=SECONDARY, i=True)], "Search note")
notes(s, "Each turn, the agent plays an equilibrium of PokaiTrainer's Bayesian matrix game: one strategy x against "
      "an opponent that best-responds in each sampled world. We solve it as one LP, or with "
      "RM-plus self-play when there is a single world, whose convergence is no-regret dynamics giving minimax. PIMC averaging instead solves each world "
      "separately and suffers strategy fusion. The actual learning happens across games: Hedge over per-turn experts, "
      "and a Bayesian opponent memory whose update is Hedge with step one.")

# 5. Across games: an experts problem
s, S = content(2, "Across games: an experts problem")
S.text(LEFT, TOP, WIDTH, Inches(0.8),
       [P("We cast choosing how to play as the experts problem. The experts are fixed rules applied "
          "to each turn's solved game:")], "Intro")
experts = [("Top move", [P("argmax of ", ("$", "x"), size=SECONDARY, align="ctr")]),
           ("Sampled", [P("play ", ("$", "x"), size=SECONDARY, align="ctr")]),
           ("Pure maximin", [P("the security row", size=SECONDARY, align="ctr")]),
           ("ε-safe best response", [P("to the predicted opponent ", ("$", r"\hat{q}"), size=SECONDARY,
                                       align="ctr")])]
gy = TOP + Inches(0.85)
for k, (name, rule) in enumerate(experts):
    x = LEFT + k * (COL4 + GAP)
    S.header(x, gy, COL4, HDR_H, [P(name, size=SECONDARY, b=True, align="ctr")], f"Expert {k + 1} header")
    S.box(x, gy + HDR_H, COL4, Inches(0.7), rule, f"Expert {k + 1} box")
by = gy + HDR_H + Inches(0.7) + Inches(0.2)
S.text(LEFT, by, WIDTH, BOTTOM - by, [
    P("Hedge mixes them:", bullet=True),
    EQ_(E_MIX, before=2),
    P("Full-information feedback: each rule is credited with what it would have earned against the "
      "observed opponent action", bullet=True, before=6),
    P("Weights kept across the games of a match", bullet=True, before=6),
    P("Hedge competes with the best rule in hindsight: fixed actions become fixed rules",
      bullet=True, before=6),
], "Hedge bullets")
notes(s, "Across games, we cast choosing how to play as the experts problem. Each expert is a fixed rule "
      "applied to the turn's solved game: top move, sampled, pure maximin, and an epsilon-safe best response to the "
      "predicted opponent. Hedge mixes them with full-information feedback: each rule earns what it would have earned "
      "against the opponent's observed action. Weights carry across the games of a match, so Hedge competes with the "
      "best rule in hindsight.")

# 6. Refinements: the Bayesian opponent model
s, S = content(2, "Refinements: the Bayesian opponent model")
S.text(LEFT, TOP, WIDTH, BOTTOM - TOP, [
    P("A style posterior over five behavioural types", bullet=True),
    P("stage equilibrium, best response to uniform, logit best response to our mixture, cautious (maximin), uniform",
      size=SECONDARY, indent=True, before=2),
    P("Bayes = Hedge with ", ("$", "η = 1"), " on log loss, so its regret is at most ", ("$", r"\ln{5}"),
      bullet=True, before=16),
    P("The prediction ", ("$", r"\hat{q}"), " feeds the ε-safe best response: exploit it, keeping the stage value "
      "minus 0.3% of the payoff range", bullet=True, before=16),
    P("Per-world style likelihood: score each style on each sampled world's own legal actions, "
      "not on the world-averaged matrix", bullet=True, before=16),
    P("Forgetting ", ("$", "(λ < 1)"), " or fixed share tracks a switching opponent → the detection-delay result",
      bullet=True, before=16),
], "Opponent model bullets")
notes(s, "The opponent model is Bayesian. A posterior over five behavioural types predicts the opponent; Bayes is "
      "Hedge with step one on log loss, so its regret is at most log five. That prediction feeds the epsilon-safe best "
      "response, which exploits while keeping the stage value minus 0.3 percent of the range. The per-world likelihood "
      "scores styles on each world's legal actions. Forgetting or fixed share tracks a switching opponent.")

# 7. Case study: one turn of a real Showdown game
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
       [P("A real Showdown game: each turn the agent solves the Bayesian matrix game over 4 sampled opponent teams; "
          "bars show our mixed strategy, cells our estimated win chance.", size=CAPTION, i=True)],
       "Animation caption")
notes(s, "This is a real Showdown game against Foul Play; the names at the top are the bots' account names. Each "
      "turn the agent samples 4 possible opponent teams and solves one Bayesian matrix game over them. The panel shows "
      "a slice: rows are our options, columns the opponent's, cells our estimated win chance. The bars are our mixed "
      "strategy, and the outlined cell is what we and Foul Play actually played.")

# ===================== Section 3: Theory =====================
# 8. The no-regret results we build on, and how we use them
s, S = content(3, "No-regret results we build on")
S.text(LEFT, TOP, WIDTH, BOTTOM - TOP, [
    P(("MWU", {"b": True}), ("   Littlestone & Warmuth 1994", {"size": SECONDARY})),
    EQ_(E_MWU_BOUND, before=2),
    P(("Hedge", {"b": True}), ("   Freund & Schapire 1997", {"size": SECONDARY}), before=12),
    EQ_(E_HEDGE_BOUND, before=2),
    P(("No-regret dynamics ⇒ minimax", {"b": True}), ("   Freund & Schapire 1999", {"size": SECONDARY}),
      before=12),
    EQ_(E_MINIMAX, before=2),
    P("How we use them", b=True, before=20),
    P("Inside a turn: no-regret self-play (RM⁺) gives an approximate minimax strategy of each turn's matrix",
      size=SECONDARY, bullet=True, before=4),
    P("Over a battle: summing the per-turn error over turns gives the whole-battle theorem (next slide)",
      size=SECONDARY, bullet=True, before=4),
    P("Across games: Hedge over per-turn experts (anytime step), so its average payoff approaches the best expert's",
      size=SECONDARY, bullet=True, before=4),
    P("Opponent memory: Bayes over 5 styles is Hedge with η = 1 → at most ln 5 extra log loss per match; "
      "after a style switch the delay grows with how long the old style was played (tested live)",
      size=SECONDARY, bullet=True, before=4),
], "Base results and uses")
notes(s, "Three standard no-regret results carry our theory. Multiplicative weights loses at most a factor one plus "
      "eta of the best expert, plus log n over eta. Hedge's regret grows like the square root of T log n. And if both "
      "players of a zero-sum game are no-regret, their averages are an approximate minimax equilibrium, off by the sum "
      "of their average regrets. We use them four ways: RM-plus self-play inside a turn; summing per-turn errors into "
      "the whole-battle theorem; Hedge over experts across games; and the opponent memory, which is Hedge with step "
      "one, so it loses at most log five per match but notices a style switch late.")

# 9. The whole-battle bound
s, S = content(3, "The whole-battle bound")
S.text(LEFT, TOP, WIDTH, BOTTOM - TOP, [
    P("If every turn we play an approximate minimax strategy of an estimated stage matrix, our exploitability "
      "over the battle is at most the expected length times the per-turn error:"),
    EQ_(E_BOUND, before=6),
    P("With no-regret self-play each turn:", before=12),
    EQ_(E_COR, before=4),
    P(("$", "ε_s"), "  solver error in the estimated matrix: shrinks with compute", bullet=True, before=18),
    P(("$", "ε_v"), "  stage-matrix error from the heuristic leaf evaluation: does not", bullet=True, before=SP),
    P(("$", r"\bar{H}"), "  expected battle length against a best response (tight)", bullet=True, before=SP),
    P("Hidden teams: the same bound plus sampling, belief and non-locality terms", size=SECONDARY, i=True,
      before=18),
    P("Builds on: Freund & Schapire; Zinkevich et al. (CFR); Cesa-Bianchi & Lugosi; PokaiTrainer",
      size=SECONDARY, before=4),
], "Bound and terms")
notes(s, "Our main theorem: if every turn we play an approximate minimax strategy of an estimated stage matrix, "
      "exploitability over the battle is at most H-bar, the expected battle length against a best response, times "
      "the per-turn error, epsilon-s plus two epsilon-v. Epsilon-s is the solver error; with no-regret self-play it is "
      "at most the two average regrets, so it shrinks with compute. Epsilon-v, the stage-matrix error from the "
      "heuristic leaf evaluation, does not. With hidden teams the bound gains sampling, belief and non-locality terms.")

# ===================== Section 4: Experiments =====================
# 10. Exploitability and regret
s, S = content(4, "Exploitability and regret")
fw = WIDTH
fh = int(fw * 416 / 1395)  # 1395 x 416 px figure
pic = s.shapes.add_picture(FIG_MAIN, LEFT, TOP, fw, fh)
pic.name = "Exploitability and regret figure"
pic._element.nvPicPr.cNvPr.set(
    "descr", "Four panels: exact exploitability against the whole-battle bound; RM+ duality gap over iterations; "
             "Hedge regret per scored turn over 10 matches; memory regret against the bound ln 5")
by = TOP + fh + Inches(0.1)
S.text(LEFT, by, WIDTH, BOTTOM - by, [
    P("Exploitability lower bounds (informed best response): ", ("0.250", {"b": True}), " top move, ",
      ("0.090", {"b": True}), " sampled play", size=SECONDARY, bullet=True),
    P("Exact endgames: the whole-battle bound holds on all 22 rows, loose by ", ("6.9–155×", {"b": True}),
      "; leaf error ", ("$", "ε_v"), " dominates", size=SECONDARY, bullet=True, before=6),
    P("RM⁺ gap ", ("7.0×10⁻⁵", {"b": True}), " of the payoff range at T = 1000 (bound 0.38), slope ≈ −1.26",
      size=SECONDARY, bullet=True, before=6),
    P("Hedge regret ", ("0.13–0.16", {"b": True}), " of the anytime bound (", ("0.0045", {"b": True}),
      " per scored turn); memory regret ends at exactly ln 5", size=SECONDARY, bullet=True, before=6),
], "Exploitability and regret bullets")
notes(s, "Exploitability first. An informed best response that knows our policy gives 95 percent lower bounds of "
      "0.250 for top move and 0.090 for sampled play, so randomising helps. On exact endgames the whole-battle bound "
      "holds on all 22 rows, loose by 6.9 to 155 times, and the leaf error carries it. Regret next: RM-plus reaches a "
      "gap of 7 times ten to the minus five at a thousand iterations, far under its bound. Hedge sits at 0.13 to 0.16 of "
      "its anytime bound, and the memory's regret ends at exactly log five.")

# 11. Opponent memory
s, S = content(4, "Opponent memory")
S.text(LEFT, TOP, WIDTH, BOTTOM - TOP, [
    P("Better prediction: the per-world style likelihood lowers log loss by ≈", ("0.3", {"b": True}),
      " nats per scored turn, replicated", bullet=True),
    P("Detection delay matches its prediction per match (ρ = ", ("0.86", {"b": True}),
      "); no pre-registered prediction refuted", bullet=True, before=18),
    P("Doubling the pre-switch phase multiplies the delay by ", ("2.09", {"b": True}), " (predicted 1.5–2.7)",
      size=SECONDARY, indent=True, before=4),
    P("No win-rate gain: the style posterior is level with Hedge, and the per-world gain did not replicate",
      bullet=True, before=18),
    P("Better prediction shrinks only the belief term of the hidden-teams bound, not the leaf error",
      size=SECONDARY, i=True, before=36),
], "Opponent memory bullets")
notes(s, "The opponent memory does what the theory says. Scoring styles on each sampled world's legal actions lowers "
      "next-action log loss by about 0.3 nats per scored turn, and a pre-registered replication confirmed it. In the "
      "live detection-delay test, each match's delay matched its own prediction, rho 0.86, and doubling the pre-switch "
      "phase multiplied the delay by 2.09, inside the predicted range; no prediction was refuted. But better prediction "
      "did not translate into wins: it shrinks only the belief term, not the leaf error.")

# 12. Win rates
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
    P("MCTS beats equilibrium play at matched time: ", ("0.760", {"b": True}), " vs sampled, ",
      ("0.650", {"b": True}), " vs top move", bullet=True),
    P("The Bayesian LP beats PIMC averaging with sampled play: ", ("0.559", {"b": True}), " on Showdown",
      bullet=True, before=18),
    P("It loses to Foul Play at every search time; Foul Play rates ", ("135–192", {"b": True}),
      " Elo above", bullet=True, before=18),
], "Win-rate bullets")
notes(s, "Do the guarantees, which hold only in the estimated matrix, translate into wins? Not against search. At "
      "matched time, MCTS beats equilibrium play, 0.760 against sampled and 0.650 against top move. Solving one "
      "Bayesian game over the sampled worlds does beat PIMC averaging with sampled play, 0.559 on Showdown. But the "
      "agent loses to Foul Play at every search time we tried, and in the round-robin tournament Foul Play rates 135 "
      "to 192 Elo above it.")

# ===================== Section 5: Conclusions =====================
# 13. Takeaways and next steps
s, S = content(5, "Takeaways and next steps")
items = ["Per-turn equilibrium play and Hedge across games compose into a whole-battle bound, and the measured "
         "regrets sit well under their bounds",
         "The guarantees hold in the estimated matrix, whose leaf evaluation error does not shrink with compute "
         "and plausibly dominates; better beliefs do not move win rate",
         "Next: train the evaluation on exact terminal values"]
S.text(LEFT, TOP, WIDTH, BOTTOM - TOP, [
    P(txt, number=True, before=0 if k == 0 else 22) for k, txt in enumerate(items)
] + [P("Also in the paper: the hidden-teams bound, carried vs reset Hedge weights, joint set×style and "
       "lookahead-posterior nulls, a case study", size=SECONDARY, i=True, before=36)], "Takeaways")
notes(s, "Three takeaways. First, per-turn equilibrium play and Hedge across games compose into a whole-battle "
      "bound, and the regrets we measured sit well under their bounds. Second, those guarantees hold in the estimated "
      "matrix, whose leaf evaluation error does not shrink with compute and plausibly dominates, which is consistent "
      "with better beliefs improving prediction but not win rate. Third, the next step is to train the evaluation on "
      "exact terminal values. Thank you.")

add_sections(prs, groups)
prs.core_properties.title = "No-Regret Learning and Bayesian Opponent Memory in Pokémon Random Battles"
prs.core_properties.author = "[Names]"
prs.save(OUT)
print("wrote", OUT)
