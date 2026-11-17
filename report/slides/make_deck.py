"""Build deck.pptx: the 11-slide, ~4-minute CS535 final-project talk, in five sections.

Built on the user's template ("random template.pptx"): its master, theme, 16:9 size,
title-slide style and its five-box section tracker, which lives in the template's
"1_Title and Content" ... "5_Title and Content" layouts. Only the tracker labels change.
Content is plain text (Helvetica Neue, 20 pt, secondary 16 pt) with equations as native
PowerPoint equations (OMML, Cambria Math, 22 pt), wrapped like the template's slide 6:
mc:AlternateContent with an a14:m choice and a plain Unicode text fallback. Two slides keep
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
UPRIGHT = {"Reg", "expl", "gap", "val"}  # multi-letter names set upright
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
E_REG = r"Reg_T = \max_e{\sum_t{r_{t,e}}} − \sum_t{p_t·r_t}"
E_EXPL = r"expl(x) = val(A) − \min_y{x^⊤Ay}"
E_GAP = r"gap(x, y) = \max_i{{(Ay)}_i} − \min_j{{(x^⊤A)}_j}"
E_MWU = r"w_e ← w_e(1 − ηℓ_e)"
E_HEDGE = r"w_e ← w_e e^{ηr_e}"
E_SOFTMAX = r"p_t(e) ∝ \exp{(η\sum_{j<t}{r_{j,e}})}"
E_FACT1 = r"ε = \frac{Reg_1 + Reg_2}{T}"
E_BLP = r"\max_x{\sum_k{β_k \min_j{{(M_k^⊤x)}_j}}}"
E_MIX = r"p_t = \sum_e{w_e q_e}"
E_GAP_ROOT = r"gap = \frac{Reg^1 + Reg^2}{N}"
E_GAP_N = r"gap(\bar{x}_N, \bar{y}_N) = \frac{Reg^1_N + Reg^2_N}{N}"
E_BOUND = r"expl(π) ≤ \bar{H} · \sup{(ε_s + 2ε_v)}"
E_HEDGE_BOUND = r"Reg_T ≤ \sqrt{T \ln{m_E} / 2}"
E_ANYTIME = r"Reg_T ≤ \frac{3}{2}\sqrt{T \ln{m_E} / 2} + \sqrt{\ln{m_E} / 8}"


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
            f'<p:spPr/><p:txBody><a:bodyPr/><a:lstStyle/><a:p><a:fld id="{{7E75810F-9510-A844-820C-E3462544137D}}" '
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
    P(("Regret", {"b": True}), "  the best expert in hindsight minus what we earned", before=22),
    EQ_(E_REG, before=4),
    P(("Exploitability", {"b": True}), "  loss against a best-responding opponent, relative to the game value",
      before=22),
    EQ_(E_EXPL, before=4),
    P(("Duality gap", {"b": True}), "  the exploitability of both players together; zero exactly at equilibrium",
      before=22),
    EQ_(E_GAP, before=4),
], "Definitions")
notes(s, "A battle is a zero-sum stochastic game with simultaneous moves, chance and hidden teams, so each turn "
      "we face a stage matrix. Three yardsticks. Regret: the best expert in hindsight minus what we earned. "
      "Exploitability: how much a best-responding opponent takes from us, relative to the game value. The duality gap "
      "is the exploitability of both players together; it is zero exactly at equilibrium.")

# 3. From lecture: MWU -> Hedge, and regret => equilibrium
s, S = content(1, "MWU → Hedge, and regret ⇒ equilibrium")
ch = Inches(1.75)
S.text(LEFT, TOP, COL2, ch, [
    P("MWU (lecture)", b=True),
    EQ_(E_MWU, before=8),
    P("linear in the loss", size=SECONDARY, align="ctr", before=6),
], "MWU column")
S.text(LEFT + COL2 + GAP, TOP, COL2, ch, [
    P("Hedge (ours)", b=True),
    EQ_(E_HEDGE, before=8),
    EQ_(E_SOFTMAX, before=4),
], "Hedge column")
S.text(LEFT, TOP + ch + Inches(0.1), WIDTH, BOTTOM - TOP - ch - Inches(0.1), [
    P("Up- and down-weighting coincide after normalising", bullet=True),
    P("Bayes = Hedge with ", ("$", "η = 1"), " on log loss", bullet=True, before=6),
    P(("Fact 1", {"b": True}), " (Freund & Schapire 1999): both players no-regret ⇒ time averages are an ",
      ("$", "ε"), "-equilibrium", before=24),
    EQ_(E_FACT1, before=6),
    P("Averaged joint play is an ε-CCE, which in zero-sum games is minimax; "
      "CFR: regret bounds the exploitability of the average strategy", size=SECONDARY, align="ctr", before=6),
], "Hedge and Fact 1")
notes(s, "Lecture gave us multiplicative weights, a linear update on losses. Our code uses Hedge, an exponential "
      "update on rewards, so the strategy is a softmax of cumulative reward; up- and down-weighting coincide after "
      "normalising. Bayesian updating is Hedge with step one on log loss. Fact 1: if both players have no regret, the "
      "time averages are an epsilon-equilibrium, and CFR uses regret to bound exploitability.")

# ===================== Section 2: Method =====================
# 4. Where no-regret learning enters
s, S = content(2, "Applying No-Regret Learning")
cols = [
    ("Inside a turn", [
        P("Solve the Bayesian matrix game: one LP over the sampled worlds, or RM⁺ self-play", before=8),
        EQ_(E_BLP, before=10),
        P("PIMC averaging instead solves each world as if known, then averages → strategy fusion",
          size=SECONDARY, before=10)]),
    ("Across games", [
        P("Hedge over per-turn experts", before=8),
        P("plus a Bayesian opponent memory", before=8)]),
    ("At the search root", [
        P("Bandit self-play (UCB1 / Exp3): the visit shares obey", before=8),
        EQ_(E_GAP_ROOT, before=10)]),
]
for k, (head, paras) in enumerate(cols):
    S.text(LEFT + k * (COL3 + GAP), TOP, COL3, BOTTOM - TOP, [P(head, b=True)] + paras, f"{head} column")
notes(s, "We apply no-regret learning in three places. Inside a turn: we solve PokaiTrainer's Bayesian matrix game, "
      "one LP over the sampled worlds, or run RM-plus self-play: one strategy x against an opponent that "
      "best-responds in each sampled world, unlike PIMC averaging, which suffers strategy fusion. Across games: Hedge "
      "over per-turn experts plus a Bayesian opponent memory. At the search root: bandit self-play, where the visit "
      "shares' duality gap equals summed regret over N.")

# 5. Across games: an experts problem
s, S = content(2, "Across games: an experts problem")
S.text(LEFT, TOP, WIDTH, Inches(0.8),
       [P("We cast choosing how to play as the lecture's experts problem. The experts are fixed rules applied "
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
    P("Hedge competes with the best rule in hindsight: the lecture's fixed actions become fixed rules",
      bullet=True, before=6),
], "Hedge bullets")
notes(s, "Across games, we cast choosing how to play as the lecture's experts problem. Each expert is a fixed rule "
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

# ===================== Section 3: Experiments =====================
# 7. Engine experiments
s, S = content(3, "In the engine")
S.text(LEFT, TOP, WIDTH, BOTTOM - TOP, [
    P("MCTS at matched time beats equilibrium play: ", ("0.760", {"b": True}), " vs sampled, ",
      ("0.650", {"b": True}), " vs top move", bullet=True),
    P("95% lower confidence bounds on exploitability: ", ("0.250", {"b": True}), " for top move vs ",
      ("0.090", {"b": True}), " for sampled play", bullet=True, before=18),
    P("The Bayesian LP beats PIMC averaging: ", ("0.585", {"b": True}), " (sampled play, 400 games)",
      bullet=True, before=18),
    P("Exact endgames: the bound holds, loose by ", ("6.9–155×", {"b": True}),
      "; consistent with the leaf evaluation error dominating", bullet=True, before=18),
], "Engine results")
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
       [P("Round-robin tournament, 1,800 games; Elo-scale ratings, 95% CIs", size=CAPTION, i=True)],
       "Figure caption")
rx = LEFT + fw + GAP
S.text(rx, TOP, RIGHT - rx, BOTTOM - TOP, [
    P("Below Foul Play, even at its shortest search time", bullet=True),
    P("Level with the other one-turn solvers", bullet=True, before=SP),
    P("Hedge regret ", ("0.0044", {"b": True}), " per scored turn vs its best expert", bullet=True, before=SP),
    P("Detection delay matches its prediction per match (ρ = ", ("0.86", {"b": True}), "); no prediction refuted",
      bullet=True, before=SP),
    P("Better opponent prediction (≈", ("0.3", {"b": True}), " nats per scored turn), no detectable win-rate "
      "change (replicated)", bullet=True, before=SP),
], "Showdown bullets")
notes(s, "On real Showdown, a round-robin tournament of 1,800 games: Foul Play at every search time rates above "
      "our Bayesian LP, which is level with the other one-turn solvers. Hedge's regret against its best expert is 0.0044 "
      "per scored turn. The pre-registered detection-delay test refuted no prediction, rho 0.86. Better opponent "
      "prediction, about 0.3 nats per scored turn, gave no detectable win-rate change, and the null replicated.")

# ===================== Section 4: Theory =====================
# 9. Base bounds and how we adapt them
s, S = content(4, "Base bounds and where they come from")
bh = Inches(1.22)
S.text(LEFT, TOP, COL2, bh, [
    P("Hedge regret (Freund & Schapire 1997; Cesa-Bianchi & Lugosi 2006)", size=SECONDARY, b=True),
    EQ_(E_HEDGE_BOUND, before=6),
], "Base: Hedge")
S.text(LEFT + COL2 + GAP, TOP, COL2, bh, [
    P("No-regret self-play ⇒ ε-equilibrium (Freund & Schapire 1999; lecture)", size=SECONDARY, b=True),
    EQ_(E_FACT1, before=2),
], "Base: self-play")
TC = [Inches(3.0), Inches(4.6)]
TC.append(WIDTH - TC[0] - TC[1] - 2 * Inches(0.25))
TX = [LEFT, LEFT + TC[0] + Inches(0.25), LEFT + TC[0] + TC[1] + Inches(0.5)]
ty = TOP + bh + Inches(0.02)
for k, head in enumerate(["Base result (source)", "Adaptation", "Ours"]):
    S.text(TX[k], ty, TC[k], Inches(0.4), [P(head, size=SECONDARY, b=True)], f"Table header {k + 1}")
S.line(LEFT, ty + Inches(0.4), RIGHT, ty + Inches(0.4), "Table rule", w=12700)
rows = [
    (Inches(0.9),
     [P("Self-play bound, per turn (Freund & Schapire 1999)", size=SECONDARY)],
     [P("stage exploitability ", ("$", "ε_s"), ", composed over a battle with the stage-matrix error ",
        ("$", "ε_v"), " by backward induction (Shapley)", size=SECONDARY)],
     [P(("$", E_BOUND), size=SECONDARY), P("whole-battle bound; adapted", size=SECONDARY, i=True)]),
    (Inches(0.89),
     [P("Self-play bound (Freund & Schapire 1999)", size=SECONDARY)],
     [P("bandit feedback at the search root (cf. Lisý et al. 2013)", size=SECONDARY)],
     [P(("$", E_GAP_N), size=SECONDARY), P("an identity, not an inequality; ours", size=SECONDARY, i=True)]),
    (Inches(0.92),
     [P("Hedge bound (Freund & Schapire 1997; Cesa-Bianchi & Lugosi 2006)", size=SECONDARY)],
     [P("anytime step, slightly tighter constant (FTRL); Bayes = Hedge with ", ("$", "η = 1"),
        " (Cesa-Bianchi & Lugosi 2006)", size=SECONDARY)],
     [P(("$", E_ANYTIME), size=SECONDARY),
      P("FTRL constant (ext. report); ln 5: known", size=SECONDARY, i=True)]),
    (Inches(0.9),
     [P("Fixed-comparator regret (Cesa-Bianchi & Lugosi 2006)", size=SECONDARY)],
     [P("a switching opponent: tracking via fixed share (Herbster & Warmuth 1998; Cesa-Bianchi & Lugosi 2006)",
        size=SECONDARY)],
     [P("detection-delay prediction ", ("$", r"\hat{d}"), size=SECONDARY),
      P("adapted; tested live", size=SECONDARY, i=True)]),
]
ry = ty + Inches(0.45)
for k, (h, *cells) in enumerate(rows):
    for c, paras in enumerate(cells):
        S.text(TX[c], ry, TC[c], h, paras, f"Row {k + 1} column {c + 1}")
    ry += h
notes(s, "Our theory starts from two base results. Hedge's regret is at most root T log m over 2, from Freund and "
      "Schapire 1997 and Cesa-Bianchi and Lugosi; and no-regret self-play gives an epsilon-equilibrium with epsilon "
      "the summed regret over T, from Freund and Schapire 1999 and the lecture. We adapt them four ways. Applied per "
      "turn, the self-play bound gives the stage exploitability epsilon-s; composed over a battle with the stage-matrix "
      "error by Shapley's backward induction, it gives our whole-battle bound. With bandit feedback at the search root "
      "the inequality becomes an identity: the duality gap of the visit shares equals the summed regret over N; that "
      "identity is ours, building on Lisy and colleagues. Across games we use Hedge's anytime version, with a slightly "
      "tighter constant by the standard FTRL argument, and Bayes is Hedge with step one, so the memory's regret is at "
      "most log five; that is known. Against a switching opponent, tracking by fixed share, from Herbster and Warmuth, "
      "gives our detection-delay prediction, which we tested live.")

# 10. The whole-battle bound
s, S = content(4, "The whole-battle bound")
S.text(LEFT, TOP, WIDTH, Inches(3.0), [
    EQ_(E_BOUND),
    P("Shapley's backward induction over histories, one turn at a time; the core step is Lean-checked",
      size=SECONDARY, align="ctr", before=6),
    P(("$", "ε_s"), "  stage exploitability: shrinks with compute (regret ⇒ equilibrium; CFR)", before=22),
    P(("$", "ε_v"), "  stage-matrix error from the heuristic evaluation: does not", before=SP),
    P(("$", r"\bar{H}"), "  expected length against a best response (tight)", before=SP),
], "Bound and terms")
anchors = [("Freund & Schapire", [P("→ Fact 1 / ", ("$", "ε_s"), size=SECONDARY, align="ctr")]),
           ("Zinkevich et al. (CFR)", [P("→ exploitability metric; regret ⇒ exploitability", size=SECONDARY,
                                         align="ctr")]),
           ("Cesa-Bianchi & Lugosi", [P("→ Hedge, Bayes = Hedge, tracking", size=SECONDARY, align="ctr")]),
           ("PokaiTrainer", [P("→ the Bayesian matrix game", size=SECONDARY, align="ctr")])]
ay = TOP + Inches(3.15)
for k, (who, what) in enumerate(anchors):
    x = LEFT + k * (COL4 + GAP)
    S.header(x, ay, COL4, HDR_H, [P(who, size=SECONDARY, b=True, align="ctr")], f"Anchor {k + 1} header")
    S.box(x, ay + HDR_H, COL4, Inches(0.75), what, f"Anchor {k + 1} box")
S.text(LEFT, ay + HDR_H + Inches(0.85), WIDTH, Inches(0.4),
       [P("Also: a hidden-information version with sampling and belief terms", size=SECONDARY, i=True)],
       "Theory footer")
notes(s, "The headline theorem, adapted: exploitability is at most H-bar times the sup of epsilon-s plus two "
      "epsilon-v, by Shapley's backward induction over histories; the core step is checked in Lean. "
      "Epsilon-s, the stage exploitability, shrinks with compute, by Freund and Schapire's regret-to-equilibrium fact and "
      "CFR. Epsilon-v, the stage-matrix error from the heuristic evaluation, does not. H-bar, the length against a best "
      "response, is tight. Hedge and tracking come from Cesa-Bianchi and Lugosi, the Bayesian matrix game from PokaiTrainer.")

# ===================== Section 5: Conclusions =====================
# 11. Takeaways and next steps
s, S = content(5, "Takeaways and next steps")
items = ["No-regret learning composes cleanly from the turn to the search root; the measured regrets are small",
         "Bound and experiments are consistent with the stage-matrix (leaf) error dominating; "
         "better beliefs don't move win rate",
         "Next: an evaluation function trained on exact terminal values; log the opponent's side to measure "
         "two-sided regret more fully"]
S.text(LEFT, TOP, WIDTH, BOTTOM - TOP, [
    P(txt, number=True, before=0 if k == 0 else 22) for k, txt in enumerate(items)
] + [P("Also in the paper: joint set×style belief and lookahead-posterior nulls; case study",
       size=SECONDARY, i=True, before=36)], "Takeaways")
notes(s, "Three takeaways. First, no-regret learning composes cleanly across a turn, a match and a search root, "
      "and the measured regrets are small. Second, the bound and the experiments are consistent with the stage-matrix "
      "error from the leaf evaluation dominating: better beliefs improve prediction, not win rate. Third, next steps: an "
      "evaluation function trained on exact terminal values, and logging the opponent's side to measure two-sided regret. Thank you.")

add_sections(prs, groups)
prs.core_properties.title = "No-Regret Learning and Bayesian Opponent Memory in Pokémon Random Battles"
prs.core_properties.author = "[Names]"
prs.save(OUT)
print("wrote", OUT)
