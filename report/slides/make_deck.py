"""Build deck.pptx: the 12-slide, ~4-minute CS535 final-project talk, in five sections.

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
E_MINIMAX = r"\max_x{x^⊤A\bar{y}} − \min_y{\bar{x}^⊤Ay} ≤ ϵ_R + ϵ_C"
E_BOUND = r"expl(π) ≤ \bar{H}\sup{(ε_s + 2ε_v)} ≤ \bar{H}(ϵ_R + ϵ_C + 2ε_v)"
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

# 2. The game and the yardsticks
s, S = content(1, "The game and the yardsticks")
S.text(LEFT, TOP, WIDTH, BOTTOM - TOP, [
    P("A battle is a zero-sum stochastic game: simultaneous moves, chance, hidden teams → a stage matrix each turn"),
    P(("Regret", {"b": True}), "  our total loss minus the best expert's total loss in hindsight", before=20),
    EQ_(E_REG, before=4),
    P(("$", "n"), " experts, ", ("$", "T"), " rounds;  ", ("$", "p^t"), ": our probabilities over experts;  ",
      ("$", "l_i^t ∈ [0,1]"), ": expert ", ("$", "i"), "'s loss in round ", ("$", "t"), size=SECONDARY, align="ctr",
      before=4),
    P(("Exploitability", {"b": True}), "  the best worst-case payoff any strategy can secure, minus our strategy's "
      "worst case", before=20),
    EQ_(E_EXPL, before=4),
    P(("$", "A"), ": the turn's payoff matrix (our action × theirs);  ", ("$", "x, y"),
      ": mixed strategies;  first term = the game value; 0 exactly for an equilibrium strategy",
      size=SECONDARY, align="ctr", before=4),
], "Definitions")
notes(s, "A battle is a zero-sum stochastic game with simultaneous moves, chance and hidden teams, so each turn "
      "we face a payoff matrix A: rows are our actions, columns theirs, and x and y are mixed strategies. Regret, in "
      "the experts problem with n experts over T rounds: we put probabilities p^t on the experts, expert i suffers loss "
      "l_i^t, and regret is our total loss minus the best expert's in hindsight; no-regret means it grows slower than T. "
      "Exploitability: every strategy has a worst case, what it earns when the opponent best-responds. The best "
      "achievable worst case is the game value. Exploitability is how far our strategy's worst case falls below it, "
      "zero exactly for an equilibrium strategy.")

# 3. MWU -> Hedge, and regret => equilibrium
s, S = content(1, "MWU → Hedge, and regret ⇒ equilibrium")
ch = Inches(1.75)
S.text(LEFT, TOP, COL2, ch, [
    P("MWU", b=True),
    EQ_(E_MWU, before=8),
    P("linear in the loss", align="ctr", before=6),
    EQ_(E_MWUB, before=6),
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
    P("both players no-regret, with average regrets ", ("$", "ϵ_R"), " and ", ("$", "ϵ_C"), ":", before=4),
    EQ_(E_MINIMAX, before=6),
    P("and the time-average play is an ε-coarse correlated equilibrium", align="ctr", before=6),
], "Hedge and no-regret dynamics")
notes(s, "Multiplicative weights updates linearly in the loss. We use Hedge, the exponential form of multiplicative "
      "weights. With small steps it behaves like MWU, but it is the form the no-regret-in-games literature analyses, "
      "its anytime version suits matches of unknown length, and Bayesian updating is exactly Hedge with step one, so "
      "our opponent memory inherits the same no-regret guarantee. Inside a turn we use regret matching (RM-plus), as "
      "CFR and PokaiTrainer do. The key result: if both players are no-regret, the average strategies are an "
      "approximate minimax equilibrium, off by at most the sum of their average regrets, and time-average play is an "
      "approximate coarse correlated equilibrium.")

# ===================== Section 2: Method =====================
# 4. Each turn an equilibrium; across games Hedge over experts
s, S = content(2, "Equilibrium each turn, Hedge across games")
S.text(LEFT, TOP, WIDTH, Inches(1.55), [
    P(("Each turn: ", {"b": True}), "sample 4 possible opponent teams (worlds ", ("$", "k"), ", matrices ",
      ("$", "M_k"), ") and pick the one mixed strategy ", ("$", "x"),
      " that does best on average when the opponent best-responds in every world (as PokaiTrainer does for VGC)"),
    EQ_(E_BLP, before=6),
], "Each turn")
S.text(LEFT, TOP + Inches(1.6), WIDTH, Inches(0.45),
       [P(("Across games: ", {"b": True}), "Hedge over four ways of playing ", ("$", "x"),
          " on the averaged matrix ", ("$", r"\bar{A} = \sum_k{β_k M_k}"))], "Across games")
experts = [("Top move", [P(("$", r"a = arg\max_i{x_i}"), size=SECONDARY, align="ctr")]),
           ("Sampled", [P(("$", "a ∼ x"), size=SECONDARY, align="ctr")]),
           ("Pure maximin", [P(("$", r"a = arg\max_i{\min_j{\bar{A}_{ij}}}"), size=SECONDARY, align="ctr")]),
           ("ε-safe best response", [P("best reply to ", ("$", r"\hat{q}"), size=SECONDARY, align="ctr"),
                                     P("with worst case ≥ ", ("$", "x"), "'s − ", ("$", "ε"), size=SECONDARY,
                                       align="ctr")])]
gy = TOP + Inches(2.15)
for k, (name, rule) in enumerate(experts):
    x = LEFT + k * (COL4 + GAP)
    S.header(x, gy, COL4, HDR_H, [P(name, size=SECONDARY, b=True, align="ctr")], f"Expert {k + 1} header")
    S.box(x, gy + HDR_H, COL4, Inches(0.85), rule, f"Expert {k + 1} box")
by = gy + HDR_H + Inches(0.85) + Inches(0.2)
S.text(LEFT, by, WIDTH, BOTTOM - by, [
    P("Each expert is credited with what it would have earned against the observed action; weights carried "
      "across a match", bullet=True),
], "Hedge bullets")
notes(s, "Each turn, the agent samples four possible opponent teams. Each gives a payoff matrix M_k. It then picks one "
      "mixed strategy x that does best on average when the opponent best-responds separately in each world: the "
      "formula on the slide, a single linear program. PokaiTrainer solves the same problem for Pokemon VGC. Across "
      "games, Hedge mixes four ways of playing this solution on the averaged matrix: top move plays x's most likely "
      "action; sampled draws an action from x; pure maximin plays the single action with the best worst case; and "
      "the epsilon-safe best response best-replies to the memory's prediction of the opponent while keeping its worst "
      "case within epsilon, 0.3 percent of the payoff range, of x's. Each expert is credited with what it would have "
      "earned against the observed action, and the weights carry across the games of a match.")

# 5. The Bayesian opponent memory
s, S = content(2, "The Bayesian opponent memory")
steps = [("1. Observe", [P("the opponent plays ", ("$", "b_t"), size=SECONDARY, align="ctr")]),
         ("2. Score styles", [P(("$", r"q_θ(b_t | M_k)"), size=SECONDARY, align="ctr"),
                              P("each style ", ("$", "θ"), ", world ", ("$", "k"), size=SECONDARY, align="ctr", before=4),
                              P("(0 if illegal there)", size=SECONDARY, align="ctr")]),
         ("3. Update posterior", [P(("$", r"Π_t(θ) ∝ Π_{t−1}(θ)^λ ×"), size=SECONDARY, align="ctr"),
                                  P(("$", r"\sum_k{β_k q_θ(b_t | M_k)}"), size=SECONDARY, align="ctr", before=4)]),
         ("4. Predict, exploit", [P(("$", r"\hat{q} = \sum_θ{Π(θ) q_θ}"), size=SECONDARY, align="ctr"),
                                  P("→ ε-safe best response", size=SECONDARY, align="ctr", before=4)])]
BH = Inches(1.35)
for k, (name, paras) in enumerate(steps):
    x = LEFT + k * (COL4 + GAP)
    S.header(x, TOP, COL4, HDR_H, [P(name, size=SECONDARY, b=True, align="ctr")], f"Step {k + 1} header")
    S.box(x, TOP + HDR_H, COL4, BH, paras, f"Step {k + 1} box")
    if k < 3:
        ym = TOP + HDR_H + BH // 2
        S.line(x + COL4 + Inches(0.05), ym, x + COL4 + GAP - Inches(0.05), ym, f"Arrow {k + 1}", arrow=True)
y2 = TOP + HDR_H + BH + Inches(0.45)
H2 = Inches(2.0)
cols = [("Five styles", [P("stage equilibrium", size=SECONDARY, align="ctr"),
                         P("best response to uniform", size=SECONDARY, align="ctr"),
                         P("logit best response to our mix", size=SECONDARY, align="ctr"),
                         P("cautious (maximin)", size=SECONDARY, align="ctr"),
                         P("uniform", size=SECONDARY, align="ctr")]),
        ("How we check it", [P("Bayes = Hedge (", ("$", "η = 1"), "):", size=SECONDARY, align="ctr"),
                             P("regret ≤ ", ("$", r"\ln{5}"), ", measured ", ("$", r"\ln{5}"), size=SECONDARY,
                               align="ctr"),
                             P("Per-world scoring:", size=SECONDARY, align="ctr", before=8),
                             P("≈ 0.3 nats/turn better", size=SECONDARY, align="ctr"),
                             P("Switch delay = prediction (live)", size=SECONDARY, align="ctr", before=8)]),
        ("Extension: sets × style jointly", [P(("$", r"Π_t(k, θ) ∝ Π_{t−1}(k, θ) q_θ(b_t | M_k)"),
                                               size=SECONDARY, align="ctr"),
                                             P("which team and how they play, updated together", size=SECONDARY,
                                               align="ctr", before=6),
                                             P("adds nothing measurable (+0.001 nats/turn)", size=SECONDARY,
                                               align="ctr", before=6)])]
for k, (name, paras) in enumerate(cols):
    x = LEFT + k * (COL3 + GAP)
    S.header(x, y2, COL3, HDR_H, [P(name, size=SECONDARY, b=True, align="ctr")], f"Memory {k + 1} header")
    S.box(x, y2 + HDR_H, COL3, H2, paras, f"Memory {k + 1} box")
notes(s, "Here is the opponent memory step by step. The opponent plays an action. For each of five styles and each "
      "of our four sampled worlds, we ask how likely that action was under that style in that world, giving zero if "
      "the action is illegal there. We average over worlds and multiply into the posterior over styles, with "
      "forgetting lambda below one for a switching opponent. The posterior's prediction feeds the epsilon-safe best "
      "response. We check it three ways: Bayes is Hedge with step one, so its log-loss regret is at most log five, "
      "and we measure exactly log five; scoring styles world by world improves prediction by about 0.3 nats per turn; "
      "and the delay after a style switch matches its predicted value in a live test. The extension tracks the team "
      "and the style jointly, but it adds nothing measurable.")

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
# 7. The whole-battle bound, Hedge across games, RL exploiters
s, S = content(3, "The whole-battle bound")
S.text(LEFT, TOP, WIDTH, BOTTOM - TOP, [
    P(("Exploitability = sum of per-turn losses", {"b": True}), "   ", ("$", E_LOSSES)),
    P("each turn loses at most ", ("$", "ε_s + 2ε_v"), ", so", before=6),
    EQ_(E_BOUND, before=4),
    P(("$", "ε_s"), " solver error, shrinks with compute  ·  ", ("$", "ε_v"), " leaf error, does not  ·  ",
      ("$", r"\bar{H}"), " battle length", size=SECONDARY, align="ctr", before=6),
    P(("Hedge across games", {"b": True}), " (in our estimated matrices)", before=22),
    P("Exploits: the best expert's payoff minus its regret", bullet=True, before=4),
    P("Stays safe: within its regret of the equilibrium value", bullet=True, before=4),
    P("Carried weights: per-game overhead → 0 over a match", bullet=True, before=4),
    P(("RL best response = Hedge in every state", {"b": True}), " → RL exploiters lower-bound exploitability",
      before=22),
], "Bound, Hedge across games, RL")
notes(s, "Exploitability equals the expected sum of per-turn losses against a best response, the game analogue of the "
      "performance-difference lemma. Each turn loses at most epsilon-s plus two epsilon-v, so the battle loses at most "
      "H-bar times that, a game version of Singh and Yee's bound with H-bar as the horizon. Epsilon-s shrinks with "
      "compute; the leaf error epsilon-v does not. Hedge across games exploits, earning the best expert's payoff minus "
      "its regret, and stays safe, within its regret of the equilibrium value; with carried weights the per-game "
      "overhead vanishes over a match. These hold in our matrices; on Showdown only the first applies. And since "
      "natural policy gradient is Hedge in every state, an RL exploiter's score lower-bounds exploitability.")

# ===================== Section 4: Experiments =====================
# 8. Exploitability
s, S = content(4, "Exploitability and regret")
fw = WIDTH
fh = int(fw * 416 / 1395)  # 1395 x 416 px figure
pic = s.shapes.add_picture(FIG_MAIN, LEFT, TOP, fw, fh)
pic.name = "Exploitability and regret figure"
pic._element.nvPicPr.cNvPr.set(
    "descr", "Four panels: exact exploitability against the whole-battle bound; RM+ distance from equilibrium over iterations; "
             "Hedge regret per scored turn over 10 matches; memory regret against the bound ln 5")
by = TOP + fh + Inches(0.1)
S.text(LEFT, by, WIDTH, BOTTOM - by, [
    P("Exploitability lower bounds: ", ("0.25", {"b": True}), " top move, ", ("0.09", {"b": True}),
      " sampled play", size=18, bullet=True),
    P("Exact endgames: the bound holds, loose by ", ("7–155×", {"b": True}), "; leaf error ", ("$", "ε_v"),
      " dominates", size=18, bullet=True, before=6),
    P("Regrets (b–d) stay under their bounds", size=18, bullet=True, before=6),
], "Exploitability bullets")
notes(s, "An informed best response that knows our policy gives exploitability lower bounds of 0.25 for top move and "
      "0.09 for sampled play, so randomising helps. On exact endgames the whole-battle bound holds on every row, loose "
      "by 7 to 155 times, and the leaf error, not the solver, carries it. Panels b to d show the regrets under their "
      "bounds; the next slide gives the numbers.")

# 9. No-regret dynamics
s, S = content(4, "No-regret dynamics")
S.text(LEFT, TOP, WIDTH, BOTTOM - TOP, [
    P("Theory: if both players have average regrets ", ("$", "ϵ_R, ϵ_C"), ", their average strategies are within ",
      ("$", "ϵ_R + ϵ_C"), " of minimax"),
    EQ_(E_MINIMAX, before=4),
    P("Inside a turn: RM⁺ is within ", ("7×10⁻⁵", {"b": True}), " of the payoff range of minimax (bound 0.38)",
      bullet=True, before=20),
    P("Averages within the bound on all 120 matrices; the last iterate has a heavier tail",
      bullet=True, before=SP),
    P("Across games: Hedge regret ", ("≈ 0.004", {"b": True}), " per turn vs Foul Play (13–16% of the bound); "
      "carried beat reset weights by ", ("+0.15", {"b": True}), bullet=True, before=SP),
    P("Hedge vs the equilibrium agent: ", ("0.46", {"b": True}), " [0.42, 0.50], inside the range we predicted in advance",
      bullet=True, before=SP),
], "Dynamics bullets")
notes(s, "Theory says no-regret players' averages are within the sum of their regrets of minimax. Measured: RM-plus "
      "inside a turn reaches 7 times ten to the minus five of the payoff range, against a bound of 0.38. On 120 stage "
      "matrices, Hedge and optimistic Hedge averages stay within the bound on every one; the last iterate has a lower "
      "median, because 58 percent have a pure saddle, but a heavier tail. Across games, Hedge's regret against Foul "
      "Play is 13 to 16 percent of its bound, and carried weights beat reset weights, as the bounds order them. "
      "Against the equilibrium agent Hedge scores 0.46, inside the range we predicted in advance; theory bounds only the upper side.")

# 10. Opponent memory
s, S = content(4, "Opponent memory: results")
S.text(LEFT, TOP, WIDTH, BOTTOM - TOP, [
    P("Per-world likelihood: log loss ", ("≈ 0.3", {"b": True}), " nats per turn lower, replicated", bullet=True),
    P("Detection delay matches its prediction (median ratio ", ("1.06", {"b": True}), "); doubling the pre-switch phase "
      "→ ×2.1 (predicted 1.5–2.7)", bullet=True, before=18),
    P("Modelling sets and style jointly adds nothing measurable (+0.001 nats per turn)", bullet=True, before=18),
    P("No win-rate gain: better prediction shrinks only the belief term", bullet=True, before=18),
], "Opponent memory bullets")
notes(s, "The memory does what the theory says. Scoring styles on each sampled world's legal actions lowers log loss "
      "by about 0.3 nats per turn, confirmed in a replication on fresh seeds planned in advance. In the live detection-delay test each "
      "match's delay matched its prediction, and doubling the pre-switch phase multiplied the delay by 2.1, inside the "
      "predicted range. Modelling sets and style jointly adds nothing measurable. But better prediction did not move "
      "win rate: it shrinks only the belief term, not the leaf error.")

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

# ===================== Section 5: Conclusions =====================
# 12. Takeaways and next steps
s, S = content(5, "Takeaways and next steps")
items = ["Equilibrium play and Hedge compose into a whole-battle bound; measured regrets sit well under their "
         "bounds",
         "The guarantees hold in the estimated matrix, whose leaf error does not shrink with compute; better "
         "beliefs do not move win rate",
         "Next: train the evaluation on exact terminal values"]
S.text(LEFT, TOP, WIDTH, BOTTOM - TOP, [
    P(txt, number=True, before=0 if k == 0 else 22) for k, txt in enumerate(items)
], "Takeaways")
notes(s, "Three takeaways. First, per-turn equilibrium play and Hedge across games compose into a whole-battle bound, "
      "and the regrets we measured sit well under their bounds. Second, those guarantees hold in the "
      "estimated matrix, whose leaf evaluation error does not shrink with compute and plausibly dominates, consistent "
      "with better beliefs improving prediction but not win rate. Third, the next step is to train the evaluation on "
      "exact terminal values. Thank you.")

add_sections(prs, groups)
prs.core_properties.title = "No-Regret Learning and Bayesian Opponent Memory in Pokémon Random Battles"
prs.core_properties.author = "[Names]"
prs.save(OUT)
print("wrote", OUT)
