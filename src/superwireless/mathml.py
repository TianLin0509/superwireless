"""公式渲染：LaTeX 子集 → MathML。

**为什么不用 KaTeX。** KaTeX 要么引 CDN（离线打不开、也把页面绑到外网），
要么把 ~1 MB 的 JS + CSS + 字体内联进每一份说明书。而 MathML 是
**浏览器原生支持**的——Chrome 109+、Safari、Firefox 全都直接渲染，
零依赖、零网络、离线双击照样好看。

支持的语法是够写通信公式的那个子集：

    上下标      x_i  x^2  x_i^2  x_{ij}
    分数        \\frac{a}{b}
    根号        \\sqrt{x}
    希腊字母    \\alpha \\sigma \\eta ...
    运算符      \\cdot \\times \\le \\ge \\approx \\to \\sum \\prod
    括号        () [] 自动放大用 \\left( \\right)
    文字        \\text{...}
    矩阵转置    ^H  ^T（当成普通上标）

**不支持的直接原样输出**，不会崩、也不会渲染成乱码——
文档里的公式是人写的，写错了应该看得出来而不是被悄悄吃掉。
"""
from __future__ import annotations

import html
import re

# 希腊字母与常用符号
_SYM = {
    "alpha": "α", "beta": "β", "gamma": "γ", "delta": "δ", "epsilon": "ε",
    "zeta": "ζ", "eta": "η", "theta": "θ", "kappa": "κ", "lambda": "λ",
    "mu": "μ", "nu": "ν", "xi": "ξ", "pi": "π", "rho": "ρ", "sigma": "σ",
    "tau": "τ", "phi": "φ", "chi": "χ", "psi": "ψ", "omega": "ω",
    "Gamma": "Γ", "Delta": "Δ", "Theta": "Θ", "Lambda": "Λ", "Xi": "Ξ",
    "Pi": "Π", "Sigma": "Σ", "Phi": "Φ", "Psi": "Ψ", "Omega": "Ω",
    "cdot": "⋅", "times": "×", "div": "÷", "pm": "±", "mp": "∓",
    "le": "≤", "ge": "≥", "ne": "≠", "approx": "≈", "equiv": "≡",
    "to": "→", "rightarrow": "→", "leftarrow": "←", "Rightarrow": "⇒",
    "in": "∈", "notin": "∉", "subset": "⊂", "cap": "∩", "cup": "∪",
    "sum": "∑", "prod": "∏", "int": "∫", "infty": "∞", "partial": "∂",
    "nabla": "∇", "forall": "∀", "exists": "∃", "propto": "∝",
    "ll": "≪", "gg": "≫", "sim": "∼", "star": "⋆", "dagger": "†",
    "log": "log", "exp": "exp", "min": "min", "max": "max",
    "argmax": "argmax", "argmin": "argmin", "mathrm": "", "mathbb": "",
}

_OPS = set("+-=<>±×÷≤≥≠≈≡→←⇒∈∉⊂∩∪∝∼≪≫/")


def _tok(src: str) -> list[str]:
    """切成记号。反斜杠命令、花括号组、单字符各成一个记号。"""
    out: list[str] = []
    i, n = 0, len(src)
    while i < n:
        c = src[i]
        if c == "\\":
            m = re.match(r"\\([A-Za-z]+)", src[i:])
            if m:
                out.append("\\" + m.group(1))
                i += m.end()
                continue
            out.append(src[i:i + 2])
            i += 2
        elif c == "{":
            depth, j = 1, i + 1
            while j < n and depth:
                if src[j] == "{":
                    depth += 1
                elif src[j] == "}":
                    depth -= 1
                j += 1
            out.append(src[i:j])          # 连花括号一起，交给递归
            i = j
        elif c.isdigit():
            m = re.match(r"\d+(\.\d+)?", src[i:])
            out.append(m.group(0))
            i += m.end()
        elif c.isalpha():
            out.append(c)
            i += 1
        elif c.isspace():
            i += 1
        else:
            out.append(c)
            i += 1
    return out


def _grp(t: str) -> str:
    """``{...}`` 去壳后递归；否则当单个记号。"""
    return t[1:-1] if t.startswith("{") and t.endswith("}") else t


def _atom(t: str) -> str:
    if t.startswith("{"):
        return f"<mrow>{_render(_grp(t))}</mrow>"
    if t.startswith("\\"):
        name = t[1:]
        if name == "text":
            return ""                      # 由 _render 处理（要吃掉后面的组）
        sym = _SYM.get(name)
        if sym is None:
            return f"<mi>{html.escape(name)}</mi>"
        return (f"<mo>{sym}</mo>" if sym in _OPS or name in
                ("sum", "prod", "int", "cdot", "times", "to", "le", "ge",
                 "approx", "ne", "pm", "propto", "in", "equiv")
                else f"<mi>{sym}</mi>")
    if re.fullmatch(r"\d+(\.\d+)?", t):
        return f"<mn>{t}</mn>"
    if t.isalpha():
        return f"<mi>{t}</mi>"
    if t in ("(", ")", "[", "]", "|"):
        return f"<mo stretchy=\"true\">{html.escape(t)}</mo>"
    return f"<mo>{html.escape(t)}</mo>"


def _render(src: str) -> str:
    """记号流 → MathML。

    **关键是把「基」和「上下标」分成两步。** 早先 frac / text / right]
    这类多记号构造直接 append 就走了，后面的 ``_k`` / ``^{-1}`` 接不上去——
    渲染出来下标掉到下一行、``^{-1}`` 变成字面的 ^ -1。
    现在统一先算出一个 base，再看后面有没有 ``_`` / ``^``。

    **命令名必须写成 raw 字符串。** ``"\\frac"`` 里 ``\\f`` 是换页符，
    比较永远不成立，于是所有 frac 都掉进 else 分支被当成普通标识符。
    """
    toks = _tok(src)
    out: list[str] = []
    i = 0
    while i < len(toks):
        t = toks[i]
        base: str

        if t == r"\frac" and i + 2 < len(toks):
            base = (f"<mfrac><mrow>{_render(_grp(toks[i + 1]))}</mrow>"
                    f"<mrow>{_render(_grp(toks[i + 2]))}</mrow></mfrac>")
            i += 3
        elif t == r"\sqrt" and i + 1 < len(toks):
            base = f"<msqrt>{_render(_grp(toks[i + 1]))}</msqrt>"
            i += 2
        elif t in (r"\text", r"\mathrm", r"\mathbb") and i + 1 < len(toks):
            base = f"<mtext>{html.escape(_grp(toks[i + 1]))}</mtext>"
            i += 2
        elif t in (r"\left", r"\right") and i + 1 < len(toks):
            d = toks[i + 1]
            base = ("<mo>&#x2061;</mo>" if d == "."
                    else f'<mo stretchy="true">{html.escape(d)}</mo>')
            i += 2
        elif t == r"\quad":
            out.append('<mspace width="1em"/>')
            i += 1
            continue
        elif t in (r"\begin", r"\end"):
            i += 2 if i + 1 < len(toks) else 1
            continue
        else:
            base = _atom(t)
            i += 1

        # 下标与上标可以连着来，顺序任意：x_i^2 / x^2_i
        sub = sup = None
        while i < len(toks) and toks[i] in ("_", "^") and i + 1 < len(toks):
            which, arg = toks[i], toks[i + 1]
            rendered = _render(_grp(arg))
            if which == "_":
                sub = rendered
            else:
                sup = rendered
            i += 2
        if sub is not None and sup is not None:
            out.append(f"<msubsup>{base}<mrow>{sub}</mrow><mrow>{sup}</mrow></msubsup>")
        elif sub is not None:
            out.append(f"<msub>{base}<mrow>{sub}</mrow></msub>")
        elif sup is not None:
            out.append(f"<msup>{base}<mrow>{sup}</mrow></msup>")
        else:
            out.append(base)
    return "".join(out)


def render(latex: str, *, block: bool = True) -> str:
    """LaTeX 子集 → MathML。渲染不了的原样退回带样式的纯文本。"""
    try:
        body = _render(latex.strip())
        if not body:
            raise ValueError("空公式")
        disp = ' display="block"' if block else ""
        # **整体必须包一层 <mrow>。** 不包的话浏览器把顶层的每个子元素
        # 当成可独立断行的单位，`SINR_k = <分数> - 1` 会被拆成三行，
        # 等号和减号各自孤零零挂着，看起来像三个公式。
        return (f'<math xmlns="http://www.w3.org/1998/Math/MathML"{disp}>'
                f"<mrow>{body}</mrow></math>")
    except Exception:  # noqa: BLE001
        # **渲染不了就原样显示**，不吞掉也不糊弄——公式是人写的，
        # 写错了应该看得出来。
        return f'<code class="fml-raw">{html.escape(latex)}</code>'


def css() -> str:
    """配套样式。MathML 由浏览器原生渲染，这里只调字号与间距。"""
    return (
        "math{font-size:1.06em;font-family:'Cambria Math','STIX Two Math',"
        "'Latin Modern Math',serif}"
        "math[display='block']{display:block;margin:10px 0;text-align:left}"
        ".fml-raw{color:var(--tint-red-ink);background:var(--tint-red)}"
    )
