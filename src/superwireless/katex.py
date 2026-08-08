"""内联 KaTeX：公式的排版增强层，**MathML 仍是兜底**。

## 为什么是两层而不是二选一

用户 2026-08-03 批准了内联 KaTeX（"内联如果只有 1MB，感觉完全可接受"），
核心诉求是"公式显示美观、直观"。但把 MathML 整个换掉是有代价的：
KaTeX 靠 JS 在页面加载后渲染，只要脚本没跑起来（资产缺失、被 CSP 拦、
浏览器禁用 JS、页面被别的工具重新打包），公式就变成一堆裸 LaTeX——
**而那恰恰是最需要看懂它的场合**。

所以现在的结构是：

    <span class="kx" data-tex="\\frac{S}{I+N}"><math>…MathML…</math></span>
                     └─ KaTeX 的输入          └─ 没有 JS 时看到的东西

页面末尾一小段脚本把每个 ``.kx`` 的内容替换成 KaTeX 渲染结果。
脚本没跑 = 看到 MathML，照样是排好版的公式，只是不如 KaTeX 精致。
**降级路径上没有任何一步会露出源码。**

## 体积

CSS 359 KB（20 个 woff2 字体全部 base64 内联）+ JS 269 KB = **628 KB**，
比批准的 1 MB 预算低 37%。字体一个不删的理由见 ``scripts/vendor_katex.py``。

资产由 ``python scripts/vendor_katex.py`` 生成，不手改。
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

__all__ = ["available", "meta", "head_assets", "upgrade_script", "wrap"]

_ASSETS = Path(__file__).resolve().parent / "assets"


@lru_cache(maxsize=1)
def _read() -> tuple[str, str, dict[str, Any]]:
    try:
        css = (_ASSETS / "katex.css").read_text(encoding="utf-8")
        js = (_ASSETS / "katex.js").read_text(encoding="utf-8")
        meta_ = json.loads((_ASSETS / "katex.json").read_text(encoding="utf-8"))
        return css, js, meta_
    except Exception:  # noqa: BLE001
        return "", "", {}


def available() -> bool:
    """资产在不在。不在时说明书照常出，公式退回 MathML。"""
    css, js, _ = _read()
    return bool(css and js)


def meta() -> dict[str, Any]:
    return dict(_read()[2])


def head_assets() -> str:
    """放进 ``<head>`` 的内联样式与脚本。资产缺失时返回空串。"""
    css, js, _ = _read()
    if not (css and js):
        return ""
    # **不能用 f-string 或任何会碰到内容的字符串操作**——CSS 里有大量
    # base64 与花括号，JS 里有 ``</`` 之外的各种转义。原样拼接就好。
    return "<style>" + css + "</style>\n<script>" + js + "</script>"


def upgrade_script() -> str:
    """把页面里的 ``.kx`` 就地升级成 KaTeX 渲染结果。

    **失败必须静默且无损**：任何一条公式渲染不出来就保留它原来的 MathML，
    不清空、不抛错、不打断后面的公式。一个写错的 LaTeX 不该让整页公式消失。
    """
    if not available():
        return ""
    return (
        "<script>(function(){"
        "if(typeof katex==='undefined')return;"
        "var n=document.querySelectorAll('.kx');"
        "for(var i=0;i<n.length;i++){"
        "var e=n[i],t=e.getAttribute('data-tex');"
        "if(!t)continue;"
        "try{"
        "var d=e.getAttribute('data-display')==='1';"
        "var h=katex.renderToString(t,{displayMode:d,throwOnError:false,"
        "output:'html',strict:false});"
        "e.innerHTML=h;e.classList.add('kx-ok');"
        "}catch(err){/* 保留 MathML 兜底，不动它 */}"
        "}})();</script>"
    )


def wrap(tex: str, fallback_html: str, *, display: bool = False) -> str:
    """包一层可被 KaTeX 升级的容器，内容是 MathML 兜底。

    ``tex`` 原样进 ``data-tex``，所以调用方不要预先做 HTML 转义之外的处理。
    """
    esc = (str(tex).replace("&", "&amp;").replace("<", "&lt;")
           .replace(">", "&gt;").replace('"', "&quot;"))
    d = ' data-display="1"' if display else ""
    return f'<span class="kx"{d} data-tex="{esc}">{fallback_html}</span>'
