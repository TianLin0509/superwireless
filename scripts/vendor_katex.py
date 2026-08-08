"""把 KaTeX 打包成两个自包含的资产文件，供说明书内联。

**为什么要内联而不是引 CDN。** 说明书是要发给人、存进归档、在没有网络的
内网机器上打开的。引 CDN 意味着断网时公式变成一堆裸 LaTeX 源码——
而这恰恰是最需要看懂它的场合。用户 2026-08-03 明确批准了内联方案
（"katex 内联如果只有 1MB，感觉完全可接受"）。

实测体积：CSS 22.8 KB + JS 269 KB + 20 个 woff2 字体 253.7 KB，
字体转 base64 后膨胀到 338 KB，**合计约 630 KB**，比预算低 37%。

**字体一个都不删。** 少一个就可能有某个符号悄悄退回系统字体——
渲染不会报错，只是那一个字符变丑，而且只有肉眼盯着才看得出来。
630 KB 还在预算内，没必要冒这个险。

只保留 woff2：原 CSS 里每个 ``@font-face`` 同时列了 woff2 / woff / ttf 三种，
后两种是给十年前的浏览器兜底的，全留会让体积翻三倍。

用法::

    python scripts/vendor_katex.py [--version 0.16.11]

产出 ``src/superwireless/assets/katex.css`` 与 ``katex.js``。
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import re
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "src" / "superwireless" / "assets"


def fetch(base: str, path: str) -> bytes:
    with urllib.request.urlopen(base + path, timeout=60) as r:
        return r.read()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", default="0.16.11")
    args = ap.parse_args()
    base = f"https://cdn.jsdelivr.net/npm/katex@{args.version}/dist/"

    css = fetch(base, "katex.min.css").decode("utf-8")
    js = fetch(base, "katex.min.js").decode("utf-8")

    names = sorted(set(re.findall(r"fonts/(KaTeX_[A-Za-z0-9\-]+)\.woff2", css)))
    if not names:
        print("没在 CSS 里找到字体引用，KaTeX 的打包结构可能变了", file=sys.stderr)
        return 1
    blobs: dict[str, bytes] = {}
    for n in names:
        blobs[n] = fetch(base, f"fonts/{n}.woff2")

    # 每个 @font-face 的 src 整体换成单个 woff2 data URI，丢掉 woff/ttf 兜底
    def repl(m: re.Match[str]) -> str:
        body = m.group(0)
        fm = re.search(r"fonts/(KaTeX_[A-Za-z0-9\-]+)\.woff2", body)
        if not fm:
            return body
        b64 = base64.b64encode(blobs[fm.group(1)]).decode("ascii")
        uri = f"url(data:font/woff2;base64,{b64}) format('woff2')"
        return re.sub(r"src:[^;}]+", f"src:{uri}", body, count=1)

    css_inlined = re.sub(r"@font-face\s*\{[^}]*\}", repl, css)
    if "fonts/KaTeX" in css_inlined:
        print("还有没被替换掉的字体 URL，检查正则", file=sys.stderr)
        return 1

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "katex.css").write_text(css_inlined, encoding="utf-8")
    (OUT / "katex.js").write_text(js, encoding="utf-8")
    meta = {
        "version": args.version,
        "source": base,
        "fonts": len(names),
        "css_bytes": len(css_inlined.encode()),
        "js_bytes": len(js.encode()),
        "css_sha256": hashlib.sha256(css_inlined.encode()).hexdigest(),
        "js_sha256": hashlib.sha256(js.encode()).hexdigest(),
        "license": "MIT (KaTeX, https://github.com/KaTeX/KaTeX/blob/main/LICENSE)",
    }
    (OUT / "katex.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    total = (meta["css_bytes"] + meta["js_bytes"]) / 1024
    print(f"KaTeX {args.version}：CSS {meta['css_bytes'] / 1024:.0f} KB"
          f"（含 {len(names)} 个内联字体）+ JS {meta['js_bytes'] / 1024:.0f} KB"
          f" = {total:.0f} KB → {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
