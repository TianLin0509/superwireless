"""说明书回传桥：本地环回 HTTP + 收件箱。

**为什么要有这一层。** 说明书是离线 HTML。用 ``file://`` 打开时它既写不了文件、
也发不出请求，唯一的回传通道就是"复制 → 切窗口 → 粘回对话框"。多敲两下本身
不算负担，问题是它把一次调参拆成三个动作，中间任何一步走神就断了，
而且粘错、粘半截没有任何东西会拦。

改成从 ``127.0.0.1`` 上的环回服务打开**同一份 HTML**，页面就能直接 POST 回来：
用户点一下「应用到仿真」就结束，agent 这边 ``sw_await_config`` 正等着。
复制粘贴那条路原样保留，作为服务起不来时的兜底——**不是二选一，是主备**。

三条安全约定：

* 只绑 ``127.0.0.1``，不绑 ``0.0.0.0``。本机以外连不上。
* URL 里带一段每进程随机的 token，POST 也要带同一个。同机别的进程猜不到路径。
* 只接受白名单里的参数名，值必须是标量。页面是我们自己生成的，
  但接口一旦开着就得按"任何人都能戳"来写。

服务是**惰性启动**的：没人调 `serve()` 就不开端口。起不来（防火墙、沙箱、
端口耗尽）时返回 ``None`` 而不是抛异常——说明书本身还是能看，
只是退回复制粘贴。
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
import uuid
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from .paths import artifacts_root

# 单个 payload 的上限。页面只回几十个标量，1 MB 已经宽得离谱了。
_MAX_BODY = 1 << 20
_MAX_KEYS = 64


@dataclass
class Submission:
    """用户在页面上点「应用」之后回传的一次改动。"""

    spec_id: str
    overrides: dict[str, Any]
    title: str = ""
    text: str = ""
    at: float = field(default_factory=time.time)

    def as_dict(self) -> dict[str, Any]:
        return {
            "spec_id": self.spec_id,
            "overrides": self.overrides,
            "title": self.title,
            "text": self.text,
            "at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(self.at)),
        }


class _Page:
    __slots__ = ("html", "title", "allowed")

    def __init__(self, html: str, title: str, allowed: frozenset[str]) -> None:
        self.html = html
        self.title = title
        self.allowed = allowed


_lock = threading.Lock()
_server: ThreadingHTTPServer | None = None
_token: str = ""
_pages: dict[str, _Page] = {}
_inbox: list[Submission] = []
_seen: set[str] = set()          # 幂等键，防重发变成两份改动
_arrived = threading.Event()
_waiters = 0                     # 当前有几个 sw_await_config 正阻塞着等


def _dbg(msg: str) -> None:
    # stdio 传输下 stdout 是 JSON-RPC 通道，调试只能走 stderr。
    if os.environ.get("SUPERWIRELESS_DEBUG"):
        print(f"[superwireless.bridge] {msg}", file=sys.stderr, flush=True)


def inbox_dir():
    p = artifacts_root() / "inbox"
    p.mkdir(parents=True, exist_ok=True)
    return p


# ---------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------
class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "superwireless"
    sys_version = ""

    def log_message(self, fmt: str, *args: Any) -> None:  # noqa: A003
        _dbg("http " + (fmt % args))

    # -- 工具 ---------------------------------------------------------------
    def _send(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        # 页面只在本机自用，不给任何跨源可乘之机。
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(body)
        # 显式 flush：回执必须真的出去。收下了却让页面看到"失败"，
        # 用户会再发一遍，agent 就收到两份。
        self.wfile.flush()

    def _json(self, code: int, obj: dict[str, Any]) -> None:
        self._send(code, json.dumps(obj, ensure_ascii=False).encode("utf-8"),
                   "application/json; charset=utf-8")

    def _local_only(self) -> bool:
        host = self.client_address[0]
        if host in ("127.0.0.1", "::1"):
            return True
        self._json(403, {"ok": False, "error": "仅接受本机请求"})
        return False

    # -- 路由 ---------------------------------------------------------------
    def do_GET(self) -> None:  # noqa: N802
        if not self._local_only():
            return
        parts = [p for p in self.path.split("?")[0].split("/") if p]
        if len(parts) == 3 and parts[0] == "s" and parts[1] == _token:
            page = _pages.get(parts[2])
            if page is not None:
                self._send(200, page.html.encode("utf-8"), "text/html; charset=utf-8")
                return
        self._json(404, {"ok": False, "error": "not found"})

    def do_POST(self) -> None:  # noqa: N802
        if not self._local_only():
            return
        parts = [p for p in self.path.split("?")[0].split("/") if p]
        if len(parts) != 2 or parts[0] != "apply" or parts[1] != _token:
            self._json(404, {"ok": False, "error": "not found"})
            return
        try:
            n = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            n = 0
        if n <= 0 or n > _MAX_BODY:
            self._json(413, {"ok": False, "error": "payload 尺寸不合法"})
            return
        try:
            data = json.loads(self.rfile.read(n).decode("utf-8"))
        except Exception as exc:  # noqa: BLE001
            self._json(400, {"ok": False, "error": f"JSON 解析失败：{exc}"})
            return

        spec_id = str(data.get("id") or "")
        page = _pages.get(spec_id)
        if page is None:
            self._json(404, {"ok": False, "error": "未知说明书"})
            return
        ok, cleaned, why = _sanitize(data.get("overrides"), page.allowed)
        if not ok:
            self._json(400, {"ok": False, "error": why})
            return

        # **幂等键。** 回执可能在路上丢（服务正好退出、socket 被掐），页面因此会重发。
        # 没有 nonce 的话重发就是第二份改动，agent 那边看起来像用户点了两次。
        nonce = str(data.get("nonce") or "")[:64]
        with _lock:
            dup = bool(nonce) and nonce in _seen
            if nonce:
                _seen.add(nonce)
        if dup:
            self._json(200, {"ok": True, "dup": True, "n": len(cleaned),
                             "msg": "这次改动之前已经送到了"})
            return

        sub = Submission(spec_id=spec_id, overrides=cleaned, title=page.title,
                         text=str(data.get("text") or "")[:4000])
        # 先进内存收件箱（纯内存、不会失败），再回执，最后才落盘——
        # 回执之前不做任何可能阻塞的 I/O。
        with _lock:
            _inbox.append(sub)
            waiting = _waiters > 0
            _arrived.set()
        # **告诉用户改动落到哪一步了，别只说"已送达"。**
        # agent 正等着 vs 改动躺在收件箱里，这两件事对他完全不同：
        # 前者马上有回应，后者要等 agent 下次调工具才被看见。
        # 之前统一说"已送达 agent，回到对话框看结果"，结果 agent 没在等的时候
        # 对话框里什么都不会发生——用户以为没生效。
        self._json(200, {
            "ok": True, "n": len(cleaned), "waiting": waiting,
            "msg": ("agent 正在等，马上就会回应你"
                    if waiting else
                    "已收下（agent 当前在忙）。它下一次动作时就会看到并跟你确认。"),
        })
        _persist(sub)


def _sanitize(raw: Any, allowed: frozenset[str]) -> tuple[bool, dict[str, Any], str]:
    """只放行白名单里的键与标量值。

    页面是我们自己生成的，但**开着的接口要按任何人都能戳来写**——
    不做这一步，一个越界的键就能悄悄进入后面的 ``sw_revise``。
    """
    if not isinstance(raw, dict):
        return False, {}, "overrides 必须是对象"
    if len(raw) > _MAX_KEYS:
        return False, {}, f"改动项过多（{len(raw)} > {_MAX_KEYS}）"
    out: dict[str, Any] = {}
    for k, v in raw.items():
        key = str(k)
        if key not in allowed:
            return False, {}, f"不认识的参数 {key!r}"
        if isinstance(v, bool) or v is None:
            return False, {}, f"{key} 的值类型不支持"
        if isinstance(v, (int, float)):
            out[key] = v
        elif isinstance(v, str):
            if len(v) > 128:
                return False, {}, f"{key} 的值过长"
            out[key] = v
        else:
            return False, {}, f"{key} 的值必须是数字或字符串"
    return True, out, ""


def _persist(sub: Submission) -> None:
    """落一份盘：agent 当时没在等（在跑仿真、或会话已经翻页）时还能补捡。"""
    try:
        stamp = time.strftime("%Y%m%d-%H%M%S", time.localtime(sub.at))
        f = inbox_dir() / f"{stamp}-{uuid.uuid4().hex[:6]}.json"
        f.write_text(json.dumps(sub.as_dict(), ensure_ascii=False, indent=2),
                     encoding="utf-8")
    except Exception as exc:  # noqa: BLE001
        _dbg(f"落盘失败（不影响回传）：{exc}")


# ---------------------------------------------------------------------------
# 生命周期
# ---------------------------------------------------------------------------
def enabled() -> bool:
    return os.environ.get("SUPERWIRELESS_NO_SERVE", "") not in ("1", "true", "TRUE")


def start() -> str | None:
    """惰性启起环回服务，返回 ``http://127.0.0.1:PORT``；起不来返回 ``None``。"""
    global _server, _token
    if not enabled():
        return None
    with _lock:
        if _server is not None:
            return f"http://127.0.0.1:{_server.server_address[1]}"
        try:
            srv = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
        except OSError as exc:  # noqa: BLE001
            _dbg(f"起不来，退回 file://：{exc}")
            return None
        srv.daemon_threads = True
        _token = uuid.uuid4().hex
        # daemon=True：MCP 进程要退就退，别被这个线程吊住。
        threading.Thread(target=srv.serve_forever, name="sw-bridge",
                         daemon=True).start()
        _server = srv
        _dbg(f"listening on 127.0.0.1:{srv.server_address[1]}")
        return f"http://127.0.0.1:{srv.server_address[1]}"


def serve(spec_id: str, html: str, *, title: str,
          allowed: frozenset[str] | set[str]) -> str | None:
    """把一份说明书挂上去，返回可打开的 URL；服务起不来返回 ``None``。"""
    base = start()
    if base is None:
        return None
    _pages[spec_id] = _Page(html, title, frozenset(allowed))
    return f"{base}/s/{_token}/{spec_id}"


def allowed_for(spec_id: str) -> frozenset[str]:
    """某份说明书当前生效的白名单。测试用它核对没有另抄一份。"""
    page = _pages.get(spec_id)
    return page.allowed if page is not None else frozenset()


def apply_url() -> str | None:
    """页面 POST 回来的地址。没起服务时是 ``None``（页面据此退回复制粘贴）。"""
    if _server is None:
        return None
    return f"http://127.0.0.1:{_server.server_address[1]}/apply/{_token}"


# ---------------------------------------------------------------------------
# 收件箱
# ---------------------------------------------------------------------------
def drain(spec_id: str | None = None) -> list[Submission]:
    """取走已到达的改动。取走即清空，同一条不会被读两次。"""
    with _lock:
        if spec_id is None:
            got, _inbox[:] = list(_inbox), []
        else:
            got = [s for s in _inbox if s.spec_id == spec_id]
            _inbox[:] = [s for s in _inbox if s.spec_id != spec_id]
        if not _inbox:
            _arrived.clear()
    return got


def pending_count() -> int:
    """收件箱里还没被取走的改动数。**每个 MCP 工具的返回值都会带上它**——
    MCP 没有推送通道，这是让用户的点击"被看见"的唯一办法。"""
    return len(_inbox)


def await_submission(timeout_s: float = 90.0,
                     spec_id: str | None = None) -> list[Submission]:
    """等用户在页面上点「应用到仿真」。超时返回空列表——**超时不是错误**。

    先看一眼收件箱：用户可能在 agent 还没开始等的时候就点了。

    等待期间登记 ``_waiters``，页面据此告诉用户"agent 正在等"还是
    "已收下、它下次动作时会看到"——两种情况对用户的意义完全不同。
    """
    global _waiters
    got = drain(spec_id)
    if got:
        return got
    deadline = time.time() + max(1.0, float(timeout_s))
    with _lock:
        _waiters += 1
    try:
        while time.time() < deadline:
            if _arrived.wait(timeout=min(1.0, max(0.05, deadline - time.time()))):
                got = drain(spec_id)
                if got:
                    return got
        return []
    finally:
        with _lock:
            _waiters = max(0, _waiters - 1)


def open_url(target: str) -> bool:
    """用系统默认浏览器打开。失败返回 False，绝不抛。

    Windows 上直接走 ``os.startfile``——``webbrowser`` 在没有 DISPLAY /
    被沙箱限制的环境里会去试一串命令行浏览器，其中有些会往 **stdout** 写东西，
    而 stdio 传输下 stdout 是 JSON-RPC 通道，一个字节的杂音就能让会话崩掉。
    """
    if os.environ.get("SUPERWIRELESS_NO_BROWSER", "") in ("1", "true", "TRUE"):
        return False
    try:
        if sys.platform == "win32":
            os.startfile(target)  # noqa: S606
            return True
        cmd = ["open"] if sys.platform == "darwin" else ["xdg-open"]
        subprocess.Popen([*cmd, target],  # noqa: S603
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except Exception as exc:  # noqa: BLE001
        _dbg(f"打不开浏览器：{exc}")
        return False


def status() -> dict[str, Any]:
    return {
        "enabled": enabled(),
        "running": _server is not None,
        "base_url": (f"http://127.0.0.1:{_server.server_address[1]}"
                     if _server is not None else None),
        "pages": len(_pages),
        "pending": len(_inbox),
    }
