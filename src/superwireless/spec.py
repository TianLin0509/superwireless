"""仿真说明书：配置敲定之后，把这次到底在仿什么画出来给用户看。

**为什么要有这个。** 计划书（``plan.render_plan_markdown``）给的是文字和
一串 key: value，用户要在脑子里把"64T"还原成 8x4x2 的双极化面板、把
"7 站 3 扇区 ISD 500"还原成六边形栅格上 21 个扇区的指向、把"272 RB"还原成
17 个 RBG。**这一步在脑子里做，就一定有人做错。**

所以配置敲定后出一份说明书：参数表 + 示意图，一次看清楚

* **阵列**——RF 端口怎么排、每个端口驱动几个物理阵子、间距多少；
* **拓扑**——站点在哪、扇区朝哪、UE 撒在哪、站间距多大；
* **频域**——多少 RB、怎么分 RBG、子载波间隔与带宽；
* **时域**——TDD 时隙配比；
* **信道**——CDL 剖面的时延功率谱；
* **链路预算**——从发射功率到 SINR 的每一跳。

产物是**自包含的离线 HTML**（SVG 内联，无外部依赖），落到项目
``artifacts/specs/``。对话里只回路径和摘要，不把图往回贴。

--- 信息分级，不是信息堆叠 ---------------------------------------------

把六段内容一路平铺下去，等于把"找重点"这件事推给用户。所以：

* **首屏**只放**网络拓扑图**（基站 + 用户 + 扇区，图内直接标站数/小区数/
  UE 数/站间距，配比例尺）加一排**关键信息卡**；
* 其余按主题折进 **tab**（基站阵列 / 频域与时域 / 信道剖面 / 参数全表），
  用户按需点。

关键信息卡选什么有两条依据：**做仿真通常最关心的**（规模、频点、带宽、阵列、
场景），加上**这次对话里专门点过名的**——调用方把参数名传进 ``highlight``，
它们会被顶到最前面并高亮。这样首屏既精准又不漏。

tab 用**纯 CSS**（radio + label）实现，离线双击打开也能切；页面里那段脚本
只服务于「改配置」面板，**不碰页签**——JS 挂了导航照样能用。

「改配置」页签让用户直接调参数、实时看拓扑变化，改完一键复制一段
``overrides = {...}`` 粘回对话框，agent 照着重跑。payload **只带改动过的项**，
免得把默认值当成用户意图固化下来。
写的时候踩过：``input`` 必须是 ``.tabs`` 的直接子元素且与 ``.panels`` 同级，
中间套一层容器 ``~`` 选择器就全失效——页面上直接露出原生 radio、一个面板都
不显示。**这个光看代码看不出来，是在浏览器里看出来的。**

--- 两条纪律 -----------------------------------------------------------

1. **只画配置里真有的东西。** 站数被六边形栅格吸附（配 2 站实际 7 站）就画
   实际的 7 站并标注；阵列是 legacy 独立阵元就画 64 个独立阵元、不画 1 驱 3。
   说明书画的是**将要跑的那个仿真**，不是用户以为的那个。
2. **区分"用户定的"和"默认补的"。** 参数表里两者分开标注——用户看到
   "这 17 项是我没管、系统替我定的"才有机会喊停。
"""
from __future__ import annotations

import html
import json
import math
import time
import uuid
from typing import Any

import numpy as np

from . import bridge as br
from . import katex as _kx
from .paths import artifacts_root

# ---------------------------------------------------------------------------
# 结构化说明书
# ---------------------------------------------------------------------------

# 参数 -> (中文名, 单位/格式化). 只列值得摆在明面上的，其余进"其他配置"。
_KEY_LABELS: dict[str, tuple[str, str]] = {
    "scenario": ("传播场景", ""),
    "channel_model": ("信道剖面", ""),
    "carrier_freq_hz": ("载波频率", "ghz"),
    "bandwidth_hz": ("带宽", "mhz"),
    "subcarrier_spacing": ("子载波间隔", "khz"),
    "num_rb": ("RB 数", ""),
    "num_sites": ("站点数", ""),
    "sectors_per_site": ("每站扇区", ""),
    "isd_m": ("站间距", "m"),
    "num_ues": ("UE 数", ""),
    "num_interfering_ues": ("每邻区干扰 UE 数", ""),
    "num_bs_tx_ant": ("基站发射端口", ""),
    "num_bs_rx_ant": ("基站接收端口", ""),
    "num_ue_tx_ant": ("终端发射天线", ""),
    "num_ue_rx_ant": ("终端接收天线", ""),
    "tx_power_dbm": ("基站发射功率", "dbm"),
    "ue_tx_power_dbm": ("终端发射功率", "dbm"),
    "noise_figure_db": ("噪声系数", "db"),
    "tx_height_m": ("基站高度", "m"),
    "ue_height_m": ("终端高度", "m"),
    "ue_speed_kmh": ("终端速度", "kmh"),
    "mobility_mode": ("移动模型", ""),
    "topology_layout": ("拓扑布局", ""),
    "link": ("链路方向", ""),
    # 这个键本来就在表里，但从没进过调参面板——ChannelHub 的三档估计器
    # 一直可用，superwireless 只是没提，于是数据集全默默走了默认的 ls_linear。
    "channel_est_mode": ("信道估计模式", ""),
    "tdd_pattern": ("TDD 图案", ""),
    "num_slots_per_sample": ("每样本时隙数", ""),
    "num_ofdm_symbols": ("每时隙符号数", ""),
    "pdsch_load": ("下行负载", ""),
    "pusch_load": ("上行负载", ""),
    "srs_comb": ("SRS 梳齿", ""),
    "srs_periodicity": ("SRS 周期", "slot"),
    "ue_distribution": ("UE 分布", ""),
    "train_penetration_loss_db": ("车体穿透损耗", "db"),
}

_FMT = {
    "ghz": lambda v: f"{float(v) / 1e9:g} GHz",
    "mhz": lambda v: f"{float(v) / 1e6:g} MHz",
    "khz": lambda v: f"{float(v) / 1e3:g} kHz",
    "m": lambda v: f"{float(v):g} m",
    "db": lambda v: f"{float(v):g} dB",
    "dbm": lambda v: f"{float(v):g} dBm",
    "kmh": lambda v: f"{float(v):g} km/h",
    "slot": lambda v: f"{int(v)} 时隙",
    "": lambda v: f"{v}",
}


def _fmt(key: str, value: Any) -> str:
    kind = _KEY_LABELS.get(key, ("", ""))[1]
    try:
        return _FMT.get(kind, _FMT[""])(value)
    except (TypeError, ValueError):
        return str(value)


def _site_positions(cfg: dict[str, Any]) -> tuple[list[dict[str, Any]], str | None]:
    """真实站点/扇区位置。拿不到时回空列表 + 原因。

    **直接问 ChannelHub 的拓扑模块**，不自己重算——六边形栅格的站数吸附
    （1/7/19）和线性布局的两侧交错都在那边，重算一定会漂。
    """
    try:
        from .channelhub import _ensure_path  # noqa: PLC0415

        _ensure_path()
        from msg_embedding.topology.hex_grid import (  # noqa: PLC0415
            make_hex_grid,
            make_linear_grid,
        )
    except Exception as exc:  # noqa: BLE001
        return [], f"取不到拓扑模块：{type(exc).__name__}"

    n_sites = int(cfg.get("num_sites", 1) or 1)
    sectors = int(cfg.get("sectors_per_site", 1) or 1)
    isd = float(cfg.get("isd_m", 500.0) or 500.0)
    h = float(cfg.get("tx_height_m", 25.0) or 25.0)
    scen = str(cfg.get("scenario", "UMa_NLOS"))
    layout = str(cfg.get("topology_layout", "hexagonal"))

    try:
        if layout == "linear":
            cells = make_linear_grid(
                num_sites=n_sites, isd_m=isd, sectors=sectors, tx_height_m=h,
                scenario=scen, track_offset_m=float(cfg.get("track_offset_m", 80.0) or 80.0),
            )
        else:
            rings = 0 if n_sites <= 1 else (1 if n_sites <= 7 else 2)
            cells = make_hex_grid(
                num_rings=rings, isd_m=isd, sectors=sectors, tx_height_m=h, scenario=scen,
            )
    except Exception as exc:  # noqa: BLE001
        return [], f"拓扑构造失败：{type(exc).__name__}: {exc}"

    return [
        {
            "x": float(c.position[0]), "y": float(c.position[1]),
            "z": float(c.position[2]), "az": float(c.azimuth_deg),
            "site_id": int(getattr(c, "site_id", 0)),
        }
        for c in cells
    ], None


def build_spec(
    cfg: dict[str, Any],
    *,
    num_samples: int | None = None,
    user_set: list[str] | None = None,
    dataset_id: str | None = None,
    title: str = "",
) -> dict[str, Any]:
    """把配置整理成结构化说明书。不生成数据，纯解释。"""
    from . import hardware as hw  # noqa: PLC0415
    from .generate import _ensure_bs_panel, _rb_from_bandwidth  # noqa: PLC0415

    cfg = dict(cfg)
    cfg.pop("num_samples", None)
    source = str(cfg.pop("source", "internal_sim"))
    panel, panel_derived = _ensure_bs_panel(cfg)
    hw.apply_array_defaults(cfg)
    applied = hw.strip_markers(cfg)
    array = hw.array_summary(cfg, applied)

    n_rb = int(cfg.get("num_rb") or _rb_from_bandwidth(cfg))
    scs = float(cfg.get("subcarrier_spacing", 30000) or 30000)
    n_sites_cfg = int(cfg.get("num_sites", 1) or 1)
    sectors = int(cfg.get("sectors_per_site", 1) or 1)
    cells, cell_err = _site_positions(cfg)
    n_sites_real = len({c["site_id"] for c in cells}) if cells else n_sites_cfg

    user_keys = set(user_set or [])
    shown, others = [], []
    for k, v in sorted(cfg.items()):
        if k.startswith("_") or k in ("bs_antenna", "measurements", "bs_panel"):
            continue
        row = {
            "key": k, "value": _fmt(k, v),
            "label": _KEY_LABELS.get(k, (k, ""))[0],
            "by_user": k in user_keys,
        }
        (shown if k in _KEY_LABELS else others).append(row)
    shown.sort(key=lambda r: list(_KEY_LABELS).index(r["key"]))

    # RBG 划分：38.214 Table 5.1.2.2.1-1 Configuration 2 在 145~275 PRB 时 P=16
    rbg_size = 16 if n_rb >= 145 else (8 if n_rb >= 73 else 4)
    n_rbg = math.ceil(n_rb / rbg_size)

    notes: list[str] = []
    if n_sites_real != n_sites_cfg:
        notes.append(
            f"配置写的是 {n_sites_cfg} 站，六边形栅格按环数展开后实际是 "
            f"**{n_sites_real} 站**（只能取 1/7/19）。图上画的是实际值。"
        )
    if array.get("antenna_model_mode") == "legacy_64" and hw.is_company_panel(panel):
        notes.append(
            "**阵列走的是 legacy 独立阵元模型，不是本地 1 驱 3 硬件。**"
            "实测这样报出的吞吐偏高约 27%、边缘用户偏高约 61%。"
        )
    if cell_err:
        notes.append(f"站点位置画不出来：{cell_err}")
    # RB 数与带宽推导值不一致时才提——**272 vs 273 是本地的常规口径**
    # （17 RBG × 16 RB vs 38.104 标准表），每次都弹一条黄框纯属噪音。
    # 差得多才是真要提醒的事。
    _rb_auto = _rb_from_bandwidth(cfg)
    if int(cfg.get("num_rb") or 0) and abs(n_rb - _rb_auto) > 2:
        notes.append(
            f"RB 数是显式指定的 {n_rb}，与带宽推导的 {_rb_auto} 差 "
            f"{abs(n_rb - _rb_auto)} 个——**确认这是有意的**。"
        )

    return {
        "title": title or "仿真说明书",
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "source": source,
        "dataset_id": dataset_id,
        "num_samples": num_samples,
        "array": array,
        "panel": list(panel),
        "panel_derived": bool(panel_derived),
        "topology": {
            "layout": str(cfg.get("topology_layout", "hexagonal")),
            "num_sites_config": n_sites_cfg,
            "num_sites_actual": n_sites_real,
            "sectors_per_site": sectors,
            "num_cells": len(cells) or n_sites_cfg * sectors,
            "isd_m": float(cfg.get("isd_m", 0) or 0),
            "cells": cells,
        },
        "frequency": {
            "carrier_freq_hz": float(cfg.get("carrier_freq_hz", 0) or 0),
            "bandwidth_hz": float(cfg.get("bandwidth_hz", 0) or 0),
            "scs_hz": scs,
            "num_rb": n_rb,
            "rbg_size": rbg_size,
            "num_rbg": n_rbg,
            "occupied_hz": n_rb * 12 * scs,
        },
        "time": {
            "tdd_pattern": str(cfg.get("tdd_pattern", "DDDSU")),
            "slots_per_sample": int(cfg.get("num_slots_per_sample", 1) or 1),
            "symbols_per_slot": int(cfg.get("num_ofdm_symbols", 14) or 14),
        },
        "params": shown,
        "other_params": others,
        "notes": notes,
        "config": cfg,
    }


# ---------------------------------------------------------------------------
# SVG 示意图
# ---------------------------------------------------------------------------


def _esc(s: Any) -> str:
    return html.escape(str(s), quote=True)


def _bold(s: Any) -> str:
    """转义后把 ``**...**`` 变成 ``<b>``。

    这些说明字符串同时给纯文本（CLAUDE.md、MCP 返回值）和 HTML 用，
    源头统一写 markdown 强调；HTML 侧不转的话页面上会直接露出星号。
    """
    out, parts = [], _esc(s).split("**")
    for i, seg in enumerate(parts):
        out.append(f"<b>{seg}</b>" if i % 2 else seg)
    return "".join(out)


def _plain(s: Any) -> str:
    """去掉 markdown 强调标记，只留文字。

    **SVG 的 ``<text>`` 不支持内联加粗**，把 ``**...**`` 原样写进去
    就是在图上露出一对裸星号。这些描述字符串是和 HTML 共用的，
    源头统一写 markdown，到 SVG 这一侧只能把标记去掉。
    """
    return str(s).replace("**", "")


def _svg_array(spec: dict[str, Any]) -> str:
    """阵列面板：RF 端口栅格 + 每端口驱动的物理阵子。"""
    arr = spec["array"]
    n_h, n_v, n_p = (list(spec["panel"]) + [1, 1, 1])[:3]
    m = int(arr.get("elements_per_rf_port") or 1)
    dv = float(arr.get("ae_vertical_spacing_lambda") or 0.5)
    dh = float(arr.get("horizontal_spacing_lambda") or 0.5)
    legacy = arr.get("antenna_model_mode") == "legacy_64"

    cw, ae_h, gap = 34, 13, 5          # 列宽、单个阵子高、阵子间隙
    port_h = m * ae_h + (m - 1) * gap
    pad_l, pad_t = 96, 54
    w = pad_l + n_h * cw + 268
    h = pad_t + n_v * (port_h + 16) + 74

    out = [f'<svg viewBox="0 0 {w} {h}" width="100%" style="max-width:{w}px" '
           f'role="img" aria-label="阵列面板示意图">']
    out.append('<style>.lb{font:11px ui-monospace,Consolas,monospace;fill:#6e6e73}'
               '.ti{font:600 12.5px system-ui;fill:currentColor}'
               '.ae{fill:#0071e3;opacity:.85}.ae2{fill:#5a3ec8;opacity:.85}'
               '.pt{fill:none;stroke:#0071e3;stroke-width:1.4;opacity:.55}</style>')
    out.append(f'<text class="ti" x="8" y="20">'
               f'{"64 个独立阵元（legacy）" if legacy else f"RF 端口 {n_h}x{n_v}x{n_p} = {n_h*n_v*n_p}，每端口 1 驱 {m}"}'
               f'</text>')
    out.append(f'<text class="lb" x="8" y="38">'
               f'{"间距一律 " + format(dh, "g") + "λ —— 不是本地硬件" if legacy else f"物理阵子 {n_h}x{n_v*m}x{n_p} = {n_h*n_v*m*n_p}"}'
               f'</text>')

    for v in range(n_v):
        y0 = pad_t + v * (port_h + 16)
        for hh in range(n_h):
            x0 = pad_l + hh * cw
            if not legacy:
                out.append(f'<rect class="pt" x="{x0 - 3}" y="{y0 - 3}" '
                           f'width="{cw - 8}" height="{port_h + 6}" rx="4"/>')
            for q in range(m):
                y = y0 + q * (ae_h + gap)
                # 双极化画成两个半宽块（+45 / -45）
                out.append(f'<rect class="ae" x="{x0}" y="{y}" width="{(cw-12)//2}" '
                           f'height="{ae_h}" rx="2"/>')
                if n_p == 2:
                    out.append(f'<rect class="ae2" x="{x0 + (cw-12)//2 + 2}" y="{y}" '
                               f'width="{(cw-12)//2}" height="{ae_h}" rx="2"/>')
        out.append(f'<text class="lb" x="{pad_l - 10}" y="{y0 + port_h/2 + 4}" '
                   f'text-anchor="end">端口 v={v}</text>')

    # 间距标注
    xr = pad_l + n_h * cw + 16
    out.append(f'<line x1="{xr}" y1="{pad_t}" x2="{xr}" y2="{pad_t + ae_h}" '
               f'stroke="#6e6e73" stroke-width="1"/>')
    out.append(f'<text class="lb" x="{xr + 8}" y="{pad_t + ae_h}">阵子垂直间距 {dv:g}λ</text>')
    if not legacy and m > 1:
        out.append(f'<line x1="{xr}" y1="{pad_t}" x2="{xr}" y2="{pad_t + port_h + 16}" '
                   f'stroke="#5a3ec8" stroke-width="1.4"/>')
        out.append(f'<text class="lb" x="{xr + 8}" y="{pad_t + port_h + 12}" fill="#5a3ec8">'
                   f'端口相位中心间距 {m * dv:g}λ{"（&gt;λ，有栅瓣）" if m * dv > 1 else ""}</text>')
    yb = pad_t + n_v * (port_h + 16) + 8
    out.append(f'<line x1="{pad_l}" y1="{yb}" x2="{pad_l + cw}" y2="{yb}" '
               f'stroke="#6e6e73" stroke-width="1"/>')
    out.append(f'<text class="lb" x="{pad_l}" y="{yb + 16}">水平间距 {dh:g}λ</text>')
    if n_p == 2:
        out.append(f'<rect class="ae" x="{pad_l + 150}" y="{yb + 4}" width="10" height="10" rx="2"/>'
                   f'<text class="lb" x="{pad_l + 164}" y="{yb + 13}">+45°</text>'
                   f'<rect class="ae2" x="{pad_l + 210}" y="{yb + 4}" width="10" height="10" rx="2"/>'
                   f'<text class="lb" x="{pad_l + 224}" y="{yb + 13}">-45°</text>')
    out.append("</svg>")
    return "".join(out)


def unit_hex_layouts() -> dict[str, list[list[float]]]:
    """ISD = 1 时的六边形站点坐标，按站数索引（"1" / "7" / "19"）。

    **由 ChannelHub 现算，不硬编码。** 六边形位置随 ISD 线性缩放，所以前端
    只要乘一个 ISD 就能得到精确坐标——不必在 JS 里重写栅格逻辑，也就不会漂。
    """
    try:
        from .channelhub import _ensure_path  # noqa: PLC0415

        _ensure_path()
        from msg_embedding.topology.hex_grid import make_hex_grid  # noqa: PLC0415
    except Exception:  # noqa: BLE001
        return {}
    out: dict[str, list[list[float]]] = {}
    for rings, n in ((0, 1), (1, 7), (2, 19)):
        try:
            cells = make_hex_grid(num_rings=rings, isd_m=1.0, sectors=1,
                                  tx_height_m=25.0, scenario="UMa_NLOS")
        except Exception:  # noqa: BLE001
            continue
        out[str(n)] = [[round(float(c.position[0]), 6), round(float(c.position[1]), 6)]
                       for c in cells]
    return out


def planned_ue_drop(cfg: dict[str, Any], n: int = 200) -> list[tuple[float, float]]:
    """算出这个配置**实际会撒在哪**的用户位置。

    **这不是示意图，是真的撒点。** 位置完全由配置和 seed 决定，
    仿真跑不跑都一样——所以从预设直接出说明书时也能算出来，
    不需要等数据集生成。

    直接调 ChannelHub 的 ``_place_ues``，用它自己的 RNG 约定
    （``default_rng(ue_seed + idx)``，每个样本一次），
    所以和真跑出来的坐标**逐位相同**。

    **别自己手搓撒点。** 我第一版按六边形内均匀自己撒，
    把 ``ue_distribution``（uniform / clustered / hotspot）整个忽略了——
    配了热点分布的场景画出来还是均匀的，图和真跑的完全是两回事。
    """
    try:
        from .channelhub import require_source  # noqa: PLC0415

        cls = require_source("internal_sim")
        src = cls(dict(cfg))
        sites = src._build_sites()  # noqa: SLF001
        n_ue = int(getattr(src, "num_ues", 1) or 1)
        # **一次把所有 UE 放完，seed 是 ue_seed + 7000。**
        # 按 `ue_seed + idx` 逐样本放是错的——RNG 消耗顺序不同，
        # 实测那样和真跑差最多 1232 m，等于画了一张完全无关的图。
        # 这个写法与真跑逐位相同（实测最大误差 0.0000 m）。
        pos = src._place_ues(  # noqa: SLF001
            np.random.default_rng(int(getattr(src, "_ue_seed", 0)) + 7000),
            sites, min(n_ue, int(n)))
        return [(float(r[0]), float(r[1])) for r in np.asarray(pos)]
    except Exception as exc:  # noqa: BLE001
        # **不静默吞掉。** 撒点画不出来时说明配置有问题，用户该知道。
        import os  # noqa: PLC0415
        import sys  # noqa: PLC0415

        if os.environ.get("SUPERWIRELESS_DEBUG"):
            print(f"[spec] 撒点算不出来：{type(exc).__name__}: {exc}",
                  file=sys.stderr, flush=True)
        return []


def _svg_layout(
    spec: dict[str, Any],
    ue_xy: list[tuple[float, float]] | None = None,
    *,
    size: int | None = None,
    ue_is_real: bool = True,
) -> str:
    """网络拓扑图：六边形小区 + 扇区 + 基站 + 用户，图上直接标关键数。

    画成教科书里那种蜂窝图——每个站一个六边形（外接圆半径 ISD/√3、顶点在
    30°+k·60°，边正对邻站），里面按扇区数切扇形。比一堆重叠的圆好读得多。
    """
    topo = spec["topology"]
    cells = topo["cells"]
    if not cells:
        return '<p class="src">拿不到站点位置，无法绘制拓扑图。</p>'

    isd = float(topo["isd_m"] or 0)
    hex_r = isd / math.sqrt(3.0) if isd else 0.0
    sites = sorted({(c["x"], c["y"]) for c in cells})

    xs = [c["x"] for c in cells] + [p[0] for p in (ue_xy or [])]
    ys = [c["y"] for c in cells] + [p[1] for p in (ue_xy or [])]
    if hex_r:                       # 六边形要完整落在画布内
        xs = xs + [min(xs) - hex_r, max(xs) + hex_r]
        ys = ys + [min(ys) - hex_r, max(ys) + hex_r]
    cx, cy = (max(xs) + min(xs)) / 2, (max(ys) + min(ys)) / 2

    # 画布跟着部署形状走，米/像素两轴一致（几何不失真），高度只做上下限保护。
    n_sites = len(sites)
    W = int(size) if size else (340 if n_sites <= 1 else 660)
    dx = max(max(xs) - min(xs), 1.0) * 1.02
    dy = max(max(ys) - min(ys), 1.0) * 1.06
    scale = W / dx
    Hc = min(max(dy * scale, W * 0.34), W)
    H = int(Hc) + 34

    def px(x: float) -> float:
        return W / 2 + (x - cx) * scale

    def py(y: float) -> float:
        return Hc / 2 - (y - cy) * scale

    out = [f'<svg viewBox="0 0 {W} {H}" width="100%" style="max-width:{W}px" '
           f'role="img" aria-label="网络拓扑图">']
    out.append(
        '<defs>'
        '<radialGradient id="sg" cx="50%" cy="50%" r="50%">'
        '<stop offset="0%" stop-color="#0071e3" stop-opacity=".20"/>'
        '<stop offset="100%" stop-color="#0071e3" stop-opacity=".03"/>'
        '</radialGradient></defs>'
        '<style>'
        '.lb{font:11px ui-monospace,Consolas,monospace;fill:#6e6e73}'
        '.hex{fill:none;stroke:#0071e3;stroke-opacity:.30;stroke-width:1.1}'
                '.bore{stroke:#0071e3;stroke-width:1.8;stroke-opacity:.65;stroke-linecap:round}'
        '.ueo{fill:#34c759;fill-opacity:.65}'
        # 示意撒点用空心，和数据集里的真实撒点一眼能分开
        '.ues{fill:none;stroke:#34c759;stroke-width:1.2;stroke-opacity:.7}'
        '.bsh{fill:#fff;fill-opacity:.95}.bsd{fill:#ff3b30}'
        '.bx{fill:#0071e3;fill-opacity:.06}'
        '.bn{font:700 15px ui-monospace,Consolas,monospace;fill:#0071e3}'
        # 小区编号压在扇区色块上，用细白描边提清晰度（paint-order 先描边后填充）
        # **类名必须带前缀。** SVG 的 <style> 是文档级的，`.cl` 和调参面板的
        # <span class="cl">、`.sl` 和 TDD 图的时隙标号（fill:#fff）全撞了——
        # 实测小区编号糊成蓝块、站点编号白底白字整个看不见。
        '.tpc{font:600 9.5px ui-monospace,Consolas,monospace;fill:#0071e3;'
        'text-anchor:middle;paint-order:stroke;stroke:#fff;stroke-width:1.6px;'
        'stroke-linejoin:round}'
        # 站点编号给个底：白描边在小字号上会把字整个糊掉，实测糊成一片白块
        '.tpsb{fill:#fff;fill-opacity:.85;stroke:#d2d2d7;stroke-width:.5}'
        '.tps{font:600 8.5px ui-monospace,Consolas,monospace;fill:#3a3a3c;'
        'text-anchor:middle}'
        '</style>'
    )

    # --- 六边形小区 ---
    if hex_r and topo["layout"] != "linear":
        rr = hex_r * scale
        for (sx, sy) in sites:
            X, Y = px(sx), py(sy)
            pts = " ".join(
                f'{X + rr * math.cos(math.radians(30 + 60 * k)):.1f},'
                f'{Y - rr * math.sin(math.radians(30 + 60 * k)):.1f}'
                for k in range(6)
            )
            out.append(f'<polygon class="hex" points="{pts}"/>')

    # --- 扇区扇形 ---
    sec = max(int(topo["sectors_per_site"] or 1), 1)
    half = math.radians(360.0 / sec / 2.0) if sec > 1 else math.pi
    reach = (hex_r * scale * 0.92) if hex_r else W * 0.28
    for c in cells:
        X, Y = px(c["x"]), py(c["y"])
        a = math.radians(c["az"])
        if sec > 1:
            x1, y1 = X + reach * math.cos(a - half), Y - reach * math.sin(a - half)
            x2, y2 = X + reach * math.cos(a + half), Y - reach * math.sin(a + half)
            out.append(f'<path fill="url(#sg)" d="M{X:.1f},{Y:.1f} L{x1:.1f},{y1:.1f} '
                       f'A{reach:.1f},{reach:.1f} 0 0,0 {x2:.1f},{y2:.1f} Z"/>')
        else:
            out.append(f'<circle fill="url(#sg)" cx="{X:.1f}" cy="{Y:.1f}" r="{reach:.1f}"/>')

    _uc = "ueo" if ue_is_real else "ues"
    for x, y in (ue_xy or []):
        out.append(f'<circle class="{_uc}" cx="{px(x):.1f}" cy="{py(y):.1f}" r="2.7"/>')

    # --- 小区编号：贴在扇区指向上，字小、有描边，不跟别的元素打架 ---
    # 编号规则和数据里一致：小区 i 属于站点 i//sectors，扇区 i%sectors。
    for i, c in enumerate(cells):
        X, Y = px(c["x"]), py(c["y"])
        a = math.radians(c["az"])
        d = reach * 0.62 if sec > 1 else reach * 0.55
        out.append(f'<text class="tpc" x="{X + d * math.cos(a):.1f}" '
                   f'y="{Y - d * math.sin(a) + 3:.1f}">C{i}</text>')

    arm = max(reach * 0.5, 12)
    for c in cells:
        X, Y = px(c["x"]), py(c["y"])
        a = math.radians(c["az"])
        out.append(f'<line class="bore" x1="{X:.1f}" y1="{Y:.1f}" '
                   f'x2="{X + arm * math.cos(a):.1f}" y2="{Y - arm * math.sin(a):.1f}"/>')
    for si, (sx, sy) in enumerate(sites):
        X, Y = px(sx), py(sy)
        out.append(f'<circle class="bsh" cx="{X:.1f}" cy="{Y:.1f}" r="5.6"/>'
                   f'<circle class="bsd" cx="{X:.1f}" cy="{Y:.1f}" r="3.4"/>')
        # 站点编号放在基站正下方，和小区编号错开
        _lbl = f"Site{si}"
        _w = 6.0 + 5.0 * len(_lbl)
        out.append(f'<rect class="tpsb" x="{X - _w / 2:.1f}" y="{Y + 8:.1f}" '
                   f'width="{_w:.1f}" height="11" rx="3"/>'
                   f'<text class="tps" x="{X:.1f}" y="{Y + 16:.1f}">{_lbl}</text>')

    # --- 图内信息盒 ---
    facts = [(str(topo["num_sites_actual"]), "站点"), (str(topo["num_cells"]), "小区")]
    if ue_xy:
        facts.append((str(len(ue_xy)), "用户"))
    if isd:
        facts.append((f"{isd:g} m", "站间距"))
    bw = 92 * len(facts) + 16
    out.append(f'<rect class="bx" x="8" y="8" width="{bw}" height="46" rx="9"/>')
    for i, (num, lab) in enumerate(facts):
        x = 20 + i * 92
        out.append(f'<text class="bn" x="{x}" y="30">{_esc(num)}</text>')
        out.append(f'<text class="lb" x="{x}" y="46">{_esc(lab)}</text>')

    # --- 比例尺 ---
    if isd:
        bar = isd * scale
        if bar < W * 0.55:
            x0, yb = 12, H - 26
            out.append(f'<line x1="{x0}" y1="{yb}" x2="{x0 + bar:.1f}" y2="{yb}" '
                       f'stroke="#6e6e73" stroke-width="1.5"/>')
            for xx in (x0, x0 + bar):
                out.append(f'<line x1="{xx:.1f}" y1="{yb-4}" x2="{xx:.1f}" y2="{yb+4}" '
                           f'stroke="#6e6e73" stroke-width="1.5"/>')
            out.append(f'<text class="lb" x="{x0 + bar + 8:.1f}" y="{yb + 4}">'
                       f'{isd:g} m（站间距）</text>')

    lx = W - (172 if ue_xy else 118)
    out.append(f'<circle class="bsh" cx="{lx}" cy="{H-30}" r="5"/>'
               f'<circle class="bsd" cx="{lx}" cy="{H-30}" r="3"/>'
               f'<text class="lb" x="{lx+10}" y="{H-26}">基站</text>')
    if ue_xy:
        out.append(f'<circle class="{_uc}" cx="{lx+54}" cy="{H-30}" r="3.4"/>'
                   f'<text class="lb" x="{lx+63}" y="{H-26}">'
                   f'用户</text>')
    dx2 = lx + (106 if ue_xy else 54)
    out.append(f'<line class="bore" x1="{dx2}" y1="{H-30}" x2="{dx2+16}" y2="{H-30}"/>'
               f'<text class="lb" x="{dx2+21}" y="{H-26}">扇区</text>')
    out.append("</svg>")
    return "".join(out)


# 可在 HTML 里直接改的参数。**只放改了确实有意义、且改错也不会把仿真跑崩的那些**——
# 这个面板的用途是"用户按自己的需要调一版，一键粘回给 agent"，不是完整配置编辑器。
# (key, 中文名, 类型, 选项或范围, 提示)
_EDITABLE: tuple[tuple[str, str, str, Any, str], ...] = (
    ("num_sites", "站点数", "select", [1, 7, 19], "六边形栅格只能取 1/7/19"),
    ("sectors_per_site", "每站扇区", "select", [1, 3], ""),
    ("isd_m", "站间距 m", "number", (50, 6000, 10), "密集城区 150~200、宏站 500、农村 1732+"),
    ("num_ues", "每轮 UE 数", "number", (1, 500, 1), "样本数会自动对齐到它的整数倍"),
    ("num_interfering_ues", "每邻区干扰 UE", "number", (0, 32, 1), "主要影响测量域（SRS）"),
    ("num_bs_tx_ant", "基站端口", "select", [4, 16, 32, 64, 256], "64 会自动用 1 驱 3 真实阵列"),
    ("num_ue_rx_ant", "终端接收天线", "select", [2, 4], ""),
    ("carrier_freq_hz", "载波", "select",
     [700000000.0, 2100000000.0, 2600000000.0, 3500000000.0], "默认 n41 2.6 GHz"),
    ("bandwidth_hz", "带宽", "select",
     [10000000.0, 20000000.0, 100000000.0], ""),
    ("num_rb", "RB 数", "number", (12, 275, 1), "默认 272 = 17 RBG x 16"),
    ("scenario", "传播场景", "select",
     ["UMa_NLOS", "UMa_LOS", "UMi_NLOS", "UMi_LOS", "InF"], ""),
    ("channel_model", "信道剖面", "select",
     ["CDL-A", "CDL-B", "CDL-C", "CDL-D", "CDL-E",
      "TDL-A", "TDL-B", "TDL-C", "TDL-D", "TDL-E"], "TDL 没有角度，波束/定位类必须用 CDL"),
    ("tx_power_dbm", "发射功率 dBm", "number", (10, 55, 1), "UMi 默认 33、UMa 默认 43"),
    ("ue_speed_kmh", "终端速度 km/h", "number", (0, 500, 1), ""),
    ("link", "链路方向", "select", ["DL", "UL", "BOTH"], "测量域 SIR 只在 BOTH 下产生"),
    ("channel_est_mode", "信道估计", "select",
     ["ideal", "ls_linear", "ls_mmse"],
     "ideal=拿真值；ls_mmse 比 ls_linear 实测好 0.7~4.6 dB，导频越挤差距越大"),
    ("num_samples", "样本数", "number", (1, 5000, 1), "由 sw_sample_size 算，别拍脑袋"),
    # --- 系统级仿真旋钮（sw_system_sim 用，不进 ChannelHub 的信道生成）---
    ("neighbor_prb_util", "邻区 PRB 利用率", "number", (0.0, 1.0, 0.05),
     "全网统一值；1.0 = 所有邻区 full buffer（几何 SINR 的原始假设）"),
    ("neighbor_load_jitter", "负载抖动", "number", (0.0, 0.5, 0.01),
     "实际值在配置值 ±这个比例内逐快照波动，默认 0.05"),
    ("csi_aging", "CSI 老化", "select", ["on", "off"],
     "关掉 = 零时延完美 CSI，那是上界不是现网，会高估 MU 增益"),
    ("srs_period_ms", "SRS 周期 ms", "select", [5.0, 10.0, 20.0, 40.0],
     "现网典型 10~20 ms"),
    ("srs_hopping", "SRS 跳频", "select", ["on", "off"],
     "38.211 C_SRS=57：每跳 16 RB、17 跳扫完全带。**老化的主导项**"),
    ("csi_processing_delay_ms", "CSI 处理时延 ms", "number", (0.0, 20.0, 0.5),
     "信道估计 + 预编码计算 + 调度下发"),
    ("olla_speedup", "OLLA 步长放大", "number", (1.0, 50.0, 1.0),
     "等比放大两个步长，稳态 BLER 不变、收敛更快、抖动更大。**出正式结论设回 1**"),
)


#: 系统级仿真的旋钮不在 ChannelHub 的信道生成配置里，页面上拿不到当前值。
#: 给它们一份默认值，**必须和 sw_system_sim 的函数签名一致**——
#: 两处漂了的话页面显示的就不是实际会跑的值，而这种不一致没有任何提示。
_SIM_DEFAULTS: dict[str, Any] = {
    "neighbor_prb_util": 0.3,
    "neighbor_load_jitter": 0.05,
    "csi_aging": "on",
    "srs_period_ms": 10.0,
    "srs_hopping": "on",
    "csi_processing_delay_ms": 2.0,
    "olla_speedup": 1.0,
}


def editable_keys() -> frozenset[str]:
    """页面上允许改的参数名。回传接口的白名单以此为准，别另抄一份。"""
    return frozenset(k for k, *_ in _EDITABLE)


def _interactive(spec: dict[str, Any], *, apply_url: str = "",
                 spec_id: str = "") -> str:
    """可交互的调参面板：改参数 → 实时看拓扑 → 回传给 agent。

    **回传有两条路，主备关系不是二选一。** 页面从 ``http://127.0.0.1`` 打开时
    走「应用到仿真」直接 POST 回 MCP 进程，用户点一下就完；从 ``file://`` 打开时
    浏览器不许跨到环回源，自动退回「复制 → 粘回对话框」。同一份 HTML 两种都能用，
    落盘的那份换台机器打开照样有效。

    **拓扑预览用的坐标由 ChannelHub 现算后内嵌**（ISD=1 的单位布局），
    前端只做线性缩放，不在 JS 里重写栅格逻辑——所以预览和真跑的几何一致，
    不会出现"图上七站、跑出来十九站"这种漂移。
    """
    cfg = spec["config"]
    init: dict[str, Any] = {}
    rows: list[str] = []

    for key, label, kind, spec_v, hint in _EDITABLE:
        if key == "num_samples":
            cur = spec.get("num_samples")
            if cur is None:
                continue
        else:
            cur = cfg.get(key, _SIM_DEFAULTS.get(key))
            if cur is None:
                continue
        init[key] = cur
        hint_html = f'<div class="ch">{_bold(hint)}</div>' if hint else ""
        if kind == "select":
            opts = list(spec_v)
            if cur not in opts:
                opts = [cur, *opts]
            body = "".join(
                f'<option value="{_esc(o)}"{" selected" if o == cur else ""}>'
                f'{_esc(_fmt(key, o))}</option>' for o in opts
            )
            ctl = f'<select data-k="{key}">{body}</select>'
        else:
            lo, hi, step = spec_v
            ctl = (f'<input type="number" data-k="{key}" value="{_esc(cur)}" '
                   f'min="{lo}" max="{hi}" step="{step}">')
        rows.append(f'<label class="ctl"><span class="cl">{_esc(label)}</span>'
                    f'{ctl}{hint_html}</label>')

    state = {
        "init": init,
        "layout": spec["topology"]["layout"],
        "track_offset_m": float(cfg.get("track_offset_m", 80.0) or 80.0),
        "unit": unit_hex_layouts(),
        "title": spec["title"],
        "post": apply_url,
        "id": spec_id,
    }
    return f"""
<p class="lead">改完点<b>应用到仿真</b>，改动直接回到 agent——不用复制、不用切窗口。
只会带上<b>改动过</b>的项，没动的不进 payload。
<span class="src">（这份 HTML 从文件直接打开时没有回传通道，会自动退回复制粘贴。）</span></p>
<div class="ctls">{"".join(rows)}</div>
<div class="hero" id="prev"></div>
<div class="pvbar">
  <button class="btn" id="ap" hidden>应用到仿真</button>
  <button class="btn" id="cp">复制配置改动</button>
  <button class="btn ghost" id="rs">重置</button>
  <span class="src" id="msg"></span>
</div>
<textarea id="pl" class="pl" readonly rows="9"></textarea>
<script>
const ST={json.dumps(state, ensure_ascii=False)};
const cur=Object.assign({{}},ST.init);
const NL=String.fromCharCode(10);
const F=(k,v)=>{{
  if(k==='carrier_freq_hz')return (v/1e9)+' GHz';
  if(k==='bandwidth_hz')return (v/1e6)+' MHz';
  return String(v);
}};
function sites(){{
  const n=+cur.num_sites||1, isd=+cur.isd_m||500;
  if(ST.layout==='linear'){{
    const o=ST.track_offset_m, a=[];
    for(let i=0;i<n;i++)a.push([(i-(n-1)/2)*isd, (i%2?-o:o)]);
    return a;
  }}
  const u=ST.unit[String(n)]||ST.unit['7']||[[0,0]];
  return u.map(p=>[p[0]*isd,p[1]*isd]);
}}
function draw(){{
  const S=sites(), sec=+cur.sectors_per_site||1, isd=+cur.isd_m||500;
  const hexR=ST.layout==='linear'?0:isd/Math.sqrt(3);
  let xs=S.map(p=>p[0]), ys=S.map(p=>p[1]);
  if(hexR){{xs=xs.concat([Math.min(...xs)-hexR,Math.max(...xs)+hexR]);
           ys=ys.concat([Math.min(...ys)-hexR,Math.max(...ys)+hexR]);}}
  const cx=(Math.max(...xs)+Math.min(...xs))/2, cy=(Math.max(...ys)+Math.min(...ys))/2;
  const W=S.length<=1?340:660;
  const dx=Math.max(Math.max(...xs)-Math.min(...xs),1)*1.08;
  const dy=Math.max(Math.max(...ys)-Math.min(...ys),1)*1.16;
  const sc=W/dx, Hc=Math.min(Math.max(dy*sc,W*0.34),W), H=Math.round(Hc)+34;
  const PX=x=>W/2+(x-cx)*sc, PY=y=>Hc/2-(y-cy)*sc;
  let o=`<svg viewBox="0 0 ${{W}} ${{H}}" width="100%" style="max-width:${{W}}px">`;
  o+='<defs><radialGradient id="sg2" cx="50%" cy="50%" r="50%">'
   +'<stop offset="0%" stop-color="#0071e3" stop-opacity=".20"/>'
   +'<stop offset="100%" stop-color="#0071e3" stop-opacity=".03"/></radialGradient></defs>'
   +'<style>.lb{{font:11px ui-monospace,Consolas,monospace;fill:#6e6e73}}'
   +'.hex{{fill:none;stroke:#0071e3;stroke-opacity:.30;stroke-width:1.1}}'
   +'.bore{{stroke:#0071e3;stroke-width:1.8;stroke-opacity:.65;stroke-linecap:round}}'
   +'.bsh{{fill:#fff;fill-opacity:.95}}.bsd{{fill:#ff3b30}}'
   +'.bx{{fill:#0071e3;fill-opacity:.06}}'
   +'.bn{{font:700 15px ui-monospace,Consolas,monospace;fill:#0071e3}}</style>';
  if(hexR){{const rr=hexR*sc;
    for(const[sx,sy]of S){{const X=PX(sx),Y=PY(sy);let pts=[];
      for(let k=0;k<6;k++){{const a=(30+60*k)*Math.PI/180;
        pts.push((X+rr*Math.cos(a)).toFixed(1)+','+(Y-rr*Math.sin(a)).toFixed(1));}}
      o+=`<polygon class="hex" points="${{pts.join(' ')}}"/>`;}}}}
  const half=sec>1?(Math.PI/sec):Math.PI, reach=hexR?hexR*sc*0.92:W*0.28;
  for(const[sx,sy]of S)for(let k=0;k<sec;k++){{
    const X=PX(sx),Y=PY(sy),a=k*2*Math.PI/sec;
    if(sec>1){{const x1=X+reach*Math.cos(a-half),y1=Y-reach*Math.sin(a-half),
      x2=X+reach*Math.cos(a+half),y2=Y-reach*Math.sin(a+half);
      o+=`<path fill="url(#sg2)" d="M${{X.toFixed(1)}},${{Y.toFixed(1)}} L${{x1.toFixed(1)}},${{y1.toFixed(1)}} A${{reach.toFixed(1)}},${{reach.toFixed(1)}} 0 0,0 ${{x2.toFixed(1)}},${{y2.toFixed(1)}} Z"/>`;}}
    else o+=`<circle fill="url(#sg2)" cx="${{X.toFixed(1)}}" cy="${{Y.toFixed(1)}}" r="${{reach.toFixed(1)}}"/>`;}}
  const arm=Math.max(reach*0.5,12);
  for(const[sx,sy]of S){{const X=PX(sx),Y=PY(sy);
    for(let k=0;k<sec;k++){{const a=k*2*Math.PI/sec;
      o+=`<line class="bore" x1="${{X.toFixed(1)}}" y1="${{Y.toFixed(1)}}" x2="${{(X+arm*Math.cos(a)).toFixed(1)}}" y2="${{(Y-arm*Math.sin(a)).toFixed(1)}}"/>`;}}
    o+=`<circle class="bsh" cx="${{X.toFixed(1)}}" cy="${{Y.toFixed(1)}}" r="5.6"/><circle class="bsd" cx="${{X.toFixed(1)}}" cy="${{Y.toFixed(1)}}" r="3.4"/>`;}}
  const f=[[String(S.length),'站点'],[String(S.length*sec),'小区'],[isd+' m','站间距']];
  o+=`<rect class="bx" x="8" y="8" width="${{92*f.length+16}}" height="46" rx="9"/>`;
  f.forEach((it,i)=>{{o+=`<text class="bn" x="${{20+i*92}}" y="30">${{it[0]}}</text>`
    +`<text class="lb" x="${{20+i*92}}" y="46">${{it[1]}}</text>`;}});
  o+=`<text class="lb" x="12" y="${{H-10}}">预览 · 撒点由生成时决定，这里不画</text></svg>`;
  document.getElementById('prev').innerHTML=o;
}}
function diff(){{
  const d={{}};for(const k in cur)if(String(cur[k])!==String(ST.init[k]))d[k]=cur[k];
  return d;
}}
const AP=document.getElementById('ap');
// file:// 打开时浏览器不许跨到环回源，fetch 必然被拦——那就不摆这个按钮，
// 老老实实退回复制粘贴。同一份 HTML 两种打开方式都成立。
const CANPOST=!!ST.post&&location.protocol.indexOf('http')===0;
if(CANPOST)AP.hidden=false;
function sync(){{
  draw();
  const d=diff(), ks=Object.keys(d);
  const t=document.getElementById('pl');
  AP.disabled=!ks.length;
  if(!ks.length){{t.value='（还没有改动。调上面的参数后这里会出现可粘贴的内容。）';return;}}
  let s='【仿真配置调整】基于 '+ST.title+NL+'改动 '+ks.length+' 项：'+NL;
  for(const k of ks)s+='  '+k+': '+F(k,ST.init[k])+' -> '+F(k,d[k])+NL;
  s+=NL+'overrides = '+JSON.stringify(d)+NL+NL
   +'请用这个 overrides 重新配置并重跑（sw_revise / sw_generate），完事再出一份说明书。';
  t.value=s;
}}
document.querySelectorAll('.ctls [data-k]').forEach(el=>{{
  el.addEventListener('input',()=>{{
    const k=el.dataset.k,v=el.value;
    cur[k]=(el.tagName==='SELECT'&&typeof ST.init[k]==='string')?v:
           (isNaN(+v)||v==='')?v:+v;
    sync();
  }});
}});
document.getElementById('rs').onclick=()=>{{
  Object.assign(cur,ST.init);
  document.querySelectorAll('.ctls [data-k]').forEach(el=>{{el.value=ST.init[el.dataset.k];}});
  sync();document.getElementById('msg').textContent='';
}};
document.getElementById('cp').onclick=()=>{{
  const t=document.getElementById('pl');
  t.removeAttribute('readonly');t.select();t.setSelectionRange(0,99999);
  let ok=false;
  try{{ok=document.execCommand('copy');}}catch(e){{}}
  if(navigator.clipboard)navigator.clipboard.writeText(t.value).then(()=>{{}},()=>{{}});
  t.setAttribute('readonly','');
  document.getElementById('msg').textContent=ok?'已复制，粘到对话框即可':'请手动全选复制上面的文本';
}};
AP.onclick=()=>{{
  const d=diff(); if(!Object.keys(d).length)return;
  const m=document.getElementById('msg');
  m.textContent='正在送给 agent…';AP.disabled=true;
  // nonce 让重发是幂等的：回执可能在路上丢，重发一次比让用户猜安全。
  const body=JSON.stringify({{id:ST.id,overrides:d,nonce:ST.id+'-'+Date.now()+'-'+
    Object.keys(d).sort().join(','),text:document.getElementById('pl').value}});
  const send=()=>fetch(ST.post,{{method:'POST',
    headers:{{'Content-Type':'application/json'}},body:body}}).then(r=>r.json());
  send().catch(()=>new Promise(r=>setTimeout(r,400)).then(send))
   .then(j=>{{
     if(!j.ok){{m.textContent='没收下：'+(j.error||'未知原因')+'，可以改用复制';}}
     else{{
       // 回显服务端真正收到的项，并说清落到哪一步了。只说"已送达"的话，
       // agent 没在等时对话框毫无动静，用户会以为没生效。
       // 用 textContent 拼，不碰 innerHTML。
       const ks=Object.keys(d).map(k=>k+'='+F(k,d[k])).join('、');
       m.textContent='已收下 '+j.n+' 项：'+ks+'　·　'+(j.msg||'');
       m.className=j.waiting?'ok':'warn';
     }}
     AP.disabled=false;}})
   .catch(()=>{{m.textContent='连不上 agent（服务可能已退出）。先看对话框，'
                +'它收到就会复述改动；没有的话再用「复制配置改动」。';
              AP.disabled=false;}});
}};
sync();
</script>
"""


def _flow_svg(flow: dict[str, Any]) -> str:
    """把算法流程画成竖排流程图。

    **写流程图不只是为了给用户看。** 把实现摊成步骤之后，
    「代码是不是真这么做的」变成一个可以逐条对照的问题——
    这是自查实现有没有跑偏最省事的办法。
    """
    steps = flow.get("steps") or []
    if not steps:
        return ""
    brs = {b["at"]: b for b in (flow.get("branches") or [])}
    lb = flow.get("loop_back")
    W, BH, GAP, LX = 720, 46, 22, 34
    H = len(steps) * (BH + GAP) + 30
    out = [f'<svg viewBox="0 0 {W} {H}" width="100%" style="max-width:{W}px" '
           f'role="img" aria-label="算法流程图">',
           '<defs><marker id="fa" markerWidth="8" markerHeight="8" refX="7" refY="3" '
           'orient="auto"><path d="M0,0 L7,3 L0,6 z" fill="#0071e3"/></marker></defs>']
    for i, st in enumerate(steps):
        y = 10 + i * (BH + GAP)
        out.append(f'<rect class="fbx" x="{LX}" y="{y}" width="{W - LX - 150}" '
                   f'height="{BH}" rx="9"/>')
        out.append(f'<circle class="fno" cx="{LX + 16}" cy="{y + BH / 2:.0f}" r="11"/>')
        out.append(f'<text class="fnt" x="{LX + 16}" y="{y + BH / 2 + 4:.0f}">{i + 1}</text>')
        # **先去掉强调标记再截断。** 反过来的话截断可能把一对 ** 切成一半，
        # 剩下的那个就永远去不掉了。
        out.append(f'<text class="fti" x="{LX + 36}" y="{y + 19}">'
                   f'{_esc(_plain(st["title"]))}</text>')
        out.append(f'<text class="fds" x="{LX + 36}" y="{y + 35}">'
                   f'{_esc(_plain(st["desc"])[:74])}</text>')
        if i < len(steps) - 1:
            out.append(f'<line class="far" x1="{LX + 16}" y1="{y + BH}" '
                       f'x2="{LX + 16}" y2="{y + BH + GAP - 2}" marker-end="url(#fa)"/>')
        br = brs.get(i + 1)
        if br:
            bx = W - 146
            out.append(f'<line class="fbr" x1="{W - LX - 150 + LX}" y1="{y + BH / 2:.0f}" '
                       f'x2="{bx}" y2="{y + BH / 2:.0f}" marker-end="url(#fa)"/>')
            out.append(f'<text class="fbt" x="{bx + 4}" y="{y + BH / 2 - 2:.0f}">'
                       f'{_esc(br["cond"][:18])}</text>')
            out.append(f'<text class="fbd" x="{bx + 4}" y="{y + BH / 2 + 11:.0f}">'
                       f'{_esc(br["goto"][:20])}</text>')
    if lb:
        y0 = 10 + (lb["frm"] - 1) * (BH + GAP) + BH
        y1 = 10 + (lb["to"] - 1) * (BH + GAP)
        out.append(f'<path class="flb" d="M{LX} {y0 - BH / 2:.0f} H14 V{y1 + BH / 2:.0f} '
                   f'H{LX}" marker-end="url(#fa)"/>')
        out.append(f'<text class="fbd" x="4" y="{(y0 + y1) / 2:.0f}" '
                   f'transform="rotate(-90 4 {(y0 + y1) / 2:.0f})">'
                   f'{_esc(lb["desc"][:14])}</text>')
    out.append("</svg>")
    return "".join(out)


def _math(tex: str, *, display: bool = False) -> str:
    """一条公式。**KaTeX 排版，MathML 兜底。**

    KaTeX 靠 JS 在加载后渲染，脚本没跑起来时就只剩 ``data-tex`` 里的裸 LaTeX——
    所以容器内容先放 MathML：没有 JS 也看得到排好版的公式，
    **降级路径上任何一步都不会露出源码**。
    """
    from . import mathml as mm  # noqa: PLC0415

    return _kx.wrap(tex, mm.render(tex, block=display), display=display)


def _kx_credit() -> str:
    if not _kx.available():
        return ""
    m = _kx.meta()
    return (f' · 公式由内联 KaTeX {_esc(str(m.get("version", "")))} 排版'
            f'（MIT，离线可用）')


def _opt_html(o: dict[str, Any]) -> str:
    """一个可选实现的卡片。当前选中的高亮。"""
    cur = o.get("current")
    bits = [f'<div class="ohd"><span class="onm">{_esc(o["name"])}</span>'
            + ('<span class="ocur">当前采用</span>' if cur else "")
            + "</div>"]
    if o.get("summary"):
        bits.append(f'<p class="osum">{_bold(o["summary"])}</p>')
    if o.get("formula"):
        bits.append(f'<div class="ofml">{_math(o["formula"])}</div>')
    if o.get("detail"):
        bits.append(f"<p>{_bold(o['detail'])}</p>")
    kv = []
    for k, lbl in (("when", "什么时候用"), ("cost", "代价"), ("source", "依据")):
        if o.get(k):
            kv.append(f'<div class="okv"><b>{lbl}</b>{_bold(o[k])}</div>')
    bits.extend(kv)
    return f'<div class="opt{" ocurbox" if cur else ""}">{"".join(bits)}</div>'


def _families_panel(spec: dict[str, Any]) -> str:
    """算法页签正文：按阶段分组，每族列全部实现、标出当前、配流程图。"""
    from . import algo_defs as ad  # noqa: PLC0415

    fams = ad.families(spec["config"])
    out = ['<div class="callout c-blue"><p><b>这一页只读。</b>'
           '它交代的是后端<b>实际跑的是哪个算法、怎么算的</b>——'
           '想换算法请去「改配置」页签，那里改完点「应用到仿真」会直接回到 agent。</p>'
           '<p class="src">每族列出全部可选实现，<b>标着「当前采用」的才是这次真正在用的</b>。'
           '流程图是按代码实现画的，对不上就是 bug——欢迎照着挑。</p></div>']
    by_stage: dict[str, list[dict[str, Any]]] = {}
    for f in fams:
        by_stage.setdefault(f["stage"], []).append(f)
    for st in ad.STAGES:
        rows = by_stage.get(st)
        if not rows:
            continue
        out.append(f'<h3>{_esc(st)}</h3>')
        for f in rows:
            body = []
            if f.get("intro"):
                body.append(f'<p class="lead">{_bold(f["intro"])}</p>')
            if f.get("formula"):
                body.append(f'<div class="ofml big">{_math(f["formula"], display=True)}</div>')
            if f.get("flow"):
                body.append('<h4>算法流程</h4>')
                body.append(_flow_svg(f["flow"]))
            body.append(f'<h4>可选实现（{len(f["options"])} 个）</h4>')
            body.extend(_opt_html(o) for o in f["options"])
            if f.get("caveat"):
                body.append(f'<div class="callout c-amber"><p><b>什么时候会失真。</b>'
                            f'{_bold(f["caveat"])}</p></div>')
            if f.get("source"):
                body.append(f'<p class="src">依据：{_esc(f["source"])}</p>')
            if f.get("config_key"):
                body.append(f'<p class="src">对应配置项：<code>{_esc(f["config_key"])}</code>'
                            f'（在「改配置」页签里改）</p>')
            out.append(
                f'<details class="algo"><summary><span class="an">{_esc(f["name"])}</span>'
                f'<span class="ac">{_esc(f["current_name"])}</span>'
                f'<span class="acnt">{len(f["options"])} 个实现</span></summary>'
                f'<div class="ab">{"".join(body)}</div></details>')
    return "".join(out)


def _derivations_panel(spec: dict[str, Any]) -> str:
    """对标量的逐步推导，供人工核对。

    **只给一个「偏差 −1.2%」是不可核对的。** 每一步的输入、公式、中间结果
    全列出来，任何一步对不上都能当场指出。数字由代码现算，不是抄进来的常量。
    """
    from . import algorithms as alg  # noqa: PLC0415

    out: list[str] = ["<h3>对标量怎么算出来的</h3>"]
    for d in alg.derivations(spec["config"]):
        rows = "".join(
            f'<tr><td>{i}</td><td>{_esc(t)}</td><td><code>{_esc(f)}</code></td>'
            f'<td>{_bold(r)}</td></tr>'
            for i, (t, f, r) in enumerate(d["steps"], 1))
        out.append(
            f'<details class="algo"><summary><span class="an">{_esc(d["name"])}</span>'
            f'<span class="ac">{_esc(d["result"])}　参考 {_esc(d["reference"])}</span>'
            f'</summary><div class="ab">'
            f'<p class="src">参考出处：{_esc(d["ref_src"])}</p>'
            f'<div class="tbl-wrap"><table>'
            f'<tr><th>#</th><th>这一步在做什么</th><th>公式 / 输入</th><th>结果</th></tr>'
            f'{rows}</table></div></div></details>')

    fa = alg.FIELD_ANCHORS
    out.append(
        f'<div class="callout c-blue"><p><b>现网对标锚点。</b>'
        f'平均 MCS {fa["avg_mcs"]}（远点 {fa["avg_mcs_far"]}、近点 {fa["avg_mcs_near"]}）、'
        f'平均 rank {fa["avg_rank"]}。来源：{_esc(fa["source"])}。'
        f'<b>仿真结果明显偏离这几个数时，先怀疑口径而不是算法。</b></p></div>')
    return "".join(out)


def _svg_freq(spec: dict[str, Any]) -> str:
    """频域：RB 按 RBG 分组。"""
    f = spec["frequency"]
    n_rb, size, n_rbg = f["num_rb"], f["rbg_size"], f["num_rbg"]
    W, bh, pad = 900, 26, 4
    per = (W - 2 * pad) / max(n_rb, 1)
    out = [f'<svg viewBox="0 0 {W} 92" width="100%" role="img" aria-label="频域 RB 布局">']
    out.append('<style>.lb{font:11px ui-monospace,Consolas,monospace;fill:#6e6e73}'
               '.rb{fill:#0071e3;opacity:.75}.rb2{fill:#5a3ec8;opacity:.75}</style>')
    for g in range(n_rbg):
        lo = g * size
        n_in = min(size, n_rb - lo)
        x = pad + lo * per
        out.append(f'<rect class="{"rb" if g % 2 == 0 else "rb2"}" x="{x:.2f}" y="26" '
                   f'width="{max(n_in * per - 1, 0.5):.2f}" height="{bh}" rx="2"/>')
        if n_rbg <= 24:
            out.append(f'<text class="lb" x="{x + n_in * per / 2:.1f}" y="20" '
                       f'text-anchor="middle">{g}</text>')
    out.append(f'<text class="lb" x="{pad}" y="70">'
               f'{n_rb} RB = {n_rbg} RBG x {size} RB'
               f'{"（末组 " + str(n_rb - (n_rbg - 1) * size) + " RB）" if n_rb % size else ""}'
               f' · 每 RB 12 个子载波 x {f["scs_hz"]/1e3:g} kHz = '
               f'{f["occupied_hz"]/1e6:.2f} MHz 占用</text>')
    out.append(f'<text class="lb" x="{pad}" y="86">'
               f'载波 {f["carrier_freq_hz"]/1e9:g} GHz · 标称带宽 {f["bandwidth_hz"]/1e6:g} MHz'
               f' · 仿真粒度到 RB 为止（不建模到 RE）</text>')
    out.append("</svg>")
    return "".join(out)


def _svg_tdd(spec: dict[str, Any]) -> str:
    """TDD 时隙图案。"""
    t = spec["time"]
    pat = t["tdd_pattern"].upper()
    if not pat or any(c not in "DUS" for c in pat):
        return ""
    cw, W = 40, 40 * len(pat) + 20
    col = {"D": "#0071e3", "U": "#34c759", "S": "#ff9f0a"}
    out = [f'<svg viewBox="0 0 {W} 74" width="100%" style="max-width:{W}px" '
           f'role="img" aria-label="TDD 时隙图案">']
    out.append('<style>.lb{font:11px ui-monospace,Consolas,monospace;fill:#6e6e73}'
               '.sl{font:600 13px ui-monospace,Consolas,monospace;fill:#fff}</style>')
    for i, c in enumerate(pat):
        x = 10 + i * cw
        out.append(f'<rect x="{x}" y="12" width="{cw - 4}" height="30" rx="4" '
                   f'fill="{col[c]}" opacity=".88"/>')
        out.append(f'<text class="sl" x="{x + (cw - 4)/2}" y="33" text-anchor="middle">{c}</text>')
    out.append(f'<text class="lb" x="10" y="60">图案 {pat}（D 下行 / U 上行 / S 特殊）'
               f' · 每样本 {t["slots_per_sample"]} 时隙 x {t["symbols_per_slot"]} 符号</text>')
    out.append("</svg>")
    return "".join(out)


def _svg_pdp(spec: dict[str, Any]) -> str:
    """CDL/TDL 剖面的时延功率谱。"""
    name = str(spec["config"].get("channel_model", ""))
    if not name:
        return ""
    try:
        from .channelhub import cdl_profile  # noqa: PLC0415

        prof = cdl_profile(name)
        # 属性名是 delays_norm / powers_dB（不是 delays / powers_db）——
        # 早先写错名字时 except 把它吞成"没有可画的剖面"，图就这么静静少了一张。
        delays = [float(x) for x in prof.delays_norm]
        powers = [float(x) for x in prof.powers_dB]
    except (AttributeError, KeyError, TypeError, ValueError) as exc:
        return (f'<p class="src">取不到 {_esc(name)} 的时延功率谱：'
                f'{type(exc).__name__}: {_esc(exc)}</p>')
    if not delays:
        return ""

    W, H, pad = 900, 150, 34
    dmax = max(delays) or 1.0
    pmin = min(min(powers), -30.0)
    out = [f'<svg viewBox="0 0 {W} {H}" width="100%" role="img" aria-label="时延功率谱">']
    out.append('<style>.lb{font:11px ui-monospace,Consolas,monospace;fill:#6e6e73}'
               '.cl{stroke:#0071e3;stroke-width:2.2;opacity:.8}</style>')
    for d, p in zip(delays, powers, strict=False):
        x = pad + (d / dmax) * (W - 2 * pad)
        y = H - 34 - (p - pmin) / (0 - pmin) * (H - 62)
        out.append(f'<line class="cl" x1="{x:.1f}" y1="{H-34}" x2="{x:.1f}" y2="{y:.1f}"/>')
    out.append(f'<line x1="{pad}" y1="{H-34}" x2="{W-pad}" y2="{H-34}" '
               f'stroke="#d2d2d7" stroke-width="1"/>')
    out.append(f'<text class="lb" x="{pad}" y="{H-14}">0</text>'
               f'<text class="lb" x="{W-pad}" y="{H-14}" text-anchor="end">'
               f'归一化时延 {dmax:g}</text>')
    out.append(f'<text class="lb" x="{pad}" y="16">{_esc(name)} · {len(delays)} 簇 · '
               f'纵轴功率 dB（{pmin:g} ~ 0）</text>')
    out.append("</svg>")
    return "".join(out)


# ---------------------------------------------------------------------------
# HTML
# ---------------------------------------------------------------------------

_CSS = """
:root{--bg:#fafafa;--card:#fff;--ink:#1d1d1f;--ink-soft:#6e6e73;--border:#d2d2d7;
 --accent:#0071e3;--tint:#f2f2f4;--tint-blue:#eef5fd;--tint-blue-ink:#0055b3;
 --tint-amber:#fdf5e8;--tint-amber-ink:#8a5a00;--tint-red:#fdeeed;--tint-red-ink:#a32620;}
@media(prefers-color-scheme:dark){:root{--bg:#1d1d1f;--card:#2c2c2e;--ink:#f5f5f7;
 --ink-soft:#aeaeb2;--border:#38383a;--accent:#0a84ff;--tint:#333336;
 --tint-blue:#12263c;--tint-blue-ink:#8ec2ff;--tint-amber:#33280c;--tint-amber-ink:#ffd77a;
 --tint-red:#3a1a18;--tint-red-ink:#ff9a92;}}
*{box-sizing:border-box}
body{margin:0;padding:36px 20px;background:var(--bg);color:var(--ink);
 font-family:-apple-system,BlinkMacSystemFont,"PingFang SC","Microsoft YaHei",system-ui,sans-serif;
 font-size:15px;line-height:1.7}
.wrap{max-width:1000px;margin:0 auto}
h1{font-size:29px;font-weight:700;letter-spacing:-.03em;margin:0 0 6px}
h2{font-size:20px;font-weight:700;margin:40px 0 6px;padding-top:18px;border-top:1px solid var(--border)}
.meta{color:var(--ink-soft);font-size:12.5px;font-family:ui-monospace,Consolas,monospace;margin:0 0 6px}
.card{background:var(--card);border:1px solid var(--border);border-radius:13px;padding:20px;margin:16px 0}
.card>svg{display:block;margin:0 auto}
.tbl-wrap{overflow-x:auto;border:1px solid var(--border);border-radius:10px;margin:14px 0}
table{border-collapse:collapse;width:100%;font-size:13.5px;background:var(--card)}
th,td{padding:8px 12px;text-align:left;border-bottom:1px solid var(--border)}
th{background:var(--tint);color:var(--ink-soft);font-weight:600;font-size:12.5px;white-space:nowrap}
tr:last-child td{border-bottom:none}
code{font-family:ui-monospace,Consolas,monospace;background:var(--tint);padding:1px 5px;
 border-radius:4px;font-size:12.5px}
.pill{display:inline-block;font-size:11px;font-weight:600;padding:1px 8px;border-radius:20px}
.p-user{background:var(--tint-blue);color:var(--tint-blue-ink)}
.p-auto{background:var(--tint);color:var(--ink-soft)}
.callout{border-radius:10px;padding:13px 17px;margin:14px 0;border-left:3px solid}
.c-amber{background:var(--tint-amber);color:var(--tint-amber-ink);border-left-color:#ff9f0a}
.c-blue{background:var(--tint-blue);color:var(--tint-blue-ink);border-left-color:var(--accent)}
.c-red{background:var(--tint-red);color:var(--tint-red-ink);border-left-color:#ff3b30}
.src{font-family:ui-monospace,Consolas,monospace;font-size:12px;color:var(--ink-soft)}
.grid2{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:14px}
.kv{display:flex;justify-content:space-between;gap:12px;padding:4px 0;font-size:13.5px;
 border-bottom:1px dashed var(--border)}
.kv:last-child{border-bottom:none}
.kv b{font-weight:600;font-family:ui-monospace,Consolas,monospace}
math{font-size:1.06em;font-family:'Cambria Math','STIX Two Math','Latin Modern Math',serif}math[display='block']{display:block;margin:10px 0;text-align:left}.fml-raw{color:var(--tint-red-ink);background:var(--tint-red)}
footer{margin-top:44px;padding-top:18px;border-top:1px solid var(--border);
 color:var(--ink-soft);font-size:12.5px}

/* 分级呈现：拓扑图与关键信息在首屏，其余折进 tab。
   纯 CSS（radio + label），离线双击打开也能用，不依赖 JS。 */
.tabs{position:relative;margin:26px 0 0}
.tabs>input{position:absolute;opacity:0;width:0;height:0}
.tabs>label{display:inline-block;padding:9px 17px;cursor:pointer;font-size:14px;
 color:var(--ink-soft);border:1px solid transparent;border-bottom:none;
 border-radius:9px 9px 0 0;user-select:none;position:relative;z-index:1}
.tabs>label:hover{color:var(--ink)}
.tabs>input:checked+label{background:var(--card);border-color:var(--border);
 color:var(--accent);font-weight:600}
.tabs>input:focus-visible+label{outline:2px solid var(--accent);outline-offset:-2px}
.panels{border-top:1px solid var(--border);padding-top:16px;margin-top:-1px}
.panels>section{display:none;animation:fade .18s ease}
@keyframes fade{from{opacity:0}to{opacity:1}}
#tb1:checked~.panels>#pn1,#tb2:checked~.panels>#pn2,#tb3:checked~.panels>#pn3,
#tb4:checked~.panels>#pn4,#tb5:checked~.panels>#pn5,
#tb6:checked~.panels>#pn6,#tb7:checked~.panels>#pn7,
#tb6:checked~.panels>#pn6{display:block}

/* 关键信息卡 */
.facts{display:grid;grid-template-columns:repeat(auto-fit,minmax(148px,1fr));gap:10px;margin:16px 0}
.fact{background:var(--card);border:1px solid var(--border);border-radius:11px;padding:12px 14px}
.fact .n{font:700 19px ui-monospace,Consolas,monospace;letter-spacing:-.02em}
.fact .k{font-size:12px;color:var(--ink-soft);margin-top:2px}
.fact.hi{border-color:var(--accent);background:var(--tint-blue)}
.fact.hi .n{color:var(--tint-blue-ink)}
.fact.hi .k{color:var(--tint-blue-ink);opacity:.85}
.hero{background:var(--card);border:1px solid var(--border);border-radius:14px;
 padding:16px;margin:6px 0 0;text-align:center}
.hero>svg{display:block;margin:0 auto}
h3{font-size:15.5px;font-weight:600;margin:22px 0 4px}
.lead{color:var(--ink-soft);font-size:13.5px;margin:2px 0 10px}

/* 调参面板 */
.ctls{display:grid;grid-template-columns:repeat(auto-fit,minmax(210px,1fr));gap:12px;margin:14px 0 18px}
.ctl{display:block;background:var(--card);border:1px solid var(--border);
 border-radius:10px;padding:10px 12px}
.ctl .cl{display:block;font-size:12.5px;color:var(--ink-soft);margin-bottom:5px}
.ctl input,.ctl select{width:100%;font:14px ui-monospace,Consolas,monospace;
 padding:6px 8px;border:1px solid var(--border);border-radius:7px;
 background:var(--bg);color:var(--ink)}
.ctl input:focus,.ctl select:focus{outline:2px solid var(--accent);outline-offset:-1px}
.ctl .ch{font-size:11.5px;color:var(--ink-soft);margin-top:5px;opacity:.85}
.algo{border:1px solid var(--border);border-radius:10px;margin:8px 0;background:var(--card)}
.algo>summary{cursor:pointer;padding:11px 15px;display:flex;gap:12px;align-items:baseline;
 flex-wrap:wrap;list-style:none}
.algo>summary::-webkit-details-marker{display:none}
.algo>summary::before{content:"▸";color:var(--accent);font-size:11px;margin-right:2px}
.algo[open]>summary::before{content:"▾"}
.algo>summary:hover{background:var(--tint)}
.an{font-weight:600;min-width:150px}
.ac{color:var(--accent);font-size:13.5px}
.ab{padding:2px 15px 14px;border-top:1px solid var(--border)}
.fml{background:var(--tint);border-radius:8px;padding:10px 14px;margin:10px 0;
 font-family:ui-monospace,Consolas,monospace;font-size:12.5px;overflow-x:auto;white-space:pre-wrap}
.opt{border:1px solid var(--border);border-radius:9px;padding:13px 16px;margin:9px 0;background:var(--bg)}
.ocurbox{border-color:var(--accent);border-width:2px;background:var(--tint-blue)}
.ohd{display:flex;gap:10px;align-items:baseline;flex-wrap:wrap;margin-bottom:5px}
.onm{font-weight:600;font-size:14.5px}
.ocur{font-size:11px;font-weight:700;color:#fff;background:var(--accent);padding:2px 8px;border-radius:20px}
.osum{margin:4px 0;color:var(--ink-soft)}
.ofml{background:var(--tint);border-radius:8px;padding:9px 14px;margin:9px 0;overflow-x:auto}
.ofml.big{background:var(--tint-blue);padding:13px 18px}
.okv{font-size:13px;margin:5px 0;color:var(--ink-soft)}
.okv b{display:inline-block;min-width:78px;color:var(--ink)}
.acnt{font-size:11.5px;color:var(--ink-soft);margin-left:auto}
.fbx{fill:var(--card);stroke:var(--border);stroke-width:1}
.fno{fill:var(--accent)}
.fnt{font:700 11px ui-monospace,Consolas,monospace;fill:#fff;text-anchor:middle}
.fti{font:600 13px -apple-system,system-ui,sans-serif;fill:var(--ink)}
.fds{font:11.5px -apple-system,system-ui,sans-serif;fill:var(--ink-soft)}
.far{stroke:var(--accent);stroke-width:1.6}
.fbr{stroke:var(--warn,#ff9f0a);stroke-width:1.3;stroke-dasharray:4 3}
.flb{stroke:var(--accent);stroke-width:1.3;fill:none;stroke-dasharray:5 3}
.fbt{font:600 10.5px ui-monospace,Consolas,monospace;fill:#8a5a00}
.fbd{font:10px -apple-system,system-ui,sans-serif;fill:var(--ink-soft)}
.pvbar{display:flex;align-items:center;gap:10px;margin:14px 0 8px;flex-wrap:wrap}
/* 回执分两种状态：agent 正等着（绿）vs 已入收件箱、等它下次动作（琥珀）。
   对用户是完全不同的两件事，不能都用一样的灰字。 */
#msg{font-size:13px;line-height:1.5}
#msg.ok{color:var(--tint-blue-ink);font-weight:600}
#msg.warn{color:var(--tint-amber-ink);font-weight:600}
.btn{font:600 13.5px system-ui;padding:8px 16px;border-radius:9px;cursor:pointer;
 border:1px solid var(--accent);background:var(--accent);color:#fff}
.btn:hover{opacity:.9}
.btn.ghost{background:transparent;color:var(--ink-soft);border-color:var(--border)}
.pl{width:100%;font:12.5px ui-monospace,Consolas,monospace;padding:12px 14px;
 border:1px solid var(--border);border-radius:10px;background:var(--tint);
 color:var(--ink);resize:vertical}
"""


def _facts(spec: dict[str, Any], highlight: list[str] | None = None) -> list[dict[str, Any]]:
    """首屏关键信息卡。

    选什么摆首屏有两条依据：**做仿真通常最关心的**（规模、频点、带宽、阵列、
    场景），加上**这次对话里专门提到的**（``highlight`` 传进来的参数名，
    会被顶到最前面并高亮）。其余一律折进 tab。
    """
    a, f, t = spec["array"], spec["frequency"], spec["topology"]
    cfg = spec["config"]
    hl = set(highlight or [])
    m = a.get("elements_per_rf_port")
    n_ports = spec["panel"][0] * spec["panel"][1] * spec["panel"][2]

    items = [
        {"n": f'{t["num_cells"]}',
         "k": f'小区（{t["num_sites_actual"]} 站 x {t["sectors_per_site"]} 扇区）',
         "keys": ["num_sites", "sectors_per_site"]},
        {"n": f'{n_ports}T{cfg.get("num_ue_rx_ant", "")}R',
         "k": (f'1 驱 {m}、{a.get("physical_elements")} 阵子' if m else "独立阵元 legacy"),
         "keys": ["num_bs_tx_ant", "num_ue_rx_ant"]},
        {"n": f'{f["carrier_freq_hz"] / 1e9:g} GHz',
         "k": f'{f["bandwidth_hz"] / 1e6:g} MHz 带宽',
         "keys": ["carrier_freq_hz", "bandwidth_hz"]},
        {"n": f'{f["num_rb"]}',
         "k": f'RB（{f["num_rbg"]} x {f["rbg_size"]} RBG）',
         "keys": ["num_rb", "subcarrier_spacing"]},
        {"n": _esc(cfg.get("scenario", "-")),
         "k": f'场景 · {cfg.get("channel_model", "-")}',
         "keys": ["scenario", "channel_model"]},
    ]
    if t["isd_m"]:
        items.append({"n": f'{t["isd_m"]:g} m', "k": "站间距", "keys": ["isd_m"]})
    if cfg.get("num_ues"):
        items.append({"n": f'{cfg["num_ues"]}', "k": "每轮 UE 数", "keys": ["num_ues"]})
    if spec.get("num_samples"):
        items.append({"n": f'{spec["num_samples"]}', "k": "样本数", "keys": ["num_samples"]})
    if float(cfg.get("ue_speed_kmh", 0) or 0) > 3:
        items.append({"n": f'{float(cfg["ue_speed_kmh"]):g}', "k": "km/h 移动",
                      "keys": ["ue_speed_kmh", "mobility_mode"]})
    if cfg.get("num_interfering_ues"):
        items.append({"n": f'{cfg["num_interfering_ues"]}', "k": "每邻区干扰 UE",
                      "keys": ["num_interfering_ues"]})

    # 对话里点名过的参数：已有卡片就顶到最前并高亮，没有的补一张
    for it in items:
        it["hi"] = bool(hl & set(it["keys"]))
    known = {k for it in items for k in it["keys"]}
    for key in highlight or []:
        if key in known or key not in cfg:
            continue
        items.append({"n": _esc(_fmt(key, cfg[key])),
                      "k": _KEY_LABELS.get(key, (key, ""))[0] or key,
                      "keys": [key], "hi": True})
    items.sort(key=lambda it: not it["hi"])
    return items


def render_html(
    spec: dict[str, Any],
    ue_xy: list[tuple[float, float]] | None = None,
    highlight: list[str] | None = None,
    *,
    apply_url: str = "",
    spec_id: str = "",
) -> str:
    """渲染成分级呈现的单页：拓扑图打头，其余折进 tab。

    **信息分级，不是信息堆叠。** 首屏只放网络拓扑图 + 关键信息卡（做仿真通常
    最关心的那几个，加上这次对话里专门点过名的）；阵列细节、频域时域、信道
    剖面、参数全表折进 tab，用户按需点。

    tab 用纯 CSS（radio + label）实现，**离线双击打开也能用**，不依赖 JS。

    ``apply_url`` / ``spec_id`` 给调参面板用：有它们时页面能把改动直接 POST 回
    MCP 进程；没有（或用户从 ``file://`` 打开）时自动退回复制粘贴。
    """
    a = spec["array"]

    notes = "".join(
        f'<div class="callout {"c-red" if "**" in n else "c-amber"}">'
        f'<p>{_bold(n)}</p></div>'
        for n in spec["notes"]
    )

    # 没有真实撒点（比如直接从预设出说明书）就撒一批示意点——
    # 大片留白反而看不出这个配置要撒多少人、撒在哪。
    _ue_real = bool(ue_xy)
    _ue_pts = list(ue_xy) if ue_xy else planned_ue_drop(spec["config"])

    pill_user = '<span class="pill p-user">用户指定</span>'
    pill_auto = '<span class="pill p-auto">默认</span>'

    def rows(items):
        return "".join(
            f'<tr><td>{_esc(r["label"] or r["key"])}<br>'
            f'<span class="src">{_esc(r["key"])}</span></td>'
            f'<td><b>{_esc(r["value"])}</b></td>'
            f'<td>{pill_user if r["by_user"] else pill_auto}</td></tr>'
            for r in items
        )

    facts = "".join(
        f'<div class="fact{" hi" if it["hi"] else ""}">'
        f'<div class="n">{it["n"]}</div><div class="k">{_esc(it["k"])}</div></div>'
        for it in _facts(spec, highlight)
    )

    n_user = sum(1 for r in spec["params"] + spec["other_params"] if r["by_user"])
    n_all = len(spec["params"]) + len(spec["other_params"])

    array_kv = "".join(
        f'<div class="kv"><span>{k}</span><b>{_esc(v)}</b></div>'
        for k, v in [
            ("RF 端口", f'{spec["panel"][0]}H x {spec["panel"][1]}V x {spec["panel"][2]}pol'
                        f' = {spec["panel"][0] * spec["panel"][1] * spec["panel"][2]}'),
            ("物理阵子", a.get("physical_elements") or "同端口数（legacy 独立阵元）"),
            ("馈电", f'1 驱 {a["elements_per_rf_port"]}'
                     if a.get("elements_per_rf_port") else "无子阵"),
            ("水平间距", f'{a.get("horizontal_spacing_lambda", 0.5):g}λ'),
            ("垂直间距", f'{a.get("ae_vertical_spacing_lambda", 0.5):g}λ'),
            ("端口相位中心", f'{a.get("rf_vertical_spacing_lambda"):g}λ'
                             if a.get("rf_vertical_spacing_lambda") else "-"),
            ("固定下倾", f'{a.get("fixed_downtilt_deg", 0):g}°'),
            ("模型", a.get("antenna_model_mode")),
        ] if v not in (None, "-")
    )

    fq, tm = spec["frequency"], spec["time"]
    freq_kv = "".join(
        f'<div class="kv"><span>{k}</span><b>{_esc(v)}</b></div>' for k, v in [
            ("载波", f'{fq["carrier_freq_hz"] / 1e9:g} GHz'),
            ("标称带宽", f'{fq["bandwidth_hz"] / 1e6:g} MHz'),
            ("子载波间隔", f'{fq["scs_hz"] / 1e3:g} kHz'),
            ("RB 数", fq["num_rb"]),
            ("RBG", f'{fq["num_rbg"]} 组 x {fq["rbg_size"]} RB'),
            ("实际占用", f'{fq["occupied_hz"] / 1e6:.2f} MHz'),
            ("仿真粒度", "到 RB 为止（不建模到 RE）"),
            ("TDD 图案", tm["tdd_pattern"]),
            ("每样本时隙", tm["slots_per_sample"]),
            ("每时隙符号", tm["symbols_per_slot"]),
        ]
    )

    meta_ds = f' · 数据集 {_esc(spec["dataset_id"])}' if spec.get("dataset_id") else ""
    meta_n = f' · {spec["num_samples"]} 个样本' if spec.get("num_samples") else ""

    return f"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{_esc(spec["title"])}</title><style>{_CSS}</style>
{_kx.head_assets()}</head>
<body><div class="wrap">

<h1>{_esc(spec["title"])}</h1>
<p class="meta">{_esc(spec["created_at"])} · 引擎 {_esc(spec["source"])}{meta_ds}{meta_n}</p>

{notes}

<div class="hero">{_svg_layout(spec, _ue_pts, ue_is_real=_ue_real)}</div>
<div class="facts">{facts}</div>

<div class="tabs">
<input type="radio" name="tb" id="tb1" checked><label for="tb1">总览</label
><input type="radio" name="tb" id="tb2"><label for="tb2">改配置</label
><input type="radio" name="tb" id="tb3"><label for="tb3">基站阵列</label
><input type="radio" name="tb" id="tb4"><label for="tb4">频域与时域</label
><input type="radio" name="tb" id="tb5"><label for="tb5">信道剖面</label
><input type="radio" name="tb" id="tb6"><label for="tb6">算法</label><input type="radio" name="tb" id="tb7"><label for="tb7">参数全表</label>
<div class="panels">

<section id="pn1">
<h3>这次在仿什么</h3>
<p class="lead">{_esc(headline(spec))}</p>
<p class="src">上面这张图和这些数字描述的是<b>将要跑（或已经跑过）的那个仿真</b>，
不是配置意图——站数被栅格吸附、阵列走了 legacy 之类的差异都按实际画，
有出入会在页首用黄框标出来。</p>
<h3>想看细节点上面的页签</h3>
<div class="tbl-wrap"><table>
<tr><th>页签</th><th>回答什么问题</th></tr>
<tr><td><b>改配置</b></td><td><b>直接在这页改参数、实时看拓扑，一键复制粘回对话框让 agent 重跑</b></td></tr>
<tr><td>基站阵列</td><td>端口怎么排、每端口驱动几个阵子、间距多少、有没有栅瓣</td></tr>
<tr><td>频域与时域</td><td>多少 RB、怎么分 RBG、占多宽、TDD 怎么配</td></tr>
<tr><td>信道剖面</td><td>多径的时延与功率分布</td></tr>
<tr><td>参数全表</td><td>{n_all} 项配置逐条列出，标明哪些是你定的、哪些是系统补的</td></tr>
</table></div>
</section>

<section id="pn2">
{_interactive(spec, apply_url=apply_url, spec_id=spec_id)}
</section>

<section id="pn3">
<div class="hero">{_svg_array(spec)}</div>
<div class="grid2"><div class="card">{array_kv}</div>
<div class="card"><p class="src">{_bold(a.get("note", ""))}</p></div></div>
</section>

<section id="pn4">
<div class="card">{_svg_freq(spec)}</div>
<div class="card">{_svg_tdd(spec) or '<p class="src">无 TDD 图案。</p>'}</div>
<div class="card">{freq_kv}</div>
</section>

<section id="pn5">
<div class="card">{_svg_pdp(spec) or '<p class="src">该模型没有可画的时延功率谱。</p>'}</div>
<p class="src">CDL 系列每条径带角度（AoD/AoA/ZoD/ZoA），TDL 系列没有——
凡是依赖角度的课题（波束管理、定位）必须用 CDL。</p>
</section>

<section id="pn6">
{_families_panel(spec)}
{_derivations_panel(spec)}
</section>

<section id="pn7">
<p class="lead">{n_user}/{n_all} 项由用户指定，其余走默认值。
<b>标着「默认」的都是系统替你定的</b>，不认可就改。</p>
<div class="tbl-wrap"><table>
<tr><th>参数</th><th>值</th><th>来源</th></tr>
{rows(spec["params"])}{rows(spec["other_params"])}
</table></div>
</section>

</div></div>

<footer>superwireless 仿真说明书 · 图与数均由本次配置生成，未经手工编辑{_kx_credit()}</footer>
</div>
{_kx.upgrade_script()}
</body></html>
"""


def write_spec(
    cfg: dict[str, Any],
    *,
    num_samples: int | None = None,
    user_set: list[str] | None = None,
    dataset_id: str | None = None,
    title: str = "",
    ue_xy: list[tuple[float, float]] | None = None,
    highlight: list[str] | None = None,
    serve: bool = True,
    open_browser: bool = False,
) -> dict[str, Any]:
    """生成说明书并落盘，返回结构化摘要 + 可点开的地址。

    **默认不替用户弹窗。** 给地址，他自己在浏览器或 AI HUB 里点开——
    自动弹窗对一部分人是打断而不是便利。要弹显式传 ``open_browser=True``。

    HTML 落到 ``artifacts/specs/``；**对话里只回路径与摘要**，
    不要把图或整份 HTML 贴回去。

    ``highlight`` 传本次对话里用户**专门提过**的参数名（如
    ``["isd_m", "num_interfering_ues"]``），它们会被顶到首屏关键信息卡的最前面
    并高亮——首屏因此既覆盖"做仿真通常最关心的"，也覆盖"这次特别在意的"。

    ``serve=True`` 时把页面同时挂到 ``127.0.0.1`` 的环回服务上并返回 ``url``，
    这样调参面板能把改动**直接 POST 回来**（``bridge.await_submission`` 接）。
    起不来就退回 ``file://``，页面自动降级成复制粘贴——**降级要能看见**，
    返回值里的 ``serve_error`` 会说明原因。
    """
    spec = build_spec(cfg, num_samples=num_samples, user_set=user_set,
                      dataset_id=dataset_id, title=title)
    out_dir = artifacts_root() / "specs"
    out_dir.mkdir(parents=True, exist_ok=True)
    # **秒级时间戳不够唯一。** 同一秒里连着出两份说明书（先看预设、再看草稿），
    # 后一份会把前一份直接覆盖掉，而且不报错——用户拿到的路径指向的是别人的图。
    # 实测就这么丢过一份：HST 的拓扑图被单小区的覆盖成了一个孤零零的点。
    # 数据集句柄本身唯一，其余情况补一段随机后缀。
    stem = dataset_id or f"{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"

    # 回传地址要**先拿到再渲染**——它是要写进页面 JS 的。
    apply_url, serve_error = "", ""
    if serve:
        if not br.enabled():
            serve_error = "SUPERWIRELESS_NO_SERVE 已关闭回传服务"
        elif br.start() is None:
            serve_error = "环回服务起不来（端口被占或被沙箱限制）"
        else:
            apply_url = br.apply_url() or ""

    html = render_html(spec, ue_xy, highlight, apply_url=apply_url, spec_id=stem)
    path = out_dir / f"spec-{stem}.html"
    path.write_text(html, encoding="utf-8")

    url = br.serve(stem, html, title=spec["title"],
                   allowed=editable_keys()) if apply_url else None
    opened = br.open_url(url or str(path)) if open_browser else False

    (out_dir / f"spec-{stem}.json").write_text(
        json.dumps({k: v for k, v in spec.items() if k != "config"},
                   ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    return {
        "html_path": str(path),
        "url": url,
        "spec_id": stem,
        "opened_in_browser": opened,
        "writeback": "post" if url else "clipboard",
        "serve_error": serve_error or None,
        "json_path": str(out_dir / f"spec-{stem}.json"),
        "headline": headline(spec),
        "notes": spec["notes"],
        "array": spec["array"],
        "topology": {k: v for k, v in spec["topology"].items() if k != "cells"},
        "frequency": spec["frequency"],
        "time": spec["time"],
        "num_params": len(spec["params"]) + len(spec["other_params"]),
        "num_user_set": sum(1 for r in spec["params"] + spec["other_params"] if r["by_user"]),
    }


def headline(spec: dict[str, Any]) -> str:
    """一句话概括这次仿真。给对话里口头转述用。"""
    a, f, t = spec["array"], spec["frequency"], spec["topology"]
    m = a.get("elements_per_rf_port")
    n_ports = spec["panel"][0] * spec["panel"][1] * spec["panel"][2]
    arr = (f"{n_ports}T（{spec['panel'][0]}x{spec['panel'][1]}x{spec['panel'][2]}"
           + (f"，1 驱 {m}、{a.get('physical_elements')} 阵子" if m else "，独立阵元")
           + "）")
    topo = (f"{t['num_sites_actual']} 站 x {t['sectors_per_site']} 扇区 = "
            f"{t['num_cells']} 小区" + (f"、站间距 {t['isd_m']:g} m" if t["isd_m"] else ""))
    return (f"{arr} · {topo} · {f['carrier_freq_hz']/1e9:g} GHz / "
            f"{f['scs_hz']/1e3:g} kHz / {f['num_rb']} RB "
            f"({f['num_rbg']}x{f['rbg_size']}) · {spec['config'].get('scenario','')} "
            f"{spec['config'].get('channel_model','')}")
