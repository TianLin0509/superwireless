"""本地默认硬件与载波配置：64T 1驱3 AAU + 2.6 GHz / 100 MHz NR。

这个文件是**默认信道配置的唯一真相源**。之前 superwireless 一直走 ChannelHub
的 ``antenna_model_mode="legacy_64"``——把 64 个端口当成 64 个**独立**阵元、
间距一律 0.5λ。真实硬件不是这样：

===============================  ==========================================
项                                真实 AAU
===============================  ==========================================
RF / 数字端口                      8H x 4V x 2pol = **64**
物理阵子                           8H x 12V x 2pol = **192**
馈电                              每个 RF 端口固定驱动同一 (h, pol) 列上
                                  **垂直相邻的 3 个阵子**（1 驱 3）
水平阵子间距                       **0.5λ**
垂直阵子间距                       **0.67λ**（不是 0.5λ）
RF 端口垂直相位中心间距             3 x 0.67 = **2.01λ**（> λ，有栅瓣）
===============================  ==========================================

ChannelHub 的 ``phy_sim/effective_array.py`` 就是照这套硬件写的
（模块文档里 "Target AAU" 一节逐条对得上），只是默认没启用。启用它要两件事：
``antenna_model_mode="effective_subarray"`` 加一个 ``bs_antenna`` 配置块。

**实测影响**（同 seed、64T/4R、2.6 GHz、272 RB）：

* ``h_serving_true`` 与 legacy 的**相对差 4.03**——完全是另一个信道。
  所有从信道算出来的量（预编码、谱效、吞吐、CSI 压缩）都跟着变。
* ``effective_subarray`` 与 ``physical_reference``（真跑 192 阵子再用 F 投影）
  的相对差 **4.8e-7**——快路径复现了参考路径，可以放心用快的。
* **几何 SINR / SIR / IoT 逐位不变**。这条必须说清楚：ChannelHub 的几何
  SINR 走 ``_system_sinr.py`` 自己那套简化模型（水平 0.5λ DFT 码本、
  垂直方向完全平坦），**不读这里的阵列模型**。所以换阵列模型不会改变
  干扰画像，只改信道矩阵。

垂直 0.67λ 这个数是用户实测纠正过的值（早期按 0.5λ 算，全盘产物失真），
见记忆 ``project_reconfig_mimo_sim`` 的方法论教训第 4 条。**别改回 0.5。**

--- 载波与带宽 ---------------------------------------------------------

面向 5G，n41 频段 2.6 GHz、30 kHz 子载波间隔、100 MHz 带宽。

RB 数用 **272**（17 个 RBG x 每 RBG 16 个 RB）。注意 3GPP 38.104 的标准表
在 100 MHz / 30 kHz 下给的是 **273**——272 是按 RBG 对齐取的整数倍，
差的那 1 个 RB 在 RBG 划分里本来就是残块。两个数都对，只是口径不同，
所以这里显式写死 272 而不是让它走标准表。

仿真粒度到 **RB 为止**：每个 RB 有 12 个子载波（RE），但 RE 级建模的复杂度
换不来系统级结论的精度，所以信道矩阵的频率轴就是 272 个 RB。
"""
from __future__ import annotations

from typing import Any

# --- 阵列 -----------------------------------------------------------------

COMPANY_RF_PANEL: list[int] = [8, 4, 2]          # N_H, N_V, N_pol -> 64 端口
COMPANY_ELEMENTS_PER_PORT = 3                     # 1 驱 3
COMPANY_H_SPACING_LAMBDA = 0.5
COMPANY_V_SPACING_LAMBDA = 0.67                   # 实测值，别改回 0.5
COMPANY_NUM_PORTS = COMPANY_RF_PANEL[0] * COMPANY_RF_PANEL[1] * COMPANY_RF_PANEL[2]
COMPANY_NUM_ELEMENTS = COMPANY_NUM_PORTS * COMPANY_ELEMENTS_PER_PORT   # 192

# --- 载波 -----------------------------------------------------------------

COMPANY_CARRIER_HZ = 2.6e9        # n41
COMPANY_SCS_HZ = 30_000
COMPANY_BANDWIDTH_HZ = 100e6
COMPANY_NUM_RBG = 17
COMPANY_RB_PER_RBG = 16
COMPANY_NUM_RB = COMPANY_NUM_RBG * COMPANY_RB_PER_RBG   # 272
COMPANY_SC_PER_RB = 12            # 只作记录，仿真到 RB 为止
NR_TABLE_NUM_RB_100M_30K = 273    # 38.104 标准表值，与上面的 272 口径不同

# --- 收发 -----------------------------------------------------------------

COMPANY_UE_RX_ANT = 4             # 默认 4R 接收
COMPANY_UE_TX_ANT = 2
COMPANY_LINK = "DL"               # 默认下行


def company_antenna_block(
    *,
    carrier_freq_hz: float = COMPANY_CARRIER_HZ,
    fixed_downtilt_deg: float = 6.0,
) -> dict[str, Any]:
    """ChannelHub ``bs_antenna`` 配置块（1 驱 3、0.5λ/0.67λ）。

    ``fixed_downtilt_deg`` 是**馈电网络内部**的固定电下倾，正值把主瓣压到
    水平面以下。它做在子阵内部，所有端口共用同一套馈电，改它等于换硬件
    校准版本——所以 ``calibration_id`` 跟着带上。
    """
    return {
        "horizontal_port_spacing_lambda": COMPANY_H_SPACING_LAMBDA,
        "reference_frequency_hz": float(carrier_freq_hz),
        "element_pattern": {
            # parametric_temporary 是 3GPP 式的抛物线幅度模型，**不是实测方向图**。
            # ChannelHub 会把这一点写进 element_pattern_is_measured=False，
            # 对外说明时不能说成实测。
            "source": "parametric_temporary",
            "horizontal_hpbw_deg": 65.0,
            "vertical_hpbw_deg": 65.0,
            "peak_gain_dbi": 8.0,
            "xpd_db": 8.0,
        },
        "fixed_vertical_subarray": {
            "elements_per_rf_port": COMPANY_ELEMENTS_PER_PORT,
            "ae_vertical_spacing_lambda": COMPANY_V_SPACING_LAMBDA,
            "fixed_downtilt_deg": float(fixed_downtilt_deg),
            "calibration_id": "company-64T-1to3-192ae-v1",
        },
    }


def is_company_panel(panel: Any) -> bool:
    """这个面板是不是 64T 的 8x4x2。

    1 驱 3 是**这一款 AAU** 的硬件事实，不是通用规律。16T / 256T 之类的
    面板套上去只会得到一个查无实据的阵列——所以只对 8x4x2 生效。
    """
    try:
        p = [int(x) for x in panel]
    except (TypeError, ValueError):
        return False
    return p == COMPANY_RF_PANEL


def apply_array_defaults(cfg: dict[str, Any]) -> dict[str, Any]:
    """给配置补上真实阵列模型。**就地修改并返回同一个 dict。**

    只在三个条件都满足时才动手：

    1. 调用方没有显式指定 ``antenna_model_mode``（显式指定的一律尊重）；
    2. 面板是 64T 的 8x4x2（见 :func:`is_company_panel`）；
    3. 没有已存在的 ``bs_antenna`` 块。

    返回的 dict 里带一个 ``_array_defaults_applied`` 标记供上层记录；
    这个键在下发给 ChannelHub 之前会被 :func:`strip_markers` 摘掉。
    """
    if cfg.get("antenna_model_mode"):
        cfg["_array_defaults_applied"] = "explicit"
        return cfg
    panel = cfg.get("bs_panel")
    if not is_company_panel(panel):
        cfg["_array_defaults_applied"] = "skipped_non_64t"
        return cfg
    if cfg.get("bs_antenna"):
        cfg["_array_defaults_applied"] = "explicit_bs_antenna"
        return cfg

    cfg["antenna_model_mode"] = "effective_subarray"
    cfg["bs_antenna"] = company_antenna_block(
        carrier_freq_hz=float(cfg.get("carrier_freq_hz") or COMPANY_CARRIER_HZ)
    )
    cfg["_array_defaults_applied"] = "company_1to3_192ae"
    return cfg


def strip_markers(cfg: dict[str, Any]) -> str | None:
    """摘掉内部标记键并返回它的值。ChannelHub 不认识这个键。"""
    return cfg.pop("_array_defaults_applied", None)


def array_summary(cfg: dict[str, Any], applied: str | None) -> dict[str, Any]:
    """给 summary.json 用的阵列口径说明。"""
    mode = cfg.get("antenna_model_mode", "legacy_64")
    out: dict[str, Any] = {
        "antenna_model_mode": mode,
        "applied": applied,
        "bs_panel": list(cfg.get("bs_panel") or []),
    }
    if mode == "legacy_64":
        out["note"] = (
            "64 个端口按**独立阵元**建模、间距一律 0.5λ —— 这不是真实 AAU。"
            "真实硬件是 1 驱 3、192 阵子、垂直 0.67λ。"
            "面板为 8x4x2 时会自动切到真实模型；其他面板保持 legacy。"
        )
        return out
    ant = cfg.get("bs_antenna") or {}
    sub = ant.get("fixed_vertical_subarray") or {}
    m = int(sub.get("elements_per_rf_port", COMPANY_ELEMENTS_PER_PORT))
    dv = float(sub.get("ae_vertical_spacing_lambda", COMPANY_V_SPACING_LAMBDA))
    out.update({
        "elements_per_rf_port": m,
        "physical_elements": (
            int(cfg["bs_panel"][0]) * int(cfg["bs_panel"][1]) * int(cfg["bs_panel"][2]) * m
            if cfg.get("bs_panel") else None
        ),
        "horizontal_spacing_lambda": float(
            ant.get("horizontal_port_spacing_lambda", COMPANY_H_SPACING_LAMBDA)
        ),
        "ae_vertical_spacing_lambda": dv,
        "rf_vertical_spacing_lambda": round(m * dv, 4),
        "fixed_downtilt_deg": float(sub.get("fixed_downtilt_deg", 0.0)),
        "calibration_id": sub.get("calibration_id"),
        "element_pattern_is_measured": False,
        "note": (
            "真实 AAU：1 驱 3、192 物理阵子、水平 0.5λ / 垂直 0.67λ。"
            "**几何 SINR / IoT 不受它影响**——ChannelHub 的几何 SINR 走另一套"
            "简化模型（水平 0.5λ、垂直平坦），阵列模型只改信道矩阵，"
            "因而只影响预编码/谱效/吞吐/CSI 这一类量。"
            "阵元方向图是 3GPP 式参数化模型，不是实测方向图。"
        ),
    })
    return out


# --- 全套默认 -------------------------------------------------------------

def company_carrier_defaults() -> dict[str, Any]:
    """载波侧默认值。只在调用方没写的键上生效。"""
    return {
        "carrier_freq_hz": COMPANY_CARRIER_HZ,
        "subcarrier_spacing": COMPANY_SCS_HZ,
        "bandwidth_hz": COMPANY_BANDWIDTH_HZ,
        "num_rb": COMPANY_NUM_RB,
        "num_bs_tx_ant": COMPANY_NUM_PORTS,
        "num_bs_rx_ant": COMPANY_NUM_PORTS,
        "num_ue_rx_ant": COMPANY_UE_RX_ANT,
        "num_ue_tx_ant": COMPANY_UE_TX_ANT,
        "bs_panel": list(COMPANY_RF_PANEL),
        "link": COMPANY_LINK,
    }


def describe() -> dict[str, Any]:
    """人可读的默认配置说明，给 sw_capabilities / 文档用。"""
    return {
        "array": {
            "rf_ports": COMPANY_NUM_PORTS,
            "rf_shape": f"{COMPANY_RF_PANEL[0]}H x {COMPANY_RF_PANEL[1]}V x "
                        f"{COMPANY_RF_PANEL[2]}pol",
            "physical_elements": COMPANY_NUM_ELEMENTS,
            "feed": f"1 驱 {COMPANY_ELEMENTS_PER_PORT}（每端口驱动垂直相邻 "
                    f"{COMPANY_ELEMENTS_PER_PORT} 个阵子）",
            "horizontal_spacing_lambda": COMPANY_H_SPACING_LAMBDA,
            "ae_vertical_spacing_lambda": COMPANY_V_SPACING_LAMBDA,
            "rf_vertical_spacing_lambda": round(
                COMPANY_ELEMENTS_PER_PORT * COMPANY_V_SPACING_LAMBDA, 4),
            "grating_lobe": "RF 端口垂直间距 2.01λ > λ，垂直方向有栅瓣",
        },
        "carrier": {
            "band": "n41",
            "carrier_freq_hz": COMPANY_CARRIER_HZ,
            "subcarrier_spacing_hz": COMPANY_SCS_HZ,
            "bandwidth_hz": COMPANY_BANDWIDTH_HZ,
            "num_rb": COMPANY_NUM_RB,
            "rbg_layout": f"{COMPANY_NUM_RBG} RBG x {COMPANY_RB_PER_RBG} RB",
            "nr_table_num_rb": NR_TABLE_NUM_RB_100M_30K,
            "num_rb_note": (
                f"272 = {COMPANY_NUM_RBG} x {COMPANY_RB_PER_RBG}（按 RBG 对齐）；"
                f"38.104 标准表在 100 MHz/30 kHz 下是 {NR_TABLE_NUM_RB_100M_30K}。"
                "两个数口径不同，这里显式用 272。"
            ),
            "granularity": f"仿真到 RB 为止（每 RB {COMPANY_SC_PER_RB} 个子载波，不建模到 RE）",
        },
        "link": {
            "direction": COMPANY_LINK,
            "ue_rx_ant": COMPANY_UE_RX_ANT,
            "ue_tx_ant": COMPANY_UE_TX_ANT,
        },
    }
