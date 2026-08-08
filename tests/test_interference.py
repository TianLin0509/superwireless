"""干扰强度量化的测试：IoT 推导、分级、测量域、场景预设。

直接运行：python tests/test_interference.py
"""
from __future__ import annotations

import math
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

# 说明书默认会弹浏览器——跑测试时不要。**必须在 import spec 之前设**，
# 不然 write_spec 一调就是十几个窗口糊满屏幕。
os.environ["SUPERWIRELESS_NO_BROWSER"] = "1"

# Windows 中文控制台是 GBK，统一兜底（见 test_gates.py 同样的处理）。
sys.stdout.reconfigure(errors="replace")

from superwireless import channelhub as ch  # noqa: E402
from superwireless import generate as gen  # noqa: E402
from superwireless import interference as itf  # noqa: E402
from superwireless import load  # noqa: E402
from superwireless import plan as pl  # noqa: E402

FAILED: list[str] = []


def check(cond: bool, label: str) -> None:
    print(("  PASS  " if cond else "  FAIL  ") + label)
    if not cond:
        FAILED.append(label)


def sect(t: str) -> None:
    print("\n" + "=" * 70 + f"\n{t}\n" + "=" * 70)


# ---------------------------------------------------------------------------
sect("1  IoT 推导的解析正确性")

# 直接构造 S/I/N，三个量都算得出来，检查 iot_db 与解析值逐位相符。
cases = [
    (1.0, 10.0, 1.0), (1.0, 0.01, 1.0), (1.0, 1.0, 1.0),
    (10.0, 3.0, 0.5), (100.0, 0.001, 1.0), (1.0, 1000.0, 1.0),
]
worst = 0.0
for S, I_pow, N in cases:
    sinr = 10 * math.log10(S / (I_pow + N))
    sir = 10 * math.log10(S / I_pow)
    want = 10 * math.log10((I_pow + N) / N)
    got = float(itf.iot_db(sinr, sir))
    worst = max(worst, abs(want - got))
print(f"  {len(cases)} 组 (S,I,N) 的最大偏差 {worst:.2e} dB")
check(worst < 1e-9, "IoT = SIR/(SIR-SINR) 与 (I+N)/N 解析一致")

# 这条是本模块存在的理由：**不能**用 snr - sinr。
# 构造一个 ChannelHub 口径的例子：snr 不含阵列增益且多减了 10log10(RB)。
RB, N_ant = 273, 64
S, I_pow, N = 1.0, 10.0, 1.0
sinr_ch = 10 * math.log10(S * N_ant / (I_pow * N_ant + N))   # 几何 SINR：含阵列增益
snr_ch = 10 * math.log10(S / N) - 10 * math.log10(RB)     # ChannelHub 的 snr_dB
naive = snr_ch - sinr_ch
true_iot = 10 * math.log10((I_pow * N_ant + N) / N)
print(f"  naive(snr-sinr) = {naive:.2f} dB，真值 = {true_iot:.2f} dB，"
      f"差 {abs(naive - true_iot):.1f} dB")
check(abs(naive - true_iot) > 20.0,
      "snr_dB - sinr_dB 与真 IoT 相差 20 dB 以上（所以模块里禁用这个式子）")

# 向量化与广播
v = itf.iot_db(np.array([-5.0, 0.0, 5.0]), np.array([0.0, 5.0, 10.0]))
check(v.shape == (3,) and np.all(np.isfinite(v)), "支持数组输入")
check(np.isnan(itf.iot_db(np.nan, 10.0)), "输入非有限值时回 nan 而不是瞎算")
check(math.isinf(float(itf.iot_db(10.0, 10.0))),
      "SIR == SINR（噪声为零）时回 inf，不是回 0")
check(math.isinf(float(itf.iot_db(10.0, 5.0))),
      "SIR < SINR（物理不可能）时回 inf 而不是负数")

# ---------------------------------------------------------------------------
sect("2  负载换算与分级")

for load_v, want_db in ((0.5, 3.01), (0.75, 6.02), (0.9, 10.0), (0.99, 20.0)):
    got = itf.iot_from_load(load_v)
    check(abs(got - want_db) < 0.02, f"负载 {load_v} -> IoT {want_db} dB（得 {got:.2f}）")

rt = float(itf.load_factor_from_iot(itf.iot_from_load(0.87)))
check(abs(rt - 0.87) < 1e-9, "负载 <-> IoT 往返一致")

check(itf.classify_iot(22.0)["high_interference"] is True, "22 dB 判为高干扰")
check(itf.classify_iot(19.9)["high_interference"] is False, "19.9 dB 不判为高干扰")
check(itf.classify_iot(20.0)["high_interference"] is True, "门限 20 dB 取闭区间")
check(itf.classify_iot(float("nan"))["band"] == "未定义", "非有限值不硬套等级")
bands = [itf.classify_iot(x)["band"] for x in (1, 5, 10, 17, 25)]
print("  1/5/10/17/25 dB -> " + " ".join(bands))
check(len(set(bands)) == 5, "五个档位互不重叠")

# ---------------------------------------------------------------------------
sect("3  IoT 统计：不可信样本必须单独计数")

# 哨兵值：没有干扰源时 ChannelHub 填 sir_dB = 49.9
st = itf.iot_stats(np.array([20.0, 21.0, 22.0]), np.array([49.9, 49.9, 49.9]))
check(st.n_no_interferer == 3 and st.n_valid == 0,
      "sir=49.9 哨兵全部计入 n_no_interferer，不进 IoT 统计")

# 贴边：SINR 顶到 +50 dB
st = itf.iot_stats(np.array([50.0, 10.0]), np.array([55.0, 20.0]))
check(st.n_clamped == 1 and st.n_valid == 1, "贴 ±50 dB 边界的样本单独计数")

# 正常样本
sinr = np.array([0.0, 1.0, 2.0, 3.0])
sir = sinr + 0.5
st = itf.iot_stats(sinr, sir)
check(st.n_valid == 4 and st.median_db > 0, "正常样本全部有效")
check(sum(st.bands.values()) == st.n_valid, "分档计数之和等于有效样本数")

nan_st = itf.iot_stats(np.array([np.nan, np.nan]), np.array([np.nan, np.nan]))
check(nan_st.n_valid == 0 and nan_st.as_dict()["median_db"] is None,
      "全 nan 时不抛异常、中位数回 None")

# ---------------------------------------------------------------------------
sect("4  测量域：估计 NMSE 下限")

# 纯干扰受限：NMSE 底 = 1/SIR
floor = float(itf.estimation_nmse_floor_db(20.0))
check(abs(floor + 20.0) < 1e-9, "SIR 20 dB -> NMSE 底 -20 dB")
# 干扰 + 噪声：两项功率相加
floor2 = float(itf.estimation_nmse_floor_db(20.0, 20.0))
check(abs(floor2 - (-20.0 + 10 * math.log10(2))) < 1e-9,
      "同量级噪声让 NMSE 底抬高 3 dB")
check(float(itf.estimation_nmse_floor_db(5.0)) > float(itf.estimation_nmse_floor_db(25.0)),
      "SIR 越低 NMSE 底越高")

m = itf.classify_measurement_sir(3.0)
check(m["band"] == "测量严重受损", "3 dB 测量 SIR 判为严重受损")
check(itf.classify_measurement_sir(-2.0)["band"] == "测量已失效",
      "负 SIR 判为测量已失效")

# ---------------------------------------------------------------------------
sect("5  几何采集钩子")

ok = itf.install_geometry_capture()
check(ok, "钩子挂载成功")
check(itf.install_geometry_capture(), "重复挂载幂等")


class _FakeSample:
    def __init__(self, sinr, sir):
        self.sinr_dB = sinr
        self.sir_dB = sir


# 暂存为空时必须回 nan，不能回上一次的值
itf._capture.clear()
check(math.isnan(itf.take_ul_geometry_sir(_FakeSample(1.0, 2.0))),
      "暂存为空时回 nan")

# 自检：暂存的下行量与 sample 对得上才给上行值
itf._capture.update({"dl_sinr_avg": 12.0, "sir_dl_db": 18.0,
                     "ul_sinr_avg": 5.0, "sir_ul_db": 9.0})
check(abs(itf.take_ul_geometry_sir(_FakeSample(12.0, 18.0)) - 9.0) < 1e-9,
      "下行量对得上时给出上行 SIR")

itf._capture.update({"dl_sinr_avg": 12.0, "sir_dl_db": 18.0,
                     "ul_sinr_avg": 5.0, "sir_ul_db": 9.0})
check(math.isnan(itf.take_ul_geometry_sir(_FakeSample(3.0, 18.0))),
      "下行量对不上时回 nan —— 宁可没有，也不给错的上行 IoT")

# 取过一次就清空，防止串到下一个样本
itf._capture.update({"dl_sinr_avg": 1.0, "sir_dl_db": 2.0,
                     "ul_sinr_avg": 3.0, "sir_ul_db": 4.0})
itf.take_ul_geometry_sir(_FakeSample(1.0, 2.0))
check(math.isnan(itf.take_ul_geometry_sir(_FakeSample(1.0, 2.0))),
      "取走后暂存清空，同一份不会被第二个样本重复领走")

# ---------------------------------------------------------------------------
sect("6  场景预设的完整性")

presets = pl.load_presets()
print(f"  共 {len(presets)} 个预设")
check(len(presets) >= 20, "预设数量 >= 20")

groups = pl.preset_groups()
print("  分组：" + "  ".join(f"{g}({len(v)})" for g, v in groups.items()))
check("干扰场景" in groups and "测量干扰" in groups and "大站间距" in groups,
      "干扰 / 测量干扰 / 大站间距 三组都在")

for name, body in presets.items():
    check(bool(body.get("group")), f"{name} 有 group")
    check(bool(body.get("label")) and bool(body.get("summary")), f"{name} 有 label/summary")

# 测量域的量只在 paired 模式下产生，声称测量干扰的场景必须 link=BOTH
for name in groups.get("测量干扰", []):
    link = presets[name]["config"].get("link")
    check(link == "BOTH", f"{name} 的 link 是 BOTH（否则拿不到测量域 SIR）")

# 大站间距场景必须带 caveat：ChannelHub 没有 RMa 路损公式
for name in groups.get("大站间距", []):
    check("RMa" in (presets[name].get("caveat") or ""),
          f"{name} 注明了用的不是 RMa 路损公式")

# 高铁场景必须走 linear 拓扑 + 移动模型，否则 train_* 参数根本不生效
for name in groups.get("高铁", []):
    c = presets[name]["config"]
    check(c.get("topology_layout") == "linear" and
          c.get("mobility_mode") in ("linear", "track"),
          f"{name} 是 linear 拓扑 + 移动模型（否则进不了高铁模式）")
    check(float(c.get("train_penetration_loss_db", 0)) > 0,
          f"{name} 设了车体穿透损耗")

# expect 里的实测值必须自洽：iot_dl_db 与它标注的等级对得上
for name, body in presets.items():
    e = body.get("expect") or {}
    if e.get("iot_dl_db") is None:
        continue
    want = itf.classify_iot(float(e["iot_dl_db"]))["band"]
    check(want == (e.get("iot_dl_band") or want),
          f"{name} 的 expect.iot_dl_db 与标注等级一致")

summaries = pl.preset_summaries()
check(all("group" in s for s in summaries), "preset_summaries 带 group")
check(all(("expect" not in s) or s["expect"] for s in summaries),
      "expect 字段要么没有、要么非空（不放空壳）")

# ---------------------------------------------------------------------------
sect("7  设计提示")

hint = itf.design_hint(20.0)
check(hint["band"] == "高干扰", "目标 20 dB 归入高干扰档")
check(abs(hint["equivalent_load"] - 0.99) < 1e-3, "20 dB 对应等效负载 0.99")
check(len(hint["levers"]) >= 5, "至少列出 5 个旋钮")
check(all({"key", "direction", "why", "note"} <= set(x) for x in hint["levers"]),
      "每个旋钮都说清方向、原因与注意事项")
check("复核" in hint["verification"], "明确要求生成后复核，不拿估算下结论")

# ---------------------------------------------------------------------------
sect("8  端到端：多小区数据集的 IoT 与报告")

ch.warmup()
cfg = dict(pl.load_presets()["multicell_7site"]["config"])
cfg["num_rb"] = 24          # 只减计算量；几何 IoT 与 num_rb 无关
cfg["num_ues"] = 7
cfg["measurements"] = {"ssb_rsrp": False}
summ = gen.generate(cfg, num_samples=7, workers=1)
ds = load(summ["dataset_id"])

check(summ.get("interference_modeled") is True, "多小区场景的干扰确实进了 SINR")
iot_block = summ.get("iot")
check(isinstance(iot_block, dict) and "dl" in iot_block, "summary 里有 iot 块")
dl = iot_block["dl"]
print(f"  下行 IoT 中位数 {dl['median_db']} dB，{dl['classification']['band']}，"
      f"等效负载 {dl['classification']['equivalent_load']}")
check(dl["n_valid"] > 0, "有有效 IoT 样本")
check(dl["median_db"] is not None and dl["median_db"] > 0,
      "多小区场景的 IoT 大于 0 dB（干扰确实存在）")

# 逐样本复核：报告里的 IoT 必须能由落盘的 sinr/sir 重算出来
sinr = ds.scalar("sinr_dB")
sir = ds.scalar("sir_dB")
recomputed = itf.iot_stats(sinr, sir)
check(abs(recomputed.median_db - dl["median_db"]) < 0.01,
      "summary 的 IoT 可由落盘标量原样重算（不是另存的快照）")

rep = itf.interference_report(summ["dataset_id"])
check(rep["traffic_domain"]["dl"]["iot"]["n_valid"] > 0, "报告里有业务域 IoT")
check(rep["iot_exact"] is True, "num_slots_per_sample=1 时 IoT 标为精确")
check(isinstance(rep["notes"], list), "报告带 notes")

# 新增的测量域列即使在 DL-only 场景下也要存在（值为 nan），
# 否则并行合并时两块的字段集会不一致。
for name in ("ul_sir_dB", "dl_sir_dB", "num_interfering_ues", "ul_sir_geo_dB"):
    try:
        arr = ds.scalar(name)
        check(arr.shape[0] == summ["num_samples"], f"{name} 每样本一个值")
    except KeyError:
        check(False, f"{name} 落盘了")

# ---------------------------------------------------------------------------
sect("8.5  探测模式：几何量必须与全量逐位相同")

from superwireless import scenario as sc  # noqa: E402

# 探测模式压 num_rb 和 num_ofdm_symbols 换速度，前提是几何量一个不差。
# **这一节比对的是实际发货的那组参数**，不是外推——num_ofdm_symbols 在 1 处
# 有一道悬崖（实测 sir_dB 偏 16.1 dB），所以只能逐个验证、不能"2 行那 4 也行"。
probe_cfg = dict(pl.load_presets()["multicell_7site"]["config"])
probe_cfg["num_ues"] = 7
probe_cfg["seed"] = 4242
# **参照组也必须带 bs_panel。** 缺 panel 时 ChannelHub 建不出 DFT 码本，
# 几何 SINR 整条路径被跳过、sinr_dB 退化成含 -10log10(RB) 的 snr_dB，
# 于是压 num_rb 会看到 10.56 dB 的"偏差"——那是配置缺陷，不是探测模式的问题。
# probe_config 现在自己补 panel，这里对照组也补，两边才可比。
gen._ensure_bs_panel(probe_cfg)


def _geom(**over):
    cfg = dict(probe_cfg, **over, num_samples=7)
    cfg.pop("source", None)
    itf.install_geometry_capture()
    out = []
    n = 0
    for smp in ch.iter_samples("internal_sim", cfg):
        mm = smp.meta if isinstance(smp.meta, dict) else {}
        out.append((
            float(smp.sinr_dB), float(smp.sir_dB or np.nan),
            float(mm.get("pathloss_dB", np.nan)),
            float(mm.get("distance_3d_m", np.nan)),
            float(mm.get("doppler_hz", np.nan)),
            itf.take_ul_geometry_sir(smp),
        ))
        n += 1
        if n >= 7:
            break
    return np.asarray(out)


ref_geom = _geom()
shipped, _rb, _rbf = sc.probe_config(dict(probe_cfg))
cut_geom = _geom(num_rb=shipped["num_rb"],
                 num_ofdm_symbols=shipped["num_ofdm_symbols"])
both = np.isfinite(ref_geom) & np.isfinite(cut_geom)
worst_geom = float(np.max(np.abs(ref_geom[both] - cut_geom[both])))
print(f"  发货参数 num_rb={shipped['num_rb']} "
      f"num_ofdm_symbols={shipped['num_ofdm_symbols']}，"
      f"几何量最大偏差 {worst_geom:.3e}")
check(worst_geom == 0.0, "探测模式的几何量与全量逐位相同（不是近似）")

# 缺 bs_panel 时探测会失真——probe_config 必须自己把它补上
_no_panel = dict(pl.load_presets()["multicell_7site"]["config"])
_no_panel.pop("bs_panel", None)
check("bs_panel" not in _no_panel, "预设本身不带 bs_panel（所以补齐这步不能省）")
check("bs_panel" in sc.probe_config(_no_panel)[0],
      "probe_config 自动补 bs_panel（否则 sinr_dB 会退化成 RB 相关的 snr_dB）")

# 悬崖回归：符号数降到 1 会让几何量失真，PROBE_NUM_SYM 绝不能滑到这里
cliff_geom = _geom(num_rb=shipped["num_rb"], num_ofdm_symbols=1)
both_c = np.isfinite(ref_geom) & np.isfinite(cliff_geom)
worst_cliff = float(np.max(np.abs(ref_geom[both_c] - cliff_geom[both_c])))
print(f"  num_ofdm_symbols=1 时几何量最大偏差 {worst_cliff:.2f} dB")
check(worst_cliff > 1.0,
      "num_ofdm_symbols=1 确实会破坏几何量（所以 PROBE_NUM_SYM 不能取 1）")
check(sc.PROBE_NUM_SYM > sc.PROBE_NUM_SYM_CLIFF,
      "PROBE_NUM_SYM 在悬崖之上")

# 移动场景每个 UE 至少要 2 个样本，否则多普勒恒为 0
_hst = sc.probe(dict(pl.load_presets()["hst_350kmh"]["config"]), num_samples=21)
print(f"  hst 探测 21 样本 -> 实跑 {_hst['num_samples']} 个"
      f"（每 UE {_hst['samples_per_ue']} 个），"
      f"多普勒中位 {_hst['geometry']['doppler_hz']['median']} Hz")
check(_hst["samples_per_ue"] >= 2, "移动场景自动把样本数补到每 UE >= 2")
check((_hst["geometry"]["doppler_hz"]["median"] or 0) > 100,
      "补够之后多普勒不再是 0（350 km/h @ 2.6 GHz 应有几百 Hz）")
check("num_samples_note" in _hst, "补样本这件事写进了报告，不是静默发生")

_static = sc.probe(dict(probe_cfg), num_samples=21)
check("num_samples_note" not in _static, "静止场景不做补样本（不白花时间）")

# 探测模式不支持射线追踪，必须直说而不是给一份假的探测报告
try:
    sc.probe({"source": "sionna_rt", "scene": "munich"}, num_samples=1)
    check(False, "射线追踪配置应当被拒绝")
except ValueError as exc:
    check("internal_sim" in str(exc), "射线追踪配置被明确拒绝并说明原因")

# ---------------------------------------------------------------------------
sect("9  端到端：paired 模式下的测量域 SIR")

cfg2 = dict(pl.load_presets()["srs_congested"]["config"])
cfg2["num_rb"] = 24
cfg2["num_ues"] = 7
cfg2["num_interfering_ues"] = 12
cfg2["measurements"] = {"ssb_rsrp": False}
summ2 = gen.generate(cfg2, num_samples=7, workers=1)
ds2 = load(summ2["dataset_id"])
rep2 = itf.interference_report(summ2["dataset_id"])

md = rep2.get("measurement_domain", {})
print("  测量域：" + ", ".join(md) if md else "  测量域：空")
check("ul_srs" in md, "paired 模式下拿到了 SRS 测量域 SIR")
if "ul_srs" in md:
    srs = md["ul_srs"]
    print(f"  SRS 测量 SIR 中位数 {srs['sir_dB']['median']} dB -> "
          f"{srs['classification']['band']}；NMSE 底 {srs['nmse_floor_db']} dB")
    check(srs["sir_dB"]["n"] > 0, "SRS 测量 SIR 有有效样本")
    check(srs["nmse_floor_db"] is not None, "给出了估计 NMSE 下限")

# 业务域与测量域是两个独立的量，不该恰好相等
if "ul_srs" in md and rep2["traffic_domain"].get("dl"):
    a = md["ul_srs"]["sir_dB"]["median"]
    b = rep2["traffic_domain"]["dl"]["sir_dB"]["median"]
    check(a is not None and b is not None and abs(a - b) > 0.01,
          f"测量域 SIR({a}) 与业务域 SIR({b}) 是不同的量")

# ---------------------------------------------------------------------------
sect("9.5  本地默认硬件：64T 1驱3 / 192 阵子 / 0.67λ")

from superwireless import hardware as hw  # noqa: E402

check(hw.COMPANY_RF_PANEL == [8, 4, 2], "RF 面板是 8H x 4V x 2pol")
check(hw.COMPANY_NUM_PORTS == 64, "RF 端口 64")
check(hw.COMPANY_ELEMENTS_PER_PORT == 3, "1 驱 3")
check(hw.COMPANY_NUM_ELEMENTS == 192, "物理阵子 192")
check(hw.COMPANY_H_SPACING_LAMBDA == 0.5, "水平间距 0.5λ")
# 这一条是实测纠正过的硬件值，写错会让全盘产物失真（见记忆 reconfig_mimo_sim）
check(hw.COMPANY_V_SPACING_LAMBDA == 0.67, "垂直间距 0.67λ（**不是 0.5**）")
check(abs(hw.COMPANY_ELEMENTS_PER_PORT * hw.COMPANY_V_SPACING_LAMBDA - 2.01) < 1e-9,
      "RF 端口垂直相位中心间距 2.01λ")
check(hw.COMPANY_CARRIER_HZ == 2.6e9, "载波 2.6 GHz (n41)")
check(hw.COMPANY_SCS_HZ == 30000, "子载波间隔 30 kHz")
check(hw.COMPANY_NUM_RB == 272 == hw.COMPANY_NUM_RBG * hw.COMPANY_RB_PER_RBG,
      "272 RB = 17 RBG x 16 RB")
check(hw.NR_TABLE_NUM_RB_100M_30K == 273,
      "同时记住 38.104 标准表是 273（口径不同，不是笔误）")
check(hw.COMPANY_UE_RX_ANT == 4 and hw.COMPANY_LINK == "DL", "默认 4R 下行")

# 自动挂载规则：只对 64T 面板生效，显式指定一律尊重
c1 = {"bs_panel": [8, 4, 2]}
hw.apply_array_defaults(c1)
check(hw.strip_markers(c1) == "company_1to3_192ae", "64T 面板自动切真实阵列")
check(c1["antenna_model_mode"] == "effective_subarray", "模式为 effective_subarray")
check(c1["bs_antenna"]["fixed_vertical_subarray"]["ae_vertical_spacing_lambda"] == 0.67,
      "挂上去的垂直间距是 0.67λ")

c2 = {"bs_panel": [16, 8, 2]}
hw.apply_array_defaults(c2)
check(hw.strip_markers(c2) == "skipped_non_64t", "非 64T 面板不套 1 驱 3（它是这款硬件的事实，不是通用规律）")
check("antenna_model_mode" not in c2, "非 64T 面板保持 ChannelHub 默认")

c3 = {"bs_panel": [8, 4, 2], "antenna_model_mode": "legacy_64"}
hw.apply_array_defaults(c3)
check(hw.strip_markers(c3) == "explicit" and c3["antenna_model_mode"] == "legacy_64",
      "显式指定 legacy_64 时不被覆盖（对照实验要用）")

# 预设：默认组必须真的走真实阵列
_pg = pl.preset_groups()
check("本地默认" in _pg, "有「本地默认」分组")
for name in _pg.get("本地默认", []):
    c = dict(pl.load_presets()[name]["config"])
    check(int(c.get("num_rb", 0)) == 272, f"{name} 用 272 RB")
    check(float(c.get("carrier_freq_hz", 0)) == 2.6e9, f"{name} 用 2.6 GHz")
    check(int(c.get("num_ue_rx_ant", 0)) == 4, f"{name} 默认 4R 接收")
    # **不在 preset 里写死 bs_panel**：写死会让 4T4R 这类天线覆盖失效
    # （num_bs_tx_ant 改了、panel 还是 64 口，两者矛盾）。让它由
    # _ensure_bs_panel 从 num_bs_tx_ant 推导，64 -> [8,4,2] 正是要的。
    check("bs_panel" not in c, f"{name} 不写死 bs_panel（由端口数推导）")

# sw_plan 的兜底预设应当是本地默认配置
_d, _prof = pl.create_draft("验证一个 CSI 压缩的想法")
check(_d.preset == "company_64t4r", f"通用意图默认挑 company_64t4r（实得 {_d.preset}）")

# 端到端：summary 必须带阵列口径
_cfg = dict(pl.load_presets()["company_64t4r"]["config"])
_cfg["num_rb"] = 24
_cfg["num_ues"] = 4
_s = gen.generate(_cfg, num_samples=8, workers=1)
_am = _s.get("antenna_model") or {}
print(f"  summary.antenna_model: mode={_am.get('antenna_model_mode')} "
      f"AE={_am.get('physical_elements')} dv={_am.get('ae_vertical_spacing_lambda')}")
check(_am.get("antenna_model_mode") == "effective_subarray", "summary 记录了真实阵列模式")
check(_am.get("physical_elements") == 192, "summary 记录了 192 物理阵子")
check(_am.get("element_pattern_is_measured") is False,
      "明示阵元方向图不是实测的（是 3GPP 式参数化模型）")
check("几何 SINR / IoT 不受它影响" in (_am.get("note") or ""),
      "明示阵列模型不影响几何 SINR/IoT")

# ---------------------------------------------------------------------------
sect("9.8  仿真说明书")

import re as _re  # noqa: E402
import xml.etree.ElementTree as _ET  # noqa: E402

from superwireless import algo_defs as _alg_defs  # noqa: E402
from superwireless import spec as sp  # noqa: E402


def _svgs(path):
    """返回 (完整 HTML, 文档里的静态 SVG 列表)。

    **必须先把 <script> 剥掉再找 SVG。** 交互面板的 JS 里用字符串拼 SVG，
    直接正则会把那段 JS 也当成一张图捞出来——它带 ${...} 模板占位，
    XML 解析必然失败，看起来像"生成的 SVG 坏了"。
    """
    h = Path(path).read_text(encoding="utf-8")
    body = _re.sub(r"<script>.*?</script>", "", h, flags=_re.S)
    return h, _re.findall(r"<svg.*?</svg>", body, _re.S)


# 多小区（六边形栅格）
_r1 = sp.write_spec(dict(pl.load_presets()["company_64t4r_multicell"]["config"]),
                    num_samples=100, title="test-hex")
_h1, _s1 = _svgs(_r1["html_path"])
print(f"  hex: {_r1['headline'][:60]}")
# 5 张示意图（阵列/拓扑/频域/TDD/剖面）+ 15 张算法流程图
_n_fam = len(_alg_defs.families(dict(pl.load_presets()["company_64t4r_multicell"]["config"])))
check(len(_s1) == 5 + _n_fam,
      f"5 张示意图 + {_n_fam} 张算法流程图，实得 {len(_s1)}")
for _i, _sv in enumerate(_s1):
    try:
        _ET.fromstring(_sv)
    except _ET.ParseError as _e:
        check(False, f"svg{_i} XML 格式正确（{_e}）")
check(_h1.rstrip().endswith("</html>"), "HTML 完整闭合")

_lay = [x for x in _s1 if "网络拓扑" in x][0]
_sites = len(_re.findall(r'<circle class="bsd"', _lay)) - 1   # 减去图例那个
check(_sites == 7, f"六边形 7 站都画出来了（实得 {_sites}）")
check(len(_re.findall(r'<line class="bore"', _lay)) - 1 == 21, "21 个扇区指向都画出来了")
check(len(_re.findall(r'<polygon class="hex"', _lay)) == 7, "7 个六边形小区都画出来了")

# 线性拓扑（高铁）：站点沿轨道两侧交错，不能画成一个点
_r2 = sp.write_spec(dict(pl.load_presets()["hst_350kmh"]["config"]), title="test-linear")
_h2, _s2 = _svgs(_r2["html_path"])
_lay2 = [x for x in _s2 if "网络拓扑" in x][0]
_cx = [float(x) for x in _re.findall(r'<circle class="bsd" cx="([-\d.]+)"', _lay2)][:-1]
print(f"  linear: {len(_cx)} 站，横坐标跨度 {max(_cx) - min(_cx):.0f}")
check(len(_cx) == 7, f"线性拓扑 7 站（实得 {len(_cx)}）")
check(max(_cx) - min(_cx) > 200, "线性拓扑真的铺开了，不是挤成一个点")

# **同一秒内连出两份不能互相覆盖。** 秒级时间戳做文件名时踩过：
# 后一份直接盖掉前一份且不报错，用户拿到的路径指向的是别人的图。
_a = sp.write_spec(dict(pl.load_presets()["company_64t4r"]["config"]), title="a")
_b = sp.write_spec(dict(pl.load_presets()["hst_350kmh"]["config"]), title="b")
check(_a["html_path"] != _b["html_path"], "同一秒生成的两份说明书文件名不冲突")
_na = len(_re.findall(r'<circle class="bsd"',
                      [x for x in _svgs(_a["html_path"])[1] if "网络拓扑" in x][0]))
_nb = len(_re.findall(r'<circle class="bsd"',
                      [x for x in _svgs(_b["html_path"])[1] if "网络拓扑" in x][0]))
check(_na != _nb, "两份内容各自独立（没被对方覆盖）")

# 阵列图必须如实反映实际用的模型
check("1 驱 3" in _h1 and "192" in _h1, "64T 说明书画的是 1 驱 3 / 192 阵子")
check("0.67" in _h1, "标注了 0.67λ 垂直间距")
_r3 = sp.write_spec(dict(pl.load_presets()["company_64t4r_legacy_array"]["config"]),
                    title="test-legacy")
_h3 = Path(_r3["html_path"]).read_text(encoding="utf-8")
check("legacy" in _h3, "legacy 配置画的是独立阵元，不冒充 1 驱 3")
check(any("legacy" in n for n in _r3["notes"]),
      "64T 却走 legacy 时在 notes 里明确警告")

# 参数来源要分得清
_r4 = sp.write_spec(dict(pl.load_presets()["company_64t4r"]["config"]),
                    user_set=["scenario", "num_ues"], title="test-user")
check(_r4["num_user_set"] == 2, f"用户指定项计数正确（实得 {_r4['num_user_set']}）")
check(_r4["num_params"] > _r4["num_user_set"], "其余标为默认值")

# 生成时自动带一份
_cfgs = dict(pl.load_presets()["company_64t4r"]["config"])
_cfgs["num_rb"] = 24
_cfgs["num_ues"] = 4
_ss = gen.generate(_cfgs, num_samples=8, workers=1)
_sheet = _ss.get("spec_sheet") or {}
check("html_path" in _sheet, "sw_generate 自动产出说明书")
check(Path(_sheet.get("html_path", "")).is_file(), "说明书文件真的落盘了")
check("UE" in Path(_sheet["html_path"]).read_text(encoding="utf-8"),
      "生成后的说明书带真实撒点")

# 分级呈现：拓扑图打头，其余折进 tab
check(_h1.count('name="tb"') == 7, "七个 tab 都在")
check(_h1.count('<section id="pn') == 7, "七个面板都在")
check('id="tb1" checked' in _h1, "默认停在总览")
# tab 必须是纯 CSS 的：**JS 挂了、被禁用、或者浏览器老旧，页签也得能切**。
# 页面现在确实有一段脚本（交互调参面板），但它不许碰页签。
_js_all = "".join(_re.findall(r"<script>(.*?)</script>", _h1, _re.S))
check('name="tb"' not in _js_all and "tb1" not in _js_all,
      "脚本不碰页签（JS 挂了也不影响导航）")
check("#tb1:checked~.panels>#pn1" in _h1, "页签靠 CSS :checked 切换，不是 JS")
# input 必须是 .tabs 的直接子元素且与 .panels 同级，~ 选择器才成立。
# 早先套了一层 .tabbar，页面上直接露出原生 radio、一个面板都不显示——
# 光看代码看不出来，是在浏览器里看出来的。
check('<div class="tabs">\n<input' in _h1 or '<div class="tabs">\r\n<input' in _h1,
      "input 是 .tabs 的直接子元素（否则 CSS 兄弟选择器失效、tab 变成裸 radio）")
check("#tb1:checked~.panels>#pn1" in _h1, "选择器按 id 一一对应")

# 拓扑图在 tab 之外（首屏就能看到），不是折进某个页签
_hero_at = _h1.index('class="hero"')
_tabs_at = _h1.index('<div class="tabs">')
check(_hero_at < _tabs_at, "拓扑图在 tab 之前 —— 打开就看得见，不用点")
check(_h1.index('class="facts"') < _tabs_at, "关键信息卡也在首屏")

# highlight：对话里点过名的参数顶到最前并高亮
_r5 = sp.write_spec(dict(pl.load_presets()["company_64t4r_multicell"]["config"]),
                    title="test-hl", highlight=["isd_m"])
_h5 = Path(_r5["html_path"]).read_text(encoding="utf-8")
# 注意别拿 '<div class="fact' 去找第一张卡片——它会先命中容器
# '<div class="facts">'。从容器结束的位置往后找。
_box = _h5.index('<div class="facts">') + len('<div class="facts">')
check(_h5[_box:_box + 40].startswith('<div class="fact hi"'),
      "被 highlight 的信息卡排在第一个并高亮")
_r6 = sp.write_spec(dict(pl.load_presets()["company_64t4r_multicell"]["config"]),
                    title="test-nohl")
_h6 = Path(_r6["html_path"]).read_text(encoding="utf-8")
check('class="fact hi"' not in _h6, "没传 highlight 时不乱高亮")

# 说明文字里的 ** 要变成 <b>，不能在页面上露出星号
_body_only = _re.sub(r"<details class=\"algo\".*?</details>", "",
                     _h1.split("<footer>")[0], flags=_re.S)
check("**" not in _body_only, "页面正文没有裸露的 markdown 星号")

# 画布随规模自适应：单站不该用多小区那么大的画布
_r7 = sp.write_spec(dict(pl.load_presets()["company_64t4r"]["config"]), title="test-1cell")
_h7 = Path(_r7["html_path"]).read_text(encoding="utf-8")
_lay7 = [x for x in _re.findall(r"<svg.*?</svg>", _h7, _re.S) if "网络拓扑" in x][0]
_lay1 = [x for x in _s1 if "网络拓扑" in x][0]
_w7 = int(_re.search(r'viewBox="0 0 (\d+)', _lay7).group(1))
_w1 = int(_re.search(r'viewBox="0 0 (\d+)', _lay1).group(1))
print(f"  画布：单站 {_w7} / 多小区 {_w1}")
check(_w7 < _w1, "单站用更小的画布（大画布只会得到一片空白加中间一个点）")

# --- 交互式调参面板 ---
check(_h1.count('name="tb"') == 7, "七个 tab（改配置 + 算法）")
check('id="pn2"' in _h1 and 'data-k="isd_m"' in _h1, "调参面板在第 2 个页签")
for _k in ("num_sites", "sectors_per_site", "isd_m", "num_ues", "num_bs_tx_ant",
           "carrier_freq_hz", "scenario", "channel_model", "link"):
    check(f'data-k="{_k}"' in _h1, f"{_k} 可在页面上改")
check('id="cp"' in _h1 and 'id="rs"' in _h1, "有复制与重置按钮")
check('id="pl"' in _h1, "有可粘贴的 payload 文本框")
check("overrides = " in _h1, "payload 里带 overrides，agent 能直接照做")

# 拓扑预览的坐标由 ChannelHub 现算后内嵌，前端只做线性缩放——
# 不在 JS 里重写栅格逻辑，就不会出现"图上七站、跑出来十九站"
_u = sp.unit_hex_layouts()
check(set(_u) == {"1", "7", "19"}, f"内嵌 1/7/19 三套单位布局（实得 {sorted(_u)}）")
check([len(v) for v in (_u["1"], _u["7"], _u["19"])] == [1, 7, 19], "站数对得上")
_d7 = max(abs(x) for x, y in _u["7"])
check(abs(_d7 - 1.0) < 1e-6, f"单位布局按 ISD=1 归一（最大横坐标 {_d7}）")
check('const ST=' in _h1 and '"unit"' in _h1, "单位布局内嵌进页面")

# **JS 语法回归。** payload 里要换行，早先写成 \n 要穿过两层 f-string，
# 结果塌成真换行落进单引号字符串里 —— 整段脚本 SyntaxError、页面点不动，
# 而 HTML 结构检查完全看不出来。现在用 String.fromCharCode(10)，不带转义。
_all_js = _re.findall(r"<script>(.*?)</script>", _h1, _re.S)
# 页面上现在有三段脚本：KaTeX 本体（第三方，不体检）、调参面板、KaTeX 升级器。
# **只体检我们自己手写的那段**——第三方压缩产物里当然全是跨行引号与转义。
_own = [x for x in _all_js if "katex" not in x[:400].lower()
        and "const ST=" in x]
check(len(_all_js) == 3, f"三段内联脚本：KaTeX 本体 + 调参面板 + 升级器（实得 {len(_all_js)}）")
check(len(_own) == 1, "调参面板的脚本唯一可辨认")
_js = _own
_bad = []
for _i, _ln in enumerate(_js[0].splitlines(), 1):
    _stripped = _re.sub(r"\\.", "", _ln)          # 去掉转义对
    if _stripped.count("'") % 2 or _stripped.count('"') % 2:
        _bad.append(_i)
check(not _bad, f"脚本里没有跨行的引号字符串（可疑行：{_bad[:5]}）")
check("String.fromCharCode(10)" in _js[0], "换行用字符码而不是反斜杠转义")

# **SVG 的 <style> 是文档级的，不是这张图私有的。** 页面上两张图时后注入的
# 会盖掉同名 class —— 预览图的 .sec{fill:url(#sg2)} 曾把静态图的扇区填充
# 整个抹掉（引用到藏在隐藏 tab 里、渲染不出来的那个渐变）。
# 渐变改成写在元素上的 presentation attribute，跨图不再打架。
check('class="sec"' not in _h1, "扇区填充不靠 class（SVG style 是文档级的，会跨图串）")
check(_h1.count('fill="url(#sg)"') == 21, "静态图 21 个扇区都有渐变填充")
check("url(#sg2)" in _js[0], "预览图用自己的渐变 id，不与静态图重名")
check("\\n" not in _js[0], "脚本里不残留反斜杠转义的换行")

# ---------------------------------------------------------------------------
sect("9.9  回传桥：页面把改动直接送回 agent，不用复制粘贴")

import json as _json  # noqa: E402
import time  # noqa: E402
import urllib.error as _ue  # noqa: E402
import urllib.request as _urq  # noqa: E402

from superwireless import bridge as _br  # noqa: E402

_rb = sp.write_spec(dict(pl.load_presets()["company_64t4r_multicell"]["config"]),
                    num_samples=60, title="test-bridge", open_browser=False)
check(_rb["writeback"] == "post", f"回传通道通了（实得 {_rb['writeback']}，"
                                  f"{_rb.get('serve_error')}）")
check(bool(_rb["url"]) and _rb["url"].startswith("http://127.0.0.1:"),
      "只绑环回地址，本机以外连不上")
_hb = Path(_rb["html_path"]).read_text(encoding="utf-8")
check('id="ap"' in _hb and "应用到仿真" in _hb, "页面上有「应用到仿真」按钮")
check("location.protocol" in _hb,
      "页面自己判断能不能 POST（file:// 下退回复制粘贴）")

# 落盘那份和挂出去那份必须是同一个东西——不然用户拷走的 HTML 和他刚才看的不一样
check(_urq.urlopen(_rb["url"]).read().decode("utf-8") == _hb,
      "服务端返回的 HTML 与落盘的逐字相同")


def _post(payload):
    req = _urq.Request(_br.apply_url(), data=_json.dumps(payload).encode("utf-8"),
                       headers={"Content-Type": "application/json"})
    try:
        r = _urq.urlopen(req)
        return r.status, _json.loads(r.read().decode("utf-8"))
    except _ue.HTTPError as e:
        return e.code, _json.loads(e.read().decode("utf-8"))


_code, _body = _post({"id": _rb["spec_id"], "overrides": {"num_sites": 19, "isd_m": 300},
                      "nonce": "n1", "text": "改两项"})
check(_code == 200 and _body.get("ok"), f"合法回传被收下（{_code} {_body}）")
_got = _br.await_submission(3, _rb["spec_id"])
check(len(_got) == 1 and _got[0].overrides == {"num_sites": 19, "isd_m": 300},
      f"agent 侧原样收到（实得 {[s.overrides for s in _got]}）")
check(not _br.await_submission(1, _rb["spec_id"]),
      "取走即清空，同一条不会被读两次；没人点时返回空列表而不是抛异常")

# **幂等键。** 回执可能在路上丢（服务正好退出、socket 被掐），页面会重发一次。
# 没有 nonce 的话重发就是第二份改动，agent 看起来像用户点了两次。
_post({"id": _rb["spec_id"], "overrides": {"isd_m": 400}, "nonce": "n2"})
_c2, _b2 = _post({"id": _rb["spec_id"], "overrides": {"isd_m": 400}, "nonce": "n2"})
check(_c2 == 200 and _b2.get("dup") is True, f"同 nonce 重发被识别为重复（{_b2}）")
check(len(_br.await_submission(3, _rb["spec_id"])) == 1, "重发只算一份改动")
check("nonce" in _js[0] and "setTimeout" in _js[0], "页面失败后会带同一个 nonce 重发一次")

# --- 接口是开着的，就得按"任何人都能戳"来写 ---
_c3, _b3 = _post({"id": _rb["spec_id"], "overrides": {"rm_rf": "/"}})
check(_c3 == 400 and "rm_rf" in _b3.get("error", ""), f"白名单外的参数名被拒（{_b3}）")
_c4, _b4 = _post({"id": _rb["spec_id"], "overrides": {"isd_m": {"nested": 1}}})
check(_c4 == 400, f"非标量值被拒（{_b4}）")
_c5, _b5 = _post({"id": "不存在的说明书", "overrides": {"isd_m": 300}})
check(_c5 == 404, f"未注册的说明书被拒（{_b5}）")
check(_br.allowed_for(_rb["spec_id"]) == sp.editable_keys() and "isd_m" in sp.editable_keys(),
      "白名单就是页面上那些控件，没有另抄一份")
try:
    _urq.urlopen(_br.status()["base_url"] + "/s/deadbeef/" + _rb["spec_id"])
    check(False, "错 token 的 URL 被拒")
except _ue.HTTPError as _e:
    check(_e.code == 404, f"错 token 的 URL 被拒（实得 {_e.code}）")

# --- 送达感知：MCP 没有推送通道，用户点了必须有办法被看见 ---
# **这条是从真实事故来的。** 用户点了「应用到仿真」，agent 当时没在等，
# 改动躺在收件箱里，CLI 上毫无动静——用户以为功能没生效。
check(_br.pending_count() == 0, "起点收件箱是空的")
_post({"id": _rb["spec_id"], "overrides": {"num_sites": 1}, "nonce": "n3"})
check(_br.pending_count() == 1, "未取走的改动能被计数")

# 每个 MCP 工具的返回值都要挂上它 —— 这是唯一能让用户点击"被看见"的通道
from superwireless import server as _srv2  # noqa: E402

_wrapped = _srv2._with_pending({"foo": 1})
check("pending_config_changes" in _wrapped, "工具返回值挂上了未处理回传的通知")
check(_wrapped["pending_config_changes"]["count"] == 1, "通知里带条数")
check("sw_await_config" in _wrapped["pending_config_changes"]["action"],
      "通知里直接给出该调什么，不让 agent 自己猜")
check(_wrapped["foo"] == 1, "原返回值原样保留")
check(_srv2._with_pending("不是字典") == "不是字典", "非 dict 返回原样放行")

# agent 有没有在等，页面上要说不同的话 —— 对用户是完全不同的两件事
_c6, _b6 = _post({"id": _rb["spec_id"], "overrides": {"isd_m": 600}, "nonce": "n4"})
check(_b6.get("waiting") is False, f"agent 没在等时回执标 waiting=False（{_b6.get('msg')}）")
check("在忙" in _b6.get("msg", ""), "话术如实说明改动是入了收件箱而不是被立刻处理")

import threading as _th  # noqa: E402

_seen_waiting = {}


def _wait_bg():
    _seen_waiting["got"] = _br.await_submission(6, _rb["spec_id"])


_t = _th.Thread(target=_wait_bg, daemon=True)
_br.drain(_rb["spec_id"])          # 清干净，让后台线程真的进入等待
_t.start()
time.sleep(0.8)                     # 等它挂上去
_c7, _b7 = _post({"id": _rb["spec_id"], "overrides": {"isd_m": 700}, "nonce": "n5"})
check(_b7.get("waiting") is True, f"agent 正在等时回执标 waiting=True（{_b7.get('msg')}）")
_t.join(8)
check(_seen_waiting.get("got") and _seen_waiting["got"][-1].overrides == {"isd_m": 700},
      "正在等的那次调用确实拿到了改动")
check(_br.pending_count() == 0, "取走后计数归零，工具返回值不再挂通知")
check("pending_config_changes" not in _srv2._with_pending({"foo": 1}),
      "收件箱空时不打扰")

# 页面要把服务端真正收到的项回显出来，并区分两种状态
check("j.waiting" in _js[0] and "m.className" in _js[0], "页面按 waiting 区分两种回执状态")
check("m.innerHTML" not in _js[0], "回执用 textContent 拼，不往 innerHTML 塞服务端字符串")

# 关掉服务时必须**看得见地**降级，不能假装还能回传
os.environ["SUPERWIRELESS_NO_SERVE"] = "1"
try:
    _rn = sp.write_spec(dict(pl.load_presets()["company_64t4r"]["config"]),
                        title="test-noserve", open_browser=False)
    check(_rn["writeback"] == "clipboard" and _rn["url"] is None,
          "关掉服务后退回复制粘贴")
    check(bool(_rn["serve_error"]), f"降级原因说清楚了（{_rn['serve_error']}）")
    check("应用到仿真" in Path(_rn["html_path"]).read_text(encoding="utf-8"),
          "按钮仍在 HTML 里（换台机器起了服务照样能用），只是页面自己不显示")
finally:
    os.environ.pop("SUPERWIRELESS_NO_SERVE", None)
# ---------------------------------------------------------------------------
sect("9.10  算法页签：这次用了哪些算法，全部可见")

from superwireless import algorithms as _alg  # noqa: E402

_ai = _alg.algorithm_list(dict(pl.load_presets()["company_64t4r_multicell"]["config"]))
check(len(_ai) >= 13, f"算法清单至少 13 条（实得 {len(_ai)}）")
check({a["stage"] for a in _ai} <= set(_alg.stages()), "每条都归到已知阶段")
for _k in ("rank_adaptation", "noise_reference", "mcs_selection", "mu_pairing",
           "su_mu_adaptation", "scheduler", "traffic", "experienced_throughput",
           "harq", "receiver", "channel_est"):
    check(any(a["key"] == _k for a in _ai), f"{_k} 在清单里")

# **每条都得说清什么时候会失真** —— 只写"用了 XX 算法"没有价值
_no_caveat = [a["key"] for a in _ai if not a.get("caveat")]
check(not _no_caveat, f"每条算法都写了失真条件（缺：{_no_caveat}）")
_no_why = [a["key"] for a in _ai if not a.get("why")]
check(not _no_why, f"每条算法都写了为什么这么选（缺：{_no_why}）")

# 清单必须跟着配置变，不能是写死的一张纸
_leg = _alg.algorithm_list({"num_bs_tx_ant": 4, "num_sites": 1, "sectors_per_site": 1})
_a64 = next(a for a in _ai if a["key"] == "antenna_model")
_a4 = next(a for a in _leg if a["key"] == "antenna_model")
check("1 驱 3" in _a64["choice"] and "legacy" in _a4["choice"],
      "阵列模型那条随配置变（64T 走 1 驱 3、4T 走 legacy）")
_r64 = next(a for a in _ai if a["key"] == "receiver")
_r4 = next(a for a in _leg if a["key"] == "receiver")
check(_r64["caveat"] != _r4["caveat"], "接收机那条在单/多小区下说法不同")

# 现网锚点必须带出处
check(_alg.FIELD_ANCHORS["avg_rank"] == 2.7 and _alg.FIELD_ANCHORS["avg_mcs"] == 15.0,
      "现网锚点是用户给的那两个数")
check("2026-08-02" in _alg.FIELD_ANCHORS["source"], "锚点带出处与日期")

# --- 页面上真的能看见 ---
_ha = _h1                                            # 多小区那份说明书
check(_ha.count('name="tb"') == 7, "七个 tab（多了「算法」）")
check('<label for="tb6">算法</label' in _ha, "算法页签有入口")
check(_ha.count('<details class="algo"') >= 13, "13 条算法都渲染进页面")
check("#tb6:checked~.panels>#pn6" in _ha, "算法页签能被 CSS 切出来")
check("#tb7:checked~.panels>#pn7" in _ha, "参数全表顺延到第 7 个且仍可切换")

# 默认折叠、点开才看细节 —— 否则一屏塞不下
check("<details" in _ha and "open>" not in _ha.split('id="pn6"')[1][:4000],
      "算法条目默认折叠")
# **CSS 转义的坑**：▸ 用字面字符写，别用 \25B8（Python 会当八进制吃掉 \2）
check("\25B8" not in _ha and "\25BE" not in _ha, "展开标记不用十六进制转义")
check("\u25b8" in _ha and "\u25be" in _ha, "展开标记是字面的 ▸ ▾")
check(not any(ord(c) < 9 or 11 <= ord(c) < 32 for c in _ha),
      "页面里没有被转义咬坏的控制字符")

# 关键结论必须出现在页面上，而不只是藏在代码注释里
for _phrase in ("12 dB", "信道求逆功控", "单码字", "缓冲区非空", "秩 1", "平均 rank 2.7"):
    check(_phrase in _ha, f"页面上写清了「{_phrase}」")

# --- 公式渲染：KaTeX 排版 + MathML 兜底 ---
# 用户 2026-08-03 批准内联 KaTeX（"内联如果只有 1MB，感觉完全可接受"）。
# **两层而不是二选一**：KaTeX 靠 JS 渲染，脚本没跑起来时只剩裸 LaTeX，
# 而那恰恰是最需要看懂公式的场合。所以容器里先放 MathML 兜底。
from superwireless import katex as _kx  # noqa: E402

check(_kx.available(), "内联 KaTeX 资产在位（缺了公式会静默退回 MathML）")
_kxm = _kx.meta()
check(_kxm.get("fonts") == 20, f"20 个 woff2 字体全部内联（实得 {_kxm.get('fonts')}）")
_kb = (_kxm.get("css_bytes", 0) + _kxm.get("js_bytes", 0)) / 1024
print(f"  KaTeX {_kxm.get('version')}：{_kb:.0f} KB")
check(_kb < 1024, f"内联体积在 1 MB 预算内（{_kb:.0f} KB）")

_tex = _re.findall(r'<span class="kx"[^>]*data-tex="([^"]*)"', _ha)
check(len(_tex) >= 40, f"页面上至少 40 条公式（实得 {len(_tex)}）")
check(_ha.count('class="kx"') == _ha.count("<math"),
      f"每条公式都带 MathML 兜底（kx {_ha.count(chr(34) + 'kx' + chr(34))} "
      f"vs math {_ha.count('<math')}）")
check("katex.renderToString" in _ha, "KaTeX 升级脚本内联在页面里")
check("data:font/woff2;base64," in _ha, "字体是 base64 内联的，不是外链")
# **断网也要好看。** 引 CDN 的话离线打开公式就没了。
for _host in ("cdn.jsdelivr.net", "unpkg.com", "cdnjs.cloudflare.com", "//fonts.g"):
    check(_host not in _ha, f"页面不引外部资源（{_host}）")

# --- 系统级旋钮：页面默认值必须和工具签名一致 ---
# 漂了的话页面显示的就不是实际会跑的值，而这种不一致没有任何提示。
import inspect as _insp  # noqa: E402

from superwireless import server as _srv0  # noqa: E402

_sig = _insp.signature(getattr(_srv0.sw_system_sim, "__wrapped__", _srv0.sw_system_sim))
for _k, _v in sp._SIM_DEFAULTS.items():
    _p = _sig.parameters.get(_k)
    _d = _p.default if _p else None
    if isinstance(_d, bool):
        _d = "on" if _d else "off"
    check(_p is not None and _d == _v,
          f"面板默认值 {_k}={_v!r} 与 sw_system_sim 签名一致（签名 {_d!r}）")
    check(_k in sp.editable_keys(), f"{_k} 在回传白名单里")
for _k in ("neighbor_prb_util", "csi_aging", "srs_period_ms", "srs_hopping",
           "olla_speedup"):
    check(f'data-k="{_k}"' in _ha, f"改配置页上有 {_k} 控件")

# --- 对标量的逐步推导，供人工核对 ---
_dv = _alg.derivations({})
check(len(_dv) >= 4, f"至少四项推导（实得 {len(_dv)}）")
for _d in _dv:
    check(len(_d["steps"]) >= 4, f"{_d['name']} 的推导至少 4 步")
    check(all(len(x) == 3 for x in _d["steps"]), f"{_d['name']} 每步都有(做什么,公式,结果)")
    check(bool(_d["ref_src"]), f"{_d['name']} 带参考出处")

# **数字必须现算，不能是抄进来的常量。** 抄的话改了 MCS 表这里不会跟着变。
_peak = next(d for d in _dv if d["key"] == "peak_se")
from superwireless import linkadapt as _la  # noqa: E402

_m27 = _la.MCS_TABLES[3][27]
check(f"{4 * _m27.se:.3f}" in _peak["result"],
      f"峰值谱效由 MCS 表现算（表里 SE={_m27.se:.4f}×4，页面写 {_peak['result']}）")
check(str(_m27.q_m) in _peak["steps"][0][2], "调制阶数取自 MCS 表")

_rate = next(d for d in _dv if d["key"] == "peak_rate")
check("38.306" in _rate["ref_src"], "峰值速率引 38.306 §4.1.2")
check("transport_block_size" in " ".join(x[1] for x in _rate["steps"]),
      "峰值速率走真实的 TBS 函数，不是另算一套")

# 页面上要能展开看到
check(_ha.count("这一步在做什么") == len(_dv), f"{len(_dv)} 张推导表都渲染了")
check("参考出处" in _ha, "每张表带参考出处")
_dtl = _ha.count('<details class="algo"')
check(_dtl >= 13 + len(_dv), f"折叠项 = 13 条算法 + {len(_dv)} 项推导（实得 {_dtl}）")


# ---------------------------------------------------------------------------
sect("10  文档里的数字必须和代码对得上")

# "19 项体检"这句话在 README / SKILL.md / 两份 HTML 里写了八处，而 full_report
# 从第一版（f44b46a）起就只有 16 项 —— 数字是凭印象写的，从没对过账。
# 这一节让文档里的计数与代码绑死，省得下次再漂。
import re  # noqa: E402

from superwireless import server as _srv  # noqa: E402
from superwireless import validate as _val  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
n_checks = len(_val.full_report(ds).checks)
print(f"  full_report 实有 {n_checks} 项检查")

_pat = re.compile(r"(\d+)\s*项(?:体检|可信度检查|检查)")
for name in ("README.md", "skills/channel-sim/SKILL.md",
             "CAPABILITIES.html", "SETUP.html"):
    path = ROOT / name
    if not path.is_file():
        continue
    claims = {int(m) for m in _pat.findall(path.read_text(encoding="utf-8"))}
    check(all(c == n_checks for c in claims),
          f"{name} 声称的体检项数都等于 {n_checks}（文中出现 {sorted(claims)}）")

n_tools = len([n for n in vars(_srv) if n.startswith("sw_")])
_m = re.search(r"MCP 工具（(\d+) 个）", (ROOT / "README.md").read_text(encoding="utf-8"))
print(f"  server 实有 {n_tools} 个 sw_ 工具，README 写 {_m.group(1) if _m else '未写'}")
check(bool(_m) and int(_m.group(1)) == n_tools, f"README 声称的 MCP 工具数等于 {n_tools}")

# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
if FAILED:
    print(f"FAILED {len(FAILED)} 项：")
    for f in FAILED:
        print("  -", f)
    sys.exit(1)
print("干扰量化、场景预设全部通过。")
