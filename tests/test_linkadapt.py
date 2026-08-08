"""链路自适应（链路到系统映射）与并行生成的测试。

直接运行：python tests/test_linkadapt.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.stdout.reconfigure(errors="replace")

from superwireless import channelhub as ch  # noqa: E402
from superwireless import generate as gen  # noqa: E402
from superwireless import linkadapt as la  # noqa: E402
from superwireless import load  # noqa: E402

FAILED: list[str] = []


def check(cond: bool, label: str) -> None:
    print(("  PASS  " if cond else "  FAIL  ") + label)
    if not cond:
        FAILED.append(label)


def sect(t: str) -> None:
    print("\n" + "=" * 70 + f"\n{t}\n" + "=" * 70)


def main() -> None:
    ch.warmup()

    # -----------------------------------------------------------------------
    sect("1  QAM 约束容量（精确计算，可自检）")

    snr_db = np.array([-20., -10., 0., 10., 20., 30., 40.])
    shannon = np.log2(1 + 10 ** (snr_db / 10))
    print(f"  {'SNR':>5} {'香农':>8} " + " ".join(f"{m:>8}" for m in (4, 16, 64, 256)))
    for i, s in enumerate(snr_db):
        vals = [la.qam_mi(m, s)[0] for m in (4, 16, 64, 256)]
        print(f"  {s:>5.0f} {shannon[i]:>8.3f} " + " ".join(f"{v:>8.3f}" for v in vals))

    for m in (4, 16, 64, 256):
        mi = la.qam_mi(m, snr_db)
        check(bool(np.all(mi <= shannon + 1e-6)), f"{m}QAM 互信息恒 ≤ 香农")
        check(bool(np.all(np.diff(mi) >= -1e-9)), f"{m}QAM 互信息随信噪比单调")
        check(abs(mi[-1] - np.log2(m)) < 0.02, f"{m}QAM 高信噪比饱和到 log2(M)")

    # 低信噪比处应与香农重合 —— 这条能抓住 sigma 定义差 3 dB 那类错误
    lo = la.qam_mi(4, -25.0)[0]
    check(abs(lo - np.log2(1 + 10 ** (-2.5))) < 1e-3, "低信噪比处与香农重合（口径正确）")

    # 反解要能还原
    for m in (4, 64):
        g0 = 12.0
        back = la.qam_mi_inverse(m, la.qam_mi(m, g0)[0])[0]
        check(abs(back - g0) < 0.3, f"{m}QAM 互信息反解可还原信噪比")

    # -----------------------------------------------------------------------
    sect("2  38.214 表格自检")

    v = la.verify_tables()
    print(f"  核对 {v['n_checked']} 行：SE == q_m·R/1024")
    check(v["consistent"], "MCS/CQI 表内蕴一致（抄错一个数就会不一致）")
    check(len(la.MCS_TABLE_1) == 29, "MCS 表 1 共 29 档")
    check(len(la.MCS_TABLE_2) == 28, "MCS 表 2 共 28 档")
    check(len(la.CQI_TABLE_1) == 15, "CQI 表 1 共 15 档")
    check(max(m.q_m for m in la.MCS_TABLE_1) == 6, "表 1 最高 64QAM")
    check(max(m.q_m for m in la.MCS_TABLE_2) == 8, "表 2 含 256QAM")

    # 标准表在调制切换点上 SE 故意重叠，这不是抄错
    check(la.MCS_TABLE_1[9].se < la.MCS_TABLE_1[10].se, "MCS9→10 SE 重叠（标准如此）")
    check(la.MCS_TABLE_1[16].se > la.MCS_TABLE_1[17].se, "MCS16→17 SE 回落（标准如此）")

    # -----------------------------------------------------------------------
    sect("3  传输块大小（38.214 §5.1.3.2 两支都要覆盖）")

    n_re = la.re_per_slot(273, n_symbols=12, n_dmrs_per_prb=12)
    print(f"  273 PRB → n_re={n_re}")
    check(n_re == 273 * 132, "RE 数 = PRB × (12·符号 − DMRS)")
    check(la.re_per_slot(273, n_symbols=14, n_dmrs_per_prb=0) == 273 * 156,
          "每 PRB 的 RE 数封顶 156（标准明写）")

    small = la.transport_block_size(50, 0.234, 2, 1)
    check(small in la._TBS_SMALL, "小包走查表分支，落在 Table 5.1.3.2-1 上")

    big = la.transport_block_size(n_re, 0.926, 6, 4)
    check(big > 3824, "大包走量化分支")
    check(big % 8 == 0, "大包 TBS 是 8 的倍数")
    print(f"  273PRB MCS28 4层 → TBS {big} bit → {big/0.5e-3/1e6:.0f} Mbps 峰值")
    check(1_200 < big / 0.5e-3 / 1e6 < 2_200, "100MHz 4 层峰值吞吐在 1.2~2.2 Gbps")

    # TBS 应随各因素单调
    check(la.transport_block_size(n_re, 0.5, 6, 2) > la.transport_block_size(n_re, 0.5, 6, 1),
          "层数翻倍 TBS 变大")
    check(la.transport_block_size(n_re, 0.9, 6, 1) > la.transport_block_size(n_re, 0.3, 6, 1),
          "码率越高 TBS 越大")

    # -----------------------------------------------------------------------
    sect("4  码块分段")

    for tbs, r, want_ge in ((200, 0.117, 1), (8448, 0.5, 2), (200808, 0.926, 20)):
        c, k = la.code_blocks(tbs, r)
        print(f"  TBS {tbs:7d} R={r:.3f} → {c:3d} 块 × {k} bit")
        check(c >= want_ge, f"TBS {tbs} 至少分成 {want_ge} 块")
        check(k <= 8448 + 24, "每块不超过 BG1 的 K_cb")
    check(la.code_blocks(200, 0.117)[0] == 1, "小 TB 不分段")

    # -----------------------------------------------------------------------
    sect("5  有效 SINR（链路到系统映射）")

    flat = np.full(273, 15.0)
    for meth in ("miesm", "eesm"):
        e = la.effective_sinr(flat, method=meth, m_order=64)
        check(abs(e - 15.0) < 0.15, f"{meth}：平坦信道下等于原值")

    rng = np.random.default_rng(0)
    prev = None
    for spread in (2.0, 6.0, 12.0):
        g = 15.0 + rng.normal(0, spread, 273)
        e = la.effective_sinr(g, method="miesm", m_order=64)
        lin = 10 * np.log10(np.mean(10 ** (g / 10)))
        print(f"  起伏 σ={spread:4.1f} dB → 有效 {e:6.2f} dB，线性均值 {lin:6.2f} dB")
        check(e < lin, "有效 SINR 低于线性均值（好 RE 补不了坏 RE）")
        if prev is not None:
            check(e < prev, "频选越严重有效 SINR 越低")
        prev = e

    # -----------------------------------------------------------------------
    sect("6  BLER 模型的门限锚点")

    a = la.DEFAULT_BLER.anchor_check(table=1, n_coded_bits=20000)
    lo_db, hi_db = a["span_db"]
    print(f"  MCS0 需要 {lo_db:.2f} dB，MCS28 需要 {hi_db:.2f} dB")
    print(f"  调制切换点回落：{a['modulation_switch_drops']}")
    check(a["monotonic_within_modulation"], "同一调制内门限单调上升")
    check(a["above_shannon_limit"], "每档门限都高于其香农极限（不可能优于容量）")
    check(-8 <= lo_db <= -3, "MCS0 门限落在公开曲线的常见区间 −5~−7 dB 附近")
    check(18 <= hi_db <= 24, "MCS28 门限落在公开曲线的常见区间 20~23 dB 附近")
    check(all(0 < d["drop_db"] < 1.0 for d in a["modulation_switch_drops"]),
          "切换点回落幅度很小（标准表设计使然，非缺陷）")
    check("不是实测" in a["caveat"], "如实标注 BLER 是模型不是实测")

    # BLER 必须随信噪比单调下降
    m28 = la.MCS_TABLE_1[28]
    b = la.DEFAULT_BLER.bler(np.array([10., 15., 20., 25., 30.]), m28, 20000, 3)
    check(bool(np.all(np.diff(b) <= 1e-12)), "BLER 随信噪比单调下降")
    check(b[0] > 0.9 and b[-1] < 1e-3, "低信噪比几乎必错、高信噪比几乎必对")

    # 分段越多 TB 越容易错
    b1 = float(la.DEFAULT_BLER.bler(19.0, m28, 20000, 1)[0])
    b24 = float(la.DEFAULT_BLER.bler(19.0, m28, 20000, 24)[0])
    print(f"  同信噪比下 1 块 BLER {b1:.4f}，24 块 {b24:.4f}")
    check(b24 > b1, "码块越多 TB 级 BLER 越高（任一块错则整块错）")

    # -----------------------------------------------------------------------
    sect("6.5  用户提供的表驱动 NewTx/ReTx BLER 曲线")

    cv = la.bc.verify_curves()
    print(f"  {cv['n_mcs']} 个 MCS / {cv['n_curves']} 条曲线 / {cv['n_points']} 个点")
    check(cv["consistent"], "曲线哈希、覆盖、单调性和 10% 门限全部自洽")
    check(cv["hash_matches"], "曲线数据与导入时 SHA-256 一致")
    check(len(la.MCS_TABLE_3) == 28, "表 3 共 28 档，覆盖 MCS 0..27")
    check(max(m.q_m for m in la.MCS_TABLE_3) == 8, "表 3 含 256QAM")

    c15n = la.bc.get_curve(15, "newtx")
    c15r = la.bc.get_curve(15, "retx")
    check(c15n.q_m == 6 and abs(c15n.code_rate - 0.650) < 1e-12,
          "MCS15 NewTx 映射到 64QAM R=0.650")
    check(abs(c15r.code_rate - 0.333) < 1e-12,
          "MCS15 ReTx 映射到 R=0.333")
    check(abs(float(c15n.evaluate(14.00)[0]) - 0.132) < 1e-12 and
          abs(float(c15n.evaluate(14.05)[0]) - 0.0949) < 1e-12,
          "MCS15 NewTx 在原始网格点逐值还原")
    check(abs(c15n.required_sinr_db(0.1) - 14.0421) < 1e-3,
          "MCS15 NewTx 10% BLER 门限为 14.042 dB")
    check(abs(c15r.required_sinr_db(0.1) - 7.7429) < 1e-3,
          "MCS15 ReTx 10% BLER 门限为 7.743 dB")
    check(float(c15n.evaluate(0.0)[0]) == 1.0 and
          abs(float(c15n.evaluate(30.0)[0]) - c15n.bler_points[-1]) < 1e-12,
          "曲线范围外保守钳位，不伪造外推尾部")

    tab = la.link_adaptation(np.full(273, 14.2), n_prb=273, layers=1, mcs_table=3)
    check(tab.mcs_index == 15, "表 3 在 14.2 dB 选择 MCS15")
    check(tab.bler_source == "company_20b_256qam", "结果显式标出表驱动 BLER 来源")
    check(tab.retx_bler is not None and tab.retx_bler < tab.bler,
          "HARQ 首传后使用独立 ReTx 曲线")
    check(tab.harq_model == "newtx_then_retx_curve_reused",
          "结果显式标出多次重传复用 ReTx 曲线的假设")
    check("SINR" in tab.bler_axis_source and "MMSE" in tab.bler_axis_source,
          "结果明确曲线横轴为经典 MMSE 接收机 SINR")

    # -----------------------------------------------------------------------
    sect("6.6  TDD CQI → BF Gain → MCS → OLLA")

    cqi0 = la.cqi_to_mcs_by_se(0)
    check(cqi0["scheduled"] is False and cqi0["mcs"] is None,
          "CQI0 明确表示不调度，不静默降成 MCS0")

    cqi1 = la.cqi_to_mcs_by_se(1)
    check(cqi1["mcs"] == 0 and cqi1["clamped_low"] is True,
          "CQI1 低于公司表最低谱效时钳到 MCS0 并留痕")

    cqi9 = la.cqi_to_mcs_by_se(9)
    check(cqi9["mcs"] == 15 and cqi9["clamped_low"] is False,
          "CQI9 先按谱效映射到公司表 MCS15")

    tdd = la.tdd_mcs_adaptation(
        9,
        [[13.0, 10.0], [15.0, 12.0]],
        [[10.0, 8.0], [12.0, 10.0]],
        olla_mcs_offset=-0.2,
        feedback_ack=False,
    )
    check(tdd["scheduled"] is True and tdd["rank"] == 2 and tdd["n_rb"] == 2,
          "TDD 决策保留逐 RB、逐流维度")
    check(abs(tdd["cqi_mcs_sinr_db"] - 14.0421) < 1e-3,
          "初始 MCS15 转成 NewTx 10% BLER SINR 门限")
    check(tdd["bf_gain_per_stream_db"] == [3.0, 2.0],
          "BF Gain 逐流等于 SVD post-MMSE SINR 减 PMI post-MMSE SINR")
    check(abs(tdd["bf_gain_user_db"] - 2.5) < 1e-12,
          "用户 BF Gain 在所有 RB×流上做 dB 域算术平均")
    check(abs(tdd["user_sinr_db"] - 16.5421) < 1e-3,
          "用户 SINR 等于初始门限叠加逐 RB/流 BF Gain 后的 dB 域平均")
    check("dB domain" in tdd["sinr_aggregation"],
          "结果显式声明 dB 域平均口径")
    check(tdd["mcs_after_bf"] == 17,
          "叠加 BF Gain 后按 NewTx 门限重映射到 MCS17")
    check(abs(tdd["mcs_before_floor"] - 16.8) < 1e-12 and
          tdd["mcs_after_floor"] == 16 and tdd["final_mcs"] == 16,
          "OLLA 在 MCS 域相加后严格向下取整并钳位")
    check(0.0 <= tdd["final_mcs_newtx_bler"] <= 1.0,
          "最终 MCS 返回公司曲线对应的 NewTx BLER")
    check(tdd["receiver"] == "classic MMSE" and
          "only precoding weight changes" in tdd["fairness_contract"],
          "结果钉住经典 MMSE 与只改变预编码权的公平对照")
    check(abs(tdd["olla_update"]["delta_mcs"] + 0.9) < 1e-12 and
          abs(tdd["olla_next_offset_mcs"] + 1.1) < 1e-12,
          "10% 目标下 NACK 令下一时刻 OLLA 减 0.9 MCS")

    ack = la.update_olla_mcs(0.3, True)
    check(abs(ack["next_offset_mcs"] - 0.4) < 1e-12,
          "ACK 令下一时刻 OLLA 加 0.1 MCS")

    floor_edge = la.tdd_mcs_adaptation(
        9, [[14.0]], [[14.0]], olla_mcs_offset=-0.01,
    )
    check(floor_edge["mcs_after_bf"] == 15 and floor_edge["final_mcs"] == 14,
          "极小负 OLLA 也按数学 floor 降一档，不做截零取整")

    try:
        la.tdd_mcs_adaptation(9, [[1.0, 2.0]], [[1.0]])
        shape_rejected = False
    except ValueError:
        shape_rejected = True
    check(shape_rejected, "SVD/PMI 的 RB×流形状不一致时拒绝计算")

    try:
        la.tdd_mcs_adaptation(9, [[float("nan")]], [[1.0]])
        nan_rejected = False
    except ValueError:
        nan_rejected = True
    check(nan_rejected, "非有限 SINR 不进入 BF Gain 与 MCS 决策")

    # -----------------------------------------------------------------------
    sect("7  链路自适应端到端")

    rng = np.random.default_rng(3)
    prev_tp = None
    for mean_db in (-5, 5, 15, 25):
        g = mean_db + rng.normal(0, 3, 273)
        r = la.link_adaptation(g, n_prb=273, layers=2)
        print(f"  SINR≈{mean_db:3d} dB → MCS {r.mcs_index:2d} CQI {r.cqi:2d} "
              f"{r.modulation:<6} 吞吐 {r.throughput_bps/1e6:7.1f} Mbps "
              f"达成 {r.efficiency_vs_shannon:5.1%}")
        check(0 <= r.mcs_index <= 28, "MCS 索引合法")
        check(0 <= r.cqi <= 15, "CQI 合法")
        check(r.bler <= 0.1 + 1e-9 or r.mcs_index == 0, "选中的 MCS 满足目标 BLER")
        check(r.se_achieved <= r.se_shannon, "实际谱效不超香农上界")
        check(r.throughput_bps <= r.throughput_ideal_bps + 1e-6, "有效吞吐不超名义吞吐")
        if prev_tp is not None:
            check(r.throughput_bps >= prev_tp, "信噪比越高吞吐越高")
        prev_tp = r.throughput_bps

    # 目标 BLER 越严，选的 MCS 越保守
    g = np.full(273, 15.0)
    strict = la.link_adaptation(g, n_prb=273, target_bler=0.01)
    loose = la.link_adaptation(g, n_prb=273, target_bler=0.1)
    print(f"  目标 BLER 1% → MCS {strict.mcs_index}，10% → MCS {loose.mcs_index}")
    check(strict.mcs_index <= loose.mcs_index, "目标 BLER 越严 MCS 越保守")

    # 256QAM 表在高信噪比下更强
    hi = np.full(273, 28.0)
    t1 = la.link_adaptation(hi, n_prb=273, mcs_table=1)
    t2 = la.link_adaptation(hi, n_prb=273, mcs_table=2)
    print(f"  28 dB：表1 {t1.throughput_bps/1e6:.0f} Mbps，表2 {t2.throughput_bps/1e6:.0f} Mbps")
    check(t2.throughput_bps > t1.throughput_bps, "高信噪比下 256QAM 表吞吐更高")

    # -----------------------------------------------------------------------
    sect("8  吞吐统计与边缘用户")

    rng = np.random.default_rng(5)
    res = [la.link_adaptation(rng.normal(12, 8) + rng.normal(0, 3, 273), n_prb=273)
           for _ in range(40)]
    st = la.throughput_stats(res)
    print(st.text())
    check(st.n == 40, "样本数正确")
    check(st.cell_edge_mbps <= st.median_mbps <= st.peak_mbps, "5% ≤ 中位 ≤ 95%")
    check(sum(st.mcs_distribution.values()) == 40, "MCS 分布计数完整")
    check("边缘用户" in st.as_dict()["note"], "说明 5% 分位的含义")

    # -----------------------------------------------------------------------
    sect("9  耗时预估的标定")

    pts = [
        (dict(num_sites=1, sectors_per_site=1, num_bs_tx_ant=32, num_rb=51), 24),
        (dict(num_sites=7, sectors_per_site=3, num_bs_tx_ant=32, num_rb=51), 410),
        (dict(num_sites=1, sectors_per_site=1, num_bs_tx_ant=64, num_rb=273), 191),
        (dict(num_sites=7, sectors_per_site=3, num_bs_tx_ant=64, num_rb=273), 2054),
    ]
    for cfg, meas in pts:
        est = gen.estimate_seconds(cfg, 1) * 1000
        err = est / meas - 1
        print(f"  实测 {meas:5d} ms / 预估 {est:6.0f} ms  误差 {err:+.0%}")
        check(abs(err) < 0.35, f"耗时预估误差在 35% 内（实测 {meas} ms）")

    light = dict(num_sites=1, sectors_per_site=1, num_bs_tx_ant=32, num_rb=51)
    heavy = dict(num_sites=7, sectors_per_site=3, num_bs_tx_ant=64, num_rb=273)
    check(gen._resolve_workers("auto", 200, light) == 1, "轻配置不起进程（启动成本不划算）")
    check(gen._resolve_workers("auto", 200, heavy) > 4, "重配置自动多进程")
    check(gen._resolve_workers("auto", 2, heavy) <= 2, "进程数不超过样本数")
    check(gen._resolve_workers(1, 200, heavy) == 1, "显式 workers=1 强制串行")

    # -----------------------------------------------------------------------
    sect("10  并行生成与串行等价")

    cfg = dict(scenario="UMa_NLOS", channel_model="CDL-C", num_sites=7,
               sectors_per_site=3, isd_m=500.0, num_ues=6, num_bs_tx_ant=16,
               num_ue_rx_ant=2, num_ue_tx_ant=2, bandwidth_hz=20e6,
               subcarrier_spacing=30000, carrier_freq_hz=3.5e9, link="DL", seed=42)
    N = 24
    t0 = time.perf_counter()
    s1 = gen.generate(dict(cfg), num_samples=N, workers=1)
    t_ser = time.perf_counter() - t0
    t0 = time.perf_counter()
    sp = gen.generate(dict(cfg), num_samples=N, workers=4)
    t_par = time.perf_counter() - t0
    d1, dp = load(s1["dataset_id"]), load(sp["dataset_id"])
    print(f"  串行 {t_ser:.1f}s / 并行(4) {t_par:.1f}s")
    print(f"  并行摘要 {sp['parallel']}")

    check(s1["num_samples"] == sp["num_samples"] == N, "两种路径样本数一致")
    check(set(d1.keys()) == set(dp.keys()), "字段集一致")
    check(d1.h_true.shape[1:] == dp.h_true.shape[1:], "样本形状一致")
    check(sp["parallel"]["workers"] == 4, "摘要记录了进程数")
    check("统计等价但逐样本不同" in (sp["parallel"]["note"] or ""), "如实说明并行不是逐样本复现")
    check(sp["parallel"]["fallback_reason"] is None, "并行未降级")

    from scipy import stats as sst
    p = float(sst.ks_2samp(d1.sinr_dB, dp.sinr_dB).pvalue)
    print(f"  SINR 分布 KS 检验 p={p:.3f}")
    check(p > 0.01, "串行与并行的 SINR 分布统计上相容")

    # 同 seed 同 workers 必须可复现
    sp2 = gen.generate(dict(cfg), num_samples=N, workers=4)
    check(np.allclose(dp.sinr_dB, load(sp2["dataset_id"]).sinr_dB),
          "同 seed 同 workers 可复现")

    # -----------------------------------------------------------------------
    sect("11  数据集级链路自适应")

    r = d1.link_adaptation(0)
    print(f"  {r.text().splitlines()[0]}")
    check(r.n_re > 0 and r.tbs_bits > 0, "单样本链路自适应可用")
    st = d1.throughput(max_samples=12)
    print(f"  {st.text().splitlines()[0]}")
    check(st.n == 12, "整批吞吐统计可用")
    check(st.mean_mbps > 0, "吞吐为正")

    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    if FAILED:
        print(f"FAILED {len(FAILED)} 项：")
        for f in FAILED:
            print("  -", f)
        sys.exit(1)
    print("链路自适应与并行生成全部通过。")


if __name__ == "__main__":
    main()
