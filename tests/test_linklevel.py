"""链路性能、可信度验证、物理层工具箱的测试。

直接运行：python tests/test_linklevel.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from superwireless import generate as gen  # noqa: E402
from superwireless import linklevel as ll  # noqa: E402
from superwireless import physical as ph  # noqa: E402
from superwireless import plan as pl  # noqa: E402
from superwireless import validate as va  # noqa: E402
from superwireless import load  # noqa: E402

FAILED: list[str] = []


def check(cond: bool, label: str) -> None:
    print(("  PASS  " if cond else "  FAIL  ") + label)
    if not cond:
        FAILED.append(label)


def sect(t: str) -> None:
    print("\n" + "=" * 70 + f"\n{t}\n" + "=" * 70)


def make(n: int = 40, **ov):
    base = {"num_samples": n, "num_ues": max(n // 2, 4), "antenna_preset": "32T4R",
            "bandwidth_hz": 20000000.0}
    base.update(ov)
    d, p = pl.create_draft("验证 CSI 压缩", overrides=base)
    cfg, own = pl.resolved_config(d)
    cfg.pop("num_samples", None)
    s = gen.generate(cfg, num_samples=n)
    return load(s["dataset_id"]), s


# ---------------------------------------------------------------------------
sect("1  谱效链路：预编码 → SINR → 谱效")
ds, summ = make(40)
print(f"  数据集 {summ['shape']}")

r = ds.link(0, snr_db=20.0, method="svd", receiver="mmse")
print(f"  谱效 {r.spectral_efficiency:.3f} bit/s/Hz，容量上界 {r.capacity_bound:.3f}，"
      f"达成 {100*r.spectral_efficiency/r.capacity_bound:.0f}%")
print(f"  逐层 SINR {[round(x,1) for x in r.sinr_per_layer_db]} dB，rank={r.rank}")
check(r.spectral_efficiency > 0, "谱效为正")
check(r.spectral_efficiency <= r.capacity_bound * 1.001, "谱效不超容量上界")
check(len(r.sinr_per_layer_db) == r.rank, "逐层 SINR 数量等于 rank")
check(r.sinr_per_rb_db.shape[0] == ds.h_true.shape[2], "逐 RB SINR 维度正确")

# 信噪比越高谱效越高
se_lo = ll.link_performance(ds.h_true[0], snr_db=5.0).spectral_efficiency
se_hi = ll.link_performance(ds.h_true[0], snr_db=30.0).spectral_efficiency
print(f"  SNR 5dB→30dB：谱效 {se_lo:.2f} → {se_hi:.2f}")
check(se_hi > se_lo, "谱效随信噪比单调上升")

# ---------------------------------------------------------------------------
sect("2  接收机对比")
# SVD 预编码会把有效信道对角化，层间干扰为零，三种接收机等价——
# 那是物理正确的结果，但验证不到接收机之间的差异。改用 identity 预编码
# （不做空间处理，层间干扰保留）才能看出区别。
print("  用 SVD 预编码（有效信道已对角化）：")
svd_rx = {rx: ll.link_performance(ds.h_true[0], snr_db=20.0, method="svd",
                                  receiver=rx).spectral_efficiency
          for rx in ("mmse", "zf", "mrc")}
for rx, v in svd_rx.items():
    print(f"    {rx:<6} {v:7.3f}")
check(max(svd_rx.values()) - min(svd_rx.values()) < 0.01,
      "SVD 预编码下三种接收机等价（层间干扰已消除）")

print("  用 identity 预编码（保留层间干扰）：")
id_rx = {rx: ll.link_performance(ds.h_true[0], snr_db=20.0, method="identity",
                                 max_rank=4, receiver=rx).spectral_efficiency
         for rx in ("mmse", "zf", "mrc")}
for rx, v in id_rx.items():
    print(f"    {rx:<6} {v:7.3f}")
check(id_rx["mmse"] >= id_rx["zf"] - 1e-6, "MMSE ≥ ZF（理论性质）")
check(id_rx["mmse"] >= id_rx["mrc"] - 1e-6, "MMSE ≥ MRC（MRC 不消除层间干扰）")
check(max(id_rx.values()) - min(id_rx.values()) > 0.01, "有层间干扰时接收机确有差异")

# ---------------------------------------------------------------------------
sect("3  预编码方案排序")
cmp = ll.compare_precoders(ds.h_true[:20], snr_db=20.0, n_h=8, n_v=2)
print(f"  {'方案':<14}{'谱效':>9}{'vs SVD':>9}{'收敛':>7}")
for m, v in cmp.items():
    print(f"  {m:<14}{v['se_mean']:>9.3f}{v['vs_svd_pct']:>8.0f}%{str(v['converged']):>7}")
order = ["svd", "svd_wideband", "type1", "dft"]
seq = [cmp[m]["se_mean"] for m in order if m in cmp]
check(all(seq[i] >= seq[i + 1] - 1e-6 for i in range(len(seq) - 1)),
      "SVD ≥ 宽带SVD ≥ TypeI ≥ DFT")
check(cmp["svd"]["vs_svd_pct"] == 100.0, "SVD 作为基准是 100%")

# ---------------------------------------------------------------------------
sect("4  蒙特卡洛收敛判断")
small = ll.monte_carlo(ds.h_true[:5], snr_db=20.0)
big = ll.monte_carlo(ds.h_true, snr_db=20.0)
print(f"  5 个样本   置信区间相对宽度 {small.relative_ci_width:.1%}  收敛={small.converged}")
print(f"  {ds.n} 个样本  置信区间相对宽度 {big.relative_ci_width:.1%}  收敛={big.converged}")
check(big.relative_ci_width < small.relative_ci_width, "样本越多置信区间越窄")
check(big.se_ci95[0] < big.se_mean < big.se_ci95[1], "均值落在置信区间内")
check(set(big.se_percentiles) == {"p5", "p50", "p95"}, "给出分位数")

# ---------------------------------------------------------------------------
sect("5  CSI 误差的代价")
print(f"  数据集自带的估计 NMSE 中位数 {np.median(ds.estimation_error_nmse_db()):.1f} dB")
ideal = ds.monte_carlo(method="svd")
est = ds.monte_carlo(method="svd", channels_for_precoding=ds.h_est)
print(f"  理想 CSI {ideal.se_mean:.3f} → 数据集估计 CSI {est.se_mean:.3f}"
      f"  损失 {100*(1-est.se_mean/max(ideal.se_mean,1e-9)):.2f}%（估计已很准，差异接近噪声）")

# 人为注入不同程度的 CSI 误差，才看得出趋势
tau_s = float(summ["tau_rms_ns"]) * 1e-9
print(f"\n  人为劣化 CSI 后的谱效（10 个样本，SNR=20dB）：")
print(f"    {'导频间隔':<10}{'CSI NMSE':>11}{'谱效':>10}{'相对理想':>10}")
base = ll.monte_carlo(ds.h_true[:10], snr_db=20.0, method="svd").se_mean
trend = []
for sp in (2, 8, 16):
    h_est_list, nmses = [], []
    for i in range(10):
        r = ph.estimate_channel(ds.h_true[i], method="ls", snr_db=15.0,
                                pilot_spacing=sp, tau_rms_s=tau_s)
        h_est_list.append(r["h_hat"])
        nmses.append(r["nmse_db"])
    se = ll.monte_carlo(ds.h_true[:10], snr_db=20.0, method="svd",
                        channels_for_precoding=np.stack(h_est_list)).se_mean
    trend.append(se)
    print(f"    {sp:<10}{np.mean(nmses):>10.1f}dB{se:>10.3f}{100*se/base:>9.0f}%")
check(trend[0] >= trend[-1] - 1e-6, "CSI 越差谱效越低（导频越稀疏损失越大）")
check(all(t <= base * 1.02 for t in trend), "用有误差的 CSI 预编码不优于理想 CSI")

# ---------------------------------------------------------------------------
sect("6  可信度验证")
rep = ds.validate()
print(rep.text())
d_ = rep.as_dict()
check(d_["n_failed_error"] == 0, f"自洽配置全部硬性检查通过（失败 {d_['n_failed_error']} 项）")
check(any(c["name"] == "预编码性能排序" for c in d_["checks"]), "含预编码排序检查")
check(any(c["name"] == "蒙特卡洛收敛" for c in d_["checks"]), "含收敛检查")

# ---------------------------------------------------------------------------
sect("7  验证能抓出矛盾配置")
ds_bad, _ = make(30, scenario="UMa_LOS", channel_model="CDL-C")
rep_bad = ds_bad.validate()
names = [c.name for c in rep_bad.checks if not c.passed]
print(f"  UMa_LOS + CDL-C（视距场景配非视距剖面）→ 未通过：{names}")
check("场景与信道模型自洽" in names, "抓出场景与信道模型矛盾")

ds_ok, _ = make(30, scenario="UMa_LOS", channel_model="CDL-D")
rep_ok = ds_ok.validate()
check(
    "场景与信道模型自洽" not in [c.name for c in rep_ok.checks if not c.passed],
    "UMa_LOS + CDL-D 自洽配置放行",
)

# ---------------------------------------------------------------------------
sect("8  物理层工具箱")
print(f"  100MHz@30kHz -> {ph.nr_rb_count(100e6, 30000)} RB（标准表，非简单除法）")
check(ph.nr_rb_count(100e6, 30000) == 273, "NR RB 表正确")
check(ph.nr_rb_count(20e6, 30000) == 51, "20MHz -> 51 RB")

tdd = ph.tdd_pattern_info("DDDSU")
print(f"  DDDSU: 周期 {tdd.get('period_slots')} 时隙 / {tdd.get('periodicity_ms')} ms，"
      f"特殊时隙 {tdd.get('special_slot')}")
check(tdd.get("period_slots") == 5, "TDD 周期正确")
check(len(ph.list_tdd_patterns()) >= 5, "至少 5 种 TDD 配比")

srs = ph.srs_config(273, b_srs=1, b_hop=0)
print(f"  SRS b_srs=1: 跳频周期 {srs['hopping_cycle_length']}，"
      f"每跳 {srs['rb_per_hop']} RB，覆盖 {srs['coverage_ratio']:.0%}")
check(srs["hopping_enabled"], "b_hop < b_srs 时跳频启用")
check(srs["hopping_cycle_length"] > 1, "跳频周期大于 1")
check(not ph.srs_config(273, b_srs=0)["hopping_enabled"], "b_srs=0 时不跳频")

zc = ph.zadoff_chu(25, 139)
c = ph.sequence_correlation(zc)
print(f"  ZC 序列：恒模误差 {np.std(np.abs(zc)):.2e}，自相关峰旁比 {c['peak_to_sidelobe_db']:.0f} dB")
check(np.std(np.abs(zc)) < 1e-6, "ZC 恒模")
check(c["peak_to_sidelobe_db"] > 60, "ZC 理想周期自相关")

ssb = ph.ssb_sequences(42)
check(ssb["pss"].shape == (127,) and ssb["sss"].shape == (127,), "PSS/SSS 长度 127")
cb = ph.dft_codebook(8, 4, 2)
print(f"  DFT 码本 8H4V2P -> {cb.shape}")
check(cb.shape[1] == 64, "码本端口数 = 8×4×2")

# ---------------------------------------------------------------------------
sect("9  信道估计基线")
tau = float(summ["tau_rms_ns"]) * 1e-9
print(f"  {'方法':<7}{'SNR=5dB':>10}{'SNR=15dB':>11}{'SNR=25dB':>11}")
res = {}
for m in ("ls", "mmse"):
    row = [ph.estimate_channel(ds.h_true[0], method=m, snr_db=x, pilot_spacing=2,
                               tau_rms_s=tau)["nmse_db"] for x in (5, 15, 25)]
    res[m] = row
    print(f"  {m:<7}{row[0]:>10.2f}{row[1]:>11.2f}{row[2]:>11.2f}")
check(all(res["mmse"][i] <= res["ls"][i] + 0.1 for i in range(3)), "MMSE 不劣于 LS")
check(res["ls"][2] < res["ls"][0], "信噪比越高 LS 估计越准")
check(ph.estimate_channel(ds.h_true[0], method="ideal")["nmse_db"] < -100, "理想模式无误差")

# ---------------------------------------------------------------------------
sect("10  IRC：干扰抑制合并")

# 造一组可控干扰：每个干扰小区秩 1，方向互相正交 —— 这是 IRC 的教科书工况，
# 4 根接收天线应当能把 3 个这样的干扰全部零陷掉。
_rng = np.random.default_rng(3)
_RB, _BS, _UE, _K = 6, 8, 4, 3
_h = (_rng.standard_normal((1, _RB, _BS, _UE))
      + 1j * _rng.standard_normal((1, _RB, _BS, _UE))) / np.sqrt(2)
_dirs = np.linalg.qr(_rng.standard_normal((_UE, _UE))
                     + 1j * _rng.standard_normal((_UE, _UE)))[0]
_hi = np.zeros((_K, 1, _RB, _BS, _UE), dtype=complex)
for _k in range(_K):
    _a = (_rng.standard_normal((_BS, 1)) + 1j * _rng.standard_normal((_BS, 1))) / np.sqrt(2)
    for _f in range(_RB):
        _hi[_k, 0, _f] = _a @ _dirs[:, _k][None, :]      # 秩 1，方向固定且互相正交

_C = ll.interference_covariance(_hi, model="precoded")
check(_C.shape == (_RB, _UE, _UE), f"R_uu 形状 [RB,UE,UE]（实得 {_C.shape}）")
_er = ll.effective_rank(_C)
print(f"  R_uu 有效秩 {_er:.2f}（{_K} 个正交秩 1 干扰）")
check(abs(_er - _K) < 0.01, f"有效秩等于独立干扰方向数（实得 {_er}）")

_mmse = ll.link_performance(_h, snr_db=20.0, receiver="mmse", h_interferers=_hi)
_irc = ll.link_performance(_h, snr_db=20.0, receiver="irc", h_interferers=_hi)
print(f"  MMSE {_mmse.spectral_efficiency:.3f} → IRC {_irc.spectral_efficiency:.3f} "
      f"bit/s/Hz（+{_irc.spectral_efficiency - _mmse.spectral_efficiency:.3f}）")
check(_irc.spectral_efficiency > _mmse.spectral_efficiency,
      "干扰有空间结构时 IRC 严格优于把干扰当白噪声的 MMSE")
check(_irc.receiver == "irc" and _mmse.receiver == "mmse", "接收机名字如实带回结果")
check(_irc.interference_rank is not None and _irc.interference_model == "precoded",
      "R_uu 的秩与建模方式跟着结果一起走")

# **干扰真白时两者必须重合。** 这条是反向自检：IRC 的增益只能来自非白性，
# 如果白干扰下 IRC 还"更好"，那就是实现里多算了什么。
_wh = np.zeros((_UE, 1, _RB, _BS, _UE), dtype=complex)
for _k in range(_UE):
    for _f in range(_RB):
        _wh[_k, 0, _f] = np.ones((_BS, 1)) @ np.eye(_UE)[:, _k][None, :] / np.sqrt(_BS)
_m2 = ll.link_performance(_h, snr_db=20.0, receiver="mmse", h_interferers=_wh)
_i2 = ll.link_performance(_h, snr_db=20.0, receiver="irc", h_interferers=_wh)
print(f"  白干扰下 MMSE {_m2.spectral_efficiency:.4f} / IRC {_i2.spectral_efficiency:.4f}")
check(abs(_i2.spectral_efficiency - _m2.spectral_efficiency) < 1e-6,
      "干扰各向同性时 IRC 与 MMSE 必须重合（没有结构可利用）")

# 无干扰时两者也必须完全一样，且不带 R_uu 元信息
_n1 = ll.link_performance(_h, snr_db=20.0, receiver="mmse")
_n2 = ll.link_performance(_h, snr_db=20.0, receiver="irc")
check(abs(_n1.spectral_efficiency - _n2.spectral_efficiency) < 1e-12, "无干扰时 IRC 退化成 MMSE")
check(_n2.interference_rank is None and _n2.r_uu_source is None, "没干扰就不报 R_uu 元信息")

# 样本协方差 + 对角加载：接收机真实能拿到的东西，不该优于真值
_s1 = ll.link_performance(_h, snr_db=20.0, receiver="irc", h_interferers=_hi,
                          r_uu_source="sample", r_uu_samples=8, diagonal_loading=0.01)
print(f"  R_uu 真值 {_irc.spectral_efficiency:.3f} → 样本估计 {_s1.spectral_efficiency:.3f}")
check(_s1.spectral_efficiency <= _irc.spectral_efficiency + 1e-9,
      "样本估计的 R_uu 不会优于真值（真值是上界）")
check(_s1.r_uu_source == "sample", "R_uu 来源如实带回")

# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
if FAILED:
    print(f"FAILED {len(FAILED)} 项：")
    for f in FAILED:
        print("  - " + f)
    sys.exit(1)
print("链路性能、可信度验证、物理层工具箱全部通过。")
