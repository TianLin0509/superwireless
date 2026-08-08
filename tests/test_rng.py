"""系统级随机数体系：分流、重复、公共随机数、置信区间。

分节：
1. 派生机制——显式 spawn_key 与 numpy 的 .spawn() 必须逐位等价
2. 子流互相独立：改一个流的消耗不影响另一个流
3. `seed+1` 与 spawn 的差别——**实测证明的是"撞车"，不是"相关"**
4. 同 master seed 完全可复现
5. 置信区间随重复次数按 1/√n 收窄
6. 公共随机数（CRN）让 A/B 的置信区间更窄——实测对比
7. 效应小于置信区间时必须判"不显著"
8. 接进系统级仿真：分流、多重复、建表与种子无关
9. 配对契约与重复次数下界
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("SUPERWIRELESS_NO_BROWSER", "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402

from superwireless import gates as gt  # noqa: E402
from superwireless import rng as rg  # noqa: E402
from superwireless import system as sy  # noqa: E402

_n_pass = 0
_n_fail = 0


def check(cond: bool, msg: str) -> None:
    global _n_pass, _n_fail  # noqa: PLW0603
    if cond:
        _n_pass += 1
        print(f"  PASS  {msg}")
    else:
        _n_fail += 1
        print(f"  FAIL  {msg}")


def section(title: str) -> None:
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def fake_tables(n_ue: int = 6, n_snap: int = 8, sinr_lo: float = 2.0,
                sinr_hi: float = 24.0, seed: int = 0) -> list[sy.UeLinkTable]:
    """一批链路表。**不跑 SVD**——这里测的是随机数体系，不是物理。"""
    r = np.random.default_rng(seed)
    geo = np.linspace(sinr_hi, sinr_lo, n_ue)
    out = []
    for u in range(n_ue):
        s = geo[u] + r.normal(0, 1.5, (n_snap, 4))
        mcs = np.clip(((s + 6.0) * 0.9).astype(int), 0, 27)
        se = np.maximum(0.1, (mcs + 1) * 0.22 * np.arange(1, 5)[None, :])
        best = np.argmax(se, axis=1)
        out.append(sy.UeLinkTable(
            ue=u, sinr_db=s, mcs=mcs, se=se, best_rank=best + 1,
            best_se=se[np.arange(n_snap), best], geo_sinr_db=float(geo[u]),
            outage=np.zeros(n_snap, dtype=bool), iot_db=12.0, sir_db=15.0,
            se_gnb=se, best_se_gnb=se[np.arange(n_snap), best]))
    return out


def flat_tables(n_ue: int = 6, n_snap: int = 4) -> list[sy.UeLinkTable]:
    """所有用户**完全相同**的表。用来逼出调度器的平局。"""
    s = np.full((n_snap, 4), 15.0)
    mcs = np.full((n_snap, 4), 14, dtype=int)
    se = np.tile(np.array([3.0, 5.0, 4.0, 3.5]), (n_snap, 1))
    best = np.argmax(se, axis=1)
    return [sy.UeLinkTable(ue=u, sinr_db=s, mcs=mcs, se=se, best_rank=best + 1,
                           best_se=se[np.arange(n_snap), best], geo_sinr_db=15.0,
                           outage=np.zeros(n_snap, dtype=bool), iot_db=10.0,
                           sir_db=15.0, se_gnb=se,
                           best_se_gnb=se[np.arange(n_snap), best])
            for u in range(n_ue)]


# ---------------------------------------------------------------------------
section("1  派生机制：显式 spawn_key 必须与 numpy 的 .spawn() 逐位等价")
# ---------------------------------------------------------------------------
# 本模块不用有状态的 .spawn()，改用显式 spawn_key。**这不是另起炉灶**：
# numpy 的 spawn() 内部就是构造 SeedSequence(entropy, spawn_key=key+(i,))，
# 下面把这条等价性逐位钉死。钉不住就说明我们用的不是 numpy 认可的那套派生。
_ss = np.random.SeedSequence(12345)
_kids = _ss.spawn(4)
check(all(np.array_equal(
        _kids[i].generate_state(8),
        np.random.SeedSequence(12345, spawn_key=(i,)).generate_state(8))
        for i in range(4)),
      "SeedSequence(s).spawn(k)[i] 与 SeedSequence(s, spawn_key=(i,)) 逐位相同")
check(np.array_equal(
        np.random.SeedSequence(7, spawn_key=(3,)).spawn(3)[2].generate_state(8),
        np.random.SeedSequence(7, spawn_key=(3, 2)).generate_state(8)),
      "嵌套 spawn 等价于两级 spawn_key（(3,) 再 spawn 第 2 个 = (3,2)）")

# 派生必须是**纯函数**：与调用顺序、与别的流有没有被创建过都无关。
# 用 .spawn() 的话 n_children_spawned 会推进，先要 traffic 还是先要 harq
# 拿到的流就不一样——一次无关的代码顺序调整就能悄悄换掉全部随机数。
_bk = rg.RngBook(42, 3)
_first = _bk.generator("harq").random(5)
_ = _bk.generator("traffic").random(10_000)
_ = _bk.generator("scheduler").random(999)
check(np.array_equal(_first, _bk.generator("harq").random(5)),
      "先取别的流不影响 harq 流（派生是纯函数，不是有状态的 spawn）")

check(rg.stream_key("traffic") == rg.stream_key("traffic"), "流键在同进程内稳定")
check(rg.stream_key("traffic") != rg.stream_key("harq"), "不同流名给出不同的流键")
# Python 内置 hash() 对 str 是**每进程随机加盐**的，拿它当种子会让
# "同 master seed 完全可复现"在换进程时失效，而本进程内自洽——最难查的一类。
import zlib  # noqa: E402

check(rg.stream_key("traffic") == zlib.crc32(b"traffic") == 1432364592,
      f"流键是 crc32 而不是 hash()（跨进程稳定，实得 {rg.stream_key('traffic')}）")

try:
    rg.RngBook(0, 0).generator("trafic")      # 故意拼错
    check(False, "拼错的流名应当被拒")
except ValueError as _e:
    check("未登记" in str(_e), "拼错流名被硬拦截（否则会悄悄多出一条独立的流）")

# ---------------------------------------------------------------------------
section("2  子流互相独立：改一个流的消耗不影响另一个流")
# ---------------------------------------------------------------------------
# **这是分流的全部意义。** 分流之前 simulate() 里一个 rng 同时喂话务和 HARQ：
# 改一下 arrival_rate_hz，抽到的到达次数变了，HARQ 的伯努利序列整个错位，
# 于是"话务模型的影响"里混着"HARQ 换了一批随机数"。下面把两种写法摆在一起。
_shared = np.random.default_rng(0)
_ = _shared.random(37)                        # 假装话务抽了 37 次
_harq_a = _shared.random(6)
_shared2 = np.random.default_rng(0)
_ = _shared2.random(41)                       # 换个话务参数，抽了 41 次
_harq_b = _shared2.random(6)
check(not np.array_equal(_harq_a, _harq_b),
      "【反面】共用一个 rng 时，话务多抽 4 次就把 HARQ 序列整个挪位")

_bk = rg.RngBook(0, 0)
_g = _bk.generator("traffic")
_ = _g.random(37)
_h1 = _bk.generator("harq").random(6)
_g = _bk.generator("traffic")
_ = _g.random(41)
_h2 = _bk.generator("harq").random(6)
check(np.array_equal(_h1, _h2), "【正面】分流后话务抽多少次都不动 HARQ 流")

# 五条流互相之间不能有任何重合
_streams = sorted(rg.STREAMS)
_draws = {s: rg.RngBook(11, 2).generator(s).random(200) for s in _streams}
_worst = 0.0
for i, a in enumerate(_streams):
    for b in _streams[i + 1:]:
        check(not np.array_equal(_draws[a], _draws[b]), f"{a} 与 {b} 不是同一条流")
        _worst = max(_worst, abs(float(np.corrcoef(_draws[a], _draws[b])[0, 1])))
check(_worst < 0.25, f"五条流两两相关系数都在噪声量级（最大 |r|={_worst:.3f}）")
check(set(_streams) >= {"channel", "traffic", "scheduler", "harq", "neighbor_load"},
      f"至少分出五条流：{_streams}")

# ---------------------------------------------------------------------------
section("3  `seed+1` 与 spawn 的差别（老实说：证明的是撞车，不是相关）")
# ---------------------------------------------------------------------------
# **原始猜想「seed 与 seed+1 会给出相关的流」在现代 numpy 上是错的**，
# 下面这条实测就是为了不让它进结论：SeedSequence 用带雪崩效应的整数散列
# 混合 entropy，"Two input seeds that are very close to each other will produce
# initial states that are very far from each other"（numpy 并行随机数文档）。
_c = [float(np.corrcoef(np.random.default_rng(s).random(2000),
                        np.random.default_rng(s + 1).random(2000))[0, 1])
      for s in range(200)]
check(max(abs(x) for x in _c) < 0.12,
      f"相邻种子的流**测不出相关**（200 对里最大 |r|={max(abs(x) for x in _c):.4f}）"
      f"——所以「seed+1 会相关」这条主张不成立，已从设计理由里去掉")

# **真正的问题是跨批次撞车**，numpy 文档把 `root_seed + worker_id` 明确标成
# "UNSAFE! Do not do this!"，理由正是 "multiple invocations of the program with
# different seeds will get overlapping sets of worker seeds"。
_camp1 = [np.random.default_rng(100 + i).random(5) for i in range(8)]
_camp2 = [np.random.default_rng(105 + i).random(5) for i in range(8)]
_dups = sum(1 for a in _camp1 for b in _camp2 if np.array_equal(a, b))
check(_dups == 3,
      f"【反面】seed+1：两批（base 100 / 105）之间有 {_dups} 对流**逐位相同**"
      f"——不是相关，是同一条流被当成两次独立重复")

_s1 = [rg.RngBook(100, i).generator("traffic").random(5) for i in range(8)]
_s2 = [rg.RngBook(105, i).generator("traffic").random(5) for i in range(8)]
check(sum(1 for a in _s1 for b in _s2 if np.array_equal(a, b)) == 0,
      "【正面】spawn_key 派生：换 master 后两批 16 条流没有一条重合")
# 换 master 也不能让某条流退化成另一条流的平移
check(not np.array_equal(rg.RngBook(0, 1).generator("harq").random(50),
                         rg.RngBook(1, 0).generator("harq").random(50)),
      "(master=0,rep=1) 与 (master=1,rep=0) 不是同一条流")

# ---------------------------------------------------------------------------
section("4  同 master seed 完全可复现")
# ---------------------------------------------------------------------------
for _s in ("traffic", "harq", "scheduler", "neighbor_load", "channel"):
    check(np.array_equal(rg.RngBook(2024, 5).generator(_s).random(64),
                         rg.RngBook(2024, 5).generator(_s).random(64)),
          f"{_s} 流两次派生逐位相同")
check(rg.derive_generate_seed(7, 0) == rg.derive_generate_seed(7, 0)
      and rg.derive_generate_seed(7, 0) != rg.derive_generate_seed(7, 1),
      "derive_generate_seed 可复现且逐 replication 不同")
check(0 <= rg.derive_generate_seed(7, 0) < 2 ** 63,
      f"派生的整数种子落在 63 位内（实得 {rg.derive_generate_seed(7, 0)}）")
_bks = rg.replications(3, 4, start=2)
check([b.replication for b in _bks] == [2, 3, 4, 5]
      and all(b.master_seed == 3 for b in _bks),
      "replications(start=...) 只推进 replication，不动 master_seed")

_t = fake_tables()
_c1 = sy.simulate(_t, sys_cfg=sy.SystemConfig(duration_s=2.0),
                  rng=rg.RngBook(5, 1)).cell
_c2 = sy.simulate(_t, sys_cfg=sy.SystemConfig(duration_s=2.0),
                  rng=rg.RngBook(5, 1)).cell
check(all(_c1[k] == _c2[k] for k in _c1 if isinstance(_c1[k], (int, float))),
      "整条系统级仿真在同一个 RngBook 下逐位复现")
check(_c1["cell_experienced_mbps"]
      != sy.simulate(_t, sys_cfg=sy.SystemConfig(duration_s=2.0),
                     rng=rg.RngBook(5, 2)).cell["cell_experienced_mbps"],
      "换 replication 确实换了一批随机数（结果不同）")

# ---------------------------------------------------------------------------
section("5  置信区间随重复次数按 1/√n 收窄")
# ---------------------------------------------------------------------------
from scipy import stats  # noqa: E402

_r = np.random.default_rng(0)
_pop = _r.normal(150.0, 17.0, 4096)
_hw = {}
for _n in (4, 8, 16, 64, 256):
    _st = rg.summarize(_pop[:_n])
    _hw[_n] = _st.half_width
    _want = float(stats.t.ppf(0.975, _n - 1)) * _st.std / np.sqrt(_n)
    check(abs(_st.half_width - _want) < 1e-9,
          f"n={_n} 半宽 = t_{{0.975,n-1}}·s/√n（{_st.half_width:.4f}）")
print(f"  半宽：{ {k: round(v, 3) for k, v in _hw.items()} }")
check(_hw[4] > _hw[16] > _hw[64] > _hw[256], "半宽随 n 单调收窄")

# **"按 1/√n 收窄"精确成立的是标准误 s/√n，不是置信区间半宽。**
# 半宽还乘着 t_{0.975,n-1}，小 n 时 t 大得多（t₃=3.18 vs t₁₅=2.13），
# 所以半宽实际收得**比 1/√n 更快**。混为一谈会在 n=4→16 上差 30%，
# 下面把两件事分开钉：先用"人为固定样本标准差"的构造消掉抽样噪声。
def _fixed_std(n: int, s0: float = 17.0, mean: float = 150.0) -> np.ndarray:
    z = np.random.default_rng(n).normal(0, 1, n)
    z = (z - z.mean()) / z.std(ddof=1)        # 标准差精确等于 1
    return mean + s0 * z


for _a, _b in ((4, 16), (8, 32), (16, 64), (64, 256)):
    _sa, _sb = rg.summarize(_fixed_std(_a)), rg.summarize(_fixed_std(_b))
    check(abs(_sa.sem / _sb.sem - np.sqrt(_b / _a)) < 1e-9,
          f"标准误 n {_a}→{_b} 精确按 1/√n 收窄（比值 {_sa.sem / _sb.sem:.4f}"
          f" = √{_b // _a}）")
    _got = _sb.half_width / _sa.half_width
    _ideal = float(np.sqrt(_a / _b) * stats.t.ppf(0.975, _b - 1)
                   / stats.t.ppf(0.975, _a - 1))
    check(abs(_got - _ideal) < 1e-9,
          f"半宽 n {_a}→{_b} 比 {_got:.4f} = 1/√n × t 修正（纯 1/√n 会是 "
          f"{np.sqrt(_a / _b):.3f}）")

# 蒙特卡洛：随机抽样下期望半宽的比值同样落在解析预测上。
# 预测里还要带样本标准差的偏置修正 c4(n)=√(2/(n−1))·Γ(n/2)/Γ((n−1)/2)——
# E[s] = c4·σ，n 小时 s 系统性偏小，这一项在 n=4 上还有 8%。
from scipy.special import gammaln  # noqa: E402


def _c4(n: int) -> float:
    return float(np.sqrt(2.0 / (n - 1)) * np.exp(gammaln(n / 2) - gammaln((n - 1) / 2)))


_mc = {n: float(np.mean([rg.summarize(_r.normal(0, 1, n)).half_width
                         for _ in range(1500)])) for n in (4, 16, 64)}
for _a, _b in ((4, 16), (16, 64)):
    _got = _mc[_b] / _mc[_a]
    _pred = float(np.sqrt(_a / _b) * stats.t.ppf(0.975, _b - 1)
                  / stats.t.ppf(0.975, _a - 1) * _c4(_b) / _c4(_a))
    check(abs(_got / _pred - 1.0) < 0.06,
          f"蒙特卡洛平均半宽 n {_a}→{_b} 比值 {_got:.3f}，解析预测 {_pred:.3f}")

check(np.isnan(rg.summarize([3.0]).std) and np.isnan(rg.summarize([3.0]).half_width),
      "n=1 时 std/半宽是 nan 而不是 0——一次运行测不出离散度")
check("num_replications" in (rg.summarize([3.0]).as_dict().get("note") or ""),
      "n=1 的 as_dict 带一条明确告警")
check(rg.summarize([3.0]).as_dict()["ci95"] == [None, None],
      "nan 不进 JSON（写成 null，不是 NaN 字面量）")

# ---------------------------------------------------------------------------
section("6  公共随机数（CRN）让 A/B 的置信区间更窄")
# ---------------------------------------------------------------------------
# 原理：Var(a−b) = Var(a) + Var(b) − 2·Cov(a,b)。CRN 就是把那个协方差做正。
# 先用一个干净的构造把机理钉死：共同噪声 σ=10，真实效应 +1。
_rr = np.random.default_rng(3)
_N = 8
_common = _rr.normal(0, 10.0, _N)
_a_crn = _common + 1.0 + _rr.normal(0, 0.3, _N)
_b_crn = _common + _rr.normal(0, 0.3, _N)
_b_ind = _rr.normal(0, 10.0, _N) + _rr.normal(0, 0.3, _N)
_p_crn = gt.paired_compare(_a_crn, _b_crn)
_p_ind = gt.paired_compare(_a_crn, _b_ind)
_h_crn = (_p_crn.ci_high - _p_crn.ci_low) / 2
_h_ind = (_p_ind.ci_high - _p_ind.ci_low) / 2
print(f"  构造：CRN 半宽 {_h_crn:.3f}，独立 {_h_ind:.3f}，比值 {_h_ind / _h_crn:.1f}x")
check(_h_ind > 5 * _h_crn, f"共同噪声被差分抵消，CRN 区间窄 {_h_ind / _h_crn:.0f} 倍")
check(_p_crn.ci_excludes_zero and not _p_ind.ci_excludes_zero,
      "同一个真实效应：CRN 下判显著，独立随机数下判不出来")

# 再在**真的系统级仿真**上量一遍：两臂只差 PF 窗长。
_T = fake_tables(n_ue=8, seed=4)
_books = rg.replications(0, 8)


def _run(books, window):
    return np.array([sy.simulate(
        _T, sys_cfg=sy.SystemConfig(duration_s=3.0),
        sched=sy.SchedulerConfig(pf_window_tti=window), rng=b).cell[
            "cell_experienced_mbps"] for b in books])


_A = _run(_books, 50)
_B_crn = _run(_books, 500)
_B_ind = _run(rg.replications(987654321, 8), 500)
_cmp_crn = rg.compare_replications(_A, _B_crn, arm_a="pf_window=50",
                                   arm_b="pf_window=500",
                                   books_a=_books, books_b=_books)
_cmp_ind = rg.compare_replications(_A, _B_ind, arm_a="pf_window=50",
                                   arm_b="pf_window=500", books_a=_books,
                                   books_b=rg.replications(987654321, 8),
                                   require_crn=False)
_r_crn, _r_ind = _cmp_crn["ci95_half_width"], _cmp_ind["ci95_half_width"]
print(f"  系统级：CRN 半宽 {_r_crn:.3f} Mbps，独立 {_r_ind:.3f} Mbps，"
      f"比值 {_r_ind / _r_crn:.2f}x")
check(_r_ind > _r_crn * 1.3,
      f"系统级实测：独立随机数的置信区间比 CRN 宽 {_r_ind / _r_crn:.2f} 倍")
check(_cmp_crn["crn"] is True and _cmp_ind["crn"] is False,
      "结果里显式标出这次比较是不是 CRN")
check(_cmp_crn["paired"]["std_diff"] < _cmp_ind["paired"]["std_diff"],
      f"差值标准差：CRN {_cmp_crn['paired']['std_diff']:.3f} < "
      f"独立 {_cmp_ind['paired']['std_diff']:.3f}")
# CRN 省下来的样本量是可以算的（gates.required_samples 的直接后果）
_need_crn = gt.required_samples(_cmp_crn["paired"]["std_diff"], 2.0)
_need_ind = gt.required_samples(_cmp_ind["paired"]["std_diff"], 2.0)
print(f"  检出 2 Mbps 的效应：CRN 需 {_need_crn} 次重复，独立需 {_need_ind} 次")
check(_need_crn < _need_ind, "同一效应下 CRN 需要的重复次数更少")

# ---------------------------------------------------------------------------
section("7  效应小于置信区间时必须判「不显著」")
# ---------------------------------------------------------------------------
# **这一节针对的是真实事故**：上一轮报「Type I 权老化后 +14%」，
# 而同配置只改种子的噪声 1σ 就有 11.4%。
_rr = np.random.default_rng(9)
_base = _rr.normal(152.0, 17.0, 8)
_noise_only = _base + _rr.normal(0.0, 17.0, 8)      # 两臂其实没有差别
_c = rg.compare_replications(_noise_only, _base, metric="cell_experienced_mbps",
                             arm_a="新算法", arm_b="基线")
print("  ", _c["verdict_text"])
check(_c["verdict"] == "inconclusive", "纯噪声的两臂判为 inconclusive")
check(_c["effect_exceeds_ci"] is False, "effect_exceeds_ci 明确为 False")
check("不能下结论" in _c["verdict_text"] and "不要报这个百分比" in _c["verdict_text"],
      "判决语明确说「不能下结论，也不要报这个百分比」")
check("至少需要" in _c["verdict_text"], "顺带给出需要多少次重复才能站住")
check(abs(_c["effect"]) < _c["ci95_half_width"],
      f"|效应| {abs(_c['effect']):.3f} < 半宽 {_c['ci95_half_width']:.3f}"
      f"（与「区间跨零」是同一件事）")
check(_c["gate_conclusion"]["passed"] is False, "门 3 同步判不通过（复用 gates.py）")

# 真有效应时必须判得出来
_strong = _base + 40.0
_c2 = rg.compare_replications(_strong, _base, arm_a="强效应", arm_b="基线")
print("  ", _c2["verdict_text"])
check(_c2["verdict"] == "significant", "真实大效应判为 significant")
check(_c2["effect_exceeds_ci"] is True and _c2["gate_conclusion"]["passed"] is True,
      "区间不跨零且门 3 通过")
check(_c2["paired"]["decision_test"] in ("wilcoxon", "paired_t")
      and _c2["paired"]["decision_test"] in _c2["verdict_text"].replace(
          "Wilcoxon 符号秩检验", "wilcoxon").replace("配对 t 检验", "paired_t"),
      f"判决语写明用的是哪个检验（{_c2['paired']['decision_test']}）")
# 区间不跨零但非参检验不显著时同样不能放行（CLAUDE.md 门 3 那条血泪）
_d = np.array([-0.0811, 1.5561, 0.5308, 1.9896, 3.2605, -0.1125, 1.6908, -0.2045])
_c3 = rg.compare_replications(_d, np.zeros(8), arm_a="回归样本", arm_b="零")
check(_c3["paired"]["t_significant"] and not _c3["paired"]["wilcoxon_significant"],
      "回归样本：t 显著、Wilcoxon 不显著（与 test_gates 第 6.5 节同一组数）")
check(_c3["verdict"] == "inconclusive",
      "两检验冲突时以 Wilcoxon 为准，判 inconclusive（不走 t 的宽松通道）")

# ---------------------------------------------------------------------------
section("8  接进系统级仿真：分流、多重复、建表与种子无关")
# ---------------------------------------------------------------------------
_T = fake_tables(n_ue=8, seed=6)
_rep = sy.simulate_replications(_T, num_replications=8, master_seed=0,
                               sys_cfg=sy.SystemConfig(duration_s=3.0))
check(_rep.n_rep == 8, f"跑满 8 次重复（实得 {_rep.n_rep}）")
for _k in ("cell_experienced_mbps", "ue_experienced_p5_mbps", "avg_mcs",
           "avg_rank", "bler_first_tx", "occupancy"):
    _d = _rep.cell[_k]
    check(all(x in _d for x in ("mean", "std", "ci95", "n_rep"))
          and _d["n_rep"] == 8,
          f"{_k} 报 mean/std/ci95/n_rep（{_d['mean']} ± {_d['rel_half_width']}）")
check(_rep.cell["cell_experienced_mbps"]["ci95"][0]
      < _rep.cell["cell_experienced_mbps"]["mean"]
      < _rep.cell["cell_experienced_mbps"]["ci95"][1],
      "均值落在自己的置信区间里")
check(all(isinstance(u["experienced_mbps"], dict) and "ci95" in u["experienced_mbps"]
          for u in _rep.users), "用户级体验速率也带区间")
check("95% 置信区间" in _rep.text(), "summary 文本里带区间")
check(_rep.as_dict()["replications"] == list(range(8)),
      "结果里记下用了哪几个 replication（配对的 ID 契约）")
check(_rep.runs[0].config["rng"]["master_seed"] == 0
      and "spawn_key" in _rep.runs[0].config["rng"]["scheme"],
      "单次结果里也记下随机数方案")

# 重复次数少时区间必须更宽——这是"多跑几次"值不值的直接依据
_rep2 = sy.simulate_replications(_T, num_replications=2, master_seed=0,
                                 sys_cfg=sy.SystemConfig(duration_s=3.0))
check(_rep2.cell["cell_experienced_mbps"]["rel_half_width"]
      > _rep.cell["cell_experienced_mbps"]["rel_half_width"],
      f"n=2 的相对半宽 {_rep2.cell['cell_experienced_mbps']['rel_half_width']:.3f}"
      f" > n=8 的 {_rep.cell['cell_experienced_mbps']['rel_half_width']:.3f}")
check(any("Wilcoxon" in s for s in _rep2.notes),
      "n≤5 时 notes 里明说判决检验永远不可能显著")

# **建表与种子无关**——这是"只建一次表"的前提，破了整个优化就不成立
_r0 = np.random.default_rng(0)
_hs = [((_r0.standard_normal((4, 24, 16, 4))
         + 1j * _r0.standard_normal((4, 24, 16, 4))) / np.sqrt(2)).astype(np.complex64)
       for _ in range(4)]
_geo = [18.0, 12.0, 6.0, 0.0]
_tb_a = sy.build_link_tables(_hs, _geo, num_snapshots=4)
_tb_b = sy.build_link_tables(_hs, _geo, num_snapshots=4)
check(all(np.array_equal(a.se, b.se) and np.array_equal(a.mcs, b.mcs)
          and np.array_equal(a.best_rank, b.best_rank)
          for a, b in zip(_tb_a, _tb_b, strict=True)),
      "build_link_tables 逐位确定：与随机种子无关，所以只建一次是安全的")

# 调度器决胜流：所有用户完全相同时，平局必须随机打破而不是按 UE 编号
_flat = flat_tables(n_ue=6)
_rf = sy.simulate(_flat, sys_cfg=sy.SystemConfig(duration_s=3.0),
                  traffic=sy.TrafficConfig(model="full_buffer"),
                  sched=sy.SchedulerConfig(algorithm="max_ci", mu_enabled=False),
                  rng=rg.RngBook(0, 0))
_share = np.array([u["sched_tti"] for u in _rf.users], dtype=float)
_share = _share / max(_share.sum(), 1)
print(f"  平局时各 UE 的调度份额：{np.round(_share, 3).tolist()}")
check(_share.max() < 0.5,
      f"max-C/I + 完全相同的用户：调度份额随机摊开（最大 {_share.max():.3f}）"
      f"——按 UE 编号决胜的话 UE0 会独吞 100%")
check(_share.min() > 0.05, f"没有用户被系统性饿死（最小份额 {_share.min():.3f}）")
_rf2 = sy.simulate(_flat, sys_cfg=sy.SystemConfig(duration_s=3.0),
                   traffic=sy.TrafficConfig(model="full_buffer"),
                   sched=sy.SchedulerConfig(algorithm="max_ci", mu_enabled=False),
                   rng=rg.RngBook(0, 1))
check([u["sched_tti"] for u in _rf.users] != [u["sched_tti"] for u in _rf2.users],
      "换 replication 会换一套决胜结果（决胜确实走了 scheduler 流）")

# 老接口不给 rng 时按 sys_cfg.seed 走，行为可复现
_o1 = sy.simulate(_T, sys_cfg=sy.SystemConfig(duration_s=2.0, seed=3)).cell
_o2 = sy.simulate(_T, sys_cfg=sy.SystemConfig(duration_s=2.0),
                  rng=rg.RngBook(master_seed=3, replication=0)).cell
check(_o1["cell_experienced_mbps"] == _o2["cell_experienced_mbps"],
      "不给 rng 时等价于 RngBook(master_seed=sys_cfg.seed, replication=0)")

# ---------------------------------------------------------------------------
section("9  配对契约与重复次数下界")
# ---------------------------------------------------------------------------
# **配对的有效性靠 ID 契约，统计查不出错位**（CLAUDE.md 里那条实测：
# 把两臂 ID 顺序错开一位，p 值可以一模一样）。系统级的 ID 就是
# (master_seed, replication)。
_ba = rg.replications(0, 8)
check(rg.check_pairable(_ba, rg.replications(0, 8)) == [], "同一批 replication 可配对")
_shuf = [_ba[i] for i in [1, 0, 2, 3, 4, 5, 6, 7]]
_iss = rg.check_pairable(_ba, _shuf)
check(len(_iss) == 1 and "不可观测" in _iss[0]["detail"],
      "顺序被打乱时硬拦截，并点明这种错位统计层面不可观测")
check(len(rg.check_pairable(_ba, rg.replications(0, 4))) == 1, "重复次数不同时拦截")
check(len(rg.check_pairable(_ba, rg.replications(99, 8))) == 1, "master 不同时拦截")

_blocked = rg.compare_replications(np.arange(8.0), np.arange(8.0) + 1,
                                   books_a=_ba, books_b=rg.replications(99, 8))
check(_blocked["verdict"] == "not_pairable" and "p 值没有意义" in _blocked["verdict_text"],
      "非 CRN 且 require_crn=True 时直接拒绝比较，不给 p 值")
check("paired" not in _blocked, "拒绝时不返回任何统计量（避免被当成结论引用）")

# **查不到就不能当它对。** 没给 books 时 crn 是 None（三态）而不是 True——
# 这和 CLAUDE.md「外部结果的 CSI 口径只能靠声明」是同一条原则：不能假装查过了。
_nod = rg.compare_replications(np.arange(8.0), np.arange(8.0) + 1.0)
check(_nod["crn"] is None and "无法核对" in (_nod["crn_note"] or ""),
      "没给 RngBook 时 crn=None 并明说无法核对，而不是默认 True")
check(rg.compare_replications(np.arange(8.0), np.arange(8.0) + 1.0,
                              books_a=_ba, books_b=_ba)["crn"] is True,
      "给了 books 且成立时 crn=True")

# n ≤ 5 时 Wilcoxon 最小可达 p = 2/2^n > 0.05，**无论数据多干净都不可能显著**
for _n, _pmin in ((4, 0.125), (5, 0.0625), (6, 0.03125), (8, 0.0078125)):
    _got = float(stats.wilcoxon(np.arange(1.0, _n + 1)).pvalue)
    check(abs(_got - _pmin) < 1e-12,
          f"n={_n} 的双侧 Wilcoxon 最小可达 p = 2/2^{_n} = {_got:g}")
check(rg.min_replications_note(5) is not None
      and "永远不可能显著" in rg.min_replications_note(5),
      "n=5 触发告警：这个实验无论跑出什么都不足以支撑结论")
check(rg.min_replications_note(6) is None and rg.min_replications_note(8) is None,
      "n≥6 不再告警（6 是硬下界，默认 8 留了余量）")
check(rg.min_replications_note(1) is not None
      and "11.4%" in rg.min_replications_note(1),
      "n=1 告警里带上实测的种子间变异系数")

try:
    rg.replications(0, 0)
    check(False, "重复次数 0 应当被拒")
except ValueError:
    check(True, "重复次数 0 被拒")
try:
    sy.simulate_replications(fake_tables(n_ue=3), num_replications=0)
    check(False, "simulate_replications 的重复次数 0 应当被拒")
except ValueError:
    check(True, "simulate_replications 的重复次数 0 被拒")

# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print(f"随机数体系：{_n_pass} 通过，{_n_fail} 失败")
print("=" * 70)
if _n_fail:
    sys.exit(1)
print("随机数体系与置信区间全部通过。")
