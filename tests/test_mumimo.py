"""MU-MIMO：等效信道、配对、多用户预编码、功率分配、逐用户 SINR。

直接运行：python tests/test_mumimo.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.stdout.reconfigure(errors="replace")

from superwireless import mumimo as mu  # noqa: E402

FAILED: list[str] = []


def check(cond: bool, label: str) -> None:
    print(("  PASS  " if cond else "  FAIL  ") + label)
    if not cond:
        FAILED.append(label)


def sect(title: str) -> None:
    print("\n" + "=" * 70 + f"\n{title}\n" + "=" * 70)


def make_users(n_k=6, n_rb=4, n_bs=16, n_ue=4, seed=0, gains=None):
    """造一批用户信道 [T, RB, BS, UE]，可指定各自的增益差。"""
    rng = np.random.default_rng(seed)
    g = np.ones(n_k) if gains is None else np.asarray(gains, dtype=float)
    return [
        ((rng.standard_normal((1, n_rb, n_bs, n_ue))
          + 1j * rng.standard_normal((1, n_rb, n_bs, n_ue))) / np.sqrt(2) * g[k])
        for k in range(n_k)
    ]


# ---------------------------------------------------------------------------
sect("1  等效信道：把 UE 天线折叠成行向量")

_H = make_users()
_he = mu.effective_user_channels(_H, streams_per_user=1)
check(_he.shape == (6, 1, 4, 16), f"形状 [K,S,RB,BS]（实得 {_he.shape}）")
_he2 = mu.effective_user_channels(_H, streams_per_user=2)
check(_he2.shape == (6, 2, 4, 16), "多流时 S 维展开")
# 第一条等效行的范数就是最大奇异值，第二条是次大 —— 顺序不能乱
_dl = _H[0].mean(axis=0)[0].conj().T
_sv = np.linalg.svd(_dl, compute_uv=False)
check(abs(np.linalg.norm(_he2[0, 0, 0]) - _sv[0]) < 1e-9, "第一流对应最大奇异值")
check(abs(np.linalg.norm(_he2[0, 1, 0]) - _sv[1]) < 1e-9, "第二流对应次大奇异值")

# ---------------------------------------------------------------------------
sect("2  配对：SUS 半正交用户选择")

# 造一组"有两个用户几乎共线"的信道，SUS 必须把其中一个踢掉
_rng = np.random.default_rng(4)
_base = (_rng.standard_normal((16,)) + 1j * _rng.standard_normal((16,))) / np.sqrt(2)
_Hc = make_users(n_k=5, seed=1)
_Hc[1] = _Hc[0] * 0.98 + _Hc[1] * 0.02          # 1 号与 0 号几乎共线
_hec = mu.effective_user_channels(_Hc)
_pr = mu.pair_users(_hec, criterion="sus", max_users=4, corr_threshold=0.5)
print(f"  选中 {_pr.users}，因相关被剔除 {_pr.dropped_by_corr}")
check(not (0 in _pr.users and 1 in _pr.users), "几乎共线的两个用户不会被配到一起")
check(bool(_pr.dropped_by_corr), "被剔除的用户如实记录，不是静默丢掉")
check(all(c <= 0.5 + 1e-9 for c in _pr.correlations),
      f"入选者与已选集的相关都不超过门限（实得 {np.round(_pr.correlations, 3)}）")

_pr_all = mu.pair_users(_hec, criterion="all")
check(len(_pr_all.users) == 5, "all 准则不做筛选")
_pr_one = mu.pair_users(_hec, criterion="best_single")
check(len(_pr_one.users) == 1, "best_single 只选一个")

# 流数不能超过发射天线数 —— 上限必须由代码兜住
_pr_cap = mu.pair_users(mu.effective_user_channels(make_users(n_k=40, n_bs=8),
                                                   streams_per_user=1),
                        criterion="sus", max_users=99, corr_threshold=0.99)
check(len(_pr_cap.users) <= 8, f"配对数不超过发射天线数（实得 {len(_pr_cap.users)}）")

# 权重必须真的改变选择结果（比例公平的基础）
_w = np.ones(5) * 0.01
_w[4] = 100.0
_pr_w = mu.pair_users(_hec, criterion="sus", max_users=1, weights=_w)
check(_pr_w.users == [4] and _pr_w.weights_used, "权重能把弱用户顶进配对（比例公平的基础）")

# ---------------------------------------------------------------------------
sect("3  预编码：方向与功率必须解耦")

_hs = mu.effective_user_channels(make_users(n_k=4, n_bs=16, seed=2))
_W, _p = mu.mu_precoder(_hs, method="zf", noise_power=0.01)
check(_W.shape == (4, 16, 4) and _p.shape == (4, 4), f"形状 (W {_W.shape}, p {_p.shape})")
_col = np.linalg.norm(_W[0], axis=0)
check(np.allclose(_col, 1.0), f"预编码逐列单位范数，只表示方向（实得 {np.round(_col, 4)}）")
check(np.allclose(_p.sum(axis=1), 1.0), "逐 RB 总功率归一到 1（与 SU 口径一致）")
check(np.allclose(_p[0], 0.25), "equal 分配就是等分")

_Wr, _pr2 = mu.mu_precoder(_hs, method="rzf", noise_power=0.01)
check(np.allclose(np.linalg.norm(_Wr[0], axis=0), 1.0), "RZF 同样逐列归一")
_Ww, _pw = mu.mu_precoder(_hs, method="zf", noise_power=0.01,
                          power_allocation="waterfilling")
check(np.allclose(_pw.sum(axis=1), 1.0, atol=1e-9), "注水后总功率仍归一")
check(not np.allclose(_pw[0], _pw[0][0]), "注水会给不同流不同功率（不是等分）")

# ZF 的定义就是把用户间干扰清零
_hm = _hs[:, :, 0, :].reshape(4, 16)
_G = _hm @ _W[0]
_off = np.abs(_G - np.diag(np.diag(_G))).max()
check(_off < 1e-8, f"ZF 后用户间耦合为零（最大非对角 {_off:.2e}）")

# 流数超过天线数必须直接报错，不能给一个看似正常的解
try:
    mu.mu_precoder(mu.effective_user_channels(make_users(n_k=20, n_bs=8)), method="zf")
    check(False, "流数超过天线数时报错")
except ValueError as _e:
    check("超过" in str(_e), f"流数超过天线数时报错（{_e}）")

# ---------------------------------------------------------------------------
sect("4  功率分配不能退化成信道求逆功控")

# **踩过的坑。** 早先用一个全局标量把 tr(WW^H) 归一，ZF 满足 HW=c·I，
# 于是所有用户接收电平被强行拉平、弱用户吃掉大部分功率，
# 公平度恒等于 1.000 —— 看起来像"MU 天生公平"，其实是功率分配被写死了。
_Hg = make_users(n_k=4, n_bs=16, seed=3, gains=[1.0, 1.0, 1.0, 0.2])
_res = mu.mu_link_performance(_Hg, noise_power=0.01, precoder="zf", criterion="all")
print(f"  逐用户谱效 {np.round(_res.se_per_user, 3)}  Jain {_res.jain_fairness:.4f}")
check(_res.se_per_user.std() > 1e-6,
      "增益差 5 倍的用户不该拿到一模一样的谱效（那是信道求逆功控的症状）")
check(_res.jain_fairness < 1.0 - 1e-6, f"公平度不恒等于 1（实得 {_res.jain_fairness}）")
check(_res.se_per_user[3] < _res.se_per_user[0], "弱用户谱效确实更低")
check(_res.power_allocation == "equal", "功率分配方式跟着结果一起返回")

# ---------------------------------------------------------------------------
sect("5  MU 增益与 CSI 敏感性")

_Hm = make_users(n_k=8, n_bs=32, n_ue=4, seed=5)
_su = float(np.mean([  # SU 对照：同样总功率，一次只服务一个用户
    mu.mu_link_performance([h], noise_power=0.01, criterion="all").sum_se for h in _Hm
]))
_mu4 = mu.mu_link_performance(_Hm, noise_power=0.01, precoder="rzf",
                              criterion="sus", max_users=4)
print(f"  SU（单用户单流）{_su:.3f} → MU 配 {len(_mu4.users)} 个 {_mu4.sum_se:.3f}")
check(_mu4.sum_se > _su, "MU 和谱效高于单用户单流（空间复用确实有增益）")
check(len(_mu4.users) > 1, "确实配了多个用户")

# CSI 变差 -> ZF 零陷变浅 -> 残余干扰上升 -> 谱效下降。三者必须同向。
_prev_leak, _prev_se = -1.0, 1e9
print(f"  {'CSI 误差':<12}{'和谱效':>9}{'残余干扰':>12}")
for _err in (0.0, 0.03, 0.1, 0.3):
    _rg = np.random.default_rng(7)
    _He = [h + (_rg.standard_normal(h.shape) + 1j * _rg.standard_normal(h.shape))
           * _err * float(np.std(np.abs(h))) for h in _Hm]
    _r = mu.mu_link_performance(_Hm, h_users_for_precoding=_He, noise_power=0.01,
                                precoder="zf", criterion="sus", max_users=4)
    print(f"  {_err:<12.2f}{_r.sum_se:>9.3f}{_r.leakage_ratio:>12.3e}")
    if _err > 0:
        check(_r.leakage_ratio > _prev_leak, f"CSI 越差残余干扰越大（err={_err}）")
        check(_r.sum_se < _prev_se, f"CSI 越差和谱效越低（err={_err}）")
        check(_r.csi_for_precoding == "h_est", "CSI 口径如实带回结果")
    _prev_leak, _prev_se = _r.leakage_ratio, _r.sum_se

# 理想 CSI + ZF 必须零残余干扰 —— 这是 ZF 的定义，破了就是实现错了
_ideal = mu.mu_link_performance(_Hm, noise_power=0.01, precoder="zf",
                                criterion="sus", max_users=4)
check(_ideal.leakage_ratio < 1e-12,
      f"理想 CSI 下 ZF 残余干扰为零（实得 {_ideal.leakage_ratio:.2e}）")
check(_ideal.csi_for_precoding == "h_true", "没传估计信道时标成 h_true")
# ---------------------------------------------------------------------------
sect("6  单码字谱效与 rank 自适应")

# 用户级 SINR：RBG 内线性平均、RBG 间与流间 dB 域平均
_s = np.full((32, 1), 10.0)
check(abs(mu.user_sinr_db(_s, rb_per_rbg=16) - 10.0) < 1e-9, "全平信道的用户级 SINR 就是它本身")
# **dB 域平均必须比线性平均保守** —— 单码字会被深衰的 RBG 拖下去
_v = np.array([[100.0]] * 16 + [[0.01]] * 16)
_db = mu.user_sinr_db(_v, rb_per_rbg=16)
_lin = 10 * np.log10(_v.mean())
print(f"  半好半坏：dB 域平均 {_db:.2f} dB，线性平均 {_lin:.2f} dB")
check(_db < _lin - 10, "dB 域平均显著低于线性平均（单码字被深衰 RBG 拖累）")
check(abs(_db) < 1e-6, f"两个 RBG 各 +20/-20 dB，dB 域平均是 0（实得 {_db}）")

# 谱效 = rank x MCS 谱效
_se1, _m1 = mu.se_from_sinr(20.0, 1)
_se2, _m2 = mu.se_from_sinr(20.0, 2)
check(abs(_se2 - 2 * _se1) < 1e-9, "同 SINR 下谱效严格正比于 rank")
check(mu.se_from_sinr(30.0, 1)[1].index >= _m1.index, "SINR 越高 MCS 不降")

# rank 自适应
_rng2 = np.random.default_rng(11)
_hh = ((_rng2.standard_normal((1, 32, 16, 4)) + 1j * _rng2.standard_normal((1, 32, 16, 4)))
       / np.sqrt(2))
_lo = mu.su_rank_adaptation(_hh, noise_power=mu.noise_from_geometric_sinr(_hh, 0.0))
_hi = mu.su_rank_adaptation(_hh, noise_power=mu.noise_from_geometric_sinr(_hh, 30.0))
print(f"  几何 SINR 0 dB -> rank {_lo.rank} MCS {_lo.mcs}；30 dB -> rank {_hi.rank} MCS {_hi.mcs}")
check(_lo.rank <= _hi.rank, "信噪比高时选的秩不低于低信噪比时")
check(len(_hi.candidates) == 4, "四个 rank 候选都算过并留在结果里")
check(all(c["rank"] == i + 1 for i, c in enumerate(_hi.candidates)), "候选按 rank 排列")
check(_hi.se == max(c["se"] for c in _hi.candidates), "选中的就是谱效最高的候选")

# **噪声口径**：用 mean(|h|^2) 反推会把阵列增益算两遍，实测差 12 dB
_n_anchor = mu.noise_from_geometric_sinr(_hh, 15.0)
_hb = _hh.mean(axis=0)
_n_naive = float(np.mean(np.abs(_hb) ** 2)) / 10 ** 1.5
_delta = 10 * np.log10(_n_anchor / _n_naive)
print(f"  两种噪声口径相差 {_delta:.1f} dB")
check(_delta > 6.0, f"锚定口径的噪声显著高于 mean(|h|²) 口径（差 {_delta:.1f} dB）")
check(mu.su_rank_adaptation(_hh, noise_power=_n_naive).mcs
      > mu.su_rank_adaptation(_hh, noise_power=_n_anchor).mcs,
      "错口径会系统性高估 MCS —— 这正是它危险的地方")

_r1 = [c for c in mu.su_rank_adaptation(
    _hh, noise_power=mu.noise_from_geometric_sinr(_hh, 12.0)).candidates
    if c["rank"] == 1][0]
check(abs(_r1["sinr_db"] - 12.0) < 1.5,
      f"rank1 的用户级 SINR 锚在几何 SINR 上（实得 {_r1['sinr_db']}，目标 12.0）")

# ---------------------------------------------------------------------------
sect("7  SU / MU 自适应")

_Hs = make_users(n_k=6, n_rb=32, n_bs=32, n_ue=4, seed=13)
_npow = mu.noise_from_geometric_sinr(_Hs[0], 15.0)
_dec = mu.su_mu_adaptation(_Hs, noise_power=_npow)
print(f"  {_dec.note}")
print(f"  判决 {_dec.mode}：小区谱效 {_dec.cell_se:.3f}"
      f"（SU {_dec.su_se:.3f} / MU {_dec.mu_se:.3f}）")
check(_dec.mode in ("SU", "MU"), "给出明确判决")
check(abs(_dec.cell_se - max(_dec.su_se, _dec.mu_se)) < 1e-9, "小区谱效取两者中的高者")
check(_dec.su_rank <= mu.SU_MAX_RANK, f"SU 秩不超过 {mu.SU_MAX_RANK}")
check(all(d["rank"] <= mu.MU_MAX_RANK for d in _dec.mu_per_user),
      f"MU 每用户秩不超过 {mu.MU_MAX_RANK}（工程约束）")
check(bool(_dec.mu_users), "MU 方案确实配了人")
check(len(_dec.mu_per_user) == len(_dec.mu_users), "逐用户明细齐全")

# 功率按流均分：rank2 的用户拿 2 份
_he2 = mu.effective_user_channels(_Hs[:3], streams_per_user=2)
_, _pp = mu.mu_precoder(_he2, method="zf", noise_power=_npow)
check(abs(_pp[0].sum() - 1.0) < 1e-9, "总功率仍归一")
check(abs(_pp[0][0] - 1.0 / 6) < 1e-9, "6 条流每流 1/6，即 rank2 的用户拿 1/3")



# ---------------------------------------------------------------------------
sect("8  RBG 粒度：降 16 倍算量而不改结论")

_rng3 = np.random.default_rng(31)
_hb = ((_rng3.standard_normal((272, 32, 4)) + 1j * _rng3.standard_normal((272, 32, 4)))
       / np.sqrt(2))
_red = mu.rbg_reduce(_hb, 16)
check(_red.shape == (17, 32, 4), f"272 RB -> 17 RBG（实得 {_red.shape}）")
check(mu.rbg_reduce(_hb, 1).shape == (272, 32, 4), "rb_per_rbg=1 退回 RB 粒度")

_h4 = _hb[None]
_np = mu.noise_from_geometric_sinr(_h4, 15.0)
_rb_res = mu.su_rank_adaptation(_h4, noise_power=_np, rb_per_rbg=1)
_rbg_res = mu.su_rank_adaptation(_h4, noise_power=_np, rb_per_rbg=16)
print(f"  RB 粒度 rank {_rb_res.rank} MCS {_rb_res.mcs} SE {_rb_res.se:.3f}")
print(f"  RBG粒度 rank {_rbg_res.rank} MCS {_rbg_res.mcs} SE {_rbg_res.se:.3f}")
check(_rb_res.rank == _rbg_res.rank, "两种粒度选出同一个 rank")
check(abs(_rb_res.mcs - _rbg_res.mcs) <= 1, "MCS 最多差一档")
check(abs(_rb_res.se - _rbg_res.se) / max(_rb_res.se, 1e-9) < 0.05,
      f"谱效差 <5%（实得 {abs(_rb_res.se - _rbg_res.se) / max(_rb_res.se, 1e-9):.1%}）")

# **取代表点而不是平均。** 平均会把频选衰落抹平、抬高信道条件数，进而高估 rank。
_flat = np.repeat(_hb.mean(axis=0, keepdims=True), 17, axis=0)
_sv_avg = np.linalg.svd(_flat[0].conj().T, compute_uv=False)
_sv_rep = np.linalg.svd(_red[0].conj().T, compute_uv=False)
print(f"  平均后奇异值比 σ4/σ1 = {_sv_avg[3] / _sv_avg[0]:.3f}；"
      f"取代表点 {_sv_rep[3] / _sv_rep[0]:.3f}")
check(_sv_rep[3] / _sv_rep[0] < _sv_avg[3] / _sv_avg[0] * 3,
      "取代表点保留了真实的奇异值分布，没有被平均抹平")
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
if FAILED:
    print(f"FAILED {len(FAILED)} 项：")
    for f in FAILED:
        print("  - " + f)
    sys.exit(1)
print("MU-MIMO 配对、预编码、功率分配、CSI 敏感性全部通过。")
