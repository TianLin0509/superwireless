"""CSI 反馈时延与老化。

**基站永远不知道"现在"的信道。** 它知道的是上一次 SRS 探到的那个信道，
再加上估计、算预编码、下发调度的时间。TDD 下行靠互易性从上行 SRS 取 CSI，
所以这条时延链是：

    SRS 发送 → 信道估计 → 预编码计算 → PDSCH 发送
    └───────────── 这段时间信道一直在变 ─────────────┘

平台在此之前默认基站拿到的是**零时延完美 CSI**——预编码和评估用同一个矩阵，
于是 SVD 预编码永远精确匹配、ZF 零陷永远打得准。这在现网是不成立的，
而且它系统性地**高估 MU 增益**：MU 的全部收益都建立在零陷打得准上。

## 时延从哪来

两部分，量级差很远：

* **SRS 周期** ``T_SRS``——现网典型 5 / 10 / 20 / 40 ms。
* **SRS 跳频**——这才是大头。SRS 为了省上行开销与提高导频功率密度，
  一次只探一小段带宽，靠跳频扫完全带。

## 跳频的 17 是标准里逐字有的

38.211 Table 6.4.1.4.3-1 的 **C_SRS = 57 行**：

    m_SRS = (272, 16, 4, 4)      N = (1, 17, 4, 1)

取 ``B_SRS = 1``、``b_hop = 0``：每次 SRS 占 **16 RB（正好 1 个 RBG）**，
要 **17 跳**才扫完 272 RB。这和本项目的 17 RBG × 16 RB 载波配置 1:1 对上。

后果很直接：``T_SRS = 10 ms`` 时全带扫一遍要 **170 ms**，
某个 RBG 的 CSI 年龄在 0 ~ 160 ms 之间轮转，**平均 80 ms**。
2.6 GHz、30 km/h 的相干时间只有约 6 ms——CSI 早就过期了。

跳频序列直接调 ChannelHub 的 ``srs_rb_indices``（它实现了 38.211 §6.4.1.4.3
的完整跳频树），不自己写。实测 C_SRS=57 / B_SRS=1 给出的顺序就是
RBG 0 → 1 → … → 16 循环。

## 老化怎么进 SINR

**不是给 SINR 打个折扣，是让预编码真的算错。**

    W = SVD(H_stale)            ← 基站用陈旧信道算预编码
    SINR = MMSE(H_true, W)      ← 实际传输吃的是当前信道

零时延时 ``H_stale == H_true``，``H_true·W`` 恰好对角化，
逐流 SINR 退化成 ``σ_k²·P/rank/σ_n²``——**和原来的
``su_rank_adaptation`` 逐位相同**。这条恒等式是本模块的核心自检
（``test_csi_aging`` 第 1 节），它保证老化模型不是叠加上去的第二套物理。

有老化时 ``H_true·W`` 不再对角，流间泄漏进入 MMSE 的分母，
表现为 BF 增益下降 + 流间干扰——这正是现网的物理。

参考：Sionna 的 CSI 反馈链路同样把预编码信道与评估信道分成两个输入
（``sionna.phy.mimo`` 的 precoding 与 detection 是解耦的），
本模块的 ``h_prec`` / ``h_eval`` 沿用同一分工。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from . import mumimo as mu

__all__ = [
    "SRS_PERIOD_CHOICES",
    "CsiConfig",
    "hop_order",
    "rbg_age_ms",
    "rbg_lag_snapshots",
    "stale_channel",
    "svd_precoder",
    "mmse_stream_sinr",
    "AgedRankChoice",
    "rank_adaptation_aged",
    "jakes_correlation",
    "aging_summary",
]

_EPS = 1e-12

#: 现网典型 SRS 周期（ms）。**只允许这四个值**——它们对应 38.331 里
#: ``periodicityAndOffset`` 在 30 kHz SCS 下的 sl10 / sl20 / sl40 / sl80 时隙。
SRS_PERIOD_CHOICES: tuple[float, ...] = (5.0, 10.0, 20.0, 40.0)

#: 38.211 Table 6.4.1.4.3-1 里 m_SRS,0 = 272（全带）且 m_SRS,1 = 16（1 个 RBG）的那一行
SRS_C_SRS_FULL_BAND = 57
SRS_B_SRS_ONE_RBG = 1
DEFAULT_HOP_FACTOR = 17


@dataclass
class CsiConfig:
    """CSI 时延链的配置。

    ``enabled=False`` 时整条链退化成零时延完美 CSI，也就是本模块出现之前的行为。
    保留这个开关是为了能做 A/B 对比——**老化的代价必须能被量出来**，
    而不是悄悄地混进所有结果里。
    """

    enabled: bool = True
    #: SRS 周期（ms），取值见 :data:`SRS_PERIOD_CHOICES`
    srs_period_ms: float = 10.0
    #: 跳频开关。关掉时每次 SRS 探全带，年龄只剩周期内的相位 + 处理时延
    hopping: bool = True
    #: 跳频倍数。默认 17 = C_SRS 57 / B_SRS 1，每跳 1 个 RBG
    hop_factor: int = DEFAULT_HOP_FACTOR
    #: 信道估计 + 预编码计算 + 调度下发的固定时延（ms）
    processing_delay_ms: float = 2.0

    def __post_init__(self) -> None:
        if self.srs_period_ms not in SRS_PERIOD_CHOICES:
            raise ValueError(
                f"srs_period_ms 只支持 {SRS_PERIOD_CHOICES}，收到 {self.srs_period_ms}"
            )
        if self.hop_factor < 1:
            raise ValueError("hop_factor 至少为 1")
        if self.processing_delay_ms < 0:
            raise ValueError("processing_delay_ms 不能为负")

    @property
    def full_sweep_ms(self) -> float:
        """扫完全带宽需要多久。不跳频时就是一个 SRS 周期。"""
        return self.srs_period_ms * (self.hop_factor if self.hopping else 1)

    @property
    def mean_age_ms(self) -> float:
        """全带宽平均 CSI 年龄。

        跳频时某个 RBG 的年龄在 ``0 ~ (H-1)·T`` 之间均匀轮转，均值 ``(H-1)·T/2``；
        再加上周期内相位的均值 ``T/2`` 与固定处理时延。
        """
        hops = (self.hop_factor - 1) if self.hopping else 0
        return hops * self.srs_period_ms / 2.0 + self.srs_period_ms / 2.0 + \
            self.processing_delay_ms

    def as_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "srs_period_ms": self.srs_period_ms,
            "hopping": self.hopping,
            "hop_factor": self.hop_factor if self.hopping else 1,
            "processing_delay_ms": self.processing_delay_ms,
            "full_sweep_ms": round(self.full_sweep_ms, 2),
            "mean_age_ms": round(self.mean_age_ms, 2),
            "standard": (
                f"38.211 Table 6.4.1.4.3-1 C_SRS={SRS_C_SRS_FULL_BAND} "
                f"B_SRS={SRS_B_SRS_ONE_RBG}：m_SRS=(272,16,4,4) N=(1,17,4,1)，"
                f"每跳 16 RB、17 跳扫完全带"
            ) if self.hopping else "不跳频，每次 SRS 探全带",
            "note": (
                "预编码用陈旧 CSI、评估用当前信道。enabled=False 时退化成零时延"
                "完美 CSI（本模块出现之前的行为），可用于 A/B 对比。"
            ),
        }


# ---------------------------------------------------------------------------
# 跳频序列
# ---------------------------------------------------------------------------
def hop_order(num_rbg: int, *, rb_per_rbg: int = 16,
              hop_factor: int = DEFAULT_HOP_FACTOR) -> tuple[np.ndarray, str]:
    """第 j 次 SRS 机会探的是哪个 RBG。返回 ``(order, source)``。

    走 ChannelHub 的 ``srs_rb_indices``——它实现了 38.211 §6.4.1.4.3 的完整
    跳频树（``F_b`` 递推、``n_RRC`` 偏置、奇偶 ``N_b`` 的镜像规则）。
    自己重写一遍只会引入分歧。

    **取不到时退回恒等扫描并如实标注 source**，不静默假装用了标准实现。
    这条兜底路径的输出和标准实现**碰巧一模一样**（C_SRS=57 就是顺序扫描），
    所以除了看 ``source`` 没有任何办法发现自己没在用标准实现——
    测试因此直接断言 ``source`` 以 ``channelhub:`` 开头。
    """
    try:
        from .channelhub import _ensure_path  # noqa: PLC0415

        _ensure_path()
        from msg_embedding.ref_signals.srs import (  # noqa: PLC0415
            SRSResourceConfig,
            srs_hopping_cycle_length,
            srs_rb_indices,
        )

        cfg = SRSResourceConfig(
            C_SRS=SRS_C_SRS_FULL_BAND, B_SRS=SRS_B_SRS_ONE_RBG,
            K_TC=2, n_RRC=0, b_hop=0, n_SRS_ID=0, T_SRS=1, T_offset=0,
        )
        cycle = int(srs_hopping_cycle_length(cfg))
        if cycle != hop_factor:
            raise ValueError(f"标准表给出 {cycle} 跳，配置要 {hop_factor} 跳")
        total_rb = num_rbg * rb_per_rbg
        # T_SRS=1 时 n_SRS 恰好等于 slot 序号（srs.py:453），所以直接传 j
        order = np.array(
            [int(srs_rb_indices(cfg, j, 0, total_rb)[0]) // rb_per_rbg
             for j in range(cycle)], dtype=int)
        return order, "channelhub:38.211-6.4.1.4.3"
    except Exception as exc:  # noqa: BLE001
        return (np.arange(hop_factor) % max(1, num_rbg),
                f"fallback:identity-sweep（{type(exc).__name__}: {exc}）")


def rbg_age_ms(cfg: CsiConfig, num_rbg: int, t_ms: float, *,
               rb_per_rbg: int = 16) -> np.ndarray:
    """时刻 ``t_ms`` 上，每个 RBG 的 CSI 年龄（ms）。返回 ``[num_rbg]``。

    年龄 = 距上次探到该 RBG 的 SRS 机会过了多久 + 处理时延。
    跳频时各 RBG 的年龄**随时间轮转**——同一个 RBG 在不同 TTI 上年龄不同，
    所以长时间平均下来所有 RBG 是等价的，不会有"某几个 RBG 永远最差"。
    """
    if not cfg.enabled:
        return np.zeros(num_rbg)
    t = float(t_ms)
    per = cfg.srs_period_ms
    n = int(np.floor(t / per + 1e-9))            # 最近一次 SRS 机会的序号
    within = t - n * per                          # 距那次机会过了多久
    if not cfg.hopping:
        return np.full(num_rbg, within + cfg.processing_delay_ms)

    order, _ = hop_order(num_rbg, rb_per_rbg=rb_per_rbg, hop_factor=cfg.hop_factor)
    h = len(order)
    # 同一个 RBG 在一个周期里可能被探多次，取最近的那次
    hops = np.full(num_rbg, h - 1, dtype=int)
    for j, k in enumerate(order):
        if 0 <= int(k) < num_rbg:
            hops[int(k)] = min(int(hops[int(k)]), (n - j) % h)
    return hops * per + within + cfg.processing_delay_ms


def rbg_lag_snapshots(cfg: CsiConfig, num_rbg: int, *, snapshot_ms: float,
                      snapshot_index: int, rb_per_rbg: int = 16) -> np.ndarray:
    """把每个 RBG 的年龄折成**整数个信道快照**。返回 ``[num_rbg]`` 非负整数。

    信道快照之间隔 ``snapshot_ms``（由 :func:`system.snapshot_interval_ms` 算出，
    默认 5 ms），所以年龄只能量化到这个粒度。

    **量化误差要报出来，不能假装没有。** 默认配置下主导项是跳频
    （T_SRS=10 ms 时年龄跨度 0~160 ms），2 ms 的处理时延落在量化噪声里；
    但如果有人把 SRS 周期设成 5 ms 又关掉跳频，年龄就全在一个快照以内，
    这时老化模型基本失效——:func:`aging_summary` 会警告。
    """
    if not cfg.enabled:
        return np.zeros(num_rbg, dtype=int)
    age = rbg_age_ms(cfg, num_rbg, snapshot_index * float(snapshot_ms),
                     rb_per_rbg=rb_per_rbg)
    return np.maximum(0, np.round(age / max(float(snapshot_ms), _EPS))).astype(int)


def stale_channel(snaps: list[np.ndarray], snapshot_index: int,
                  lags: np.ndarray) -> np.ndarray:
    """按逐 RBG 的滞后拼出基站"以为"的信道。

    ``snaps[s]`` 形状 ``[RBG, BS, UE]``；返回同形状，第 ``k`` 行取自
    ``snaps[snapshot_index - lags[k]]``。

    索引越界（仿真刚开始、还没攒够历史）时**钳到最早的那个快照**，
    而不是回绕——回绕会把未来的信道当成过去用，是彻头彻尾的假数据。
    """
    cur = np.asarray(snaps[snapshot_index])
    out = np.array(cur, copy=True)
    for k in range(cur.shape[0]):
        s = max(0, int(snapshot_index) - int(lags[k]))
        out[k] = np.asarray(snaps[s])[k]
    return out


# ---------------------------------------------------------------------------
# 预编码失配下的 SINR
# ---------------------------------------------------------------------------
def svd_precoder(h_prec: np.ndarray) -> np.ndarray:
    """从（陈旧的）信道算 SVD 预编码。``[RBG, BS, UE]`` → ``[RBG, BS, K]``。

    列按奇异值降序、单位范数——**方向而已，功率另给**
    （和 :func:`mumimo.mu_precoder` 同一约定）。
    """
    hb = np.asarray(h_prec)
    hm = np.conj(np.transpose(hb, (0, 2, 1)))          # [F, N_rx, N_tx]
    _, _, vh = np.linalg.svd(hm, full_matrices=False)  # vh: [F, K, N_tx]
    return np.conj(np.transpose(vh, (0, 2, 1)))        # [F, N_tx, K]


def mmse_stream_sinr(h_eval: np.ndarray, w: np.ndarray, *,
                     power_per_stream: float, noise_power: float) -> np.ndarray:
    """预编码 ``w`` 打在信道 ``h_eval`` 上，MMSE 接收机的逐流 SINR。

    ``h_eval`` ``[RBG, BS, UE]``，``w`` ``[RBG, BS, rank]``，返回 ``[RBG, rank]``（线性）。

    模型：``y = H W s + n``，``E[ss^H] = p·I``、``E[nn^H] = σ²·I``，
    MMSE 均衡后第 k 流的后处理 SINR 是标准结果::

        SINR_k = 1 / [ (I + (p/σ²)·(HW)^H(HW))^{-1} ]_kk  −  1

    **零时延时它必须退化成 ``σ_k²·p/σ²``。** 因为那时 ``HW = UΣ_r``、
    ``(HW)^H(HW) = Σ_r²`` 是对角阵，逆的对角元就是 ``1/(1+p·σ_k²/σ²)``。
    这条恒等式保证老化模型不是叠加上去的第二套物理，而是同一套物理的推广。
    """
    hb = np.asarray(h_eval)
    hm = np.conj(np.transpose(hb, (0, 2, 1)))          # [F, N_rx, N_tx]
    g = hm @ np.asarray(w)                             # [F, N_rx, r]
    gram = np.conj(np.transpose(g, (0, 2, 1))) @ g     # [F, r, r]
    r = gram.shape[-1]
    a = np.eye(r) + (float(power_per_stream) / max(float(noise_power), _EPS)) * gram
    diag = np.real(np.einsum("fkk->fk", np.linalg.inv(a)))
    return np.maximum(1.0 / np.maximum(diag, _EPS) - 1.0, 0.0)


@dataclass
class AgedRankChoice:
    """老化下的 rank 自适应结果。**基站以为的**与**真实的**必须分开。"""

    rank: int                            # 基站选的 rank（按它自己的陈旧 CSI）
    sinr_db: float                       # 该 rank 下的真实接收 SINR
    mcs: int                             # 真实 SINR 对应的 MCS
    se: float                            # 真实谱效
    se_gnb: float                        # 基站以为的谱效
    candidates: list[dict[str, Any]]     # 逐 rank 的真实量
    gnb_candidates: list[dict[str, Any]]  # 逐 rank 的基站估计量


def rank_adaptation_aged(h_prec: np.ndarray, h_eval: np.ndarray, *,
                         noise_power: float, max_rank: int = mu.SU_MAX_RANK,
                         table: int = 3, target_bler: float = 0.1,
                         total_power: float = 1.0,
                         rb_per_rbg: int = 1,
                         w_override: np.ndarray | None = None) -> AgedRankChoice:
    """预编码用 ``h_prec``、评估用 ``h_eval`` 的 rank 自适应。

    两者都是 ``[RBG, BS, UE]``（已经降过粒度）。除了预编码信道来源不同，
    判据与 :func:`mumimo.su_rank_adaptation` 完全一致：遍历 rank 1..max_rank，
    单码字口径压成用户级 SINR，取 ``rank × MCS谱效`` 最高的那个。

    ``rb_per_rbg`` 直通 :func:`mumimo.user_sinr_db`：输入已降到 RBG 粒度时传 1
    （每行就是一个 RBG），停在 RB 粒度时传 16（由它按 16 分组）。
    **传错会改变单码字的聚合口径**，SINR 会差几个 dB。

    **rank 由基站按自己的 CSI 选，不是按真实信道选。** 这一条很容易写错，
    而写错的方向恰好是"老化看起来没那么糟"：如果拿真实 SINR 去挑 rank，
    等于让基站预知信道变成了什么样，它会自动避开老化最狠的那个 rank，
    损失被凭空抹掉一大半。高速下真实的现象正是"基站点了 rank 4、
    实际只撑得住 rank 1"——这是老化损失的重要一环，必须留在结果里。

    零时延时 ``h_prec is h_eval``，两套量逐位相同，退化成原来的行为。

    ``w_override`` 给定时用它当发射权（形状 ``[F, N_tx, K]``，列单位范数），
    不再从 ``h_prec`` 做 SVD——用来把**实际发送权换成 Type I 码本**，
    看码本权在老化下是不是比 SVD 更耐受（自由度少，能算错的地方也少）。
    注意它仍必须是从**陈旧** CSI 算出来的，否则又变成基站预知信道了。
    """
    hp = np.asarray(h_prec)
    he = np.asarray(h_eval)
    if hp.shape != he.shape:
        raise ValueError(f"预编码与评估信道形状必须一致，收到 {hp.shape} vs {he.shape}")
    r_max = max(1, min(int(max_rank), hp.shape[1], hp.shape[2]))
    if w_override is not None:
        w_full = np.asarray(w_override)
        if w_full.shape[:2] != hp.shape[:2]:
            raise ValueError(f"w_override 形状 {w_full.shape} 与信道 {hp.shape} 对不上")
        r_max = max(1, min(r_max, w_full.shape[2]))
    else:
        w_full = svd_precoder(hp)                      # [F, N_tx, K]

    cands: list[dict[str, Any]] = []
    gnb: list[dict[str, Any]] = []
    best_r, best_se_gnb = 1, -1.0
    for r in range(1, r_max + 1):
        p_per = float(total_power) / r
        w = w_full[:, :, :r]
        # 真实：陈旧预编码打在当前信道上
        s_true = mu.user_sinr_db(
            mmse_stream_sinr(he, w, power_per_stream=p_per, noise_power=noise_power),
            rb_per_rbg=rb_per_rbg)
        se_t, mcs_t = mu.se_from_sinr(s_true, r, table=table, target_bler=target_bler)
        cands.append({"rank": r, "sinr_db": round(s_true, 2), "mcs": mcs_t.index,
                      "se": round(se_t, 4)})
        # 基站以为的：陈旧预编码打在陈旧信道上（它只有这个）
        s_gnb = mu.user_sinr_db(
            mmse_stream_sinr(hp, w, power_per_stream=p_per, noise_power=noise_power),
            rb_per_rbg=rb_per_rbg)
        se_g, mcs_g = mu.se_from_sinr(s_gnb, r, table=table, target_bler=target_bler)
        gnb.append({"rank": r, "sinr_db": round(s_gnb, 2), "mcs": mcs_g.index,
                    "se": round(se_g, 4)})
        if se_g > best_se_gnb:
            best_r, best_se_gnb = r, se_g

    return AgedRankChoice(
        rank=best_r,
        sinr_db=float(cands[best_r - 1]["sinr_db"]),
        mcs=int(cands[best_r - 1]["mcs"]),
        se=float(cands[best_r - 1]["se"]),
        se_gnb=float(best_se_gnb),
        candidates=cands, gnb_candidates=gnb,
    )


# ---------------------------------------------------------------------------
# 报告
# ---------------------------------------------------------------------------
def jakes_correlation(delay_ms: float, speed_kmh: float,
                      carrier_hz: float = 2.6e9) -> float:
    """Jakes 模型下滞后 ``delay_ms`` 的信道时间相关系数 ``|J₀(2π·f_d·τ)|``。

    ``f_d = v·f_c/c``。用来判断"这个时延到底算不算长"——
    相干时间的常用定义是相关系数掉到 0.5 的那个 τ。
    """
    from scipy.special import j0  # noqa: PLC0415

    f_d = float(speed_kmh) / 3.6 * float(carrier_hz) / 299_792_458.0
    return float(abs(j0(2.0 * np.pi * f_d * float(delay_ms) / 1000.0)))


def coherence_time_ms(speed_kmh: float, carrier_hz: float = 2.6e9) -> float:
    """相干时间（ms）：相关系数首次掉到 0.5 的滞后。速度为 0 时返回 inf。"""
    if speed_kmh <= 0:
        return float("inf")
    lo, hi = 0.0, 1.0
    while jakes_correlation(hi, speed_kmh, carrier_hz) > 0.5 and hi < 1e5:
        hi *= 2.0
    for _ in range(60):
        mid = (lo + hi) / 2.0
        if jakes_correlation(mid, speed_kmh, carrier_hz) > 0.5:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


def aging_summary(cfg: CsiConfig, *, num_rbg: int = 17, snapshot_ms: float = 5.0,
                  speed_kmh: float = 3.0, carrier_hz: float = 2.6e9,
                  rb_per_rbg: int = 16) -> dict[str, Any]:
    """把这套配置下的老化画像算出来，供说明书与结果摘要引用。"""
    order, source = hop_order(num_rbg, rb_per_rbg=rb_per_rbg,
                              hop_factor=cfg.hop_factor)
    ages = rbg_age_ms(cfg, num_rbg, 0.0, rb_per_rbg=rb_per_rbg) if cfg.enabled \
        else np.zeros(num_rbg)
    lags = rbg_lag_snapshots(cfg, num_rbg, snapshot_ms=snapshot_ms,
                             snapshot_index=0, rb_per_rbg=rb_per_rbg)
    t_c = coherence_time_ms(speed_kmh, carrier_hz)
    mean_age = cfg.mean_age_ms if cfg.enabled else 0.0
    warn: list[str] = []
    if cfg.enabled and max(lags) == 0:
        warn.append(
            f"所有 RBG 的滞后都量化成 0 个快照（快照间隔 {snapshot_ms:g} ms，"
            f"平均年龄 {mean_age:.1f} ms）——这套配置下老化模型几乎不起作用，"
            f"结果与零时延完美 CSI 基本相同。")
    if cfg.enabled and mean_age > 0 and t_c > 0 and mean_age / t_c > 5:
        warn.append(
            f"平均 CSI 年龄 {mean_age:.0f} ms 是相干时间 {t_c:.0f} ms 的 "
            f"{mean_age / t_c:.0f} 倍——预编码基本是在对一个无关的信道做匹配，"
            f"MU 增益会接近甚至低于 SU。")
    return {
        "config": cfg.as_dict(),
        "hop_order": [int(x) for x in order],
        "hop_order_source": source,
        "rbg_age_ms": [round(float(x), 2) for x in ages],
        "rbg_lag_snapshots": [int(x) for x in lags],
        "snapshot_ms": snapshot_ms,
        "mean_age_ms": round(mean_age, 2),
        "max_age_ms": round(float(max(ages)) if len(ages) else 0.0, 2),
        "speed_kmh": speed_kmh,
        "doppler_hz": round(speed_kmh / 3.6 * carrier_hz / 299_792_458.0, 2),
        "coherence_time_ms": round(t_c, 2) if np.isfinite(t_c) else None,
        "jakes_rho_at_mean_age": round(
            jakes_correlation(mean_age, speed_kmh, carrier_hz), 4),
        "warnings": warn,
    }
