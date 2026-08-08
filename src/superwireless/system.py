"""系统级仿真：连续 TTI、话务模型、PF 调度、体验速率。

**这一层回答的问题和链路级不一样。** 链路级问"这个信道能跑多快"，
系统级问"**这个小区里的用户实际体验到多快**"——后者要把话务的到达与结束、
调度器在多用户间的取舍、HARQ 重传、缓冲区排空全部算进去。

体验速率是现网真正上报的 KPI，它**不是**吞吐量的平均：

* 只在"有数据要发"的时间段里算（没数据的时候不算你慢）
* **掐尾**——把清空缓冲区的那个 TTI 排除掉（3GPP TS 28.552 §5.1.1.3）。
  不掐的话，一个只用半个 TTI 就发完的小包会被算成"半个 TTI 的速率"，
  数值虚高得离谱。
* **掐头**——运营商话统里通常还会排除首个 TTI（含调度时延与 BSR 上报往返）。
  两种口径都实现了，见 :class:`KpiConfig`。

架构上分两相，这是能跑十万 TTI 的关键：

    第一相（贵）：逐 UE、逐信道快照，把 rank 1..4 的 SINR / MCS / 谱效
                  全部算好存成表。SVD 只在这里做。
    第二相（便宜）：TTI 主循环只查表 + 算 PF 度量 + 更新缓冲区，
                  没有任何矩阵运算。

实测 20000 TTI × 12 UE 在第二相里是秒级；如果把 SVD 放进主循环，
同样规模要几十分钟。
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np

from . import csi_aging as ca
from . import mumimo as mu
from . import rng as rg

_EPS = 1e-12

#: S 时隙折合成多少个下行 TTI。大部分符号是下行，但有 GP 与上行符号。
#: **主循环与 dl_ratio 必须用同一个数**，否则实际调度的下行比报告的多。
S_SLOT_DL_FRACTION = 0.7

TrafficModel = Literal["full_buffer", "ftp3", "cbr", "bimodal"]
SchedAlgorithm = Literal["pf", "rr", "max_ci"]
ThroughputTrim = Literal["none", "tail", "head_tail"]


# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------
@dataclass
class TrafficConfig:
    """话务模型。

    ``ftp3`` 是 3GPP 的 FTP Model 3（见 TR 36.814 Annex A.2.1.3.1 与
    TR 38.802 §A.2.1.3）：每个用户按泊松过程到达固定大小的文件，
    到达率控制负载。**它是评价体验速率的标准话务模型**——
    full buffer 下"体验速率"没有意义，因为缓冲区永远不空、没有 burst 边界。
    """

    model: TrafficModel = "ftp3"
    file_bytes: int = 500_000            # FTP3 常用 0.5 MB
    arrival_rate_hz: float = 2.0         # 每用户每秒到达几个文件
    cbr_mbps: float = 5.0                # CBR 模式的恒定速率
    # --- bimodal：现网话务按**占用 RBG 数**的分布，两头高中间低 ---
    # 用户 2026-08-02 给的现网口径：
    #   1 个 RBG（小包）约 30%、17 个 RBG（满带宽）约 30%、
    #   2~16 个 RBG 相对均匀分布，折合平均 PRB 利用率约 30%
    #   （另有约 30% 的 TTI 根本没有调度，0 个 RBG）
    # **这是"一次传输占多少频域资源"的分布，不是文件大小的分布。**
    # 我第一版理解成了文件大小，两者完全不同：前者决定单次调度的 TBS，
    # 后者决定一个 burst 要发多少个 TTI。
    p_small_rbg: float = 0.30            # 只占 1 个 RBG
    p_full_rbg: float = 0.30             # 占满全部 RBG
    p_idle_tti: float = 0.30             # 根本没有调度的 TTI 占比

    def as_dict(self) -> dict[str, Any]:
        d = {"model": self.model}
        if self.model == "ftp3":
            d |= {"file_bytes": self.file_bytes, "arrival_rate_hz": self.arrival_rate_hz,
                  "offered_load_mbps_per_ue":
                      round(self.file_bytes * 8 * self.arrival_rate_hz / 1e6, 3)}
        elif self.model == "cbr":
            d |= {"cbr_mbps": self.cbr_mbps}
        elif self.model == "bimodal":
            d |= {"p_small_rbg": self.p_small_rbg, "p_full_rbg": self.p_full_rbg,
                  "p_idle_tti": self.p_idle_tti,
                  "expected_prb_utilization": round(self.expected_prb_util(), 4),
                  "note": ("**按占用 RBG 数分布，不是按文件大小。** 现网两头高中间低："
                           "1 个 RBG 约 30%、满带宽约 30%、中间相对均匀，"
                           "另有约 30% 的 TTI 根本没有调度。"
                           "**小包测不到体验速率**——一个 TTI 就发完，"
                           "3GPP 掐尾口径下没有可测量的时间。")}
        return d

    def expected_prb_util(self, num_rbg: int = 17) -> float:
        """这套分布折合出来的平均 PRB 利用率。**这是设计意图，不是仿真结果。**

        ``p_idle_tti`` **不驱动任何仿真行为**——它只出现在这个解析式里。
        真实的空闲 TTI 来自"没有用户有数据"，由到达率与信道共同决定，
        由主循环如实测出来（``cell.occupancy``）。

        **强行按概率随机拒绝调度是假物理**：真实调度器不会在有数据时掷骰子
        放弃这个 TTI。所以这个旋钮保留为对标锚点而不是输入，
        实测与它偏离超过 10 个百分点时 :func:`simulate` 会在 ``notes`` 里告警——
        那说明到达率没调到位，而不是仿真错了。
        """
        p_mid = max(0.0, 1.0 - self.p_small_rbg - self.p_full_rbg)
        mid_mean = (2 + num_rbg - 1) / 2.0 / num_rbg     # 2~16 均匀的均值
        busy = (self.p_small_rbg * (1.0 / num_rbg)
                + self.p_full_rbg * 1.0 + p_mid * mid_mean)
        return float(busy * (1.0 - self.p_idle_tti))


@dataclass
class SchedulerConfig:
    """调度器。

    ``pf`` 比例公平：度量 ``R_inst / R_avg``，``R_avg`` 按指数窗更新::

        R_avg(t+1) = (1 - 1/Tc)·R_avg(t) + (1/Tc)·R_served(t)

    ``Tc`` 就是 ``pf_window_tti``。它决定公平的时间尺度：太小接近 max-C/I
    （只喂近点用户），太大接近轮询（不利用信道起伏）。
    """

    algorithm: SchedAlgorithm = "pf"
    pf_window_tti: int = 100
    # --- OLLA（外环链路自适应）---
    # 发送端按无干扰选 MCS，接收端吃着干扰误码，OLLA 把偏置压下来。
    # 步长按目标 BLER 不对称：ACK 加 up、NACK 减 down，
    # 稳态时 BLER → up/(up+down)。**现网基线是 +0.01/−0.1**（≈9.1% BLER）。
    # 步长放大能加快收敛但会在稳态附近抖得更厉害——要快收敛就临时调大，
    # 出正式结论用基线值。
    olla_enabled: bool = True
    # **步长比决定稳态 BLER，与步长绝对值无关。** 推导见 :func:`olla_step_down_for`。
    # 用户 2026-08-02 给的现网粗估是 +0.01/−0.1，但那对应稳态 9.09% 而不是 10%；
    # 2026-08-03 他自己也指出 NACK 应该是 −0.09 左右。按目标 10% 精确解就是 −0.09。
    olla_step_up_db: float = 0.01        # 现网基线（用户 2026-08-02）
    olla_step_down_db: float = 0.09       # 稳态 BLER -> 0.01/(0.01+0.09) = 10.0%
    olla_min_db: float = -20.0
    olla_max_db: float = 3.0
    # **加速收敛用的等比放大系数**（用户 2026-08-03 批准，条件是必须告知）。
    # 两个步长同乘一个数，稳态 BLER = up/(up+down) **完全不变**，
    # 变的只有收敛速度和稳态附近的抖动幅度。
    # 现网基线 +0.01/−0.1 在整数 MCS 档上常常压不动一档，8 秒仿真里
    # BLER 还停在 0.16~0.22；放大 10 倍能在同样时长内收敛，
    # 代价是稳态抖动更大。**出正式结论时用 1.0。**
    olla_speedup: float = 1.0
    mu_enabled: bool = True              # 是否允许 MU 配对（SU/MU 自适应）
    max_mu_users: int = 4

    @property
    def step_up(self) -> float:
        return self.olla_step_up_db * max(float(self.olla_speedup), _EPS)

    @property
    def step_down(self) -> float:
        return self.olla_step_down_db * max(float(self.olla_speedup), _EPS)

    def as_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "algorithm": self.algorithm, "pf_window_tti": self.pf_window_tti,
            "mu_enabled": self.mu_enabled, "max_mu_users": self.max_mu_users,
            "olla_enabled": self.olla_enabled,
            "olla_baseline_steps_db": [self.olla_step_up_db, self.olla_step_down_db],
            "olla_speedup": self.olla_speedup,
            "olla_effective_steps_db": [round(self.step_up, 6),
                                        round(self.step_down, 6)],
            "olla_target_bler": round(
                self.olla_step_up_db / (self.olla_step_up_db
                                        + self.olla_step_down_db), 3)}
        if self.olla_speedup != 1.0:
            d["olla_speedup_warning"] = (
                f"**OLLA 步长已等比放大 {self.olla_speedup:g} 倍**"
                f"（{self.olla_step_up_db:g}/{self.olla_step_down_db:g} → "
                f"{self.step_up:g}/{self.step_down:g}）。稳态 BLER 不变"
                f"（仍是 {self.olla_step_up_db / (self.olla_step_up_db + self.olla_step_down_db):.1%}），"
                f"但稳态附近抖动更大。这是为了在短仿真里收敛，"
                f"**出正式结论请把 olla_speedup 设回 1.0**。")
        return d


@dataclass
class KpiConfig:
    """KPI 统计口径。**换口径数字会明显变，所以它必须跟着结果一起走。**"""

    trim: ThroughputTrim = "tail"
    min_burst_tti: int = 2               # 短于这个的 burst 不计入体验速率
    warmup_tti: int = 200                # 前多少个 TTI 不计入统计（PF 均值要收敛）

    def as_dict(self) -> dict[str, Any]:
        return {"trim": self.trim, "min_burst_tti": self.min_burst_tti,
                "warmup_tti": self.warmup_tti,
                "trim_note": {
                    "none": "不掐，含清空缓冲区的那个 TTI（数值虚高，不建议）",
                    "tail": "掐尾：排除清空缓冲区的最后一个 TTI（3GPP TS 28.552 §5.1.1.3）",
                    "head_tail": "掐头去尾：再排除首个 TTI（运营商话统常用口径）",
                }[self.trim]}


@dataclass
class NeighborLoadConfig:
    """邻区负载。**不能假设所有小区都是 full buffer。**

    ChannelHub 的几何 SINR 是按**所有邻区都在发**算出来的，等于 100% PRB
    利用率。真实网络 5G 典型平均 PRB 利用率是 10% / 30% / 50%——
    邻区没在发的那些 PRB 上，本小区用户根本不受干扰。
    按 full buffer 算会把干扰放大到不真实的程度。

    折算方式：干扰功率按利用率 ``η`` 线性缩放，噪声不变::

        SINR' = S / (η·I + N)，其中 I = S/SIR、N = S/SNR

    ``prb_utilization = 1.0`` 时退化成原来的 full buffer 行为。

    **当前只支持全网配同一个负载值**（用户 2026-08-03 定的口径）。
    真实网络里各小区负载当然不同，但那要一张逐小区的负载表，
    而 ChannelHub 的几何 SINR 只给出**聚合**的 SIR——拿不到"哪个邻区贡献了多少"，
    没法把逐小区负载映射回来。所以现在是一个标量。

    ``jitter`` 让**实际生效值**在配置值的 ±5% 内随机波动。这不是装饰：
    现网负载本来就是逐 TTI 抖的，一个恒定值会让所有快照的干扰完全一样，
    结果看起来比真实情况干净。波动是乘性的，``0.3 → [0.285, 0.315]``。
    """

    prb_utilization: float = 0.3          # 5G 典型：0.1 / 0.3 / 0.5
    jitter: float = 0.05                  # 实际值在 ±5% 内波动（用户 2026-08-03）
    seed: int = 0

    def realized(self, n: int, rng: np.random.Generator | None = None) -> np.ndarray:
        """抽 ``n`` 个实际生效的利用率。``jitter=0`` 时就是 n 份配置值。"""
        r = rng if rng is not None else np.random.default_rng(self.seed)
        base = float(self.prb_utilization)
        if self.jitter <= 0:
            return np.full(n, base)
        lo, hi = base * (1.0 - self.jitter), base * (1.0 + self.jitter)
        return np.clip(r.uniform(lo, hi, size=n), 0.0, 1.0)

    def as_dict(self) -> dict[str, Any]:
        return {"prb_utilization": self.prb_utilization,
                "jitter": self.jitter,
                "realized_range": [round(self.prb_utilization * (1 - self.jitter), 4),
                                   round(self.prb_utilization * (1 + self.jitter), 4)],
                "scope": "network_wide_single_value",
                "note": ("邻区按这个 PRB 利用率折算干扰；1.0 等于假设所有邻区"
                         "full buffer（ChannelHub 几何 SINR 的原始假设）。"
                         f"实际生效值逐快照在 ±{self.jitter * 100:.0f}% 内波动。"
                         "当前只支持全网统一值——几何 SIR 是聚合量，"
                         "拿不到逐邻区的贡献，没法映射逐小区负载。")}


def apply_neighbor_load(sinr_db: float, sir_db: float, utilization: float) -> float:
    """把几何 SINR 按邻区负载折算。返回新的 SINR（dB）。

    推导：令 S=1，则 I = 1/SIR_lin、N = 1/SINR_lin − I。
    邻区只在 ``η`` 比例的 PRB 上发，干扰变成 ``η·I``，噪声不变::

        SINR' = 1 / (η·I + N)

    ``sir_db`` 拿不到（单小区哨兵 49.9）时原样返回——没有干扰可折算。
    """
    u = min(max(float(utilization), 0.0), 1.0)
    if u >= 1.0 or not np.isfinite(sir_db) or sir_db >= 49.0:
        return float(sinr_db)
    s_lin = 10.0 ** (float(sinr_db) / 10.0)
    i_lin = 10.0 ** (-float(sir_db) / 10.0)
    n_lin = 1.0 / s_lin - i_lin
    if n_lin <= 0:                        # 口径对不上时不硬算
        return float(sinr_db)
    return float(10.0 * np.log10(1.0 / (u * i_lin + n_lin)))


@dataclass
class SystemConfig:
    duration_s: float = 5.0
    scs_khz: int = 30                    # 30 kHz → slot 0.5 ms
    num_rbg: int = 17
    rb_per_rbg: int = 16
    tdd_pattern: str = "DDDSU"           # 只统计 D 时隙
    # **信道快照之间隔多久，由 ChannelHub 决定，不能拍脑袋。**
    # internal_sim.py:3252 把 UE 每个"时隙"推进
    #     speed × max(srs_periodicity, csirs_periodicity) × slot_duration_s
    # 默认 10 × 0.5 ms = **5 ms**——它们是连续的 SRS/CSI-RS 机会，不是连续 TTI。
    #
    # 我原来拍了 10.0，差 2 倍；更要命的是在量 CSI 老化时把「滞后 1 个快照」
    # 读成了 0.5 ms（一个 TTI），实际是 5 ms，**整整差 10 倍**。
    # 验证方法：Jakes 的 ρ(τ)=|J0(2π·fd·τ)| 首零点在 τ=2.405/(2π·fd)，
    # 3 km/h 时是 53 ms；实测极小值落在第 10 个快照 → 每快照 5.3 ms，对上。
    snapshot_update_ms: float = 5.0
    seed: int = 0

    @property
    def tti_ms(self) -> float:
        return 1.0 / (self.scs_khz / 15.0)          # 15→1ms, 30→0.5ms, 60→0.25ms

    @property
    def num_tti(self) -> int:
        return int(round(self.duration_s * 1000.0 / self.tti_ms))

    @property
    def dl_ratio(self) -> float:
        """TDD 图案里下行时隙占比。S 时隙按 0.7 个下行折算（大部分符号是 D）。"""
        p = self.tdd_pattern.upper() or "D"
        return (p.count("D") + S_SLOT_DL_FRACTION * p.count("S")) / len(p)

    def as_dict(self) -> dict[str, Any]:
        return {"duration_s": self.duration_s, "scs_khz": self.scs_khz,
                "tti_ms": self.tti_ms, "num_tti": self.num_tti,
                "num_rbg": self.num_rbg, "rb_per_rbg": self.rb_per_rbg,
                "num_rb": self.num_rbg * self.rb_per_rbg,
                "tdd_pattern": self.tdd_pattern,
                "dl_slot_ratio": round(self.dl_ratio, 4),
                "snapshot_update_ms": self.snapshot_update_ms, "seed": self.seed}


# ---------------------------------------------------------------------------
# 第一相：把信道压成查表
# ---------------------------------------------------------------------------
@dataclass
class UeLinkTable:
    """一个 UE 在各个信道快照下、各个 rank 的链路能力。TTI 主循环只读它。"""

    ue: int
    sinr_db: np.ndarray                  # [snapshot, rank] 用户级 SINR
    mcs: np.ndarray                      # [snapshot, rank]
    se: np.ndarray                       # [snapshot, rank] = rank × MCS 谱效
    best_rank: np.ndarray                # [snapshot] rank 自适应选中的秩（1-indexed）
    best_se: np.ndarray                  # [snapshot]
    geo_sinr_db: float
    outage: np.ndarray | None = None     # [snapshot] 该快照下根本调度不动
    iot_db: float = float("nan")         # 干扰抬升：(I+N)/N，>20 dB 算高干扰
    iot_sample_valid: float = 1.0        # 这个 UE 有多少比例的**快照**算得出 IoT
    sir_db: float = float("nan")
    # **发送侧与接收侧是两个 SINR。** 发送端一开始不知道干扰，
    # 按无干扰（或 CQI 反馈的粗略统计）选 MCS；接收端实打实吃着干扰。
    # 两者的差通过误码由 OLLA 收敛回来——这是干扰影响吞吐的第一性路径。
    sinr_tx_db: np.ndarray | None = None  # [snapshot, rank] CQI 门限 + BF Gain
    mcs_tx: np.ndarray | None = None      # [snapshot, rank] 发送端据此定的 MCS
    # --- 发送侧 SINR 的拆解，供审计与说明书引用 ---
    bf_gain_db: np.ndarray | None = None   # [snapshot, rank] SVD − PMI（基站自算）
    pmi_sinr_db: np.ndarray | None = None  # [snapshot, rank] Type I 权下的用户级 SINR
    cqi_index: np.ndarray | None = None    # [rank] 长期滤波后上报的 CQI
    csi_lag_snapshots: np.ndarray | None = None  # [snapshot] 平均 CSI 滞后（快照数）
    # **基站以为的谱效**：rank 自适应与 PF 调度都只能看它，不能看真实值。
    # 拿真实谱效去调度等于让基站预知信道，老化损失会被凭空抹掉一大半。
    # 零时延时它与 ``se`` / ``best_se`` 逐位相同。
    se_gnb: np.ndarray | None = None       # [snapshot, rank]
    best_se_gnb: np.ndarray | None = None  # [snapshot]


def la_sel(sinr_db: float, table: int, target_bler: float) -> int:
    """选 MCS 的薄封装，建表时用。"""
    from . import linkadapt as la  # noqa: PLC0415

    return int(la.select_mcs(float(sinr_db), table=table,
                             target_bler=target_bler).index)


def olla_step_down_for(target_bler: float, step_up: float = 0.01) -> float:
    """给定目标 IBLER 与 ACK 步长，反解 NACK 步长。

    OLLA 是个随机逼近：ACK 加 ``s_up``、NACK 减 ``s_down``。偏置在期望漂移
    为零时稳态::

        (1 − p)·s_up = p·s_down   ⟹   p = s_up / (s_up + s_down)
                                  ⟹   s_down = s_up · (1 − p) / p

    **稳态 BLER 只取决于两个步长的比**，与绝对值无关——绝对值只影响收敛速度
    和稳态附近的抖动（这正是 ``olla_speedup`` 能等比放大的原因）。

    目标 10%、``s_up = 0.01`` 时 ``s_down = 0.09``。
    **现网常说的 +0.01/−0.1 其实对应 9.09% 而不是 10%**，差得不多但不是一回事。

    注意这是**连续偏置**下的理想稳态。实际 MCS 是整数档，偏置要累积到跨过
    一整档才会真正改变发送，所以实测 BLER 会围绕理论值抖，且与信道的
    档位间隔有关——:func:`simulate` 会把实测值报出来，别只信理论值。
    """
    p = float(target_bler)
    if not 0.0 < p < 1.0:
        raise ValueError(f"target_bler 必须在 (0,1)，收到 {target_bler}")
    return float(step_up) * (1.0 - p) / p


def _type1_precoder(h_rbg: np.ndarray, rank: int) -> np.ndarray:
    """38.214 Type I **宽带** PMI，强制到指定 rank。``[F,BS,UE]`` → ``[F,BS,rank]``。

    宽带意味着全带共用一个权（``compute_precoder`` 内部就是在频率平均的信道上
    搜码本再广播回各 RBG），这正对应用户口径里的**全带 CQI**——
    不做子带 CQI、不做频选调度。
    """
    from . import linklevel as ll  # noqa: PLC0415

    return ll.compute_precoder(np.asarray(h_rbg)[None], method="type1",
                               max_rank=int(rank), rank_threshold=0.0).w


def _cqi_of(sinr_db: float, target_bler: float) -> int:
    """用户级 SINR → CQI index（38.214 Table 5.2.2.1-3，即 CQI 表 2）。"""
    from . import linkadapt as la  # noqa: PLC0415

    if not np.isfinite(sinr_db):
        return 0
    return int(la.select_cqi(float(sinr_db), table=2,
                             target_bler=float(target_bler)))


def _cqi_threshold_sinr(cqi_index: int, target_bler: float) -> float:
    """CQI → 按谱效映射的初始 MCS → 该 MCS 在目标 BLER 下的 NewTx SINR 门限。

    这是现场 TDD AMC 链路的前两步。CQI=0（不可调度）时返回 −inf，
    让下游把它判成发不出去，而不是悄悄当成 MCS 0。
    """
    from . import bler_curves as bc  # noqa: PLC0415
    from . import linkadapt as la  # noqa: PLC0415

    m = la.cqi_to_mcs_by_se(int(cqi_index), cqi_table=2, mcs_table=3)
    if not m["scheduled"]:
        return float("-inf")
    return float(bc.get_curve(int(m["mcs"]), "newtx").required_sinr_db(float(target_bler)))


def _nan_safe(fn, values, *args) -> float:
    """全是 NaN 时返回 NaN 而不是让 numpy 抛 RuntimeWarning。"""
    v = [x for x in values if np.isfinite(x)]
    return float(fn(v, *args)) if v else float("nan")


def interference_free_sinr(sinr_db: float, sir_db: float) -> float:
    """从含干扰的几何 SINR 反推**无干扰**的 SNR（同口径）。

    令 S=1：``I = 1/SIR``、``I+N = 1/SINR``，所以 ``N = 1/SINR − 1/SIR``，
    无干扰时 ``SNR = 1/N``。

    **这是发送端一开始看到的世界。** 发送端不知道瞬时干扰，按无干扰
    （或 CQI 反馈的粗略统计）选 MCS；接收端实打实吃着干扰，SINR 更低，
    于是误码，OLLA 把偏置压下来。干扰越大，OLLA 收敛到的偏置越负——
    这就是"干扰越大、接收 SINR 越低、吞吐越低"的第一性路径。

    **别用 ChannelHub 的 snr_dB 代替。** 它是另一个口径（不含阵列增益、
    额外减了 10log10(RB)），和 sinr_dB 差几十 dB，见 CLAUDE.md。
    """
    if not (np.isfinite(sinr_db) and np.isfinite(sir_db)) or sir_db >= 49.0:
        return float(sinr_db)
    s_lin = 10.0 ** (float(sinr_db) / 10.0)
    i_lin = 10.0 ** (-float(sir_db) / 10.0)
    n_lin = 1.0 / s_lin - i_lin
    if n_lin <= 0:
        return float(sinr_db)
    return float(-10.0 * np.log10(n_lin))


def _iot(sinr_db: float, sir_db: float) -> float:
    """IoT = SIR/(SIR−SINR)（线性域）。**只能用同口径的两个量**。

    体现的是干扰主导还是噪声主导：IoT 接近 0 dB 说明几乎没有干扰、
    完全是噪声受限；密集城区经常到 20 dB 以上，那时干扰是绝对主导，
    再加发射功率也没用（信号和干扰同步上涨）。
    """
    from . import interference as itf  # noqa: PLC0415

    if not (np.isfinite(sinr_db) and np.isfinite(sir_db)):
        return float("nan")
    # **SIR < SINR 物理上不可能**（SINR = S/(I+N) ≤ S/I = SIR）。
    # 出现它只有一个原因：两个量不同口径。实测 num_slots_per_sample=4 时
    # sinr_dB 是各 slot 的 dB 均值、sir_dB 只取最后一个 slot，
    # 20 个样本里 12 个的 sir−sinr 是负的（最小 −9.5 dB），IoT 直接算成 inf。
    # 单时隙下同一场景 IoT 中位 32.2 dB —— 正对应现网密集城区 >20 dB。
    # 宁可返回 nan 也不给一个偏低到误导人的数。
    if sir_db < sinr_db - 1e-6:
        return float("nan")
    return float(np.asarray(itf.iot_db(sinr_db, sir_db)).item())


def snapshot_interval_ms(cfg: dict[str, Any]) -> float:
    """由配置算出信道快照之间隔多久（ms）。

    ChannelHub 的多时隙输出不是连续 TTI，而是**连续的 SRS/CSI-RS 机会**：
    ``internal_sim.py:3252`` 把 UE 每个时隙推进
    ``speed × max(srs_periodicity, csirs_periodicity) × slot_duration_s``。

    默认 10 × 0.5 ms = 5 ms。**把它当成一个 TTI（0.5 ms）会让所有
    时间相关的结论差 10 倍**——CSI 老化、多普勒、移动性全部受影响。
    """
    scs = float(cfg.get("subcarrier_spacing", 30_000) or 30_000)
    slot_ms = 1.0 / (scs / 15_000.0)
    per = max(int(cfg.get("srs_periodicity", 10) or 10),
              int(cfg.get("csirs_periodicity", 10) or 10))
    return slot_ms * per


def group_samples_by_ue(n_samples: int, num_ues: int) -> list[list[int]]:
    """把数据集里的样本按 UE 分组。

    **样本数不等于用户数。** ChannelHub 一次生成 ``num_samples`` 个样本，
    分布在 ``num_ues`` 个 UE 位置上（轮转分配，每 UE
    ``num_samples/num_ues`` 个）。把每个样本当成一个独立用户，
    小区就被塞进了 4 倍的人——实测 40 样本 / 10 UE 的配置下，
    每用户谱效从应有的 0.32 掉到 0.08，**看起来像边缘用户被饿死**，
    其实是分母大了 4 倍。

    同一个 UE 的多个样本是**时间相关的**（多普勒就是从相邻样本的位移算的），
    所以它们正好当这个 UE 的信道快照序列用。
    """
    n_ue = max(1, min(int(num_ues), int(n_samples)))
    return [list(range(u, int(n_samples), n_ue)) for u in range(n_ue)]


def build_link_tables(
    h_users: list[np.ndarray],
    geo_sinr_db: list[float],
    *,
    geo_sir_db: list[float] | None = None,
    neighbor_load: float = 1.0,
    max_rank: int = mu.SU_MAX_RANK,
    table: int = 3,
    target_bler: float = 0.1,
    num_snapshots: int = 1,
    num_ues: int | None = None,
    rb_per_rbg: int = 16,
    csi: ca.CsiConfig | None = None,
    snapshot_ms: float = 5.0,
    load_jitter_rng: np.random.Generator | None = None,
    precoder: str = "svd",
) -> list[UeLinkTable]:
    """第一相：逐 UE 把 rank 1..max_rank 的 SINR / MCS / 谱效全部算好。

    **SVD 只在这里做。** 主循环里再也不碰矩阵——这是十万 TTI 能跑完的原因。

    ``h_users[i]`` 形状 ``[T, RB, BS, UE]``；``T > 1`` 时把每个时隙当一个
    独立快照（ChannelHub 的多时隙是时间相关的，正好用来表达信道起伏）。

    ``num_ues`` 给定时，按 :func:`group_samples_by_ue` 把样本合并成这么多个
    用户，同一 UE 的多个样本当作它的快照序列。**不给的话每个样本算一个用户**
    ——那通常不是你想要的，见该函数的说明。

    ``csi`` 给定且 ``enabled`` 时走 CSI 老化：预编码用滞后若干个快照的信道，
    评估用当前快照。逐 RBG 的滞后由 SRS 周期与跳频决定，见 :mod:`csi_aging`。
    不给的话是零时延完美 CSI——**那是个上界，不是现网**。

    ``load_jitter_rng`` 只用来抽邻区负载的逐快照抖动，**应当来自
    ``rng.RngBook(...).generator("neighbor_load")``**，不要在调用处写
    ``default_rng(seed + 常数)``（NumPy 并行随机数文档把它标成
    "UNSAFE! Do not do this!"）。

    **除了它，本函数完全确定性**——SVD、码本搜索、MCS 查表都不含随机。
    这正是 :func:`simulate_replications` 能"建一次表、重跑 n 次主循环"的前提；
    ``tests/test_rng.py`` 第 8 节逐位断言了这条。

    ``precoder`` 决定**实际发射权**：``svd``（逐 RBG 特征波束，理论最优）
    或 ``type1``（38.214 Type I 宽带码本）。注意 Type I 权在两种模式下都要算——
    它是 CQI 与 BF Gain 的参照系；``precoder="type1"`` 只是把它同时当成发射权，
    于是 BF Gain 恒为 0（发射权就是参照权）。
    """
    if precoder not in ("svd", "type1"):
        raise ValueError(f"precoder 只支持 'svd' / 'type1'，收到 {precoder!r}")
    sir_in = list(geo_sir_db) if geo_sir_db is not None else [float("nan")] * len(h_users)
    # 逐快照的几何量。**合并成一个均值会把动态范围压掉一半**——
    # 实测 40 个样本的 SINR 跨度 20.7 dB，按 UE 取均值后只剩 11.9 dB，
    # "5% 边缘用户"于是变成了一个中等信道的用户，边缘 MCS 报 8.2 而不是 <5。
    # 现在每个快照保留自己的 SINR/SIR，只有对外报的标量才取均值。
    per_snap_sinr: list[list[float]] = [[float(x)] for x in geo_sinr_db]
    per_snap_sir: list[list[float]] = [[float(x)] for x in sir_in]
    if num_ues is not None and num_ues < len(h_users):
        groups = group_samples_by_ue(len(h_users), num_ues)
        merged_h, merged_g, merged_s = [], [], []
        for g in groups:
            per_sample = [np.asarray(h_users[i]) for i in g]
            merged_h.append(np.concatenate(
                [x.reshape(-1, *x.shape[-3:]) for x in per_sample], axis=0))
            # 每个样本贡献 T 个快照，它的几何量在这 T 个快照上重复
            merged_g.append([float(geo_sinr_db[i])
                             for i, x in zip(g, per_sample, strict=True)
                             for _ in range(x.shape[0] if x.ndim == 4 else 1)])
            merged_s.append([float(sir_in[i])
                             for i, x in zip(g, per_sample, strict=True)
                             for _ in range(x.shape[0] if x.ndim == 4 else 1)])
        h_users = merged_h
        per_snap_sinr, per_snap_sir = merged_g, merged_s
        geo_sinr_db = [float(np.nanmean(v)) for v in merged_g]
        sir_in = [float(np.nanmean(v)) for v in merged_s]

    # **邻区不是 full buffer。** 按 PRB 利用率折算干扰后再建表——
    # 折算必须发生在算 SINR/MCS/rank 之前，事后乘系数是补不回来的。
    if neighbor_load < 1.0:
        # **SINR 和 SIR 必须一起折算。** 干扰降到 η 倍，SIR 就升高 1/η；
        # 只改 SINR 会让后面的 IoT = SIR/(SIR−SINR) 用两个不同口径的量算，
        # 直接报 inf。这和 CLAUDE.md 里「IoT 不是 snr 减 sinr」是同一类错。
        #
        # 负载**逐快照抖动**（NeighborLoadConfig.jitter，默认 ±5%）时，
        # 每个快照拿自己那份 η——所以下面是逐元素而不是一个全局标量。
        def _one(g: float, sr: float, u: float) -> tuple[float, float]:
            u_db = 10.0 * np.log10(max(u, 1e-12))
            new_sir = (sr - u_db) if np.isfinite(sr) and sr < 49.0 else sr
            return apply_neighbor_load(g, sr, u), new_sir

        loads = [
            (load_jitter_rng.uniform(neighbor_load * 0.95, neighbor_load * 1.05, len(gs))
             if load_jitter_rng is not None else np.full(len(gs), neighbor_load))
            for gs in per_snap_sinr
        ]
        _pairs = [[_one(g, sr, float(u)) for g, sr, u in zip(gs, ss, us, strict=True)]
                  for gs, ss, us in zip(per_snap_sinr, per_snap_sir, loads, strict=True)]
        per_snap_sinr = [[p[0] for p in row] for row in _pairs]
        per_snap_sir = [[p[1] for p in row] for row in _pairs]
        geo_sinr_db = [float(np.nanmean(v)) for v in per_snap_sinr]
        sir_in = [float(np.nanmean(v)) for v in per_snap_sir]

    out: list[UeLinkTable] = []
    aging = csi is not None and csi.enabled
    for i, h in enumerate(h_users):
        hh = np.asarray(h)
        snaps = [hh[t:t + 1] for t in range(hh.shape[0])] if hh.ndim == 4 else [hh]
        if len(snaps) < num_snapshots:
            # 时隙不够就复用，但**不伪造起伏**——复用会在结果里如实标注
            snaps = [snaps[t % len(snaps)] for t in range(num_snapshots)]
        n_s = len(snaps)

        # --- 压成二维再降粒度，老化与 BF Gain 都在这个粒度上算 ---
        _2d = [np.asarray(x).mean(axis=0) if np.asarray(x).ndim == 4 else np.asarray(x)
               for x in snaps]
        if rb_per_rbg > 1:
            snaps_u = [mu.rbg_reduce(x, rb_per_rbg) for x in _2d]   # 每行 = 1 RBG
            grp, rows_per_rbg = 1, 1
        else:
            snaps_u = _2d                                           # 每行 = 1 RB
            grp, rows_per_rbg = mu.RB_PER_RBG, mu.RB_PER_RBG
        n_rows = snaps_u[0].shape[0]
        n_rbg_eff = max(1, int(np.ceil(n_rows / rows_per_rbg)))

        # --- Type I 宽带 PMI：逐 UE 逐 rank 只搜一次码本 ---
        # 宽带 PMI 是**慢时间尺度**的量（38.214 里它的上报周期远长于一个 TTI），
        # 逐快照重搜既慢又不符合物理。在时间平均信道上搜一次就是它该有的样子。
        h_mean = np.mean(np.stack(snaps_u), axis=0)
        # **只搜一次。** Type I 的选波束是**增量贪心**（选一层、投影掉、再选下一层），
        # 所以 rank R 结果的前 r 列与直接搜 rank r **逐位相同**（实测偏差 0.0）。
        # 逐 rank 各搜一遍白花 4 倍时间——实测码本搜索本来就占建表的 47%。
        _w_all = _type1_precoder(h_mean, max_rank)
        w_pmi = {r: _w_all[:, :, :r] for r in range(1, max_rank + 1)}

        sinr = np.zeros((n_s, max_rank))
        mcs = np.zeros((n_s, max_rank), dtype=int)
        se = np.zeros((n_s, max_rank))
        sinr_tx = np.zeros((n_s, max_rank))
        mcs_tx = np.zeros((n_s, max_rank), dtype=int)
        bf_gain = np.zeros((n_s, max_rank))
        pmi_sinr = np.zeros((n_s, max_rank))
        lag_used = np.zeros(n_s)
        se_gnb = np.zeros((n_s, max_rank))       # 基站以为的谱效，rank 与调度都看它
        rank_gnb = np.ones(n_s, dtype=int)
        _gs = per_snap_sinr[i] if i < len(per_snap_sinr) else [geo_sinr_db[i]]
        _ss = per_snap_sir[i] if i < len(per_snap_sir) else [sir_in[i]]
        for s, hs in enumerate(snaps):
            _g = _gs[s % len(_gs)]
            # 逐快照用它自己的几何 SINR，不用 UE 的均值——保住动态范围
            npow = mu.noise_from_geometric_sinr(hs, _g)

            # **基站看到的信道**：零时延时就是当前快照；开老化时逐 RBG 滞后。
            if aging:
                lag_rbg = ca.rbg_lag_snapshots(csi, n_rbg_eff, snapshot_ms=snapshot_ms,
                                               snapshot_index=s, rb_per_rbg=rb_per_rbg)
                lags = np.repeat(lag_rbg, rows_per_rbg)[:n_rows]
                h_prec = ca.stale_channel(snaps_u, s, lags)
                lag_used[s] = float(np.mean(lag_rbg))
            else:
                h_prec = snaps_u[s]

            # 预编码用 h_prec、评估用当前快照。零时延时两者相同，
            # 结果与 mumimo.su_rank_adaptation **逐位相同**（test_csi_aging 第 1 节）。
            # Type I 权是**在陈旧信道的时间平均上**搜的（宽带 PMI 本就是慢量），
            # 所以它同样吃老化——只是自由度少，能算错的地方也少。
            _wov = _w_all if precoder == "type1" else None
            rc = ca.rank_adaptation_aged(h_prec, snaps_u[s], noise_power=npow,
                                         max_rank=max_rank, table=table,
                                         target_bler=target_bler, rb_per_rbg=grp,
                                         w_override=_wov)
            for c in rc.candidates:
                r = c["rank"] - 1
                sinr[s, r], mcs[s, r], se[s, r] = c["sinr_db"], c["mcs"], c["se"]
            for c in rc.gnb_candidates:
                se_gnb[s, c["rank"] - 1] = c["se"]
            rank_gnb[s] = rc.rank

            # --- BF Gain = SVD − PMI，**两者都在基站自己的（陈旧）CSI 上算** ---
            # 基站是从 SRS 拿的信道，它能自己算出 BF Gain，但算的是滞后那一刻的。
            # 老化时这会让它**高估**增益（以为预编码是匹配的），于是 MCS 点高了，
            # 误码上来，再由 OLLA 拉回去——这正是现网的机制。
            # BF Gain 是**实际发射权**相对 PMI 参照权的增益。
            # precoder="type1" 时两者是同一个权，所以它恒为 0——这不是特例处理，
            # 是定义的直接后果：码本发送没有额外的 BF 增益可加。
            w_tx_prec = _w_all if precoder == "type1" else ca.svd_precoder(h_prec)
            for r in range(1, max_rank + 1):
                p_per = 1.0 / r
                s_tx = ca.mmse_stream_sinr(h_prec, w_tx_prec[:, :, :r],
                                           power_per_stream=p_per, noise_power=npow)
                s_pmi = ca.mmse_stream_sinr(h_prec, w_pmi[r],
                                            power_per_stream=p_per, noise_power=npow)
                g_tx = mu.user_sinr_db(s_tx, rb_per_rbg=grp)
                g_pmi = mu.user_sinr_db(s_pmi, rb_per_rbg=grp)
                bf_gain[s, r - 1] = g_tx - g_pmi
                # CQI 是终端在**真实信道**上用 PMI 权测的，所以这里用当前快照
                pmi_sinr[s, r - 1] = mu.user_sinr_db(
                    ca.mmse_stream_sinr(snaps_u[s], w_pmi[r],
                                        power_per_stream=p_per, noise_power=npow),
                    rb_per_rbg=grp)

        # --- 发送侧 SINR = CQI 门限 + BF Gain（用户 2026-08-03 定的口径）---
        # 现场流程（CLAUDE.md 已固化）：
        #   CQI → 按谱效映射初始 MCS → 该 MCS 的目标 BLER SINR 门限
        #   → + BF Gain → 按 SINR 重映射 MCS → + OLLA → floor
        # 这里只走到"+BF Gain"为止，OLLA 留在 TTI 主循环里逐 TTI 更新。
        #
        # **CQI 是长期滤波的宽带量**（终端上报周期远长于一个 TTI），
        # 所以取该用户 PMI SINR 在全部快照上的均值再量化；
        # **BF Gain 是瞬时的**，基站每次调度都能从自己的 CSI 算出来。
        # 早先版本把发送侧写成"接收 SINR 的长期均值"，那是个事后诸葛亮的量——
        # 它已经包含了 SVD 的增益，等于假设基站预先知道自己波束打得准不准。
        cqi_idx = np.zeros(max_rank, dtype=int)
        for _r in range(max_rank):
            mean_pmi = float(np.nanmean(pmi_sinr[:, _r]))
            cqi_idx[_r] = _cqi_of(mean_pmi, target_bler)
            thr = _cqi_threshold_sinr(int(cqi_idx[_r]), target_bler)
            # **CQI=0 不能退化成 −inf。** 它的意思是"低于 CQI 表下界"，
            # 而不是"这个用户不存在"——真实接收 SINR 可能还有几个 dB。
            # 退回实测 PMI SINR：粗糙但有限，OLLA 还能在它上面工作。
            if not np.isfinite(thr):
                thr = mean_pmi if np.isfinite(mean_pmi) else -20.0
            sinr_tx[:, _r] = thr + bf_gain[:, _r]
            for _s in range(n_s):
                mcs_tx[_s, _r] = la_sel(sinr_tx[_s, _r], table, target_bler)
        # **rank 由基站按自己的 CSI 挑**（零时延时 se_gnb 与 se 逐位相同）
        best = rank_gnb - 1
        # **覆盖判定。** 用户级 SINR 连 MCS 0 的 10% BLER 门限都够不到时，
        # 这个快照下他根本调度不动——发了也是白发。必须显式标出来：
        # PF 的度量是 R_inst/R_avg，一个永远发不成功的用户 R_avg 会趋近 0，
        # 度量发散，调度器于是死盯着他，把整个小区拖垮。这是 PF 的经典病理。
        outage = np.array([
            _bler_lookup(int(mcs[t, best[t]]), float(sinr[t, best[t]])) > 0.5
            for t in range(n_s)
        ])
        out.append(UeLinkTable(
            ue=i, sinr_db=sinr, mcs=mcs, se=se,
            best_rank=best + 1, best_se=se[np.arange(n_s), best],
            geo_sinr_db=float(geo_sinr_db[i]), outage=outage,
            # **IoT 逐快照算再取中位，不能拿平均后的 SINR/SIR 去算。**
            # 两个量各自平均后相减，差值可以塌到 0，IoT 直接报 inf——
            # 实测逐样本算出来是 5~41 dB，从来不是 inf。
            iot_db=_nan_safe(np.nanmedian, (_iots := [_iot(g, r) for g, r in
                                                     zip(_gs, _ss, strict=False)])),
            # **逐样本的有效率，不是逐用户。** 一个用户 8 个快照里 4 个算不出 IoT，
            # nanmedian 照样给出有限值 → 这个用户被算成"有效" → 小区级有效率报 100%，
            # 而实际一半样本被丢了。实测 ds_9625340c：逐用户 100%、逐样本只有 46%。
            # 粒度错了的后果不是"少报一个警告"，是**报错了另一个警告**：
            # 系统会去怪站间距和邻区负载，而真因是这个量本身在多时隙下就不成立。
            iot_sample_valid=float(np.mean([np.isfinite(x) for x in _iots]))
            if _iots else 0.0,
            sir_db=float(sir_in[i]), sinr_tx_db=sinr_tx, mcs_tx=mcs_tx,
            bf_gain_db=bf_gain, pmi_sinr_db=pmi_sinr, cqi_index=cqi_idx,
            csi_lag_snapshots=lag_used, se_gnb=se_gnb,
            best_se_gnb=se_gnb[np.arange(n_s), best],
        ))
    return out


def measure_mu_gain(
    h_users: list[np.ndarray],
    geo_sinr_db: list[float],
    *,
    num_ues: int | None = None,
    max_mu_users: int = 4,
    max_snapshots: int = 4,
    csi: ca.CsiConfig | None = None,
    snapshot_ms: float = 5.0,
    rb_per_rbg: int = 16,
) -> dict[str, Any]:
    """实测 MU 相对 SU 的小区谱效比，供 TTI 主循环使用。

    **主循环里不可能逐 TTI 真做配对**——那要在每个 TTI 上做 SVD 与矩阵求逆，
    十万 TTI 直接跑不完。折中是：在建表阶段用
    :func:`mumimo.su_mu_adaptation` 在若干个快照上真配一遍，
    把 MU/SU 的小区谱效比测出来，主循环按这个比例折算。

    **这是当前最大的简化，必须说清楚。** 它假设 MU 增益在时间上是稳定的，
    而真实的配对增益随用户瞬时信道起伏。返回值里带 ``per_snapshot``，
    比值的离散程度就是这个假设的可信度——波动大就说明不该用一个标量。

    ``csi`` 开启老化时，配对的预编码走**陈旧信道**、评估走当前信道。
    **MU 受老化的打击远重于 SU**：ZF 的全部价值就是把配对用户之间的干扰
    零陷掉，而零陷是按基站以为的信道打的——信道一变，零陷就落空，
    残余干扰直接进分母。SU 只是波束没对准，损失温和得多。
    """
    if num_ues is not None and num_ues < len(h_users):
        groups = group_samples_by_ue(len(h_users), num_ues)
        h_users = [np.asarray(h_users[g[0]]) for g in groups]
        geo_sinr_db = [float(np.nanmean([geo_sinr_db[i] for i in g])) for g in groups]

    aging = csi is not None and csi.enabled
    ratios: list[float] = []
    modes: list[str] = []
    n = min(max_snapshots, max(1, min(np.asarray(h).shape[0] for h in h_users)))
    # **两条路径必须同粒度，否则比的不是老化。** 早先老化侧降到 RBG、
    # 完美侧留在 RB，su_mu_adaptation 内部又按 16 分组，等于两边口径不同，
    # 算出来的"老化损失"里混着粒度差。现在一律留在 RB 粒度，
    # 只把逐 RBG 的滞后展开到逐 RB（一跳本来就覆盖 16 个连续 RB）。
    seq = [[np.asarray(h)[t] for t in range(np.asarray(h).shape[0])] for h in h_users]
    n_rb = seq[0][0].shape[0] if seq and seq[0] else 1
    n_rbg = max(1, int(np.ceil(n_rb / max(1, rb_per_rbg))))
    for t in range(n):
        snaps = [np.asarray(h)[t:t + 1] for h in h_users]
        npow = mu.noise_from_geometric_sinr(snaps[0], geo_sinr_db[0])
        prec = None
        if aging:
            assert csi is not None
            lag_rbg = ca.rbg_lag_snapshots(csi, n_rbg, snapshot_ms=snapshot_ms,
                                           snapshot_index=t, rb_per_rbg=rb_per_rbg)
            lags = np.repeat(lag_rbg, max(1, rb_per_rbg))[:n_rb]
            # su_mu_adaptation 吃 [T,RB,BS,UE] 或 [RB,BS,UE]，补回一个时隙维
            prec = [ca.stale_channel(s, t, lags)[None] for s in seq]
        try:
            dec = mu.su_mu_adaptation(snaps, noise_power=npow,
                                      h_users_for_precoding=prec,
                                      max_mu_users=max_mu_users)
        except Exception:  # noqa: BLE001
            continue
        if dec.su_se > 0:
            ratios.append(dec.mu_se / dec.su_se)
            modes.append(dec.mode)
    if not ratios:
        return {"ratio": 1.0, "measured": False,
                "note": "配对测不出来（用户数或天线数不足），MU 按 1.0 处理"}
    r = float(np.median(ratios))
    spread = float(np.std(ratios) / max(abs(r), _EPS))
    return {
        "ratio": r, "measured": True, "per_snapshot": [round(x, 3) for x in ratios],
        "mode_share_mu": modes.count("MU") / len(modes),
        "relative_spread": round(spread, 3),
        "csi_aging": bool(aging),
        "note": (f"**这是一个标量近似**：在 {len(ratios)} 个快照上真配了一遍取中位数，"
                 f"主循环按它折算，没有逐 TTI 重新配对。"
                 f"比值离散度 {spread * 100:.0f}%——超过 30% 就说明 MU 增益"
                 f"随时间起伏很大，用一个标量会失真。"
                 + ("配对预编码用的是**陈旧 CSI**（已开老化）。" if aging else
                    "配对预编码用的是**零时延完美 CSI**，这是上界不是现网。")),
    }


# ---------------------------------------------------------------------------
# 话务
# ---------------------------------------------------------------------------
@dataclass
class _Burst:
    start_tti: int
    bytes_total: int
    bytes_left: int
    first_tti: int = -1
    last_tti: int = -1
    n_tti: int = 0
    bytes_first: int = 0
    bytes_last: int = 0
    prev_tti: int = -1                   # 倒数第二次被服务的 TTI，掐尾时用
    is_small: bool = False               # bimodal 的小包（只占 1 个 RBG）


class _Traffic:
    """按话务模型往每个 UE 的缓冲区里投 burst。"""

    def __init__(self, cfg: TrafficConfig, n_ue: int, tti_ms: float,
                 rng: np.random.Generator, small_bytes: int = 1500,
                 num_rbg: int = 17) -> None:
        self.cfg, self.n_ue, self.tti_ms, self.rng = cfg, n_ue, tti_ms, rng
        self.active: list[_Burst | None] = [None] * n_ue
        self.queue: list[list[_Burst]] = [[] for _ in range(n_ue)]
        self.done: list[list[_Burst]] = [[] for _ in range(n_ue)]
        self._p_arrive = cfg.arrival_rate_hz * tti_ms / 1000.0
        self.offered_bytes = 0
        # 小包只占 1 个 RBG：按 1/num_rbg 的 RE 数、中等 MCS 估个字节数。
        # 它小到一个 TTI 就发完，所以体验速率完全由调度时延决定。
        self._per_rbg_bytes = max(50, int(small_bytes or 1500))
        self.num_rbg = int(num_rbg)
        self.rbg_hist: list[int] = []
        self._cbr_per_tti = int(cfg.cbr_mbps * 1e6 * tti_ms / 1000.0 / 8)

    def step(self, tti: int) -> None:
        if self.cfg.model == "full_buffer":
            for u in range(self.n_ue):
                if self.active[u] is None:
                    self.active[u] = _Burst(tti, 1 << 62, 1 << 62)
            return
        if self.cfg.model == "cbr":
            for u in range(self.n_ue):
                b = self.active[u]
                if b is None:
                    self.active[u] = _Burst(tti, self._cbr_per_tti, self._cbr_per_tti)
                else:
                    b.bytes_left += self._cbr_per_tti
                    b.bytes_total += self._cbr_per_tti
                self.offered_bytes += self._cbr_per_tti
            return
        # ftp3 / bimodal：泊松到达（每 TTI 用伯努利近似，p 很小时等价）
        for u in range(self.n_ue):
            if self.rng.random() < self._p_arrive:
                if self.cfg.model == "bimodal":
                    n_rbg, small = self.draw_rbg(self.num_rbg)
                    # 一次调度占 n_rbg 个 RBG，burst 大小按它一个 TTI 的承载算
                    n_bytes = max(200, int(self._per_rbg_bytes * n_rbg))
                else:
                    small, n_bytes = False, self.cfg.file_bytes
                b = _Burst(tti, n_bytes, n_bytes, is_small=small)
                self.offered_bytes += n_bytes
                if self.active[u] is None:
                    self.active[u] = b
                else:
                    self.queue[u].append(b)

    def draw_rbg(self, num_rbg: int | None = None) -> tuple[int, bool]:
        """抽一次传输占几个 RBG。两头高中间低。返回 ``(RBG 数, 是不是小包)``。

        **num_rbg 必须跟着配置走。** 早先签名给了默认值 17、调用处又不传，
        于是 ``num_rbg=8`` 的配置照样抽出 1~17 个 RBG——
        实测平均 9.03 个 RBG 却只有 8 个可用，"满带宽占比"也从 0.30 变成 0.586。
        """
        num_rbg = int(num_rbg if num_rbg is not None else self.num_rbg)
        x = self.rng.random()
        if x < self.cfg.p_small_rbg:
            n = 1
        elif x < self.cfg.p_small_rbg + self.cfg.p_full_rbg:
            n = num_rbg
        else:
            n = int(self.rng.integers(2, max(3, num_rbg)))   # 2~16 均匀
        self.rbg_hist.append(n)
        return n, n == 1

    def has_data(self, u: int) -> bool:
        return self.active[u] is not None

    def bytes_left(self, u: int) -> int:
        """当前 burst 还剩多少没发。SU/MU 判决要用它——一个 TTI 能传完就不配对。"""
        b = self.active[u]
        return int(b.bytes_left) if b is not None else 0

    def serve(self, u: int, tti: int, n_bytes: int) -> int:
        """给这个 UE 发 ``n_bytes``，返回实际发出去的字节数。"""
        b = self.active[u]
        if b is None or n_bytes <= 0:
            return 0
        sent = min(n_bytes, b.bytes_left)
        b.bytes_left -= sent
        if b.first_tti < 0:
            b.first_tti, b.bytes_first = tti, sent
        b.prev_tti = b.last_tti
        b.last_tti, b.bytes_last = tti, sent
        b.n_tti += 1
        if b.bytes_left <= 0:
            self.done[u].append(b)
            self.active[u] = self.queue[u].pop(0) if self.queue[u] else None
        return sent


# ---------------------------------------------------------------------------
# KPI
# ---------------------------------------------------------------------------
def _burst_throughput_mbps(b: _Burst, tti_ms: float, cfg: KpiConfig) -> float | None:
    """单个 burst 的体验速率（Mbps）。不合格返回 ``None``。

    3GPP TS 28.552 §5.1.1.3 的口径：**排除清空缓冲区的最后一个 slice**，
    因为那个 TTI 通常只用了一部分就把数据发完了，把它算进去等于用
    "半个 TTI 的时间"去除"半个 TTI 的数据"，得到一个虚高的瞬时速率。
    单 slice 的 burst 因此完全无法测量，只能整个丢掉。

    **分母是一段时间，不是被调度的 TTI 数。** 这两个差得很远——
    用户排队等调度的那些 TTI 也在消耗体验。早先按被调度 TTI 数算，
    12 个用户各报出 583 Mbps、小区合计 8.2 Gbps，对一个 100 MHz 小区
    物理上不可能（峰值约 1.2 Gbps）——每个用户被算成"轮到我就独享整个小区"。

    起点按 ``trim`` 分两种（用户 2026-08-02 明确）：

    * ``none`` / ``tail``：从**数据到达**算起，等调度的时间计入分母
    * ``head_tail``：从**首次被调度的 TTI** 算起，
      **话务到达但还没被调度的等待时间不计入**
    """
    if b.n_tti < max(2, cfg.min_burst_tti) or b.last_tti < 0:
        return None
    vol = b.bytes_total
    # 掐头 = 起点从"到达"挪到"首次被调度"
    t0 = b.first_tti if cfg.trim == "head_tail" else b.start_tti
    n = b.last_tti - t0 + 1
    if cfg.trim in ("tail", "head_tail"):
        # 掐尾：排除清空缓冲区的最后一个 slice，时间与数据同时扣
        vol -= b.bytes_last
        n -= (b.last_tti - b.prev_tti) if b.prev_tti >= 0 else 1
    if n <= 0 or vol <= 0:
        return None
    return vol * 8.0 / (n * tti_ms / 1000.0) / 1e6


@dataclass
class UeKpi:
    ue: int
    geo_sinr_db: float
    iot_db: float
    experienced_mbps: float
    served_mbps: float                   # 端到端平均（含空闲，用于对照）
    bursts: int
    avg_mcs: float
    avg_rank: float
    bler_first_tx: float
    residual_bler: float
    sched_tti: int
    retx_tti: int

    def as_dict(self) -> dict[str, Any]:
        return {k: (round(v, 4) if isinstance(v, float) else v)
                for k, v in self.__dict__.items()}


@dataclass
class SystemResult:
    """一次系统级仿真的全部结果，小区级与用户级都在。"""

    config: dict[str, Any]
    cell: dict[str, Any]
    users: list[dict[str, Any]]
    elapsed_s: float
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {"config": self.config, "cell": self.cell, "users": self.users,
                "elapsed_s": round(self.elapsed_s, 3), "notes": self.notes}

    def text(self) -> str:
        c = self.cell
        return (
            f"小区体验速率 {c['cell_experienced_mbps']:.2f} Mbps"
            f"（用户中位 {c['ue_experienced_median_mbps']:.2f}、"
            f"5% 边缘 {c['ue_experienced_p5_mbps']:.2f}）\n"
            f"平均调度 MCS {c['avg_mcs']:.1f}（5% 边缘 {c['edge_mcs_p5']:.1f}），"
            f"平均 rank {c['avg_rank']:.2f}，首传 BLER {c['bler_first_tx']:.3f}\n"
            f"IoT 中位 {c['iot_db_median']:.1f} dB"
            f"（{c['high_iot_ue_share']:.0%} 的用户 ≥20 dB 属高干扰），"
            f"MU 配对占 RBG {c['mu_rbg_share']:.1%}\n"
            f"调度 {c['scheduled_tti']} 个 TTI / 共 {c['dl_tti']} 个下行 TTI"
            f"（占用率 {c['occupancy']:.1%}），MU 占比 {c['mu_share']:.1%}"
        )


# ---------------------------------------------------------------------------
# 主循环
# ---------------------------------------------------------------------------
def simulate(
    tables: list[UeLinkTable],
    *,
    sys_cfg: SystemConfig | None = None,
    traffic: TrafficConfig | None = None,
    sched: SchedulerConfig | None = None,
    kpi: KpiConfig | None = None,
    mu_se_ratio: float = 1.0,
    rng: rg.RngBook | None = None,
    progress: Any = None,
) -> SystemResult:
    """跑 TTI 主循环。**这里没有任何矩阵运算**，全是查表加算术。

    ``mu_se_ratio`` 是 MU 相对 SU 的小区谱效比（由 :func:`mumimo.su_mu_adaptation`
    在第一相测出来）。>1 时调度器会在有足够用户排队时切到 MU。

    ``rng`` 是 :class:`rng.RngBook`，**按用途分流**：话务到达、HARQ 误码抽样、
    调度器决胜各拿一条互相独立的流。不给的话从 ``sys_cfg.seed`` 构造
    ``RngBook(master_seed=sys_cfg.seed, replication=0)``，老调用方不用改。

    **分流前是一个 rng 同时喂话务和 HARQ**，改一下 ``arrival_rate_hz`` 会让
    HARQ 的伯努利序列整个错位——"话务模型的影响"里于是混着"HARQ 换了一批随机数"。
    这类污染在结果里完全看不出来。分流后改话务只动话务流。
    """
    sys_cfg = sys_cfg or SystemConfig()
    traffic = traffic or TrafficConfig()
    sched = sched or SchedulerConfig()
    kpi = kpi or KpiConfig()
    t0 = time.perf_counter()

    book = rng if rng is not None else rg.RngBook(master_seed=int(sys_cfg.seed))
    rng_traffic = book.generator("traffic")
    rng_harq = book.generator("harq")
    rng_sched = book.generator("scheduler")

    n_ue = len(tables)
    # 小包只占 1 个 RBG：按该 RBG 的 RE 数 × 中等 MCS 谱效估承载
    # 1 个 RBG 一个 TTI 的承载：RB×12 子载波×12 数据符号×中等谱效
    _small_b = max(200, int(sys_cfg.rb_per_rbg * 12 * 12 * 3.0 / 8))
    tr = _Traffic(traffic, n_ue, sys_cfg.tti_ms, rng_traffic, small_bytes=_small_b,
                  num_rbg=sys_cfg.num_rbg)

    n_rb = sys_cfg.num_rbg * sys_cfg.rb_per_rbg
    # 每 TTI 可用 RE：RB × 12 子载波 × 12 个数据符号（扣 DM-RS 与控制开销）
    re_per_tti = n_rb * 12 * 12
    # **S 时隙不是满下行。** 主循环把 D 和 S 一视同仁地当整个下行 TTI 调度，
    # 而 SystemConfig.dl_ratio 报告时又把 S 折成 0.7——同一个量两套口径，
    # 于是"实际调度的下行"比"报告的下行"多。现在按同一个系数折 RE。
    _re_of = {"D": re_per_tti, "S": int(re_per_tti * S_SLOT_DL_FRACTION)}
    snap_every = max(1, int(round(sys_cfg.snapshot_update_ms / sys_cfg.tti_ms)))
    n_snap = tables[0].sinr_db.shape[0]

    r_avg = np.full(n_ue, 1e-6)
    served = np.zeros(n_ue)
    sched_cnt = np.zeros(n_ue, dtype=int)
    retx_cnt = np.zeros(n_ue, dtype=int)
    mcs_sum = np.zeros(n_ue)
    rank_sum = np.zeros(n_ue)
    nack_first = np.zeros(n_ue)
    tx_first = np.zeros(n_ue)
    nack_final = np.zeros(n_ue)
    harq_pending: dict[int, tuple[int, int]] = {}   # ue -> (剩余重传次数, TB bytes)

    dl_tti = 0
    busy_tti = 0
    mu_tti = 0
    outage_tti = 0
    su_fits_skip = 0
    mu_rbg = 0
    olla_db = np.zeros(n_ue)              # 每用户的 OLLA 偏置（dB）
    pattern = sys_cfg.tdd_pattern.upper() or "D"

    from . import linkadapt as la  # noqa: PLC0415

    # **调度器只能用基站自己估的谱效。** 用真实谱效等于让它预知信道，
    # 它会自动绕开 CSI 老化最严重的用户，老化的代价就凭空消失了。
    # 零时延时 best_se_gnb 与 best_se 逐位相同，行为不变。
    # 在循环外解引用一次——这是每 TTI 每用户都要碰的量。
    _sched_se = [t.best_se_gnb if t.best_se_gnb is not None else t.best_se
                 for t in tables]

    for tti in range(sys_cfg.num_tti):
        _slot = pattern[tti % len(pattern)]
        if _slot not in ("D", "S"):
            continue                                   # 上行时隙不调度下行
        re_per_tti = _re_of[_slot]                     # S 时隙的 RE 少三成
        dl_tti += 1
        tr.step(tti)
        snap = (tti // snap_every) % n_snap

        cand = [u for u in range(n_ue) if tr.has_data(u)
                and not (tables[u].outage is not None and tables[u].outage[snap])]
        blocked = sum(1 for u in range(n_ue) if tr.has_data(u)
                      and tables[u].outage is not None and tables[u].outage[snap])
        outage_tti += blocked
        if not cand:
            r_avg *= (1.0 - 1.0 / sched.pf_window_tti)
            continue

        # --- 调度判决 ---
        # 用的是 _sched_se：**基站自己估的**谱效（见主循环外的说明）
        inst_se = np.array([_sched_se[u][snap] for u in cand])
        if sched.algorithm == "pf":
            metric = inst_se / np.maximum(r_avg[cand], 1e-9)
        elif sched.algorithm == "max_ci":
            metric = inst_se
        else:                                          # rr
            metric = np.array([-((tti + u) % n_ue) for u in cand], dtype=float)
        # **决胜（tie-break）要随机，不能按 UE 编号。** 度量打平时 argsort 稳定排序
        # 恒把编号小的排前面，于是同信道、同队列的两个用户里编号小的**系统性**多拿
        # 调度机会——PF 的公平性判据看不出来（它只看 R_avg，而 R_avg 确实被拉平了），
        # 但逐用户 KPI 会带一个与编号相关的偏置。
        # 只在**真有平局**时才抽签：没有平局时 lexsort 与 argsort 结果相同，
        # 但抽签会白白消耗 scheduler 流，也让"没有平局的配置"变得不可复现比对。
        _m = metric.tolist()
        if len(_m) > 1 and len(set(_m)) < len(_m):
            order = np.lexsort((rng_sched.random(len(_m)), -metric))
        else:
            order = np.argsort(-metric)

        # **SU 能一个 TTI 传完就不触发 MU**（用户 2026-08-02 的现场准则）——
        # 数据都发完了，配对没有意义，还白白引入用户间干扰。
        _top = cand[order[0]]
        _su_bytes = int(la.transport_block_size(
            re_per_tti, la.MCS_TABLES[3][int(tables[_top].mcs[
                snap, int(tables[_top].best_rank[snap]) - 1])].rate,
            la.MCS_TABLES[3][int(tables[_top].mcs[
                snap, int(tables[_top].best_rank[snap]) - 1])].q_m,
            layers=int(tables[_top].best_rank[snap])) // 8)
        _fits_in_su = tr.bytes_left(_top) <= _su_bytes
        use_mu = (sched.mu_enabled and mu_se_ratio > 1.0
                  and len(cand) >= 2 and not _fits_in_su)
        if _fits_in_su and len(cand) >= 2:
            su_fits_skip += 1
        picked = [cand[i] for i in order[:sched.max_mu_users]] if use_mu else [cand[order[0]]]
        if use_mu:
            mu_tti += 1
            mu_rbg += sys_cfg.num_rbg          # MU 时整band 都是配对的
        busy_tti += 1

        # --- 发送 ---
        # **MU 是空间复用，不是频率复用。** 配对的每个用户都拿**全带宽**，
        # 靠不同的空间波束区分。早先按 1/K 分 RE，MU 的聚合吞吐就和 SU 一模一样——
        # 等于把空间复用做成了时频复用，MU 增益整个消失。
        n_pair = len(picked)
        for u in picked:
            r = int(tables[u].best_rank[snap])
            if use_mu:
                r = min(r, mu.MU_MAX_RANK)      # MU 每用户硬顶 rank2（工程约束）
            # **发送端按无干扰的 SINR + OLLA 偏置选 MCS**，
            # 接收端按含干扰的 SINR 判误码。两者的差就是 OLLA 要收敛掉的东西。
            if tables[u].sinr_tx_db is not None and sched.olla_enabled:
                _tx = float(tables[u].sinr_tx_db[snap, r - 1]) + olla_db[u]
                m = la.select_mcs(_tx, table=3, target_bler=0.1).index
            else:
                m = int(tables[u].mcs[snap, r - 1])
            mcs_obj = la.MCS_TABLES[3][m]
            tbs_bits = la.transport_block_size(
                re_per_tti, mcs_obj.rate, mcs_obj.q_m, layers=r)
            if use_mu:
                # 配对后每人只分到 1/K 的功率、还要吃残余干扰。
                # mu_se_ratio 是建表阶段用真实 SU/MU 自适应测出来的**聚合**比值，
                # 所以这里除以配对数，使 K 个用户加起来 = ratio x 单用户 SU。
                tbs_bits *= mu_se_ratio / n_pair
            tb_bytes = max(1, int(tbs_bits // 8))

            # HARQ：首传按该 MCS 的 BLER 判 ACK/NACK，失败进重传
            pend = harq_pending.get(u)
            if pend is not None:
                left, size = pend
                # 重传查 ReTx 曲线（合并增益体现在曲线本身更靠左）。
                # 用上一次的 SINR 近似——真软合并要 LLR，本项目明确不做。
                #
                # **查的必须是实发的 MCS，不是"这个 SINR 该用的 MCS"。**
                # 早先写成 tables[u].mcs[snap, r-1]，那是拿真实 SINR 反查出来的
                # **理想档**；而实发的 m 来自发送侧 SINR + OLLA，通常更高
                # （首传之所以失败正是因为点高了，开 CSI 老化后更明显）。
                # 用低档去查 ReTx 曲线 → BLER 偏低 → 重传几乎必然成功 →
                # **残留 BLER 系统性偏低**，而这个偏差不会以任何方式报出来。
                bler = _bler_lookup(m, float(tables[u].sinr_db[snap, r - 1]), "retx")
                retx_cnt[u] += 1
                if rng_harq.random() > bler:
                    # **重传成功也要计入 served。** 早先这里漏了，
                    # 字节进了缓冲区却没进统计，对账差 4.5%。
                    served[u] += tr.serve(u, tti, size)
                    harq_pending.pop(u, None)
                elif left > 1:
                    harq_pending[u] = (left - 1, size)
                else:
                    harq_pending.pop(u, None)
                    nack_final[u] += 1
                sched_cnt[u] += 1
                mcs_sum[u] += m
                rank_sum[u] += r
                continue

            sinr = float(tables[u].sinr_db[snap, r - 1])
            bler = float(la.bler_curve(m, "newtx")["bler_at"](sinr)) \
                if False else _bler_lookup(m, sinr)
            tx_first[u] += 1
            sched_cnt[u] += 1
            mcs_sum[u] += m
            rank_sum[u] += r
            if rng_harq.random() > bler:
                sent = tr.serve(u, tti, tb_bytes)
                served[u] += sent
                if sched.olla_enabled:      # ACK：小步上调
                    olla_db[u] = min(olla_db[u] + sched.step_up,
                                     sched.olla_max_db)
            else:
                nack_first[u] += 1
                harq_pending[u] = (3, tb_bytes)
                if sched.olla_enabled:      # NACK：大步下调
                    olla_db[u] = max(olla_db[u] - sched.step_down,
                                     sched.olla_min_db)

        # --- PF 平均速率更新 ---
        inst = np.zeros(n_ue)
        for u in picked:
            # **PF 的瞬时速率必须和实发口径一致。** MU 下实发被限到 rank≤2，
            # 而 best_se 可能是 rank4 的——记错了会让 PF 以为给足了，
            # 公平性判据整个偏掉。
            if use_mu:
                _r = min(int(tables[u].best_rank[snap]), mu.MU_MAX_RANK)
                inst[u] = tables[u].se[snap, _r - 1] * mu_se_ratio / len(picked)
            else:
                inst[u] = tables[u].best_se[snap]
        a = 1.0 / sched.pf_window_tti
        r_avg = (1.0 - a) * r_avg + a * inst
        if progress and tti % 5000 == 0:
            progress(tti, sys_cfg.num_tti)

    # --- KPI 汇总 ---
    offered_bytes = tr.offered_bytes
    users: list[UeKpi] = []
    small_thp: list[float] = []
    large_thp: list[float] = []
    for u in range(n_ue):
        _done = [b for b in tr.done[u] if b.start_tti >= kpi.warmup_tti]
        thps = [x for x in (_burst_throughput_mbps(b, sys_cfg.tti_ms, kpi)
                            for b in _done) if x is not None]
        # **小包和大包要分开报。** 小包的体验速率被调度时延主导、大包才反映
        # 信道能力，混在一起平均会得到一个谁都不像的数。
        _sm = [x for b in _done if b.is_small
               for x in [_burst_throughput_mbps(b, sys_cfg.tti_ms, kpi)] if x is not None]
        _lg = [x for b in _done if not b.is_small
               for x in [_burst_throughput_mbps(b, sys_cfg.tti_ms, kpi)] if x is not None]
        small_thp.append(float(np.mean(_sm)) if _sm else float("nan"))
        large_thp.append(float(np.mean(_lg)) if _lg else float("nan"))
        users.append(UeKpi(
            ue=u, geo_sinr_db=tables[u].geo_sinr_db, iot_db=tables[u].iot_db,
            experienced_mbps=float(np.mean(thps)) if thps else 0.0,
            served_mbps=served[u] * 8 / max(sys_cfg.duration_s, _EPS) / 1e6,
            bursts=len(thps),
            avg_mcs=float(mcs_sum[u] / max(sched_cnt[u], 1)),
            avg_rank=float(rank_sum[u] / max(sched_cnt[u], 1)),
            bler_first_tx=float(nack_first[u] / max(tx_first[u], 1)),
            residual_bler=float(nack_final[u] / max(tx_first[u], 1)),
            sched_tti=int(sched_cnt[u]), retx_tti=int(retx_cnt[u]),
        ))

    exp = np.array([x.experienced_mbps for x in users if x.bursts > 0])
    cell = {
        # **小区体验速率是各用户体验速率的平均，不是求和。** 用户是时分复用的，
        # 求和会得到"每个用户都独享整个小区"的假数——实测过一次 8.2 Gbps
        # 落在 100 MHz 小区上，物理峰值只有约 1.2 Gbps。
        "cell_experienced_mbps": float(np.mean(exp)) if exp.size else 0.0,
        "ue_experienced_mean_mbps": float(np.mean(exp)) if exp.size else 0.0,
        "ue_experienced_median_mbps": float(np.median(exp)) if exp.size else 0.0,
        "ue_experienced_p5_mbps": float(np.percentile(exp, 5)) if exp.size else 0.0,
        "cell_served_mbps": float(np.sum([x.served_mbps for x in users])),
        "avg_mcs": float(np.sum(mcs_sum) / max(np.sum(sched_cnt), 1)),
        "avg_rank": float(np.sum(rank_sum) / max(np.sum(sched_cnt), 1)),
        "bler_first_tx": float(np.sum(nack_first) / max(np.sum(tx_first), 1)),
        "residual_bler": float(np.sum(nack_final) / max(np.sum(tx_first), 1)),
        "dl_tti": dl_tti, "scheduled_tti": busy_tti,
        "occupancy": busy_tti / max(dl_tti, 1),
        "mu_share": mu_tti / max(busy_tti, 1),
        "measured_bursts": int(np.sum([x.bursts for x in users])),
        # bimodal 下小包与大包分开报：前者由调度时延主导，后者反映信道能力
        "rbg_size_hist": (
            {"p_1rbg": round(float(np.mean(np.array(tr.rbg_hist) == 1)), 3),
             "p_full": round(float(np.mean(np.array(tr.rbg_hist) >= sys_cfg.num_rbg)), 3),
             "mean_rbg": round(float(np.mean(tr.rbg_hist)), 2),
             "n": len(tr.rbg_hist)} if tr.rbg_hist else None),
        "small_pkt_experienced_mbps": (float(np.nanmean(small_thp))
                                       if np.any(np.isfinite(small_thp)) else None),
        "large_pkt_experienced_mbps": (float(np.nanmean(large_thp))
                                       if np.any(np.isfinite(large_thp)) else None),
        "outage_ue": int(sum(1 for t in tables
                             if t.outage is not None and t.outage.all())),
        "outage_skips": int(outage_tti),
        # **OLLA 收敛到多少，就说明发送端把干扰低估了多少。**
        # 它应当与 IoT 同向：干扰越大、偏置越负。
        "olla_db_mean": float(np.mean(olla_db)),
        "olla_db_p5": float(np.percentile(olla_db, 5)),
        "olla_db_p95": float(np.percentile(olla_db, 95)),
        "olla_target_bler": round(sched.olla_step_up_db
                                  / (sched.olla_step_up_db + sched.olla_step_down_db), 4),
        # **MU 配对比例**：MU 配对的 RBG 数占已调度 RBG 总数。
        # 现场经验值：30%~50% PRB 利用率下大约 5%~20%。
        "mu_rbg_share": mu_rbg / max(busy_tti * sys_cfg.num_rbg, 1),
        "su_fits_skips": int(su_fits_skip),
        # **IoT = (I+N)/N**：干扰主导还是噪声主导。密集城区常 >20 dB。
        "iot_db_median": _nan_safe(np.nanmedian, [t.iot_db for t in tables]),
        "iot_db_p5": _nan_safe(np.nanpercentile, [t.iot_db for t in tables], 5),
        "iot_db_p95": _nan_safe(np.nanpercentile, [t.iot_db for t in tables], 95),
        "iot_sample_valid_share": float(np.mean(
            [t.iot_sample_valid for t in tables])) if tables else 0.0,
        "iot_valid_ue_share": float(np.mean(
            [bool(np.isfinite(t.iot_db)) for t in tables])),
        "high_iot_ue_share": float(np.mean([
            (t.iot_db >= 20.0) if np.isfinite(t.iot_db) else False for t in tables])),
        # **边缘用户 MCS**：现场经验通常 < 5。它比平均 MCS 更能暴露覆盖问题。
        "edge_mcs_p5": _nan_safe(np.nanpercentile,
                                 [x.avg_mcs for x in users if x.sched_tti > 0], 5),
        # 守恒对账：到达了多少、发完了多少、还压着多少。
        # 不报这三个的话，"实际吞吐 105 Mbps vs 话务负载 144 Mbps"
        # 这种缺口只能靠猜——它可能是队列积压（正常），也可能是漏数据（bug）。
        "offered_mbps": round(offered_bytes * 8 / max(sys_cfg.duration_s, _EPS) / 1e6, 3),
        "completed_bursts": int(sum(len(x) for x in tr.done)),
        "backlog_bursts": int(sum(1 for x in tr.active if x is not None)
                              + sum(len(q) for q in tr.queue)),
        "backlog_bytes": int(sum((x.bytes_left for x in tr.active if x is not None), 0)
                             + sum(b.bytes_left for q in tr.queue for b in q)),
    }
    _acct = cell["cell_served_mbps"] * sys_cfg.duration_s * 1e6 / 8 + cell["backlog_bytes"]
    cell["accounting_error_pct"] = round(
        abs(_acct - offered_bytes) / max(offered_bytes, 1) * 100, 3)
    notes: list[str] = []
    if n_snap < 4:
        notes.append(f"**信道快照只有 {n_snap} 个**，时间起伏被严重低估，"
                     "PF 的多用户分集增益拿不到——生成时把 num_slots_per_sample 调大。")
    if cell["measured_bursts"] < 20:
        notes.append(f"只有 {cell['measured_bursts']} 个 burst 进入体验速率统计，"
                     "样本太少。**加长 duration_s 或提高到达率**。")
    if cell["backlog_bytes"] > 0.15 * max(offered_bytes, 1):
        notes.append(
            f"**队列积压 {cell['backlog_bytes']*8/1e6:.1f} Mb**"
            f"（占到达量 {cell['backlog_bytes']/max(offered_bytes,1):.0%}）——"
            "系统在这个负载下没有收敛，体验速率被排队时间拖低。"
            "降低 arrival_rate_hz 或加长 duration_s 再看。")
    if cell["accounting_error_pct"] > 1.0:
        notes.append(f"**字节对不上账（差 {cell['accounting_error_pct']}%）**——"
                     "发出去的 + 还压着的 应该等于到达的。这是 bug 不是现象。")
    if np.isfinite(cell["edge_mcs_p5"]) and cell["edge_mcs_p5"] > 8:
        notes.append(
            f"**5% 边缘用户的 MCS 是 {cell['edge_mcs_p5']:.1f}，偏高**"
            "（现场经验通常 <5）。多半是撒点没覆盖到真正的边缘，"
            "或者邻区负载设得太低、干扰被低估了。")
    # **p_idle_tti 是对标锚点，不是仿真输入。** 它只进解析式 expected_prb_util，
    # 不生成任何空闲 TTI——真实的空闲来自"没人有数据"。两者差太多说明
    # 到达率没调到位，得说出来，否则用户会以为设了 30% 就真是 30%。
    if traffic.model == "bimodal":
        _want_idle = float(traffic.p_idle_tti)
        _got_idle = 1.0 - float(cell["occupancy"])
        if abs(_got_idle - _want_idle) > 0.10:
            notes.append(
                f"**空闲 TTI 实测 {_got_idle:.0%}，而 p_idle_tti 设的是 "
                f"{_want_idle:.0%}。** p_idle_tti **不驱动仿真**——它只是个对标锚点，"
                f"真实的空闲 TTI 由到达率与信道决定。要对齐现网就调 "
                f"arrival_rate_hz，改 p_idle_tti 只会改报告里的 expected_prb_util，"
                f"不会改任何实际行为。")
    _tgt = cell["olla_target_bler"]
    if sched.olla_enabled and cell["bler_first_tx"] > _tgt * 1.6:
        notes.append(
            f"**首传 BLER {cell['bler_first_tx']:.3f} 明显高于 OLLA 的稳态目标 "
            f"{_tgt:.3f}，说明外环还没收敛完。** 现网基线步长 +0.01/−0.1 很慢，"
            f"每次 NACK 只压 0.1 dB，而 MCS 是整数档、小步长常常压不动一档。"
            "要看稳态结论就加长 duration_s；要快收敛就临时把步长调大"
            "（比例不变则稳态 BLER 不变）。")
    # **判据必须是逐样本有效率。** 逐用户的那个恒等于 1（nanmedian 会把
    # 半数 nan 的用户也算成有效），于是这条正确的告警从不触发，
    # 反而触发下面那条"检查站间距"——把用户支使去查一个根本没问题的配置。
    _iot_ok = cell.get("iot_sample_valid_share", 1.0)
    if _iot_ok < 0.9:
        notes.append(
            f"**IoT 不可信：只有 {_iot_ok:.0%} 的样本算得出来**"
            f"（逐用户口径会报 {cell['iot_valid_ue_share']:.0%}，那个数会骗人）。"
            "根因是生成时 num_slots_per_sample > 1——那时 sinr_dB 是各 slot 的"
            "dB 均值、sir_dB 只取最后一个 slot，两者不同口径，"
            "会出现 SIR < SINR 这种物理上不可能的值。"
            "**别去查站间距和邻区负载，配置没问题，是这个量本身在多时隙下不成立。**"
            "要看 IoT 就用 num_slots_per_sample=1 单独生成一批"
            "——但那批做不了系统级仿真（PF 拿不到时间分集、CSI 老化恒为 0），"
            "**这两个需求当前无法在同一个数据集上同时满足**。")
    elif np.isfinite(cell["iot_db_median"]) and cell["iot_db_median"] < 3:
        notes.append(
            f"**IoT 中位只有 {cell['iot_db_median']:.1f} dB**，几乎是噪声受限。"
            "密集城区实际常在 20 dB 以上——检查是不是站间距太大、"
            "或者邻区负载 prb_utilization 设得过低。")
    if traffic.model == "bimodal":
        _u = traffic.expected_prb_util(sys_cfg.num_rbg)
        if abs(_u - 0.30) > 0.05:
            notes.append(
                f"**这套 RBG 尺寸分布折合出来的 PRB 利用率是 {_u:.1%}，"
                f"现网口径约 30%**。差在中间段——2~{sys_cfg.num_rbg - 1} 个 RBG "
                f"均匀分布的均值是 {(2 + sys_cfg.num_rbg - 1) / 2 / sys_cfg.num_rbg:.2f}，"
                "偏高，把中间段改成偏小的分布才能对齐。"
                "**别指望调 p_idle_tti**——它不驱动仿真，只改这个报告数字，"
                "真实的空闲 TTI 由 arrival_rate_hz 决定。"
                "**这个参数我没有替你调，因为它直接决定负载**。")
    if traffic.model == "bimodal" and cell["small_pkt_experienced_mbps"] is None:
        notes.append(
            "**小包的体验速率测不出来**：它们在一个 TTI 内就发完了，"
            "而 3GPP TS 28.552 的掐尾口径要排除清空缓冲区的那个 slice，"
            "单 slice 的 burst 因此没有可测量的时间。"
            "**这不是 bug，是这个 KPI 的固有盲区**——现网话统里小包同样测不到，"
            "它们的体验由调度时延而非速率决定。要看小包体验请看调度时延分布。")
    if cell["outage_ue"]:
        notes.append(
            f"**{cell['outage_ue']} 个用户全程处于覆盖外**（用户级 SINR 够不到 MCS 0 的门限），"
            "已从调度中剔除。他们不进 BLER 与体验速率统计——"
            "但这本身就是个结论：这些点位需要补站或降配。")
    if cell["occupancy"] > 0.98:
        notes.append("**下行时隙几乎占满**，系统已过载——此时体验速率反映的是"
                     "容量上限而不是用户体验，降低到达率再测。")
    return SystemResult(
        config={"system": sys_cfg.as_dict(), "traffic": traffic.as_dict(),
                "scheduler": sched.as_dict(), "kpi": kpi.as_dict(),
                "mu_se_ratio": round(float(mu_se_ratio), 4),
                "rng": book.as_dict()},
        cell=cell, users=[x.as_dict() for x in users],
        elapsed_s=time.perf_counter() - t0, notes=notes,
    )


# ---------------------------------------------------------------------------
# 多次重复：所有 KPI 带置信区间
# ---------------------------------------------------------------------------
@dataclass
class ReplicationResult:
    """n 次重复的汇总。**每个 KPI 都是 mean / std / ci95 / n_rep，不是一个裸数。**"""

    runs: list[SystemResult]
    books: list[rg.RngBook]
    cell: dict[str, dict[str, Any]]
    users: list[dict[str, Any]]
    config: dict[str, Any]
    elapsed_s: float
    build_elapsed_s: float = 0.0
    notes: list[str] = field(default_factory=list)

    @property
    def n_rep(self) -> int:
        return len(self.runs)

    def stat(self, key: str) -> rg.KpiStat:
        """某个小区级 KPI 在各次重复上的分布，拿去做 A/B 用。"""
        return rg.summarize([r.cell[key] for r in self.runs if key in r.cell], key)

    def as_dict(self) -> dict[str, Any]:
        return {"config": self.config, "cell": self.cell, "users": self.users,
                "n_rep": self.n_rep,
                "replications": [b.as_dict()["replication"] for b in self.books],
                "elapsed_s": round(self.elapsed_s, 3),
                "build_tables_s": round(self.build_elapsed_s, 3),
                "notes": self.notes}

    def text(self) -> str:
        def _s(k: str) -> str:
            d = self.cell.get(k) or {}
            m, lo, hi = d.get("mean"), *(d.get("ci95") or [None, None])
            if m is None:
                return "n/a"
            return f"{m:.2f}" if lo is None else f"{m:.2f} [{lo:.2f}, {hi:.2f}]"
        head = (f"（{self.n_rep} 次重复，方括号内是 95% 置信区间）"
                if self.n_rep > 1 else
                "（**只跑了 1 次，下面所有数字都没有置信区间**，"
                "不能用来做比较）")
        return (
            f"{head}\n"
            f"小区体验速率 {_s('cell_experienced_mbps')} Mbps"
            f"（5% 边缘 {_s('ue_experienced_p5_mbps')}）\n"
            f"平均调度 MCS {_s('avg_mcs')}，平均 rank {_s('avg_rank')}，"
            f"首传 BLER {_s('bler_first_tx')}"
        )


def simulate_replications(
    tables: list[UeLinkTable],
    *,
    num_replications: int = 8,
    master_seed: int = 0,
    sys_cfg: SystemConfig | None = None,
    traffic: TrafficConfig | None = None,
    sched: SchedulerConfig | None = None,
    kpi: KpiConfig | None = None,
    mu_se_ratio: float = 1.0,
    build_elapsed_s: float = 0.0,
    progress: Any = None,
) -> ReplicationResult:
    """跑 n 次独立重复，所有 KPI 报 ``mean / std / ci95 / n_rep``。

    **关键优化：只重跑 TTI 主循环，不重建链路表。** :func:`build_link_tables`
    与随机种子完全无关（它只做 SVD、码本搜索、MCS 查表），所以建一次表就够了。
    多跑 n 次的代价因此是 ``(n−1)·T_loop / (T_build + T_loop)``——
    实测 ds_6e9715bc 上建表 5.14 s、单次主循环 0.99 s（交错 3 轮取中位，
    建表轮间波动 11.3%、主循环 3.2%），n=8 是 13.0 s vs 单次 6.1 s，**多 113%**。
    建表越贵这个比例越低：按 10.5 s / 1.1 s 算是 +66%。

    重复实验换的是 ``replication``（对应 ns-3 的 ``RngRun``）而不是 ``master_seed``
    （对应 ``RngSeed``），理由见 :mod:`rng` 的模块文档。

    **这个置信区间覆盖什么、不覆盖什么，必须说清楚。** 各次重复共用同一批信道
    与同一张链路表，所以区间反映的是**话务到达、HARQ 误码、调度决胜**这三条流。
    邻区负载抖动在建表阶段就定死了，**它不进区间**。

    这个取舍是量过的，不是拍的（``measurements/rng_replication.json``）：
    64 次 replication（表固定）与 32 次 master seed 扫描（每次重建表、
    负载抖动重抽）的变异系数对照——

    ===========================  ==================  ==================
    KPI                          replication (n=64)  master seed (n=32)
    ===========================  ==================  ==================
    ``cell_experienced_mbps``    9.40% [8.0, 11.4]   5.93% [4.8, 7.9]
    ``ue_experienced_p5_mbps``   18.62% [15.9, 22.6] 18.93% [15.2, 25.2]
    ``avg_mcs``                  8.14% [6.9, 9.9]    10.46% [8.4, 13.9]
    ``avg_rank``                 2.86% [2.4, 3.5]    2.51% [2.0, 3.3]
    ``bler_first_tx``            8.84% [7.5, 10.7]   10.59% [8.5, 14.1]
    ===========================  ==================  ==================

    方括号是**变异系数自身**的 95% 区间（χ²）。五个 KPI 里四个两列的区间重叠，
    也就是说**冻结链路表并没有可分辨地把离散度报小**——系统级的主导方差就是
    话务与 HARQ，正好是区间覆盖的那几条流。

    顺带一个必须记住的量级：n=8 时变异系数自身的 95% 区间是 ``0.66×~2.04×``。
    ``measurements/seed_variance.json`` 里那个 11.4% 是 8 个种子测的，
    真值可能在 7.5%~23% 之间——**那张表上的 CoV 只精确到大约 2 倍**，
    不要拿它去做精细比较。

    信道实现本身的不确定度是**另一个、更大的方差分量**，要覆盖它得用不同 seed
    重新生成数据集，本函数不做也做不到。
    """
    n = int(num_replications)
    if n < 1:
        raise ValueError(f"重复次数至少 1 次，收到 {n}")
    t0 = time.perf_counter()
    books = rg.replications(int(master_seed), n)
    runs: list[SystemResult] = []
    for i, bk in enumerate(books):
        runs.append(simulate(tables, sys_cfg=sys_cfg, traffic=traffic, sched=sched,
                             kpi=kpi, mu_se_ratio=mu_se_ratio, rng=bk))
        if progress:
            progress(i + 1, n)

    cell = rg.summarize_runs([r.cell for r in runs])
    # 用户级也带区间：只挑真正有意义的两个量，避免把 users 撑成一堵墙
    users: list[dict[str, Any]] = []
    for u in range(len(runs[0].users)):
        row: dict[str, Any] = {"ue": u}
        for k in ("geo_sinr_db", "iot_db"):
            row[k] = runs[0].users[u].get(k)
        for k in ("experienced_mbps", "avg_mcs", "bler_first_tx"):
            row[k] = rg.summarize([r.users[u][k] for r in runs], k).as_dict()
        users.append(row)

    # notes 去重但保序。**按原文去重是不够的**：像"首传 BLER 0.287 高于目标"
    # 这种 note 把逐次重复的数值嵌在文本里，8 次重复就是 8 条只差几个数字的
    # 告警，把真正不同的那几条淹掉。所以去重键是**抹掉数字后的模板**，
    # 保留第一条原文并标注命中了几次——数字不同这件事本身不是新信息。
    import re  # noqa: PLC0415

    _seen: dict[str, int] = {}
    _order: list[tuple[str, str]] = []
    for r in runs:
        for s in r.notes:
            k = re.sub(r"[0-9]+(?:\.[0-9]+)?", "#", s)
            if k not in _seen:
                _seen[k] = 0
                _order.append((k, s))
            _seen[k] += 1
    notes = [(txt if _seen[k] <= 1 else
              f"{txt}（{_seen[k]}/{n} 次重复都触发；上面的数值取自第 1 次）")
             for k, txt in _order]
    warn = rg.min_replications_note(n)
    if warn:
        notes.insert(0, warn)
    # 相对区间最宽的那个 KPI 值得单独点名——它决定了这组数字能说到多细。
    # **只在头条 KPI 里挑**：backlog_bytes 这类均值贴近 0 的量相对半宽动辄
    # 几百个百分点（实测 140%），点名它只会把注意力引到一个没人要下结论的字段上。
    _HEADLINE = ("cell_experienced_mbps", "ue_experienced_p5_mbps",
                 "ue_experienced_median_mbps", "cell_served_mbps",
                 "avg_mcs", "avg_rank", "bler_first_tx")
    _worst = max(
        ((k, v) for k, v in cell.items()
         if k in _HEADLINE and v.get("rel_half_width") is not None and v.get("mean")),
        key=lambda kv: kv[1]["rel_half_width"], default=None)
    if _worst and _worst[1]["rel_half_width"] > 0.05:
        notes.append(
            f"**头条 KPI 里 {_worst[0]} 的 95% 置信区间最宽，半宽是均值的 "
            f"{_worst[1]['rel_half_width']:.1%}**（n_rep={n}）——"
            f"比这更小的差异，这次实验分辨不出来。要下更细的结论就加 num_replications，"
            f"区间按 1/√n 收窄（注意还带 t 修正，收得比 1/√n 更快一些）。")

    cfg = dict(runs[0].config)
    cfg["rng"] = {**runs[0].config["rng"], "replication": f"0..{n - 1}",
                  "num_replications": n}
    return ReplicationResult(
        runs=runs, books=books, cell=cell, users=users, config=cfg,
        elapsed_s=time.perf_counter() - t0, build_elapsed_s=float(build_elapsed_s),
        notes=notes,
    )


_BLER_CACHE: dict[tuple[int, int, str], float] = {}


def _bler_lookup(mcs: int, sinr_db: float, tx_mode: str = "newtx") -> float:
    """查表 BLER，按 0.5 dB 量化后缓存——主循环里会被叫十万次。

    量化到 0.5 dB 是有意的：BLER 曲线在门限附近很陡，但 0.5 dB 的分辨率
    足够（一档 MCS 的间隔约 1~2 dB），而缓存命中率因此接近 100%。
    """
    # **nan 要在这里兜住。** int(round(nan*2)) 直接 ValueError，
    # 而 nan SINR 是能真到这儿的（被拒样本、全零信道、几何 SINR 缺失）。
    # 一个用户的一个快照能把整条系统级仿真挂掉，报的错还看不出是谁。
    if sinr_db != sinr_db:                      # nan
        return 1.0                              # 发不出去
    key = (int(mcs), int(round(min(max(sinr_db, -60.0), 60.0) * 2)), tx_mode)
    v = _BLER_CACHE.get(key)
    if v is None:
        from . import bler_curves as bc  # noqa: PLC0415

        try:
            v = float(np.atleast_1d(
                bc.get_curve(int(mcs), tx_mode).evaluate(key[1] / 2.0))[0])
        except Exception:  # noqa: BLE001
            v = 0.1
        _BLER_CACHE[key] = float(min(max(v, 0.0), 1.0))
    return _BLER_CACHE[key]
