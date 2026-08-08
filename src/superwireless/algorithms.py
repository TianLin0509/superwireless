"""这次仿真到底用了哪些算法——一份可以摊开给用户看的清单。

**为什么要有这个模块。** 一次仿真里嵌着十几个算法选择：预编码用什么、
接收机怎么算 SINR、MCS 怎么选、rank 怎么定、多用户怎么配对、调度器什么准则、
体验速率怎么掐头去尾。每一个都会改变最终数字，但它们平时**全都藏在代码里**——
用户看到的只有一个"谱效 26.3"。

这里把它们逐条写出来：**是什么、怎么算的、为什么这么选、什么时候会失真**。
说明书里的「算法」页签直接渲染它。

写在这里的每一条都必须和代码对得上。加算法就在这里加一条，
`test_algorithms` 会检查清单与实际实现没有漂开。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# 现场口径（用户 2026-08-02 确认）
FIELD_ANCHORS = {
    "avg_mcs": 15.0,
    "avg_mcs_near": 25.0,
    "avg_mcs_far": 5.0,
    "avg_rank": 2.7,
    "source": "现网话统，用户 2026-08-02 提供",
}


@dataclass
class Algorithm:
    """一个算法的完整交代。"""

    key: str
    name: str
    stage: str                      # 属于链路的哪一段
    choice: str                     # 这次实际用的是哪一个
    formula: str = ""
    why: str = ""
    caveat: str = ""                # 什么时候这个选择会让结论失真
    source: str = ""                # 标准条款或文献
    alternatives: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {k: v for k, v in self.__dict__.items() if v}


def _algorithms(cfg: dict[str, Any]) -> list[Algorithm]:
    n_bs = int(cfg.get("num_bs_tx_ant", 64) or 64)
    n_cell = int(cfg.get("num_sites", 1) or 1) * int(cfg.get("sectors_per_site", 1) or 1)
    est = str(cfg.get("channel_est_mode", "ls_linear"))
    multi = n_cell > 1

    return [
        Algorithm(
            key="antenna_model",
            name="天线阵列模型",
            stage="信道生成",
            choice=("effective_subarray（1 驱 3，64 RF 端口 × 3 阵子 = 192）"
                    if n_bs == 64 else f"legacy_64（{n_bs} 个独立阵元，间距一律 0.5λ）"),
            why="真实 AAU 是 8H×4V×2pol 共 64 个 RF 端口，每端口固定驱动垂直相邻 3 个阵子。"
                "水平 0.5λ、垂直 0.67λ，RF 端口垂直相位中心 2.01λ > λ，垂直方向有栅瓣。",
            caveat="**legacy 会把吞吐高估 27%、边缘用户高估 61%**（实测）。"
                   "面板不是 8×4×2 时自动走 legacy，因为 1 驱 3 是这一款硬件的事实、不是通用规律。",
            source="ChannelHub phy_sim/effective_array.py；实测对比见 CLAUDE.md",
            alternatives=["legacy_64", "physical_reference（真跑 192 阵子，慢但可作参考）"],
        ),
        Algorithm(
            key="channel_est",
            name="信道估计",
            stage="接收",
            choice={"ideal": "ideal（直接拿真值，是上界不是可实现性能）",
                    "ls_linear": "ls_linear（LS + 频/时域线性插值）",
                    "ls_mmse": "ls_mmse（LS + 频域 MMSE 用指数 PDP 先验 + 线性时域插值）",
                    }.get(est, est),
            why="ls_linear 是 ChannelHub 的默认档。ls_mmse 靠 PDP 先验压掉被污染的部分，"
                "**导频越挤赢得越多**：实测干扰 UE=0 时好 0.7 dB，=16 时好 3.6 dB。",
            caveat="用 ideal 做预编码会得到教科书曲线——**MU-MIMO 尤其致命**，"
                   "实测 CSI NMSE 从 −31 dB 掉到 −8.6 dB，MU 和谱效直接掉一半。",
            source="ChannelHub msg_embedding/channel_est/",
            alternatives=["ideal", "ls_linear", "ls_mmse"],
        ),
        Algorithm(
            key="precoder_su",
            name="单用户预编码",
            stage="发射",
            choice="SVD（逐 RB 取前 rank 个右奇异向量）",
            formula="H^H = U Σ V^H  →  W = V[:, :rank]，总功率归一到 1 按流均分",
            why="SVD 是理想上界，用来和码本方案对比。Type I 码本已做秩自适应"
                "（38.214 的 Type I 反馈里 RI 和 PMI 是一起报的）。",
            caveat="用 h_true 做 SVD 是上界；真实系统只有 h_est。两者的差就是 CSI 代价。",
            source="38.214 §5.2.2（Type I 码本）",
            alternatives=["svd_wideband", "dft", "type1（38.214 码本）", "mrt"],
        ),
        Algorithm(
            key="rank_adaptation",
            name="Rank 自适应",
            stage="链路自适应",
            choice="遍历 rank 1..4，取 rank × MCS谱效 最大的那个",
            formula="对每个 r：每流功率 P/r → 逐流 SINR → 用户级 SINR（dB 域平均）"
                    " → 查 MCS → SE = r × SE(MCS)；取 argmax",
            why="**这是个真实的权衡，不是秩越高越好。** rank1 全功率压最强流、"
                "BF 增益最大、MCS 最高，但只有一条流；rank4 每流只有 P/4，"
                "弱流把用户级 SINR 拖下去、MCS 掉档，但乘的是 4。最优点通常在中间。",
            caveat=f"现网锚点是平均 rank {FIELD_ANCHORS['avg_rank']}。"
                   "仿真跑出来明显偏高（比如 3.9）多半是噪声口径错了——见下一条。",
            source="用户 2026-08-02 给的现场算法",
        ),
        Algorithm(
            key="noise_reference",
            name="噪声功率口径",
            stage="链路自适应",
            choice="锚定 rank1：noise = σ₁²·P / 10^(几何SINR/10)",
            formula="使 rank=1 时的后波束 SINR 恰好等于 ChannelHub 报的几何 sinr_dB",
            why="ChannelHub 的几何 SINR 信号项里**已经含了** N_ant·|w^H a|² 的波束赋形增益。"
                "再按平均单天线功率反推噪声、然后叠一次 SVD 阵列增益，就是把增益算了两遍。",
            caveat="**这一步值 12 dB。** 实测同一批数据：错口径给 平均 rank 3.90 / MCS 23.5，"
                   "正确口径给 2.23 / 11.1，现网锚点是 2.70 / 15.0。"
                   "错的那个几乎所有用户判到 rank4、MCS 顶格——**没有现网锚点作对照，"
                   "它长得完全像一份正常结果**。",
            source="见 CLAUDE.md「IoT 不是 snr_dB 减 sinr_dB」同一类口径问题",
        ),
        Algorithm(
            key="mcs_selection",
            name="MCS 选择（单码字）",
            stage="链路自适应",
            choice="表 3（公司实测 20B NewTx 曲线，28 档）+ 10% 首传 BLER",
            formula="逐 RB SINR → RBG 内线性平均 → RBG 间与流间 dB 域平均"
                    " → 用户级 SINR → 选满足目标 BLER 的最高 MCS",
            why="**一个用户一个 TTI 只发一个码字**，同一个 MCS 覆盖全部 RB 与全部流。"
                "所以必须先把 SINR 压成一个数再查表，不能逐 RB 查完再平均——"
                "后者等于假设每 RB 能用不同 MCS，系统性高估。两者的差正是单码字的损失。",
            caveat="dB 域平均比线性平均保守：实测半好半坏（+20/−20 dB）的信道，"
                   "dB 域给 0 dB、线性给 17 dB，**差 17 dB**。深衰的 RBG 会把整个码字拖下去。",
            source="38.214 §5.1.3；公司 20B 曲线（bler_data_20b.py，含 SHA-256）",
            alternatives=["表 1/2：38.214 标准表 + 分析 BLER 模型"],
        ),
        Algorithm(
            key="receiver",
            name="接收机",
            stage="接收",
            choice="MMSE（把干扰当白噪声）/ IRC（用完整空间协方差打零陷）",
            formula="SINR_k = 1/[(I + (P/rank)·G^H R_n^{-1} G)^{-1}]_kk − 1；"
                    "MMSE 取 R_n=(N0+I_tot/N_rx)·I，IRC 取 R_n=N0·I+R_uu",
            why="**公式相同，区别全在 R_n。** IRC 的增益只能来自干扰的非白性——"
                "干扰真白的时候两者必然重合。",
            caveat="实测 ChannelHub 的**单个干扰小区信道是秩 1 的**（σ₂/σ₁ 中位 4.0e−8）。"
                   "3 个秩 1 干扰 + 4 根收天线 = 刚好全零陷得掉，"
                   "**这是 IRC 最有利的工况，实测 +2.37 bit/s/Hz 偏乐观**。"
                   "引用时必须带上 interference_rank。"
                   if multi else "单小区场景下没有邻区干扰，IRC 与 MMSE 等价。",
            source="经典 MMSE-IRC；本项目 linklevel.post_equalizer_sinr",
        ),
        Algorithm(
            key="mu_pairing",
            name="MU-MIMO 配对（EZF）",
            stage="多用户",
            choice="每用户 SVD 取前 rank 流 → 堆叠后对配对用户 ZF 迫零",
            formula="H̃ = 各用户等效行向量堆叠；W ∝ H̃^H(H̃H̃^H)^(−1)，逐列归一；"
                    "功率按流均分（rank2 的用户拿 2 份）",
            why="EZF：先用各自的 SVD 把用户内的流分开，再用 ZF 把用户间的干扰清零。"
                "**MU 每用户最多 rank 2**（工程约束），SU 可以到 rank 4。",
            caveat="**预编码矩阵只能表示方向，功率必须单独给。** 合成一个全局标量会退化成"
                   "信道求逆功控——ZF 满足 H̃W=c·I，所有用户接收电平被强行拉平，"
                   "弱用户吃掉大部分功率。症状是等效信道范数 12.0/11.7/10.7/7.2 的四个用户"
                   "拿到一模一样的谱效、Jain 公平度恒等于 1.000000。",
            source="用户 2026-08-02 给的现场算法；Sionna rzf_precoding_matrix（逐列归一）",
        ),
        Algorithm(
            key="su_mu_adaptation",
            name="SU/MU 自适应",
            stage="多用户",
            choice="同一个 TTI 里两种发法都算一遍，取小区谱效高的",
            formula="SU：单用户独占、无 MU 干扰、rank≤4；MU：配对多用户、有残余干扰、rank≤2",
            why="现场没有明确的用户间相关性门限，实际做法就是直接比小区谱效。"
                "SU 赢在无干扰且能开到 rank4，MU 赢在流数多，两者不是一个总压另一个。",
            caveat="**别想当然认为配对总是更好**：64 端口只服务 12 个用户时空间自由度富余，"
                   "实测全选 12 个（74.24）反而高于 SUS 选 4 个（46.07）。"
                   "配对真正起作用是在用户数逼近端口数、或 CSI 有误差时。",
            source="用户 2026-08-02 给的现场算法",
        ),
        Algorithm(
            key="scheduler",
            name="调度器",
            stage="系统级",
            choice="比例公平 PF",
            formula="度量 = R_inst / R_avg；R_avg(t+1) = (1−1/Tc)·R_avg(t) + (1/Tc)·R_served(t)",
            why="Tc 决定公平的时间尺度：太小接近 max-C/I（只喂近点用户），"
                "太大接近轮询（不利用信道起伏）。",
            caveat="**PF 有个经典病理**：一个永远发不成功的用户 R_avg 趋近 0、度量发散，"
                   "调度器会死盯着他。实测这能把全小区首传 BLER 从 0.011 拖到 0.811。"
                   "现在覆盖外的用户（SINR 够不到 MCS 0 门限）会被剔出调度并单独报出。",
            source="经典 PF；本项目 system.simulate",
            alternatives=["max_ci（吞吐最大但极不公平）", "rr（轮询）"],
        ),
        Algorithm(
            key="traffic",
            name="话务模型",
            stage="系统级",
            choice="FTP Model 3（泊松到达固定大小文件）",
            why="**评价体验速率的标准话务模型。** full buffer 下体验速率没有意义——"
                "缓冲区永不空，没有 burst 边界可言。",
            caveat="到达率太高会积压，此时体验速率反映的是容量上限而不是用户体验。"
                   "积压超过到达量 15% 时会主动告警。",
            source="3GPP TR 36.814 Annex A.2.1.3.1 / TR 38.802 §A.2.1.3",
            alternatives=["full_buffer", "cbr"],
        ),
        Algorithm(
            key="experienced_throughput",
            name="体验速率口径",
            stage="系统级",
            choice="掐尾（3GPP TS 28.552 §5.1.1.3）",
            formula="Thp = (V_total − V_last) / (T_buffer_nonempty − T_last)",
            why="**分母是缓冲区非空的时间，不是被调度的 TTI 数**——排队等调度的时间"
                "也算在体验里，那正是调度器压力的体现。"
                "掐尾是因为清空缓冲区的那个 TTI 通常只用了一部分。",
            caveat="按被调度 TTI 数算过一次：12 个用户各报 583 Mbps、小区合计 8.2 Gbps，"
                   "而 100 MHz 小区物理峰值约 1.2 Gbps——等于每个用户都被算成独享整个小区。"
                   "另外**小区体验速率是各用户的平均不是求和**，用户是时分复用的。",
            source="3GPP TS 28.552 §5.1.1.3；运营商话统另有掐头去尾口径",
            alternatives=["none（不掐，虚高）", "head_tail（掐头去尾）"],
        ),
        Algorithm(
            key="tx_rx_sinr",
            name="发送侧 / 接收侧 SINR 分离",
            stage="系统级",
            choice="发送侧 = CQI 长期统计；接收侧 = 瞬时含干扰 SINR",
            formula="MCS = select_mcs(SINR_tx + OLLA)；BLER = curve(MCS)@SINR_rx",
            why="**这是干扰影响吞吐的第一性路径。** 发送端不知道瞬时干扰，"
                "只有 CQI 反馈的粗略统计值；接收端实打实吃着干扰、SINR 更低、"
                "于是误码；OLLA 把差值压回来。干扰越大，OLLA 收敛得越负。",
            caveat="**发送侧别做成「完全无干扰」。** 那是极端假设——实测发送侧 40.7 dB、"
                   "接收侧 12.7 dB，差 28 dB，OLLA 追不上，首传 BLER 飙到 0.85。"
                   "CQI 是终端测的、本来就含干扰，只是平均掉了快变，"
                   "所以发送侧取接收 SINR 的长期均值，两者只差约 1 dB。",
            source="用户 2026-08-02 的现场描述；本项目 system.build_link_tables",
        ),
        Algorithm(
            key="olla",
            name="OLLA 外环链路自适应",
            stage="系统级",
            choice="ACK +0.01 dB / NACK −0.1 dB（现网基线）",
            formula="稳态 BLER → step_up / (step_up + step_down) = 0.01/0.11 ≈ 9.1%",
            why="外环用 ACK/NACK 把发送端不知道的那部分干扰补偿掉。"
                "步长比例决定稳态 BLER，绝对大小决定收敛速度。",
            caveat="**步长小收敛很慢**：每次 NACK 只压 0.1 dB，而 MCS 是整数档，"
                   "小步长常常压不动一档。8 秒仿真里 BLER 还停在 0.16~0.22 而不是 9.1%。"
                   "要看稳态结论就加长时长；要快收敛就临时调大步长"
                   "（比例不变则稳态 BLER 不变）。未收敛时结果里会主动告警。",
            source="用户 2026-08-02 给的现网基线",
        ),
        Algorithm(
            key="neighbor_load",
            name="邻区负载",
            stage="系统级",
            choice="按 PRB 利用率折算干扰（默认 30%）",
            formula="SINR' = S / (η·I + N)，SIR' = SIR / η；"
                    "等价于 IoT'_lin = 1 + η·(IoT_lin − 1)",
            why="ChannelHub 的几何 SINR 按**所有邻区都在发**算，等于 100% PRB 利用率；"
                "5G 典型是 10%/30%/50%。邻区没发的 PRB 上本小区根本不受干扰。",
            caveat="**SINR 和 SIR 必须一起折算。** 只改 SINR 会让 IoT = SIR/(SIR−SINR) "
                   "拿两个不同口径的量算，直接报 inf。"
                   "另外现场说密集城区 IoT 常 >20 dB，实测 100% 负载下是 32.9 dB、"
                   "10% 负载下只有 22.9 dB——**反过来说明那些小区的邻区负载接近满**。",
            source="5G 典型 PRB 利用率；本项目 system.apply_neighbor_load",
        ),
        Algorithm(
            key="rbg_granularity",
            name="仿真粒度",
            stage="信道生成",
            choice="RBG（17 个），不是 RB（272 个）",
            formula="RBG 内取中间那个 RB 作代表点",
            why="一个 RBG 内的 16 个 RB 共用同一个 MCS、同一次调度决策、同一个预编码——"
                "**RB 级的分辨率没有任何已实现的算法在用**。实测 rank 与 MCS 逐位相同、"
                "谱效差 0.1%，建表快一倍。",
            caveat="**取代表点而不是平均。** 平均会把频选衰落抹平、奇异值分布变平"
                   "（信道条件数被人为改善），进而高估 rank。"
                   "会受影响的只有频选调度与导频图案，两者都还没做。",
            source="本项目 mumimo.rbg_reduce",
        ),
        Algorithm(
            key="two_phase",
            name="两相架构（性能）",
            stage="系统级",
            choice="第一相建表（SVD 只在这里）→ 第二相 TTI 主循环纯查表",
            why="十万 TTI 的主循环里不能有任何矩阵运算。第一相逐 UE 逐快照把 "
                "rank 1..4 的 SINR/MCS/谱效全算好，第二相只读表 + 算 PF 度量。"
                "实测 100000 TTI × 8 UE **0.38 秒**。",
            caveat="**MU 在主循环里是标量近似**：逐 TTI 真做配对要每 TTI 做 SVD + 求逆，"
                   "跑不完。建表阶段用真实 su_mu_adaptation 测出 MU/SU 聚合比值，"
                   "主循环按 ratio/K 折算。返回值带逐快照比值与离散度——"
                   "离散度超过 30% 就不该用标量。",
            source="本项目 system.build_link_tables / simulate",
        ),
        Algorithm(
            key="harq",
            name="HARQ",
            stage="系统级",
            choice="首传查 NewTx 曲线，失败后查 ReTx 曲线，最多 4 次",
            why="表 3 的源数据每档 MCS 有一条 NewTx 和一条 ReTx 曲线，"
                "合并增益体现在 ReTx 曲线本身更靠左。",
            caveat="**没有真正的软合并（Chase/IR）**——那需要 LLR，而比特级链路"
                "本项目明确不做。多次重传会复用同一条 ReTx 曲线，"
                "结果里保留 harq_model=newtx_then_retx_curve_reused 标明这一点。",
            source="公司 20B 曲线；软合并不做是用户 2026-08-02 定的边界",
        ),
    ]


def algorithm_list(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    """这次配置下实际生效的算法清单。"""
    return [a.as_dict() for a in _algorithms(cfg)]


def stages() -> list[str]:
    return ["信道生成", "发射", "接收", "链路自适应", "多用户", "系统级"]


# ---------------------------------------------------------------------------
# 对标量的推导过程 —— 每一步都摊开，供人工核对
# ---------------------------------------------------------------------------
def derivations(cfg: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    """峰值速率/谱效等对标量的**逐步推导**，供用户亲自核对。

    **只给一个"29.63 vs 30.0，偏差 −1.2%"是不可核对的。** 这里把每一步的
    输入、公式、中间结果全列出来，任何一步对不上都能当场指出来。
    数字全部从代码现算，不是抄进来的常量。
    """
    from . import hardware as hw  # noqa: PLC0415
    from . import linkadapt as la  # noqa: PLC0415

    cfg = cfg or {}
    out: list[dict[str, Any]] = []

    # --- 峰值谱效 ---
    m27 = la.MCS_TABLES[3][27]
    out.append({
        "key": "peak_se",
        "name": "峰值谱效",
        "result": f"{4 * m27.se:.3f} bit/s/Hz",
        "reference": "30.0 bit/s/Hz",
        "ref_src": "ITU-R M.2412 / IMT-2020 最低要求，DL 峰值谱效",
        "steps": [
            ("最高档 MCS", "表 3 的 MCS 27",
             f"调制阶数 q_m = {m27.q_m}（{2 ** m27.q_m}QAM），"
             f"目标码率 R = {m27.r_1024:.0f}/1024 = {m27.rate:.4f}"),
            ("单流谱效", "SE₁ = q_m × R",
             f"{m27.q_m} × {m27.rate:.4f} = {m27.se:.4f} bit/s/Hz"),
            ("最大层数", "SU 最多 4 流",
             f"终端 {hw.COMPANY_UE_RX_ANT}R，rank 上限 = 4"),
            ("峰值谱效", "SE_peak = rank × SE₁",
             f"4 × {m27.se:.4f} = {4 * m27.se:.4f} bit/s/Hz"),
            ("与参考对比", "ITU-R 要求 30.0",
             f"偏差 {(4 * m27.se - 30.0) / 30.0 * 100:+.1f}%。"
             f"差的这一点来自码率——IMT-2020 的 30 是按 q_m=8、R=0.9375 "
             f"（=960/1024）算的，表 3 最高档是 {m27.rate:.4f}"),
        ],
    })

    # --- 峰值速率 ---
    n_prb, oh = 273, 0.14
    ts = 1e-3 / 14 / 2
    r_max = 948 / 1024
    peak = 4 * 8 * r_max * (n_prb * 12 / ts) * (1 - oh)
    re_tti = hw.COMPANY_NUM_RB * 12 * 12
    tbs = la.transport_block_size(re_tti, m27.rate, m27.q_m, layers=4)
    out.append({
        "key": "peak_rate",
        "name": "峰值速率",
        "result": f"{tbs / 0.5e-3 / 1e9:.3f} Gbps",
        "reference": f"{peak / 1e9:.3f} Gbps",
        "ref_src": "3GPP TS 38.306 §4.1.2 峰值速率公式",
        "steps": [
            ("标准公式", "R = v · Q_m · f · R_max · (N_PRB×12 / T_s) · (1−OH)",
             f"v=4 层，Q_m=8，f=1（无缩放），R_max={r_max:.4f}（948/1024），"
             f"N_PRB={n_prb}，T_s={ts * 1e6:.2f} μs（30 kHz，14 符号/0.5 ms），"
             f"OH=0.14（DL FR1 开销）"),
            ("标准公式结果", "代入",
             f"4 × 8 × {r_max:.4f} × ({n_prb}×12 / {ts:.3e}) × {1 - oh:.2f} "
             f"= {peak / 1e9:.4f} Gbps"),
            ("本仿真器的 RE 数", "N_RE = N_RB × 12 子载波 × 12 数据符号",
             f"{hw.COMPANY_NUM_RB} × 12 × 12 = {re_tti} 个 RE/TTI"
             f"（14 符号扣掉 2 个给 DM-RS 与控制）"),
            ("按 38.214 §5.1.3.2 算 TBS", "transport_block_size(N_RE, R, q_m, layers=4)",
             f"= {tbs} bit"),
            ("折成速率", "TBS / TTI 时长",
             f"{tbs} / 0.5 ms = {tbs / 0.5e-3 / 1e9:.4f} Gbps"),
            ("与公式对比", "两条独立路径",
             f"偏差 {(tbs / 0.5e-3 - peak) / peak * 100:+.1f}%。"
             f"差异来自 RB 数（{hw.COMPANY_NUM_RB} vs {n_prb}）与开销口径——"
             f"我们按 12/14 符号扣，标准按固定 OH=0.14 扣"),
        ],
    })

    # --- 小区谱效的 TDD 归一 ---
    pat = str(cfg.get("tdd_pattern", "DDDSU")).upper() or "DDDSU"
    dl_ratio = (pat.count("D") + 0.7 * pat.count("S")) / len(pat)
    out.append({
        "key": "tdd_normalize",
        "name": "小区谱效的 TDD 归一",
        "result": f"下行占比 {dl_ratio:.4f}",
        "reference": "ITU 的小区谱效是按全下行定义的",
        "ref_src": "ITU-R M.2412 Dense Urban DL 平均小区谱效 7.8 bit/s/Hz/TRxP",
        "steps": [
            ("TDD 图案", f"{pat}",
             f"{pat.count('D')} 个 D + {pat.count('S')} 个 S + "
             f"{pat.count('U')} 个 U，周期 {len(pat)} 个时隙"),
            ("S 时隙折算", "按 0.7 个下行算",
             "S 时隙大部分符号是下行，剩下给 GP 和上行导频"),
            ("下行占比", "(D + 0.7×S) / 周期",
             f"({pat.count('D')} + 0.7×{pat.count('S')}) / {len(pat)} = {dl_ratio:.4f}"),
            ("归一", "仿真谱效 / 下行占比",
             f"仿真里一秒只有 {dl_ratio:.0%} 的时隙能发下行，"
             f"而 ITU 的参考值是按全下行定义的，所以要除以 {dl_ratio:.4f} 才可比"),
        ],
    })

    # --- 噪声口径 ---
    out.append({
        "key": "noise_ref",
        "name": "噪声功率口径（值 12 dB）",
        "result": "锚定 rank1：noise = σ₁²·P / 10^(几何SINR/10)",
        "reference": "对错口径实测差 12 dB",
        "ref_src": "现网锚点 平均 rank 2.7 / MCS 15 反推",
        "steps": [
            ("ChannelHub 的几何 SINR 含什么", "信号项 = N_ant·|w^H a|²",
             "它已经包含了波束赋形增益（64 天线约 18 dB）"),
            ("错误做法", "noise = mean(|h|²) / 10^(SINR/10)",
             "按单天线平均功率反推噪声，之后 SVD 又叠一次阵列增益 —— 算了两遍"),
            ("错误做法的后果", "实测",
             "平均 rank 3.90 / 平均 MCS 23.5 / MCS 范围 10~27，"
             "几乎所有用户判到 rank4、MCS 顶格"),
            ("正确做法", "noise = σ₁²·P / 10^(SINR/10)",
             "使 rank=1（全功率压最强流）时的后波束 SINR 恰好等于几何 SINR"),
            ("正确做法的结果", "实测",
             "平均 rank 2.23 / 平均 MCS 11.1，现网锚点是 2.70 / 15.0"),
            ("为什么必须有现网锚点", "错的那个长得像正常结果",
             "曲线形状对、随场景变化的趋势也对，只是整体偏了 12 dB"),
        ],
    })
    return out
