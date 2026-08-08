"""算法族定义（续）：信道生成、发射、话务与 KPI 侧。

和 :mod:`algo_defs` 拆成两个文件纯粹是为了每个文件不至于太长，
结构与约定完全一致——每族列全部可选实现、标出当前采用、配一张流程图。
"""
from __future__ import annotations

from typing import Any

from .algo_defs import Family, Flow, Option


def _antenna(n_bs: int) -> Family:
    return Family(
        key="antenna_model",
        name="天线阵列模型",
        stage="信道生成",
        current="effective_subarray" if n_bs == 64 else "legacy_64",
        config_key="num_bs_tx_ant",
        intro="同样写「64T」，把它当 64 个独立阵元还是当 64 个 RF 端口各驱动 3 个阵子，"
              "算出来的信道完全是两回事。",
        formula=r"H_{eff} = F^H H_{phys}, \quad F \in \mathbb{C}^{192 \times 64}",
        caveat="**legacy 会把吞吐高估 27%、边缘用户高估 61%**（实测）。"
               "2026-07-31 之前生成的所有谱效与吞吐数字都偏乐观。"
               "1 驱 3 是这一款 AAU 的硬件事实、不是通用规律，"
               "所以只在面板是 8×4×2 时自动生效。",
        source="ChannelHub phy_sim/effective_array.py",
        options=[
            Option("effective_subarray", "effective_subarray（1 驱 3 真实阵列）",
                   formula=r"d_H = 0.5\lambda, \quad d_V = 0.67\lambda, \quad "
                           r"d_{RF,V} = 3 \times 0.67\lambda = 2.01\lambda",
                   summary="64 个 RF 端口，每端口固定驱动垂直相邻 3 个阵子，共 192 阵子",
                   detail="真实 AAU 的样子：8H × 4V × 2pol = 64 个 RF 端口。"
                          "RF 端口的垂直相位中心间距 2.01λ **大于一个波长**，"
                          "所以垂直方向有栅瓣——这是 legacy 模型完全看不到的物理。",
                   when="面板是 8×4×2 时自动启用（默认）",
                   cost="与 legacy 相同（用等效阵列快路径）"),
            Option("legacy_64", "legacy_64（独立阵元）",
                   formula=r"d_H = d_V = 0.5\lambda",
                   summary="把 64 个端口当 64 个独立阵元，间距一律半波长",
                   detail="ChannelHub 的历史默认。没有 1 驱 3 的耦合、没有栅瓣，"
                          "自由度被高估。",
                   when="面板不是 8×4×2 时自动落回；或要对照历史结果",
                   cost="最省"),
            Option("physical_reference", "physical_reference（真跑 192 阵子）",
                   summary="按 192 个物理阵子建模再用耦合矩阵投影回 64 端口",
                   detail="慢路径的参考实现。实测与 effective_subarray 相对差 **4.8e−7**，"
                          "说明快路径复现了参考路径——放心用快的。",
                   when="要验证快路径没写错",
                   cost="慢很多，只用于校验"),
        ],
        flow=Flow(steps=[
            ("按物理阵子建信道", "192 个阵子各自的 38.901 信道系数"),
            ("组耦合矩阵 F", "每个 RF 端口固定驱动垂直相邻 3 个阵子，F 是 192×64 稀疏阵"),
            ("投影到 RF 端口", "H_eff = F^H · H_phys，得到 64 端口的等效信道"),
            ("检查栅瓣", "RF 端口垂直相位中心 2.01λ > λ，垂直方向必有栅瓣"),
        ], branches=[(1, "面板不是 8×4×2", "落回 legacy_64，直接按 N 个独立阵元建")]),
    )


def _precoder_su() -> Family:
    return Family(
        key="precoder_su",
        name="单用户预编码",
        stage="发射",
        current="svd",
        config_key="precoder",
        intro="把要发的几条流映射到多个发射天线上。SVD 是理想上界，码本是真实系统能做到的。",
        formula=r"H^H = U \Sigma V^H, \quad W = V[:, 1{:}r]",
        caveat="用 h_true 做 SVD 得到的是**上界**，真实系统只有 h_est。"
               "两者的差就是 CSI 反馈的代价——这正是 CSI 类课题的落点。",
        source="38.214 §5.2.2（Type I 码本）",
        options=[
            Option("svd", "SVD（奇异值分解）",
                   formula=r"W = V[:, 1{:}r], \quad \text{每流功率} = P/r",
                   summary="取信道的前 r 个右奇异向量，理论最优",
                   detail="逐 RB 做 SVD，取前 rank 个右奇异向量。"
                          "**这是上界，不是可实现性能**——它要求发送端知道完整的瞬时信道。",
                   when="要一个理想上界做对照",
                   cost="逐 RB 一次 SVD"),
            Option("svd_wideband", "宽带 SVD",
                   summary="全带宽共用一组预编码，而不是逐 RB 各算各的",
                   detail="更接近真实系统（反馈开销受限），比逐 RB SVD 低一些。",
                   when="要看宽带预编码的损失",
                   cost="一次 SVD"),
            Option("type1", "Type I 码本（38.214）",
                   formula=r"W \in \mathcal{W}_{DFT}, \quad "
                           r"(i_{1,1}, i_{1,2}, i_2) \to W",
                   summary="从 DFT 波束码本里选一个，**含秩自适应**",
                   detail="38.214 的 Type I 反馈里 RI 和 PMI 是一起报的，"
                          "所以码本方案必须做秩自适应——早期版本把秩硬定成 max_rank，"
                          "在低秩信道上会输给 rank-1 的 DFT 波束，"
                          "看起来像「码本不如单波束」，其实是没做秩自适应。",
                   when="要贴近真实系统的 CSI 反馈",
                   cost="遍历码本",
                   source="38.214 §5.2.2.2.1"),
            Option("dft", "DFT 波束",
                   summary="固定的 DFT 波束，不看信道细节",
                   detail="最简单的波束赋形，作为下界对照。",
                   when="对照实验",
                   cost="最省"),
            Option("mrt", "MRT（最大比发射）",
                   formula=r"W \propto H^H",
                   summary="共轭匹配，单流时最优",
                   when="rank=1",
                   cost="最省"),
        ],
        flow=Flow(steps=[
            ("取用于预编码的信道", "h_true（上界）或 h_est（真实系统）——**这一步决定结论性质**"),
            ("逐 RB 做 SVD", "H^H = U Σ V^H"),
            ("秩判决", "按奇异值门限决定开几流（码本方案也必须做，不能硬定）"),
            ("取前 rank 个右奇异向量", "W = V[:, :rank]"),
            ("功率归一", "总功率 P=1 在 rank 条流上均分"),
        ], branches=[(1, "用的是 h_est", "预编码不再匹配真实信道，损失即 CSI 代价")]),
    )


def _rbg() -> Family:
    return Family(
        key="rbg_granularity",
        name="仿真粒度",
        stage="信道生成",
        current="rbg",
        config_key="rb_per_rbg",
        intro="272 个 RB 还是 17 个 RBG？一个 RBG 内的 16 个 RB 共用同一个 MCS、"
              "同一次调度决策、同一个预编码——**RB 级的分辨率没有任何已实现的算法在用**。",
        formula=r"H_{RBG}[b] = H_{RB}[16b + 8]",
        caveat="**取代表点而不是平均。** 平均会把频选衰落抹平、奇异值分布变平"
               "（信道条件数被人为改善），进而**高估 rank**。"
               "会受影响的只有频选调度与导频图案，两者都还没做。",
        source="本项目 mumimo.rbg_reduce",
        options=[
            Option("rbg", "RBG 粒度（17 个）",
                   summary="每个 RBG 取中间那个 RB 作代表",
                   detail="实测 rank 与 MCS **逐位相同**、谱效差 0.1%、建表快一倍。",
                   when="默认",
                   cost="272 → 17，SVD 少算 16 倍"),
            Option("rb", "RB 粒度（272 个）",
                   summary="逐 RB 全算",
                   detail="做频选调度或导频图案时必须退回这一档。",
                   when="频选调度 / 导频图案（都还没做）",
                   cost="16 倍的 SVD"),
        ],
        flow=Flow(steps=[
            ("拿到逐 RB 的信道", "[272, BS, UE]"),
            ("选代表 RB", "每 16 个取中间那个（第 8 个）"),
            ("后续全部在 RBG 上算", "SVD、SINR、MCS 都只跑 17 次而不是 272 次"),
        ], branches=[(2, "rb_per_rbg = 1", "跳过，退回 RB 粒度")]),
    )


def _traffic() -> Family:
    return Family(
        key="traffic",
        name="话务模型",
        stage="系统级",
        current="ftp3",
        config_key="traffic_model",
        intro="用户什么时候有数据要发、一次发多少。**体验速率这个 KPI 只在有 burst 边界时才有意义。**",
        formula=r"P(\text{到达}) = \lambda \cdot T_{TTI}, \quad "
                r"\text{负载} = \lambda \cdot S_{file} \cdot 8",
        caveat="full buffer 下**体验速率没有意义**——缓冲区永不空、没有 burst 边界。"
               "到达率太高会积压，此时体验速率反映的是容量上限而不是用户体验，"
               "积压超过到达量 15% 会主动告警。",
        source="3GPP TR 36.814 Annex A.2.1.3.1 / TR 38.802 §A.2.1.3",
        options=[
            Option("ftp3", "FTP Model 3",
                   formula=r"\text{到达} \sim \text{Poisson}(\lambda), \quad "
                           r"S_{file} = \text{const}",
                   summary="泊松到达固定大小的文件",
                   detail="3GPP 评价体验速率的**标准话务模型**。到达率控制负载。",
                   when="默认；对标 3GPP 参考值时",
                   cost="最省",
                   source="TR 36.814 Annex A.2.1.3.1"),
            Option("bimodal", "现网两头高中间低",
                   formula=r"P(1\,\text{RBG}) = 0.3, \quad P(N_{RBG}) = 0.3, "
                           r"\quad P(\text{空闲 TTI}) = 0.3",
                   summary="按**占用 RBG 数**分布：小包和满带宽各占 30%，中间均匀",
                   detail="现网口径（用户 2026-08-02）。**这是一次传输占多少频域资源的分布，"
                          "不是文件大小的分布**——两者完全不同。"
                          "小包与大包的体验速率分开报，因为前者由调度时延主导。",
                   when="要贴近现网话务",
                   cost="最省"),
            Option("full_buffer", "full buffer",
                   summary="永远有数据要发",
                   detail="用来测容量上限。**体验速率在这个模型下没有意义。**",
                   when="测小区容量、对标 ITU 的平均小区谱效",
                   cost="最省"),
            Option("cbr", "CBR（恒定速率）",
                   summary="每 TTI 固定字节数到达",
                   when="模拟固定码率业务",
                   cost="最省"),
        ],
        flow=Flow(steps=[
            ("每 TTI 抽一次到达", "伯努利近似泊松，p = λ·T_TTI"),
            ("决定这次的大小", "ftp3 用固定文件大小；bimodal 抽 RBG 数再折算字节"),
            ("进缓冲区", "该 UE 没有活跃 burst 就直接激活，否则排队"),
            ("被调度时扣减", "记录首次/末次被调度的 TTI，供体验速率统计用"),
            ("发完出队", "下一个排队的 burst 接上"),
        ], loop_back=(5, 1, "持续到仿真结束")),
    )


def _exp_thp() -> Family:
    return Family(
        key="experienced_throughput",
        name="体验速率口径",
        stage="系统级",
        current="tail",
        config_key="trim",
        intro="**这是现网真正上报的 KPI，也是最容易算错的一个。** "
              "它不是吞吐量的平均，只在有数据要发的那段时间里算，而且要掐头去尾。",
        formula=r"Thp = \frac{V_{total} - V_{last}}{T_{buffer \neq \emptyset} - T_{last}}",
        caveat="**分母是一段时间，不是被调度的 TTI 数。** 按后者算过一次，"
               "12 个用户各报 583 Mbps、小区合计 8.2 Gbps——"
               "而 100 MHz 小区物理峰值约 1.2 Gbps，等于每个用户都被算成独享整个小区。"
               "另外**小区体验速率是各用户的平均不是求和**，用户是时分复用的。",
        source="3GPP TS 28.552 §5.1.1.3",
        options=[
            Option("tail", "掐尾（3GPP 标准口径）",
                   formula=r"V \leftarrow V - V_{last}, \quad T \leftarrow T - T_{last}",
                   summary="排除清空缓冲区的最后一个 slice",
                   detail="那个 TTI 通常只用了一部分就把数据发完，算进去等于用"
                          "半个 TTI 的时间去除半个 TTI 的数据，得到虚高的瞬时速率。"
                          "**单 slice 的 burst 因此完全无法测量**——小包测不到体验速率"
                          "不是 bug，是这个 KPI 的固有盲区。",
                   when="对标 3GPP / 标准话统",
                   cost="零",
                   source="TS 28.552 §5.1.1.3"),
            Option("head_tail", "掐头去尾（运营商口径）",
                   formula=r"T \leftarrow T_{last} - T_{first\_sched}",
                   summary="起点从数据到达挪到**首次被调度**",
                   detail="话务到达但还没被调度的等待时间**不计入分母**"
                          "（用户 2026-08-02 明确）。轻载时两者差别很大。",
                   when="对标你们的现网话统",
                   cost="零"),
            Option("none", "不掐",
                   summary="含清空缓冲区的那个 TTI",
                   detail="**数值虚高，不建议**。只作为理解口径影响的对照。",
                   when="对照实验",
                   cost="零"),
        ],
        flow=Flow(steps=[
            ("找出这个 burst 的边界", "数据到达 tti、首次被调度 tti、末次被调度 tti"),
            ("定起点", "tail/none 从到达算；head_tail 从首次被调度算"),
            ("扣掉最后一个 slice", "数据与时间同时扣——它把缓冲区清空了，不完整"),
            ("太短的丢掉", "少于 2 个 slice 无法测量，整个 burst 不计入"),
            ("按用户聚合", "该用户所有合格 burst 的速率取平均"),
            ("按小区聚合", "**各用户体验速率的平均，不是求和**"),
        ], branches=[(4, "只有 1 个 slice", "丢弃——小包的固有盲区")]),
    )


def _harq() -> Family:
    return Family(
        key="harq",
        name="HARQ",
        stage="系统级",
        current="curve_reuse",
        config_key="",
        intro="首传失败后重传。合并增益体现在 ReTx 曲线本身比 NewTx 曲线更靠左。",
        formula=r"P(\text{成功}) = 1 - \prod_{k=1}^{N} \text{BLER}_k",
        caveat="**没有真正的软合并（Chase / IR）**——那需要 LLR，而比特级链路"
               "本项目明确不做。多次重传复用同一条 ReTx 曲线，"
               "结果里保留 <code>harq_model=newtx_then_retx_curve_reused</code> 标明这一点。"
               "**重传增益会被低估。**",
        source="公司 20B 曲线；软合并不做是用户 2026-08-02 定的边界",
        options=[
            Option("curve_reuse", "NewTx / ReTx 双曲线",
                   summary="首传查 NewTx 曲线，失败后查 ReTx 曲线",
                   detail="表 3 的源数据每档 MCS 有一条 NewTx 和一条 ReTx。"
                          "多次重传复用同一条 ReTx——**这是个已知的天花板**。",
                   when="默认（也是唯一可用的）",
                   cost="查表"),
            Option("chase", "Chase 合并（未实现）",
                   formula=r"\gamma_{comb} = \sum_{k} \gamma_k",
                   summary="把多次重传的软信息按 SINR 相加",
                   detail="需要 LLR。**本项目明确不做比特级链路，所以这条做不了。**",
                   when="不可用",
                   cost="需要比特级链路"),
        ],
        flow=Flow(steps=[
            ("首传", "按发送侧 SINR + OLLA 选 MCS，按接收侧 SINR 查 NewTx 曲线"),
            ("抽 ACK / NACK", "伯努利，概率就是查到的 BLER"),
            ("NACK 则进重传队列", "该 UE 在重传完成前不开新的首传"),
            ("重传查 ReTx 曲线", "曲线更靠左，体现合并增益"),
            ("最多 4 次", "还失败就丢弃，计入残留 BLER"),
        ], loop_back=(4, 3, "还没成功就再来一次"),
           branches=[(2, "ACK", "数据交付，OLLA 上调偏置")]),
    )


def _neighbor() -> Family:
    return Family(
        key="neighbor_load",
        name="邻区负载",
        stage="系统级",
        current="jitter",
        config_key="neighbor_prb_util",
        intro="ChannelHub 的几何 SINR 是按**所有邻区都在发**算的，等于 100% PRB 利用率。"
              "真实网络 5G 典型是 10% / 30% / 50%。",
        formula=r"SINR' = \frac{S}{\eta I + N}, \quad SIR' = \frac{SIR}{\eta}, "
                r"\quad IoT'_{lin} = 1 + \eta (IoT_{lin} - 1)",
        caveat="**SINR 和 SIR 必须一起折算。** 只改 SINR 会让 "
               "IoT = SIR/(SIR−SINR) 拿两个不同口径的量算，直接报 inf。"
               "另外实测现网密集城区 IoT &gt;20 dB 对应的是**接近满负载**："
               "100% 负载下 32.9 dB、10% 负载下只有 22.9 dB。"
               "<br>**当前只支持全网统一值**（用户 2026-08-03 定）。"
               "<br>**这个限制的原因 2026-08-07 被推翻了一半。** 原来写的是"
               "「几何 SIR 只给聚合量、拿不到逐邻区贡献」——存下来的确实只有聚合量，"
               "但分解是可恢复的：干扰求和式是 "
               "<code>I = Σ_k rx_lin[k]·N_ant·avg_leak_k</code>，而全码本平均下 "
               "<code>avg_leak</code> 跨小区是常数（Parseval），"
               "所以<b>在期望意义上逐小区份额精确等于 RSRP 份额</b>；"
               "ChannelHub 已经算出了全部小区的 RSRP，只是 superwireless 的"
               "标量字段过滤把这个数组丢了，补上只要约 168 字节/样本。"
               "<br>但**单次实现下这个分解不成立**："
               "<code>n_dl_sched = max(1, round(ues_per_cell·pdsch_load))</code> "
               "在默认预设下恒等于 1，每个干扰小区只随机抽<b>一个</b>波束，"
               "实测由此带来 <b>4.74 dB</b> 的抽签噪声。"
               "**要做逐小区负载，得先把这个解决掉。**",
        source="5G 典型 PRB 利用率；本项目 system.apply_neighbor_load",
        options=[
            Option("scaled", "按 PRB 利用率线性折算",
                   formula=r"I' = \eta I, \quad N' = N",
                   summary="干扰按利用率缩放，噪声不变",
                   detail="邻区没在发的那些 PRB 上，本小区用户根本不受干扰。"
                          "η=1 时退化成原来的 full buffer 行为。",
                   when="默认（0.3）",
                   cost="零"),
            Option("jitter", "带 ±5% 抖动的线性折算",
                   formula=r"\eta_s \sim \mathcal{U}\big(0.95\,\eta,\; 1.05\,\eta\big)",
                   summary="每个快照抽一份自己的利用率",
                   detail="恒定负载会让所有快照的干扰**完全一样**，结果比现网干净。"
                          "真实网络的负载逐 TTI 就在抖。抖动是乘性的，"
                          "0.3 → [0.285, 0.315]。<b>这是当前默认</b>"
                          "（用户 2026-08-03：「实际结果可以在配置值 ±5% 范围内波动」）。",
                   when="默认",
                   cost="零"),
            Option("full", "full buffer（η = 1）",
                   summary="所有邻区都在发",
                   detail="ChannelHub 几何 SINR 的原始假设。**要复现现网 IoT &gt;20 dB "
                          "就该用接近这个值。**",
                   when="对标现网高干扰场景",
                   cost="零"),
        ],
        flow=Flow(steps=[
            ("拿到几何 SINR 与 SIR", "都来自 ChannelHub 的同一次几何计算，口径一致"),
            ("反推干扰与噪声", "令 S=1：I = 1/SIR，N = 1/SINR − I"),
            ("按利用率缩放干扰", "I' = η·I，噪声不动"),
            ("重算 SINR 与 SIR", "SINR' = 1/(η I + N)，SIR' = SIR/η —— **必须一起改**"),
            ("再算 IoT", "用折算后的同口径两个量"),
        ], branches=[(1, "单小区（SIR 是 49.9 哨兵）", "没有干扰可折算，原样返回")]),
    )


def _two_phase() -> Family:
    return Family(
        key="two_phase",
        name="两相架构（性能）",
        stage="系统级",
        current="table_lookup",
        config_key="",
        intro="十万个 TTI 的主循环里**不能有任何矩阵运算**。"
              "把贵的都挪到第一相，主循环只查表。",
        formula=r"O(N_{UE} \cdot N_{snap} \cdot N_{rank}) \text{ 次 SVD} "
                r"+ O(N_{TTI} \cdot N_{UE}) \text{ 次查表}",
        caveat="**MU 在主循环里是标量近似**：逐 TTI 真做配对要每 TTI 做 SVD + 矩阵求逆，"
               "跑不完。建表阶段用真实的 su_mu_adaptation 测出 MU/SU 聚合比值，"
               "主循环按 ratio/K 折算。返回值带逐快照比值与离散度——"
               "实测离散度 3.7%~13.1%，**超过 30% 就不该用标量**。",
        source="本项目 system.build_link_tables / simulate",
        options=[
            Option("table_lookup", "两相：建表 + 查表",
                   summary="SVD 只在第一相做，主循环纯查表加算术",
                   detail="实测 **100000 TTI × 8 UE 只要 0.38 秒**。"
                          "把 SVD 放进主循环的话同规模要几十分钟。",
                   when="默认",
                   cost="第一相 0.55 秒，第二相每 10 万 TTI 0.38 秒"),
            Option("per_tti", "逐 TTI 全算（未实现）",
                   summary="每个 TTI 重新做 SVD 与配对",
                   detail="最准，但十万 TTI 跑不完。**如果哪天要精确的逐 TTI MU 配对，"
                          "得先把这一层的性能问题解决掉。**",
                   when="不可用",
                   cost="慢两个数量级"),
        ],
        flow=Flow(steps=[
            ("第一相：逐 UE 逐快照", "对每个 rank 1..4 算 SINR / MCS / 谱效，存成表"),
            ("第一相：测 MU/SU 比值", "在若干快照上跑真实的 SU/MU 自适应，取中位数"),
            ("第一相：判覆盖外", "用户级 SINR 够不到 MCS 0 门限的快照标出来"),
            ("第二相：TTI 主循环", "只读表 + 算 PF 度量 + 更新缓冲区，**无矩阵运算**"),
            ("第二相：BLER 查表", "按 0.5 dB 量化后缓存，命中率接近 100%"),
        ], branches=[(2, "MU 比值离散度 > 30%", "标量近似不成立，结果里告警")]),
    )


def _csi_aging() -> Family:
    return Family(
        key="csi_aging",
        name="CSI 反馈时延与老化",
        stage="发射",
        current="srs_hop_17",
        config_key="srs_period_ms",
        intro="**基站永远不知道「现在」的信道。** TDD 下行靠互易性从上行 SRS 取 CSI，"
              "所以从探测到发送之间隔着一整条时延链：SRS 发送 → 信道估计 → "
              "预编码计算 → PDSCH 发送。这段时间信道一直在变。"
              "平台在此之前默认零时延完美 CSI——预编码与评估用同一个矩阵，"
              "SVD 永远精确匹配、ZF 零陷永远打得准，"
              "**这系统性地高估 MU 增益**，因为 MU 的全部收益就建立在零陷打得准上。",
        formula=r"W = \mathrm{SVD}(H_{t-\tau}), \quad "
                r"\mathrm{SINR}_k = \frac{1}{\left[\left(I + "
                r"\tfrac{P}{r} (H_t W)^H (H_t W)\right)^{-1}\right]_{kk}} - 1",
        caveat="**零时延时这套公式必须逐位退化成 σ_k²·P/rank/σ_n²**，"
               "也就是原来的 su_rank_adaptation 用的那个特征值公式——"
               "因为那时 H_t W = UΣ_r 是对角的。这条恒等式是老化模型的地基，"
               "不成立就说明它是叠加上去的第二套物理，任何「老化损失」都不可解释。"
               "test_csi_aging 第 1 节实测最大偏差 0 dB。"
               "<br>**另一个极易写错的地方：rank 必须由基站按自己的陈旧 CSI 选。** "
               "拿真实 SINR 去挑 rank 等于让基站预知信道，它会自动避开老化最狠的 rank，"
               "损失被凭空抹掉一大半。",
        source="38.211 §6.4.1.4.3 与 Table 6.4.1.4.3-1；跳频序列直接调 ChannelHub 的 "
               "srs_rb_indices，不自己重写",
        options=[
            Option("srs_hop_17", "SRS 跳频（C_SRS=57，17 跳 × 16 RB）",
                   formula=r"\text{age}(k) = \big((n - k) \bmod 17\big) \cdot T_{SRS} "
                           r"+ \delta_{proc}",
                   summary="每次 SRS 只探 1 个 RBG，17 跳扫完全带",
                   detail="38.211 Table 6.4.1.4.3-1 的 C_SRS=57 行："
                          "m_SRS=(272,16,4,4)、N=(1,17,4,1)。取 B_SRS=1 时"
                          "每次 SRS 占 <b>16 RB，正好 1 个 RBG</b>，"
                          "要 <b>17 跳</b>才扫完 272 RB——和本项目的 17 RBG × 16 RB "
                          "载波配置 1:1 对上。"
                          "<br><b>这是老化的主导项</b>：T_SRS=10 ms 时全带扫一遍要 "
                          "<b>170 ms</b>，某个 RBG 的 CSI 年龄在 0~160 ms 之间轮转，"
                          "平均 80 ms。而 2.6 GHz、30 km/h 的相干时间只有约 3 ms。"
                          "<br>年龄<b>随时间轮转</b>，不会有某几个 RBG 永远最差。",
                   when="默认（现网为省上行开销、提高导频功率密度普遍开跳频）",
                   cost="实测 MU/SU 比值 0.816 → 0.449（−45%），SU 谱效 −27%",
                   source="38.211 Table 6.4.1.4.3-1 第 57 行"),
            Option("srs_nohop", "不跳频（每次探全带）",
                   formula=r"\text{age} = (t \bmod T_{SRS}) + \delta_{proc}",
                   summary="全带年龄相同，只剩周期内相位 + 处理时延",
                   detail="上行开销大得多（一次要占满 272 RB），"
                          "但 CSI 新鲜得多。实测 SU 谱效只掉 10%（跳频掉 27%）。",
                   when="SRS 资源充裕、或要单独看跳频的代价",
                   cost="上行开销 × 17"),
            Option("perfect", "零时延完美 CSI（关掉老化）",
                   summary="预编码与评估用同一个信道矩阵",
                   detail="<b>这是上界，不是现网。</b>保留它是为了能做 A/B 对比——"
                          "老化的代价必须能被量出来，而不是悄悄混进所有结果里。",
                   when="要上界基线时",
                   cost="系统性高估 MU 增益"),
        ],
        flow=Flow(steps=[
            ("定 SRS 周期", "5 / 10 / 20 / 40 ms，对应 38.331 的 sl10/20/40/80（30 kHz）"),
            ("查跳频序列", "调 ChannelHub 的 srs_rb_indices（38.211 §6.4.1.4.3 完整跳频树），"
                           "C_SRS=57 / B_SRS=1 给出 RBG 0→1→…→16 循环"),
            ("算逐 RBG 年龄", "age(k) = ((n−k) mod 17)·T_SRS + 周期内相位 + 处理时延"),
            ("量化成整数快照", "lag(k) = round(age(k) / 快照间隔)，快照间隔默认 5 ms"),
            ("拼出基站以为的信道", "第 k 个 RBG 取自 lag(k) 个快照之前；"
                                   "**越界钳到最早快照，绝不回绕**（回绕=拿未来当过去）"),
            ("用陈旧信道算预编码", "W = SVD(H_stale)，逐 RBG"),
            ("用当前信道评估", "SINR = MMSE(H_true, W)，失配表现为 BF 增益下降 + 流间泄漏"),
            ("rank 也按陈旧 CSI 选", "基站不知道真实信道支持几流——"
                                     "高速下「点了 rank4、实际只撑得住 rank1」正是老化损失的一环"),
        ], branches=[
            (0, "关掉老化", "H_stale = H_true，整条链退化成零时延，结果与原实现逐位相同"),
            (3, "滞后全部量化成 0", "老化模型此时几乎不起作用，aging_summary 主动告警"),
        ]),
    )


def _tx_sinr() -> Family:
    return Family(
        key="tx_sinr",
        name="发送侧 SINR（CQI + BF Gain）",
        stage="链路自适应",
        current="cqi_bf",
        config_key="",
        intro="**发送侧和接收侧是两个 SINR。** 基站选 MCS 时手里只有 CQI 反馈和"
              "它自己能算的 BF 增益；接收端实打实吃着干扰。两者的差由误码经 OLLA "
              "收敛回来——这是干扰影响吞吐的第一性路径。",
        formula=r"\mathrm{SINR}_{tx} = \underbrace{\Gamma\big(\mathrm{MCS}"
                r"(\mathrm{CQI})\big)}_{\text{长期滤波的宽带上报}} + "
                r"\underbrace{\overline{\mathrm{SINR}_{SVD} - \mathrm{SINR}_{PMI}}}"
                r"_{\text{基站自算，逐次调度}}",
        caveat="**CQI 是长期滤波的宽带量，BF Gain 是瞬时量**——这个分工不能混。"
               "CQI 由终端在真实信道上用 PMI 权测得、上报周期远长于一个 TTI；"
               "BF Gain 基站从自己的 SRS 信道算，所以开老化时它算的是"
               "<b>滞后那一刻</b>的增益，会系统性高估（以为预编码是匹配的），"
               "于是 MCS 点高了、误码上来、OLLA 再拉回去。"
               "<br>早先版本把发送侧写成「接收 SINR 的长期均值」，"
               "那是个<b>事后诸葛亮</b>的量：它已经包含了 SVD 的实际增益，"
               "等于假设基站预先知道自己波束打得准不准。"
               "<br><b>已核查的一处近似：</b>Type I 码本是在<b>时频平均后的信道</b>"
               "上贪心选波束的（<code>measure.pmi_type_i</code>），"
               "而教科书口径是在<b>空间协方差</b> "
               "<code>R = Σ h hᴴ</code> 上选（相位不敏感）。"
               "实测两者差 <b>0.2 ~ 1.0 dB</b>（协方差法略优），已知但没改——"
               "改的是一个被多条路径共用的函数，为 1 dB 不值得。"
               "另外 iid 瑞利信道上 BF 增益高达 13 dB <b>不是 bug 而是真实物理</b>："
               "空间白信道没有结构，任何宽带码本波束都对不准；"
               "换成有角度结构的多径信道后是 2.6 ~ 5 dB，正对得上现场量级。",
        source="现场 TDD AMC 流程（用户 2026-08-03 确认）；38.214 §5.2.2",
        options=[
            Option("cqi_bf", "CQI 门限 + BF Gain",
                   formula=r"\mathrm{CQI} \to \mathrm{MCS} \to \Gamma_{10\%} "
                           r"\to +\,\mathrm{BFGain} \to \mathrm{MCS}' \to "
                           r"+\,\mathrm{OLLA} \to \lfloor \cdot \rfloor",
                   summary="现场口径：CQI 按谱效映射 MCS，取该 MCS 的目标 BLER SINR 门限，加 BF 增益",
                   detail="CQI 用 38.214 Table 5.2.2.1-3（表 2，含 256QAM）。"
                          "PMI 走 <b>Type I 宽带码本</b>——全带共用一个权，"
                          "正对应现场的<b>全带 CQI</b>（不做子带 CQI、不做频选调度）。"
                          "<br>宽带 PMI 是慢时间尺度的量，所以逐 UE 逐 rank 在时间平均信道上"
                          "搜一次码本就够，不逐快照重搜。"
                          "<br><b>CQI=0 不退化成 −inf</b>：它的意思是「低于 CQI 表下界」，"
                          "不是「这个用户不存在」，退回实测 PMI SINR，OLLA 还能在它上面工作。",
                   when="默认",
                   cost="每 UE 每 rank 一次 Type I 码本搜索，约 40 ms"),
            Option("rx_longterm", "接收 SINR 的长期均值（已弃用）",
                   summary="拿接收侧 SINR 在快照上取均值当发送侧",
                   detail="<b>事后诸葛亮</b>：这个量里已经含了 SVD 的实际增益，"
                          "等于让基站预知波束打得准不准。开 CSI 老化后它的问题变得致命——"
                          "老化的全部代价就是「基站以为打准了其实没有」，"
                          "而这个口径直接把它抹平了。",
                   when="不再使用",
                   cost="抹掉 CSI 老化的主要损失"),
            Option("interference_free", "完全无干扰（已弃用）",
                   summary="按反推出的无干扰 SNR 选 MCS",
                   detail="极端假设。实测发送侧 40.7 dB、接收侧 12.7 dB，<b>差 28 dB</b>，"
                          "OLLA 的钳位根本追不上，首传 BLER 飙到 0.85。",
                   when="不再使用",
                   cost="OLLA 发散"),
        ],
        flow=Flow(steps=[
            ("终端测 CQI", "用基站下发的 Type I 宽带 PMI 权，在**真实信道**上测，"
                           "含干扰；长期滤波后上报一个宽带值"),
            ("量化成 CQI index", "38.214 Table 5.2.2.1-3，满足 10% BLER 的最高档"),
            ("CQI → 初始 MCS", "按谱效就近映射到 MCS 表 3"),
            ("取该 MCS 的 SINR 门限", "该档 NewTx 曲线上 BLER=10% 对应的 SINR"),
            ("基站自算 BF Gain", "同一信道、同一 rank、同一功率、同一接收机下 "
                                 "SINR_SVD − SINR_PMI，逐 RBG 逐流在 dB 域平均；"
                                 "**开老化时算的是陈旧信道上的增益**"),
            ("相加得发送侧 SINR", "Γ(MCS(CQI)) + BF Gain"),
            ("重映射 MCS", "按发送侧 SINR 选满足目标 BLER 的最高 MCS"),
            ("加 OLLA 偏置", "逐 TTI 更新的 dB 域偏置，ACK +0.01 / NACK −0.1"),
            ("接收端判误码", "用**真实**接收 SINR 查 BLER 曲线——发送侧与接收侧的差就在这里变成误码"),
        ], branches=[
            (1, "CQI = 0（低于表下界）", "退回实测 PMI SINR，不用 −inf"),
        ], loop_back=(8, 7, "ACK/NACK 反馈驱动 OLLA，下一次调度生效")),
    )


def extra_families(cfg: dict[str, Any]) -> list[Family]:
    n_bs = int(cfg.get("num_bs_tx_ant", 64) or 64)
    return [
        _antenna(n_bs),
        _rbg(),
        _precoder_su(),
        _csi_aging(),
        _tx_sinr(),
        _traffic(),
        _exp_thp(),
        _harq(),
        _neighbor(),
        _two_phase(),
    ]
