"""决策点知识表：这个任务该和用户对齐哪些事。

这是 superwireless 唯一不可替代的部分。参数转发谁都能写，但"哪几件事
一改结论就翻盘、为什么、选错会怎样"是无线领域知识。

分两层，顺序不能颠倒：

* **实验设计层**（design）——跟什么基线比？用什么指标？结论要推广到哪？
  参数配错了重跑就行，实验设计错了整个结论作废。所以先问这层。
* **仿真参数层**（decisions）——信道模型、天线、带宽……

另外两样东西同样重要：
* **对比组建议**（sweeps）——很多结论必须靠 A/B 才立得住，MCP 主动提。
* **常见陷阱**（pitfalls）——这类课题上大家反复踩的坑。

收敛式，不是发散式：参数空间有限且已知，不让模型自由发挥。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# ---------------------------------------------------------------------------
# 数据结构
# ---------------------------------------------------------------------------


@dataclass
class Option:
    """一个可选项。

    ``recommended`` 参考 superpowers 的做法：给选项的同时明确说出推荐哪个
    并讲理由，用户才好判断，而不是面对一排平铺的选项自己权衡。
    """

    value: Any
    label: str
    note: str = ""
    recommended: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "value": self.value,
            "label": self.label,
            "note": self.note,
            "recommended": self.recommended,
        }


@dataclass
class Decision:
    """一个值得问用户的仿真参数。"""

    key: str
    question: str
    default: Any
    why: str  # 为什么这个选择会改变结论 —— 最关键的一栏
    options: list[Option] = field(default_factory=list)
    priority: int = 5  # 越小越该先问

    def as_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "question": self.question,
            "default": self.default,
            "why": self.why,
            "options": [o.as_dict() for o in self.options],
            "priority": self.priority,
        }


@dataclass
class DesignQuestion:
    """实验设计层的问题。没有默认值——这些必须用户自己想清楚。

    它不影响任何仿真参数，但决定了这批数据能不能支撑用户想要的结论。

    **给选项，不要只抛问题。** 开放式提问会让用户卡壳——"你要跟什么比"
    这种问题，很多人第一反应是答不上来，但看到"理想信道的 SVD 上界"
    这个选项就会想起来还有这条路。选项的作用是启发，不是限制，
    所以每题都留"其他/让我自己说"。
    """

    key: str
    question: str
    why: str
    options: list[Option] = field(default_factory=list)
    optional: bool = False  # True 表示用户不答也能走
    allow_free: bool = True  # 是否允许自由作答

    def as_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "question": self.question,
            "why": self.why,
            "options": [o.as_dict() for o in self.options],
            # examples 是 options 的扁平化，保留给只想要一句话提示的调用方
            "examples": [o.label for o in self.options],
            "optional": self.optional,
            "allow_free": self.allow_free,
        }


@dataclass
class Sweep:
    """建议的对比维度：一次生成多组数据，结论才站得住。"""

    key: str
    values: list[Any]
    label: str
    why: str

    def as_dict(self) -> dict[str, Any]:
        return {"key": self.key, "values": self.values, "label": self.label, "why": self.why}


@dataclass
class TaskProfile:
    """一类仿真任务的完整画像。"""

    task: str
    label: str
    keywords: tuple[str, ...]
    decision_keys: tuple[str, ...]
    design_keys: tuple[str, ...] = ()
    sweeps: tuple[Sweep, ...] = ()
    pitfalls: tuple[str, ...] = ()
    config_hints: dict[str, Any] = field(default_factory=dict)
    guards: tuple[str, ...] = ()


# ---------------------------------------------------------------------------
# 实验设计层问题
# ---------------------------------------------------------------------------

_DESIGN: dict[str, DesignQuestion] = {
    q.key: q
    for q in (
        DesignQuestion(
            key="baseline",
            question="你的方法要跟什么比？",
            why=(
                "没有基线的仿真结果没法解读——"
                "「NMSE 达到 -18 dB」本身不说明任何问题，"
                "关键是它比现有方案好还是差。基线定下来，"
                "才知道要不要同时生成对照数据。"
            ),
            options=[
                Option("type1_2", "3GPP Type I 或 Type II 码本",
                       "最常见的基线，几乎所有 CSI 相关工作都会比它",
                       recommended=True),
                Option("svd_bound", "理想信道的 SVD 预编码",
                       "理论天花板，看你离上界还差多少"),
                Option("published", "某篇已发表方法",
                       "告诉我哪篇，我尽量对齐它的实验设置便于横向比"),
                Option("none_yet", "还没定，先看可行性",
                       "先跑一组基准数据，基线之后再补"),
            ],
        ),
        DesignQuestion(
            key="metric",
            question="用什么指标判断好坏？",
            why=(
                "指标决定了要留哪些量，也决定了需要多少样本——不同指标的方差"
                "差一个数量级，谱效均值收敛得快，5% 边缘用户谱效要慢得多。\n"
                "谱效的口径是 `SE = mean_rb Σ_layer log2(1 + 后处理SINR)`，"
                "各层等功率、总发射功率归一，接收机默认 MMSE。"
            ),
            options=[
                Option("se", "平均谱效 bit/s/Hz（香农口径）",
                       "最常用。含预编码与接收机全链路，收敛也最快",
                       recommended=True),
                Option("throughput", "真实吞吐 Mbps（含 MCS/BLER）",
                       "走链路到系统映射：调制受限 + 码率离散 + 有限码长都算进来。"
                       "比香农谱效低 25~60%，但那才是系统真能拿到的"),
                Option("cell_edge", "5% 边缘用户吞吐 / 谱效",
                       "3GPP 评估里的公平性指标。**方差大得多**，样本数要按均值的 3~5 倍准备"),
                Option("sinr", "SINR 分布",
                       "不经过预编码，直接看链路质量；适合干扰协调、功控类工作"),
            ],
        ),
        DesignQuestion(
            key="effect_size",
            question="你预期的增益有多大？低于多少就不值得做？",
            why=(
                "**这题决定样本数，所以它比样本数本身更该问。**\n"
                "样本量不是拍脑袋定的：给定期望效应 Δ 和逐样本差值的标准差 σ_d，"
                "N ≥ ((1.96+0.84)·σ_d/Δ)^2。反过来，样本数定了就能算出这个实验"
                "最小能检出多大效应——如果它比你期望的增益还大，这次实验"
                "**无论跑出什么结果都不足以下结论**，该加样本而不是跑完再解释。\n"
                "流程上先跑 20 个样本做试点量 σ_d，再算正式样本数（`sw_sample_size`）。"
            ),
            options=[
                Option("large", "明显增益：谱效 +10% 以上",
                       "样本数需求低，几十个通常够", recommended=True),
                Option("moderate", "中等增益：谱效 +3~10%",
                       "典型量级，一般需要一两百个样本"),
                Option("small", "小幅增益：谱效 +1~3%",
                       "**要几百到上千个样本**。先想清楚这个量级的增益值不值得这么跑"),
                Option("unknown", "不知道，先探路",
                       "那就先跑 20 个试点，用实测的方差反推正式样本数"),
            ],
        ),
        DesignQuestion(
            key="csi_basis",
            question="预编码用理想信道还是估计信道？",
            why=(
                "这题不问清楚，最容易出的错是**两边口径不一致**——自己的方法用理想"
                "信道预编码、基线用估计信道，那测出来的增益里混着"
                "「知道了答案」的部分。这是无线论文评审最常抓的一条，"
                "门 2 会强制两臂口径一致。\n"
                "理想 CSI 给的是上界，估计 CSI 给的是实际系统能做到的，两者都有意义，"
                "但必须在同一口径下比。"
            ),
            options=[
                Option("ideal", "理想信道（h_true）",
                       "算法本身的上界，排除估计误差干扰。适合先看想法成不成立",
                       recommended=True),
                Option("estimated", "估计信道（h_est）",
                       "贴近实际系统。导频开销与估计误差的代价都会体现出来"),
                Option("both", "两个都跑",
                       "两者之差就是 CSI 误差的代价，本身常常是个有价值的结论"),
            ],
            optional=True,
        ),
        DesignQuestion(
            key="scope",
            question="这个结论想推广到什么范围？",
            why=(
                "只在一种场景下验证，结论就只在那种场景成立。"
                "想说「在城区普遍有效」，至少要覆盖散射强弱两类；"
                "想说「对高速用户也管用」，就得扫速度。"
                "范围定下来，需要几组对照就清楚了。"
            ),
            options=[
                Option("single", "就这一个场景，先看可行性",
                       "最快，一组数据即可", recommended=True),
                Option("urban_general", "城区宏站普遍适用",
                       "建议扫散射丰富度和视距比例，两组对照"),
                Option("real_world", "要在真实地形上站得住",
                       "统计信道定型后，用射线追踪的真实城市地图复核"),
                Option("paper", "要写进论文或对外汇报",
                       "建议至少两个维度对照，并明确报告视距比例与信噪比分布"),
            ],
            optional=True,
        ),
        DesignQuestion(
            key="hypothesis",
            question="你预期会看到什么？",
            why=(
                "先说出预期能暴露隐含假设。如果结果和预期一致但原因不对"
                "（比如信道恰好低秩，什么方法都表现好），事先想过就容易发现。"
            ),
            options=[
                Option("clear_win", "明显优于基线（3 dB 以上）", ""),
                Option("small_win", "小幅优于基线（1~2 dB）",
                       "差距小则样本数要够，建议 200 以上才有统计显著性"),
                Option("conditional", "特定条件下更好（低信噪比 / 高速等）",
                       "那这个条件维度就得扫，否则看不出来"),
                Option("unknown", "没有明确预期，先看看",
                       "探索性实验也是实验"),
            ],
            optional=True,
        ),
    )
}


# ---------------------------------------------------------------------------
# 仿真参数决策点
# ---------------------------------------------------------------------------

_CHANNEL_MODEL = Decision(
    key="channel_model",
    question="信道模型用哪个？",
    default="CDL-C",
    why=(
        "CDL-A/B/C 是非视距剖面，CDL-D/E 是视距剖面（K 因子 13.3/22 dB，"
        "信道近似低秩）。TDL 系列没有每条径的角度信息，依赖角度的算法不能用它。\n"
        "**重要**：视距与否由几何按 38.901 的 LOS 概率逐链路判定，"
        "不是由你选的剖面决定。选的剖面类别和判定结果不符时，仿真器会自动换成"
        "同类别的默认剖面（非视距→CDL-C，视距→CDL-D）——所以"
        "「选 CDL-D」不等于「得到视距信道」，在非视距判定下它实际会退成 CDL-C。"
        "想调视距比例要改场景和站间距（距离近则视距概率高），"
        "生成后 los_ratio 会报告实际比例。"
    ),
    options=[
        Option("CDL-C", "非视距 · 城区基准", "最常用的对比基线", recommended=True),
        Option("CDL-A", "非视距 · 强散射", "角度与时延扩展更大，适合与 CDL-C 做对照"),
        Option("CDL-D", "视距 · K=13.3 dB", "仅在链路判为视距时生效，见上文说明"),
        Option("TDL-C", "非视距 · 仅抽头", "无角度信息，不能做波束/定位"),
    ],
    priority=1,
)

_SNR = Decision(
    key="snr_range_dB",
    question="要不要限定信噪比范围？",
    default=None,
    why=(
        "信噪比不是可以直接设定的参数——它由路损、发射功率和噪声共同决定，"
        "取决于用户撒在哪里。默认不限定，让它自然分布（生成后会报告实际分布）。"
        "确实需要特定区间时会用拒绝采样实现，代价是变慢，"
        "区间偏离实际分布太远时可能一个都取不到。"
        "想整体抬高或压低信噪比，调发射功率或站间距比筛选更有效。"
    ),
    options=[
        Option(None, "不限定", "自然分布，最快", recommended=True),
        Option([-5.0, 5.0], "-5~5 dB · 只看小区边缘", "走拒绝采样，可能慢"),
        Option([25.0, 45.0], "25~45 dB · 只看近点", "走拒绝采样，可能慢"),
    ],
    priority=2,
)

_BANDWIDTH = Decision(
    key="bandwidth_hz",
    question="带宽多少？",
    default=100e6,
    why=(
        "带宽决定频域维度（100 MHz@30kHz 对应 273 个 RB），"
        "既是压缩率的分母，也决定时延分辨率——做定位/时延估计时，"
        "分辨率约等于光速除以带宽，100 MHz 对应 3 米。\n"
        "完整档位有 13 档（5/10/15/20/25/30/40/50/60/70/80/90/100 MHz），"
        "上面只列常用的，其余直接写数值即可。"
    ),
    options=[
        Option(100e6, "100 MHz", "273 RB @30kHz，n78 典型", recommended=True),
        Option(20e6, "20 MHz", "51 RB，跑得快，适合先验证流程"),
        Option(50e6, "50 MHz", "133 RB，折中"),
        Option(5e6, "5 MHz", "11 RB，窄带物联场景"),
    ],
    priority=3,
)

_SPEED = Decision(
    key="ue_speed_kmh",
    question="终端移动速度？",
    default=3.0,
    why=(
        "速度通过多普勒决定信道老化速度。静止/步行（3 km/h）时上下行互易性"
        "几乎不衰减；60 km/h 以上时 SRS 测量到实际使用之间信道已经变了，"
        "依赖互易性的方案会明显掉性能。"
    ),
    options=[
        Option(3.0, "3 km/h · 步行", "信道近似静止", recommended=True),
        Option(60.0, "60 km/h · 快速路", "信道老化开始显著"),
        Option(120.0, "120 km/h · 高速", "互易性明显退化"),
    ],
    priority=4,
)

_ANTENNA = Decision(
    key="antenna_preset",
    question="基站天线配置？",
    default="64T4R",
    why=(
        "端口数直接决定 CSI 的维度和码本大小。64 口按 8H4V 双极化解读，"
        "这是 3.5 GHz 宏站的主流配置；32 口及以下码本更小、搜索更快。"
    ),
    options=[
        Option("64T4R", "64 发 4 收", "8H4V 双极化，宏站主流", recommended=True),
        Option("32T4R", "32 发 4 收", "8H2V 双极化，码本更小搜索更快"),
        Option("4T4R", "4 发 4 收", "最小配置，先跑通流程用"),
    ],
    priority=3,
)

_EST_MODE = Decision(
    key="channel_est_mode",
    question="信道估计方式？",
    default="ls_linear",
    why=(
        "ideal 给理想信道，适合先验证算法上界；ls_linear 是带导频和噪声的"
        "实际估计。两者的差距就是估计误差对你算法的影响。"
        "注意理想信道会一并给出，所以选实际估计不会损失真值参照。"
    ),
    options=[
        Option("ls_linear", "LS + 线性插值", "贴近实际实现；理想信道会一并给出", recommended=True),
        Option("ls_mmse", "LS + MMSE", "更好的估计器"),
        Option("ideal", "理想信道", "无估计误差，只在算上界时用"),
    ],
    priority=2,
)

_PILOT = Decision(
    key="pilot_type",
    question="导频类型？",
    default="csi_rs_gold",
    why=(
        "下行用 CSI-RS（Gold 序列），上行用 SRS（ZC 序列）。做上行互易性、"
        "SRS 跳频相关的课题必须走 srs_zc，否则拿不到 SRS 侧的量。"
    ),
    options=[
        Option("csi_rs_gold", "CSI-RS · Gold 序列", "下行", recommended=True),
        Option("srs_zc", "SRS · ZC 序列", "上行，互易性相关课题用"),
    ],
    priority=4,
)

_LINK = Decision(
    key="link",
    question="上行还是下行？",
    default="DL",
    why="下行看 CSI 反馈与预编码，上行看 SRS 与互易性。做互易性对比需要成对的上下行。",
    options=[
        Option("DL", "下行", "看 CSI 反馈与预编码", recommended=True),
        Option("UL", "上行", ""),
        Option("both", "上下行成对", "互易性课题必选，数据量翻倍"),
    ],
    priority=4,
)

_NUM_SAMPLES = Decision(
    key="num_samples",
    question="生成多少个样本？",
    default=200,
    why=(
        "统计类结论（累积分布、平均性能）通常需要几百个样本；先跑 20 个"
        "验证流程通不通，再放大到几百个正式出图，能省不少时间。"
        "要做分布尾部分析（如 5% 边缘用户）则需要上千。"
    ),
    options=[
        Option(20, "20 · 先验证流程", "几秒出结果，确认配置对不对", recommended=True),
        Option(200, "200 · 正式实验", "够画分布图"),
        Option(1000, "1000 · 尾部统计", "看 5% 边缘用户时需要"),
    ],
    priority=6,
)

_SCENARIO = Decision(
    key="scenario",
    question="传播场景？",
    default="UMa_NLOS",
    why=(
        "城区宏站（UMa）站高 25 m、站间距 500 m；城区微站（UMi）站高 10 m、"
        "站距更小；室内工厂（InF）时延扩展与角度分布都明显不同。\n"
        "带 _LOS 后缀的版本改用视距路损公式——实测同一配置下 UMa_LOS 路损 83.9 dB、"
        "UMa_NLOS 116.3 dB，差 32 dB，是完全不同的两类信道。"
        "注意它改的是**路损模型**；每条链路的视距标志仍按 38.901 的 LOS 概率判定。"
    ),
    options=[
        Option("UMa_NLOS", "城区宏站 · 非视距", "38.901 基准，站高 25m 站距 500m", recommended=True),
        Option("UMa_LOS", "城区宏站 · 视距", "路损比非视距低约 30 dB，SINR 显著更高"),
        Option("UMi_NLOS", "城区微站 · 非视距", "密集城区，站高 10m 站距更小"),
        Option("UMi_LOS", "城区微站 · 视距", "密集城区视距"),
        Option("InF", "室内工厂", "38.901 §7.2，时延扩展与城区差别明显"),
    ],
    priority=3,
)

_NUM_SITES = Decision(
    key="num_sites",
    question="几个站点？",
    default=7,
    why=(
        "小区数 = 站数 × 扇区数。**六边形栅格的站数只能是 1/7/19**（按环数展开），"
        "配其他值会向上吸附——配 2 站实际会生成 7 站。需要精确站数（比如正好 6 小区）"
        "要用 topology_layout='linear' 或 custom_site_positions 指定坐标。\n"
        "单站看不到小区间干扰。7 站 21 小区是密集城区常用规模；"
        "19 站 57 小区是 38.901 Case 1 基准，中心小区的干扰环境最完整，"
        "但耗时约为 7 站的三倍。"
    ),
    options=[
        Option(7, "7 站 21 小区", "一圈邻区，干扰环境完整", recommended=True),
        Option(19, "19 站 57 小区", "两圈邻区，38.901 基准，耗时约三倍"),
        Option(1, "1 站 · 单小区", "无小区间干扰，干扰类课题不适用"),
    ],
    priority=1,
)

_UE_DIST = Decision(
    key="ue_distribution",
    question="用户怎么撒点？",
    default="uniform",
    why=(
        "均匀撒点是标准做法；热点分布会把用户往少数几处集中，边缘用户比例"
        "和干扰强度都上升，更能拉开干扰协调类算法的差距。撒点方式一换，"
        "SINR 分布整体平移，跨实验对比会失效。"
    ),
    options=[
        Option("uniform", "均匀分布", "标准做法", recommended=True),
        Option("hotspot", "热点分布", "边缘用户比例高，更能拉开干扰算法差距"),
        Option("clustered", "成簇分布", "若干簇中心"),
    ],
    priority=2,
)

_PRB_UTIL = Decision(
    key="prb_utilization",
    question="邻区负载率？",
    default=1.0,
    why=(
        "负载率决定邻区有多大比例的时频资源在发射。满载（1.0）是干扰最坏情况；"
        "轻载下干扰问题会被掩盖，容易得出「算法没用」的错误结论。"
    ),
    options=[
        Option(1.0, "满载", "干扰最坏情况", recommended=True),
        Option(0.7, "0.7 · 忙时典型", ""),
        Option(0.3, "0.3 · 轻载", "干扰弱，容易低估协调算法的价值"),
    ],
    priority=3,
)

_TDD = Decision(
    key="tdd_pattern",
    question="TDD 时隙配比？",
    default="DDDSU",
    why=(
        "配比决定上下行时隙比例，也决定 SRS 发送机会的间隔——"
        "间隔越长，用 SRS 推下行时信道老化越严重。"
    ),
    options=[
        Option("DDDSU", "DDDSU", "国内主流 2.5 ms 单周期", recommended=True),
        Option("DDSUU", "DDSUU", "上行更多，SRS 更密"),
        Option("DDDDDDDSUU", "DDDDDDDSUU", "下行为主，SRS 间隔长"),
        Option("DDDSUDDSUU", "DDDSUDDSUU", "5 ms 双周期"),
        Option("DSUUD", "DSUUD", "上行占比高"),
    ],
    priority=5,
)

_MOBILITY = Decision(
    key="mobility_mode",
    question="移动模型？",
    default="static",
    why=(
        "静止时每个样本独立；直线/随机游走会产生有时间相关性的轨迹，"
        "做切换、跟踪、信道预测类课题必须用它，否则样本间没有时序关系。"
    ),
    options=[
        Option("static", "静止", "样本间独立", recommended=True),
        Option("linear", "直线移动", "轨迹连续"),
        Option("random_walk", "随机游走", ""),
        Option("random_waypoint", "随机路点", ""),
    ],
    priority=5,
)

_NUM_SLOTS = Decision(
    key="num_slots_per_sample",
    question="每个样本取几个连续时隙？",
    default=1,
    why=(
        "默认 1 个时隙，样本之间没有时序关系。做信道预测、跟踪、老化分析"
        "必须取多个连续时隙——后续时隙由 Jakes 模型按多普勒演进，"
        "才有真实的时间相关性。数据量按时隙数线性增长。"
    ),
    options=[
        Option(14, "14 个", "覆盖一个 TDD 周期以上", recommended=True),
        Option(5, "5 个", "短时序，数据量小"),
        Option(1, "1 个", "单快照，无时序结构"),
    ],
    priority=2,
)


_SCS = Decision(
    key="subcarrier_spacing",
    question="子载波间隔？",
    default=30000,
    why=(
        "子载波间隔同时决定三件事：时隙长度（15/30/60/120 kHz 对应 1/0.5/0.25/0.125 ms）、"
        "同带宽下的 RB 数、以及抗多普勒能力。做高速移动或低时延课题时它是关键变量；"
        "间隔越大时隙越短、跟踪越快，但同带宽下频域分辨率越粗。"
    ),
    options=[
        Option(30000, "30 kHz", "n78 主流，时隙 0.5 ms", recommended=True),
        Option(15000, "15 kHz", "低频段，时隙 1 ms，频域分辨率最细"),
        Option(60000, "60 kHz", "时隙 0.25 ms，抗多普勒更好"),
        Option(120000, "120 kHz", "毫米波，时隙 0.125 ms"),
    ],
    priority=4,
)

_TOPOLOGY_LAYOUT = Decision(
    key="topology_layout",
    question="站点怎么布？",
    default="hexagonal",
    why=(
        "六边形栅格是 38.901 的标准蜂窝布局；线性布局把站点排成一条线，"
        "用于高铁、公路、隧道这类沿线覆盖场景——两者的干扰几何完全不同。"
    ),
    options=[
        Option("hexagonal", "六边形栅格", "标准蜂窝布局", recommended=True),
        Option("linear", "线性布站", "高铁 / 公路 / 隧道沿线覆盖"),
    ],
    priority=3,
)

_HYPERCELL = Decision(
    key="hypercell_size",
    question="几个小区合并成一个超级小区？",
    default=1,
    why=(
        "超级小区把 N 个物理小区合并成一个逻辑小区（共用 PCI），用户在其中移动不触发切换。"
        "高铁场景常用，代价是小区内干扰无法通过调度规避。1 表示不合并。"
    ),
    options=[
        Option(1, "不合并", "标准蜂窝", recommended=True),
        Option(3, "3 个合并", "常用规模"),
        Option(5, "5 个合并", "覆盖更长的沿线区段"),
    ],
    priority=4,
)

ALL_DECISIONS: dict[str, Decision] = {
    d.key: d
    for d in (
        _CHANNEL_MODEL, _SNR, _BANDWIDTH, _SPEED, _ANTENNA, _EST_MODE,
        _PILOT, _LINK, _NUM_SAMPLES, _SCENARIO, _NUM_SITES, _UE_DIST,
        _PRB_UTIL, _TDD, _MOBILITY, _NUM_SLOTS,
        _SCS, _TOPOLOGY_LAYOUT, _HYPERCELL,
    )
}


# ---------------------------------------------------------------------------
# 任务画像
# ---------------------------------------------------------------------------

_SCATTER_SWEEP = Sweep(
    key="channel_model",
    values=["CDL-A", "CDL-C"],
    label="强散射 vs 城区基准",
    why=(
        "两者都是非视距剖面但角度/时延扩展差异明显，能看出方法对散射丰富度是否敏感。"
        "注意别用 CDL-C vs CDL-D 做这个对比——视距与否由几何判定，"
        "非视距链路上选 CDL-D 会被自动退成 CDL-C，两组数据会完全相同。"
    ),
)

_LOS_SWEEP = Sweep(
    key="isd_m",
    values=[200.0, 800.0],
    label="近距（视距多）vs 远距（视距少）",
    why=(
        "视距比例是几何决定的，改站间距才能真正改变它——站距近则视距概率高。"
        "视距下信道近似低秩，很多方法会显得特别好，两种几何都跑结论才站得住。"
        "生成后看 los_ratio 确认实际比例。"
    ),
)

_SPEED_SWEEP = Sweep(
    key="ue_speed_kmh",
    values=[3.0, 60.0, 120.0],
    label="低速 / 中速 / 高速",
    why="信道老化随速度急剧变化，单一速度下的结论无法外推。",
)

_ANTENNA_SWEEP = Sweep(
    key="antenna_preset",
    values=["32T4R", "64T4R"],
    label="不同天线规模",
    why="维度变化会改变压缩率与码本搜索开销，只测一种规模难说方法可扩展。",
)

# **别用 prb_utilization 扫干扰强度。** 实测（42 样本、21 小区、num_ues=21）
# 0.2 与 1.0 两组的 SINR/SIR/IoT **逐位相同** —— 几何模型里调度 UE 数是
# max(1, round(num_ues/K x load))，每小区只有 1 个 UE 时恒为 1。
# 真正能改下行干扰强度的是站间距，实测 IoT：
#     ISD 100 m -> 38.3 dB，200 m -> 24.9 dB，500 m -> 4.4 dB，1732 m -> 0.2 dB
_LOAD_SWEEP = Sweep(
    key="isd_m",
    values=[150.0, 500.0],
    label="强干扰 vs 弱干扰（靠站间距）",
    why=(
        "站间距是这套几何模型里唯一能大范围改变干扰强度的旋钮，"
        "实测 IoT 从 38 dB 一路降到 4 dB。"
        "轻载下干扰问题被掩盖，容易低估干扰协调算法的价值——"
        "但**别用 prb_utilization 来造这个对比**：每小区 UE 数不足 2 时它完全无效，"
        "两组会生成一模一样的数据（sw_plan 的 guards 会拦）。"
        "生成后用 sw_interference_report 复核两组的 IoT 确实拉开了。"
    ),
)

_UL_LOAD_SWEEP = Sweep(
    key="pusch_load",
    values=[0.2, 1.0],
    label="上行轻载 vs 满载",
    why=(
        "上行干扰按同时发射的邻区 UE 数线性叠加，pusch_load 是有效旋钮——"
        "**但前提是每小区 UE 数 >= 2**。实测 21 小区 105 UE（每小区 5 个）时，"
        "满载让上行 IoT 从 1.8 dB 抬到 5.4 dB；21 UE（每小区 1 个）时毫无变化。"
    ),
)

TASK_PROFILES: tuple[TaskProfile, ...] = (
    TaskProfile(
        task="csi_compression",
        label="CSI 压缩 / 反馈",
        keywords=("csi", "压缩", "反馈", "feedback", "compress", "码本量化", "自编码", "csinet"),
        design_keys=("baseline", "metric", "scope"),
        decision_keys=("channel_model", "antenna_preset", "snr_range_dB", "bandwidth_hz", "num_samples"),
        sweeps=(_SCATTER_SWEEP, _LOS_SWEEP, _ANTENNA_SWEEP),
        pitfalls=(
            "只在视距场景验证会高估压缩率——视距信道本身就低秩，压什么都好压。",
            "压缩率要跟反馈开销对齐比较：比 Type II 码本省多少比特，而不是只看重建误差。",
            "只测高信噪比容易掩盖量化噪声与信道噪声的相互作用。",
        ),
        config_hints={"link": "DL", "channel_est_mode": "ls_linear"},
    ),
    TaskProfile(
        task="beam_management",
        label="波束管理 / 波束搜索",
        keywords=("波束", "beam", "赋形", "beamforming", "波束搜索", "扫描", "码本搜索", "波束跟踪"),
        design_keys=("baseline", "metric"),
        decision_keys=("channel_model", "antenna_preset", "ue_speed_kmh", "scenario", "num_samples"),
        sweeps=(_SCATTER_SWEEP, _LOS_SWEEP, _SPEED_SWEEP),
        pitfalls=(
            "TDL 模型没有每条径的角度，波束类算法在它上面跑出的结果没有物理意义。",
            "只用视距信道验证波束搜索，会严重高估命中率——非视距下最强径未必对着用户。",
            "搜索开销要跟穷举比较，只报命中率不报搜索次数说明不了问题。",
        ),
        config_hints={"link": "DL"},
        guards=("require_cdl",),
    ),
    TaskProfile(
        task="channel_estimation",
        label="信道估计 / 插值 / 降噪",
        keywords=("信道估计", "估计", "插值", "estimation", "interpolat", "降噪", "去噪", "denois"),
        design_keys=("baseline", "metric"),
        decision_keys=("channel_est_mode", "pilot_type", "snr_range_dB", "ue_speed_kmh", "channel_model"),
        sweeps=(_SPEED_SWEEP,),
        pitfalls=(
            "选 ideal 模式就没有估计误差可分析了——理想信道会作为真值一并给出，选实际估计不损失参照。",
            "低信噪比区间才是估计算法拉开差距的地方，只测高信噪比看不出优劣。",
            "导频密度和插值方法要一起说，否则 NMSE 不可比。",
        ),
        config_hints={},
    ),
    TaskProfile(
        task="positioning",
        label="定位 / 时延估计",
        keywords=("定位", "position", "toa", "时延估计", "测距", "localization", "tdoa", "aoa估计"),
        design_keys=("baseline", "metric", "scope"),
        decision_keys=("bandwidth_hz", "channel_model", "scenario", "num_sites", "num_samples"),
        sweeps=(_SCATTER_SWEEP, _LOS_SWEEP),
        pitfalls=(
            "时延分辨率约等于光速除以带宽——100 MHz 只有 3 米，定位精度先被带宽卡死。",
            "非视距下首径未必是直达径，视距比例直接决定定位误差分布，必须报告。",
            "单站只能测角，多站才能交汇定位。",
        ),
        config_hints={},
        guards=("require_cdl",),
    ),
    TaskProfile(
        task="precoding",
        label="预编码 / 码本设计",
        keywords=("预编码", "precod", "码本", "codebook", "pmi", "svd", "波束成形权", "mu-mimo配对"),
        design_keys=("baseline", "metric"),
        decision_keys=("channel_model", "antenna_preset", "snr_range_dB", "channel_est_mode", "num_samples"),
        sweeps=(_SCATTER_SWEEP, _LOS_SWEEP, _ANTENNA_SWEEP),
        pitfalls=(
            "拿理想信道算预编码增益是上界，实际系统用的是估计信道，两者差距可能很大。",
            "码本方案要跟 SVD 理想预编码比，才知道量化损失有多少。",
        ),
        config_hints={"link": "DL"},
    ),
    TaskProfile(
        task="interference",
        label="干扰协调 / 调度",
        keywords=("干扰", "interference", "协调", "调度", "schedul", "comp", "协作", "icic", "功控"),
        design_keys=("baseline", "metric", "scope"),
        decision_keys=("num_sites", "ue_distribution", "prb_utilization", "tdd_pattern", "num_samples"),
        sweeps=(_LOAD_SWEEP, _UL_LOAD_SWEEP),
        pitfalls=(
            "轻载场景下干扰问题被掩盖，容易得出「算法没用」的结论。",
            "边缘用户（5% 分位）才是干扰协调的目标，只看平均吞吐看不出效果。",
            "单小区场景没有干扰源，这类课题必须多站。",
            "**下行的 pdsch_load / prb_utilization 改不动干扰强度**：几何模型里每个"
            "邻区都无条件贡献泄漏，负载只改变对几个波束取平均。要造强弱干扰对比请动 "
            "isd_m（实测 IoT：100 m 38.3 dB / 200 m 24.9 / 500 m 4.4 / 1732 m 0.2）。",
            "上行负载是有效旋钮，但每小区 UE 数必须 >= 2，否则取整后恒为 1、完全无效。",
            "先用 sw_probe_scenario 确认 IoT 真的拉开了，再花时间跑正式对比。",
        ),
        # ChannelHub 默认不保存干扰小区的信道矩阵（只把干扰体现在 SINR 里），
        # 但干扰协调算法必须拿到干扰信道本身，所以这里显式打开。
        # 代价：数据量按干扰小区数翻倍。
        config_hints={"link": "DL", "measurements": {"interferer_channels": True}},
        guards=("require_multicell",),
    ),
    TaskProfile(
        task="mobility",
        label="移动性 / 切换",
        keywords=("切换", "handover", "移动性", "mobility", "轨迹", "重选", "乒乓"),
        design_keys=("baseline", "metric"),
        decision_keys=("ue_speed_kmh", "mobility_mode", "num_sites", "ue_distribution", "num_samples"),
        sweeps=(_SPEED_SWEEP,),
        pitfalls=(
            "静止模型下样本之间没有时序关系，切换判决无从谈起——必须选移动模型。",
            "乒乓切换要看连续轨迹，单快照数据分析不出来。",
        ),
        config_hints={"mobility_mode": "linear"},
        guards=("require_multicell", "require_trajectory"),
    ),
    TaskProfile(
        task="reciprocity",
        label="上下行互易性 / SRS 老化",
        keywords=("互易", "reciprocity", "srs", "老化", "aging", "上下行", "srs跳频"),
        design_keys=("baseline", "metric"),
        decision_keys=("ue_speed_kmh", "tdd_pattern", "channel_est_mode", "channel_model", "num_samples"),
        sweeps=(_SPEED_SWEEP,),
        pitfalls=(
            "速度低于 10 km/h 时信道几乎不老化，看不出任何差异。",
            "TDD 配比决定 SRS 间隔，是老化程度的另一个关键变量，别只扫速度。",
        ),
        config_hints={"link": "both", "pilot_type": "srs_zc"},
    ),
    TaskProfile(
        task="channel_prediction",
        label="信道预测 / 时序建模",
        keywords=("预测", "predict", "时序", "外推", "extrapolat", "信道跟踪", "lstm", "时间相关"),
        design_keys=("baseline", "metric", "scope"),
        decision_keys=("num_slots_per_sample", "ue_speed_kmh", "channel_model", "antenna_preset", "num_samples"),
        sweeps=(_SPEED_SWEEP,),
        pitfalls=(
            "默认每样本只有 1 个时隙，样本之间独立，根本没有时序可预测——必须调大连续时隙数。",
            "预测增益要跟「直接用上一时刻的信道」这个平凡基线比，否则容易自我感觉良好。",
            "多普勒决定可预测的时间跨度，速度不同结论完全不同。",
        ),
        config_hints={"num_slots_per_sample": 14, "mobility_mode": "linear"},
        guards=("require_multislot",),
    ),
    TaskProfile(
        task="link_adaptation",
        label="链路自适应 / CQI / MCS 选择",
        keywords=("链路自适应", "cqi", "mcs", "调制编码", "自适应", "link adaptation", "bler", "外环"),
        design_keys=("baseline", "metric"),
        decision_keys=("snr_range_dB", "channel_est_mode", "ue_speed_kmh", "prb_utilization", "num_samples"),
        sweeps=(_SPEED_SWEEP, _LOAD_SWEEP),
        pitfalls=(
            "CQI 上报到实际调度之间有延迟，静止场景下这个延迟无害，高速下才是主要误差源。",
            "只看平均吞吐会掩盖误块率的波动，两个指标要一起报。",
        ),
        config_hints={"link": "DL"},
    ),
    TaskProfile(
        task="channel_charting",
        label="信道表征 / 嵌入学习",
        keywords=("表征", "embedding", "charting", "嵌入", "自监督", "对比学习", "mae", "预训练"),
        design_keys=("baseline", "metric", "scope"),
        decision_keys=("channel_model", "antenna_preset", "scenario", "ue_distribution", "num_samples"),
        sweeps=(_SCATTER_SWEEP, _LOS_SWEEP),
        pitfalls=(
            "表征学习需要的样本量比常规仿真大一个量级，先估算好数据体积。",
            "如果要跟位置对齐评估（如信道图），必须保留 UE 坐标——geometry 里有。",
            "ChannelHub 本身有一套 16-token 特征是为 MAE 服务的，"
            "本项目给的是物理量，两者不要混用。",
        ),
        config_hints={},
    ),
    TaskProfile(
        task="generic",
        label="通用信道仿真",
        keywords=(),
        design_keys=("metric",),
        decision_keys=("scenario", "channel_model", "antenna_preset", "snr_range_dB", "num_samples"),
        sweeps=(),
        pitfalls=(),
        config_hints={},
    ),
)


def classify_intent(intent: str) -> TaskProfile:
    """从自然语言意图识别任务类型。命中最多关键词者胜，都不命中走通用。"""
    text = (intent or "").lower()
    best: TaskProfile | None = None
    best_hits = 0
    for prof in TASK_PROFILES:
        hits = sum(1 for kw in prof.keywords if kw in text)
        if hits > best_hits:
            best, best_hits = prof, hits
    return best or TASK_PROFILES[-1]


def decisions_for(profile: TaskProfile, *, limit: int = 6) -> list[Decision]:
    """取该任务应当询问的仿真参数，按优先级排序，最多 limit 个。"""
    picked = [ALL_DECISIONS[k] for k in profile.decision_keys if k in ALL_DECISIONS]
    picked.sort(key=lambda d: d.priority)
    return picked[:limit]


def design_questions_for(profile: TaskProfile, *, limit: int = 3) -> list[DesignQuestion]:
    """取实验设计层的问题。必答的排在前面。"""
    picked = [_DESIGN[k] for k in profile.design_keys if k in _DESIGN]
    picked.sort(key=lambda q: (q.optional,))
    return picked[:limit]


# ---------------------------------------------------------------------------
# 多轮提问
# ---------------------------------------------------------------------------
# superpowers 的 brainstorming 是「一次一个问题、问到你真正understand为止」。
# 我们的参数空间有限且已知，一次一个会拖到八九轮，所以改成
# **一轮 2~4 个问题、按主题分轮、答完再决定要不要下一轮**。
# 每轮都能直接生成——用户随时可以说「就这样吧」。

# 一轮最多问几个。从 4 提到 6 是因为**轮次比单轮长度更让人烦**：
# 6 个带选项的问题一次看完，比分三轮各 2~4 个来回三次要轻松。
_MAX_PER_ROUND = 6


def next_round(
    profile: TaskProfile,
    answered_design: set[str],
    answered_params: set[str],
    *,
    round_no: int = 1,
) -> dict[str, Any]:
    """根据已回答的内容决定这一轮问什么，以及还要不要继续问。

    **目标是 2 轮问完，最多 3 轮。**

      第 1 轮 —— 实验设计（跟什么比、看什么指标）**加上**高影响参数
                （信道模型、天线、拓扑这类一改结论就变的）。
      第 2 轮 —— 剩下的次要参数 + 可选的设计问题（推广范围、预期结果）。
      之后 —— 不再主动问，用户想调什么自己说。

    **设计层和参数层没有先后依赖**，早先把它们拆成两轮纯粹是分类整齐，
    对用户是白白多一次来回：选"跟 Type I 码本比"和选"用 CDL-C 还是
    CDL-A"互不影响，完全可以一次问完。所以这两类现在合并在第 1 轮，
    上限 ``_MAX_PER_ROUND``。

    真正**有依赖**的只有一处：样本数依赖试点方差，所以它压根不问用户，
    由 ``sw_sample_size`` 算（见 CONCLUSION_TEMPLATE 那一节）。

    返回的 ``can_generate`` 恒为 True：任何一轮之后都能直接生成，
    没答的走默认值。提问是为了让结论更可靠，不是准入门槛。
    """
    design_all = [_DESIGN[k] for k in profile.design_keys if k in _DESIGN]
    param_all = [ALL_DECISIONS[k] for k in profile.decision_keys if k in ALL_DECISIONS]
    param_all.sort(key=lambda d: d.priority)

    design_todo = [q for q in design_all if q.key not in answered_design and not q.optional]
    design_opt_todo = [q for q in design_all if q.key not in answered_design and q.optional]
    param_todo = [d for d in param_all if d.key not in answered_params]
    param_key = [d for d in param_todo if d.priority <= 2]
    param_rest = [d for d in param_todo if d.priority > 2]

    if round_no <= 1 and (design_todo or param_key):
        # 设计层优先占位（它错了整个结论作废），剩余名额给高影响参数。
        # 两类互不依赖，合在一轮问完。
        picked_d = design_todo[:_MAX_PER_ROUND]
        picked_p = param_key[: max(0, _MAX_PER_ROUND - len(picked_d))]
        focus = "实验设计 + 关键参数"
        rationale = (
            "基线和指标错了整个结论作废，这几个参数一改结论就变——"
            "两类互不依赖，一次问完。"
        )
    elif design_todo or param_key or design_opt_todo or param_rest:
        # 第 2 轮：把所有剩下的一次问完，不再往后拖。
        picked_d = (design_todo + design_opt_todo)[:_MAX_PER_ROUND]
        picked_p = (param_key + param_rest)[: max(0, _MAX_PER_ROUND - len(picked_d))]
        focus = "补齐剩余"
        rationale = "剩下的一次问完，答不上来的走默认值。"
    else:
        picked_d, picked_p = [], []
        focus = "已问完"
        rationale = "该确认的都确认了，可以生成。"

    remaining = (
        len(design_todo) + len(design_opt_todo) + len(param_todo)
        - len(picked_d) - len(picked_p)
    )
    return {
        "round": round_no,
        "focus": focus,
        "rationale": rationale,
        "design_questions": [q.as_dict() for q in picked_d],
        "questions": [d.as_dict() for d in picked_p],
        "has_more": remaining > 0,
        "remaining_count": max(remaining, 0),
        "can_generate": True,
        "target_rounds": "2 轮问完，最多 3 轮",
        # 剩下的问题是不是"全都有合理默认值"。为 True 时 Agent 应当把第 2 轮
        # 包装成一句可跳过的话（"剩下这些都有默认值，要不要直接跑？"），
        # 而不是又摆一屏选项——**必答的设计问题一个都不剩才算**。
        "remaining_all_optional": bool(
            not [q for q in design_todo if q.key not in {x.key for x in picked_d}]
            and not [d for d in param_key if d.key not in {x.key for x in picked_p}]
        ),
        "stop_hint": (
            "用户说「随便 / 默认就行 / 就这样」时立刻停止提问直接生成，"
            "不要再问下一轮。没答的项会用默认值，生成后会如实列出。"
        ),
    }


# ---------------------------------------------------------------------------
# 「缺什么就问什么」—— 由结论模板反推
# ---------------------------------------------------------------------------
# 上面的分轮是按主题走的。但真正决定该问什么的，是**最终那句结论还缺哪几个词**。
# 一次蒙特卡洛仿真的产出，说到底就是这么一句话：
#
#     在【场景】下，【方法】相对【基线】在【指标】上 【效应 ± 置信区间】（n 样本），
#     该结论在【扫描维度】上成立。
#
# 每个方括号是一个必须填的槽。槽空着 → 那句话就说不完整 → 该问。
# 反过来，能算的就别问：样本数由效应量和试点方差算出来（superpowers 那条
# "prevent agents from delegating prioritization back to humans" 的精神），
# 问用户"你要多少样本"是把该自己做的功课推回去。

CONCLUSION_TEMPLATE = (
    "在【{scenario}】下，【{method}】相对【{baseline}】在【{metric}】上"
    "{effect}（n={n}），该结论在【{scope}】上成立。"
)

# 槽位 → (是哪个设计问题 / 参数决策点在填它, 空着的后果)
_SLOTS: tuple[tuple[str, str, str, str], ...] = (
    ("baseline", "design", "baseline",
     "没有基线，「谱效 28 bit/s/Hz」这个数说明不了任何事"),
    ("metric", "design", "metric",
     "指标没定就不知道该留哪些量，也算不出需要多少样本"),
    ("effect", "design", "effect_size",
     "不知道要检出多大的效应，样本量就只能拍脑袋；样本不够时结论正负全凭运气"),
    ("csi_basis", "design", "csi_basis",
     "两臂 CSI 口径不一致时，增益里混着「偷看答案」的部分"),
    ("scenario", "param", "scenario",
     "场景决定路损与散射强度，换个场景结论可能反号"),
    ("scope", "design", "scope",
     "推广范围没定，就不知道要扫哪几个维度，结论只能限定在这一组配置上"),
)


def missing_slots(answered_design: set[str], answered_params: set[str]) -> list[dict[str, Any]]:
    """结论模板里还空着的槽，按"空着的代价"从大到小排。

    这是给 Agent 的提问依据：**问空着的槽，不问已经填上的**。
    """
    out = []
    for slot, kind, key, cost in _SLOTS:
        answered = key in (answered_design if kind == "design" else answered_params)
        if answered:
            continue
        q = _DESIGN.get(key) if kind == "design" else ALL_DECISIONS.get(key)
        out.append(
            {
                "slot": slot,
                "kind": kind,
                "key": key,
                "cost_if_empty": cost,
                "question": q.question if q else "",
                "options": [o.as_dict() for o in q.options] if q else [],
            }
        )
    return out


def sample_size_advice(
    std_diff: float | None = None,
    expected_effect: float | None = None,
    *,
    n_current: int | None = None,
) -> dict[str, Any]:
    """样本数建议。**算出来的，不是问出来的。**

    三种输入组合：

    * 有 σ_d 和期望效应 → 直接给需要的 N；
    * 有 σ_d 和当前 N   → 给这个实验的最小可检出效应；
    * 什么都没有       → 给试点流程（先跑 20 个量方差）。
    """
    from .gates import detectable_effect, required_samples

    if std_diff is not None and expected_effect:
        need = required_samples(std_diff, expected_effect)
        return {
            "mode": "由效应量定样本数",
            "std_diff": round(float(std_diff), 4),
            "expected_effect": float(expected_effect),
            "required_n": need,
            "note": (
                f"α=0.05、功效 80% 下需要 {need} 个样本。"
                f"这是**配对**设计的样本数（两个方案跑在同一批信道上）；"
                f"若两臂用不同数据集，需求会高得多。"
            ),
        }
    if std_diff is not None and n_current:
        mde = detectable_effect(std_diff, int(n_current))
        return {
            "mode": "由样本数定可检出效应",
            "std_diff": round(float(std_diff), 4),
            "n": int(n_current),
            "min_detectable_effect": round(mde, 4),
            "note": (
                f"{n_current} 个样本最小能可靠检出 {mde:.4f} 的差异。"
                f"比这更小的差异，本实验分辨不出来——**这时候结果无论正负都不该下结论**。"
            ),
        }
    return {
        "mode": "先做试点",
        "pilot_n": 20,
        "note": (
            "先用 20 个样本把两个方案都跑一遍，拿逐样本差值的标准差 σ_d，"
            "再回来算正式样本数。直接猜样本数是这类实验最常见的浪费来源："
            "猜少了结论不成立，猜多了白等几小时。"
        ),
    }


def also_configurable(profile: TaskProfile) -> list[str]:
    """没被问到、但用户可以主动提的参数——只给名字，不展开。

    这解决的是「你不知道你能要什么」：不列出来用户想不到还能调移动速度，
    全展开又会淹没那几个真正关键的。
    """
    asked = set(profile.decision_keys)
    labels = {
        "scenario": "传播场景", "channel_model": "信道模型", "antenna_preset": "天线配置",
        "snr_range_dB": "信噪比范围", "bandwidth_hz": "带宽", "ue_speed_kmh": "移动速度",
        "channel_est_mode": "信道估计方式", "pilot_type": "导频类型", "link": "上下行",
        "tdd_pattern": "TDD配比", "num_sites": "站点数", "ue_distribution": "撒点方式",
        "prb_utilization": "邻区负载", "mobility_mode": "移动模型", "num_samples": "样本数",
        "num_slots_per_sample": "连续时隙数", "subcarrier_spacing": "子载波间隔",
        "topology_layout": "站点布局", "hypercell_size": "超级小区规模",
    }
    extra = [
        "载波频率", "终端天线数", "用户数", "站间距", "发射功率",
        "噪声系数", "SRS跳频", "干扰用户数", "随机种子", "扇区数",
        "多TRP联合", "高铁穿透损耗", "轨道偏移", "自定义站点坐标", "自定义用户坐标",
    ]
    return [v for k, v in labels.items() if k not in asked] + extra


def sweep_suggestions(profile: TaskProfile) -> list[dict[str, Any]]:
    """建议的对比维度。用户想让结论站得住时，一次生成多组比事后补跑省事。"""
    return [s.as_dict() for s in profile.sweeps]


# ---------------------------------------------------------------------------
# 体检拦截
# ---------------------------------------------------------------------------


def check_guards(profile: TaskProfile, cfg: dict[str, Any]) -> list[dict[str, str]]:
    """检查参数组合在物理上是否讲得通。

    返回问题清单；空列表表示没问题。这里拦的是「跑得出结果但结果没意义」
    的组合——比报错更危险，因为数值看起来完全正常。
    """
    issues: list[dict[str, str]] = []

    model = str(cfg.get("channel_model", "")).upper()
    if "require_cdl" in profile.guards and model.startswith("TDL"):
        issues.append(
            {
                "severity": "block",
                "key": "channel_model",
                "message": (
                    f"{profile.label} 依赖每条径的角度信息，但 TDL 模型不含角度"
                    "（ChannelHub 内部即以此区分两类模型）。算法会跑出结果，"
                    "但那些结果没有物理意义，而且不会报错。"
                ),
                "suggestion": "改用 CDL-A~E 中的任一个（默认 CDL-C）",
            }
        )

    n_sites = int(cfg.get("num_sites", 1) or 1)
    if "require_multicell" in profile.guards and n_sites <= 1:
        issues.append(
            {
                "severity": "block",
                "key": "num_sites",
                "message": f"{profile.label} 需要小区间干扰，单站场景下没有干扰源。",
                "suggestion": "站点数设为 7（21 小区）或 19（57 小区）",
            }
        )

    n_slots = int(cfg.get("num_slots_per_sample", 1) or 1)
    if "require_multislot" in profile.guards and n_slots <= 1:
        issues.append(
            {
                "severity": "block",
                "key": "num_slots_per_sample",
                "message": (
                    f"{profile.label} 需要样本内部有时间相关性，"
                    "但每样本只取 1 个时隙时各样本相互独立，没有可预测的时序结构。"
                ),
                "suggestion": "连续时隙数设为 14（覆盖一个 TDD 周期以上）",
            }
        )

    mob = str(cfg.get("mobility_mode", "static"))
    if "require_trajectory" in profile.guards and mob == "static":
        issues.append(
            {
                "severity": "warn",
                "key": "mobility_mode",
                "message": "静止模型下用户不移动，切换/轨迹类分析没有对象。",
                "suggestion": "改用 linear 或 random_waypoint",
            }
        )

    speed = float(cfg.get("ue_speed_kmh", 3.0) or 3.0)
    if profile.task in ("reciprocity", "channel_prediction") and speed < 10.0:
        issues.append(
            {
                "severity": "warn",
                "key": "ue_speed_kmh",
                "message": f"速度 {speed:g} km/h 下信道几乎不老化，{profile.label} 看不出差异。",
                "suggestion": "建议 60 km/h 以上，或扫一组速度做对比",
            }
        )

    bw = float(cfg.get("bandwidth_hz", 100e6) or 100e6)
    if profile.task == "positioning" and bw < 50e6:
        issues.append(
            {
                "severity": "warn",
                "key": "bandwidth_hz",
                "message": (
                    f"带宽 {bw/1e6:.0f} MHz 对应时延分辨率约 "
                    f"{3e8/bw:.1f} 米，定位精度会被带宽卡住。"
                ),
                "suggestion": "定位类课题建议 100 MHz 以上",
            }
        )

    est = str(cfg.get("channel_est_mode", ""))
    if profile.task == "channel_estimation" and est == "ideal":
        issues.append(
            {
                "severity": "warn",
                "key": "channel_est_mode",
                "message": "理想信道没有估计误差，信道估计算法无从对比。",
                "suggestion": "改用 ls_linear 或 ls_mmse；理想信道会作为真值一并给出",
            }
        )

    if profile.task == "reciprocity" and str(cfg.get("link", "DL")) != "both":
        issues.append(
            {
                "severity": "warn",
                "key": "link",
                "message": "互易性课题需要成对的上下行信道，只取单向无法对比。",
                "suggestion": 'link 设为 "both"',
            }
        )

    # 负载类旋钮在小区内 UE 数不足时**完全不起作用**。
    # ChannelHub 的几何 SINR 里调度 UE 数是 max(1, round(num_ues/K x load))，
    # num_ues/K == 1 时不管 load 取 0.2 还是 1.0 都等于 1。实测（42 样本，
    # 21 小区 21 UE）两个极端的 SINR/SIR/IoT **逐位相同**。
    # 不拦的话，用户会拿两批一模一样的数据做"轻载 vs 满载"对比，
    # 得出"负载不影响性能"——这个错误结论完全是工具造成的。
    n_cells = n_sites * int(cfg.get("sectors_per_site", 1) or 1)
    load_keys = [k for k in ("prb_utilization", "pusch_load", "pdsch_load") if k in cfg]
    if load_keys and n_cells > 1:
        per_cell = int(cfg.get("num_ues", 1) or 1) // max(n_cells, 1)
        if per_cell < 2:
            issues.append(
                {
                    "severity": "warn",
                    "key": load_keys[0],
                    "message": (
                        f"设了 {'/'.join(load_keys)}，但每小区只有 {per_cell} 个 UE"
                        f"（num_ues={cfg.get('num_ues')} / {n_cells} 小区）。"
                        "调度 UE 数是 max(1, 每小区UE数 x 负载)，取整后恒为 1，"
                        "**这个旋钮在当前配置下没有任何效果**——不同负载会生成完全相同的数据。"
                    ),
                    "suggestion": f"num_ues 至少设成 {n_cells * 4}（每小区 4 个 UE）再调负载",
                }
            )

    # 下行负载在这套几何模型里**始终**不改变期望干扰功率：每个非服务小区
    # 都无条件贡献一份泄漏，负载只改变对多少个波束取平均（均值不变、方差变小）。
    if "pdsch_load" in cfg and str(cfg.get("link", "DL")).upper() == "DL":
        issues.append(
            {
                "severity": "warn",
                "key": "pdsch_load",
                "message": (
                    "下行几何模型里每个邻区都无条件贡献干扰，pdsch_load 只影响"
                    "对几个波束取平均，**不改变干扰功率的期望**。想调下行干扰强度请动"
                    "isd_m（实测 100/200/500/1732 m 对应 IoT 38.3/24.9/4.4/0.2 dB）、"
                    "num_sites、tx_power_dbm 或 noise_figure_db。"
                ),
                "suggestion": "用 sw_design_interference 看各旋钮的实测作用方向",
            }
        )

    return issues
