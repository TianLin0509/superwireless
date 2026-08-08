"""系统级仿真的随机数体系：按用途分流、重复实验、公共随机数、置信区间。

**为什么要有这个模块（实测证据）。** 同一批信道、同一套配置，只改 ``seed``
跑 8 次 ``sw_system_sim``（数据在 ``measurements/seed_variance.json``）：

===========================  ========  ==========  ================
KPI                          均值      变异系数    极差
===========================  ========  ==========  ================
``cell_experienced_mbps``    152.1     **11.4%**   123.7 ~ 184.6
``ue_experienced_p5_mbps``   53.6      **13.5%**   20.6
``avg_mcs``                  5.45      9.2%        1.31
===========================  ========  ==========  ================

单次运行报出来的"体验速率 142.3 Mbps"，小数点后那位是假的。**已经发生过把
噪声当效应的事故**：上一轮报"Type I 权老化后 +14%"，而噪声 1σ 就有 11.4%。

链路级早就有完整的三道门（配对 t / Wilcoxon / 置信区间 / 预注册，见
:mod:`gates` 与 :mod:`results`），**系统级完全绕过了**——在 ``system.py`` 里
搜 ``confidence|t_test|wilcoxon|paired`` 命中数是 0。这个模块补的就是这个不一致，
**统计一律复用 gates.py，不另写一套**。

调研（三个成熟实现怎么做）
--------------------------

**ns-3**（https://www.nsnam.org/docs/manual/html/random-variables.html）
用 L'Ecuyer 的 MRG32k3a，分成两个独立旋钮：``RngSeed`` 与 ``RngRun``。手册的原话是
"There is no guarantee that the streams produced by two random seeds will not overlap.
The only way to guarantee that two streams do not overlap is to use the substream
capability provided by the RNG implementation. **Therefore, use the substream
capability to produce multiple independent runs of the same scenario**"，并明确
"the more statistically rigorous way to configure multiple independent replications
is to use a fixed seed and to advance the run number"。
**采纳**：本模块的 ``master_seed`` 对应 ``RngSeed``（换它 = 换一个宇宙），
``replication`` 对应 ``RngRun``（同一宇宙里的第 k 次独立重复）。

**NumPy**（https://numpy.org/doc/stable/reference/random/parallel.html）
文档把 ``worker_seed = root_seed + worker_id`` 明确标成 "UNSAFE! Do not do this!"，
理由不是"相邻种子相关"，而是**跨批次撞车**："it is quite likely that multiple
invocations of the program with different seeds will get overlapping sets of worker
seeds"。``SeedSequence`` 用带雪崩效应的整数散列把 entropy 与 ``spawn_key`` 混合，
"Two input seeds that are very close to each other will produce initial states that
are very far from each other"。
**采纳 spawn 机制，但用显式 ``spawn_key`` 而不是有状态的 ``.spawn()``**——理由见
:func:`RngBook.seed_sequence`。

**Sionna**（https://nvlabs.github.io/sionna/phy/api/config.html）
``sionna.phy.config.seed`` 是一个**全局**种子，一次设好 Python / NumPy / torch
三套 RNG；同时很多层（如交织器）自己带一个可选的 ``seed`` 参数，"If explicitly
given, the global internal seed is replaced by this seed"。
**部分采纳**：全局默认 + 局部可覆盖这个两级结构是对的，但 Sionna 的局部种子是
**扁平的裸整数**（本机装的 ``sionna.rt`` 里甚至能看到
``radio_map_solver.py:411`` 写着 ``self._sampler.seed(seed + 1, num_samples)``
——正是 NumPy 文档点名的那个反模式）。本模块改成从同一个 ``master_seed``
**派生**，而不是让调用方各拍一个整数。

**Common Random Numbers（公共随机数，CRN）** 是经典的方差缩减技术：比较两个方案时
"use the same pseudo-random numbers in exactly the same ways for corresponding runs
of each of the competing systems"，目的是让观测到的差异归因于方案本身而不是随机
波动。配对估计量的方差比独立估计量少了 ``2·Cov(a,b)``。
**采纳**：A/B 两臂用**同一批** replication 流，这正是链路级
``results.check_pairable`` 那套配对思想在系统级的对应物。

分流表
------

===================  =============================================================
流名                 管什么
===================  =============================================================
``channel``          信道生成与撒点（ChannelHub 的 ``seed``，见
                     :func:`derive_generate_seed`）
``traffic``          话务到达：FTP3 泊松到达、bimodal 的 RBG 尺寸抽样
``scheduler``        调度器决胜：PF/max-C/I 度量打平时的随机打破平局
``harq``             HARQ 误码抽样：首传与重传的 ACK/NACK 伯努利
``neighbor_load``    邻区 PRB 利用率的逐快照抖动
===================  =============================================================

**分流的好处非常具体**：改话务模型不会连带改变信道实现，A/B 才是受控的。
分流之前 ``simulate()`` 里**一个** ``rng`` 同时喂话务和 HARQ——改一下
``arrival_rate_hz``，抽到的到达次数变了，后面 HARQ 的伯努利序列**整个错位**，
于是"话务模型的影响"里混着"HARQ 换了一批随机数"。这类污染在结果里看不出来。

实测（``measurements/rng_replication.json``，ds_6e9715bc、ftp3、5 s）
--------------------------------------------------------------------

**置信区间随 n 收窄**（从 64 次重复里重抽 500 次取平均半宽，消掉单次实现的
抽样噪声；总体变异系数 9.4%）::

    n        2      4      6      8     12     16     32
    半宽/均值  60.9%  13.7%   9.4%   7.6%   5.8%   5.0%   3.4%

注意"按 1/√n 收窄"精确成立的是**标准误**；置信区间半宽还乘着 ``t_{0.975,n-1}``，
小 n 时 t 大得多（``t₃=3.18`` vs ``t₁₅=2.13``），所以半宽实际收得**比 1/√n 更快**
（n 4→16 是 0.357 而不是 0.5）。混为一谈会差 30%。

**公共随机数的收益**（A/B：PF 窗 100 vs 1000，n_rep=8，真实效应约 −10 Mbps）::

                     效应        95% CI                半宽    Wilcoxon p   判决
    CRN            −10.64  [−14.14, −7.15]            3.49      0.0078   significant
    独立随机数      −14.97  [−28.66, −1.27]           13.69      0.078    inconclusive

**同一个真实效应，CRN 下判得出来，独立种子下判不出来**：区间窄 **3.92 倍**，
差值标准差 4.18 vs 16.38。顺带一提独立那一栏的区间其实不跨零，
但判决以 Wilcoxon 为准（``gates.PairedResult``），照样拦住了——
这正是 CLAUDE.md「门 3 的判决必须显式说清用哪个检验」那条要求的效果。
"""
from __future__ import annotations

import math
import zlib
from dataclasses import dataclass, field
from typing import Any

import numpy as np

_EPS = 1e-12

#: 内置的随机流登记表。**名字 → 用途**，改这里等于改现场约定。
#: 新增流用 :func:`register_stream`，不要直接改这个字典的顺序——
#: 流的派生键是**名字的 CRC32**而不是它在表里的位置，所以增删不会
#: 让已有流的随机序列漂移（这一点是有意设计的，见 :func:`stream_key`）。
STREAMS: dict[str, str] = {
    "channel": "信道生成与撒点（ChannelHub 的 seed）",
    "traffic": "话务到达：FTP3 泊松到达、bimodal 的 RBG 尺寸抽样",
    "scheduler": "调度器决胜：度量打平时的随机打破平局",
    "harq": "HARQ 误码抽样：首传与重传的 ACK/NACK 伯努利",
    "neighbor_load": "邻区 PRB 利用率的逐快照抖动",
}


def register_stream(name: str, purpose: str) -> None:
    """登记一个新的随机流。**加流不会扰动已有流**（派生键来自名字的散列）。"""
    n = str(name).strip()
    if not n:
        raise ValueError("流名不能为空")
    if n in STREAMS and STREAMS[n] != purpose:
        raise ValueError(f"流 {n!r} 已登记为 {STREAMS[n]!r}，不要改用途——"
                         f"改用途等于悄悄换掉一批随机数")
    STREAMS[n] = str(purpose)


def stream_key(name: str) -> int:
    """流名 → ``spawn_key`` 里用的 32 位整数。

    用 ``zlib.crc32`` 而**不是** Python 内置的 ``hash()``：后者对 str 是
    **每进程随机加盐**的（PYTHONHASHSEED），拿它当种子会让"同一个 master seed
    完全可复现"这条直接失效——而且只在换进程时才暴露，本进程内自洽，
    是最难查的一类不可复现。

    也不用"名字在表里的下标"：那样加一个流就会把它后面所有流的随机序列
    整体挪位，一次无关的重构就能让历史结果对不上。
    """
    return int(zlib.crc32(str(name).encode("utf-8")))


def _known(name: str) -> str:
    if name not in STREAMS:
        raise ValueError(
            f"未登记的随机流 {name!r}。已登记：{sorted(STREAMS)}。"
            f"**拼错流名不会报错、只会悄悄多出一条独立的随机流**，"
            f"所以这里是硬拦截；确实要加新流请先调 register_stream()")
    return name


# ---------------------------------------------------------------------------
# 两级种子：master_seed（ns-3 的 RngSeed）+ replication（ns-3 的 RngRun）
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class RngBook:
    """一次运行的**全套**随机流。不可变，因此可以放心到处传。

    ``master_seed`` 标识整个实验批次（对应 ns-3 的 ``RngSeed``），
    ``replication`` 标识这个批次里的第几次独立重复（对应 ns-3 的 ``RngRun``）。

    **重复实验换 replication，不换 master_seed。** ns-3 手册的理由是两个随机
    种子产生的流没有任何"不重叠"保证，只有子流机制才有；NumPy 文档给的理由是
    ``root_seed + k`` 这种写法在换一次 root 之后会和上一批**撞车**。
    :mod:`tests.test_rng` 第 3 节把后者实测出来了：base=100 与 base=105 各取 8 个
    ``default_rng(base+i)``，有 **3 对流逐位相同**——不是"相关"，是**同一条流**。
    """

    master_seed: int = 0
    replication: int = 0

    # ---- 派生 ------------------------------------------------------------
    def seed_sequence(self, stream: str) -> np.random.SeedSequence:
        """派生某条流的 ``SeedSequence``。

        用**显式 spawn_key** 而不是 ``SeedSequence(master).spawn(k)``，
        虽然两者的底层机制完全是同一个（numpy 的 ``spawn()`` 就是构造
        ``SeedSequence(entropy, spawn_key=self.spawn_key + (i,))``，
        ``tests/test_rng.py`` 第 1 节逐位验证过这条等价性）。差别在于：

        * ``.spawn()`` 是**有状态**的——``n_children_spawned`` 会随调用推进。
          先要 ``traffic`` 还是先要 ``harq``，拿到的流就不一样。一次无关的
          代码顺序调整就能悄悄换掉全部随机数。
        * 显式 key 是**纯函数**：``(master, replication, 流名)`` 一定映射到同一条流，
          与调用顺序、与有没有别的流被创建过都无关。

        key 的层级顺序是 ``(流, 重复)`` 而不是 ``(重复, 流)``，对齐 ns-3 的结构：
        每个 ``RandomVariableStream`` 有自己的 stream，``RngRun`` 在**每条
        stream 内部**选子流。
        """
        return np.random.SeedSequence(
            entropy=int(self.master_seed),
            spawn_key=(stream_key(_known(stream)), int(self.replication)),
        )

    def generator(self, stream: str) -> np.random.Generator:
        """派生某条流的 ``Generator``（PCG64）。同参数调用两次给出等价的两个流。"""
        return np.random.default_rng(self.seed_sequence(stream))

    def integer_seed(self, stream: str) -> int:
        """把某条流折成一个 63 位整数，喂给只肯收 ``int`` 的外部接口。

        ChannelHub 的 ``seed``、以及任何"只有一个整数种子"的库都走这里，
        **不要在调用处写 ``master_seed + 常数``**——那就是 NumPy 文档点名的
        反模式，换一次 master 就可能和别的流撞上。
        """
        return int(self.seed_sequence(stream).generate_state(2, dtype=np.uint64)[0]
                   >> np.uint64(1))

    # ---- 记账 ------------------------------------------------------------
    def as_dict(self) -> dict[str, Any]:
        return {
            "master_seed": int(self.master_seed),
            "replication": int(self.replication),
            "streams": sorted(STREAMS),
            "scheme": "numpy.SeedSequence(entropy=master_seed, "
                      "spawn_key=(crc32(stream), replication))",
            "note": ("重复实验换 replication 不换 master_seed（对应 ns-3 的 "
                     "RngRun 而不是 RngSeed）；A/B 两臂共用同一批 replication "
                     "即公共随机数（CRN）。"),
        }


def replications(master_seed: int = 0, n: int = 8, *, start: int = 0) -> list[RngBook]:
    """一批独立重复。``[RngBook(master, start), ..., RngBook(master, start+n-1)]``。

    **A/B 两臂拿同一个返回值就是公共随机数**——两边第 k 次重复用的是同一套
    话务到达、同一套 HARQ 抽签，差值里只剩算法本身的差别。
    """
    if int(n) < 1:
        raise ValueError(f"重复次数至少 1 次，收到 {n}")
    return [RngBook(int(master_seed), int(start) + i) for i in range(int(n))]


def derive_generate_seed(master_seed: int = 0, replication: int = 0) -> int:
    """给 ``sw_generate`` 用的 ChannelHub 种子，从 ``channel`` 流派生。

    **信道流和话务流分开的价值就在这**：改话务参数不会连带换掉信道实现，
    否则 A/B 比的是"算法差异 + 换了一批信道"。ChannelHub 只认一个 int，
    所以这里折成整数，但派生路径仍然是同一套 ``SeedSequence``。
    """
    return RngBook(int(master_seed), int(replication)).integer_seed("channel")


# ---------------------------------------------------------------------------
# KPI 的点估计 + 置信区间
# ---------------------------------------------------------------------------
@dataclass
class KpiStat:
    """一个 KPI 在 n 次重复上的分布。**所有系统级 KPI 都得长这样。**"""

    name: str
    values: list[float] = field(default_factory=list)

    @property
    def n_rep(self) -> int:
        return int(len(self._finite))

    @property
    def _finite(self) -> np.ndarray:
        v = np.asarray(self.values, dtype=float)
        return v[np.isfinite(v)]

    @property
    def mean(self) -> float:
        v = self._finite
        return float(v.mean()) if v.size else float("nan")

    @property
    def std(self) -> float:
        """**样本**标准差（ddof=1）。n=1 时是 nan，不是 0——一次运行测不出离散度。"""
        v = self._finite
        return float(v.std(ddof=1)) if v.size >= 2 else float("nan")

    @property
    def sem(self) -> float:
        return self.std / math.sqrt(self.n_rep) if self.n_rep >= 2 else float("nan")

    @property
    def half_width(self) -> float:
        """95% 置信区间的半宽 ``t_{0.975,n-1}·s/√n``。

        用 t 而不是 z：n=8 时 ``t=2.365`` 比 ``z=1.96`` 宽 21%，
        **小样本上用 z 会系统性地把区间报窄**。
        """
        if self.n_rep < 2:
            return float("nan")
        from scipy import stats  # noqa: PLC0415

        return float(stats.t.ppf(0.975, self.n_rep - 1)) * self.sem

    @property
    def ci95(self) -> tuple[float, float]:
        h = self.half_width
        return (self.mean - h, self.mean + h)

    @property
    def cv(self) -> float:
        """变异系数 ``s/|mean|``。跨 KPI 比较离散度时用它，量纲无关。"""
        m = abs(self.mean)
        return self.std / m if m > _EPS else float("nan")

    @property
    def rel_half_width(self) -> float:
        """置信区间半宽占均值的比例。**报数字前先看它**——它就是最后一位的可信度。"""
        m = abs(self.mean)
        return self.half_width / m if m > _EPS else float("nan")

    def as_dict(self) -> dict[str, Any]:
        lo, hi = self.ci95
        d: dict[str, Any] = {
            "mean": _r(self.mean), "std": _r(self.std),
            "ci95": [_r(lo), _r(hi)], "n_rep": self.n_rep,
            "cv": _r(self.cv), "rel_half_width": _r(self.rel_half_width),
            "min": _r(float(self._finite.min()) if self.n_rep else float("nan")),
            "max": _r(float(self._finite.max()) if self.n_rep else float("nan")),
        }
        if self.n_rep < 2:
            d["note"] = ("**单次运行没有置信区间**。系统级 KPI 的种子间变异系数"
                         "实测 9.4%（cell_experienced_mbps，64 次重复；"
                         "早先 8 个种子测出的 11.4% 与它统计上一致），"
                         "一次运行的数字最后一位是假的——把 num_replications 调到 ≥6。")
        return d


def _r(x: float, nd: int = 4) -> float | None:
    """JSON 里不要 nan/inf——它们不是合法 JSON，各家解析器行为还不一样。"""
    return round(float(x), nd) if np.isfinite(x) else None


def summarize(values: Any, name: str = "") -> KpiStat:
    """把一列重复值收成 :class:`KpiStat`。"""
    return KpiStat(name=str(name), values=[float(x) for x in np.asarray(values).ravel()])


def summarize_runs(runs: list[dict[str, Any]], *,
                   keys: list[str] | None = None) -> dict[str, dict[str, Any]]:
    """把 n 次重复的 ``cell`` 字典收成 ``{KPI: {mean/std/ci95/n_rep}}``。

    只收数值型字段——``rbg_size_hist`` 这类嵌套结构、``notes`` 这类文本
    原样跳过（它们不是可以求均值的量）。
    """
    if not runs:
        return {}
    if keys is None:
        keys = [k for k, v in runs[0].items()
                if isinstance(v, (int, float)) and not isinstance(v, bool)]
    out: dict[str, dict[str, Any]] = {}
    for k in keys:
        vals = [float(r[k]) for r in runs
                if isinstance(r.get(k), (int, float)) and not isinstance(r.get(k), bool)]
        if vals:
            out[k] = summarize(vals, k).as_dict()
    return out


# ---------------------------------------------------------------------------
# A/B 比较：公共随机数 + 复用 gates.py 的配对检验
# ---------------------------------------------------------------------------
def check_pairable(books_a: list[RngBook], books_b: list[RngBook]) -> list[dict[str, Any]]:
    """两臂能不能做配对（CRN）比较。返回**硬拦截**列表，空表示可以。

    照抄 ``results.check_pairable`` 的思路：**配对的有效性靠 ID 契约，统计查不出错位。**
    CLAUDE.md 里那条血泪是实测出来的——把两臂的样本 ID 顺序错开一位，配对检验算出的
    p 值可以**一模一样**（1.63e-11 → 1.63e-11），因为统计只看数值数组，
    根本不知道第 i 个数对应哪一次重复。

    系统级的"样本 ID"就是 ``(master_seed, replication)``：两臂第 k 次重复必须是
    同一套话务到达与 HARQ 抽签，差值才只反映算法差异。
    """
    issues: list[dict[str, Any]] = []
    if len(books_a) != len(books_b):
        issues.append({
            "check": "重复次数一致",
            "detail": f"A 臂 {len(books_a)} 次、B 臂 {len(books_b)} 次",
            "fix": "两臂用同一个 rng.replications(...) 的返回值",
        })
        return issues
    ka = [(b.master_seed, b.replication) for b in books_a]
    kb = [(b.master_seed, b.replication) for b in books_b]
    if ka != kb:
        first = next((i for i, (x, y) in enumerate(zip(ka, kb, strict=True)) if x != y), None)
        same_set = sorted(ka) == sorted(kb)
        issues.append({
            "check": "两臂用公共随机数（CRN）",
            "detail": (f"第 {first} 次重复的种子不同（{ka[first]} vs {kb[first]}）"
                       + ("；集合相同，只是顺序被打乱了——这种错位统计层面"
                          "**完全不可观测**" if same_set
                          else "；两臂跑的根本不是同一批随机数")),
            "fix": ("两臂都用同一个 rng.replications(master_seed, n) 的返回值。"
                    "独立随机数也能算，但置信区间会明显更宽——**实测宽 3.9 倍**"
                    "（同一个真实效应，CRN 下判显著、独立种子下判不出来），"
                    "见 measurements/rng_replication.json 与 tests/test_rng.py 第 6 节"),
        })
    return issues


def compare_replications(
    values_a: Any,
    values_b: Any,
    *,
    metric: str = "cell_experienced_mbps",
    unit: str = "Mbps",
    arm_a: str = "A",
    arm_b: str = "B",
    books_a: list[RngBook] | None = None,
    books_b: list[RngBook] | None = None,
    require_crn: bool = True,
) -> dict[str, Any]:
    """两臂 n 次重复的判决。**统计全部走 gates.py，这里不另写一套。**

    判据只有一条，但它有两种等价说法：

    * "95% 置信区间跨零"；
    * "**效应比置信区间还小**"。

    这两句说的是同一件事——对称的 t 区间是 ``mean ± h``，
    ``|mean| < h`` 与 "区间含 0" 是充要的。所以下面只判一次，
    但两句话都写进 ``verdict_text``，因为提问的人两种说法都会用。

    ``require_crn=False`` 时允许两臂用独立随机数——**结果仍然是对的，只是区间更宽**
    （``Var(a−b) = Var(a) + Var(b) − 2Cov(a,b)``，CRN 就是把那个协方差做正）。
    这条路径主要用来实测 CRN 到底省了多少。
    """
    from . import gates as gt  # noqa: PLC0415

    a = np.asarray(values_a, dtype=float).ravel()
    b = np.asarray(values_b, dtype=float).ravel()

    # **没给 books 就不能声称是 CRN。** 这里的三态是有意的：
    # True = 查过且成立、False = 查过不成立、None = 没法查。
    # 早先没给时直接返回 True，等于"查不到就当它对"——这正是 CLAUDE.md
    # 「外部结果的 CSI 口径只能靠声明」那条批评的做法：不能假装查过了。
    declared = bool(books_a) and bool(books_b)
    crn_issues = check_pairable(books_a, books_b) if declared else []
    if crn_issues and require_crn:
        return {
            "metric": metric, "unit": unit, "arm_a": arm_a, "arm_b": arm_b,
            "n_rep": int(min(a.size, b.size)),
            "crn": False, "blockers": crn_issues,
            "verdict": "not_pairable",
            "verdict_text": (f"**不能比较**：{crn_issues[0]['detail']}。"
                            f"配对检验已跳过——错配数据上的 p 值没有意义，"
                            f"而它照样会算出一个看起来很显著的数。"),
        }
    if a.size != b.size:
        raise ValueError(f"两臂重复次数必须一致，收到 {a.size} 与 {b.size}")

    paired = gt.paired_compare(a, b)
    g3 = gt.gate_conclusion(paired)

    sa, sb = summarize(a, arm_a), summarize(b, arm_b)
    hw = (paired.ci_high - paired.ci_low) / 2.0
    effect = paired.mean_diff
    rel = effect / sb.mean if abs(sb.mean) > _EPS else float("nan")
    rel_hw = hw / abs(sb.mean) if abs(sb.mean) > _EPS else float("nan")

    # **效应小于置信区间就拒绝下结论**，而不是照报百分比。
    # 还要再过一次 gates 的判决检验（以 Wilcoxon 为准，见 gates.PairedResult）——
    # 区间不跨零但非参检验不显著时同样不能声称，这正是 CLAUDE.md 里
    # 「门 3 的判决必须显式说清用哪个检验」那条要求的。
    significant = bool(paired.ci_excludes_zero and paired.decision_significant)
    if significant:
        verdict = "significant"
        vtext = (f"{arm_a} 相对 {arm_b}：{metric} {sa.mean:.3f} vs {sb.mean:.3f} {unit}，"
                 f"差值 {effect:+.3f}（{rel:+.1%}），95% CI "
                 f"[{paired.ci_low:+.3f}, {paired.ci_high:+.3f}]，n_rep={paired.n}，"
                 f"{'Wilcoxon 符号秩检验' if paired.decision_test == 'wilcoxon' else '配对 t 检验'}"
                 f" p={paired.decision_p_value:.3g}。**效应大于置信区间半宽 "
                 f"{hw:.3f}，结论成立。**")
    else:
        why = []
        if not paired.ci_excludes_zero:
            why.append(f"效应 {abs(effect):.3f} 小于 95% 置信区间半宽 {hw:.3f}"
                       f"（等价说法：区间 [{paired.ci_low:+.3f}, {paired.ci_high:+.3f}] 跨零）")
        if not paired.decision_significant:
            why.append(f"{'Wilcoxon' if paired.decision_test == 'wilcoxon' else '配对 t'} "
                       f"p={paired.decision_p_value:.3g} ≥ 0.05")
        need = gt.required_samples(paired.std_diff, abs(effect)) if abs(effect) > _EPS else -1
        verdict = "inconclusive"
        vtext = (f"{arm_a} 相对 {arm_b}：{metric} {sa.mean:.3f} vs {sb.mean:.3f} {unit}，"
                 f"点估计差 {effect:+.3f}（{rel:+.1%}），但 **{'；'.join(why)}**。"
                 f"**不能下结论，也不要报这个百分比**——"
                 f"n_rep={paired.n} 的实验分辨不出这么小的差异。"
                 + (f"要让这个效应站住至少需要 {need} 次重复"
                    f"（α=0.05，功效 80%）。" if need > 0 else
                    "点估计差本身接近 0，加样本也说明不了什么。"))

    return {
        "metric": metric, "unit": unit, "arm_a": arm_a, "arm_b": arm_b,
        "n_rep": int(paired.n),
        # 三态：True 查过且成立 / False 查过不成立 / None 调用方没给 books，查不了
        "crn": (not crn_issues) if declared else None,
        "crn_note": (None if declared else
                     "调用方没有给两臂的 RngBook，**无法核对是不是公共随机数**——"
                     "这条得你自己保证：两臂必须用同一个 rng.replications() 的返回值"),
        "blockers": crn_issues,
        "a": sa.as_dict(), "b": sb.as_dict(),
        "effect": _r(effect),
        "effect_rel": _r(rel),
        "ci95_of_effect": [_r(paired.ci_low), _r(paired.ci_high)],
        "ci95_half_width": _r(hw),
        "ci95_half_width_rel": _r(rel_hw),
        "effect_exceeds_ci": bool(paired.ci_excludes_zero),
        "paired": paired.as_dict(),
        "gate_conclusion": g3.as_dict(),
        "verdict": verdict,
        "verdict_text": vtext,
    }


def min_replications_note(n: int) -> str | None:
    """重复次数太少时的告警。**n ≤ 5 时 Wilcoxon 永远不可能显著。**

    双侧符号秩检验最小可达 p 是 ``2/2^n``：n=5 给 0.0625 > 0.05，
    也就是说**无论数据多干净，这个实验都不可能宣告显著**——
    而它照样会算出一个 p 值和一个漂亮的百分比。n=6 给 0.03125，是硬下界。
    """
    n = int(n)
    if n < 2:
        return ("**只跑了 1 次，没有置信区间可言。** 系统级 KPI 的种子间变异系数"
                "实测 11.4%（measurements/seed_variance.json），"
                "单次运行的数字不能用来做任何比较。")
    if n <= 5:
        return (f"**只有 {n} 次重复，判决检验（Wilcoxon）永远不可能显著**："
                f"双侧符号秩检验最小可达 p 是 2/2^{n} = {2 / 2 ** n:.4g} > 0.05。"
                f"这个实验无论跑出什么结果都不足以支撑结论——"
                f"把 num_replications 提到 ≥6（推荐 8）。")
    return None
