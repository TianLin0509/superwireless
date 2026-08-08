# 系统级仿真参数详解 —— `sw_system_sim`

**什么时候读这一份**：主文件「第 5 段」里的旋钮不够用、要解释某个 KPI
是怎么算的、要调话务或邻区负载、或者 `notes` 报了一条你不确定怎么处理的。

主文件里已经写死的三条不在这里重复：**每 UE 要 ≥8 个快照**、
**`cell` 与 `users` 两级都要看**、**`notes` 逐条转述**。

## 完整签名

```python
sw_system_sim(
    dataset_id,
    duration_s=5.0, traffic_model="ftp3", file_bytes=500_000, arrival_rate_hz=2.0,
    scheduler="pf", pf_window_tti=100, mu_enabled=False,
    trim="tail", tdd_pattern="DDDSU",
    neighbor_prb_util=0.3, neighbor_load_jitter=0.05,
    csi_aging=True, srs_period_ms=10.0, srs_hopping=True,
    csi_processing_delay_ms=2.0,
    olla_speedup=1.0, precoder="svd", seed=0, num_replications=8,
)
```

主循环里**没有任何矩阵运算**，全是查表加算术——所以 40000 个 TTI 只要 0.2 秒。
矩阵运算集中在 `build_link_tables` 建表那一相。

## 重复实验与置信区间 `num_replications` / `seed`

**每个 KPI 都是 `{mean, std, ci95, n_rep, cv, rel_half_width, min, max}`，不是裸数。**
默认 `num_replications=8`（对应 ns-3 的 `RngRun`），区间用 **t 分布**算——n 小的时候
用 z 会把区间报窄 20%。返回里的 `kpi_format` 块会把这套格式解释给用户。

**默认 8 是算出来的，不是拍的：**

- `n <= 5` 时 Wilcoxon 符号秩检验最小可达 p 是 `2/2^n > 0.05`，**无论数据多干净都
  不可能宣告显著**——而它照样会算出漂亮的百分比。n=6 是硬下界（p_min=0.031），
  8 留了余量（p_min=0.0078）。
- 代价可控：`build_link_tables` **与随机种子无关，只建一次表**，重复的只有 TTI 主循环。
  实测 ds_6e9715bc 上建表 5.14 s、单次主循环 0.99 s，n=8 是 13.0 s vs 单次 6.1 s，
  **墙钟 +113%**。建表越贵这个比例越低（按 10.5 s / 1.1 s 算是 +66%）。
- 区间随 n 收窄（从 64 次重复里重抽 500 次取平均半宽，总体变异系数 9.4%）：

  | n | 2 | 4 | 6 | 8 | 12 | 16 | 32 |
  |---|---|---|---|---|---|---|---|
  | 半宽/均值 | 60.9% | 13.7% | 9.4% | 7.6% | 5.8% | 5.0% | 3.4% |

  **"按 `1/√n` 收窄"精确成立的是标准误，不是区间半宽**——半宽还乘着
  `t_{0.975,n-1}`，小 n 时 t 大得多（`t₃=3.18` vs `t₁₅=2.13`），
  所以半宽实际收得**比 `1/√n` 更快**（n 4→16 是 0.357 而不是 0.5）。混为一谈会差 30%。

`num_replications=1` 退回"单次运行、无区间"，并在 `notes` 里明确告警——**这种结果
不能用来做任何比较**。

**`seed` 是实验批次的主种子（`RngSeed`），重复实验不要靠改它**——改它等于换一整个
宇宙，两批之间没有"流不重叠"的保证。要重复就调 `num_replications`。

### 随机流是分开的

`rng.STREAMS` 把随机数按用途分流：`channel`（生成与撒点）/ `traffic`（FTP3 泊松到达、
bimodal 的 RBG 尺寸抽样）/ `scheduler`（度量打平时的随机决胜）/ `harq`（ACK/NACK 伯努利）
/ `neighbor_load`（邻区利用率逐快照抖动）。

**分流的好处非常具体**：改话务模型不会连带改变信道实现，A/B 才是受控的。
分流之前一个 `rng` 同时喂话务和 HARQ——改一下 `arrival_rate_hz`，抽到的到达次数变了，
后面 HARQ 的伯努利序列**整个错位**，于是"话务模型的影响"里混着"HARQ 换了一批随机数"。
**这类污染在结果里看不出来。**

### A/B 必须用公共随机数（CRN）—— `rng.compare_replications`

CRN 是经典的方差缩减技术：比较两个方案时对应的两次运行用**同一批伪随机数**，
让观测到的差异归因于方案本身而不是随机波动（`Var(a−b) = Var(a)+Var(b)−2Cov(a,b)`，
CRN 就是把那个协方差做正）。**两臂拿同一个 `rng.replications(master_seed, n)` 的
返回值就是 CRN。**

实测收益（A/B：PF 窗 100 vs 1000，n_rep=8，真实效应约 −10 Mbps）：

| | 效应 | 95% CI | 半宽 | Wilcoxon p | 判决 |
|---|---|---|---|---|---|
| CRN | −10.64 | [−14.14, −7.15] | 3.49 | 0.0078 | significant |
| 独立随机数 | −14.97 | [−28.66, −1.27] | 13.69 | 0.078 | inconclusive |

**同一个真实效应，CRN 判得出来、独立种子判不出来**：区间窄 **3.92 倍**，
差值标准差 4.18 vs 16.38。注意独立那一栏的区间其实不跨零，但**判决以 Wilcoxon 为准**
（复用 `gates.paired_compare`，就是门 3 那套），照样拦住了。

`rng.check_pairable(books_a, books_b)` 是**硬拦截**：两臂重复次数不一致、
或者第 k 次重复的 `(master_seed, replication)` 对不上，直接返回
`verdict="not_pairable"` 而不给 p 值。**顺序被打乱这种错位在统计层面完全不可观测**
——统计只看数值数组，不知道第 i 个数对应哪一次重复，错配数据照样算得出漂亮的 p 值。
没传 `books` 时 `crn` 报 `None`（"没法查"），**不会当成查过了**。
`require_crn=False` 允许独立随机数，结果仍然是对的，只是区间明显更宽。

判据有两种等价说法，`verdict_text` 两句都写：**"95% 置信区间跨零"**与
**"效应比置信区间还小"**——对称 t 区间下 `|mean| < h` 与"区间含 0"是充要的。

**`rng.compare_replications` 目前只是库函数，没有对应的 MCP 工具。**
要判决就写脚本导入它跑；**不要臆造 `sw_compare_system_arms` 这类不存在的工具**。
另外 `sw_system_sim` 的返回里只有聚合后的 `KpiStat`，**没有逐次重复的原始值**，
所以这条路必须在脚本里从 `simulate_replications` 拿 `runs` 才走得通。

### 区间覆盖什么、不覆盖什么

各次重复共用**同一批信道与同一张链路表**，所以 `ci95` 覆盖的是**话务到达、HARQ 误码、
调度决胜**这三条流。返回里的 `rng.covered_by_ci` / `not_covered_by_ci` / `ci_scope`
把这件事显式写出来——**别把它当成"全部不确定度"**。

**信道实现本身的不确定度是另一个、更大的方差分量**，这个函数不做也做不到——
要覆盖它得用不同 `seed` 重新 `sw_generate` 再比。

这个取舍是量过的（`measurements/rng_replication.json`）：64 次 replication（表固定）
与 32 次 master seed 扫描（每次重建表、负载抖动重抽）的变异系数对照里，
五个 KPI 有四个的区间重叠——**冻结链路表并没有可分辨地把离散度报小**，
系统级的主导方差就是话务与 HARQ，正好是区间覆盖的那几条流。

顺带一个必须记住的量级：**n=8 时变异系数自身的 95% 区间是 0.66×~2.04×**。
`measurements/seed_variance.json` 里那个 11.4% 是 8 个种子测的，真值可能在 7.5%~23%
之间——**那张表上的 CoV 只精确到大约 2 倍**，别拿它做精细比较。

### `rel_half_width` 是这次实验的分辨率

区间半宽 / 均值。**比它小的差异，这次实验分辨不出来。** `notes` 会在头条 KPI
（体验速率、边缘体验速率、`cell_served_mbps`、`avg_mcs`、`avg_rank`、`bler_first_tx`）
里挑相对区间最宽的那个单独点名。要下更细的结论就加 `num_replications`。

**这条规则有过真实事故：** 同一批信道、同一套配置只改种子，`cell_experienced_mbps`
的变异系数实测 11.4%，而上一轮把这 11.4% 的噪声报成了「+14% 提升」。

## 话务 `traffic_model`

| 取值 | 是什么 | 什么时候用 |
|---|---|---|
| `ftp3` | 3GPP FTP Model 3，泊松到达的固定大小文件 | **默认**，评价体验速率的标准话务 |
| `bimodal` | 现网话务两头高中间低：约 30% 只占 1 个 RBG 的小包、约 30% 占满全带宽、约 30% 的 TTI 根本没调度 | 要贴现网 PRB 利用率（约 30%）时 |
| `full_buffer` | 缓冲区永不空 | 只看容量上限。**体验速率在这个模型下没有意义** |
| `cbr` | 恒定比特率 | 固定码率业务 |

`ftp3` 的负载由 `file_bytes × 8 × arrival_rate_hz` 决定，
返回的 `config.traffic.offered_load_mbps_per_ue` 直接给出每用户提供负载。
**太高会积压**，`notes` 会拦（`backlog_bytes > 15%` 的到达量时报出来）。

`bimodal` 是按**占用 RBG 数**分布，不是按文件大小。它的小包与大包体验速率
分开报（`small_pkt_experienced_mbps` / `large_pkt_experienced_mbps`）；
**小包的体验速率经常测不出来**——一个 TTI 就发完，3GPP 掐尾口径下没有可测量
的时间。这不是 bug，是这个 KPI 的固有盲区，现网话统里同样测不到。

## KPI 口径 `trim` / `warmup`

- `trim="tail"`（默认）：排除清空缓冲区的最后一个 slice，时间与数据同时扣
  （3GPP TS 28.552 §5.1.1.3）
- `trim="head_tail"`：再排掉首个 TTI，运营商话统常用口径
- `trim="none"`：不掐，数值虚高，不建议
- 短于 `min_burst_tti=2` 的 burst 不计入；前 `warmup_tti=200` 个 TTI 不计入
  （PF 的滑动均值要先收敛）

**换口径数字会明显变，所以报数时必须带上用的是哪个 trim。**

## 邻区负载 `neighbor_prb_util`

ChannelHub 的几何 SINR 是按**所有邻区都在发**算出来的，等于 100% PRB 利用率。
真实 5G 网络典型是 10% / 30% / 50%。按 full buffer 算会把干扰放大到不真实的程度，
所以默认取 **0.3**；设 1.0 退化成原行为。

**当前只支持全网统一值**——几何 SIR 是聚合量，拿不到逐邻区贡献。

`neighbor_load_jitter=0.05` 让实际生效负载在配置值 ±5% 内逐快照波动。
恒定负载会让所有快照的干扰完全一样，结果比现网干净。

注意这和信道生成阶段的 `pdsch_load` 不是一回事：后者在下行**完全不起作用**
（见 `scenarios-and-interference.md`），`neighbor_prb_util` 是系统级仿真自己
在链路表上做的折算。

## CSI 老化 `csi_aging` / `srs_period_ms` / `srs_hopping`

**默认开。** 关掉退化成零时延完美 CSI——那是个上界不是现网，MU 增益会被系统性高估。
保留这个开关是为了能做 A/B，把老化的代价量出来，而不是让它悄悄混进所有结果。

- `srs_period_ms` **只接受 5 / 10 / 20 / 40**，别的值直接报错
- `srs_hopping` 默认开，对应 38.211 Table 6.4.1.4.3-1 的 `C_SRS=57` / `B_SRS=1`：
  `m_SRS=(272,16,4,4)`、`N=(1,17,4,1)`，每跳 16 RB 正好一个 RBG，**17 跳**扫完 272 RB
- **跳频是老化的主导项**：10 ms 周期下全带扫一遍要 170 ms，某个 RBG 的年龄在
  0~160 ms 之间轮转（平均 80 ms），而 2.6 GHz、30 km/h 的相干时间只有约 3 ms。
  实测 MU/SU 比值 0.816 → 0.449（−45%），SU 谱效 −27%
- `csi_processing_delay_ms=2.0` 是信道估计 + 预编码计算 + 调度下发的固定时延

返回里的 `csi_aging` 块给出 `full_sweep_ms` 与 `mean_age_ms`，可以直接转述。

**快照间隔不是一个 TTI。** ChannelHub 的多时隙输出是连续的 SRS/CSI-RS 机会，
默认 `10 × 0.5 ms = 5 ms`，由 `system.snapshot_interval_ms(cfg)` 从配置算出。
把它当成 0.5 ms 会让**所有时间相关的结论差 10 倍**。

## 调度与 OLLA

`scheduler="pf"` 比例公平，度量 `R_inst / R_avg`，`R_avg` 按 `pf_window_tti=100`
的指数窗更新。窗太小接近 max-C/I（只喂近点用户），太大接近轮询（不利用信道起伏）。

**基站按陈旧 CSI 选 rank 和调度**（`best_se_gnb`），不是按真实 SINR——
拿真实 SINR 挑 rank 等于让基站预知信道，老化损失会被凭空抹掉一大半。

OLLA 步长默认 **+0.01 / −0.09**，稳态 BLER = `up/(up+down)` = 正好 10%。
（现网口头常说的 −0.1 对应的是 9.09%。）
**稳态与步长绝对值无关**，所以 `olla_speedup` 等比放大只改收敛速度与稳态抖动：
实测 k=1 时 8 秒内 IBLER 还停在 0.394、k=20 收敛到 0.100，而体验速率
142.3 → 142.5 几乎不动。短仿真里基线常常压不动一档 MCS，可临时设 10；
**出正式结论设回 1.0**，非 1.0 时结果里会带一条显式告警。

实测 IBLER 一致地比理论高 4~5%（相对），六个取值全部如此——那是 MCS 整数档的
系统性偏置（`select_mcs` 取「满足目标的最高档」，天然偏激进半档），不是噪声。

`avg_mcs` 报的是 **OLLA 之后**的 MCS，即实际调度下去的档位。

## MU `mu_enabled`

默认 **False**，先看清 SU 基线。开了之后 `measure_mu_gain` 会在建表阶段用真实的
`su_mu_adaptation` 测出**聚合比值**，主循环按 `ratio/K` 折算。

**这是标量近似**：逐 TTI 真配对要在每个 TTI 做 SVD + 矩阵求逆，十万 TTI 跑不完。
返回的 `mu_gain` 带逐快照比值与离散度，**离散度就是这个近似的可信度**——
实测 3.7%~13.1%，**超过 30% 就不该用标量**，那时的 MU 结论不要报。

实测在 10 用户 / 64 端口下 MU/SU 比值常 < 1（密集城区 0.755、城区宏站 0.917），
自适应因此全程选 SU。**这不是 bug**：SU 无干扰且能到 rank4，MU 硬顶 rank2
且每人只分 1/K 功率，自由度富余时 SU 本来就该赢。

## 发射权 `precoder`

- `svd`（默认）：逐 RBG 特征波束，理论最优
- `type1`：用 38.214 Type I 宽带码本当发射权。码本自由度少，**在 CSI 老化下
  反而可能更耐受**——能算错的地方也少。`type1` 时 BF Gain 恒为 0
  （发射权就是 CQI 的参照权）

CQI 的参照权始终是 `type1_wideband`，返回的 `precoder` 块会写明两者。

## `notes` 全清单

按触发条件列，每一条都是"这组数字在什么条件下不成立"：

| 触发条件 | 说的是什么 |
|---|---|
| 快照数 < 4 | 时间起伏被严重低估，PF 的多用户分集拿不到 |
| 快照数 ≤ 1 且开着老化 | 陈旧信道与当前信道是同一个矩阵，**老化效果恒为 0** |
| `csi_aging=False` | 预编码用零时延完美信道，是上界不是现网，MU 增益被系统性高估 |
| `measured_bursts < 20` | 进入体验速率统计的 burst 太少，加长 `duration_s` 或提高到达率 |
| 积压 > 15% 到达量 | 系统在这个负载下没收敛，体验速率被排队时间拖低 |
| 对账误差 > 1% | **这是 bug 不是现象**（发出去的 + 还压着的 应等于到达的） |
| `edge_mcs_p5 > 8` | 边缘 MCS 偏高（现场经验 <5），多半是撒点没覆盖到边缘或邻区负载设太低 |
| 首传 BLER > 目标 ×1.6 | 外环还没收敛完 |
| 有效 IoT 用户 < 90% | 多半是 `num_slots_per_sample > 1` 导致 SIR 与 SINR 口径不同 |
| IoT 中位 < 3 dB | 几乎是噪声受限，检查站间距或邻区负载 |
| `bimodal` PRB 利用率偏离 30% | 折合负载和现网口径对不上 |
| `bimodal` 小包体验速率为 None | 掐尾口径下单 slice burst 没有可测量的时间，固有盲区 |
| `outage_ue > 0` | 有用户全程够不到 MCS 0 的门限，已从调度剔除——**这本身就是结论** |
| 占用率 > 98% | 已过载，此时体验速率反映的是容量上限而不是用户体验 |
| `olla_speedup != 1.0` | 步长被放大，稳态抖动更大，出正式结论要设回 1.0 |
| `num_replications < 6` | 判决检验结构上不可能显著，区间也不可信 |
| 头条 KPI 相对半宽 > 5% | 点名最宽的那个——比它更小的差异这次实验分辨不出来 |

多次重复的 `notes` **按"抹掉数字后的模板"去重**并标注命中几次
（"（6/8 次重复都触发；上面的数值取自第 1 次）"）——同一条告警在 8 次重复里
只差几个数字，全列出来会把真正不同的那几条淹掉。

## 对账三件套

`offered_mbps` / `completed_bursts` / `backlog_bytes` 一起报，
才能解释"实际吞吐 105 Mbps vs 话务负载 144 Mbps"这种缺口——
它可能是队列积压（正常），也可能是漏数据（bug）。`accounting_error_pct`
就是用来分辨这两种的。
