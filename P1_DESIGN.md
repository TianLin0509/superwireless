# P1 方案设计：话务与包模型 / 调度时延与 HARQ RTT / 多小区联合调度

> 2026-08-08。本文只做设计与取证，没有改任何代码。
> 所有实测数字都是写这份文档时在本机跑出来的，配置写在数字旁边。
> 我不确定的地方标了 **【存疑】**，不要把它们当成已核实的事实。
>
> **行号的时效性**：引用的行号取自本文成稿时的 `system.py`（1410 行）。
> 写作期间另有 agent 在并行改这个文件，主循环区域的行号已经整体下移约 14 行
> （`re_per_tti` 1049 → 1063、`harq_pending` 1062 → 1076、
> `harq_pending[u] = (3, ...)` 1194 → 1218、TDD 时隙判断 1083 → 1097）。
> **定位以引用的代码片段为准，行号只当近似路标。**

---

## 0 先说结论

**三条课题里，最贵的那条（P1-C）比想象中便宜，最便宜的那条（P1-A）比想象中贵。**

| | 我原以为 | 查完之后 |
|---|---|---|
| P1-A | 加个 RBG 循环，中等工作量 | KPI 口径要重定义（体验速率、PRB 利用率、burst 定义都变），历史数字全部不可比 |
| P1-B | 中等 | 确实中等，但**必须和 HARQ 进程池一起做**，只加 RTT 不加进程会把吞吐打崩，那是假结果 |
| P1-C | 要重构架构、拿不到逐小区干扰 | **逐小区干扰拿得到，而且几乎不要钱**——`meta["rx_power_all_dbm"]` 里有全部 K 个小区的 RSRP，只是 superwireless 从来没存过它 |

推荐顺序：**P1-C 的第一阶段 → P1-B → P1-A → P1-C 的第二阶段**。理由见第 5 节，核心是"先把最便宜、能立刻改变干扰画像的那一半做掉"。

另外有一条**必须先修的口径 bug**（第 4.2 节 E14）：默认预设下 ChannelHub 存进 `sir_dB` 的那个数，有约 **4.7 dB 标准差是纯粹的邻区波束抽签噪声**，不是物理。这条会污染 P1-C 的全部结论，也在污染现在已经出的干扰类结论。

---

## 1 共同前提：本文核过的事实

写方案之前先把地基查清楚。下面每一条都有出处或实测。

### 1.1 现在的两相架构长什么样

```
第一相 build_link_tables()      system.py:494-731
  逐 UE、逐快照做 SVD / Type I 码本搜索 / MMSE 逐流 SINR
  → UeLinkTable[snapshot, rank] 的 sinr/mcs/se + 基站视角的 se_gnb
第二相 simulate()               system.py:1018-1382
  TTI 主循环：查表 + PF 度量 + 缓冲区更新，没有任何矩阵运算
```

**实测性能（本机，2026-08-08）**

| 项目 | 配置 | 耗时 |
|---|---|---|
| `build_link_tables` | 12 UE × 8 快照，24 RB × 64 BS × 4 UE | 1.94 s |
| `simulate` | 20000 TTI（10 s），12 UE | **0.56 s = 27.8 µs/TTI** |
| `simulate` | 4000 TTI（2 s），12 UE | 0.14 s = 35.1 µs/TTI |

任务书里说的"20000 TTI 要 4.5 秒"我没复现出来，本机是 0.56 s。差别应该在 UE 数或 MU 开关上——按 UE 数线性外推 21 UE 约 1.0 s。**下面所有性能估算都以 27.8 µs/TTI @ 12 UE 为基准**，用别的基准请自己换算。

### 1.2 第一相已经算出来、但被扔掉的东西

这一条对 P1-A 至关重要：

```
csi_aging.mmse_stream_sinr()    csi_aging.py:283-305   返回 [RBG, rank] 的逐 RBG 线性 SINR
mumimo.user_sinr_db()           mumimo.py:44-71        立刻把 RBG 维压成一个标量
```

**逐 RBG 的 SINR 在第一相里已经算出来了，然后被 `user_sinr_db` 平均掉。** 把它留下来是纯内存开销，**零额外计算**——17 个 RBG × 4 个 rank × 8 个快照 × 12 个 UE = 6528 个 float，可以忽略。

这意味着 P1-A 里"分到的 RBG 上真实 BLER 是多少"这个问题**不需要任何新的物理计算**就能答。

### 1.3 ChannelHub 给了什么、没给什么

`sample.meta` 里有（生成时不用开任何开关，本文实测确认存在）：

| 字段 | 内容 | superwireless 存了吗 |
|---|---|---|
| `rx_power_all_dbm` | **全部 K 个小区**的接收功率（dBm，含路损与扇区天线增益，不含阵列增益） | ❌ `generate.py:32-39` 的 `_SCALAR_META_FIELDS` 里没有 |
| `pathloss_all_db` | 全部 K 个小区的路损 | ❌ |
| `antenna_gain_all_db` | 全部 K 个小区的扇区天线增益 | ❌ |
| `serving_cell_index` / `serving_pci` | 服务小区索引 | 只存了 `serving_pci` |
| `interferer_distances_m` | K−1 个干扰小区的距离 | ❌ |
| `num_cells` | 小区总数 | ❌ |

**只有 `rx_power_serving_dbm`（单个标量）被存下来了，全小区那个数组没存。** 这就是"拿不到逐小区贡献"这个判断的真正来源——不是 ChannelHub 没给，是我们没接。

`h_interferers` 的情况：

* 形状 `[K−1, T, RB, BS, UE]`，`loader.py:82-84`
* **默认只存最强的 3 个**，由 `internal_sim.py:1100` 的 `max_per_ue_intf_cells`（默认 3）控制，是个**可配置的键**
* 实测把它设成 20：`h_interferers` 变成 `(20, 1, 272, 64, 4)`，**11.14 MB/样本**，而服务小区信道只有 0.56 MB —— **20 倍于服务信道**
* 排序是"按 `rx_power_dbm` 降序"（`internal_sim.py:2237`），所以**只要有 `rx_power_all_dbm` + `serving_cell_index` 就能精确还原 `h_interferers` 的第 i 层是哪个小区**，不需要额外的 ID 字段（meta 里确实没有 `intf_cell_ids`，实测确认）
* 幅度定标是 `sqrt(P_k/P_serving)`（`internal_sim.py:2119-2131`），即 **RSRP 域**的相对功率，不含波束赋形增益

---

## 2 P1-A：完整的话务与包模型

### 2.1 现状精确描述

**"所有用户都是 full buffer"这个说法不准确。** 话务模型是有的：`TrafficConfig`（`system.py:47-98`）支持 `full_buffer` / `ftp3` / `cbr` / `bimodal`，`_Traffic`（`system.py:833-925`）有 burst 到达、单活跃 burst + FIFO 队列、逐 burst 的服务记账，体验速率按 TS 28.552 掐尾/掐头去尾（`system.py:931-962`）。**这一层是真的。**

真正缺的是下面三条，而且第一条最要命：

**（1）频域完全没有复用——一个 TTI 只服务一个用户，而且强制占满全带**

```python
# system.py:1047-1049
n_rb = sys_cfg.num_rbg * sys_cfg.rb_per_rbg
re_per_tti = n_rb * 12 * 12          # 272 × 144 = 39168 RE，恒定
```

```python
# system.py:1123
picked = [cand[i] for i in order[:sched.max_mu_users]] if use_mu else [cand[order[0]]]
```

`use_mu` 为假时 `picked` 只有一个人，`re_per_tti` 原样喂给 `transport_block_size`（`system.py:1146-1147`）。MU 时 K 个人**也都拿全带**（这是对的，MU 是空间复用），所以**频域维度在整个 `simulate` 里根本不存在**。

`bimodal` 抽出来的 RBG 数只走到两个地方：

```python
# system.py:872-877   只用来定 burst 字节数和打 is_small 标记
n_rbg, small = self.draw_rbg(self.num_rbg)
n_bytes = max(200, int(self._per_rbg_bytes * n_rbg))
b = _Burst(tti, n_bytes, n_bytes, is_small=small)
```

`is_small` 之后只在 KPI 分组（`system.py:1226-1228`）和直方图（`1263-1267`）里出现。**没有任何一处让它影响资源占用。** 我用 grep 逐处核过。

**（2）`p_idle_tti` 是个死旋钮**

`TrafficConfig.p_idle_tti`（`system.py:71`）在整个文件里只出现在 `expected_prb_util()`（`system.py:98`）——一个纯报告用的解析式。**它不生成任何空闲 TTI。** 实际的空闲率完全由 `arrival_rate_hz` 与信道能力决定。所以"现网口径 30% PRB 利用率"这句话在 `bimodal` 下**不是仿真出来的，是算出来印在配置里的**。

**（3）没有包层**

粒度是 burst（一坨字节）。`_Burst`（`system.py:819-831`）只有 `bytes_total` / `bytes_left` / 首末 TTI。没有 IP 包、没有分段、没有逐包时延。全文件搜 `delay` / `latency` / `时延`，只命中 CSI 老化相关的注释——**系统级没有任何时延 KPI**。

### 2.2 缺什么、为什么重要

**频域的账，用实测数字说**（表 3 MCS 表，本文实测）：

| MCS / rank | 全带 272 RB 一个 TTI 能传 | 1 个 RBG（16 RB）能传 | 比值 |
|---|---|---|---|
| MCS 5 / rank 2 | 14 347 B | 848 B | 16.9 |
| MCS 12 / rank 2 | 29 722 B | 1 729 B | 17.2 |
| MCS 20 / rank 4 | 104 497 B | 6 147 B | 17.0 |
| MCS 27 / rank 4 | 143 436 B | 8 448 B | 17.0 |

一个 **1500 字节的 IP 包，在 MCS12 / rank2 下只需要 1 个 RBG**（1729 B ≥ 1500 B）。现在的做法是给它 17 个 RBG，**94% 的频域资源在这半毫秒里空转**。

具体场景：一个小区里 12 个用户，都在刷微信（小包为主）。现在的模型说"每半毫秒只能服务 1 个人"，于是第 12 个人要排 6 毫秒的队。真实基站在同一个 TTI 里就把这 12 个人全发了。**这不是"精度差一点"，是把小区的用户容量硬顶低了一个数量级。**

反方向也错：一个 500 KB 的 FTP3 文件在 MCS20 / rank4 全带下**只要 5 个 TTI**（本文实测）。而体验速率的掐尾口径要扣掉最后一个 slice，`min_burst_tti=2` 又要求至少 2 个 TTI——**5 个 TTI 的样本里有 1 个被扣掉，测量分辨率本来就很粗**。给了全带宽反而让 burst 短到测不准。

**包层的账**：现在能回答"这个用户的体验速率是多少"，不能回答"这个包等了多久"。而对小包业务，体验就是时延——`system.py:1361-1367` 那条 note 自己写着"小包的体验速率测不出来……要看小包体验请看调度时延分布"，**而调度时延分布并不存在**。这条 note 指向了一个不存在的东西。

### 2.3 实现方案

#### 2.3.1 数据结构：burst → packet queue，但 burst 概念要保留（且要改定义）

```python
@dataclass
class _Packet:
    seq: int
    arrival_tti: int
    size_bytes: int
    bytes_left: int
    first_tx_tti: int = -1        # 首次有字节被发出去
    done_tti: int = -1            # 最后一个字节发出去

@dataclass
class _Flow:                       # 每个 UE 一个
    q: collections.deque[_Packet]  # FIFO，队头是 HoL 包
    bytes_backlog: int             # 增量维护，别每次求和（主循环里每 TTI 都要读）
    burst_start_tti: int = -1      # 缓冲区从空变非空的那一刻
    burst_bytes: int = 0
    burst_tti_span: ...
```

**`_Burst` 的定义要改，而且这是个口径变更。** 现在一个 burst = 一个文件（`system.py:877` 到达时就 new 一个）。TS 28.552 §5.1.1.3 的 data burst 是**缓冲区连续非空的那一段时间**。两个文件如果到达时间重叠，现网话统算**一个** burst，现在的代码算**两个**——第二个文件的排队时间被算进它自己的分母，等于把排队惩罚记了两遍。

**【存疑】** 这是我读 28.552 的理解，我没有标准原文在手核对。改之前请你确认一次"data burst 的边界是文件边界还是缓冲区忙期边界"。如果是前者，那现状是对的，只要在文档里写清楚就行。

新的 burst 由 `_Flow` 在"缓冲区 0 → 非 0"时开始、"非 0 → 0"时结束，与包无关。

#### 2.3.2 调度器：RBG 维度的按需分配

**关键约束：不能引入频选调度。** 你明确否掉了子带 CQI，所以**分给谁哪几个 RBG 必须与信道无关**——只按需求量分，不按哪段频率好分。实现上就是：分配顺序由 PF 度量定（PF 度量仍然用全带的 `best_se_gnb`，不变），分到手的具体是哪几个 RBG 用**连续编号**（0,1,2…），不做任何挑选。

```
每个下行 TTI：
  cand = 有数据 且 非 outage 的 UE                      # 不变
  metric = inst_se / r_avg （或 max_ci / rr）           # 不变，仍用 best_se_gnb
  order = argsort(-metric)

  free = num_rbg                                        # 17
  alloc = {}                                            # ue -> rbg 数
  for u in order:
      if free == 0 or len(alloc) >= max_ue_per_tti: break
      m, r = mcs_and_rank(u)                            # 查表，不变
      need = rbg_needed[m][r][bytes_left(u)]            # O(1) 查表，见下
      n = min(need, free, max_rbg_per_ue)
      if n < min_rbg_per_ue: break                      # 剩不下一个最小分配单元就收工
      alloc[u] = n; free -= n

  # 尾料：free > 0 时全部补给 order[0]（现网常见做法之一）
  if free and alloc: alloc[order[0]] += free
```

`max_ue_per_tti` 是**PDCCH 容量的代理**（AUDIT 里点名的遗漏）。现网一个 TTI 的 PDCCH 通常能下 8~16 个 DCI。默认给 **8**，并在结果里显式报"被 PDCCH 顶住的 TTI 占比"——如果这个数很高，说明结论受这个假设支配，必须告诉用户。

#### 2.3.3 TBS 按分到的 RBG 数算

```python
n_re = n_rbg * rb_per_rbg * 12 * 12
tbs  = la.transport_block_size(n_re, mcs.rate, mcs.q_m, layers=rank)
```

`transport_block_size`（`linkadapt.py:305-333`）本来就吃 `n_re`，**不用改一行**。

反查（"要发 X 字节需要几个 RBG"）**不能用除法**——TBS 有量化和分码块，不是线性的。做法是建表：

```python
# 第一相尾巴上算一次，17 × 28 × 4 = 1904 个 int，常驻内存
TBS_TABLE[n_rbg][mcs][rank] = transport_block_size(n_rbg*256*... ) // 8
# 反查用 np.searchsorted 在 n_rbg 轴上找第一个 >= 需求的，O(log 17)
```

主循环里只有查表和整数比较，**两相架构完好**。

#### 2.3.4 被分到的 RBG 上的真实 SINR

这是唯一有物理内容的新增项，而且第 1.2 节已经说了：**数据现成，零额外计算**。

在 `UeLinkTable` 上加一个字段：

```python
sinr_per_rbg_db: np.ndarray | None   # [snapshot, rank, rbg]  真实（评估）逐 RBG SINR
```

由 `build_link_tables` 在调 `mu.user_sinr_db` 之前顺手存下 `mmse_stream_sinr` 的返回值。

**用法必须严格区分两个视角**（沿用项目已有的 `se_gnb` vs `se` 分工）：

* **选 MCS / 排调度顺序**：只用**全带** `sinr_tx_db`（CQI 门限 + BF Gain）+ OLLA。基站没有子带 CQI，它不知道哪个 RBG 好。**这一条不能破，破了就等于偷偷做了频选调度。**
* **判 ACK/NACK**：用**实际分到的那几个 RBG** 上的真实 SINR，按 `user_sinr_db` 的口径（RBG 内线性平均、RBG 间 dB 平均）压成一个数再查 BLER 曲线。

这样才会出现现网真实的效应：分到 3 个 RBG 的用户，运气不好抽到 3 个深衰的 RBG，误码率就高于全带平均——**这个惩罚是真的，而且只有做了 FDM 才会出现**。全带分配时它被平均掉了。

#### 2.3.5 KPI 口径会怎么变

| KPI | 现在 | 改后 | 方向 |
|---|---|---|---|
| PRB 利用率 | 不存在（`expected_prb_util` 是解析式，不是实测） | `Σ_t Σ_u n_rbg(u,t) / (dl_tti × 17)`，真实测出来 | 新增，**这是这个课题最大的产出** |
| `occupancy` | 有用户被调度的 TTI 占比 | 语义不变，但会明显上升（同一 TTI 多人） | ↑ |
| `cell_served_mbps` | 各用户之和 | 不变 | 小包为主时 **↑**（回收了空转的频域）；大包为主时基本不变 |
| `experienced_mbps` | 各用户平均 | 口径不变 | **方向不确定**：排队时间下降推高它，单次拿到的带宽变少推低它。**净效应我不敢预测，这正是要测的东西。** |
| burst 数 | 每个文件一个 | 每个缓冲区忙期一个 | 数量下降，单个变长（有利于测量） |
| 新增 | — | 逐包时延 p50/p95/p99、HoL 时延、PDCCH 受限 TTI 占比 | |

**历史数字全部不可比。** 改完之后 2026-08-08 之前所有的系统级体验速率都要重算才能横比。这一条要写进 CLAUDE.md。

#### 2.3.6 两相架构保不保得住

**保得住，而且新增的都是整数运算。** 逐 TTI 的新增工作量：

* 分配循环：最多 `max_ue_per_tti`=8 次迭代，每次 2 次查表 + 几次整数比较
* 逐 RBG SINR 聚合：分到的 RBG 是**连续区间**，可以预先对每个 (snapshot, rank) 算好 dB 域的**前缀和**，取任意区间 O(1)。**这一步是关键优化，不做的话每 TTI 每用户要做一次 slice + mean，会慢一个量级。**

估计主循环从 27.8 µs/TTI 涨到 **80~150 µs/TTI**（3~5×），20000 TTI 约 **1.6~3 s**。可接受。

### 2.4 工作量与风险

| 文件 | 改动 | 行数量级 |
|---|---|---|
| `system.py` | `_Packet` / `_Flow` / `_Traffic` 重写、RBG 分配器、TBS 反查表、前缀和、KPI | ~400 |
| `system.py` | `UeLinkTable.sinr_per_rbg_db` + `build_link_tables` 存它 | ~30 |
| `server.py` | `sw_system_sim` 新参数（`max_ue_per_tti` / `min_rbg_per_ue` / `packet_size_bytes` / `fdm_enabled`）+ 转述 | ~50 |
| `spec.py` | `_SIM_DEFAULTS` 必须同步（`system.py` 的注释里点名了这条约定） | ~15 |
| `tests/test_system.py` | 新增不变量 | ~180 |
| **合计** | | **~650 行** |

**风险**

1. **最大的风险不是代码，是口径**。burst 定义变更 + PRB 利用率从解析式变实测值，会让所有历史结论不可比。必须一次性做完并在 CLAUDE.md 立碑。
2. `test_system.py` 现有 65 项里有相当一部分断言的是"一个 TTI 一个人"的行为，会红一片。要逐条判断"这条是断言物理还是断言旧实现"。
3. 尾料分配策略（`free > 0` 时给谁）会影响公平性指标，而这没有唯一正确答案。**建议做成显式参数并默认给 order[0]，同时在结果里报出尾料占比**——如果尾料占比很高，说明 `min_rbg_per_ue` 设得不对。

### 2.5 验证方法

按项目的"零时延必须逐位退化"文化，写成硬不变量：

* **I-A1（退化）** `fdm_enabled=False` 时，全部 KPI 必须与今天**逐位相同**。这是 A/B 开关，也是最重要的一条——它保证 FDM 是叠加上去的能力而不是另一套物理。
* **I-A2（退化）** `fdm_enabled=True` 但只有 1 个 UE 有数据时，它必然拿到全部 17 个 RBG，TBS 与 BLER 必须与 I-A1 逐位相同。
* **I-A3（守恒）** 每个 TTI `Σ_u n_rbg(u,t) ≤ 17`，且实测 PRB 利用率 = `Σ/(dl_tti×17)`，与逐 TTI 累加值对得上（浮点误差外为 0）。
* **I-A4（守恒）** 现有的字节对账（`accounting_error_pct < 1%`）继续成立。这条抓到过真 bug（HARQ 重传漏计 served，差 4.5%），不能丢。
* **I-A5（物理方向）** 纯小包话务（`p_1rbg=1.0`）下，开 FDM 后 `cell_served_mbps` 必须**明显上升**（预期数量级：接近 `min(max_ue_per_tti, 17)` 倍，受话务量上限约束），而实测 PRB 利用率必须仍然远低于 1。**如果聚合吞吐没涨，说明分配器没生效。**
* **I-A6（不能被"修好"的盲区）** 单 slice 的 burst 在 `trim="tail"` 下仍然测不到体验速率。**如果改完之后突然测得到了，说明掐尾实现被改坏了**——这是 KPI 的固有盲区，不是 bug。
* **I-A7（频选没有偷偷溜进来）** 把逐 RBG SINR 换成"每个 RBG 都等于全带均值"（人为抹平频选），体验速率的变化必须很小（**【存疑】** 我猜 < 3%，没测过）。**如果变化很大，说明调度器在利用频域信息**，那就违背了"不做频选调度"的约束，必须查。
* **I-A8（时延自洽）** 每个包 `done_tti − arrival_tti = 排队 + 传输 + 重传`，逐包相加与分项统计对得上。

### 2.6 依赖与顺序

* **不依赖** P1-B、P1-C，可以独立做。
* **被 P1-B 依赖**：P1-B 的"小包体验被高估多少"这个结论，只有在有 FDM 和逐包时延之后才测得准。反过来 P1-B 可以先做（见第 5 节），只是那时的量级估计要打折扣。
* **和 P1-C 冲突点**：两个都要重写 `simulate` 的主循环。**建议不要并行开工**，否则合并会很痛。

---

## 3 P1-B：调度时延与 HARQ RTT（k0 / k1 / k2）

### 3.1 现状精确描述

**HARQ 是有的，但没有时间。**

```python
# system.py:1062
harq_pending: dict[int, tuple[int, int]] = {}   # ue -> (剩余重传次数, TB bytes)
```

**每个 UE 只有一个 HARQ 槽位**，不是 16 个进程。

```python
# system.py:1157-1177（节选）
pend = harq_pending.get(u)
if pend is not None:
    left, size = pend
    bler = _bler_lookup(int(tables[u].mcs[snap, r - 1]),
                        float(tables[u].sinr_db[snap, r - 1]), "retx")
    retx_cnt[u] += 1
    if rng.random() > bler:
        served[u] += tr.serve(u, tti, size)
        harq_pending.pop(u, None)
    ...
```

这段代码在 UE **下一次被调度到的那个 TTI** 立刻执行重传。对一个 PF 度量高的用户，那就是**下一个下行 TTI（0.5 ms）**。真实系统是 4~8 个 TTI。

```python
# system.py:1194
harq_pending[u] = (3, tb_bytes)      # 最多 3 次重传
```

**另外这里有两处口径不一致，是修 HARQ 时必须一起处理的：**

**（a）重传的 BLER 查错了 MCS。** 实发的 MCS 是 `m`（由 `sinr_tx_db + olla` 选出，`system.py:1140-1144`），但重传的 BLER 查的是 `tables[u].mcs[snap, r-1]`（由**真实接收 SINR** 反查出来的 MCS）。后者按定义就是"在这个 SINR 下 BLER 恰好等于目标"的那一档——**所以重传几乎必然成功**。这让残留 BLER 系统性偏低。首传那一支是对的（`system.py:1180-1181` 用 `m`）。

**（b）S 时隙被当成整个下行时隙。**

```python
# system.py:1083
if pattern[tti % len(pattern)] not in ("D", "S"):
    continue                                   # S 被放行，然后按全下行处理
```
```python
# system.py:289-291
return (p.count("D") + 0.7 * p.count("S")) / len(p)   # 这里 S 只算 0.7
```

`dl_ratio` 只被 `as_dict()` 用来报告（grep 确认），主循环不读它。所以 **DDDSU 下 S 时隙拿到了 100% 的下行 RE，而报告里说它只有 70%**。下行 TTI 数被高估约 `0.3/5 = 6%`。P1-B 要建正式的时隙表，顺手修掉。

**没有任何调度时延。** `tr.step(tti)` 投数据（`system.py:1086`），同一个 TTI 里就可能被调度、被发出去。k0 / k1 / k2 三个都是 0。

### 3.2 缺什么、为什么重要

**TDD DDDSU 下的 HARQ RTT 手算（30 kHz，1 时隙 = 0.5 ms）**

时隙相位 `n mod 5` = `[D0, D1, D2, S3, U4]`。ACK 只能落在 U（或 S 的上行符号）上。

| PDSCH 落在 | 最近的上行机会 | k1（时隙） | + gNB 处理 1 时隙 | 最早重传时隙 | RTT |
|---|---|---|---|---|---|
| D0 | U4 | 4 | 5 | 5（下一周期 D0） | **5 时隙 = 2.5 ms** |
| D1 | U4 | 3 | 4 | 5 | 4 时隙 = 2.0 ms |
| D2 | U4 | 2 | 3 | 5 | 3 时隙 = 1.5 ms |
| S3 | U4 | 1 | 2 | 5 | 2 时隙 = 1.0 ms |

再加上 38.214 §5.3 的 UE PDSCH 处理时间 N1（capability 1、30 kHz、front-loaded DMRS 约 10 个符号 < 1 个时隙），k1 至少要 ≥ 1。所以**平均 RTT 约 3.5 个时隙 ≈ 1.75 ms**，与"4~8 个 TTI"的口头印象同量级但偏低——**DDDSU 每 5 个时隙就有一个上行机会，RTT 比 DDDDDDDSUU 这类图案短得多**。这个数值本身就是一个值得报出去的结论。

**对体验速率的影响量级（粗估，请当成数量级不是精确值）**

一个 500 KB 的 FTP3 文件全带 MCS20/rank4 要 5 个 TTI（第 2.2 节实测）。10% 首传 BLER ⇒ 期望 0.5 次 NACK。

* 现状：每次 NACK 多花 1 个 TTI ⇒ burst 时长 5 + 0.5×1 = **5.5 TTI**
* 加 RTT 后：每次 NACK 多花 ~3.5 个时隙 ⇒ 5 + 0.5×3.5 = **6.75 TTI**

体验速率被高估 `6.75/5.5 − 1 ≈ **+23%**`。

**对小包更狠**：一个 1 TTI 就发完的包，首传成功时时延 1 TTI，NACK 时现状是 2 TTI、加 RTT 后是 4.5 TTI。10% BLER 下平均时延 1.1 → 1.35 TTI（+23%），但 **p99 时延从 4 TTI 跳到 ~15 TTI（3 次重传 × 3.5 + 1）**——**尾时延被高估了近 4 倍**，而尾时延正是小包业务的体验指标。

**这是这个课题最值得做的理由：均值影响 20% 量级，尾部影响 4 倍量级。**

**必须一起做 HARQ 进程池，否则结果是假的。** 现在每个 UE 只有 1 个 HARQ 槽。如果只加 RTT 不加进程数，那么一个 UE 在等 ACK 的 3.5 个时隙里**完全不能被调度**——吞吐会断崖下跌，而那个下跌是实现缺陷不是物理。38.214 规定 DL 最多 16 个进程正是为了填满这个管道。**只加 RTT 得到的"体验速率下降 60%"会是一个纯粹的假结论。**

### 3.3 实现方案

#### 3.3.1 正式的 TDD 时隙表

```python
@dataclass
class TddConfig:
    pattern: str = "DDDSU"
    s_dl_symbol_ratio: float = 0.7      # S 时隙里能用于下行的符号比例
    # 派生（__post_init__ 里算一次，主循环只查）
    is_dl: np.ndarray                   # [len(pattern)] bool
    is_ul: np.ndarray                   # ACK 能落的时隙
    re_scale: np.ndarray                # D=1.0, S=s_dl_symbol_ratio, U=0
```

`re_per_tti` 改成 `re_per_tti_base * re_scale[tti % P]`。**这一条修掉了 3.1(b)。**

#### 3.3.2 k1 求解器

```python
def k1_for(slot_phase: int, tdd: TddConfig, n1_slots: int = 1) -> int:
    """PDSCH 在 slot_phase 发出，HARQ-ACK 最早落在几个时隙之后。
    38.213 §9.2.3：k1 从 dl-DataToUL-ACK 集合里取，实际系统里就是
    '满足 UE 处理时间的、最近的上行机会'。
    """
    P = len(tdd.pattern)
    for j in range(n1_slots, n1_slots + 2 * P):
        if tdd.is_ul[(slot_phase + j) % P]:
            return j
    raise ValueError("这个 TDD 图案里没有上行时隙，HARQ 反馈无处可去")
```

`n1_slots` 由 38.214 §5.3 的 N1 符号数换算：30 kHz、capability 1、front-loaded DMRS ⇒ N1 = 10 符号 ⇒ `ceil(10/14) = 1` 个时隙。**做成参数并把出处写在 docstring 里**，别硬编码。

k0（PDCCH → PDSCH）默认 0（同时隙调度，现网常态）。k2（UL grant → PUSCH）**下行仿真用不上**，但要在文档里说清楚"不是漏了，是这个方向不需要"。

#### 3.3.3 HARQ 进程池

```python
@dataclass
class _HarqProc:
    tb_bytes: int
    mcs: int                  # **实发的那一档**，重传要用它查 ReTx 曲线
    rank: int
    n_rbg: int                # P1-A 之后才有意义
    tx_count: int             # 1 = 首传
    packets: list[...]        # 这个 TB 覆盖了哪些包的哪些字节（时延记账用）
    state: Literal["idle", "waiting_ack", "ready_retx"]
    ack_due_tti: int
    retx_ready_tti: int
```

每个 UE 一个 `list[_HarqProc]`，长度 `n_harq_proc`（默认 **16**，38.214 的 DL 上限）。

主循环里加两个 O(1) 的定时结构：

```python
ack_due: list[list[tuple[int, int]]]     # 环形缓冲，长度 max_k1+2，索引 tti % L
                                          # 元素 (ue, proc_id)
retx_ready: list[list[tuple[int, int]]]   # 同上
```

**每 TTI 的处理顺序（顺序错了会差一个 TTI）**

```
1. 结算本 TTI 到期的 ACK/NACK       ← 先结算，让这些进程有机会在本 TTI 就被排上
2. tr.step(tti)  投新数据
3. 把到期的 retx 进程标成可调度
4. 组候选：重传优先（现网准则），再按 PF 排新传
5. 分配 RBG / 选 MCS / 发送
6. 对每个发出去的 TB：抽 BLER 结果，但**不立刻应用**，
   而是把 (ue, proc) 挂到 ack_due[(tti + k1) % L]
7. 更新 PF 平均速率
```

**第 6 步的关键**：BLER 抽签在**发送时**做（因为那时才知道 SINR 和 MCS），但**结果在 k1 之后才生效**。这样 OLLA 的更新也自然被推迟到 ACK 到达时——这是对的，`SchedulerConfig` 的注释里本来就写着"反馈只作用于下一调度时刻"，现在才真的做到。

**重传的 BLER 必须用 `proc.mcs`**，修掉 3.1(a)：

```python
bler = _bler_lookup(proc.mcs, sinr_at(u, snap, proc.rank), "retx")
```

改完之后**残留 BLER 会上升**，这是正确方向，不是回归。

#### 3.3.4 时延记账

每个 `_Packet` 记 `arrival_tti` / `first_tx_tti` / `done_tti`（`done_tti` 只在**承载它最后一个字节的那个 TB 被 ACK 时**才写）。新增 KPI：

* `queue_delay_ms`：`first_tx_tti − arrival_tti`，p50 / p95 / p99
* `packet_delay_ms`：`done_tti − arrival_tti`，p50 / p95 / p99
* `harq_rtt_ms_mean` / `harq_retx_share`
* `harq_proc_occupancy`：进程池平均占用率。**这个数很重要**——接近 1 说明进程数不够，结论受实现约束支配，必须告警。

#### 3.3.5 两相架构

**完全不受影响。** 新增的全是整数、环形缓冲和 dict 查找，没有一次浮点矩阵运算。估计主循环从 27.8 µs/TTI 涨到 **45~70 µs/TTI**（1.6~2.5×）。

### 3.4 工作量与风险

| 文件 | 改动 | 行数量级 |
|---|---|---|
| `system.py` | `TddConfig` + k1 求解器 | ~80 |
| `system.py` | `_HarqProc` + 进程池 + 两个定时环 + 主循环重排 | ~180 |
| `system.py` | 时延 KPI + notes | ~70 |
| `server.py` | `sw_system_sim` 新参数（`n_harq_proc` / `n1_slots` / `gnb_proc_slots` / `harq_rtt_enabled`） | ~35 |
| `spec.py` | `_SIM_DEFAULTS` 同步 | ~10 |
| `tests/` | 新增不变量 | ~140 |
| **合计** | | **~450~520 行** |

**风险**

1. **主循环步骤顺序**（3.3.3 的 1~7）错一步就差一个 TTI，而症状是"体验速率差百分之几"——**看起来完全正常**。防御手段只有 I-B2 那条手算表对照。
2. 修 3.1(a) 之后残留 BLER 上升，会有人以为是回归。要在 CLAUDE.md 里写明"这是修正，旧的残留 BLER 偏低"。
3. `n_harq_proc` 的默认值会直接决定结论。**必须扫一遍并把饱和曲线放进文档**，不能拍 16 了事。

### 3.5 验证方法

* **I-B1（退化）** `harq_rtt_enabled=False`（等价于 k0=k1=gnb=0、`n_harq_proc=1`）时，全部 KPI 与今天**逐位相同**。
* **I-B2（对标手算）** DDDSU、`n1_slots=1` 下，k1 求解器必须给出 `{D0:4, D1:3, D2:2, S3:1}`，且 ACK 时隙相位恒为 4（U）。这张表是手算的，写进测试当金标准。
* **I-B3（RTT 真的生效了）** 把 BLER 强制成 0（无重传）时，开不开 RTT **结果必须逐位相同**。这条抓"把 RTT 错误地加到首传上"——如果首传也被延迟了，无重传场景也会变，当场暴露。
* **I-B4（进程数饱和曲线）** `n_harq_proc` 从 1 扫到 16，`cell_served_mbps` 必须**单调不降**并在某处饱和。饱和点应当 ≈ `ceil(平均 RTT / 平均下行时隙间隔)`（DDDSU 下约 3~4）。**如果 n_harq_proc=1 时吞吐没有明显掉，说明 RTT 根本没生效。**
* **I-B5（时延守恒）** 逐包 `packet_delay = queue_delay + tx_slots + Σ retx_rtt`，全体包相加与分项统计的差为 0（整数运算，不允许有误差）。
* **I-B6（S 时隙修正的可见性）** 修掉 3.1(b) 之后，DDDSU 的有效下行 RE 总量必须精确等于 `(3 + 0.7) / 5 = 0.74` 而不是 `4/5 = 0.8`。小区吞吐应当下降约 **7.5%**（0.8/0.74 − 1 = 8.1%，取整数量级）。**这个下降是修正，不是回归。**
* **I-B7（重传 MCS）** 断言重传查的是 `proc.mcs` 而不是 `tables[u].mcs`。构造一个 OLLA 偏置很大的场景，两者会明显不同，直接比对 BLER 输入值。

### 3.6 依赖与顺序

* **不依赖 P1-A**：进程池、k1、时延记账都可以在"一个 TTI 一个人、全带"的框架下做完。
* **但 P1-A 之后量级会变**：有了 FDM，一个 TTI 服务多个用户，队列更短，RTT 的相对影响更大（因为传输时间变短了，等待时间占比上升）。所以 P1-B 先做得到的量级是**偏保守**的。这不影响做，只影响引用时的措辞。
* **和 P1-C 无耦合**。
* **强烈建议 P1-B 排在 P1-A 之前**：它的改动局限在主循环的时序，不碰 KPI 口径；P1-A 会重定义 burst 和 PRB 利用率。**先做不改口径的那个，回归压力小得多。**

---

## 4 P1-C：多小区联合调度

### 4.1 现状精确描述

只有服务小区真的跑调度。邻区是一个静态标量：

```python
# system.py:213-215
prb_utilization: float = 0.3
jitter: float = 0.05
```
```python
# system.py:239-257
def apply_neighbor_load(sinr_db, sir_db, utilization):
    """SINR' = 1 / (η·I + N)，其中 I = S/SIR、N = S/SNR"""
```

折算发生在**第一相**（`system.py:561-583`），逐快照抖动 ±5%。**折算完就固化进 `UeLinkTable` 了，TTI 主循环里干扰是常数。**

`NeighborLoadConfig` 的 docstring（`system.py:203-206`）自己写着：

> **当前只支持全网配同一个负载值**……ChannelHub 的几何 SINR 只给出**聚合**的 SIR——拿不到"哪个邻区贡献了多少"，没法把逐小区负载映射回来。

### 4.2 这个判断只对了一半

**"几何 `sir_dB` 是聚合量"是对的。但"拿不到逐邻区贡献"是错的。**

我把 ChannelHub 的几何 SINR 拆开了（`_system_sinr.py:416-500`）：

```python
rx_lin = 10.0 ** (rx_power_dbm / 10.0)                    # 逐小区，[K]
s_dl = rx_lin[serving_idx] * N_ant * bf_srv               # 服务：最优波束
for t in range(num_slots):
    i_dl = 0.0
    for k in range(K):
        if k == serving_idx: continue
        sel = rng.choice(B, size=min(n_dl_sched, B), replace=False)
        avg_leak = float(np.mean(bg[k, sel]))             # 从码本里随机抽几个波束取平均
        i_dl += rx_lin[k] * N_ant * avg_leak              # ← 逐小区贡献，在这里被加掉
```

所以

$$\mathrm{SIR} = \frac{\mathrm{bf}_{srv}}{\sum_{k \neq srv} \frac{rx_k}{rx_{srv}} \cdot \overline{leak}_k}$$

**关键问题：`avg_leak_k` 是不是跟方向有关？如果无关，那逐小区的份额就完全由 `rx_k` 决定，而 `rx_k` 我们拿得到。**

**E12（实测，决定性）** 我在 `compute_geometry_sinr_single_ue` 外面包了一层，把每个小区在整个码本上的平均波束增益 `mean_b bg[k,b]` 抓出来（4 个样本，7 站 21 小区，8×4×2 面板，码本 32×64）：

```
srv= 9  interferer mean_b bg[k]: min=0.1250 max=0.1250 std/mean=0.0000
srv= 5  interferer mean_b bg[k]: min=0.1250 max=0.1250 std/mean=0.0000
srv=11  interferer mean_b bg[k]: min=0.1250 max=0.1250 std/mean=0.0000
srv=12  interferer mean_b bg[k]: min=0.1250 max=0.1250 std/mean=0.0000
```

**逐位相同的 0.1250，跨小区标准差恰好 0。** 这是 Parseval：对一个完备的 DFT 码本，`Σ_b |c_b^H a|²` 只取决于 `|a|²`，与 `a` 的方向无关。

**结论：逐小区的干扰份额在期望意义上精确等于 RSRP 份额。**

$$w_k = \frac{P_k}{\sum_{j \neq srv} P_j}, \qquad P_k = 10^{rx\_power\_all\_dbm[k]/10}$$

**逐小区干扰合成需要的额外数据只有一个数组：`meta["rx_power_all_dbm"]`，K 个 float，约 200 字节/样本。不需要 `h_interferers`。**

#### E14：一个必须先处理的口径问题

`avg_leak_k` 的**期望**是常数，但**每次实现**不是——它是从码本里抽 `n_dl_sched` 个波束的平均：

```python
# internal_sim.py:2494
ues_per_cell=max(1, self.num_ues // K)
# _system_sinr.py
n_dl_sched = max(1, round(ues_per_cell * pdsch_load))
```

默认预设 `company_64t4r_multicell` 是 `num_ues=21`、K=21 ⇒ `ues_per_cell = 1` ⇒ **`n_dl_sched = 1`，每个干扰小区只抽一个随机波束**。

**E14（实测，n=42 样本，7 站 21 小区，UMa_NLOS，isd 500 m）**

| 量 | 值 |
|---|---|
| `SIR_geo − SIR_rsrp` 均值 | **13.19 dB** |
| 标准差 | **4.74 dB** |
| 极差 | 1.3 ~ 21.6 dB |

这个差就是 `10log10(bf_srv / avg_leak_realized)`。`bf_srv` 实测在 0.95~1.00（最优波束，几乎不变），所以**这 4.74 dB 的标准差全部来自"邻区波束抽签"**，不是路损、不是阴影、不是小尺度衰落。

顺带**独立复现了 CLAUDE.md 的「负载类旋钮在下行完全不起作用」**：`pdsch_load` 取 0.5 与 1.0，42 个样本的 `sir_dB` **逐位相同**。原因现在很清楚——`max(1, round(1 × 0.5)) = max(1, round(1 × 1.0)) = 1`，两边都是 1 个波束，连抽签次数都一样。

**这对 P1-C 的影响**：如果把每个样本的 `sir_dB` 当成"这个位置的真实干扰"，再乘上逐小区活跃度，那就是在一个已经含 ±4.7 dB 抽签噪声的数上叠加更多随机性。**两条路可选，我建议第一条：**

* **（推荐）用 RSRP 域重建 SIR**：`SIR_model = SIR_rsrp + Δ`，Δ 取实测均值 13.19 dB（或按 `10log10(bf_srv/0.125)` 逐样本算，bf_srv 需要新钩子）。好处是逐小区份额与总量**内部自洽**，方差干净。代价是与今天的 `sir_dB` 不逐位一致，需要一次口径迁移。
* **（保守）仍用 `sir_dB` 锚定总量，用 RSRP 份额做拆分**。好处是与现有 `interference_report` / IoT 口径完全兼容。代价是拆分与总量在单个样本上不严格自洽（只在期望上自洽）。

**无论选哪条，都必须把 `SIR_geo − SIR_rsrp` 的分布报出来**——它是"这批数据的干扰有多少是抽签噪声"的直接度量。如果哪天有人把 `num_ues` 调到 `21×32`，这个标准差会塌下去，那时两条路合一。

#### E16：干扰高度集中，这是个好消息

同一批 42 个样本：

* 有效干扰小区数 `n_eff = 1/Σw_k²` 均值 **2.68**
* **最强的 3 个邻区占了 86.9% 的干扰功率**

所以只建模 top-3~5 个邻区就能覆盖近 90%。丢掉尾部 13.1% 会让 SIR 偏高 `10log10(1/0.869) = **0.61 dB**`——**可以精确预估、可以在结果里补偿或标注**。

### 4.3 做不到的事（诚实清单）

**（1）逐 TTI 的邻区波束方向做不到。** 几何模型把"邻区在给谁打波束"抽样了一次就冻结进 `sir_dB` 了。要逐 TTI 控制，只能自己拿 `h_interferers` 算 `H_k w_k`。

**但这条路不值得走**，理由是 CLAUDE.md 已经写死的一条事实：**ChannelHub 的单个干扰小区信道是秩 1 的**（σ₂/σ₁ 中位 4.0e-8），所以 `interference_model="precoded"` 与 `isotropic` **逐位相同**——"邻区打哪个波束"在当前信道模型下**根本没有可区分的空间结构**。花 20 倍数据量（实测 11.14 MB/样本 vs 服务信道 0.56 MB）买回来的是一个假自由度。

**如果哪天 ChannelHub 的干扰信道变成满秩了，这条结论要重新评估。** 在那之前，`max_per_ue_intf_cells` 保持默认 3。

**（2）邻区的"真实调度器"需要邻区自己的 UE，而数据集里没有足够的。**

**E19（实测）** `num_ues=21`、`num_samples=42`、21 个小区：只有 **14 个小区**有 UE 落在里面，每小区样本数 `[6,6,4,4,4,2,2,2,2,2,2,2,2,2]`。**7 个小区一个用户都没有。** 因为 `serving_idx = argmax(rx_power_dbm)`，UE 撒点是均匀的但归属是竞争的。

要每小区 8~10 个 UE × 21 个小区 × 每 UE 8 个快照，需要 `num_samples ≈ 1700`。按本机实测（272 RB / 14 符号，约 0.7~1.9 s/样本）单进程 **20~55 分钟**，并行路径可以摊薄。可行，但要专门生成一批，不是顺手就能跑的。

**所以 P1-C 要分两个阶段，而阶段一根本不需要邻区 UE。**

### 4.4 实现方案

#### 4.4.1 阶段 C1：逐小区活跃度过程（便宜，先做这个）

**思路**：不给邻区真调度器，给每个邻区一个**独立的话务活跃过程**，占空比等于它的 PRB 利用率。干扰按逐小区份额逐 TTI 合成。

**数据侧**（这是唯一的数据依赖，必须先做）

```python
# generate.py:32-39 附近，新增"逐小区数组字段"通道
_ARRAY_META_FIELDS = ("rx_power_all_dbm", "pathloss_all_db", "antenna_gain_all_db")
# 落盘为 payload[f"cellarr__{k}"]，形状 [N, K]
# 再补两个标量：serving_cell_index、num_cells
```

现在的 `_SCALAR_META_FIELDS` 通道只吃标量（`generate.py:_as_float`），数组会被 `_as_float` 变成 nan。**必须走新通道，不能塞进老的。**

`loader.py` 加访问器：

```python
@cached_property
def rx_power_all_dbm(self) -> np.ndarray | None:   # [N, K]
```

**代价**：K=21 时每样本 21 个 float64 = 168 字节，相对服务信道 0.56 MB **可以忽略**。老数据集没有这个字段，要能优雅降级（返回 None，并让上层报"这批数据不支持多小区，需重新生成"）。

**表侧：给 `UeLinkTable` 加一个干扰维**

这是唯一和两相架构真正冲突的地方，也是本方案的核心。

干扰不再是常数，`noise_power` 逐 TTI 变，`mmse_stream_sinr` 的结果就逐 TTI 变。**但 SVD 和 Type I 码本搜索与干扰无关**（`system.py:614` 的 `_type1_precoder`、`csi_aging.py:271` 的 `svd_precoder` 都只吃信道），只有 `noise_power` 这一个标量变。

所以：**在第一相里把最贵的部分（SVD + 码本）算一次，然后在一个 η 网格上重跑便宜的部分（`mmse_stream_sinr` 是 17 个 4×4 矩阵求逆）。**

```python
ETA_GRID = np.logspace(np.log10(0.02), 0.0, 12)     # 12 个点，0.02 → 1.0
UeLinkTable.sinr_db      : [snapshot, rank, eta]    # 从 [snapshot, rank] 变成三维
UeLinkTable.se_gnb       : [snapshot, rank, eta]
...
```

主循环：算出本 TTI 的 η(t)，在 `log10(η)` 上做**线性插值**（SINR 在 dB 域对 log η 很平滑），或直接取最近格点。插值是 2 次乘加，不是矩阵运算——**两相架构保住了**。

第一相成本估算：η 无关的部分（SVD、码本搜索）约占 70%（CLAUDE.md 说码本搜索单独就占 47%），η 相关的约 30%。12 个格点 ⇒ `0.70 + 12×0.30 = 4.3×`。1.94 s → **约 8.4 s**。可接受。

内存：`[12 UE, 8 snap, 4 rank, 12 eta]` × 若干个数组 ≈ 几十 KB。可忽略。

**主循环：逐 TTI 合成干扰**

```python
# 第一相尾巴上算好，[n_ue, K-1]，行归一
W = 逐小区干扰份额矩阵                  # w[u,k] = P_k(u) / Σ_j P_j(u)

# 每个 TTI
a = neighbor_activity[:, tti]           # [K-1]，0/1 或 0~1 的 RBG 占用比例
eta = W @ a                             # [n_ue] 一次矩阵向量乘，21×12 = 252 flops
# 然后逐 UE 在 eta 上插值查表
```

**这里有个必须警惕的点**：`eta` 是**逐 UE 不同的**（每个 UE 看到的邻区几何不同），不是全网一个数。这正是这个课题的价值——小区边缘用户的 η 由 1~2 个强邻区支配，中心用户的 η 接近所有邻区的加权平均。**静态标量模型把这两类用户按同一个 0.3 处理，这是最大的失真。**

**邻区活跃过程**：每个邻区一个独立的 on/off 马尔可夫链（或直接复用 `_Traffic` 的到达过程），占空比 = 配置的 PRB 利用率。参数：`neighbor_burst_len_tti`（突发长度，决定干扰的时间相关性）。**这个参数是这个课题的主变量**——突发长度趋于 0 时干扰在 TTI 尺度上白化，OLLA 追不上；突发长度很长时 OLLA 能跟住。

#### 4.4.2 阶段 C2：真正的多小区联合调度

在 C1 之上：

* 按 `serving_cell_index` 把数据集的 UE 分组到各小区（`system.group_samples_by_ue` 之后再分一层）
* 每个小区一份 `_Traffic` + 一份调度器状态 + 一份 OLLA 偏置
* TTI 主循环外层套一个小区循环；**每个 TTI 的顺序必须是：先所有小区各自做调度决策 → 再统一合成干扰 → 再各自判 ACK/NACK**。顺序反了就变成了"小区 1 知道小区 2 这个 TTI 要发什么"，是上帝视角。
* `a_k(t)` 不再是随机过程，而是小区 k 的调度器**真实**用掉的 RBG 比例

**性能**：21 个小区 × 27.8 µs/TTI ≈ **584 µs/TTI**，20000 TTI ≈ **11.7 s**。叠加 P1-A（3~5×）和 P1-B（1.6~2.5×）后约 **50~150 s**。再乘 8 次重复（`num_replications` 默认值）⇒ **7~20 分钟**。

**这个量级需要一个决定**：要么接受（多小区仿真本来就慢），要么把小区循环向量化（把 21 个小区的 PF 度量做成 `[21, n_ue]` 的矩阵一次算完）。**我倾向先接受，等真的慢到不能忍再优化**——过早向量化会让调度逻辑难读，而这一层的可读性比速度重要。

### 4.5 能回答哪些现在答不了的问题

1. **干扰的时间起伏对 OLLA 收敛的影响**。现在干扰逐 TTI 是平的，OLLA 收敛到一个固定偏置。真实干扰是突发的，OLLA 在追一个移动目标。扫 `neighbor_burst_len_tti` 就能量出"干扰突发长度 vs OLLA 稳态 BLER 偏离"。
2. **邻区话务与本区话务相关时会怎样**。忙时所有小区一起忙，边缘用户在最需要资源的时候受到最强干扰。这是现网的真实痛点，现在完全测不了。
3. **中心用户 vs 边缘用户对邻区负载的敏感度差异**。C1 就能答——`eta` 逐 UE 不同是自动的结果。
4. **"关掉最强的那个邻区能救回多少边缘用户"**（干扰协调、小区间协作的价值上界）。把 `a_k(t)` 对特定 k 置 0 即可。
5. **PRB 利用率从 10% 到 50% 的真实影响曲线**，而不是现在这条由解析式画出来的。

### 4.6 工作量与风险

| 阶段 | 文件 | 改动 | 行数量级 |
|---|---|---|---|
| C1 | `generate.py` | 逐小区数组字段通道 | ~50 |
| C1 | `loader.py` | 访问器 + 老数据集降级 | ~25 |
| C1 | 新 `multicell.py` | 份额矩阵、活跃过程、η 合成、诊断（含 E14 的抽签噪声度量） | ~280 |
| C1 | `system.py` | η 网格维、第一相重构（W 复用）、主循环插值 | ~170 |
| C1 | `validate.py` | 新体检项：数据集是否带逐小区 RSRP | ~30 |
| C1 | `server.py` | 参数 + 转述 | ~60 |
| C1 | `tests/` | | ~180 |
| **C1 合计** | | | **~800 行 + 一次数据集重生成** |
| C2 | `system.py` / `multicell.py` | 多小区 TTI 循环、逐小区调度器状态 | ~300 |
| C2 | `server.py` | 新工具 `sw_multicell_sim` | ~120 |
| C2 | `tests/` | | ~150 |
| **C2 合计** | | | **~570 行 + 一批大数据集（20~55 分钟生成）** |

**风险**

1. **E14 那 4.74 dB 的抽签噪声是最大的隐患。** 如果不先处理，C1 出来的所有"干扰起伏"结论里都混着一份与物理无关的抽签方差，而且**这份方差和真实起伏长得一模一样**，事后分不开。**必须在 C1 之前把它量出来并写进结果**。
2. η 网格的插值误差。缓解：网格加倍做敏感性（I-C3）。
3. **老数据集全部不支持 C1**（没有 `rx_power_all_dbm`）。要有清晰的降级路径和错误信息，不能报一个 KeyError 了事。
4. C2 的小区间时序（先决策后合成）写反了不会报错，只会让干扰"看起来"低一点。**只能靠 I-C6 那条因果自检拦。**

### 4.7 验证方法

* **I-C1（退化）** 所有邻区恒定全发（`a_k(t) ≡ 1`，等价 η ≡ 1）时，逐 TTI 合成的 SINR 必须与今天 `neighbor_prb_util=1.0` **逐位相同**。
* **I-C2（份额自洽）** `Σ_k w_k = 1`（浮点误差内），且 `−10log10(Σ_k w_k · 1)` 恢复出的 RSRP 域 SIR 与直接从 `rx_power_all_dbm` 算的**逐位相同**。
* **I-C3（网格）** 在 η 网格点上，插值结果必须与直接计算**逐位相同**；网格从 12 点加密到 24 点，全部 KPI 变化 < 1%。**超过 1% 说明网格太粗，别调阈值，加密网格。**
* **I-C4（抽签噪声的度量，E14 的常驻版）** 每次多小区仿真都报 `SIR_geo − SIR_rsrp` 的均值与标准差。默认预设下应当接近 **13.2 ± 4.7 dB**。**标准差 > 3 dB 就要在 notes 里告警**："这批数据的干扰有相当一部分是邻区波束抽签噪声，要压下去请把 `num_ues` 提高到 `num_cells` 的若干倍。"
* **I-C5（截断误差可预估）** 只保留 top-3 邻区 vs 保留全部，SIR 的差必须 ≈ `10log10(1/top3_share)`，默认数据上 **0.61 dB**。实测偏离超过 0.2 dB 说明份额算错了。
* **I-C6（因果，C2 专用）** 把某个小区的话务在**仿真中途**从 0 拉满，其邻区用户的 SINR 必须在**同一个 TTI 或之后**下降，**绝不能在之前**。做法是打一个阶跃并逐 TTI 比对。这条抓"先合成干扰后做调度决策"这类时序反转。
* **I-C7（凸性差，这是结论不是 bug）** 邻区各自独立 Bernoulli(η₀) 时，长时间平均的体验速率**不会**等于静态 `neighbor_prb_util=η₀` 的结果——因为 `1/(ηI+N)` 对 η 是凸的（Jensen），而吞吐对 SINR 是凹的，两个方向相反，净效应我**预测不了**。
  **这个差本身就是这个课题要交付的东西，必须显式报出来，不能当成实现误差去"修"。** 如果有人把它"修"成零，那说明他把逐 TTI 起伏又平均掉了，整个课题白做。
* **I-C8（逐 UE 的 η 确实不同）** 边缘用户（几何 SINR 低）的 `n_eff` 必须**显著低于**中心用户——边缘被 1~2 个强邻区支配。实测样本上 `n_eff` 跨度是 1.4~4.0，改完之后这个跨度必须在仿真里出现。**如果所有用户的 η 都一样，说明份额矩阵退化成了全局标量，等于什么都没做。**

### 4.8 依赖与顺序

* **C1 的数据侧改动（`generate.py` / `loader.py`）不依赖任何其他课题，可以立刻做**，而且是纯增量、不动任何现有行为。
* C1 的表侧改动（η 网格维）会碰 `build_link_tables`，**和 P1-A 的 `sinr_per_rbg_db` 改的是同一个数据结构**。两个一起改会得到 `[snapshot, rank, eta, rbg]` 四维表——**12 UE × 8 snap × 4 rank × 12 eta × 17 rbg = 78 336 个 float，仍然只有 600 KB，内存不是问题**，但一次改两个维度容易出错。
* C2 依赖 C1，且依赖一批新数据集。
* **C1 和 P1-B 完全无耦合**，可以并行。

---

## 5 建议实施顺序

```
① P1-C 的数据侧（generate.py + loader.py 存逐小区 RSRP）     ~75 行，半天
② E14 口径体检（量化抽签噪声，写进 validate + CLAUDE.md）    ~60 行，半天
③ P1-B（HARQ RTT + 进程池 + 时延 KPI + 修 S 时隙）           ~500 行
④ P1-A（包模型 + RBG 分配 + KPI 重定义）                      ~650 行
⑤ P1-C 的 C1（η 网格 + 逐小区活跃度）                         ~700 行
⑥ P1-C 的 C2（真多小区调度）                                  ~570 行 + 大数据集
```

**理由**

**① 和 ② 排最前，因为它们便宜、纯增量、且是后面所有事的前提。** ① 是往数据集里多存 168 字节/样本，不改任何行为，但**它决定了以后生成的数据能不能用于多小区**——越早做，越多数据集自动具备这个能力。晚做的代价是所有中间产出的数据集都要重生成。

② 是因为 **E14 那 4.74 dB 的抽签噪声现在就在污染已有的干扰类结论**，不只是 P1-C 的问题。而且它和 AUDIT 里那条自我更正（"+14% 其实是噪声"）是同一类病：**把实现的随机性当成了物理效应**。既然刚栽过一次，就该先把这个源头量出来。

**③ P1-B 排在 P1-A 前面**，虽然 AUDIT 把 FDM 排得更靠前。三个理由：

1. **P1-B 不改 KPI 口径，P1-A 改。** P1-A 会重定义 burst 边界和 PRB 利用率，历史数字全部作废。先做不动口径的那个，回归压力小得多。
2. **P1-B 的自检条件更硬。** `harq_rtt_enabled=False` 逐位退化、DDDSU 的 k1 手算表、BLER=0 时 RTT 无影响——三条都是二值判定。P1-A 的"体验速率会怎么变"我在 2.3.5 里连方向都不敢预测。
3. **P1-B 顺手修掉两个现存 bug**（重传查错 MCS、S 时隙按满下行算），这两个正在影响当前所有系统级数字。

**④ P1-A 排第三**，因为它虽然影响面最大，但也最容易被口径变更绊住。做的时候 ③ 已经把时延链建好了，"小包体验被高估多少"这个问题可以一次性答完整。

**⑤ C1 排在 P1-A 之后**，因为两者都要改 `build_link_tables` 的表结构。**串行做能保证每一步都有干净的 A/B 基线；并行做会得到一个四维表和两批同时红掉的测试。**

**⑥ C2 排最后**，理由和 AUDIT 一致（唯一需要架构重构的一条），再加一条新理由：**C1 做完之后，C2 还值不值得做是可以重新判断的**。如果 C1 的独立活跃过程已经把"干扰逐 TTI 起伏"的效应量出来了，而真调度器带来的额外相关性（邻区忙时本区也忙）可以用一个相关系数参数近似，那 C2 的边际收益就不大。**先做 C1，拿到数据再决定要不要 C2** ——这比一开始就承诺做 C2 更省。

### 三个课题之间谁挡谁

```
P1-C 数据侧 ①  ──┐（无依赖，最先做）
E14 体检     ②  ──┤
                  ├──> P1-B ③（独立）
                  │
                  └──> P1-A ④ ──> P1-C 的 C1 ⑤ ──> C2 ⑥
                              （同一个表结构，串行）    （需新数据集）
```

* **③ 与 ④⑤ 可以并行**（不同的人做也行），只要约定好谁先合并主循环。
* **④ 与 ⑤ 不能并行**，硬冲突在 `UeLinkTable` 的维度上。
* **⑥ 依赖 ⑤，且依赖一批 20~55 分钟才能生成的数据集**，规划时要把生成时间算进去。

---

## 6 本文引用的实测证据一览

方便复核。除注明外，配置均为 UMa_NLOS / CDL-C / 7 站 21 小区 / isd 500 m / 64 端口 8×4×2 / 4R / 2.6 GHz。

| 编号 | 结论 | 证据 |
|---|---|---|
| E1 | `re_per_tti` 恒为全带；`draw_rbg` 结果只影响字节数与 `is_small` | `system.py:1049, 872-877, 1226-1228`，grep 逐处核过 |
| E2 | `p_idle_tti` 只出现在 `expected_prb_util()`，不生成任何空闲 TTI | `system.py:71, 98`，grep 全文件 |
| E3 | 全带 vs 1 RBG 的 TBS 比 = 16.9~17.2；1500 B 的包在 MCS12/rank2 下只要 1 个 RBG | 实测，表 3 MCS 表 |
| E4 | 500 KB FTP3 文件 @ MCS20/rank4 全带 = 5 个 TTI | 实测 |
| E5 | `simulate` 20000 TTI / 12 UE = 0.56 s（27.8 µs/TTI）；`build_link_tables` = 1.94 s | 实测本机 |
| E6 | 逐 RBG SINR 在第一相已算出并被 `user_sinr_db` 丢弃 | `csi_aging.py:283-305` → `mumimo.py:44-71` |
| E7 | 每 UE 只有 1 个 HARQ 槽、立即重传、最多 3 次；重传 BLER 查的是 `tables[u].mcs` 不是实发的 `m` | `system.py:1062, 1161-1162, 1194` |
| E8 | S 时隙在主循环里按满下行处理，而 `dl_ratio` 按 0.7 报告；`dl_ratio` 不被主循环读 | `system.py:1083, 289-291`，grep 确认 |
| E9 | 系统级没有任何时延 KPI | grep `delay`/`latency`/`时延` |
| E10 | `meta` 有 `rx_power_all_dbm` / `pathloss_all_db` / `antenna_gain_all_db` / `serving_cell_index`（全 K 个小区），superwireless 一个都没存 | 实测 meta 键；`generate.py:32-39` |
| E11 | 几何 SINR 的 `i_dl` 是逐小区求和，`I_k = rx_lin[k]·N_ant·avg_leak_k` | `_system_sinr.py:416-500` |
| E12 | `mean_b bg[k,b] = 0.1250` 对每个小区**逐位相同**，跨小区标准差 0 ⇒ 干扰份额 = RSRP 份额（期望意义精确） | 实测，包了一层钩子抓 `bg`，4 个样本 |
| E13 | `n_dl_sched = max(1, round((num_ues//K) · pdsch_load))`；默认预设下 = 1 | `internal_sim.py:2494`、`_system_sinr.py` |
| E14 | `SIR_geo − SIR_rsrp`：均值 13.19 dB、**标准差 4.74 dB**、极差 1.3~21.6 dB，全部来自邻区波束抽签 | 实测 n=42 |
| E15 | `pdsch_load` 取 0.5 与 1.0，42 个样本 `sir_dB` 逐位相同（独立复现 CLAUDE.md） | 实测 |
| E16 | `n_eff = 2.68`，top-3 邻区占 86.9% 干扰功率 ⇒ 截断到 3 个的 SIR 误差 0.61 dB | 实测 n=42 |
| E17 | `h_interferers` 默认只存 3 个（`max_per_ue_intf_cells`，可配）；设成 20 时 11.14 MB/样本 vs 服务信道 0.56 MB | `internal_sim.py:1100`；实测 272 RB/14 符号 |
| E18 | `h_interferers` 按 `rx_power` 降序，可由 `rx_power_all_dbm` + `serving_cell_index` 精确还原映射；meta 里没有 `intf_cell_ids` | `internal_sim.py:2237`；实测 |
| E19 | 21 UE / 21 小区 ⇒ 只有 14 个小区有 UE，分布 `[6,6,4,4,4,2×9]` | 实测 n=42 |
| E20 | `h_interferers` 定标 = `sqrt(P_k/P_srv)`，RSRP 域，不含阵列增益 | `internal_sim.py:2119-2131` |

### 【存疑】清单

1. **TS 28.552 的 data burst 边界**是文件边界还是缓冲区忙期边界。我按后者设计（第 2.3.1 节），但没有标准原文核对。这条决定 P1-A 要不要改 burst 定义。
2. **I-A7 里"抹平频选后体验速率变化 < 3%"**是我的猜测，没测过。真做的时候要先测这个数再定阈值。
3. **P1-B 对体验速率的 +23% 高估估算**是纸面推导（5 TTI 传输 + 0.5 次 NACK × RTT），没有仿真验证。数量级可信，具体数字不要引用。
4. **C2 的性能估算（50~150 s）**是按 21× 线性外推的，没有实测。多小区循环里可能有我没预见的开销。
5. **E12 的 0.1250 是在 32×64 的 DFT 码本上测的**。如果有人换了码本尺寸或用了过采样码本，"跨小区常数"这条要重测——Parseval 只对完备正交码本严格成立。
