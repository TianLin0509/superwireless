# 链路自适应：谱效、吞吐、MCS 与 TDD

**什么时候读这一份**：用户要真实吞吐（Mbps）而不是谱效、问 MCS/CQI/BLER
是怎么来的、要 TDD 的 MCS 审计链，或者要画谱效/吞吐 vs SNR 曲线时。

## 领域事实层面的常见误读

| 心里的念头 | 实际情况 |
|---|---|
| "达成率只有 40%，算法不行" | 先看 MCS 分布：压在最高档就是表封顶，跟算法无关 |
| "表 3 算出 0.1%，所以所有配置都可靠" | 曲线只代表经典 MMSE 接收机的 SINR→BLER 映射，其他维度当前不建模 |
| "换 mcs_table=2 就一定更快" | 只有确实压在表顶时才有效。实测 29 dB 场景 1373 → 1690 Mbps，低信噪比处无差别 |
| "BLER 是 3GPP 实测的" | 表 1/2 是有限码长分析模型，表 3 是用户提供的曲线，两者都不是标准曲线 |

## 香农谱效不是吞吐 —— `sw_throughput`

`sw_link_performance` / `ds.link()` 给的是 `SE = Σ log2(1+SINR)`，**香农上界**。
真实系统达不到它，差 25~60%，原因有三条，都是可量化的：

1. **调制受限** —— 20 dB 时香农说 6.66 bit/s/Hz，64QAM 最多给 5.80
2. **码率离散** —— MCS 只有 29 档，实际码率总落在需要的码率之下
3. **有限码长 + 实现损失** —— LDPC 距容量 1~2 dB，码块越多 TB 越易错

`sw_throughput` 走业界做系统级仿真的标准路径（链路到系统映射：
有效 SINR → MCS/CQI → TBS → BLER → 吞吐），给出 **Mbps** 和
**5% 边缘用户吞吐**（3GPP 评估里的公平性指标，比均值更能说明问题）。

**返回里 `hint` 提示"大量样本压在最高档 MCS"时，一定转述给用户**——
那说明限制来自 MCS 表而不是信道或算法，换 `mcs_table=2`（含 256QAM）
通常直接提升 20% 以上。实测 29 dB 的场景：表 1 均值 1373 Mbps，表 2 1690 Mbps。

`sw_sweep_snr` 出**谱效/吞吐 vs SNR 曲线**，无线论文里最标准的那张图。
各点跑在同一批信道上、彼此配对，曲线不含信道抽样噪声。
达成率的走势最有信息量：低信噪比处 70~77%（受噪声限），
高信噪比处掉到 40% 以下（受 MCS 表封顶限）。

## 评价链路：谁出数、谁判决

- **`sw_link_performance` 出数** —— 一次调用在同一批信道上横评多个方案（默认
  svd / svd_wideband / type1 / dft，自研方案加进 `methods`），返回谱效均值、
  95% 置信区间与收敛判断。**只出数不过门**：均值差再好看，也不许直接写成结论。
  `use_estimated_csi: true` 是 CSI 反馈课题的核心对比——估计信道算预编码、
  理想信道评性能，量的是 CSI 误差的真实代价。
- **`sw_compare_arms` 判决** —— 横评筛出的决赛组合两两过它。

## MCS 表与 BLER 的口径边界

**一条必须守住的边界**：表 1/2 的 MCS/CQI/TBS 按 38.214 精确算，QAM 约束
容量精确求积，但 BLER 是**有限码长分析模型**。表 3 则是用户提供的 28 档 MCS +
56 条 NewTx/ReTx 解调曲线（1824 点），**不是 3GPP 标准曲线**。数据所有者已确认：
源标签 Es/No 就是经典 MMSE 接收机的 SINR；其他链路维度暂不参数化，
曲线范围外只能保守钳位，不能外推。

- `sw_mcs_info(table=1/2, show_bler_anchors=true)` —— 看分析模型门限
- `sw_mcs_info(table=3, show_bler_anchors=true)` —— 看用户曲线两套门限与哈希自检
- `sw_bler_curve(mcs=..., tx_mode="newtx"/"retx")` —— 取单档原始曲线或插值
- `sw_tdd_mcs(dataset_id=..., cqi=..., olla_mcs_offset=...)` —— TDD 最终 MCS 与逐流审计链

表 3 的 HARQ 首传用 NewTx、后续用 ReTx；只有一条 ReTx 曲线，因此多次重传会
复用它并显式标成假设（`harq_model=newtx_then_retx_curve_reused`）。
不要额外推断 TB/CB、块长、信道、层数或译码器的影响。
表 3 没有 CQI 曲线，CQI 仍走 38.214 Table 2 + 分析 BLER，由 `cqi_source` 明示。

## TDD 的 CQI、BF Gain 与 OLLA

用户要求 TDD MCS 或提到 CQI/BF Gain/OLLA 时，调用 `sw_tdd_mcs`，**不要在对话里
手算**。固定顺序是：`CQI → 按谱效映射初始 MCS → 该 MCS 的 NewTx 目标 BLER
SINR 门限 → + BF Gain → 重映射 MCS → + OLLA → floor → 钳位 0..27`。

- CQI 是 PMI 权测得的 **pre-BF** 索引，是**长期滤波的宽带量**；
  CQI0 不调度，但也不能当 MCS0——它的意思是"低于 CQI 表下界"，退回实测 PMI SINR
- BF Gain 是**瞬时量**，逐 RB、逐流计算 `post-MMSE SINR_SVD - post-MMSE SINR_PMI`
- 两条链路必须共用信道、CSI、rank、功率、噪声、干扰和经典 MMSE 接收机，
  只改变预编码权；**rank 不同不是 BF Gain**
- 用户 SINR 对全部 RB×流在 **dB 域做算术平均**，不做线性域平均或 MIESM
- OLLA 单位是连续 MCS 档位，不是 dB；正值更激进；先相加再严格向下取整
- 默认目标首传 BLER 10%，ACK +0.1、NACK -0.9；反馈只更新下一调度时刻

**发送侧 SINR 是 `Γ(MCS(CQI)) + BFGain`，不是接收 SINR 的均值。** 后者是个
事后诸葛亮的量——它已经包含了 SVD 的实际增益，等于假设基站预先知道波束打得准不准。
开 CSI 老化后这个错会变致命：老化的全部代价就是"基站以为打准了其实没有"。

转述结果时至少给出初始 MCS/门限、逐流 BF Gain、用户 SINR、BF 后 MCS、
OLLA offset 和最终 MCS；若 `clamped_low` / `mcs_clipped` 为真也必须说明。
