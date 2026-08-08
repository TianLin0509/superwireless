# superwireless

给 Agent 用的无线仿真信道供应站 —— **面向蒙特卡洛验证**。

你提一个无线算法优化思路，它给你可信的信道场景实例、配套的物理观察量，
以及 SINR / 谱效的完整评价链路。复用
[ChannelHub](https://github.com/wangxz0803-lab/ChannelHub_main) 的物理内核，
通过 MCP 向任意 Agent 开放。

配套的 `channel-sim` skill 提供 superpowers 式工作流：
**头脑风暴 → 计划书 → 生成 → 门 1 体检 → 跑实验 → 门 2/门 3 → 结论**。

## 七件事

**一、信道可信。** 18 项体检，分四类：对 3GPP 标准（路损逐点对 38.901、
**CDL 剖面逐簇对 Table 7.7.1-x**、Annex A.1 角度扩展）、对物理定律（时频能量守恒、
谱效不超容量上界、SISO 退化到香农）、对配置（场景与剖面视距类别、小区数是否被
栅格吸附、**干扰是否真的进了 SINR**）、对统计（收敛、信噪比覆盖）。
**不通过不会静默**，会告诉你哪里不可信、偏了多少、怎么改。

```python
print(ds.gate().text())      # 门 1：18 项，含实测偏差与容差依据
print(ds.calibrate().text()) # 3GPP §7.8 口径的校准量，对 R1-165975 参考曲线
```

**二、信道多样。** 5 个传播场景（UMa/UMi 各含视距与非视距、InF）× 10 个信道剖面
（CDL-A~E 有每径角度、TDL-A~E 无）× 任意小区数 × 10 个真实城市射线追踪场景。
上层没问到的参数一律**原样透传**——`internal_sim` 共 44 个、`sionna_rt` 49 个。

**23 个场景预设分 10 组**，每个都真跑过并把实测特征写在清单里，
不是只给个名字让你猜：

```python
# 干扰场景 · 测量干扰 · 大站间距 · 移动性 · 高铁 · 传播条件 · 多小区干扰 · 基线 · 射线追踪 · 室内与专网
r = sw_probe_scenario(preset="high_iot_dense", num_samples=63)   # 几十秒，不是几十分钟
# 干扰画像、链路预算、路损/距离/视距/多普勒分布
```

**探测模式把 `num_rb` 压到 24、`num_ofdm_symbols` 压到 4，几何量与全量逐位相同**
（实测 num_rb 273/24/12 与 nsym 14/7/4/2 各档，SINR/SIR/路损/距离/视距/多普勒/
UE 位置全部零差异），**11.5 倍速**（2602 → 226 毫秒/样本）。
唯一变的 `snr_dB` 有解析修正。nsym 在 1 处有悬崖（偏 16.1 dB），所以取 4 不取 2。
给不了谱效与吞吐，返回里会列清楚。

**三、谱效开箱即用。** 预编码 → 逐层 SINR → 频谱效率的完整链路，
含 SVD（理想上界）、宽带 SVD、38.214 Type I 码本、DFT 波束四种方案的横向对比。

```python
mc = ds.monte_carlo(method="svd")
print(f"{mc.se_mean:.2f} bit/s/Hz  收敛={mc.converged}")

for name, v in ds.compare_precoders().items():
    print(f"{name:<14}{v['se_mean']:6.2f}  (SVD 的 {v['vs_svd_pct']:.0f}%)")
# svd            30.54  (SVD 的 100%)
# svd_wideband   20.44  (SVD 的  67%)   ← 宽带损失
# type1          17.41  (SVD 的  57%)   ← 码本量化 + 秩自适应
# dft            11.19  (SVD 的  37%)   ← 单层波束
```

**四、结论守得住。** 三道评审门拦住站不住的结论。**信道对了结论照样可以是错的**
——两组配置除被测变量外还有别的不同、一边用理想 CSI 另一边用估计 CSI、
样本量不足到置信区间比效应还宽、只比均值没做检验。

```python
r = ds.compare_arms({"name": "我的方法", "method": "svd_wideband", "csi": "estimated"},
                    {"name": "基线",     "method": "type1",        "csi": "estimated"})
print(r.statement())
# 我的方法 相对 基线：谱效 20.932 vs 13.177 bit/s/Hz，差值 +7.755（+58.9%），
# 95% CI [+6.989, +8.521]，n=200，Wilcoxon 符号秩检验 p=2.72e-31。结论成立。
```

两臂跑在同一批信道上 → 天然配对，共同的场景起伏被差分抵消，
**样本量需求常比非配对少一个数量级**。门 2 拦口径不公平，门 3 拦统计站不住，
过不了门时 `statement` 自己会写"结论不成立"及原因。

**五、自研算法也进得来。** 上面的 `method` 只认六种内置预编码。你自己的
CSI 压缩、信道估计、波束管理、调度算法走结果契约：

```python
# 1. 生成前锁口径（预注册），生成时绑上
pr = sw_lock_analysis(primary_metric="spectral_efficiency", baseline="type1")
ds = sw_generate(..., prereg_id=pr.prereg_id)

# 2. 导一份评测脚本，把 my_algorithm 换成你的算法（不改也能跑）
code = sw_export_eval_template(dataset_id)["code"]

# 3. 你的脚本里注册结果 —— MCP 不执行你的代码，只收标准化的逐样本值
art = ds.register_results("我的方法", values, metric="spectral_efficiency",
                          method_metadata={"csi": "estimated"})

# 4. 交给 MCP 判决，与内置方案用同一套统计与门控
sw_compare_results(art_a.result_id, art_b.result_id)
```

注册时锁死数据集内容摘要、样本 ID **逐个按序**比对、指标与单位——
配对检验的有效性全靠"第 i 个数对应同一个信道实例"，**错配时它照样会算出
一个看起来很显著的 p 值**。

**六、香农谱效不是吞吐。** 上面的 `se_mean` 是上界，真实系统达不到。
走 38.214 的链路到系统映射，把三项真实损失算进来：

```python
st = ds.throughput(mcs_table=1)
print(st.text())
# 吞吐（n=60）：均值 1373.11 / 中位 1474.69 / 边缘用户(5%) 833.48 / 峰值(95%) 1606.61 Mbps
#   谱效 均值 19.052，边缘 11.565 bit/s/Hz
#   平均 BLER 0.38%，中断比例 0.00%
#   MCS 分布 {18: 1, 20: 1, ..., 28: 34}    ← 34/60 压在最高档 = 表封顶
```

三项损失：**调制受限**（20 dB 时香农 6.66，64QAM 只给 5.80）、**码率离散**
（MCS 只有 29 档）、**有限码长 + 实现损失**（LDPC 距容量 1~2 dB）。
QAM 约束容量用 Gauss-Hermite 求积**精确算**，表 1/2 的 MCS/CQI 与 TBS 按
38.214 精确复刻；默认 BLER 是有限码长分析模型（没有 3GPP 参考曲线兜底）。

现在另有显式可选的 `mcs_table=3`：28 档用户 MCS profile + 56 条 NewTx/ReTx
解调曲线（1824 点）。它用曲线选 MCS，HARQ 首传后切到 ReTx 曲线：

```python
st = ds.throughput(mcs_table=3)
sw_bler_curve(mcs=15, tx_mode="newtx", sinr_db_list=[14.0, 14.05])
# BLER = [0.132, 0.0949]，10% 门限 14.042 dB
```

表 3 **不是 3GPP 标准表**；源脚本标签 `Es/No` 已确认表示经典 MMSE 接收机
的 SINR。TB/CB、块长、信道模型、MIMO 层数和译码器细节暂不参数化。
`sw_mcs_info(table=3, show_bler_anchors=true)` 可查看全部门限、码率和哈希自检。

TDD AMC 已支持完整的 `CQI → 初始 MCS → NewTx SINR 门限 → BF Gain →
重映射 MCS → OLLA → floor` 链路。BF Gain 是同一信道、CSI、rank、功率、
干扰和经典 MMSE 接收机下，SVD 权相对 PMI 权的逐 RB、逐流 post-MMSE SINR
差值；用户 SINR 对全部 RB×流在 dB 域做算术平均：

```python
sw_tdd_mcs(dataset_id="ds_xxxxxxxx", cqi=9, olla_mcs_offset=-0.2,
           feedback_ack=False)
```

在 Claude Code / Codex CLI 里不需要自己写 Python，直接告诉 Agent：
“请调用 superwireless 的 `sw_tdd_mcs`，对数据集 `ds_xxxxxxxx`、CQI 9、
OLLA -0.2 MCS 计算最终 MCS，并解释逐流 BF Gain。”Agent 会调用 MCP 并返回
完整中间量。`CQI=0` 明确不调度；反馈只更新下一时刻 OLLA，当前决策不回写。

`sw_sweep_snr` 出谱效/吞吐 vs SNR 曲线——实测低信噪比达成 77%、
高信噪比因 MCS 封顶掉到 38%。

**六点五、干扰强度用 IoT 说话。** "高干扰"是个数不是形容词。
IoT（噪声抬升 `(I+N)/N`）由几何 SIR 与 SINR **精确推出**——
不是 `snr_dB - sinr_dB`，那两个字段口径不同，相减差几十 dB。

```python
sw_interference_report(dataset_id)
# traffic_domain.dl.iot   → 28.3 dB，高干扰，等效负载 0.9985
# measurement_domain.ul_srs → SIR -10.5 dB，测量已失效，NMSE 底 10.5 dB
```

**业务域和测量域是两回事。** 实测一组对照：`srs_congested` 与
`srs_clean_reference` 只差导频配置，业务域 IoT 差 **0.06 dB**（噪声），
SRS 测量域 SIR 差 **17.9 dB**（−10.50 vs +7.37）。
只看业务域 SINR 会认为这两个场景是同一件事。

哪些旋钮真的能动 IoT 是**实测过的**，`sw_design_interference` 会给出实测值——
其中两条与直觉相反：`pdsch_load` 对下行 IoT **完全无效**（0.2 与 1.0 逐位相同），
`num_interfering_ues` 影响的是测量域而非业务域上行 IoT。

**七、跑得快。** `workers="auto"` 按配置预估耗时自动决定要不要多进程。
实测单样本耗时差 85 倍（单小区 32T/20MHz 24 ms，21 小区 64T/100MHz 2054 ms），
重配置 200 样本 **842s → 246s**；轻配置自动走串行，因为起进程的开销更大。

`collect_ssb=False` 再省约 30%（交错重测中位数 3456 → 2475 ms/样本）。
**比较耗时必须交错重测**：本机基准的轮间波动就有 11.9%，
顺序跑变体会把预热效应读成"加速"——第一版基准正是这么测出一个假的 2.55×。

样本数是**算出来的**，不是问用户的：

```python
sw_sample_size(std_diff=2.14, expected_effect=1.5)   # → 需要 64 个样本
sw_sample_size(std_diff=2.14, n_current=20)          # → 最小可检出 2.70 —— 比预期还大，白跑
```

## 交互方式

```
你：帮我验证一个 CSI 压缩的想法，先弄一批单小区 64T4R 的信道数据。

Agent：配好了 64T4R、273 RB、CDL-C、100 MHz。
       第 1 轮 · 实验设计 —— 参数配错重跑就行，实验设计错了结论作废。

       ① 你的方法要跟什么比？
          1) 3GPP Type I 或 Type II 码本 —— 最常见的基线   ← 推荐
          2) 理想信道的 SVD 预编码 —— 理论天花板
          3) 某篇已发表方法 / 4) 还没定，先看可行性
       ② 用什么指标？
          1) 重建精度类：NMSE / 余弦相似度                ← 推荐
          2) 系统收益类：频谱效率或吞吐损失
          3) 任务专属：波束命中率 / 定位误差 / BLER
       或者你直接说。
```

**一轮 2~4 个问题，每题 3~4 个选项并标明推荐**，最后留"或者你直接说"。
典型 2~3 轮收敛，用户随时可以说"随便"直接生成。轮次由 MCP 自己记，
Agent 不用规划；`has_more_rounds` 为 false 或用户说"随便"就停。

设计参考 [superpowers](https://github.com/obra/superpowers) 的 brainstorming，
按仿真场景做了调整——它面对开放式设计所以一次一问，而仿真参数空间有限且已知。

## 文档

- **[安装说明 `SETUP.html`](SETUP.html)** —— 由哪几块拼成、要装什么、怎么装、装完先跑什么、排错
- **[`INSTALL_AGENT.md`](INSTALL_AGENT.md)** —— 写给 AI agent 看的安装步骤，丢给它自己装
- **[能力手册 `CAPABILITIES.html`](CAPABILITIES.html)** —— 能产生哪些信道、能拿到哪些观察量（含形状与单位）、参数全表、能力边界
- **[实测场景演示 `SHOWCASE.html`](SHOWCASE.html)** —— 真实跑过的场景对话、三道门、踩过的坑
- **[接入自研算法 `EXTERNAL_ALGO.html`](EXTERNAL_ALGO.html)** —— 让你自己的算法进门 2/门 3、预注册分析口径、边界与局限
- **[从 SINR 到真实吞吐 `LINK_ADAPTATION.html`](LINK_ADAPTATION.html)** —— L1 链路自适应、38.214 MCS/CQI、SNR 扫描曲线、并行生成
- **[测试体系说明 `TESTS.html`](TESTS.html)** —— 「635 项测试」是什么、拦住过哪 8 个真实事故、以及它**证明不了**什么
- **仿真说明书** —— `sw_spec_sheet` 出的 HTML，**默认直接在浏览器里弹出来**：拓扑图打头、其余折进页签、还能在上面改参数点「应用到仿真」把改动送回 agent（`sw_await_config` 接）。落在 `artifacts/specs/`，拷走用 `file://` 打开时自动退回复制粘贴
- **[MU-MIMO 算法流程 `MU_MIMO.html`](MU_MIMO.html)** —— 配对/预编码/功率分配逐步展开，含六个待确认的设计选择与实测数字
- **[通宵成果与待审 `TONIGHT.html`](TONIGHT.html)** —— 6 个 bug、5 个新需求提案、8 个待拍板的决策点
- **[通宵进展与待审问题 `MORNING_REVIEW.html`](MORNING_REVIEW.html)** —— 3GPP/ITU 对标结果 + 12 个待拍板的问题
- **[还缺什么 `ROADMAP.html`](ROADMAP.html)** —— 对着 Sionna / MATLAB 5G Toolbox / QuaDRiGa / 5G-LENA 逐模块点名。**只下行 · 只 TDD · BLER 一律查表**，边界写在第七节
- **[场景拓展与干扰量化 `SCENARIOS.html`](SCENARIOS.html)** —— IoT 噪声抬升、业务域 vs 测量域、21 个场景的实测画像、场景探测、哪些提速是真的

## 四条设计铁律

**一、不传数据，传取货代码。** 单个信道样本几百 KB，序列化成 JSON 会膨胀到
十几 MB——进不了任何模型的上下文。MCP 只回句柄、统计摘要和可运行的 Python。

**二、给物理量，不给训练特征。** ChannelHub 的 `data/bridge.py` 会把这些量
归一化、截断、乘门控后打包成 16 个 MAE token——那是为表征学习服务的。
本项目**绕开它**：PDP 不归一化、RSRP 不截断、SRS 给完整协方差和全部特征值、
PMI 给码本索引而非嵌入向量。

**三、生成与取货解耦。** 测量量从信道现算，改主意重新取货**实测 1 毫秒**，
不重跑仿真。

**四、分轮问，先设计后参数；能算的不问。** 样本数由期望效应量和试点方差算出来，
不问用户"你想跑多少次"——把自己该做的功课推回去是这类协作最常见的偷懒。

## 拦截"跑得出结果但没意义"的组合

| 组合 | 为什么拦 |
|---|---|
| 波束搜索 + TDL 模型 | TDL 没有每条径的角度，算法会输出看似正常的垃圾且不报错 |
| 信道预测 + 单时隙 | 样本间相互独立，没有可预测的时序结构 |
| 干扰协调 + 单小区 | 没有干扰源 |
| 视距场景 + 非视距剖面 | 路损与多径按不同假设生成，时延扩展偏离标称值数倍 |
| 射线追踪数据 + `ds.paths()` | 多径来自真实建筑几何，套用 CDL 剖面会得到与数据无关的假角度 |
| 多小区但 SINR = 纯热噪声 SNR | 干扰没进计算，干扰类结论全不成立 |
| 一臂理想 CSI、另一臂估计 CSI | 增益里混着"提前知道答案"的部分 |
| 置信区间跨零却说"有提升" | 方向都不能确定 |
| 把香农谱效当吞吐报 | 真实系统要打 4~6 折，差的是调制受限+码率离散+码长 |
| 声称实测 BLER | 表 1/2 是分析模型；表 3 是用户曲线插值，二者都不是 3GPP 实测 |

## 安装

### 最省事：让 agent 自己装

把这句话发给你的 Claude Code / Codex：

> 帮我装 superwireless：读 https://github.com/TianLin0509/superwireless/blob/main/INSTALL_AGENT.md
> 按里面的步骤装好并验证，装完告诉我能不能用。

[`INSTALL_AGENT.md`](INSTALL_AGENT.md) 是**写给 agent 看的**：每步带验证命令与预期输出，
标了哪些事该问你、哪些该自己查，附失败对照表。

### 内网 / 不能联网

在一台能联网的机器上打包，拷进去：

```bash
python scripts/make_offline_bundle.py          # 完整包 65 MB，全新 venv 可全程离线装
python scripts/make_offline_bundle.py --thin   # 轻量包 17 MB，要求目标机已有 numpy/scipy
```

产出 `dist/superwireless-offline-<包型>-<平台>-py<版本>.zip`，里面有源码、skill、
依赖 wheel、`bundle-manifest.json`（各文件 SHA-256）、`INSTALL_AGENT.md`
和给人看的 `开始安装.txt`。接收方解压后把那句话发给自己的 agent 即可。

**默认打完整包。** 轻量包不含 numpy/scipy 与构建后端，在全新 venv 里
`pip install --no-index -e .` 会失败（先卡在缺 setuptools，而报错只说
"install build dependencies did not run successfully"，看不出缺什么）。
包型写进了文件名和 manifest，`requires_preinstalled` 直接列出需自备什么。

**wheel 是平台相关的**，必须在与目标机器同平台、同 Python 大版本的机器上打包。

> 包里**不含 ChannelHub** —— 该仓库没有开源许可证，默认保留所有权利，
> 随包转发有法律风险。接收方需自备一份含 `src/msg_embedding/data/contract.py`
> 的源码树。确认自己有权分发时用 `--include-channelhub <路径>`。

### 手动

需要 Python ≥ 3.10。

```bash
git clone https://github.com/wangxz0803-lab/ChannelHub_main   # 物理内核
git clone https://github.com/TianLin0509/superwireless
cd superwireless && pip install -e .

pip install sionna-rt      # 可选，射线追踪（约 300 MB）
```

ChannelHub 会自动在同级/上级目录查找；放在别处就设 `SUPERWIRELESS_CHANNELHUB`。
不装射线追踪也能用，`sw_capabilities` 会如实报告缺什么。

```bash
claude mcp add superwireless -- python /path/to/superwireless/scripts/mcp_server.py
codex  mcp add superwireless -- python /path/to/superwireless/scripts/mcp_server.py

cp -r skills/channel-sim ~/.claude/skills/     # 可选：工作流编排
cp -r skills/channel-sim ~/.codex/skills/
```

## 评审门控

| 门 | 什么时候过 | 拦什么 |
|---|---|---|
| **门 1 · 信道可信** | 生成之后 | 18 项体检，硬性项不通过即拦截 |
| **门 2 · 比较公平** | 跑对比时 | 两臂不同数据集、配置漂移、**CSI 口径不一致** |
| **门 3 · 结论站得住** | 写结论前 | 置信区间跨零、检验不显著、单样本主导、声称值超出区间 |
| **预注册身份** | 写结论时 | 用的指标不是事先定的 → 标 `exploratory`，不许冒充主结论 |

门 3 的显著性**以 Wilcoxon 符号秩检验判决**，配对 t 只作参考——谱效的逐样本差值
分布常是偏的，t 检验的正态假设不成立、小样本下偏乐观。两个检验冲突时 `statement`
会把冲突明写出来。

门 2 的 CSI 口径检查是无线论文评审最常抓的一条——自己的方法用理想信道预编码、
基线用估计信道，测出来的"增益"里混着"提前知道答案"的部分。

3GPP 口径的校准量按 **TR 38.901 §7.8** 出：耦合损耗 CDF（§7.8.1 指标1）、
几何量含噪与不含噪两条（指标2）、时延与角度扩展 ASD/ASA/ZSD/ZSA
（§7.8.2 指标3，Annex A.1 圆周定义）、PRB 奇异值最大/次大/比值三条 CDF
（指标4，10log10 尺度）。参考曲线在 R1-165974 / R1-165975 / R1-1909704。

## MCP 工具（34 个）

| 工具 | 作用 |
|---|---|
| `sw_capabilities` / `sw_list_presets` / `sw_list_scenes` | 能力与场景发现 |
| `sw_missing_slots` | **结论模板还缺哪些槽** —— 决定该主动问什么 |
| `sw_plan` / `sw_revise` | 分轮协商：实验设计 + 参数 + 对比组 + 陷阱 |
| `sw_generate` | 生成数据集，返回句柄与统计摘要 |
| `sw_deliver` | 按自然语言点单生成取货代码 |
| `sw_validate` / `sw_gate` | **可信度体检 / 门 1**：17 项 |
| `sw_calibrate` | **3GPP §7.8 校准量**：耦合损耗、几何、时延角度扩展、PRB 奇异值 |
| `sw_link_performance` | **算谱效**：预编码 → SINR → 谱效，多方案横向对比 |
| `sw_compare_arms` | **配对比较 + 门 2 + 门 3**，返回可直接引用的结论句 |
| `sw_sample_size` | **功效分析**：样本数 ↔ 最小可检出效应 |
| `sw_lock_analysis` | **预注册**：生成前把主指标与基线定下来 |
| `sw_export_eval_template` | **自研算法评测脚本骨架**，替换一个函数即可 |
| `sw_compare_results` | **判决外部算法结果** + 门 2 + 门 3 + 预注册身份 |
| `sw_list_results` | 已注册的结果与预注册记录 |
| `sw_throughput` | **真实吞吐 Mbps** + 5% 边缘用户（链路到系统映射） |
| `sw_sweep_snr` | **谱效/吞吐 vs SNR 曲线**，各点配对无抽样噪声 |
| `sw_mcs_info` | 表 1/2：38.214 + 分析模型；表 3：用户 MCS + NewTx/ReTx 门限 |
| `sw_bler_curve` | 查单档原始 BLER 曲线、10% 门限，并在任意 SINR 点做对数域插值 |
| `sw_tdd_mcs` | **TDD AMC**：CQI → PMI/SVD BF Gain → MCS → OLLA，返回逐 RB/流审计链 |
| `sw_system_sim` | **系统级仿真**：连续几秒 TTI + PF 调度 + 话务，出体验速率等现网 KPI |
| `sw_spec_sheet` | **仿真说明书**：拓扑图 + 分级页签 + 调参面板，**默认自己弹浏览器** |
| `sw_await_config` | 等用户在说明书上点「应用到仿真」，**改动直接回来**，免复制粘贴 |
| `sw_describe_dataset` / `sw_list_datasets` | 数据集信息 |

## 观察量（12 类）

| 名称 | 内容 |
|---|---|
| `channel` | 频域信道矩阵，理想与估计两版 |
| `linkperf` | **链路性能**：预编码、逐层 SINR、谱效、容量上界、多方案对比 |
| `validate` | **可信度体检**：18 项检查 |
| `pdp` | 时延功率谱：未归一化功率 + 真实时延轴 + RMS 时延扩展 |
| `paths` | 每条径的时延、功率、角度（**CDL 才有角度**）|
| `srs` | 完整空间协方差、全部特征值、每天线增益、波束域 RSRP |
| `pmi` | 38.214 Type I 码本索引 + 预编码矩阵 + 秩 |
| `rsrp` / `sinr` / `capacity` | 功率、链路标量、容量与条件数 |
| `geometry` | 路损、阴影、3D 距离、视距判定、多普勒、位置 |
| `topology` | 多小区 SSB 测量与干扰小区信道 |

## 物理层工具箱

`superwireless.physical` 转发 ChannelHub 里已按 38.211/38.213/38.214 实现的模块，
主要用来**当基线**和**做导频层课题**：

```python
from superwireless import physical as ph

ph.nr_rb_count(100e6, 30000)       # 273（标准表，不是简单除法）
ph.tdd_pattern_info("DDDSU")       # 帧结构 + 特殊时隙符号级切分
ph.srs_config(273, b_srs=1)        # SRS 跳频：周期 17、每跳 16 RB、覆盖 6%
ph.zadoff_chu(25, 139)             # ZC 序列，实测峰旁比 151 dB
ph.ssb_sequences(42)               # PSS / SSS / PBCH-DMRS
ph.dft_codebook(8, 4, 2)           # CSI-RS 波束码本 [512, 64]
ph.estimate_channel(h, method="mmse", tau_rms_s=363e-9)   # LS / MMSE 估计基线
ph.project_interference(...)       # 干扰投影：不投影会高估干扰
```

## 场景与参数

**传播场景**：城区宏站视距/非视距 · 城区微站视距/非视距 · 室内工厂
**信道剖面**：CDL-A~E（有每径角度）· TDL-A~E（无角度）
**拓扑**：任意站数 × 扇区数（1 或 3），支持六边形栅格与线性布站、
超级小区、多 TRP、高铁车体穿透、自定义站点与用户坐标
**射线追踪**：慕尼黑 · 巴黎凯旋门 · 佛罗伦萨 · 旧金山 · 北京中关村 ·
上海陆家嘴 · 深圳福田 · 广州天河 · 杭州钱江 · 重庆解放碑
**子载波间隔**：15 / 30 / 60 / 120 kHz　**带宽**：5~100 MHz 共 13 档
**TDD 配比**：7 种　**支持任务**：12 类

加场景只改 `presets/presets.yaml`，加决策点只改 `decisions.py`。

## 已知约束

- **信噪比不能直接设定**。它由路损、发射功率和撒点位置决定；要求特定区间时
  走拒绝采样。想整体调整，改发射功率或站间距更有效。
- **视距比例由几何决定**，不是选 CDL-D 就能得到视距信道——剖面类别与几何
  判定不符时会被自动替换。想调视距比例改站间距（实测 200m→0.46、800m→0.13）。
- **射线追踪拿不到逐径几何**。ChannelHub 尚未导出 Sionna 的 `Paths` 对象，
  所以射线追踪数据集调 `ds.paths()` 会报错而非返回假角度。
- **时延扩展的频域估计有固有误差**。可观测最大时延是 `1/(12·SCS)`，
  实测比值 0.8~1.0，仅作数量级检查。
- **QuaDRiGa 未纳入**，需要 MATLAB/Octave 运行时。
- **ChannelHub 的 CDL-A/B/C 角度表与 38.901 不符**，superwireless 启动时会用
  逐字核对过的标准表覆盖（`spec38901.apply_spec_tables`）。CDL-C 原表有
  23/24 簇的角度与 Table 7.7.1-3 有出入，占总功率 93.8%，ASA 偏 14.5°。
  时延与功率两列是对的。设 `SUPERWIRELESS_CDL_SPEC=0` 可复现未修正前的结果；
  **CDL-D/E 未覆盖**（表结构含 `Cluster PAS` 列，未逐字核对）。
- **多小区必须有 `bs_panel` 干扰才会进 SINR**。ChannelHub 只在拿到面板排布时
  才建 DFT 码本，拿不到就走兜底：`sir_dB = 49.9`、`sinr_dB = snr_dB`，
  不报错不告警。superwireless 现在会由 `num_bs_tx_ant` 自动推导面板并在
  门 1 里检查——但 2026-07-29 之前生成的数据集没有这一步，那些数据集里的
  "SINR"实为纯热噪声 SNR。

## 测试

```bash
python tests/test_e2e.py         # 端到端 39 项
python tests/test_mcp_server.py  # MCP 全链路 37 项
python tests/test_raytracing.py  # 射线追踪与决策层 40 项
python tests/test_linklevel.py   # 谱效、可信度、物理层、IRC 45 项
python tests/test_gates.py       # 校准、标准表、三道门、统计判决 86 项
python tests/test_results.py     # 外部算法结果契约、预注册 80 项
python tests/test_linkadapt.py   # 链路自适应、吞吐、并行生成 135 项
python tests/test_mumimo.py      # MU-MIMO 配对、预编码、rank/SU-MU 自适应 63 项
python tests/test_system.py      # 系统级：话务、PF 调度、HARQ、体验速率口径 80 项
python tests/test_interference.py # IoT、测量域、预设、说明书、算法页、文档计数 362 项
python tests/test_csi_aging.py   # CSI 时延与老化、SRS 跳频、基站/真实视角分离 84 项
python tests/test_rng.py         # 随机数分流、多重复置信区间、公共随机数 107 项
python tests/test_sysscenes.py   # 系统级场景预设、成对受控性、expect 诚实性 69 项
```

共 **1227 项**。

## 致谢

物理计算内核来自 [ChannelHub](https://github.com/wangxz0803-lab/ChannelHub_main)。
射线追踪基于 [Sionna RT](https://nvlabs.github.io/sionna/)。
工作流设计参考 [superpowers](https://github.com/obra/superpowers)。

## License

MIT
