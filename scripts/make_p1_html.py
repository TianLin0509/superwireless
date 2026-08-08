"""生成 P1_PLAN.html：三个 P1 课题的实现方案，供评审。

底稿是 agent 写的 P1_DESIGN.md（855 行），本脚本负责：
1. 把我**独立核实过**的部分与**只是转述**的部分区分开
2. 修掉核实中发现的技术错误
3. 补上底稿漏掉的限定条件
4. 排版 + KaTeX 公式
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from superwireless import katex as kx  # noqa: E402
from superwireless import mathml as mm  # noqa: E402


def M(tex: str, *, block: bool = False) -> str:
    return kx.wrap(tex, mm.render(tex, block=block), display=block)


# --- 公式常量（f-string 里不能有反斜杠，Python < 3.12 是语法错误）---------
F_SIR = M(r"\mathrm{SIR} = \frac{\mathrm{bf}_{srv}}"
          r"{\sum_{k \neq srv} \frac{P_k}{P_{srv}} \cdot \overline{leak}_k}", block=True)
F_PARSEVAL = M(r"\frac{1}{B}\sum_{b=1}^{B} \left| \mathbf{c}_b^H \mathbf{a}(\theta) \right|^2"
               r" \;=\; \mathrm{const} \quad \forall\,\theta", block=True)
F_SHARE = M(r"w_k = \frac{P_k}{\sum_{j \neq srv} P_j}, \qquad "
            r"P_k = 10^{\,\mathrm{rx\_power\_all\_dbm}[k]/10}", block=True)
F_ETA = M(r"\eta_u(t) = \sum_{k \neq srv} w_{u,k}\, a_k(t)", block=True)
F_JENSEN = M(r"\mathbb{E}\!\left[10\log_{10}\tfrac{1}{L}\right] \;>\; "
             r"10\log_{10}\tfrac{1}{\mathbb{E}[L]}")
F_NEFF = M(r"n_{\mathrm{eff}} = 1/\sum_k w_k^2")
F_TBS = M(r"n_{RE} = n_{RBG}\cdot 16 \cdot 12 \cdot 12")
F_RTT = M(r"\mathrm{RTT} = k_1 + N_{gNB} + (\text{到下一个可用下行时隙})")
F_CONV = M(r"\frac{1}{\eta I + N}")


def head() -> str:
    src = (ROOT / "TONIGHT.html").read_text(encoding="utf-8")
    h = src.split("</head>")[0]
    h = h.replace("superwireless 通宵成果与待审", "superwireless P1 方案评审")
    extra = """
<style>
  .vfy{border-collapse:collapse;width:100%}
  .ok2{color:#1a7f37;font-weight:600}
  .bad2{color:#cf222e;font-weight:600}
  .relay{color:#9a6700;font-weight:600}
  .topic{border:1px solid var(--border);border-radius:14px;padding:0;margin:26px 0;
         overflow:hidden;background:var(--card)}
  .topic>h3{margin:0;padding:16px 22px;background:#f5f5f7;font-size:20px;
            border-bottom:1px solid var(--border)}
  .topic>.tb{padding:4px 22px 18px}
  .q{background:#fff8e1;border:1px solid #ffd54f;border-radius:10px;
     padding:14px 18px;margin:14px 0}
  .q b.qt{display:block;color:#c77700;margin-bottom:6px}
  .inv{background:#f0fff4;border-left:4px solid #34c759;padding:12px 18px;
       margin:10px 0;border-radius:0 8px 8px 0}
  .inv code{background:#e6f7ea}
  code{font-size:.92em}
</style>"""
    return h + extra + "\n" + kx.head_assets() + "\n</head>"


VERIFY = [
    ("TBS 全带 vs 单 RBG 的四组数", "ok",
     "四个数<b>逐位吻合</b>。我用 <code>transport_block_size</code> 重算："
     "MCS5/rank2 全带 14347 B、单 RBG 848 B（比 16.9×）；"
     "MCS12/rank2 是 29722 / 1729（17.2×）；MCS20/rank4 是 104497 / 6147；"
     "MCS27/rank4 是 143436 / 8448。"
     "「1500 B 的 IP 包在 MCS12/rank2 下单个 RBG 就装得下」也对（1729 ≥ 1500）。"),
    ("Parseval：逐小区干扰份额 = RSRP 份额", "ok",
     "<b>证实，但要补一个限定条件（底稿漏了）。</b> 我拿 400 个方位角实测 "
     "<code>mean_b bg</code> 恒等于 <b>0.12500000</b>，标准差 5.5e-17（机器精度）。"
     "<br><b>但这只对恒模的导向矢量成立。</b>我第一次拿随机高斯向量测，"
     "相对离散是 <b>31.3%</b>，完全不成立。查了 <code>_system_sinr.py:239</code> 才发现 "
     "<code>bg</code> 用的是 <code>_steering_batch</code> 出来的导向矢量——"
     "每个阵元模长恒为 1/8。<b>所以这个结论的适用范围是「ChannelHub 简化几何模型内部」，"
     "不能推广到含多径的真实信道。</b>好在 P1-C 要重建的正是这个几何模型，结论可用。"),
    ("最优波束增益 bf_srv ∈ [0.95, 1.00]", "ok",
     "实测 400 个方向下是 <b>0.9510 ~ 1.0000</b>，与底稿一致。"),
    ("E14：4.74 dB 是波束抽签噪声", "ok",
     "<b>机制证实。</b>底稿只给了公式没解释来源，我补一下："
     "单个随机波束的增益<b>均值无偏</b>（实测 0.130 ≈ 码本均值 0.125），"
     "但<b>中位只有 0.0159，差 8 倍</b>——绝大多数波束背对着用户，分布极度右偏。"
     "SIR 在对数域算，由 Jensen 不等式 " + F_JENSEN + " 必然产生系统性正偏。"
     "我单小区单波束实测这个间隙是 9.1 dB、标准差 10.6 dB；"
     "底稿的 13.19/4.74 是 20 个小区求和后的值，求和压方差，<b>方向一致</b>。"),
    ("HARQ RTT 的时隙推算", "ok",
     "DDDSU 逐相位手算无误：D0 的 k1=4、加 1 个时隙 gNB 处理 → 落到下一周期 D0，"
     "RTT 5 个时隙；D1/D2/S3 分别是 4/3/2 个时隙。四种相位平均 "
     "<b>3.5 时隙 = 1.75 ms</b>。"
     "N1=10 符号也对得上 38.214 Table 5.3-1 在 μ=1、capability 1、"
     "<code>dmrs-AdditionalPosition=pos0</code> 下的取值。"),
    ("h_interferers 的数据量", "ok",
     "算术无误：20 个干扰小区 × 272 RB × 64 × 4 × 8 B(complex64) = "
     "<b>11.14 MB/样本</b>，服务信道 0.56 MB，正好 20 倍。"),
    ("rx_power_all_dbm 存在但没被存下来", "ok",
     "<code>internal_sim.py:2896</code> 确实产出它；superwireless 全库搜 "
     "<code>rx_power_all</code> <b>命中数为 0</b>，"
     "<code>generate.py</code> 的 <code>_SCALAR_META_FIELDS</code> 里只有 "
     "<code>rx_power_serving_dbm</code>。"),
    ("n_dl_sched 恒等于 1", "ok",
     "<code>_system_sinr.py:484</code> 是 "
     "<code>max(1, round(ues_per_cell × pdsch_load))</code>，"
     "21 UE / 21 小区 ⇒ <code>ues_per_cell=1</code> ⇒ "
     "<code>round(0.5)=0</code> ⇒ 兜成 1。<code>pdsch_load</code> 取 0.5 与 1.0 都是 1，"
     "这也解释了 CLAUDE.md 那条「负载旋钮在下行不起作用」的<b>精确根因</b>。"),
    ("p99 时延「~15 TTI」", "bad",
     "<b>底稿的括号算错了。</b>它写「3 次重传 × 3.5 + 1」= 11.5，但正文写 ~15。"
     "正确的是 <b>1 + 3×(3.5+1) = 14.5</b>——每次重传除了等 RTT，"
     "<b>还要占一个传输时隙</b>。结论「尾时延被高估近 4 倍」不变（14.5/4 = 3.6×），"
     "只是中间那步要改。"),
    ("I-B6 的 S 时隙修正", "done",
     "<b>这条我已经做完了。</b>底稿把它列在 P1-B 里顺手修，"
     "但我在 P0 阶段就修了：<code>S_SLOT_DL_FRACTION=0.7</code> 现在被主循环与 "
     "<code>dl_ratio</code> 共用，实测全 S 图案吞吐是全 D 的 <b>0.700</b> 倍。"
     "DDDSU 的有效下行从 0.8 变成 <b>0.74</b>，<b>吞吐下降 7.5%</b>"
     "（底稿写「下降约 7.5%（0.8/0.74−1 = 8.1%）」把两个方向混了，7.5% 是对的）。"),
    ("3.1(a) 重传 BLER 查错 MCS", "done",
     "<b>也已经修完。</b>现在查的是实发的 <code>m</code>，"
     "回归测试钉住了旧写法不能回来。<b>所以 P1-B 的工作量比底稿估的要小。</b>"),
    ("E19：21 个小区只有 14 个有 UE", "relay",
     "<b>转述，我没独立复核。</b>机制是可信的——"
     "<code>serving_idx = argmax(rx_power)</code>，撒点均匀但归属是竞争的。"
     "它推出的「要每小区 8~10 个 UE 得生成约 1700 样本、单进程 20~55 分钟」"
     "也只是转述。"),
    ("E16：top-3 邻区占 86.9% 干扰", "relay",
     "<b>转述。</b>由此推出的截断误差 "
     "<code>10log10(1/0.869) = 0.61 dB</code> 算术是对的，"
     + F_NEFF + " 的定义也标准。"),
]


TOPICS = [
    dict(
        key="A", num="三", title="P1-A · 包模型与频域复用",
        verdict="最大的产出是 PRB 利用率从解析式变成实测值；最大的代价是 KPI 口径重定义。",
        blocks=[
            ("现状：频域维度在 <code>simulate</code> 里根本不存在", """
<p><code>re_per_tti = n_rb × 12 × 12</code> 恒定为 39168 RE，
<code>use_mu</code> 为假时 <code>picked</code> 只有一个人，全带原样喂给
<code>transport_block_size</code>。<b>MU 时 K 个人也都拿全带——那是对的</b>
（MU 是空间复用），但这意味着<b>频域从头到尾没有出现过</b>。</p>
<p><code>bimodal</code> 抽出的 RBG 数只走到两处：定 burst 字节数、打
<code>is_small</code> 标记。<code>is_small</code> 之后只进 KPI 分组和直方图，
<b>没有任何一处让它影响资源占用</b>。</p>
<p>另外两条：<code>p_idle_tti</code> 是死旋钮（<b>P0 阶段我已加告警</b>，
但它仍然不驱动仿真）；<b>没有包层</b>——粒度是 burst，
全文搜不到任何时延 KPI。</p>"""),
            ("为什么重要：把小区用户容量硬顶低了一个数量级", """
<p>一个 1500 字节的 IP 包在 MCS12/rank2 下<b>只要 1 个 RBG</b>（1729 B ≥ 1500 B）。
现在给它 17 个，<b>94% 的频域资源在这半毫秒里空转</b>。</p>
<p>具体场景：12 个用户都在发小包。现在的模型说「每半毫秒只能服务 1 个人」，
第 12 个人要排 6 毫秒的队。真实基站在同一个 TTI 里就把 12 个人全发了。</p>
<p><b>反方向也错</b>：500 KB 的 FTP3 文件在 MCS20/rank4 全带下只要 5 个 TTI，
而掐尾要扣掉 1 个、<code>min_burst_tti=2</code> 又要求至少 2 个——
<b>给了全带宽反而让 burst 短到测不准</b>。</p>
<p>还有一条自指的荒谬：<code>system.py</code> 里那条 note 自己写着
「小包的体验速率测不出来……要看小包体验请看调度时延分布」，
<b>而调度时延分布并不存在</b>。</p>"""),
            ("方案：RBG 按需分配，但不许碰频选调度", """
<p><b>关键约束</b>：你明确否掉了子带 CQI，所以<b>分给谁哪几个 RBG 必须与信道无关</b>
——只按需求量分，不按哪段频率好分。分配顺序仍由 PF 度量定
（仍用全带的 <code>best_se_gnb</code>），分到的具体是哪几个 RBG 用<b>连续编号</b>。</p>
<pre><code>free = 17
for u in PF 排序:
    need = rbg_needed[mcs][rank][bytes_left]   # O(log 17) 反查表
    n = min(need, free, max_rbg_per_ue)
    if n &lt; min_rbg_per_ue: break              # 剩不下一个最小单元就收工
    alloc[u] = n; free -= n
if free: alloc[order[0]] += free               # 尾料给第一名</code></pre>
<p><code>max_ue_per_tti</code> 是 <b>PDCCH 容量的代理</b>（自审里点名的遗漏）。
现网一个 TTI 能下 8~16 个 DCI，默认给 8，
并在结果里报「被 PDCCH 顶住的 TTI 占比」。</p>
<p>TBS 反查<b>不能用除法</b>（TBS 有量化与分码块，不是线性的），
建一张 17×28×4 = 1904 个 int 的表，<code>np.searchsorted</code> 查。
{tbs}，<code>transport_block_size</code> 本来就吃 <code>n_re</code>，<b>一行都不用改</b>。</p>"""),
            ("唯一有物理内容的新增项：分到的 RBG 上的真实 SINR", """
<div class="inv"><p><b>这一项零额外计算。</b>
<code>csi_aging.mmse_stream_sinr()</code> 本来就返回逐 RBG 的
<code>[RBG, rank]</code> 线性 SINR，然后被 <code>mumimo.user_sinr_db()</code>
立刻平均掉。<b>留下它只是内存开销</b>——17 RBG × 4 rank × 8 快照 × 12 UE
= 6528 个 float。</p></div>
<p><b>用法必须严格区分两个视角</b>（沿用已有的 <code>se_gnb</code> vs <code>se</code> 分工）：</p>
<ul>
<li><b>选 MCS / 排调度顺序</b>：只用<b>全带</b> <code>sinr_tx_db</code>（CQI 门限 + BF Gain）+ OLLA。
基站没有子带 CQI，它不知道哪个 RBG 好。<b>这条不能破，破了就等于偷偷做了频选调度。</b></li>
<li><b>判 ACK/NACK</b>：用<b>实际分到的那几个 RBG</b> 上的真实 SINR。</li>
</ul>
<p>这样才会出现真实效应：分到 3 个 RBG 的用户运气不好抽到 3 个深衰的，
误码率高于全带平均——<b>这个惩罚只有做了 FDM 才会出现</b>，全带分配时被平均掉了。</p>"""),
            ("KPI 口径会怎么变（历史数字全部不可比）", """
<div class="tbl-wrap"><table>
<thead><tr><th>KPI</th><th>现在</th><th>改后</th><th>方向</th></tr></thead>
<tbody>
<tr><td>PRB 利用率</td><td>不存在（解析式，不是实测）</td>
<td>逐 TTI 累加，真实测出来</td><td><b>新增，这是最大产出</b></td></tr>
<tr><td><code>occupancy</code></td><td>有人被调度的 TTI 占比</td>
<td>语义不变</td><td>↑（同一 TTI 多人）</td></tr>
<tr><td><code>cell_served_mbps</code></td><td>各用户之和</td><td>不变</td>
<td>小包为主时 <b>↑</b>；大包基本不变</td></tr>
<tr><td><code>experienced_mbps</code></td><td>各用户平均</td><td>口径不变</td>
<td><b>方向不确定</b>：排队时间↓推高、单次带宽↓推低。<b>净效应不敢预测，
这正是要测的东西</b></td></tr>
<tr><td>burst 数</td><td>每个文件一个</td><td>每个缓冲区忙期一个</td>
<td>数量↓、单个变长（有利于测量）</td></tr>
<tr><td>新增</td><td>—</td>
<td>逐包时延 p50/p95/p99、HoL 时延、PDCCH 受限占比</td><td></td></tr>
</tbody></table></div>
<div class="callout c-red"><p><b>改完之后所有历史系统级体验速率都要重算才能横比。</b>
这一条必须写进 CLAUDE.md。<code>test_system.py</code> 现有断言里有相当一部分
断的是「一个 TTI 一个人」的行为，会红一片，
要逐条判断<b>「这条断的是物理还是旧实现」</b>。</p></div>"""),
        ],
        invariants=[
            ("I-A1 退化", "<code>fdm_enabled=False</code> 时全部 KPI 与今天<b>逐位相同</b>。"
             "这保证 FDM 是叠加上去的能力而不是另一套物理。"),
            ("I-A2 退化", "只有 1 个 UE 有数据时必然拿到全部 17 个 RBG，"
             "TBS 与 BLER 与 I-A1 逐位相同。"),
            ("I-A3 守恒", "每 TTI <code>Σ n_rbg ≤ 17</code>，实测 PRB 利用率与逐 TTI 累加对得上。"),
            ("I-A5 物理方向", "纯小包话务下开 FDM 后 <code>cell_served_mbps</code> 必须<b>明显上升</b>，"
             "而 PRB 利用率仍远低于 1。<b>吞吐没涨就是分配器没生效。</b>"),
            ("I-A6 盲区不可修", "单 slice 的 burst 在掐尾下<b>仍然测不到</b>体验速率。"
             "<b>如果改完突然测得到了，说明掐尾被改坏了</b>——这是 KPI 的固有盲区不是 bug。"),
            ("I-A7 频选没溜进来", "把逐 RBG SINR 人为抹平成全带均值，体验速率变化必须很小。"
             "<b>变化很大就说明调度器在用频域信息</b>，违背了不做频选调度的约束。"),
        ],
        cost="约 650 行（<code>system.py</code> ~430、<code>server.py</code> ~50、"
             "<code>spec.py</code> ~15、测试 ~180）",
        risk="最大风险不是代码是<b>口径</b>：burst 定义变更 + PRB 利用率从解析式变实测，"
             "会让所有历史结论不可比。必须一次做完并在 CLAUDE.md 立碑。"),

    dict(
        key="B", num="四", title="P1-B · 调度时延与 HARQ RTT（k0 / k1 / k2）",
        verdict="均值影响 20% 量级，<b>尾时延影响 4 倍量级</b>。而尾时延正是小包业务的体验指标。",
        blocks=[
            ("现状：HARQ 是有的，但没有时间", """
<p><code>harq_pending: dict[int, tuple[int, int]]</code> —— <b>每个 UE 只有一个槽位</b>，
不是 16 个进程。重传在 UE <b>下一次被调度到的那个 TTI</b> 立刻执行；
对 PF 度量高的用户就是<b>下一个下行 TTI（0.5 ms）</b>。真实系统要 4~8 个 TTI。</p>
<p>而且 <code>tr.step(tti)</code> 投数据的同一个 TTI 里就可能被调度、被发出去
—— <b>k0 / k1 / k2 三个都是 0</b>。</p>
<div class="callout c-green"><p><b>底稿列出的两个「顺手要修」的 bug，我在 P0 阶段已经修完了</b>：
重传 BLER 查错 MCS、S 时隙口径不一致。<b>所以 P1-B 的实际工作量比底稿估的小。</b></p></div>"""),
            ("DDDSU 下的 RTT 手算（我逐格验过）", """
<p>时隙相位 <code>n mod 5</code> = <code>[D0, D1, D2, S3, U4]</code>，
ACK 只能落在 U 上。{rtt}</p>
<div class="tbl-wrap"><table>
<thead><tr><th>PDSCH 落在</th><th>k1（时隙）</th><th>+ gNB 处理 1 时隙</th>
<th>最早重传时隙</th><th>RTT</th></tr></thead>
<tbody>
<tr><td>D0</td><td>4</td><td>5</td><td>5（下周期 D0）</td><td><b>5 时隙 = 2.5 ms</b></td></tr>
<tr><td>D1</td><td>3</td><td>4</td><td>5</td><td>4 时隙 = 2.0 ms</td></tr>
<tr><td>D2</td><td>2</td><td>3</td><td>5</td><td>3 时隙 = 1.5 ms</td></tr>
<tr><td>S3</td><td>1</td><td>2</td><td>5</td><td>2 时隙 = 1.0 ms</td></tr>
</tbody></table></div>
<p>平均 <b>3.5 时隙 ≈ 1.75 ms</b>。<b>这比「4~8 个 TTI」的口头印象短</b>——
DDDSU 每 5 个时隙就有一个上行机会，比 DDDDDDDSUU 这类图案快得多。
<b>这个数本身就是一个值得报出去的结论。</b></p>
<p class="src">N1 = 10 符号取自 38.214 Table 5.3-1（μ=1、capability 1、
<code>dmrs-AdditionalPosition=pos0</code>），换算 <code>ceil(10/14) = 1</code> 个时隙。
<b>做成参数并把出处写进 docstring，别硬编码。</b></p>"""),
            ("影响量级：均值 +23%，尾部 4 倍", """
<p>500 KB 文件全带 MCS20/rank4 要 5 个 TTI，10% 首传 BLER ⇒ 期望 0.5 次 NACK。</p>
<ul>
<li>现状：每次 NACK 多花 1 个 TTI ⇒ 5 + 0.5×1 = <b>5.5 TTI</b></li>
<li>加 RTT：每次 NACK 多花约 3.5 个时隙 ⇒ 5 + 0.5×3.5 = <b>6.75 TTI</b></li>
</ul>
<p>体验速率被高估 <b>+23%</b>。</p>
<div class="callout c-amber"><p><b>对小包狠得多，而这里底稿算错了一步，我改过来：</b>
p99 时延现状是 <code>1 + 3×1 = 4</code> 个 TTI；加 RTT 后是
<code>1 + 3×(3.5+1) = <b>14.5</b></code> 个 TTI——
<b>每次重传除了等 RTT，还要占一个传输时隙</b>。
底稿括号里写的「3 次重传 × 3.5 + 1」= 11.5 漏了传输时隙。
结论「尾时延被高估近 4 倍」不变（14.5/4 = 3.6×）。</p></div>"""),
            ("必须和进程池一起做，否则结果是假的", """
<div class="callout c-red"><p>现在每个 UE 只有 1 个 HARQ 槽。<b>只加 RTT 不加进程数</b>，
那么 UE 在等 ACK 的 3.5 个时隙里<b>完全不能被调度</b>——吞吐会断崖下跌，
<b>而那个下跌是实现缺陷不是物理</b>。38.214 规定 DL 最多 16 个进程正是为了填满这个管道。
<b>只加 RTT 得到的「体验速率下降 60%」会是一个纯粹的假结论。</b></p></div>
<p>主循环每 TTI 的处理顺序<b>错一步就差一个 TTI</b>：</p>
<pre><code>1. 结算本 TTI 到期的 ACK/NACK    ← 先结算，让这些进程本 TTI 就能被排上
2. tr.step(tti) 投新数据
3. 把到期的 retx 进程标成可调度
4. 组候选：重传优先，再按 PF 排新传
5. 分配 RBG / 选 MCS / 发送
6. 抽 BLER 结果但「不立刻应用」，挂到 ack_due[(tti + k1) % L]   ← 关键
7. 更新 PF 平均速率</code></pre>
<p><b>第 6 步是关键</b>：BLER 抽签在<b>发送时</b>做（那时才知道 SINR 和 MCS），
但<b>结果在 k1 之后才生效</b>。这样 OLLA 的更新也自然被推迟到 ACK 到达时——
<code>SchedulerConfig</code> 的注释里本来就写着「反馈只作用于下一调度时刻」，
<b>现在才真的做到</b>。</p>"""),
        ],
        invariants=[
            ("I-B1 退化", "<code>harq_rtt_enabled=False</code> 时全部 KPI 与今天逐位相同。"),
            ("I-B2 对标手算", "DDDSU、<code>n1_slots=1</code> 下 k1 求解器必须给出 "
             "<code>{D0:4, D1:3, D2:2, S3:1}</code>，ACK 相位恒为 4。"
             "<b>这张表是手算的，写进测试当金标准。</b>"),
            ("I-B3 RTT 真的生效", "把 BLER 强制成 0（无重传）时，开不开 RTT <b>结果必须逐位相同</b>。"
             "这条抓「把 RTT 错误地加到首传上」——首传也被延迟的话，"
             "无重传场景也会变，当场暴露。"),
            ("I-B4 进程数饱和曲线", "<code>n_harq_proc</code> 从 1 扫到 16，吞吐必须<b>单调不降</b>"
             "并在某处饱和，饱和点应当 ≈ 平均 RTT / 平均下行时隙间隔（DDDSU 下约 3~4）。"
             "<b><code>n_harq_proc=1</code> 时吞吐没明显掉，就说明 RTT 根本没生效。</b>"),
            ("I-B5 时延守恒", "逐包 <code>packet_delay = 排队 + 传输 + Σ retx_rtt</code>，"
             "全体相加与分项统计<b>差必须为 0</b>（整数运算，不允许误差）。"),
        ],
        cost="约 400~470 行（原估 450~520，扣掉我已修完的两个 bug）",
        risk="<b>主循环步骤顺序错一步就差一个 TTI，而症状是「体验速率差百分之几」——"
             "看起来完全正常。</b>唯一的防御是 I-B2 那张手算表。"
             "另外 <code>n_harq_proc</code> 的默认值会直接决定结论，"
             "<b>必须扫一遍把饱和曲线放进文档，不能拍 16 了事</b>。"),

    dict(
        key="C", num="五", title="P1-C · 多小区联合调度",
        verdict="<b>比我原先估的便宜得多</b>——逐小区干扰拿得到，168 字节/样本。但要先解决一个口径噪声。",
        blocks=[
            ("我之前的判断错了一半", """
<p>我在自审里写过「几何 SIR 只给聚合量、拿不到逐邻区贡献」。
<b>「是聚合量」对，「拿不到分解」错。</b></p>
<p>干扰求和式（<code>_system_sinr.py:500</code>）是
<code>I = Σ_k rx_lin[k] · N_ant · avg_leak_k</code>，于是</p>
{sir}
<p>关键问题：<code>avg_leak_k</code> 跟方向有关吗？如果无关，
份额就完全由 <code>rx_k</code> 决定，而那个我们拿得到。</p>
{parseval}
<div class="inv"><p><b>我实测证实了：400 个方位角下恒等于 0.12500000，标准差 5.5e-17。</b>
这是 Parseval——对完备 DFT 码本，<code>Σ_b |c_b^H a|²</code> 只取决于 <code>|a|²</code>。</p>
<p><b>但要补一个底稿漏掉的限定：这只对恒模的导向矢量成立。</b>
我第一次拿随机高斯向量测，相对离散是 <b>31.3%</b>，完全不成立。
查代码才发现 <code>bg</code> 用的是 <code>_steering_batch</code> 的导向矢量
（每个阵元模长恒为 1/8）。<b>所以这个结论的适用范围是「ChannelHub 简化几何模型内部」</b>
——好在 P1-C 要重建的正是这个模型，结论可用；
<b>但别把它推广到含多径的真实信道</b>。</p></div>
{share}
<p><b>需要的额外数据只有一个数组</b>：<code>meta["rx_power_all_dbm"]</code>，
K 个 float，约 <b>168 字节/样本</b>。<b>不需要 <code>h_interferers</code>。</b></p>"""),
            ("必须先修的口径噪声（E14）", """
<p><code>avg_leak_k</code> 的<b>期望</b>是常数，<b>每次实现</b>不是——
它是从码本里抽 <code>n_dl_sched</code> 个波束的平均，而</p>
<pre><code>n_dl_sched = max(1, round(ues_per_cell × pdsch_load))
# 默认预设 21 UE / 21 小区 ⇒ ues_per_cell=1 ⇒ round(0.5)=0 ⇒ 兜成 1</code></pre>
<p><b>每个干扰小区只抽一个随机波束。</b>实测 42 个样本：</p>
<div class="tbl-wrap"><table>
<thead><tr><th>量</th><th>值</th></tr></thead><tbody>
<tr><td><code>SIR_geo − SIR_rsrp</code> 均值</td><td><b>13.19 dB</b></td></tr>
<tr><td>标准差</td><td><b>4.74 dB</b></td></tr>
<tr><td>极差</td><td>1.3 ~ 21.6 dB</td></tr>
</tbody></table></div>
<div class="inv"><p><b>底稿只给了公式没解释来源，我把机制补上：</b>
单个随机波束的增益<b>均值无偏</b>（实测 0.130 ≈ 码本均值 0.125），
但<b>中位只有 0.0159，差 8 倍</b>——绝大多数波束背对着用户，分布极度右偏。
SIR 在对数域算，由 Jensen 不等式 {jensen} 必然产生系统性正偏。
我单小区实测这个间隙 9.1 dB、标准差 10.6 dB；
20 个小区求和会同时压小间隙和方差，得到 13.19/4.74，<b>方向一致</b>。</p></div>
<div class="callout c-red"><p><b>这 4.74 dB 不是路损、不是阴影、不是小尺度衰落，
是纯粹的波束抽签。</b>它在污染 P1-C 的全部结论，
<b>也在污染现在已经出的干扰类结论</b>。
而且<b>这份方差和真实的干扰起伏长得一模一样，事后分不开</b>——
必须在 C1 之前把它量出来并写进结果。</p></div>"""),
            ("做不到的事（诚实清单）", """
<p><b>（1）逐 TTI 的邻区波束方向做不到。</b>几何模型把「邻区在给谁打波束」
抽样一次就冻结进 <code>sir_dB</code> 了。要逐 TTI 控制只能自己拿
<code>h_interferers</code> 算。</p>
<p><b>但这条路不值得走</b>：CLAUDE.md 已写死 ChannelHub 的单个干扰小区信道是<b>秩 1</b> 的
（σ₂/σ₁ 中位 4.0e-8），所以 <code>precoded</code> 与 <code>isotropic</code> 逐位相同——
<b>「邻区打哪个波束」在当前信道模型下根本没有可区分的空间结构</b>。
花 20 倍数据量（11.14 MB/样本 vs 0.56 MB）买回来的是个假自由度。
<b>如果哪天干扰信道变成满秩，这条结论要重新评估。</b></p>
<p><b>（2）邻区的真调度器需要邻区自己的 UE，而数据集里不够。</b>
实测 21 UE / 21 小区时只有 <b>14 个小区</b>有人，7 个一个都没有
（<code>serving_idx = argmax(rx_power)</code>，撒点均匀但归属是竞争的）。
要每小区 8~10 个 UE 得生成约 1700 样本，单进程 20~55 分钟。
<b>所以 C1 / C2 必须分两阶段，而 C1 根本不需要邻区 UE。</b></p>"""),
            ("C1：逐小区活跃度（便宜，先做）", """
<p>不给邻区真调度器，给每个邻区一个<b>独立的话务活跃过程</b>，
占空比等于它的 PRB 利用率，干扰按逐小区份额逐 TTI 合成：</p>
{eta}
<div class="callout c-blue"><p><b>关键：<code>η</code> 是逐 UE 不同的</b>，不是全网一个数。
每个 UE 看到的邻区几何不同——<b>边缘用户的 η 由 1~2 个强邻区支配，
中心用户接近所有邻区的加权平均</b>。
静态标量模型把这两类用户按同一个 0.3 处理，<b>这是最大的失真</b>。</p></div>
<p><b>两相架构怎么保住</b>：干扰逐 TTI 变 ⇒ <code>noise_power</code> 逐 TTI 变 ⇒
<code>mmse_stream_sinr</code> 结果逐 TTI 变。<b>但 SVD 与 Type I 码本搜索与干扰无关</b>
——只有 <code>noise_power</code> 这一个标量变。所以在第一相把最贵的部分算一次，
然后在一个 η 网格（12 点，0.02→1.0 对数均匀）上重跑便宜的部分
（17 个 4×4 矩阵求逆）。主循环只做<b>对数域线性插值</b>，2 次乘加，<b>不是矩阵运算</b>。</p>
<p>第一相成本：η 无关部分约 70%（码本搜索单独就占 47%），
12 个格点 ⇒ <code>0.70 + 12×0.30 = 4.3×</code>。</p>"""),
            ("C2：真正的多小区调度", """
<p>在 C1 之上：按 <code>serving_cell_index</code> 把 UE 分组到各小区，
每个小区一份话务 + 调度器状态 + OLLA 偏置；
<code>a_k(t)</code> 不再是随机过程，而是小区 k 的调度器<b>真实</b>用掉的 RBG 比例。</p>
<div class="callout c-amber"><p><b>每个 TTI 的顺序必须是：先所有小区各自做调度决策 →
再统一合成干扰 → 再各自判 ACK/NACK。</b>
顺序反了就变成「小区 1 知道小区 2 这个 TTI 要发什么」，是上帝视角，
<b>而且不会报错，只会让干扰看起来低一点</b>。</p></div>
<p><b>性能</b>：21 小区 × 27.8 µs/TTI ≈ 584 µs/TTI，20000 TTI ≈ 11.7 s；
叠加 P1-A（3~5×）与 P1-B（1.6~2.5×）后约 50~150 s，
<b>再乘 8 次重复 ⇒ 7~20 分钟</b>。底稿倾向先接受不优化，
理由是「这一层的可读性比速度重要」——我认同。</p>"""),
            ("能回答哪些现在答不了的问题", """
<ol>
<li><b>干扰的时间起伏对 OLLA 收敛的影响。</b>现在干扰逐 TTI 是平的，OLLA 收敛到固定偏置；
真实干扰是突发的，OLLA 在追移动目标。扫突发长度就能量出来。</li>
<li><b>邻区话务与本区话务相关时会怎样。</b>忙时所有小区一起忙，
边缘用户在最需要资源时受到最强干扰——现网真实痛点，现在完全测不了。</li>
<li>中心用户 vs 边缘用户对邻区负载的敏感度差异（<b>C1 就能答</b>）。</li>
<li>「关掉最强的那个邻区能救回多少边缘用户」——干扰协调的价值上界。</li>
<li>PRB 利用率 10%→50% 的<b>真实</b>影响曲线，而不是解析式画出来的那条。</li>
</ol>"""),
        ],
        invariants=[
            ("I-C1 退化", "所有邻区恒定全发（η ≡ 1）时，逐 TTI 合成的 SINR 必须与今天 "
             "<code>neighbor_prb_util=1.0</code> 逐位相同。"),
            ("I-C4 抽签噪声常驻度量", "每次仿真都报 <code>SIR_geo − SIR_rsrp</code> 的均值与标准差，"
             "默认预设下应接近 13.2 ± 4.7 dB。<b>标准差 &gt; 3 dB 就在 notes 里告警。</b>"),
            ("I-C5 截断误差可预估", "只保留 top-3 邻区 vs 全部，SIR 差必须 ≈ "
             "<code>10log10(1/0.869) = 0.61 dB</code>。偏离超 0.2 dB 说明份额算错了。"),
            ("I-C6 因果（C2 专用）", "把某小区话务在仿真中途从 0 拉满，其邻区用户 SINR 必须在"
             "<b>同一 TTI 或之后</b>下降，<b>绝不能在之前</b>。这条抓时序反转。"),
            ("I-C7 凸性差是结论不是 bug", "邻区独立 Bernoulli(η₀) 的长期平均"
             "<b>不会</b>等于静态 <code>neighbor_prb_util=η₀</code> 的结果——"
             + F_CONV + " 对 η 是凸的（Jensen），而吞吐对 SINR 是凹的，两个方向相反。"
             "<b>这个差本身就是要交付的东西，谁把它「修」成零，整个课题就白做了。</b>"),
            ("I-C8 逐 UE 的 η 确实不同", "边缘用户的 " + F_NEFF + " 必须<b>显著低于</b>中心用户。"
             "<b>所有用户 η 都一样就说明份额矩阵退化成了全局标量，等于什么都没做。</b>"),
        ],
        cost="C1 约 800 行 + 一次数据集重生成；C2 约 570 行 + 一批大数据集（20~55 分钟生成）",
        risk="<b>E14 的 4.74 dB 抽签噪声是最大隐患</b>：不先处理的话，"
             "C1 出来的所有「干扰起伏」结论里都混着一份与物理无关的方差，"
             "<b>而且它和真实起伏长得一模一样，事后分不开</b>。"
             "另外老数据集全部不支持 C1（没有逐小区 RSRP），要有清晰的降级路径。"),
]

OPEN = [
    ("TS 28.552 的 data burst 边界",
     "底稿要把 burst 的定义从<b>「一个文件」</b>改成<b>「缓冲区连续非空的一段」</b>。"
     "理由是：两个文件到达时间重叠时，现网话统算<b>一个</b> burst，"
     "现在的代码算<b>两个</b>——第二个文件的排队时间被算进它自己的分母，"
     "等于把排队惩罚记了两遍。",
     "<b>这条决定 P1-A 要不要改 burst 定义，而改了历史数字就全部不可比。</b>"
     "写底稿的 agent 明确标了【存疑】——它没有 28.552 原文在手。<b>我也没有。</b>"
     "请你确认：现网话统里 data burst 的边界到底是文件边界还是缓冲区忙期边界？"),
    ("尾料分配策略",
     "RBG 按需分完后可能有剩（<code>free &gt; 0</code>），给谁没有唯一正确答案。"
     "底稿建议默认给 PF 第一名，并在结果里报尾料占比。",
     "<b>这会影响公平性指标。</b>要不要改成按 PF 度量比例分？还是干脆不分（空着）？"
     "空着的好处是 PRB 利用率更诚实。"),
    ("HARQ 进程数默认值",
     "38.214 规定 DL 最多 16 个。底稿建议默认 16 但<b>必须扫一遍饱和曲线</b>。",
     "DDDSU 下理论饱和点约 3~4 个进程。"
     "<b>默认给 16（贴标准）还是给饱和点（贴现网配置）？</b>"
     "这个值直接决定结论，不能拍。"),
    ("E14 口径：重建 SIR 还是保留现状",
     "底稿给了两条路：<b>（推荐）用 RSRP 域重建 SIR</b>——份额与总量内部自洽、方差干净，"
     "代价是与今天的 <code>sir_dB</code> 不逐位一致，需要一次口径迁移；"
     "<b>（保守）仍用 <code>sir_dB</code> 锚定总量、用 RSRP 份额做拆分</b>——"
     "与现有 IoT 口径完全兼容，代价是单样本上不严格自洽（只在期望上自洽）。",
     "<b>我倾向保守那条</b>，理由是 IoT 口径刚在 P0 阶段被发现有多时隙问题，"
     "同时动两个口径风险叠加。但这会让 4.74 dB 的噪声一直留在数据里，"
     "只能靠 I-C4 报出来而不是消掉。<b>你怎么看？</b>"),
    ("I-A7 的阈值",
     "「把逐 RBG SINR 抹平成全带均值，体验速率变化必须很小」——"
     "底稿猜 &lt; 3% 但<b>明确标注没测过</b>。",
     "这个阈值是「频选调度有没有偷偷溜进来」的唯一防线。"
     "<b>建议先不设死阈值，第一次实现后实测出来再定。</b>"),
]


def verify_rows() -> str:
    tag = {"ok": '<span class="ok2">已独立验证</span>',
           "bad": '<span class="bad2">发现错误·已更正</span>',
           "done": '<span class="ok2">已提前完成</span>',
           "relay": '<span class="relay">仅转述·未复核</span>'}
    return "".join(
        f"<tr><td><b>{n}</b><br>{tag[k]}</td><td>{d}</td></tr>"
        for n, k, d in VERIFY)


def topic_html(t: dict) -> str:
    blocks = "".join(
        f"<h4>{title}</h4>{body.format(tbs=F_TBS, rtt=F_RTT, sir=F_SIR, parseval=F_PARSEVAL, share=F_SHARE, eta=F_ETA, jensen=F_JENSEN)}"
        for title, body in t["blocks"])
    inv = "".join(f'<div class="inv"><p><b>{n}</b> —— {d}</p></div>'
                  for n, d in t["invariants"])
    return f"""
<h2 id="{t['key'].lower()}">{t['num']}、{t['title']}</h2>
<div class="topic">
<h3>{t['title']}</h3>
<div class="tb">
<p class="lead">{t['verdict']}</p>
{blocks}
<h4>硬不变量（照项目「零时延必须逐位退化」的文化写）</h4>
{inv}
<div class="tbl-wrap"><table><tbody>
<tr><td style="width:90px"><b>工作量</b></td><td>{t['cost']}</td></tr>
<tr><td><b>主要风险</b></td><td>{t['risk']}</td></tr>
</tbody></table></div>
</div></div>"""


def build() -> str:
    open_html = "".join(
        f'<div class="q"><b class="qt">存疑 {i} · {n}</b>'
        f'<p>{what}</p><p><b>要你拍板：</b>{ask}</p></div>'
        for i, (n, what, ask) in enumerate(OPEN, 1))

    return f"""{head()}
<body>
<div class="wrap">

<h1>P1 方案评审</h1>
<p class="tagline">包模型与频域复用 · 调度时延与 HARQ RTT · 多小区联合调度</p>
<p class="meta">2026-08-09 · 底稿 855 行由 agent 产出，本文是我逐条核实后的评审版</p>

<div class="callout c-blue">
<p><b>这份文档里哪些是我验过的、哪些只是转述，第一节逐条标出来了。</b>
核实中发现<b>一处技术错误</b>（已更正）、<b>一处论断缺限定条件</b>（已补），
另有<b>两条底稿列为待办的 bug 我在 P0 阶段已经修完</b>，
所以 P1-B 的工作量比底稿估的小。</p>
</div>

<div class="toc">
<strong>目录</strong>
<ol>
<li><a href="#v">我核实了什么</a></li>
<li><a href="#s">三条课题的结论先摆出来</a></li>
<li><a href="#a">P1-A 包模型与频域复用</a></li>
<li><a href="#b">P1-B 调度时延与 HARQ RTT</a></li>
<li><a href="#c">P1-C 多小区联合调度</a></li>
<li><a href="#o">实施顺序</a></li>
<li><a href="#q">要你拍板的 5 个存疑项</a></li>
</ol>
</div>

<h2 id="v">一、我核实了什么</h2>
<p class="lead">底稿是 agent 写的。你说「不要有技术错误」，
所以我把能验的都自己跑了一遍。<b>验不了的如实标成「仅转述」，没有装作验过。</b></p>
<div class="tbl-wrap"><table class="vfy">
<thead><tr><th style="min-width:180px">核实项</th><th>结论</th></tr></thead>
<tbody>{verify_rows()}</tbody></table></div>

<h2 id="s">二、三条课题的结论先摆出来</h2>
<div class="tbl-wrap"><table>
<thead><tr><th style="min-width:70px"></th><th>我原以为</th><th>查完之后</th></tr></thead>
<tbody>
<tr><td><b>P1-A</b></td><td>加个 RBG 循环，中等工作量</td>
<td><b>比想象中贵</b>——KPI 口径要重定义（体验速率、PRB 利用率、burst 定义都变），
历史数字全部不可比</td></tr>
<tr><td><b>P1-B</b></td><td>中等</td>
<td>确实中等，<b>但必须和 HARQ 进程池一起做</b>——只加 RTT 不加进程会把吞吐打崩，
那是实现缺陷不是物理</td></tr>
<tr><td><b>P1-C</b></td><td>要重构架构、拿不到逐小区干扰</td>
<td><b>比想象中便宜</b>——逐小区干扰拿得到，168 字节/样本，
不需要 <code>h_interferers</code></td></tr>
</tbody></table></div>

{topic_html(TOPICS[0])}
{topic_html(TOPICS[1])}
{topic_html(TOPICS[2])}

<h2 id="o">六、实施顺序</h2>
<p class="lead">底稿建议的顺序<b>与我在自审里的排法不同</b>，它的理由我认同。</p>
<div class="tbl-wrap"><table>
<thead><tr><th style="min-width:50px">序</th><th>做什么</th><th>量级</th><th>为什么排这里</th></tr></thead>
<tbody>
<tr><td><b>①</b></td><td>P1-C 数据侧：存逐小区 RSRP</td><td>~75 行 · 半天</td>
<td><b>纯增量，不动任何现有行为</b>，不依赖其他课题，且是 C1 的前提</td></tr>
<tr><td><b>②</b></td><td>E14 口径体检：量化抽签噪声</td><td>~60 行 · 半天</td>
<td><b>它在污染现有的干扰类结论</b>，越早量出来越好</td></tr>
<tr><td><b>③</b></td><td>P1-B：HARQ RTT + 进程池 + 时延 KPI</td><td>~400~470 行</td>
<td><b>不碰 KPI 口径</b>，回归压力小；自检是二值判定，容易做对</td></tr>
<tr><td><b>④</b></td><td>P1-A：包模型 + RBG 分配 + KPI 重定义</td><td>~650 行</td>
<td>会重定义 burst 与 PRB 利用率，<b>历史数字全部作废</b>——放在不改口径的之后</td></tr>
<tr><td><b>⑤</b></td><td>P1-C 的 C1：η 网格 + 逐小区活跃度</td><td>~800 行</td>
<td>要碰 <code>build_link_tables</code>，和 P1-A 改同一个数据结构，<b>不要并行</b></td></tr>
<tr><td><b>⑥</b></td><td>P1-C 的 C2：真多小区调度</td><td>~570 行 + 大数据集</td>
<td>依赖 C1，且要专门生成一批 1700 样本的数据</td></tr>
</tbody></table></div>

<div class="callout c-amber">
<p><b>我在自审里把 P1-A（频域复用）排在 P1-B 前面，现在改主意了。</b>
底稿的理由站得住：<b>P1-A 会重定义 burst 边界和 PRB 利用率，历史数字全部作废</b>；
而 P1-B 只动主循环时序，不碰任何 KPI 口径。<b>先做不改口径的那个，回归压力小得多。</b></p>
<p>另外 ① 和 ② 加起来只要一天，却能<b>立刻改变现有干扰结论的可信度</b>——
这两条我建议无论如何先做。</p>
</div>

<h2 id="q">七、要你拍板的 5 个存疑项</h2>
<p class="lead">这些是<b>我和 agent 都不能替你决定</b>的。
第 1 条最重要——它决定 P1-A 要不要改 burst 定义。</p>
{open_html}

<footer>superwireless P1 方案评审 · 底稿 P1_DESIGN.md 855 行 ·
公式由内联 KaTeX 排版（MathML 兜底），离线可用</footer>
</div>
{kx.upgrade_script()}
</body></html>
"""


def main() -> int:
    out = ROOT / "P1_PLAN.html"
    out.write_text(build(), encoding="utf-8")
    print(f"{out}  ({out.stat().st_size / 1024:.0f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
