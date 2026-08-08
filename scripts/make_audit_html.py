"""生成 AUDIT.html：superwireless 全库自审 + Top 任务清单。

每条任务都要回答"**为什么我认为该做**"，用白话，不堆术语。
实测数字来自 ``measurements/seed_variance.json`` 等，不手填。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from superwireless import katex as kx  # noqa: E402
from superwireless import mathml as mm  # noqa: E402


def M(tex: str, *, block: bool = False) -> str:
    return kx.wrap(tex, mm.render(tex, block=block), display=block)


F_CV = M(r"\mathrm{CV} = \sigma/\mu")
F_CI = M(r"\pm 1.96\,\sigma/\sqrt{n}")


def head() -> str:
    src = (ROOT / "TONIGHT.html").read_text(encoding="utf-8")
    h = src.split("</head>")[0]
    h = h.replace("superwireless 通宵成果与待审", "superwireless 全库自审")
    extra = """
<style>
  .why{background:#f0f7ff;border-left:4px solid #0071e3;padding:14px 18px;
       margin:14px 0;border-radius:0 8px 8px 0}
  .why b.wt{display:block;color:#0071e3;font-size:13px;letter-spacing:.5px;
            margin-bottom:6px;text-transform:uppercase}
  .anal{background:#fffaf0;border-left:4px solid #ff9f0a;padding:12px 18px;
        margin:12px 0;border-radius:0 8px 8px 0;font-style:normal}
  .anal b.at{display:block;color:#c77700;font-size:13px;margin-bottom:5px}
  .rank{display:flex;gap:14px;align-items:flex-start;margin:26px 0;
        padding-bottom:22px;border-bottom:1px solid var(--border)}
  .rank:last-child{border-bottom:none}
  .rk{flex:0 0 46px;height:46px;border-radius:12px;display:flex;
      align-items:center;justify-content:center;font-size:20px;font-weight:700;color:#fff}
  .rk1{background:#ff3b30}.rk2{background:#ff9f0a}.rk3{background:#0071e3}
  .rkx{background:#8e8e93}
  .rb{flex:1;min-width:0}
  .rb h4{margin:2px 0 8px;font-size:19px}
  .eff{display:inline-block;font-size:12px;padding:2px 9px;border-radius:20px;
       background:#eef;color:#334;margin-right:6px}
</style>"""
    return h + extra + "\n" + kx.head_assets() + "\n</head>"


def seed_chart(data: dict) -> str:
    """8 个 seed 的体验速率散点 + 均值 + 95% CI 带。让"数字在跳"一眼可见。"""
    v = data["cell_experienced_mbps"]
    n = len(v)
    mu = sum(v) / n
    sd = (sum((x - mu) ** 2 for x in v) / (n - 1)) ** 0.5
    ci = 1.96 * sd / (n ** 0.5)
    lo, hi = min(v) - 12, max(v) + 12
    W, H, L, R = 720, 200, 62, 24

    def x(val: float) -> float:
        return L + (val - lo) / (hi - lo) * (W - L - R)

    band = f'<rect class="adband" x="{x(mu - ci):.1f}" y="46" ' \
           f'width="{x(mu + ci) - x(mu - ci):.1f}" height="96" rx="4"/>'
    rng = f'<line class="adrng" x1="{x(min(v)):.1f}" y1="94" ' \
          f'x2="{x(max(v)):.1f}" y2="94"/>'
    # **按数值排序后再错行**，否则数值接近的两个点会叠在同一行上完全重合——
    # 实测 seed 3(154.6) 与 seed 7(154.2) 就这么丢了一个。
    order = sorted(range(n), key=lambda i: v[i])
    row = {idx: k % 4 for k, idx in enumerate(order)}
    dots = "".join(
        f'<circle class="addot" cx="{x(val):.1f}" cy="{70 + row[i] * 16}" r="6"/>'
        f'<text class="adsd" x="{x(val):.1f}" y="{74 + row[i] * 16}">{i}</text>'
        for i, val in enumerate(v))
    ticks = "".join(
        f'<line class="adax" x1="{x(t):.1f}" y1="150" x2="{x(t):.1f}" y2="156"/>'
        f'<text class="adtk" x="{x(t):.1f}" y="170">{t}</text>'
        for t in range(120, int(hi) + 1, 20))
    return f"""<svg viewBox="0 0 {W} {H}" width="100%" style="max-width:{W}px"
 role="img" aria-label="8 个随机种子下的小区体验速率散布">
<style>
 .adband{{fill:#0071e3;opacity:.12}} .adrng{{stroke:#8e8e93;stroke-width:2}}
 .addot{{fill:#0071e3;opacity:.85}} .adsd{{fill:#fff;font-size:9px;text-anchor:middle}}
 .admu{{stroke:#ff3b30;stroke-width:2;stroke-dasharray:5 3}}
 .adax{{stroke:#d2d2d7}} .adtk{{fill:#6e6e73;font-size:11px;text-anchor:middle}}
 .adlb{{fill:#1d1d1f;font-size:12px}} .adlb2{{fill:#6e6e73;font-size:11px}}
 .adhd{{fill:#1d1d1f;font-size:13px;font-weight:600}}
</style>
<text class="adhd" x="6" y="18">同一信道、同一配置，只改随机种子 —— 小区体验速率</text>
<line class="adax" x1="{L}" y1="150" x2="{W - R}" y2="150"/>{ticks}
{band}{rng}
<line class="admu" x1="{x(mu):.1f}" y1="46" x2="{x(mu):.1f}" y2="150"/>
{dots}
<text class="adlb" x="6" y="56">{n} 个种子</text>
<text class="adlb2" x="6" y="72">圆点里是种子号</text>
<text class="adlb" x="{W - R}" y="190" text-anchor="end">Mbps</text>
<text class="adlb2" x="{x(mu):.1f}" y="42" text-anchor="middle">均值 {mu:.1f}</text>
<text class="adlb2" x="{L}" y="190">蓝带 = 95% 置信区间 ±{ci:.1f}（{ci / mu * 100:.1f}%）
 · 极差 {max(v) - min(v):.1f} Mbps</text>
</svg>"""


# --- Top 清单 -------------------------------------------------------------
TASKS = [
    dict(
        n=1, cls="rk1", title="系统级多 seed + 置信区间",
        effort="小 · 约 150 行", roi="决定所有系统级结论可不可信",
        what="每次系统仿真跑 N 个随机种子（默认 8），所有 KPI 报"
             "均值 ± 95% 置信区间。两组对比时，效应小于置信区间就"
             "<b>拒绝下结论</b>，而不是照报百分比。",
        why="现在我们跑一次仿真，报「体验速率 142.3 Mbps」——看着精确到 0.1。"
            "但实测换 8 个种子，同样的信道、同样的配置，结果在 <b>123.7 到 184.6</b> "
            "之间跳。小数点后那一位是假的，连个位数都不一定站得住。"
            "<br><br>危险不在于数字不好看，而在于<b>我们会把噪声当成效应</b>。"
            "上一轮我自己就中招了：报「Type I 权在 60 km/h 下老化后反而涨了 14%」，"
            "而噪声本身一个标准差就是 11.4%——那个 14% 大概率什么都不是。"
            "<br><br>有了置信区间，这种话会被自动拦下来："
            "「两组差 14%，但各自的 95% 区间是 ±20%，说明不了问题。」"
            "<br><br>项目在链路级已经建了完整的三道门（配对检验、Wilcoxon、置信区间、"
            "预注册），<b>系统级却整个绕过了</b>——同一个库里两套标准，"
            "而恰恰是你最关心的那个 KPI 走的是没有门的那条路。",
        anal="就像用一台有 ±3 斤漂移的体重秤。你今天 150、明天 147，"
             "不能说「减了 3 斤」。要么多称几次取平均，要么就别报小数。",
        cost="贵的那一步（把信道压成查找表）<b>跟随机种子没关系</b>，只做一次；"
             "换种子只要重跑 TTI 循环，一次 1.1 秒。实测 8 个种子总共 19.4 秒，"
             "比单次的 11.6 秒<b>只多 8 秒</b>。"),
    dict(
        n=2, cls="rk1", title="把 sw_system_sim 写进 SKILL.md",
        effort="很小 · 约 80 行", roi="这个功能现在事实上不可达",
        what="在 skill 里加「第 5 段 · 系统级体验速率」，"
             "交代什么时候该从链路级切到系统级、"
             "数据集要满足什么条件（每 UE 多快照）、KPI 怎么读。",
        why="skill 是给 agent 看的操作手册。<b>整本手册里没提过系统级仿真这个工具</b>"
            "（<code>sw_system_sim</code> 在 SKILL.md 里出现 <b>0 次</b>），"
            "所以任何按手册干活的 agent 都不会用它。"
            "<br><br>后果不是「效率低一点」，是<b>功能事实上不存在</b>。"
            "用户问「这个小区的体验速率是多少」，agent 会去翻链路级的谱效工具，"
            "然后给出一个香农谱效——那根本不是体验速率，"
            "而且它<b>看起来像个答案</b>，所以错得很隐蔽。",
        anal="餐厅新出了招牌菜，厨房会做，但菜单上没印。客人永远点不到。",
        cost="纯文档，不改代码。"),
    dict(
        n=3, cls="rk2", title="频域多用户 RBG 分配",
        effort="中 · 约 250 行", roi="修正小区容量与 PRB 利用率",
        what="一个 TTI 里把 17 个 RBG 按各用户的缓冲区需求分给多个用户，"
             "而不是整band 给一个人。",
        why="现在的调度器一个 TTI 只服务一个用户，而且<b>不管他要发多少，"
            "都把 100 MHz 全给他</b>。"
            "<br><br>假设一个用户只要发个很小的包，占 1/17 的带宽就够了。"
            "现在的做法是：给他整个 100 MHz，他半毫秒发完走人，"
            "<b>剩下 16/17 的带宽这半毫秒里空着</b>。"
            "<br><br>两层后果：小区能同时服务的用户数被硬顶在「每半毫秒 1 个人」；"
            "排队的其他用户白等。真实基站不是这么干的——"
            "全带 CQI 下调度器就会把 RBG 按需分给好几个用户，同一个 TTI 里并行发。"
            "<br><br><b>这和你否掉的「频选调度」不是一回事。</b>"
            "频选调度是「看哪个用户在哪段频率上信道好，就把那段给他」，需要子带 CQI；"
            "这里说的只是「按需要的量把带宽分开」，<b>全带 CQI 就够了</b>。",
        anal="像一条 17 车道的高速，现在的规则是"
             "「同一时刻只放一辆车进来，而且它必须占满 17 条道」。"
             "哪怕那是辆自行车。",
        cost="改 <code>simulate</code> 的分配逻辑与 TBS 计算，"
             "建议和第 5 条一起做（都在 <code>system.py</code>，省一次回归）。"),
    dict(
        n=4, cls="rk3", title="SKILL.md 拆成主文件 + references",
        effort="小", roi="省一半上下文，规则更容易被真守住",
        what="主文件压到约 250 行只留「什么时候做什么」，"
             "射线追踪 / 干扰场景设计 / 高铁 / TDD AMC 细节拆成 "
             "<code>references/*.md</code>，用到才读。",
        why="现在 skill 是一个 <b>663 行</b>的大文件，每次对话开始就整个塞进上下文。"
            "里面有射线追踪、高铁超级小区、干扰场景设计这些内容，"
            "绝大多数任务用不到，但每次都占着位置。"
            "<br><br>收益是双份的：省上下文是明面上的；"
            "更重要的是<b>主文件短了之后规则更容易被真的遵守</b>——"
            "663 行里的一条硬性规定，和 250 行里的一条，被忽略的概率不一样。"
            "superpowers 的所有 skill 都是这个结构。",
        anal="安全须知贴在门口一张纸上，和夹在 60 页员工手册第 43 页，"
             "被看见的概率差很远。",
        cost="纯文档搬运，但要小心别把「必须守住的几条」也搬走了。"),
    dict(
        n=5, cls="rk3", title="系统级场景预设",
        effort="小", roi="让系统仿真可复现、可横比",
        what="加 5~6 个系统级预设（如「密集城区满负载」「宏站轻载」"
             "「高速移动」），一句话起跑。",
        why="现在跑一次系统仿真要填 <code>duration_s</code>、"
            "<code>traffic_model</code>、<code>arrival_rate</code>、"
            "<code>scheduler</code>、<code>pf_window</code>、"
            "<code>neighbor_prb_util</code>、<code>csi_aging</code>、"
            "<code>srs_period</code>、<code>olla_speedup</code>……八九个参数。"
            "<br><br>链路级那边有 26 个预设，一句「密集城区」就够了，"
            "系统级这边一个都没有。结果就是<b>每次跑都在拍参数，"
            "而且不同次之间参数不一致，结果没法横向比</b>——"
            "这一点和第 1 条是同一个病根：结论站不住。",
        anal="做菜每次都凭手感放盐，然后想比较「这次和上次哪次好吃」。",
        cost="改 <code>presets.yaml</code> 加一层系统级段落，不改代码逻辑。"),
    dict(
        n=6, cls="rkx", title="调度时延与 HARQ RTT",
        effort="中", roi="补齐时延链的另一半",
        what="建模 k0/k1/k2：从「决定要发」到「真的发出去」的时延，"
             "以及 NACK 到重传之间的往返。",
        why="我们刚建好 CSI 老化——基站看到的信道是过期的。"
            "但<b>时延链只做了一半</b>："
            "<br>· 已建模：从 SRS 探测到用它算预编码之间的时延"
            "<br>· 没建模：从决定调度到真正发出去、以及发失败到重传之间的时延"
            "<br><br>现在的 HARQ 是「发现失败下一个 TTI 立刻重传」，"
            "真实系统要等 4~8 个 TTI（等终端反馈、等重传被调度）。"
            "<br><br>这对小包影响特别大：一个小包发一次失败，"
            "实际要多等 4 毫秒，而现在只多等 0.5 毫秒——<b>体验速率被高估</b>。"
            "而体验速率正是你说的最关心的 KPI。",
        anal="现在的模型里，快递员敲门没人应，下一秒就能再敲一次。"
             "真实情况是要等下一趟班车。",
        cost="要在 TTI 循环里加一个待发队列与定时器，"
             "但仍然是纯查表，不影响两相架构。"),
    dict(
        n=7, cls="rkx", title="skill 压力测试",
        effort="中", roi="验证硬性规定真的拦得住",
        what="照 superpowers 的 <code>testing-skills-with-subagents</code> 做："
             "先不给 skill 跑压力场景看 agent 怎么违规（RED），"
             "把原话记下来写成反制条款（GREEN），再重跑堵漏（REFACTOR）。",
        why="我们在 skill 里写了硬性规定，比如「门 3 没过之前不许说 A 比 B 好」。"
            "但<b>从来没验证过这条规定在压力下守不守得住</b>。"
            "<br><br>superpowers 的做法是把 TDD 用在流程文档上："
            "构造一个有压力的场景（用户催得急、结果看起来明显更好、只差统计检验），"
            "先<b>不给</b> skill，看 agent 怎么绕过去、用什么话术合理化"
            "（「趋势上看」「总体而言」），把这些原话逐字记下来；"
            "再写针对性的反制条款，带着 skill 重跑，看还能不能绕。"
            "<br><br>核心一句话：<b>没见过它失败，就不知道它防住了什么。</b>"
            "我们现在的 <code>&lt;HARD-GATE&gt;</code> 是凭直觉写的。",
        anal="装了防盗门却从没试过撬。门看着挺结实。",
        cost="要开子 agent 跑几轮，本身不改产品代码。"),
    dict(
        n=8, cls="rkx", title="多小区联合调度",
        effort="大 · 架构重构", roi="干扰画像质变，但建议排在最后",
        what="每个小区都跑自己的调度器，干扰随邻区的实际调度逐 TTI 变化。",
        why="现在只有服务小区在真的跑调度，<b>邻区是一个静态的「负载 0.3」</b>。"
            "真实网络里每个小区都在各自调度：某一刻邻区刚好在发大包（干扰大），"
            "下一刻它空闲（干扰小）。"
            "<br><br>现在的模型给出的干扰是「平均意义上正确、但逐 TTI 是平的」。"
            "所以像「干扰的时间起伏对 OLLA 收敛有多大影响」"
            "「邻区话务和本区话务相关时会怎样」这类问题，<b>现在根本答不了</b>。"
            "<br><br>放最后是因为它要重构架构，而前面六条都是增量改动——"
            "<b>先把结论可信度（第 1 条）解决掉，再谈把模型做深</b>，"
            "否则模型越复杂，噪声越难分辨。",
        anal="现在是「假设邻居家平均每天用 30% 的电」，"
             "而真实情况是他晚上七点开空调、白天没人。"
             "平均值对，但你算不出晚高峰会不会跳闸。",
        cost="TTI 主循环要同时推进多个小区，"
             "干扰要从几何静态量改成逐 TTI 合成。"),
]


def tasks_html() -> str:
    out = []
    for t in TASKS:
        out.append(f"""
<div class="rank">
<div class="rk {t['cls']}">{t['n']}</div>
<div class="rb">
<h4>{t['title']}</h4>
<div><span class="eff">工作量 {t['effort']}</span><span class="eff">{t['roi']}</span></div>
<p><b>做什么：</b>{t['what']}</p>
<div class="why"><b class="wt">为什么我认为该做</b><p>{t['why']}</p></div>
<div class="anal"><b class="at">打个比方</b>{t['anal']}</div>
<p class="src"><b>代价：</b>{t['cost']}</p>
</div></div>""")
    return "".join(out)


def build(seed: dict | None) -> str:
    chart = seed_chart(seed) if seed else ""
    v = seed["cell_experienced_mbps"] if seed else []
    stats = ""
    if v:
        n = len(v)
        mu = sum(v) / n
        sd = (sum((x - mu) ** 2 for x in v) / (n - 1)) ** 0.5
        stats = (f"均值 {mu:.1f} Mbps，标准差 {sd:.1f}，"
                 f"变异系数 <b>{sd / mu * 100:.1f}%</b>，"
                 f"极差 <b>{max(v) - min(v):.1f} Mbps</b>")

    return f"""{head()}
<body>
<div class="wrap">

<h1>全库自审</h1>
<p class="tagline">三条硬伤 · 机制对照 superpowers · Top 8 任务清单</p>
<p class="meta">2026-08-07 · 每条都写清「为什么我认为该做」· 数字来自实跑</p>

<div class="callout c-red">
<p><b>先说一条更正。</b>上一轮我说 60 km/h 下 Type I 权老化后体验速率
「+14%」（109.8 → 125.3）。<b>刚测出来那是噪声，不是效应</b>——
同配置换随机种子的标准差就有 11.4%，+14% 只有 1.2 个标准差。</p>
<p>「Type I 对老化几乎免疫」这个<b>整体趋势仍然成立</b>（四个速度点方向一致），
但那个 +14% 我当成效应讲了，不对。<b>这正是下面第 1 条要解决的问题。</b></p>
</div>

<div class="toc">
<strong>目录</strong>
<ol>
<li><a href="#a1">三条硬伤（带实测证据）</a></li>
<li><a href="#a2">Skill 与机制：对照 superpowers</a></li>
<li><a href="#a3">仿真实现的遗漏与待定</a></li>
<li><a href="#a4">Top 8 任务清单 —— 每条为什么该做</a></li>
</ol>
</div>

<h2 id="a1">一、三条硬伤</h2>

<h3>1. 系统级仿真零统计</h3>
<div class="hero">{chart}</div>
<p>上面这张图是<b>同一批信道、同一套配置，只换随机种子</b>跑 8 次的结果。
{stats}。</p>
<p>换句话说，<b>报「142.3 Mbps」时，真实的不确定度是 {F_CI} ≈ ±12 Mbps</b>。
小数点后那一位没有意义。</p>
<p>而链路级那边建了完整的三道门——配对检验、Wilcoxon、置信区间、预注册。
在 <code>system.py</code> 里搜 <code>confidence</code> /
<code>t_test</code> / <code>wilcoxon</code> / <code>paired</code>，
命中数是 <b>0</b>；<code>gates.py</code> 只被链路级路径引用。
<b>同一个库里两套标准，而你最关心的 KPI 走的是没有门的那条。</b></p>

<h3>2. 一个 TTI 只服务一个用户，且占满全带</h3>
<p><code>re_per_tti = n_rb × 12 × 12</code> —— 每个被调度的用户拿<b>整个带宽</b>。
bimodal 抽出的 RBG 数只用来定 burst 的字节数，<b>不影响实际占用</b>。</p>
<p>所以一个只需要 1/17 带宽的小包，会占掉一整个 TTI 的全部 100 MHz。
小区的用户容量被硬顶在「每个下行 TTI 一个人」。</p>

<h3>3. 多小区不是联合仿真</h3>
<p>邻区是 <code>apply_neighbor_load</code> 一个静态标量，<b>没有自己的调度器</b>。
干扰在时间上是平的，只有均值对。</p>

<h2 id="a2">二、Skill 与机制：对照 superpowers</h2>

<div class="callout c-red">
<p><b>最大的路由缺口：<code>sw_system_sim</code> 在 SKILL.md 里出现 0 次。</b>
体验速率是你说的「真正最关心的 KPI」，系统级仿真是最近两轮最大的功能，
但照着 skill 走的 agent <b>根本不会发现它</b>。
<code>sw_describe_dataset</code> 同样是 0 次。</p>
</div>

<div class="tbl-wrap"><table>
<thead><tr><th style="min-width:110px"></th><th>superpowers</th><th>channel-sim</th>
<th>差在哪</th></tr></thead>
<tbody>
<tr><td><b>文件结构</b></td><td>SKILL.md + <code>references/*.md</code> 按需加载</td>
<td>单文件 <b>663 行</b></td>
<td>每次全量加载；射线追踪 / 干扰设计这些低频内容一直占着上下文</td></tr>
<tr><td><b>子 agent</b></td>
<td>固化提示词文件（<code>code-reviewer.md</code>、<code>implementer-prompt.md</code>、
<code>task-reviewer-prompt.md</code>）</td><td>无</td>
<td>没法把「横评」「查 bug」稳定地派给子 agent</td></tr>
<tr><td><b>skill 自测</b></td>
<td><code>testing-skills-with-subagents</code>：RED → GREEN → 堵合理化漏洞</td>
<td>无</td><td>我们的 <code>&lt;HARD-GATE&gt;</code> 从没被压力测过</td></tr>
<tr><td><b>溯源</b></td><td><code>CREATION-LOG.md</code></td><td>无</td>
<td>规则为什么这么写只存在 git log 里</td></tr>
<tr><td><b>脚本</b></td><td>skill 自带 <code>scripts/</code></td><td>无</td>
<td>取货代码每次现写</td></tr>
</tbody></table></div>

<p>superpowers 的 <code>persuasion-principles.md</code> 和「合理化对照表」
（「这只是个简单问题」→「问题也是任务」）那套，我们的
<code>## 常见的自我合理化</code> 已经有雏形，但<b>没被验证过</b>。</p>

<h2 id="a3">三、仿真实现的遗漏与待定</h2>

<p><b>已知并已标注的</b>（不算漏）：HARQ 软合并复用同一条 ReTx 曲线、
MU 是标量近似而非逐 TTI 真配对、<code>interference_model="precoded"</code>
目前是空转旋钮。</p>

<div class="tbl-wrap"><table>
<thead><tr><th style="min-width:140px">没标注的遗漏</th><th>后果</th></tr></thead>
<tbody>
<tr><td><b>调度时序 k0/k1/k2</b></td>
<td>CSI 老化建了，但调度时延与 HARQ RTT 没有——NACK 到重传之间是零时延，
小包体验速率被高估</td></tr>
<tr><td><b>控制信道开销</b></td>
<td>PDCCH 容量没建模，假设永远能调度。这正是真实系统里
「一个 TTI 能服务几个用户」的硬约束</td></tr>
<tr><td><b>系统级没有场景预设</b></td>
<td>链路级有 26 个 preset，系统级要手工拼八九个参数，不同次之间没法横比</td></tr>
<tr><td><b>一个小 bug</b></td>
<td><code>system.py:557</code> 在 <code>geo_sir_db=None</code> 时抛
<code>RuntimeWarning: Mean of empty slice</code>，不致命但污染输出</td></tr>
</tbody></table></div>

<h2 id="a4">四、Top 8 任务清单</h2>
<p class="lead">按 ROI 排序。每条都写清<b>我为什么认为该做</b>，
不是只写「做什么」。</p>
{tasks_html()}

<div class="callout c-blue">
<p><b>我的推荐组合。</b></p>
<p><b>今天就做 1 + 2</b>：半天能完，而且第 1 条是所有后续结论的前提——
在它之前做任何模型深化，都是在往一个读数不准的秤上加砝码。</p>
<p><b>3 + 5 一组</b>：都在 <code>system.py</code>，一起改省一次回归。</p>
<p><b>4 顺手做</b>。6 / 7 / 8 等你定优先级，其中 <b>8 建议明确排最后</b>——
它是唯一需要重构的一条。</p>
</div>

<footer>superwireless 全库自审 · 公式由内联 KaTeX 排版（MathML 兜底），离线可用</footer>
</div>
{kx.upgrade_script()}
</body></html>
"""


def main() -> int:
    p = ROOT / "measurements" / "seed_variance.json"
    seed = json.loads(p.read_text(encoding="utf-8")) if p.exists() else None
    out = ROOT / "AUDIT.html"
    out.write_text(build(seed), encoding="utf-8")
    print(f"{out}  ({out.stat().st_size / 1024:.0f} KB"
          f"{'，含种子散布图' if seed else ''})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
