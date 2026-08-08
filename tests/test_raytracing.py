"""射线追踪与深化决策层测试。

射线追踪比统计信道慢一个量级，所以这里只跑极小配置。
直接运行：python tests/test_raytracing.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from superwireless import channelhub as ch  # noqa: E402
from superwireless import decisions as dec  # noqa: E402
from superwireless import generate as gen  # noqa: E402
from superwireless import plan as pl  # noqa: E402
from superwireless import scenes as sc  # noqa: E402
from superwireless import load  # noqa: E402

FAILED: list[str] = []


def check(cond: bool, label: str) -> None:
    print(("  PASS  " if cond else "  FAIL  ") + label)
    if not cond:
        FAILED.append(label)


def sect(t: str) -> None:
    print("\n" + "=" * 68 + f"\n{t}\n" + "=" * 68)


# ---------------------------------------------------------------------------
sect("1  射线追踪引擎可用性")
caps = {c.name: c for c in ch.probe_capabilities()}
rt = caps.get("sionna_rt")
if rt is None:  # 兼容旧版 probe 的提前返回；恒报三引擎后不会再触发
    rt = ch.Capability("sionna_rt", False, "引擎探测未报告 sionna_rt", ["sionna.rt"])
print(f"  sionna_rt: {'可用' if rt.available else '不可用'}  {rt.detail}")
# sionna-rt 是可选依赖。没装时本文件后半段会自动跳过实跑，但前半段的场景发现、
# 决策层、陷阱拦截都不依赖它，仍然要测。硬断言"可用"会让没装的人看到假失败。
check(
    rt.available or bool(rt.missing),
    "sionna_rt 可用性报告与事实一致" + ("（已装）" if rt.available else "（未装，已列出缺失项）"),
)
RT_OK = bool(rt.available)

# 无 ChannelHub 时，场景清单/决策/陷阱/实跑各节全部依赖内核——如实跳过，不算失败。
# （INSTALL_AGENT.md 第 7 步承诺"缺依赖自动跳过"，离线包环境正是这种情况。）
if any("ChannelHub" in c.missing for c in caps.values()):
    sect("2~5  依赖 ChannelHub —— 跳过")
    print("  未找到 ChannelHub 源码，后续各节依赖内核，全部跳过（不算失败）。")
    print("  补齐方式见 INSTALL_AGENT.md 第 2 步。")
    raise SystemExit(0)

# ---------------------------------------------------------------------------
sect("2  场景清单")
all_scenes = sc.list_scenes()
for s in all_scenes:
    tag = "内置" if s.builtin else "真实OSM"
    print(f"  {s.scene_id:<22} {tag:<8} {s.display_name[:26]:<28} presets={list(s.presets)}")
check(len(all_scenes) >= 10, f"至少 10 个场景（实际 {len(all_scenes)}）")
check(sum(1 for s in all_scenes if s.builtin) == 4, "4 个 Sionna 内置场景")
check(sum(1 for s in all_scenes if s.needs_preparation) >= 6, "6 个中国城市场景带本地资产")

# ---------------------------------------------------------------------------
sect("3  场景资产准备（修 PLY 头，不改 ChannelHub）")
t0 = time.perf_counter()
prep = sc.prepare_scene("shenzhen_futian", force=True)
print(f"  准备耗时 {time.perf_counter()-t0:.1f}s")
print(f"  PLY 总数 {prep['ply_total']}，修复 {prep['ply_fixed']} 个")
print(f"  缓存位置 {prep['osm_path']}")
check(prep["prepared"], "场景准备完成")
check(prep["ply_fixed"] >= 1, "确实修复了带 obj_info 的 PLY")

orig = sc.scenes_dir() / "shenzhen_futian" / "mesh" / "ground.ply"
if orig.is_file():
    check(b"obj_info" in orig.read_bytes()[:400], "ChannelHub 原文件未被修改")

cached = sc.prepare_scene("shenzhen_futian")
check(cached.get("cached") is True, "第二次调用命中缓存")

# ---------------------------------------------------------------------------
# 第 4~5 节要真跑射线追踪，没装 sionna-rt 就跳过。
# sionna-rt 是可选依赖（约 300 MB），主功能不依赖它——把这两节写成硬失败，
# 会让按安装文档装完但没装射线追踪的人看到 FAILED，误以为装坏了。
if not RT_OK:
    sect("4~5  射线追踪实跑 —— 跳过")
    print("  未安装 sionna-rt，跳过慕尼黑与深圳福田的实跑。")
    print("  需要射线追踪时： pip install sionna-rt")
else:
# ---------------------------------------------------------------------------
    sect("4  内置场景射线追踪（慕尼黑）")
    d, p = pl.create_draft(
        "在慕尼黑真实地图上验证覆盖",
        overrides={"num_ues": 1, "num_samples": 1, "bs_antenna": "4T4R", "bandwidth_hz": 20000000.0},
    )
    print(f"  自动选中预设: {d.preset}")
    check(d.preset.startswith("rt_"), "意图含城市名时自动走射线追踪预设")

    cfg, own = pl.resolved_config(d)
    cfg.pop("num_samples", None)
    print(f"  scenario={cfg.get('scenario')}  device={cfg.get('device')}")
    t0 = time.perf_counter()
    s1 = gen.generate(cfg, num_samples=1)
    dt = time.perf_counter() - t0
    print(f"  生成 {dt:.1f}s  形状 {s1['shape']}")
    print(f"  SINR {s1['sinr_dB']['median']} dB  路损 {s1.get('pathloss_dB', {}).get('median')} dB")
    check(s1["num_samples"] == 1, "慕尼黑场景生成成功")

    ds1 = load(s1["dataset_id"])
    mode = ds1.summary.get("sample_meta", {}).get("channel_generation_mode")
    print(f"  channel_generation_mode = {mode}")
    check(mode == "sionna_rt", "确认走的是真射线追踪，不是 TDL 回退")

# ---------------------------------------------------------------------------
    sect("5  真实城市射线追踪（深圳福田）")
    d2, p2 = pl.create_draft(
        "深圳福田密集城区覆盖分析",
        overrides={"num_ues": 1, "num_samples": 1, "bs_antenna": "4T4R", "bandwidth_hz": 20000000.0},
    )
    print(f"  预设 {d2.preset}")
    cfg2, own2 = pl.resolved_config(d2)
    cfg2.pop("num_samples", None)
    print(f"  scenario={cfg2.get('scenario')}  站点={cfg2.get('num_sites')}x{cfg2.get('sectors_per_site')}")
    print(f"  osm_path={str(cfg2.get('osm_path'))[-46:]}")
    check(cfg2.get("scenario") == "custom_osm", "真实城市走 custom_osm")
    check("artifacts" in str(cfg2.get("osm_path", "")), "osm_path 指向准备好的缓存副本")

    t0 = time.perf_counter()
    s2 = gen.generate(cfg2, num_samples=1)
    print(f"  生成 {time.perf_counter()-t0:.1f}s  形状 {s2['shape']}")
    print(f"  SINR {s2['sinr_dB']['median']} dB  视距比例 {s2.get('los_ratio')}")
    ds2 = load(s2["dataset_id"])
    mode2 = ds2.summary.get("sample_meta", {}).get("channel_generation_mode")
    print(f"  channel_generation_mode = {mode2}  小区数 {ds2.summary['sample_meta'].get('num_cells')}")
    check(mode2 == "sionna_rt", "深圳福田走真射线追踪")
    check(bool(ds2.ssb), "多小区 SSB 测量可用")

    print("\n  正确性护栏：射线追踪数据不得套用 CDL 剖面的假角度")
    check(ds2.is_ray_traced, "数据集自报为射线追踪")
    try:
        ds2.paths()
        check(False, "paths() 在射线追踪数据上应当报错")
    except NotImplementedError as e:
        print(f"    已拦截：{str(e).splitlines()[0][:70]}…")
        check(True, "paths() 在射线追踪数据上正确报错")

    from superwireless import deliver as dlv  # noqa: E402

    res_rt = dlv.build_code(s2["dataset_id"], "信道 + 角度")
    print(f"    取货提示 {len(res_rt['notes'])} 条")
    check(any("射线追踪" in n for n in res_rt["notes"]), "取货代码给出射线追踪说明")

    # 常规量在射线追踪数据上照常可用
    p_rt = ds2.pdp(0)
    srs_rt = ds2.srs(0)
    print(f"    PDP RMS 时延扩展 {p_rt.rms_delay_spread_s*1e9:.1f} ns | 主导秩 {srs_rt.dominant_rank}")
    check(p_rt.rms_delay_spread_s > 0, "射线追踪数据的 PDP 仍可用")

# ---------------------------------------------------------------------------
sect("6  实验设计层（superpowers 式头脑风暴）")
d3, p3 = pl.create_draft("验证一个 CSI 压缩的想法")
prop = pl.build_proposal(d3, p3)
print(f"  任务 {prop['task_label']}")
print("\n  实验设计问题（先问这层）：")
for q in prop["design_questions"]:
    mark = "可选" if q["optional"] else "建议问"
    print(f"    [{mark}] {q['question']}")
    print(f"           why: {q['why'][:66]}…")
    print(f"           例:  {' / '.join(q['examples'][:3])}")
check(len(prop["design_questions"]) >= 2, "提供了实验设计层问题")
check(all(q["why"] and q["examples"] for q in prop["design_questions"]), "设计问题都带 why 和示例")

print(f"\n  第 {prop['round']} 轮 · {prop['round_focus']}")
print(f"    {prop['round_rationale']}")
q0 = prop["design_questions"][0]
print(f"\n    {q0['question']}")
for i, o in enumerate(q0["options"], 1):
    star = "  ← 推荐" if o.get("recommended") else ""
    print(f"      {i}) {o['label']}  —— {o['note'][:40]}{star}")
print("      或者你直接说")

n_this_round = len(prop["design_questions"]) + len(prop["questions"])
print(f"\n    本轮 {n_this_round} 问，还剩 {prop['remaining_count']} 项")
check(2 <= n_this_round <= 6, f"一轮问 2~6 个（实际 {n_this_round}）")
check(all(3 <= len(q["options"]) <= 4 for q in prop["design_questions"] + prop["questions"]),
      "每个问题 3~4 个选项")
check(all(any(o.get("recommended") for o in q["options"])
          for q in prop["design_questions"] + prop["questions"] if not q.get("optional")),
      "必答问题都标出了推荐项")
check(all(q["allow_free"] for q in prop["design_questions"]), "允许自由作答，选项只用于启发")
check(prop["can_generate_now"], "任何一轮之后都能直接生成")

# --- 多轮推进 ---
print("\n  多轮推进：")
seen_questions: set[str] = set()
dr, pr = d3, p3
n_rounds = 0
for _ in range(6):
    pp = pl.build_proposal(dr, pr)
    n_rounds += 1
    keys = [q["key"] for q in pp["design_questions"] + pp["questions"]]
    dup = seen_questions & set(keys)
    print(f"    第 {pp['round']} 轮 · {pp['round_focus']}：{len(keys)} 问 {keys}")
    check(not dup, f"第 {pp['round']} 轮不重复问已答过的（重复：{dup or '无'}）")
    seen_questions |= set(keys)
    if not pp["has_more_rounds"]:
        print(f"    → has_more_rounds=false，停止提问")
        break
    ov = {q["key"]: q["default"] for q in pp["questions"]}
    dg = {q["key"]: "（用户已答）" for q in pp["design_questions"]}
    dr, pr, _ = pl.revise_draft(dr.draft_id, overrides=ov or None, design=dg or None)
check(not pp["has_more_rounds"], "有限轮内收敛完毕")
# 用户明确要求把提问压到 2~3 轮：轮次比单轮长度更让人烦。
# 设计层与参数层没有先后依赖，已合并在第 1 轮一起问。
print(f"    收敛用了 {n_rounds} 轮")
check(n_rounds <= 3, f"2~3 轮内问完（实际 {n_rounds} 轮）")

# 每一轮都要给出"还剩下的是不是都有默认值"，好让 Agent 把最后一轮
# 包装成一句可跳过的话，而不是又摆一屏选项
_p1 = pl.build_proposal(*pl.create_draft("验证 CSI 压缩"))
check("remaining_all_optional" in _p1 and "target_rounds" in _p1,
      "提案带 remaining_all_optional / target_rounds，供 Agent 决定要不要再问一轮")

# 用户确认默认值（值没变）也应推进轮次，否则会重复问
d_c, p_c = pl.create_draft("验证 CSI 压缩")
r_before = pl.build_proposal(d_c, p_c)["round"]
d_c, p_c, ch_c = pl.revise_draft(d_c.draft_id, design={"baseline": "Type II"})
d_c2, _, ch2 = pl.revise_draft(d_c.draft_id, overrides={"channel_model": "CDL-C"})
print(f"\n    确认默认值时 changes={ch2}（空），轮次 {r_before} → {d_c2.round_no}")
check(d_c2.round_no > r_before + 1, "确认默认值也推进轮次，不会重复问")

print("\n  建议的对比组：")
for s in prop["suggested_sweeps"]:
    print(f"    · {s['label']}: {s['key']} = {s['values']}")
    print(f"      why: {s['why'][:66]}…")
check(len(prop["suggested_sweeps"]) >= 1, "给出对比组建议")

print("\n  常见陷阱：")
for pf in prop["pitfalls"]:
    print(f"    · {pf[:76]}")
check(len(prop["pitfalls"]) >= 2, "给出常见陷阱提示")

# ---------------------------------------------------------------------------
sect("7  实验设计写进计划书")
d4, p4, changes = pl.revise_draft(
    d3.draft_id,
    design={"baseline": "3GPP Type II 码本", "metric": "NMSE 与频谱效率损失"},
)
md = pl.render_plan_markdown(d4, p4, ["channel", "pmi"])
print(md[: md.find("## 关键选择")])
check("## 实验设计" in md, "计划书含实验设计章节")
check("Type II" in md, "基线写进了计划书")

# ---------------------------------------------------------------------------
sect("8  新增任务类型与拦截")
cases = [
    ("做信道预测，用 LSTM 外推未来时隙", "channel_prediction"),
    ("CQI 上报和 MCS 选择的自适应算法", "link_adaptation"),
    ("信道表征学习，做对比学习预训练", "channel_charting"),
]
for text, expect in cases:
    prof = dec.classify_intent(text)
    ok = prof.task == expect
    print(f"  {'OK ' if ok else 'ERR'} {text[:30]:<32} → {prof.task}")
    check(ok, f"新任务类型识别：{expect}")

prof_pred = dec.classify_intent("信道预测")
issues = dec.check_guards(prof_pred, {"num_slots_per_sample": 1, "ue_speed_kmh": 3.0})
for i in issues:
    print(f"  [{i['severity']}] {i['key']}: {i['message'][:62]}…")
check(any(i["severity"] == "block" for i in issues), "信道预测 + 单时隙被拦截")

# ---------------------------------------------------------------------------
print("\n" + "=" * 68)
if FAILED:
    print(f"FAILED {len(FAILED)} 项：")
    for f in FAILED:
        print("  - " + f)
    sys.exit(1)
print("射线追踪与决策层全部通过。")
