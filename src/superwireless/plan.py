"""仿真计划：意图 → 提案 → 定稿。

一份计划书同时是给人看的实验记录和给机器执行的指令。draft 落盘保存，
所以协商可以跨会话继续，也便于事后复现。
"""
from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from . import decisions as dec
from .paths import drafts_dir, presets_file

# ---------------------------------------------------------------------------
# 抽象参数 → ChannelHub 实参 的翻译
# ---------------------------------------------------------------------------
# 决策点用的是人话（"64T4R"），ChannelHub 要的是具体键。这一层负责翻译，
# 也负责把 ChannelHub 根本不支持的参数（如 snr_range_dB）挑出来另作处理。

_ANTENNA_PRESETS: dict[str, dict[str, int]] = {
    "64T4R": {"num_bs_tx_ant": 64, "num_bs_rx_ant": 64, "num_ue_tx_ant": 4, "num_ue_rx_ant": 4},
    "32T4R": {"num_bs_tx_ant": 32, "num_bs_rx_ant": 32, "num_ue_tx_ant": 4, "num_ue_rx_ant": 4},
    "16T2R": {"num_bs_tx_ant": 16, "num_bs_rx_ant": 16, "num_ue_tx_ant": 2, "num_ue_rx_ant": 2},
    "4T4R": {"num_bs_tx_ant": 4, "num_bs_rx_ant": 4, "num_ue_tx_ant": 4, "num_ue_rx_ant": 4},
}

# ChannelHub 不认识、由 superwireless 自己消化的键
# scene 会展开成 scenario / osm_path / 站点布局（见 scenes.resolve_scene_config）
#
# 注意 antenna_preset 这个名字：它是"64T4R"这类简写标签，展开成 num_bs_tx_ant 等。
# **不能叫 bs_antenna** —— ChannelHub 自己有一个 bs_antenna 配置块（嵌套 dict，
# 含 port_order / element_pattern / fixed_vertical_subarray），是描述 1驱3 子阵这类
# 阵列细节用的。两者重名会让阵列配置被静默吞掉。
_SUPERWIRELESS_ONLY = {
    "antenna_preset", "snr_range_dB", "measurements_wanted", "scene", "scene_site_preset",
}


def antenna_label(params: dict[str, Any]) -> str | None:
    """从具体天线数反推标签。preset 直接给了 num_bs_* 时用它，避免默认标签把 preset 冲掉。"""
    bs = params.get("num_bs_tx_ant")
    ue = params.get("num_ue_rx_ant")
    if bs is None:
        return None
    for label, spec in _ANTENNA_PRESETS.items():
        if spec["num_bs_tx_ant"] == bs and spec["num_ue_rx_ant"] == ue:
            return label
    return f"{bs}T{ue}R"


def translate(params: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """拆成 (ChannelHub 实参, superwireless 自用参数)。

    antenna_preset 这类抽象参数最后展开，因为它要覆盖具体的 num_bs_* ——
    能出现在 params 里就说明是被明确选定的阵型。

    ChannelHub 自己的 ``bs_antenna``（嵌套 dict：port_order / element_pattern /
    fixed_vertical_subarray 等，1驱3 子阵就配在这里）原样透传，不做任何解释。
    为兼容早期写法，字符串形式的 bs_antenna 仍按 antenna_preset 处理。
    """
    ch: dict[str, Any] = {}
    own: dict[str, Any] = {}
    antenna: str | None = None

    for k, v in params.items():
        if k == "antenna_preset":
            antenna = str(v)
            own[k] = v
        elif k == "bs_antenna":
            if isinstance(v, str):  # 早期写法：bs_antenna="64T4R"
                antenna = v
                own["antenna_preset"] = v
            else:  # ChannelHub 的阵列配置块，原样透传
                ch[k] = v
        elif k in _SUPERWIRELESS_ONLY:
            own[k] = v
        else:
            ch[k] = v

    if antenna is not None:
        spec = _ANTENNA_PRESETS.get(antenna)
        if spec is not None:
            ch.update(spec)

    # 射线追踪场景展开：scene -> scenario / osm_path / 站点布局。
    # 真实城市场景会在这里顺带完成资产准备（复制到缓存 + 修 PLY 头）。
    scene = own.get("scene")
    if scene:
        from .scenes import resolve_scene_config  # 延迟导入，避免非 RT 路径付出代价

        scene_cfg = resolve_scene_config(str(scene), own.get("scene_site_preset"))
        scene_cfg.pop("source", None)
        for k, v in scene_cfg.items():
            # 用户显式给过的值优先，场景只补没给的
            if k in ("scenario", "osm_path", "scene_preset") or k not in ch:
                ch[k] = v
    return ch, own


# ---------------------------------------------------------------------------
# Preset
# ---------------------------------------------------------------------------


def load_presets() -> dict[str, dict[str, Any]]:
    path = presets_file()
    if not path.is_file():
        return {}
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def preset_summaries() -> list[dict[str, Any]]:
    out = []
    for name, body in load_presets().items():
        cfg = body.get("config", {}) or {}
        item = {
            "preset": name,
            "group": body.get("group", "其他"),
            "label": body.get("label", name),
            "summary": body.get("summary", ""),
            "typical_for": body.get("typical_for", []),
            "num_sites": cfg.get("num_sites"),
            "num_cells": (
                (cfg.get("num_sites") or 1) * (cfg.get("sectors_per_site") or 1)
                if cfg.get("num_sites") else None
            ),
            "isd_m": cfg.get("isd_m"),
            "link": cfg.get("link", "DL"),
        }
        # 只有真跑过、把实测值写回 preset 的场景才有 expect；没有就不给，
        # 不用"设计意图"冒充实测。
        for key in ("expect", "verify", "caveat"):
            if body.get(key):
                item[key] = body[key]
        out.append(item)
    return out


def preset_groups() -> dict[str, list[str]]:
    """按 group 归类的预设名清单。"""
    out: dict[str, list[str]] = {}
    for name, body in load_presets().items():
        out.setdefault(body.get("group", "其他"), []).append(name)
    return out


_RT_HINTS: dict[str, str] = {
    "慕尼黑": "rt_munich", "munich": "rt_munich",
    "陆家嘴": "rt_shanghai_lujiazui", "上海": "rt_shanghai_lujiazui",
    "福田": "rt_shenzhen_futian", "深圳": "rt_shenzhen_futian",
}


def _guess_preset(intent: str, profile: dec.TaskProfile) -> str:
    """按意图挑一个场景骨架。多小区类任务自动升到 7 站。"""
    text = (intent or "").lower()
    for key in load_presets():
        if key in text:
            return key

    # 提到具体城市或射线追踪，走 RT 路径
    for hint, preset in _RT_HINTS.items():
        if hint in text:
            return preset
    if any(w in text for w in ("射线追踪", "ray tracing", "raytracing", "真实地图", "真实建筑", "osm")):
        return "rt_munich"
    if any(w in text for w in ("19 站", "19站", "19 site", "57")):
        return "multicell_19site"
    if any(w in text for w in ("室内", "工厂", "indoor", "factory")):
        return "indoor_factory"
    if "require_multicell" in profile.guards or any(
        w in text for w in ("多小区", "多站", "multi-cell", "multicell", "邻区", "干扰")
    ):
        return "company_64t4r_multicell"
    if any(w in text for w in ("最小", "冒烟", "快速", "先跑通", "smoke")):
        return "single_cell_4t4r"
    # 兜底走**本地默认配置**：真实 AAU（1 驱 3 / 192 阵子 / 0.5λ 水平 0.67λ 垂直）
    # + n41 2.6 GHz / 30 kHz / 272 RB / 4R 下行。
    # 旧的 single_cell_64t4r 是 3.5 GHz + legacy 独立阵元模型，留着做对照，
    # 但不该再当默认——实测两者吞吐差 27%、边缘用户差 61%。
    return "company_64t4r"


# ---------------------------------------------------------------------------
# Draft
# ---------------------------------------------------------------------------


@dataclass
class Draft:
    draft_id: str
    intent: str
    task: str
    task_label: str
    preset: str
    params: dict[str, Any] = field(default_factory=dict)
    user_set: list[str] = field(default_factory=list)  # 用户显式指定过的键
    design: dict[str, str] = field(default_factory=dict)  # 实验设计层的回答
    round_no: int = 1  # 当前问到第几轮
    created_at: float = field(default_factory=time.time)
    history: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "draft_id": self.draft_id,
            "intent": self.intent,
            "task": self.task,
            "task_label": self.task_label,
            "preset": self.preset,
            "params": self.params,
            "user_set": self.user_set,
            "design": self.design,
            "round_no": self.round_no,
            "created_at": self.created_at,
            "history": self.history,
        }


def _draft_path(draft_id: str) -> Path:
    return drafts_dir() / f"{draft_id}.json"


def save_draft(d: Draft) -> None:
    p = _draft_path(d.draft_id)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(d.as_dict(), ensure_ascii=False, indent=2), encoding="utf-8")


def load_draft(draft_id: str) -> Draft:
    p = _draft_path(draft_id)
    if not p.is_file():
        raise KeyError(f"找不到计划 {draft_id!r}；它可能已被清理，重新 plan 一次即可")
    raw = json.loads(p.read_text(encoding="utf-8"))
    return Draft(**raw)


def create_draft(
    intent: str,
    *,
    preset: str | None = None,
    overrides: dict[str, Any] | None = None,
) -> tuple[Draft, dec.TaskProfile]:
    """从自然语言意图建一份提案。"""
    profile = dec.classify_intent(intent)
    preset_name = preset or _guess_preset(intent, profile)
    presets = load_presets()
    if preset_name not in presets:
        preset_name = "single_cell_64t4r"

    params: dict[str, Any] = dict(presets.get(preset_name, {}).get("config", {}))
    params.update(profile.config_hints)

    # preset 若已给具体天线数，就按它反推标签，不要让默认标签把 preset 冲掉
    inferred = antenna_label(params)
    if inferred:
        params["antenna_preset"] = inferred

    # 决策点的默认值补进来（preset 已给的不覆盖）
    for d in dec.decisions_for(profile, limit=99):
        translated, _ = translate({d.key: d.default})
        for k, v in translated.items():
            params.setdefault(k, v)
        if d.key in _SUPERWIRELESS_ONLY:
            params.setdefault(d.key, d.default)

    params.setdefault("num_samples", 200)
    params.setdefault("seed", 42)

    user_set: list[str] = []
    if overrides:
        params.update(overrides)
        user_set = sorted(overrides)

    d = Draft(
        draft_id="d_" + uuid.uuid4().hex[:8],
        intent=intent,
        task=profile.task,
        task_label=profile.label,
        preset=preset_name,
        params=params,
        user_set=user_set,
        history=[f"由意图创建，场景骨架 {preset_name}"],
    )
    save_draft(d)
    return d, profile


def revise_draft(
    draft_id: str,
    overrides: dict[str, Any] | None = None,
    design: dict[str, str] | None = None,
) -> tuple[Draft, dec.TaskProfile, list[str]]:
    """差分修正：只说改什么，不用重述整个需求。

    ``design`` 记录实验设计层的回答（基线、指标、推广范围）。它不影响仿真
    参数，但会写进计划书——这是三个月后回看时最有价值的部分。
    """
    d = load_draft(draft_id)
    profile = next((p for p in dec.TASK_PROFILES if p.task == d.task), dec.TASK_PROFILES[-1])

    changes: list[str] = []
    for k, v in (overrides or {}).items():
        old = d.params.get(k)
        if old != v:
            changes.append(f"{k}: {old!r} → {v!r}")
        d.params[k] = v
        if k not in d.user_set:
            d.user_set.append(k)

    for k, v in (design or {}).items():
        if v:
            d.design[k] = str(v)
            changes.append(f"实验设计 {k}: {str(v)[:40]}")

    # 用户回应过一轮就推进轮次，下次 build_proposal 问新的一批。
    # 注意即使用户只是"确认默认值"（值没变、changes 为空）也要推进——
    # 否则下一轮会把同样的问题再问一遍。
    if overrides or design:
        d.round_no += 1
        d.history.append(
            f"第 {d.round_no - 1} 轮：" + ("；".join(changes) if changes else "确认默认值")
        )
    save_draft(d)
    return d, profile, changes


def resolved_config(d: Draft) -> tuple[dict[str, Any], dict[str, Any]]:
    """定稿：拆出真正交给 ChannelHub 的配置和自用参数。"""
    return translate(d.params)


# ---------------------------------------------------------------------------
# 提案渲染
# ---------------------------------------------------------------------------


def build_proposal(
    d: Draft,
    profile: dec.TaskProfile,
    *,
    max_questions: int = 6,
) -> dict[str, Any]:
    """组装给 Agent 看的提案。

    分两层交给 Agent：

    * ``design_questions`` —— 实验设计层（跟什么比、用什么指标、推广到哪）。
      这层没有默认值，也不影响仿真参数，但决定了这批数据能不能支撑
      用户想要的结论。**应当先问这层**，参数配错重跑就行，实验设计
      错了整个结论作废。
    * ``questions`` —— 仿真参数层，每条都带 why 和默认值。

    另外 ``sweeps`` 给出建议的对比组，``pitfalls`` 是这类课题的常见坑。
    """
    ch_cfg, own = resolved_config(d)

    # 本轮该问什么由 MCP 自己算：已答的不再问，一轮最多 4 个
    rnd = dec.next_round(
        profile,
        answered_design=set(d.design),
        answered_params=set(d.user_set),
        round_no=d.round_no,
    )

    questions = []
    for item in rnd["questions"]:
        questions.append(
            {
                **item,
                "current": d.params.get(item["key"], item["default"]),
                "user_specified": item["key"] in d.user_set,
            }
        )
    design = [{**q, "answered": d.design.get(q["key"])} for q in rnd["design_questions"]]

    issues = dec.check_guards(profile, d.params)
    presets = load_presets()

    return {
        "draft_id": d.draft_id,
        "task": d.task,
        "task_label": d.task_label,
        "preset": d.preset,
        "preset_label": presets.get(d.preset, {}).get("label", d.preset),
        "preset_summary": presets.get(d.preset, {}).get("summary", ""),
        # --- 本轮提问 ---
        "round": rnd["round"],
        "round_focus": rnd["focus"],
        "round_rationale": rnd["rationale"],
        # round_questions 是本轮全部问题的合并视图（设计层在前），
        # 调用方直接照着问即可，不必自己拼两个列表。
        # 第 1 轮通常只有设计层问题，questions 会是空的，这是正常的。
        "round_questions": [{**q, "layer": "design"} for q in design]
        + [{**q, "layer": "param"} for q in questions],
        "design_questions": design,
        "questions": questions,
        "has_more_rounds": rnd["has_more"],
        "remaining_count": rnd["remaining_count"],
        "target_rounds": rnd["target_rounds"],
        "remaining_all_optional": rnd["remaining_all_optional"],
        "stop_hint": rnd["stop_hint"],
        # --- 参考信息 ---
        "also_configurable": dec.also_configurable(profile),
        "suggested_sweeps": dec.sweep_suggestions(profile),
        "pitfalls": list(profile.pitfalls),
        "issues": issues,
        "ready_to_go": not any(i["severity"] == "block" for i in issues),
        "can_generate_now": True,
        "resolved_config": ch_cfg,
        "superwireless_params": own,
        "user_specified": d.user_set,
        "answered_design": dict(d.design),
        "hint": (
            "**目标 2 轮问完，最多 3 轮。** 这一轮只问 round_questions 里的这几个，"
            "别把 also_configurable 里的也问了。设计层和参数层互不依赖，"
            "已经合并在同一轮，照着列表一次性问出来即可。"
            "每个问题都带 options，把选项编号列出来并标明推荐项（recommended=true），"
            "最后留一句「或者你直接说」。"
            "用户答完后再调 sw_revise 拿下一轮；has_more_rounds 为 false "
            "或用户说「随便」就直接生成。"
            "remaining_all_optional 为 true 时，下一轮请包装成一句可跳过的话"
            "（「剩下这些都有合理默认值，要不要直接跑？」），不要再摆一屏选项。"
        ),
    }


def render_plan_markdown(d: Draft, profile: dec.TaskProfile, wanted: list[str]) -> str:
    """计划书：上半人话，下半配置。可存档、可交给同事复现。"""
    ch_cfg, own = resolved_config(d)
    lines = [
        f"# 仿真计划：{d.task_label}",
        "",
        "## 要验证什么",
        d.intent or "（未说明）",
        "",
    ]

    # 实验设计层：三个月后回看时最有价值的部分
    if d.design:
        labels = {
            "baseline": "对比基线", "metric": "评价指标",
            "scope": "结论适用范围", "hypothesis": "预期结果",
        }
        lines.append("## 实验设计")
        for k, v in d.design.items():
            lines.append(f"- **{labels.get(k, k)}**：{v}")
        lines.append("")

    lines.append("## 关键选择与理由")
    for item in dec.decisions_for(profile, limit=99):
        cur = d.params.get(item.key, item.default)
        mark = "用户指定" if item.key in d.user_set else "默认"
        first_sentence = item.why.split("；")[0].split("。")[0]
        lines.append(f"- **{item.question.rstrip('？')}**：`{cur}`（{mark}）—— {first_sentence}")

    lines += [
        "",
        "## 产出什么",
        "、".join(wanted) + "；其余测量量后续可再取，不必重跑仿真。",
        "",
        "## 场景骨架",
        f"`{d.preset}`",
        "",
        "---",
        "以下由 superwireless 执行：",
        "",
        "```yaml",
        yaml.safe_dump(ch_cfg, allow_unicode=True, sort_keys=True).rstrip(),
        "```",
    ]
    if own:
        lines += ["", "superwireless 自用参数：", "", "```yaml",
                  yaml.safe_dump(own, allow_unicode=True, sort_keys=True).rstrip(), "```"]
    if d.history:
        lines += ["", "## 修改记录", *[f"- {h}" for h in d.history]]
    return "\n".join(lines)
