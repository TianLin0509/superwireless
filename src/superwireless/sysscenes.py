"""系统级场景预设：把一整套仿真条件打包成一个名字。

## 为什么需要它

链路级有 26 个信道预设，一句 ``company_64t4r_multicell`` 就够了。
系统级却要手工填 ``duration_s`` / ``traffic_model`` / ``arrival_rate_hz`` /
``scheduler`` / ``pf_window_tti`` / ``neighbor_prb_util`` / ``csi_aging`` /
``srs_period_ms`` / ``olla_speedup`` 八九个参数。

后果不只是麻烦——**每次跑都在拍参数，不同次之间参数不一致，结果没法横向比**。
这和"系统级没有置信区间"是同一个病根：结论站不住。

## 它比"一组默认值"多两样东西

1. **实测锚点** ``expect``。CLAUDE.md 里有条规矩：*preset 里的 label 是设计意图，
   写着「高干扰」实际只有 2 dB 的事发生过*。所以场景名必须有实测值背书，
   没测过的字段宁可留空并标 ``measured: false``，**不许照抄设计意图当实测值**。
2. **成对设计** ``pair_with``。很多系统级结论必须靠 A/B 才立得住
   （满载 vs 轻载、老化 vs 零时延）。两个场景除了要对比的那一项，
   其余条件**逐字相同**——这才叫受控对比。信道侧的
   ``high_iot_dense`` / ``low_iot_reference`` 就是这个套路。

## 三层结构

一个系统级场景 = **信道预设**（复用已有的 26 个）
+ **生成覆盖**（每 UE 要几个快照、速度）
+ **系统级参数**（话务、调度、负载、CSI 时延链）。

不重新定义信道，是因为信道预设已经各自带了实测的 ``expect``；
重复一份必然会漂。
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

__all__ = [
    "load_system_presets",
    "list_system_presets",
    "resolve",
    "SystemScene",
]

_PRESET_DIR = Path(__file__).resolve().parents[2] / "presets"
_FILE = _PRESET_DIR / "system_presets.yaml"


class SystemScene(dict):
    """一个解析好的系统级场景。dict 的薄封装，方便直接进 JSON。"""

    @property
    def generate_kwargs(self) -> dict[str, Any]:
        """喂给 ``sw_generate`` 的 ``preset`` + ``overrides``。"""
        return {"preset": self["channel_preset"],
                "overrides": dict(self.get("generate") or {}),
                "num_samples": (self.get("generate") or {}).get("num_samples")}

    @property
    def sim_kwargs(self) -> dict[str, Any]:
        """喂给 ``sw_system_sim`` 的参数（不含 ``dataset_id``）。"""
        return dict(self.get("system") or {})


@lru_cache(maxsize=1)
def load_system_presets() -> dict[str, dict[str, Any]]:
    if not _FILE.is_file():
        return {}
    data = yaml.safe_load(_FILE.read_text(encoding="utf-8")) or {}
    return dict(data)


def list_system_presets() -> list[dict[str, Any]]:
    """给用户看的清单：名字、一句话、回答什么问题、有没有对照组。"""
    out = []
    for name, s in load_system_presets().items():
        exp = s.get("expect") or {}
        out.append({
            "name": name,
            "label": s.get("label", ""),
            "answers": s.get("answers", []),
            "channel_preset": s.get("channel_preset"),
            "pair_with": s.get("pair_with"),
            "expect_measured": bool(exp.get("measured")),
            "expect_note": exp.get("note", ""),
        })
    return out


def resolve(name: str) -> SystemScene:
    """按名字取一个场景。**信道预设必须真实存在**，否则早报错。

    不做存在性校验的话，写错一个信道预设名会一路跑到生成阶段才炸，
    而那时报的是 ChannelHub 的错，看起来像环境问题。
    """
    presets = load_system_presets()
    if name not in presets:
        raise KeyError(f"没有系统级场景 {name!r}，可用的：{sorted(presets)}")
    s = dict(presets[name])

    from . import plan as _plan  # noqa: PLC0415

    ch = s.get("channel_preset")
    known = _plan.load_presets()
    if ch not in known:
        raise ValueError(
            f"系统级场景 {name!r} 引用的信道预设 {ch!r} 不存在。"
            f"可用的信道预设：{sorted(known)}")

    pair = s.get("pair_with")
    if pair and pair not in presets:
        raise ValueError(f"系统级场景 {name!r} 的对照组 {pair!r} 不存在")
    s["name"] = name
    return SystemScene(s)


def check_pairs() -> list[str]:
    """校验成对场景的**受控性**：除了要对比的那一项，其余必须逐字相同。

    这是 ``pair_with`` 唯一有价值的地方——两个场景如果同时差了三四个参数，
    比出来的差值归因不到任何一项上，那就不是对照组，只是两个场景。
    返回违规说明列表，空列表表示全部合规。
    """
    presets = load_system_presets()
    bad: list[str] = []
    for name, s in presets.items():
        pair = s.get("pair_with")
        if not pair or pair not in presets:
            continue
        if presets[pair].get("pair_with") != name:
            bad.append(f"{name} 指向 {pair}，但对方没指回来（成对关系必须双向）")
        varies = set(s.get("pair_varies") or [])
        if not varies:
            bad.append(f"{name} 声明了对照组但没写 pair_varies（要对比的是哪一项）")
            continue
        o = presets[pair]
        for block in ("generate", "system"):
            a, b = dict(s.get(block) or {}), dict(o.get(block) or {})
            diff = {k for k in set(a) | set(b) if a.get(k) != b.get(k)}
            extra = diff - varies
            if extra:
                bad.append(f"{name} vs {pair} 在 {block} 里还差了 {sorted(extra)}，"
                           f"但只声明要对比 {sorted(varies)}——这不是受控对比")
        if s.get("channel_preset") != o.get("channel_preset") \
                and "channel_preset" not in varies:
            bad.append(f"{name} vs {pair} 用的信道预设都不同，却没声明")
    return bad
