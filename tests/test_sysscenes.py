"""系统级场景预设。

分节：
1. 加载与解析
2. 引用完整性（信道预设必须真存在）
3. 成对场景的受控性 —— **这是 pair_with 唯一有价值的地方**
4. expect 的诚实性
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("SUPERWIRELESS_NO_BROWSER", "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from superwireless import plan as pl  # noqa: E402
from superwireless import sysscenes as ss  # noqa: E402

_n_pass = 0
_n_fail = 0


def check(cond: bool, msg: str) -> None:
    global _n_pass, _n_fail  # noqa: PLW0603
    if cond:
        _n_pass += 1
        print(f"  PASS  {msg}")
    else:
        _n_fail += 1
        print(f"  FAIL  {msg}")


def section(title: str) -> None:
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


# ---------------------------------------------------------------------------
section("1  加载与解析")
# ---------------------------------------------------------------------------
scenes = ss.load_system_presets()
check(len(scenes) >= 6, f"至少 6 个系统级场景（实得 {len(scenes)}）")

_need = ("label", "summary", "answers", "channel_preset", "generate",
         "system", "expect")
for name, s in scenes.items():
    miss = [k for k in _need if k not in s]
    check(not miss, f"{name} 字段齐全（缺 {miss}）")

lst = ss.list_system_presets()
check(len(lst) == len(scenes), "清单条数与预设数一致")
check(all(r["answers"] for r in lst), "每个场景都写了它回答什么问题")

r = ss.resolve("sys_urban_macro_field_csi")
check(r.generate_kwargs["preset"] == "company_64t4r_multicell",
      "resolve 出的信道预设正确")
check("duration_s" in r.sim_kwargs and "csi_aging" in r.sim_kwargs,
      "sim_kwargs 带齐系统级参数")

# ---------------------------------------------------------------------------
section("2  引用完整性")
# ---------------------------------------------------------------------------
# **信道预设名写错必须早报。** 不校验的话会一路跑到生成阶段才炸，
# 而那时报的是 ChannelHub 的错，看起来像环境问题。
known = set(pl.load_presets())
for name, s in scenes.items():
    check(s["channel_preset"] in known,
          f"{name} 引用的信道预设 {s['channel_preset']} 存在")

try:
    ss.resolve("这个场景不存在")
    check(False, "未知场景名应当被拒")
except KeyError:
    check(True, "未知场景名被拒（带可用清单）")

# **每 UE 至少 8 个快照**：少于这个数 PF 拿不到多用户分集、
# CSI 老化的「陈旧信道」和「当前信道」会是同一个矩阵，效果恒为 0。
for name, s in scenes.items():
    g = s["generate"]
    n_slots = int(g.get("num_slots_per_sample", 1))
    check(n_slots >= 8, f"{name} 的 num_slots_per_sample={n_slots} >= 8")
    check(int(g["num_samples"]) % int(g["num_ues"]) == 0,
          f"{name} 的样本数能被 UE 数整除（ChannelHub 硬约束）")

# ---------------------------------------------------------------------------
section("3  成对场景的受控性")
# ---------------------------------------------------------------------------
bad = ss.check_pairs()
for b in bad:
    print("   ", b)
check(not bad, f"全部成对场景都是受控对比（{len(bad)} 处违规）")

pairs = [(n, s["pair_with"]) for n, s in scenes.items() if s.get("pair_with")]
check(len(pairs) >= 4, f"至少两对对照组（实得 {len(pairs) // 2} 对）")
for a, b in pairs:
    check(scenes[b].get("pair_with") == a, f"{a} ↔ {b} 双向指认")

# **负向测试：校验器真的拦得住吗。** 只断言"当前合规"是不够的——
# 一个永远返回空列表的 check_pairs 也能让上面全绿。
_orig = ss.load_system_presets
try:
    _fake = {
        "A": {"pair_with": "B", "pair_varies": ["neighbor_prb_util"],
              "channel_preset": "company_64t4r",
              "system": {"neighbor_prb_util": 0.9, "duration_s": 3.0},
              "generate": {"ue_speed_kmh": 3.0}},
        "B": {"pair_with": "A", "pair_varies": ["neighbor_prb_util"],
              "channel_preset": "company_64t4r",
              # 除了声明要变的负载，还偷偷改了 duration_s 和速度
              "system": {"neighbor_prb_util": 0.1, "duration_s": 5.0},
              "generate": {"ue_speed_kmh": 30.0}},
    }
    ss.load_system_presets = lambda: _fake  # type: ignore[assignment]
    caught = ss.check_pairs()
    check(any("duration_s" in c for c in caught),
          "偷改 duration_s 被抓出来（system 块）")
    check(any("ue_speed_kmh" in c for c in caught),
          "偷改 ue_speed_kmh 被抓出来（generate 块）")

    _fake2 = {"A": {"pair_with": "B", "channel_preset": "company_64t4r"},
              "B": {"pair_with": "A", "channel_preset": "company_64t4r"}}
    ss.load_system_presets = lambda: _fake2  # type: ignore[assignment]
    check(any("pair_varies" in c for c in ss.check_pairs()),
          "声明了对照组却没写要对比哪一项，被抓出来")

    _fake3 = {"A": {"pair_with": "B", "pair_varies": ["x"],
                    "channel_preset": "company_64t4r"},
              "B": {"channel_preset": "company_64t4r"}}
    ss.load_system_presets = lambda: _fake3  # type: ignore[assignment]
    check(any("指回来" in c for c in ss.check_pairs()),
          "单向指认被抓出来（成对关系必须双向）")
finally:
    ss.load_system_presets = _orig  # type: ignore[assignment]
    ss.load_system_presets.cache_clear()

# ---------------------------------------------------------------------------
section("4  expect 的诚实性")
# ---------------------------------------------------------------------------
# CLAUDE.md：「preset 里的 label 是设计意图，写着「高干扰」实际只有 2 dB 的事
# 发生过」。所以没测过就必须标 measured: false，不许拿设计意图充数。
_num_keys = ("avg_mcs", "bler_first_tx", "cell_experienced_mbps",
             "ue_experienced_p5_mbps", "olla_db_mean", "iot_db_median")
for name, s in scenes.items():
    e = s["expect"]
    check("measured" in e, f"{name} 的 expect 显式标了 measured")
    nums = [k for k in _num_keys if k in e]
    if not e.get("measured"):
        check(not nums,
              f"{name} 标了未实测就不许有数值锚点（发现 {nums}）")
        check(bool(e.get("note")), f"{name} 未实测时写了 note 说明待办")
    else:
        check(bool(nums), f"{name} 标了已实测就必须有数值锚点")
        check(bool(e.get("dataset")), f"{name} 已实测时标出了数据来源数据集")
        check(bool(e.get("note")), f"{name} 已实测时写了口径说明")

_m = [n for n, s in scenes.items() if s["expect"].get("measured")]
print(f"  已实测 {len(_m)}/{len(scenes)}：{_m}")
check(len(_m) >= 3, "至少三个场景有实测锚点")

# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print(f"系统级场景预设：{_n_pass} 通过，{_n_fail} 失败")
print("=" * 70)
if _n_fail:
    sys.exit(1)
print("系统级场景预设全部通过。")
