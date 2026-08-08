"""信道生成与落盘。

数据一律落盘，MCP 只回句柄和摘要——信道矩阵进不了对话上下文，
详见设计文档 v1 第二节。

存储用 .npz（numpy 原生，不依赖 torch），一个数据集一个目录。
"""
from __future__ import annotations

import json
import os
import sys
import time
import uuid
from typing import Any, Callable

import numpy as np

from . import channelhub as ch
from .paths import dataset_dir

_DEBUG = bool(os.environ.get("SUPERWIRELESS_DEBUG"))


def _dbg(msg: str) -> None:
    """打点只走 stderr —— stdio 传输下 stdout 是 JSON-RPC 通道。"""
    if _DEBUG:
        print(f"[sw.gen] {msg}", file=sys.stderr, flush=True)

# ChannelSample.meta 里逐样本收集的标量物理量。
# 这些是 internal_sim 生成过程中算出来的量，本来会被丢弃。
_SCALAR_META_FIELDS = (
    "pathloss_dB", "distance_3d_m", "is_los", "los_probability",
    "rx_power_serving_dbm", "doppler_hz", "sample_tau_rms_ns",
    "noise_power_dbm", "antenna_gain_serving_db", "tau_rms_ns",
    "rician_k_db", "num_taps", "serving_pci", "ue_id",
    "tx_power_dbm", "ue_tx_power_dbm", "noise_figure_db",
    "tdd_slot_direction", "srs_active_in_slot",
)

# 逐样本收集的顶层标量字段
#
# ``ul_sir_dB`` / ``dl_sir_dB`` 是**测量域**的量（导频上的信干比），和业务域的
# ``sir_dB``（几何 SIR）完全不是一回事——前者决定信道估计准不准，后者决定吞吐。
# 早先只收了业务域那个，于是"SRS 被邻区 UE 打穿"这类场景在数据里完全看不见。
_SCALAR_SAMPLE_FIELDS = (
    "snr_dB", "sinr_dB", "sir_dB", "noise_power_dBm",
    "serving_cell_id", "dl_rank", "slot_duration_s",
    "ul_pre_sinr_dB", "ul_snr_dB", "ul_sinr_dB",
    "ul_sir_dB", "dl_sir_dB", "num_interfering_ues",
)

# 靠钩子采集、ChannelSample 里没有的字段。见 interference.install_geometry_capture。
_HOOKED_SAMPLE_FIELDS = ("ul_sir_geo_dB",)


def _as_float(v: Any) -> float:
    try:
        if v is None:
            return float("nan")
        if isinstance(v, bool):
            return float(v)
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


def estimate_size_mb(cfg: dict[str, Any], num_samples: int) -> float:
    """预估落盘体积，MB。生成前告诉用户，免得跑完才发现几个 G。"""
    rb = int(cfg.get("num_rb") or _rb_from_bandwidth(cfg))
    bs = int(cfg.get("num_bs_tx_ant", 64))
    ue = int(cfg.get("num_ue_rx_ant", 4))
    t = int(cfg.get("num_slots_per_sample", 1) or 1)
    per = t * rb * bs * ue * 8  # complex64 = 8 字节
    return per * num_samples * 2 / 1e6  # 理想 + 估计两份


def _rb_from_bandwidth(cfg: dict[str, Any]) -> int:
    bw = float(cfg.get("bandwidth_hz", 100e6) or 100e6)
    scs = float(cfg.get("subcarrier_spacing", 30000) or 30000)
    return max(1, int(bw / (12 * scs) * 0.95))


# 端口数 → [n_h, n_v, n_p] 面板排布。按 3GPP AAU 的常规做法：
# 双极化、水平优先铺满，再往垂直方向长。
_PANEL_BY_PORTS = {
    2: [1, 1, 2], 4: [2, 1, 2], 8: [2, 2, 2], 16: [4, 2, 2], 32: [8, 2, 2],
    64: [8, 4, 2], 96: [8, 6, 2], 128: [16, 4, 2], 192: [16, 6, 2], 256: [16, 8, 2],
}


def _panel_from_ports(n_ports: int) -> list[int]:
    """把端口数拆成面板排布。

    **这个函数不是可有可无的。** ChannelHub 只有拿到 ``bs_panel`` 才会构造
    DFT 码本，而几何 SINR（`internal_sim.py:2446` 的 `self._sinr_codebook is not
    None and K > 1`）依赖这个码本。码本为 None 时它走兜底分支：
    ``sir_dB = 49.9``（刚好卡在 ±50 dB 契约边界内的哨兵值）、
    ``sinr_dB = snr_db``——**小区间干扰完全不进计算**，报出来的"SINR"其实是
    纯热噪声 SNR。这条路径不报错、不告警，只能靠 `sinr == snr` 反查。

    对不在表里的端口数，退化成单极化水平线阵——能让码本建起来，
    但排布不一定符合用户预期，所以调用方应在摘要里如实说明。
    """
    n = max(int(n_ports), 1)
    if n in _PANEL_BY_PORTS:
        return list(_PANEL_BY_PORTS[n])
    if n % 2 == 0:
        return [n // 2, 1, 2]
    return [n, 1, 1]


def _ensure_bs_panel(cfg: dict[str, Any]) -> tuple[list[int], bool]:
    """确保 cfg 里有 bs_panel，返回 (排布, 是否为推导得来)。"""
    raw = cfg.get("bs_panel")
    if raw:
        p = [int(x) for x in raw]
        cfg["num_bs_tx_ant"] = p[0] * p[1] * p[2]
        cfg["num_bs_rx_ant"] = p[0] * p[1] * p[2]
        return p, False
    panel = _panel_from_ports(int(cfg.get("num_bs_tx_ant", 64) or 64))
    cfg["bs_panel"] = panel
    return panel, True


def _align_to_ues(n: int, num_ues: int) -> int:
    """向上取整到 num_ues 的倍数。

    ChannelHub 要求 num_samples 能被 num_ues 整除（每个 UE 采样轮数相同）。
    这类耦合约束不该让用户操心——多生成几个，再截到用户要的数量。
    """
    if num_ues <= 1:
        return max(n, 1)
    return max(((n + num_ues - 1) // num_ues) * num_ues, num_ues)


def _resolve_workers(workers: int | str, num_samples: int, cfg: dict[str, Any]) -> int:
    """决定用几个进程。

    ``"auto"`` 时按**预估单样本耗时**决定：多小区带干扰的配置一个样本要 0.4~2 秒，
    并行收益巨大；单小区小带宽只要 25 毫秒，起进程的开销（每个子进程要重新
    import numpy/scipy/ChannelHub，约 3 秒）反而不划算。

    判据取小区数与天线数——实测这两个是耗时的主导因素：
    单小区 32T/20MHz 24 ms，21 小区同配置 410 ms（17 倍），
    21 小区 64T/100MHz 2054 ms（87 倍）。
    """
    if isinstance(workers, int) and workers != 0:
        n = workers
    else:
        est_total_s = estimate_seconds(cfg, num_samples)
        # 每个子进程要重新 import numpy/scipy/ChannelHub，实测约 4 秒。
        # 所以只在总工作量够大时才并行，且保证**每个 worker 至少有
        # _MIN_WORK_S 的活**，否则启动成本吃掉收益（实测每 worker 只分到
        # 6 秒活时，10 进程只有 1.34 倍加速）。
        if est_total_s < _PARALLEL_MIN_TOTAL_S:
            return 1
        n = max(2, int(est_total_s // _MIN_WORK_S))
    if n <= 1:
        return 1
    return max(1, min(int(n), num_samples, (os.cpu_count() or 4)))


# 并行的两个门槛：总活少于这个数就不值得起进程；每个 worker 至少要分到这么多活。
_PARALLEL_MIN_TOTAL_S = 30.0
_MIN_WORK_S = 20.0


def estimate_seconds(cfg: dict[str, Any], num_samples: int) -> float:
    """粗估串行生成要多久（秒）。用于自动并行决策与给用户提示。

    实测标定点（本机 20 核）：

    ============================  ===========
    配置                          毫秒/样本
    ============================  ===========
    单小区 32T 20MHz                     24
    21 小区 32T 20MHz                   410
    单小区 64T 100MHz                   191
    21 小区 64T 100MHz                 2054
    ============================  ===========

    主导因素是**小区数**（多小区要算几何 SINR 与干扰，贵 17 倍）与
    **天线数 × RB 数**。射线追踪另算，慢一个量级。
    """
    cells = int(cfg.get("num_sites", 1) or 1) * int(cfg.get("sectors_per_site", 1) or 1)
    ants = int(cfg.get("num_bs_tx_ant", 64) or 64)
    rb = int(cfg.get("num_rb") or _rb_from_bandwidth(cfg))

    # 天线与 RB 的指数取 0.9 / 0.85 而不是 1.0：实测 (64T,273RB) 比
    # (32T,51RB) 只慢 8.0 倍，而线性外推会给 10.7 倍——有固定开销摊薄。
    ms = 24.0 * (ants / 32.0) ** 0.9 * (rb / 51.0) ** 0.85
    if cells > 1:
        # 多小区的代价不是常数倍：小配置下 17 倍，大配置下 11 倍。
        # 取折中的 14 倍，四个实测点都落在 ±30% 内——够做调度决策，不求准。
        ms *= 1.0 + 13.0 * min(1.0, (cells - 1) / 20.0)
    if str(cfg.get("scenario", "")) in ("munich", "custom_osm", "etoile",
                                        "florence", "san_francisco"):
        ms = max(ms, 3000.0)  # 射线追踪慢一个量级
    # 关掉 SSB 测量省约 30%（交错重测：3456 → 2475 ms，基准轮间波动 11.9%）。
    if cells > 1 and not (cfg.get("measurements") or {}).get("ssb_rsrp", True):
        ms *= 0.72
    return ms * num_samples / 1000.0


def _run_parallel(
    source_name: str,
    cfg_run: dict[str, Any],
    *,
    num_samples: int,
    n_workers: int,
    lo: float,
    hi: float,
    filtering: bool,
    base_seed: int,
    n_ues: int,
    ask_factor: int,
    progress: Callable[[int, int], None] | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, Any], int, int, int, list[float]]:
    """把样本切成若干块交给进程池，再合并。

    **分块靠给每块不同的 ``seed``**，不是切样本序号——ChannelHub 的
    ``ue_seed_offset`` 实测对撒点没有影响（同一 offset 与不同 offset 给出
    逐位相同的路损），只有 ``seed`` 真正换掉随机流。

    因此并行结果与串行结果**不是同一批样本**：串行 seed=S 跑 N 个，
    并行是 seed=S..S+W-1 各跑 N/W 个。两者统计上等价、各自可复现，
    但逐样本不同。这一点会写进 summary 的 ``parallel`` 块，别让它成为
    "换了 workers 结果就对不上"的隐形陷阱。
    """
    import tempfile
    from concurrent.futures import ProcessPoolExecutor

    per = [num_samples // n_workers] * n_workers
    for i in range(num_samples % n_workers):
        per[i] += 1
    per = [p for p in per if p > 0]

    tmpdir = tempfile.mkdtemp(prefix="sw_gen_")
    jobs = []
    for k, want in enumerate(per):
        c = dict(cfg_run)
        c["seed"] = base_seed + k
        c["num_samples"] = _align_to_ues(want * ask_factor, n_ues)
        jobs.append((source_name, c, want, lo, hi, filtering,
                     os.path.join(tmpdir, f"chunk{k}.npz")))

    paths: list[str] = []
    first_meta: dict[str, Any] = {}
    acc = att = rej = 0
    observed: list[float] = []
    done = 0
    with ProcessPoolExecutor(max_workers=len(jobs)) as pool:
        for path, fm, st in pool.map(_chunk_worker, jobs):
            if path:
                paths.append(path)
            if fm and not first_meta:
                first_meta = fm
            acc += st["accepted"]
            att += st["attempted"]
            rej += st["rejected"]
            observed.extend(st["observed_sinr"])
            done += 1
            if progress:
                progress(min(acc, num_samples), num_samples)
            _dbg(f"  worker {done}/{len(jobs)} 回来，累计 {acc} 个样本")

    payload = _merge_chunks(paths)
    # 各块可能多给一两个（对齐到 num_ues 的整数倍），统一截到要的数量
    if payload:
        n_have = len(next(iter(payload.values())))
        if n_have > num_samples:
            payload = {k: v[:num_samples] for k, v in payload.items()}
            acc = num_samples
    try:
        import shutil

        shutil.rmtree(tmpdir, ignore_errors=True)
    except OSError:
        pass
    return payload, first_meta, acc, att, rej, observed


def _collect(
    source_name: str,
    cfg_run: dict[str, Any],
    *,
    want: int,
    lo: float,
    hi: float,
    filtering: bool,
    progress: Callable[[int, int], None] | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, Any], dict[str, Any]]:
    """跑一批样本并打包成落盘用的数组字典。

    串行路径与每个并行 worker 用的都是这一个函数——**共用一份实现**，
    否则两条路径迟早会漂移，而漂移只在"并行结果和串行对不上"时才暴露。

    返回 ``(payload, first_meta, stats)``。stats 里带尝试数/拒绝数/观察到的
    信噪比，供上层合并统计与报错。
    """
    from . import interference as intf_mod  # noqa: PLC0415

    # 上行几何 SIR 在 ChannelSample 里没有位置，只能靠钩子从 ChannelHub 内部取。
    # 挂不上就少这一列，不影响其余流程（钩子内部吞异常并回 False）。
    intf_mod.install_geometry_capture()

    h_true: list[np.ndarray] = []
    h_est: list[np.ndarray] = []
    h_intf: list[np.ndarray] = []
    positions: list[np.ndarray] = []
    w_dl: list[np.ndarray] = []
    scalars: dict[str, list[float]] = {
        k: [] for k in (*_SCALAR_SAMPLE_FIELDS, *_HOOKED_SAMPLE_FIELDS)
    }
    metas: dict[str, list[Any]] = {k: [] for k in _SCALAR_META_FIELDS}
    ssb_rsrp: list[list[float]] = []
    ssb_sinr: list[list[float]] = []

    accepted = attempted = rejected = 0
    observed_sinr: list[float] = []
    first_meta: dict[str, Any] = {}
    ask = int(cfg_run.get("num_samples", want))

    for sample in ch.iter_samples(source_name, cfg_run):
        attempted += 1
        sinr = _as_float(getattr(sample, "sinr_dB", None))
        if np.isfinite(sinr):
            observed_sinr.append(sinr)
        if filtering and not (lo <= sinr <= hi):
            rejected += 1
            if attempted >= ask:
                break
            continue

        ht = ch.serving_channel(sample, estimated=False)
        he = ch.serving_channel(sample, estimated=True)
        if ht is None:
            continue

        h_true.append(np.asarray(ht, dtype=np.complex64))
        h_est.append(np.asarray(he if he is not None else ht, dtype=np.complex64))

        hi_arr = getattr(sample, "h_interferers", None)
        if hi_arr is not None:
            h_intf.append(np.asarray(hi_arr, dtype=np.complex64))

        pos = getattr(sample, "ue_position", None)
        positions.append(
            np.asarray(pos, dtype=np.float64) if pos is not None else np.full(3, np.nan)
        )

        w = getattr(sample, "w_dl", None)
        if w is not None:
            w_dl.append(np.asarray(w, dtype=np.complex64))

        for k in _SCALAR_SAMPLE_FIELDS:
            scalars[k].append(_as_float(getattr(sample, k, None)))
        # 必须紧跟着取：钩子里存的是"上一次几何 SINR 计算"的结果，
        # 隔一个样本就串了。take_ 会拿下行量与本样本核对，对不上回 nan。
        scalars["ul_sir_geo_dB"].append(intf_mod.take_ul_geometry_sir(sample))

        meta = sample.meta if isinstance(sample.meta, dict) else {}
        if not first_meta:
            first_meta = {
                k: v for k, v in meta.items()
                if isinstance(v, (str, int, float, bool, type(None)))
            }
        for k in _SCALAR_META_FIELDS:
            metas[k].append(meta.get(k))

        ssb_rsrp.append(list(getattr(sample, "ssb_rsrp_dBm", None) or []))
        ssb_sinr.append(list(getattr(sample, "ssb_sinr_dB", None) or []))

        accepted += 1
        if progress:
            progress(accepted, want)
        if accepted >= want:
            break

    stats = {
        "accepted": accepted, "attempted": attempted, "rejected": rejected,
        "observed_sinr": observed_sinr,
    }
    if accepted == 0:
        return {}, first_meta, stats

    payload: dict[str, np.ndarray] = {
        "h_true": np.stack(h_true),
        "h_est": np.stack(h_est),
        "ue_position": np.stack(positions),
    }
    if h_intf and all(a.shape == h_intf[0].shape for a in h_intf):
        payload["h_interferers"] = np.stack(h_intf)
    if w_dl and all(a.shape == w_dl[0].shape for a in w_dl):
        payload["w_dl"] = np.stack(w_dl)
    for k, vals in scalars.items():
        payload[f"scalar__{k}"] = np.asarray(vals, dtype=np.float64)
    for k, vals in metas.items():
        arr = np.asarray([_as_float(v) for v in vals], dtype=np.float64)
        if np.all(np.isnan(arr)):  # 非数值字段（如 tdd_slot_direction）存字符串
            payload[f"metastr__{k}"] = np.asarray([str(v) for v in vals])
        else:
            payload[f"meta__{k}"] = arr
    if ssb_rsrp and all(len(x) == len(ssb_rsrp[0]) for x in ssb_rsrp) and ssb_rsrp[0]:
        payload["ssb_rsrp_dBm"] = np.asarray(ssb_rsrp, dtype=np.float64)
        payload["ssb_sinr_dB"] = np.asarray(ssb_sinr, dtype=np.float64)
    return payload, first_meta, stats


def _chunk_worker(args: tuple) -> tuple[str, dict[str, Any], dict[str, Any]]:
    """并行 worker：生成一块、落到临时 npz、回句柄。

    **不把数组通过 pickle 传回主进程**——200 个样本的信道有几百 MB，
    走 IPC 既慢又容易撞进程间内存上限。落盘再由主进程合并便宜得多。

    子进程必须自己 ``warmup()``：scipy 的 C 扩展在工作线程/新进程里首次
    加载会撞 import 死锁，这条铁律对子进程同样成立（见 CLAUDE.md）。
    """
    source_name, cfg_run, want, lo, hi, filtering, tmp_path = args

    # **必须在 import numpy 之前把 BLAS 线程数压到 1。**
    # 否则每个 worker 各自开满 CPU 核数的线程：20 个 worker × 20 线程 = 400 个
    # 线程抢 20 个核，上下文切换的开销吃掉全部并行收益。实测不设的话
    # 10 个 worker 只有 1.34 倍加速，设了之后才拿到应有的加速比。
    # 进程级并行 + 单线程 BLAS 是数值计算里的标准组合。
    for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
               "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[_v] = "1"

    from . import channelhub as _ch

    _ch.warmup()
    payload, first_meta, stats = _collect(
        source_name, cfg_run, want=want, lo=lo, hi=hi, filtering=filtering
    )
    if payload:
        np.savez(tmp_path, **payload)
    return (tmp_path if payload else "", first_meta, stats)


def _merge_chunks(paths: list[str]) -> dict[str, np.ndarray]:
    """把各 worker 落的 npz 沿样本轴拼起来。

    只保留**所有块都有**的字段：某块缺 h_interferers 而别块有时，拼出来会
    出现长度不一致的数组，后面读取会错位——宁可丢掉那个字段并在摘要里说明。
    """
    if not paths:
        return {}
    opened = [np.load(p, allow_pickle=False) for p in paths]
    try:
        common = set(opened[0].files)
        for z in opened[1:]:
            common &= set(z.files)
        out: dict[str, np.ndarray] = {}
        for k in sorted(common):
            arrs = [z[k] for z in opened]
            if any(a.shape[1:] != arrs[0].shape[1:] for a in arrs):
                continue  # 形状不一致的字段直接丢，不做危险的补齐
            out[k] = np.concatenate(arrs, axis=0)
        return out
    finally:
        for z in opened:
            z.close()


def generate(
    cfg: dict[str, Any],
    *,
    num_samples: int = 200,
    snr_range_dB: list[float] | None = None,
    plan_markdown: str = "",
    draft_id: str = "",
    prereg_id: str = "",
    progress: Callable[[int, int], None] | None = None,
    max_attempts_factor: int = 5,
    workers: int | str = 1,
    collect_ssb: bool | None = None,
) -> dict[str, Any]:
    """生成数据集并落盘，返回句柄与摘要。

    snr_range_dB 用拒绝采样实现——internal_sim 没有直接设定信噪比的参数，
    信噪比由路损、发射功率和噪声共同决定，只能生成后筛。接受率会如实报告。

    collect_ssb 控制要不要算每小区的 SSB RSRP/SINR。**关掉能省约 30% 时间**
    （交错重测的中位数：3456 → 2475 ms/样本，基准自身的轮间波动 11.9%，
    所以这个差是真的）。代价是 ``Dataset.ssb`` 为空，小区选择、切换、
    波束管理类课题用不了。默认 None = 跟随配置里的 ``measurements.ssb_rsrp``，
    都没给就保留（**不静默减少数据**）。
    """
    cfg = dict(cfg)
    source_name = str(cfg.pop("source", "internal_sim"))
    cfg["num_samples"] = int(num_samples)
    panel, panel_derived = _ensure_bs_panel(cfg)

    # 真实阵列模型：64T 面板自动切到 1 驱 3 / 192 阵子 / 垂直 0.67λ。
    # 不切的话走 ChannelHub 默认的 legacy_64（64 个独立阵元、一律 0.5λ），
    # 那不是本地硬件——实测两者的 h_true 相对差 4.03，完全是另一个信道。
    from . import hardware as hw  # noqa: PLC0415

    hw.apply_array_defaults(cfg)
    array_applied = hw.strip_markers(cfg)
    array_block = hw.array_summary(cfg, array_applied)

    if collect_ssb is not None:
        meas = dict(cfg.get("measurements") or {})
        meas["ssb_rsrp"] = bool(collect_ssb)
        cfg["measurements"] = meas
    ssb_on = bool((cfg.get("measurements") or {}).get("ssb_rsrp", True))

    dataset_id = "ds_" + uuid.uuid4().hex[:8]
    out_dir = dataset_dir(dataset_id)
    out_dir.mkdir(parents=True, exist_ok=True)

    lo, hi = (float(snr_range_dB[0]), float(snr_range_dB[1])) if snr_range_dB else (-np.inf, np.inf)
    filtering = np.isfinite(lo) or np.isfinite(hi)

    accepted = 0
    attempted = 0
    rejected = 0
    observed_sinr: list[float] = []  # 含被拒样本，用于失败时给出可操作的提示
    first_meta: dict[str, Any] = {}
    t0 = time.perf_counter()

    # 拒绝采样时多要一些样本；再对齐到 num_ues 的整数倍（ChannelHub 的约束）
    ask = int(num_samples * max_attempts_factor) if filtering else int(num_samples)
    n_ues = int(cfg.get("num_ues", 1) or 1)
    ask = _align_to_ues(ask, n_ues)
    cfg_run = dict(cfg)
    cfg_run["num_samples"] = ask
    n_workers = _resolve_workers(workers, num_samples, cfg)

    _dbg(f"进入迭代 ask={ask} n_ues={n_ues} workers={n_workers} source={source_name}")

    parallel_fallback: str | None = None
    if n_workers > 1:
        try:
            payload, first_meta, accepted, attempted, rejected, observed_sinr = _run_parallel(
                source_name, cfg_run, num_samples=num_samples, n_workers=n_workers,
                lo=lo, hi=hi, filtering=filtering, base_seed=int(cfg.get("seed", 0) or 0),
                n_ues=n_ues, ask_factor=max_attempts_factor if filtering else 1,
                progress=progress,
            )
        except Exception as exc:  # noqa: BLE001
            # 多进程在某些宿主里起不来：Windows spawn 需要可导入的 __main__
            # （REPL、-c、部分 notebook 里没有），也可能撞上内存或权限限制。
            # **降级到串行而不是让整次生成失败**，但必须如实报出来，
            # 否则用户会以为并行生效了、还纳闷为什么没变快。
            parallel_fallback = f"{type(exc).__name__}: {exc}"
            _dbg(f"并行失败，降级串行：{parallel_fallback}")
            n_workers = 1

    if n_workers <= 1:
        payload, first_meta, st = _collect(
            source_name, cfg_run, want=num_samples, lo=lo, hi=hi,
            filtering=filtering, progress=progress,
        )
        accepted = st["accepted"]
        attempted = st["attempted"]
        rejected = st["rejected"]
        observed_sinr = st["observed_sinr"]

    elapsed = time.perf_counter() - t0
    if accepted == 0:
        if filtering and observed_sinr:
            obs = np.asarray(observed_sinr)
            raise RuntimeError(
                f"信噪比筛选区间 [{lo:g}, {hi:g}] dB 与该场景的实际分布不重叠：\n"
                f"  尝试了 {attempted} 个样本，实际信噪比落在 "
                f"[{obs.min():.1f}, {obs.max():.1f}] dB（中位数 {np.median(obs):.1f} dB）。\n"
                f"信噪比由路损、发射功率和噪声共同决定，不能直接设定。可以：\n"
                f"  · 去掉筛选（snr_range_dB=null），先看自然分布\n"
                f"  · 把区间改到 [{obs.min():.0f}, {obs.max():.0f}] dB 之内\n"
                f"  · 想整体压低信噪比，调小 tx_power_dbm 或调大 isd_m"
            )
        raise RuntimeError(
            "没有生成出任何样本。"
            + (f"信噪比区间 [{lo:g}, {hi:g}] dB 全部被拒。" if filtering else "请检查配置。")
        )

    _dbg(f"迭代结束 accepted={accepted}，开始写盘")
    np.savez_compressed(out_dir / "channels.npz", **payload)
    _dbg("写盘完成")

    shape = payload["h_true"].shape
    sinr_arr = payload["scalar__sinr_dB"]
    finite = sinr_arr[np.isfinite(sinr_arr)]

    # 六边形栅格会把站数吸附到环数（0→1 站、1→7 站、2→19 站），
    # 所以"配了 6 站"可能实际跑的是 7 站。这里对比配置值与实际值，
    # 不一致时在 summary 里显式记下——否则用户拿着错误的小区数下结论。
    cells_cfg = int(cfg.get("num_sites", 1) or 1) * int(cfg.get("sectors_per_site", 1) or 1)
    cells_real = first_meta.get("num_cells")
    topology_note = None
    if cells_real and int(cells_real) != cells_cfg:
        topology_note = (
            f"配置为 {cfg.get('num_sites')} 站 × {cfg.get('sectors_per_site')} 扇区 "
            f"= {cells_cfg} 小区，实际生成 {cells_real} 小区。"
            f"六边形栅格的站数只能是 1/7/19（按环数展开），会向上吸附。"
            f"需要精确站数请用 topology_layout='linear' 或 custom_site_positions。"
        )

    # 干扰是否真的进了 SINR。判据：sir_dB 恰为兜底哨兵 49.9，或 sinr 与 snr 逐点相同。
    sir_arr = payload.get("scalar__sir_dB")
    snr_arr = payload.get("scalar__snr_dB")
    sinr_is_snr = (
        snr_arr is not None and snr_arr.size and np.allclose(sinr_arr, snr_arr, atol=1e-6)
    )
    sir_sentinel = bool(sir_arr is not None and sir_arr.size and np.allclose(sir_arr, 49.9))
    interference_modeled = bool(cells_cfg > 1 and not sinr_is_snr and not sir_sentinel)
    interference_note = None
    if cells_cfg > 1 and not interference_modeled:
        interference_note = (
            "多小区配置但小区间干扰未进入 SINR —— 报出的 sinr_dB 等于纯热噪声 snr_dB。"
            "干扰相关的结论不成立。"
        )

    # IoT（噪声抬升）。由几何 SIR 与 SINR 精确推出——**不是** snr - sinr，
    # 那两个字段口径不同，相减差几十 dB（见 interference.py 模块文档）。
    iot_block: dict[str, Any] | None = None
    if interference_modeled and sir_arr is not None and sir_arr.size:
        from . import interference as _intf  # noqa: PLC0415

        st = _intf.iot_stats(sinr_arr, sir_arr)
        iot_block = {"dl": st.as_dict()}
        ul_geo = payload.get("scalar__ul_sir_geo_dB")
        ul_sinr = payload.get("scalar__ul_sinr_dB")
        if ul_geo is not None and ul_sinr is not None and np.isfinite(ul_geo).any():
            iot_block["ul"] = _intf.iot_stats(ul_sinr, ul_geo).as_dict()

    # 预注册口径随数据一起存档。**必须在生成时绑定，事后补绑没有意义**——
    # 预注册的全部价值就在于"看数据之前写下的"，事后写的只是记录。
    prereg_block = None
    if prereg_id:
        from . import analysis as an

        try:
            pr = an.load(prereg_id)
            prereg_block = {
                "prereg_id": pr.prereg_id,
                "digest": pr.digest,
                "primary_metric": pr.primary_metric,
                "metric_unit": pr.metric_unit,
                "baseline": pr.baseline,
                "csi_basis": pr.csi_basis,
                "expected_effect": pr.expected_effect,
                "higher_is_better": pr.higher_is_better,
                "secondary_metrics": pr.secondary_metrics,
                "locked_before_generation": True,
            }
        except FileNotFoundError:
            prereg_block = {"prereg_id": prereg_id, "error": "找不到该预注册，未绑定"}

    summary = {
        "dataset_id": dataset_id,
        "draft_id": draft_id,
        "prereg": prereg_block,
        "source": source_name,
        "num_samples": int(accepted),
        "requested": int(num_samples),
        "cells_configured": cells_cfg,
        "cells_actual": int(cells_real) if cells_real else None,
        "topology_note": topology_note,
        "bs_panel": list(panel),
        "bs_panel_derived": bool(panel_derived),
        "antenna_model": array_block,
        # 并行会换掉随机流的分块方式，逐样本结果与串行不同（统计等价、各自可复现）。
        # 记进摘要，免得"换了 workers 结果对不上"变成隐形陷阱。
        "parallel": {
            "workers": int(n_workers),
            "seed_layout": (
                f"seed={int(cfg.get('seed', 0) or 0)}"
                if n_workers <= 1
                else f"seed={int(cfg.get('seed', 0) or 0)}..{int(cfg.get('seed', 0) or 0) + n_workers - 1}"
            ),
            "note": (
                None if n_workers <= 1
                else "多进程分块：每块用不同 seed。与 workers=1 的结果统计等价但逐样本不同"
            ),
            "fallback_reason": parallel_fallback,
        },
        "interference_modeled": interference_modeled if cells_cfg > 1 else None,
        "interference_note": interference_note,
        "iot": iot_block,
        "collect_ssb": ssb_on,
        "shape": {
            "N": int(shape[0]), "T": int(shape[1]), "RB": int(shape[2]),
            "BS_ant": int(shape[3]), "UE_ant": int(shape[4]),
        },
        "elapsed_s": round(elapsed, 2),
        "seconds_per_sample": round(elapsed / max(accepted, 1), 3),
        "size_mb": round((out_dir / "channels.npz").stat().st_size / 1e6, 1),
        "snr_filter": {
            "enabled": bool(filtering),
            "range_dB": [lo, hi] if filtering else None,
            "attempted": attempted,
            "rejected": rejected,
            "accept_rate": round(accepted / max(attempted, 1), 3),
        },
        "sinr_dB": _distribution(finite),
        "channel_model": first_meta.get("channel_model"),
        "scenario": first_meta.get("scenario"),
        "is_cdl": str(first_meta.get("channel_model", "")).upper().startswith("CDL"),
        "tau_rms_ns": first_meta.get("tau_rms_ns"),
        "config": cfg,
        "sample_meta": first_meta,
        "created_at": time.time(),
        "path": str(out_dir),
    }

    for key, label in (("meta__pathloss_dB", "pathloss_dB"),
                       ("meta__distance_3d_m", "distance_3d_m"),
                       ("meta__doppler_hz", "doppler_hz")):
        if key in payload:
            v = payload[key][np.isfinite(payload[key])]
            if v.size:
                summary[label] = _distribution(v)
    if "meta__is_los" in payload:
        los = payload["meta__is_los"]
        los = los[np.isfinite(los)]
        if los.size:
            summary["los_ratio"] = round(float(los.mean()), 3)

    # 仿真说明书：配置敲定之后把"这次到底在仿什么"画出来。
    # 用真实撒点画拓扑图，所以放在生成之后而不是之前。
    # **失败不影响数据集**——说明书是解释性产物，不是数据的一部分。
    try:
        from . import spec as _spec  # noqa: PLC0415

        _pos = payload.get("ue_position")
        _ue_xy = (
            [(float(r[0]), float(r[1])) for r in _pos[:400] if np.isfinite(r[0])]
            if _pos is not None and _pos.ndim == 2 and _pos.shape[1] >= 2 else None
        )
        summary["spec_sheet"] = _spec.write_spec(
            dict(cfg, source=source_name),
            num_samples=int(accepted),
            dataset_id=dataset_id,
            title=f"仿真说明书 · {dataset_id}",
            ue_xy=_ue_xy,
        )
    except Exception as exc:  # noqa: BLE001
        summary["spec_sheet"] = {"error": f"{type(exc).__name__}: {exc}"}

    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    if plan_markdown:
        (out_dir / "plan.md").write_text(plan_markdown, encoding="utf-8")

    return summary


def _distribution(v: np.ndarray) -> dict[str, float]:
    if v.size == 0:
        return {}
    q = np.percentile(v, [5, 50, 95])
    return {
        "min": round(float(v.min()), 2),
        "p5": round(float(q[0]), 2),
        "median": round(float(q[1]), 2),
        "p95": round(float(q[2]), 2),
        "max": round(float(v.max()), 2),
        "mean": round(float(v.mean()), 2),
    }


def load_summary(dataset_id: str) -> dict[str, Any]:
    p = dataset_dir(dataset_id) / "summary.json"
    if not p.is_file():
        raise KeyError(f"找不到数据集 {dataset_id!r}")
    return json.loads(p.read_text(encoding="utf-8"))


def list_datasets() -> list[dict[str, Any]]:
    from .paths import datasets_dir

    out = []
    for d in sorted(datasets_dir().glob("ds_*"), key=lambda p: p.name):
        f = d / "summary.json"
        if f.is_file():
            try:
                s = json.loads(f.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                continue
            out.append(
                {
                    "dataset_id": s.get("dataset_id"),
                    "num_samples": s.get("num_samples"),
                    "shape": s.get("shape"),
                    "channel_model": s.get("channel_model"),
                    "scenario": s.get("scenario"),
                    "size_mb": s.get("size_mb"),
                    "created_at": s.get("created_at"),
                }
            )
    return out
