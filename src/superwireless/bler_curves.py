"""Table-driven BLER curves and integrity checks.

The first profile is the user-provided ``company_20b_256qam`` data set. The source
script labels its horizontal axis ``Es/No``; the data owner confirmed that it denotes
SINR for a classic MMSE receiver. This module preserves both the original label and
the confirmed physical meaning.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import numpy as np

from . import bler_data_20b as data

TX_MODES = ("newtx", "retx")


@dataclass(frozen=True)
class DemodCurve:
    """One tabulated demodulation curve for an MCS and transmission mode."""

    source_id: str
    mcs: int
    q_m: int
    tx_mode: str
    code_rate: float
    start_db: float
    step_db: float
    bler_points: tuple[float, ...]

    @property
    def modulation(self) -> str:
        return {2: "QPSK", 4: "16QAM", 6: "64QAM", 8: "256QAM"}[self.q_m]

    @property
    def sinr_db(self) -> np.ndarray:
        return self.start_db + self.step_db * np.arange(len(self.bler_points), dtype=float)

    @property
    def end_db(self) -> float:
        return self.start_db + self.step_db * (len(self.bler_points) - 1)

    def evaluate(self, sinr_db: Any) -> np.ndarray:
        """Evaluate BLER with log-domain interpolation and conservative clamping.

        Values below the measured grid clamp to BLER=1.  Values above the grid clamp
        to the last measured BLER; the function never invents an extrapolated tail.
        """
        x = np.atleast_1d(np.asarray(sinr_db, dtype=float))
        if not np.all(np.isfinite(x)):
            raise ValueError("sinr_db must contain only finite values")
        xp = self.sinr_db
        yp = np.asarray(self.bler_points, dtype=float)
        log_y = np.interp(x, xp, np.log10(yp), left=0.0, right=float(np.log10(yp[-1])))
        return np.clip(10.0 ** log_y, 0.0, 1.0)

    def required_sinr_db(self, target_bler: float = 0.1) -> float:
        """Interpolate the SINR where this curve reaches ``target_bler``."""
        target = float(target_bler)
        y = np.asarray(self.bler_points, dtype=float)
        if not (0.0 < target <= 1.0):
            raise ValueError(f"target_bler must be in (0, 1], got {target_bler}")
        if target > y[0] or target < y[-1]:
            raise ValueError(
                f"target BLER {target:g} is outside observed range [{y[-1]:g}, {y[0]:g}]"
            )
        x = self.sinr_db
        for i in range(len(y) - 1):
            if y[i] >= target >= y[i + 1]:
                if y[i] == y[i + 1]:
                    return float(x[i])
                frac = ((math.log10(target) - math.log10(y[i])) /
                        (math.log10(y[i + 1]) - math.log10(y[i])))
                return float(x[i] + frac * (x[i + 1] - x[i]))
        return float(x[-1])

    def as_dict(self, *, include_points: bool = True) -> dict[str, Any]:
        out: dict[str, Any] = {
            "source_id": self.source_id,
            "mcs": self.mcs,
            "tx_mode": self.tx_mode,
            "modulation": self.modulation,
            "q_m": self.q_m,
            "code_rate": self.code_rate,
            "axis_source_name": data.SOURCE_AXIS_NAME,
            "axis_original_label": data.SOURCE_AXIS_ORIGINAL_LABEL,
            "axis_interpretation": data.SOURCE_AXIS_USAGE,
            "receiver_model": data.RECEIVER_MODEL,
            "profile_scope": data.PROFILE_SCOPE,
            "start_db": self.start_db,
            "end_db": self.end_db,
            "step_db": self.step_db,
            "n_points": len(self.bler_points),
            "interpolation": "linear in log10(BLER); clamp outside observed range",
        }
        if include_points:
            out["sinr_db"] = [round(float(v), 10) for v in self.sinr_db]
            out["bler"] = list(self.bler_points)
        return out


@lru_cache(maxsize=64)
def get_curve(mcs: int, tx_mode: str = "newtx") -> DemodCurve:
    """Return a source curve by MCS and ``newtx``/``retx`` mode."""
    idx = int(mcs)
    mode = str(tx_mode).lower()
    if idx < 0 or idx >= len(data.MCS_CURVE_ROWS):
        raise ValueError(f"MCS must be 0..{len(data.MCS_CURVE_ROWS) - 1}, got {mcs}")
    if mode not in TX_MODES:
        raise ValueError(f"tx_mode must be one of {TX_MODES}, got {tx_mode!r}")
    row = data.MCS_CURVE_ROWS[idx]
    row_mcs, q_m = int(row[0]), int(row[1])
    rate, start, step, points = row[2 if mode == "newtx" else 3]
    return DemodCurve(
        source_id=data.SOURCE_ID,
        mcs=row_mcs,
        q_m=q_m,
        tx_mode=mode,
        code_rate=float(rate),
        start_db=float(start),
        step_db=float(step),
        bler_points=tuple(float(v) for v in points),
    )


def mcs_profile_rows() -> list[dict[str, Any]]:
    """Return the 28-entry MCS profile used by the tabulated curves."""
    rows = []
    for idx in range(len(data.MCS_CURVE_ROWS)):
        new = get_curve(idx, "newtx")
        retx = get_curve(idx, "retx")
        rows.append({
            "index": idx,
            "q_m": new.q_m,
            "modulation": new.modulation,
            "newtx_code_rate": new.code_rate,
            "retx_code_rate": retx.code_rate,
            "se": round(new.q_m * new.code_rate, 4),
        })
    return rows


def verify_curves(target_bler: float = 0.1) -> dict[str, Any]:
    """Check source hash, coverage, axes, BLER monotonicity, and target crossings."""
    issues: list[str] = []
    raw_hash = hashlib.sha256(
        json.dumps(data.MCS_CURVE_ROWS, separators=(",", ":")).encode()
    ).hexdigest()
    if raw_hash != data.DATA_SHA256:
        issues.append(f"data hash mismatch: {raw_hash} != {data.DATA_SHA256}")
    if len(data.MCS_CURVE_ROWS) != 28:
        issues.append(f"expected 28 MCS rows, got {len(data.MCS_CURVE_ROWS)}")

    n_points = 0
    new_thresholds: dict[int, list[float]] = {}
    for idx, row in enumerate(data.MCS_CURVE_ROWS):
        if int(row[0]) != idx:
            issues.append(f"row {idx} carries MCS {row[0]}")
        for mode in TX_MODES:
            curve = get_curve(idx, mode)
            y = np.asarray(curve.bler_points, dtype=float)
            n_points += len(y)
            if curve.q_m not in (2, 4, 6, 8):
                issues.append(f"MCS {idx} {mode}: unsupported q_m={curve.q_m}")
            if not (0.0 < curve.code_rate <= 1.0):
                issues.append(f"MCS {idx} {mode}: invalid code rate {curve.code_rate}")
            if curve.step_db <= 0 or len(y) < 2:
                issues.append(f"MCS {idx} {mode}: invalid grid")
            if not np.all(np.isfinite(y)) or np.any((y <= 0.0) | (y > 1.0)):
                issues.append(f"MCS {idx} {mode}: BLER outside (0,1]")
            if np.any(np.diff(y) > 1e-12):
                issues.append(f"MCS {idx} {mode}: BLER is not non-increasing")
            if not (y[-1] <= target_bler <= y[0]):
                issues.append(f"MCS {idx} {mode}: does not cross BLER={target_bler}")
            elif mode == "newtx":
                new_thresholds.setdefault(curve.q_m, []).append(
                    curve.required_sinr_db(target_bler)
                )

    for q_m, thresholds in new_thresholds.items():
        if np.any(np.diff(thresholds) < -1e-9):
            issues.append(f"Qm={q_m}: NewTx thresholds are not monotonic by MCS")

    return {
        "consistent": not issues,
        "source_id": data.SOURCE_ID,
        "n_mcs": len(data.MCS_CURVE_ROWS),
        "n_curves": len(data.MCS_CURVE_ROWS) * len(TX_MODES),
        "n_points": n_points,
        "data_sha256": raw_hash,
        "hash_matches": raw_hash == data.DATA_SHA256,
        "issues": issues,
        "caveat": (
            "User-provided demodulation curves, not a 3GPP reference table. "
            "The source label Es/No denotes post-MMSE SINR. Other link dimensions are "
            "intentionally not parameterized."
        ),
    }
