from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit


def kinetic_model_f1(x: np.ndarray, B: float, C: float, D: float) -> np.ndarray:
    A = 1.0
    return A * (1.0 - ((1.0 - np.exp(D * x)) / (1.0 - C * np.exp(D * x)))) + B


@dataclass
class FitResult:
    popt: Tuple[float, float, float]      # (B, C, D)
    perr: Tuple[float, float, float]      # std errors
    r2: float
    t95: float
    df_fit: pd.DataFrame                 # combined (seg_id, norm_time, norm_abs, fit_curve)


def _r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    residuals = y_true - y_pred
    rss = float(np.sum(residuals ** 2))
    tss = float(np.sum((y_true - np.mean(y_true)) ** 2))
    return float(1.0 - rss / tss) if tss != 0 else float("nan")


def _t95_from_params(B: float, C: float, D: float) -> float:
    k1 = -D
    k2 = -(C * D) / (1.0 - C)

    denom = (20.0 * k1 + k2)
    numer = (k1 + k2)
    if k1 == 0 or denom == 0 or numer <= 0 or denom <= 0:
        return float("nan")

    return float((-1.0 / k1) * (math.log(numer / denom)))


_seg_pat = re.compile(r"^(time|abs)_(\d+)$")


def extract_segments_from_step2_csv(df: pd.DataFrame) -> Dict[int, pd.DataFrame]:
    """
    Step2出力形式:
      time_01, abs_01, time_02, abs_02, ...
    を検出し、{seg_id: DataFrame(time, abs)} を返す。
    """
    cols = list(df.columns)
    found: Dict[int, Dict[str, str]] = {}  # seg_id -> {"time": colname, "abs": colname}

    for c in cols:
        m = _seg_pat.match(str(c))
        if not m:
            continue
        kind, sid_s = m.group(1), m.group(2)
        sid = int(sid_s)
        found.setdefault(sid, {})
        found[sid][kind] = c

    segments: Dict[int, pd.DataFrame] = {}
    for sid, mapping in sorted(found.items()):
        if "time" in mapping and "abs" in mapping:
            seg = df[[mapping["time"], mapping["abs"]]].copy()
            seg.columns = ["time", "abs"]
            # NaN行を落とす（セグメント長が違うので末尾にNaNが出る想定）
            seg = seg.dropna().reset_index(drop=True)
            if len(seg) > 0:
                segments[sid] = seg

    if not segments:
        raise ValueError("No segments found. Expected columns like time_01, abs_01 ...")

    return segments


def normalize_one_segment(seg: pd.DataFrame, plot_number: int) -> pd.DataFrame:
    """
    1セグメントの time/abs を規格化して返す（norm_time, norm_abs）。
    """
    seg = seg[["time", "abs"]].copy().reset_index(drop=True)

    # x: start at 0
    x = seg["time"].to_numpy(dtype=float) - float(seg.loc[0, "time"])

    # Amin: mean of last plot_number points
    tail = seg["abs"].tail(int(plot_number))
    abs_min = float(tail.mean())

    abs_arr = seg["abs"].to_numpy(dtype=float)
    denom = float(np.max(abs_arr) - abs_min)
    if denom == 0:
        raise ValueError("Normalization denominator is zero in a segment. Check data/plot_number.")

    y = (abs_arr - abs_min) / denom

    out = pd.DataFrame({"norm_time": x, "norm_abs": y})
    return out


def build_combined_dataset_from_segments(
    df_step2: pd.DataFrame,
    plot_number: int = 20,
) -> pd.DataFrame:
    """
    Step2 CSVからセグメントを抽出し、それぞれ規格化して縦結合する。
    出力: seg_id, norm_time, norm_abs
    """
    segments = extract_segments_from_step2_csv(df_step2)

    rows: List[pd.DataFrame] = []
    for sid, seg in segments.items():
        norm = normalize_one_segment(seg, plot_number=plot_number)
        norm.insert(0, "seg_id", sid)
        rows.append(norm)

    combined = pd.concat(rows, axis=0, ignore_index=True)

    # フィットの安定性のためtime順に並べる（seg_idは保持）
    combined = combined.sort_values(by="norm_time").reset_index(drop=True)
    return combined


def fit_f1_on_combined(
    combined: pd.DataFrame,
    p0: Tuple[float, float, float] = (0.0, 0.5, -0.005),
) -> FitResult:
    """
    combined: columns ['seg_id','norm_time','norm_abs']
    """
    x = combined["norm_time"].to_numpy(dtype=float)
    y = combined["norm_abs"].to_numpy(dtype=float)

    popt, pcov = curve_fit(kinetic_model_f1, x, y, p0=p0, maxfev=20000)
    perr = np.sqrt(np.diag(pcov))
    y_fit = kinetic_model_f1(x, *popt)

    r2 = _r2_score(y, y_fit)
    t95 = _t95_from_params(float(popt[0]), float(popt[1]), float(popt[2]))

    df_fit = combined.copy()
    df_fit["fit_curve"] = y_fit

    return FitResult(
        popt=(float(popt[0]), float(popt[1]), float(popt[2])),
        perr=(float(perr[0]), float(perr[1]), float(perr[2])),
        r2=float(r2),
        t95=float(t95),
        df_fit=df_fit,
    )


def save_fit_csv(df_fit: pd.DataFrame, out_csv: Path) -> None:
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df_fit.to_csv(out_csv, index=False)

def calc_rate_constants(B: float, C: float, D: float) -> tuple[float, float]:
    """
    Calculate k1, k2 from fitted parameters.
    """
    k1 = -D
    k2 = -(C * D) / (1.0 - C) if (1.0 - C) != 0 else float("nan")
    return float(k1), float(k2)

def build_param_df(
    popt: tuple[float, float, float],
    column_name: str,
) -> pd.DataFrame:
    B, C, D = popt
    k1, k2 = calc_rate_constants(B, C, D)

    data = [B, C, D, k1, k2]
    index = ["opt_B", "opt_C", "opt_D", "k_1", "k_2"]

    return pd.DataFrame(data, index=index, columns=[column_name])
