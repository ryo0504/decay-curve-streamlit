from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, Optional

import numpy as np
import pandas as pd


def _r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    resid = y_true - y_pred
    rss = float(np.sum(resid ** 2))
    tss = float(np.sum((y_true - np.mean(y_true)) ** 2))
    return float(1.0 - rss / tss) if tss != 0 else float("nan")


@dataclass
class LinearFitResult:
    slope: float
    intercept: float
    r2: float


@dataclass
class Step6Result:
    temperature: Optional[float]
    fit_k1: LinearFitResult
    fit_k2: LinearFitResult
    k_t1: float
    k_t2: float
    k_p1: float
    k_p2: float
    df_merged: pd.DataFrame  # columns: intensity,k_1,k_2 (+ meta)


def infer_temperature_from_fit_params(df_fit: pd.DataFrame) -> Optional[float]:
    """
    Step4の fit_params_concat は temp列を持つ想定。
    """
    if "temp" in df_fit.columns:
        try:
            vals = pd.to_numeric(df_fit["temp"], errors="coerce").dropna()
            if len(vals) > 0 and vals.nunique() == 1:
                return float(vals.iloc[0])
        except Exception:
            pass
    return None


def load_step4_step5(
    fit_params_concat_csv: Path,
    vis_integrated_csv: Path,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df_fit = pd.read_csv(fit_params_concat_csv)
    df_vis = pd.read_csv(vis_integrated_csv)

    # Step5は index=True 保存なので、先頭列が file のはず
    if "file" not in df_vis.columns:
        # 先頭列を file として扱う
        df_vis = df_vis.rename(columns={df_vis.columns[0]: "file"})
    if "light_intensity" not in df_vis.columns:
        # 旧名対応
        if "light intensity" in df_vis.columns:
            df_vis = df_vis.rename(columns={"light intensity": "light_intensity"})

    return df_fit, df_vis


def merge_by_order(
    df_fit: pd.DataFrame,
    df_vis: pd.DataFrame,
    intensity_col: str = "light_intensity",
) -> pd.DataFrame:
    """
    Jupyterと同じ「順番で合わせる」方式。
    df_fitの行順に df_vis[intensity] を付与する。
    """
    intens = pd.to_numeric(df_vis[intensity_col], errors="coerce").to_numpy()
    if len(intens) < len(df_fit):
        raise ValueError(f"vis_integrated rows ({len(intens)}) < fit_params rows ({len(df_fit)})")

    df = df_fit.copy()
    df["light_intensity"] = intens[: len(df_fit)]
    if "filter_combination" in df.columns:
        df["filter_combination"] = (
            df["filter_combination"]
            .astype(str)
            .str.replace(r"\.0$", "", regex=True)  # "10.0" -> "10"
            .str.zfill(3)                          # "10" -> "010"
        )
    return df


def linear_fit(x: np.ndarray, y: np.ndarray) -> LinearFitResult:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if len(x) < 2:
        return LinearFitResult(slope=float("nan"), intercept=float("nan"), r2=float("nan"))

    # y = a x + b
    a, b = np.polyfit(x, y, 1)
    yhat = a * x + b
    return LinearFitResult(slope=float(a), intercept=float(b), r2=_r2(y, yhat))


def compute_k_parameters(
    df_merged: pd.DataFrame,
    intensity_eval: float = 1e8,
) -> Step6Result:
    """
    df_merged must have columns: light_intensity, k_1, k_2
    """
    if "k_1" not in df_merged.columns or "k_2" not in df_merged.columns:
        raise ValueError("df_merged must contain 'k_1' and 'k_2' columns (from Step4).")

    x = pd.to_numeric(df_merged["light_intensity"], errors="coerce").to_numpy()
    y1 = pd.to_numeric(df_merged["k_1"], errors="coerce").to_numpy()
    y2 = pd.to_numeric(df_merged["k_2"], errors="coerce").to_numpy()

    fit1 = linear_fit(x, y1)  # k1(I) = a1 I + b1
    fit2 = linear_fit(x, y2)  # k2(I) = a2 I + b2

    # 元コード踏襲
    k_t1 = fit1.intercept
    k_p1 = fit1.slope * intensity_eval + fit1.intercept

    k_t2 = fit2.intercept
    # func_kp2: -a*x + b
    k_p2 = (-fit2.slope) * intensity_eval + fit2.intercept

    temp = infer_temperature_from_fit_params(df_merged)

    return Step6Result(
        temperature=temp,
        fit_k1=fit1,
        fit_k2=fit2,
        k_t1=float(k_t1),
        k_t2=float(k_t2),
        k_p1=float(k_p1),
        k_p2=float(k_p2),
        df_merged=df_merged,
    )


def save_parameters_csv(
    result: Step6Result,
    out_csv: Path,
) -> None:
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    idx = result.temperature if result.temperature is not None else "unknown"

    df = pd.DataFrame(
        [[
            # derived parameters
            result.k_t1, result.k_t2, result.k_p1, result.k_p2,
            # k1 linear fit
            result.fit_k1.slope, result.fit_k1.intercept, result.fit_k1.r2,
            # k2 linear fit
            result.fit_k2.slope, result.fit_k2.intercept, result.fit_k2.r2,
        ]],
        index=[idx],
        columns=[
            "k_t1", "k_t2", "k_p1", "k_p2",
            "k1_slope", "k1_intercept", "k1_R2",
            "k2_slope", "k2_intercept", "k2_R2",
        ],
    )
    df.to_csv(out_csv, index=True)



def save_merged_csv(df_merged: pd.DataFrame, out_csv: Path) -> None:
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df_merged.to_csv(out_csv, index=False)
