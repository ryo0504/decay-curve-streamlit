from __future__ import annotations

import codecs
import glob
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from numpy import trapezoid


@dataclass
class IntegrateConfig:
    wavelength_nm_min: float = 400.0
    wavelength_nm_max: float = 550.0
    file_glob: str = "data/**/vis/*.txt"   # default例（UIで上書き想定）


def _nm_to_wavenumber_cm1(wavelength_nm: np.ndarray) -> np.ndarray:
    """
    wavelength [nm] -> wavenumber [cm^-1]
    1 nm = 1e-7 cm
    wavenumber = 1 / (wavelength_cm)
    """
    wl_cm = wavelength_nm * 1e-7
    return 1.0 / wl_cm


def _read_vis_txt(path: str) -> pd.DataFrame:
    """
    元コードを踏襲: 5列のテーブルとして読んで、c00,c01を使用する。
    """
    with codecs.open(path, "r", "UTF-8", "ignore") as f:
        col_names = [f"c{i:02d}" for i in range(5)]
        df = pd.read_table(f, names=col_names)

    df = df.rename(columns={"c00": "wavelength", "c01": "photon_count"})
    df = df.drop(columns=["c02", "c03", "c04"], errors="ignore")

    # 型を揃える（文字混入対策）
    df["wavelength"] = pd.to_numeric(df["wavelength"], errors="coerce")
    df["photon_count"] = pd.to_numeric(df["photon_count"], errors="coerce")
    df = df.dropna().reset_index(drop=True)
    return df


def integrate_one_file(
    path: str,
    wavelength_nm_min: float,
    wavelength_nm_max: float,
) -> float:
    """
    指定波長範囲を wavenumber に変換し、photon_count を trapz で積分。
    元コードのロジック（wavenumber範囲で抽出）を踏襲。
    """
    df = _read_vis_txt(path)
    df["wavenumber"] = _nm_to_wavenumber_cm1(df["wavelength"].to_numpy())

    # 積分範囲（nm → cm^-1）へ
    x1 = _nm_to_wavenumber_cm1(np.array([wavelength_nm_min]))[0]
    x2 = _nm_to_wavenumber_cm1(np.array([wavelength_nm_max]))[0]

    # 元コードと同じ条件: x2 < wavenumber < x1 （nm_minが短波長ならx1が大きい）
    lo = min(x1, x2)
    hi = max(x1, x2)
    df_target = df[(df["wavenumber"] > lo) & (df["wavenumber"] < hi)]

    if len(df_target) < 2:
        return float("nan")

    x = df_target["wavenumber"].to_numpy(dtype=float)
    y = df_target["photon_count"].to_numpy(dtype=float)

    # trapzはxの順序に依存するので昇順にしておく
    order = np.argsort(x)
    x = x[order]
    y = y[order]

    val = float(trapezoid(y, x))
    return abs(val)  # 元コードの apply_abs


def integrate_vis_folder(
    file_glob: str,
    wavelength_nm_min: float = 400.0,
    wavelength_nm_max: float = 550.0,
) -> pd.DataFrame:
    """
    file_glob に合う全ファイルを処理し、
    index=file_stem, column=['light_intensity'] のDFを返す。
    """
    paths = sorted(glob.glob(file_glob))
    if not paths:
        raise FileNotFoundError(f"No files found for glob: {file_glob}")

    result: Dict[str, float] = {}
    for p in paths:
        file_name = Path(p).stem
        result[file_name] = integrate_one_file(
            p, wavelength_nm_min=wavelength_nm_min, wavelength_nm_max=wavelength_nm_max
        )

    df_out = pd.DataFrame.from_dict(result, orient="index", columns=["light_intensity"])
    df_out = df_out.sort_values("light_intensity", ascending=False)
    df_out.index.name = "file"
    return df_out


def save_integral_csv(df_out: pd.DataFrame, out_csv: Path) -> None:
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(out_csv, index=True)
