from __future__ import annotations

import re
from pathlib import Path
from typing import Optional, Tuple, List

import pandas as pd


FILTERS = {"001", "010", "011", "100", "101", "110"}


def _parse_column_name(name: str) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[str]]:
    """
    column_name 例:
      "30 490 010"
      "30 25 490 010"
    戻り: (temp, period_3, wavelength, filter_combination)
    """
    parts = str(name).strip().split()
    # 末尾はfilter
    if len(parts) < 3:
        return (None, None, None, None)

    filt = parts[-1]
    if filt not in FILTERS:
        # 一応ゆるく：末尾が3桁なら許容
        if not re.fullmatch(r"\d{3}", filt):
            return (None, None, None, None)

    # 先頭から数値を解釈
    try:
        if len(parts) == 3:
            temp = float(parts[0])
            wavelength = float(parts[1])
            period_3 = None
        else:
            temp = float(parts[0])
            period_3 = float(parts[1])
            wavelength = float(parts[2])
    except Exception:
        return (None, None, None, None)

    return (temp, period_3, wavelength, filt)


def load_and_concat_fit_params(fit_params_dir: Path) -> pd.DataFrame:
    """
    fit_params_dir配下の *.csv を読み、1行=1条件に整形して返す。
    返り値: columns = [temp, period_3, wavelength, filter_combination, opt_B, opt_C, opt_D, k_1, k_2, source_file, column_name]
    """
    fit_params_dir = Path(fit_params_dir)
    paths: List[Path] = sorted(fit_params_dir.glob("*.csv"))
    if not paths:
        raise FileNotFoundError(f"No csv found in: {fit_params_dir}")

    # 1) 各CSVを「index=params, column=column_name」の形で読み込んで横結合
    df_all = pd.DataFrame()
    meta = {}  # column_name -> source_file
    for p in paths:
        df = pd.read_csv(p)
        # Step3の保存形式（df_params.to_csv）を想定:
        # 1列目が index相当（params）
        first = df.columns[0]
        df = df.rename(columns={first: "params"}).set_index("params")

        # ここで df は 1列（column_name）になっているはず
        for col in df.columns:
            meta[str(col)] = p.name

        df_all = pd.concat([df_all, df], axis=1)

    # 2) 転置して 1行=1条件へ
    df_t = df_all.T.copy()
    df_t.index.name = "column_name"
    df_t = df_t.reset_index()

    # 3) column_name をパースして測定条件列を作る
    parsed = df_t["column_name"].apply(_parse_column_name)
    df_t["temp"] = parsed.apply(lambda x: x[0])
    df_t["period_3"] = parsed.apply(lambda x: x[1])
    df_t["wavelength"] = parsed.apply(lambda x: x[2])
    df_t["filter_combination"] = parsed.apply(lambda x: x[3])

    # 4) source_file を付与
    df_t["source_file"] = df_t["column_name"].map(meta)

    # 5) 並び替え（temp → period_3 → wavelength → filter）
    df_t = df_t.sort_values(
        by=["temp", "period_3", "wavelength", "filter_combination"],
        na_position="last",
        kind="mergesort",  # 安定ソート
    ).reset_index(drop=True)

    # 6) 列順の整形（存在しない可能性に備えて安全に）
    preferred = [
        "temp", "period_3", "wavelength", "filter_combination",
        "opt_B", "opt_C", "opt_D", "k_1", "k_2",
        "column_name", "source_file",
    ]
    cols = [c for c in preferred if c in df_t.columns] + [c for c in df_t.columns if c not in preferred]
    return df_t[cols]


def save_concat_csv(df_concat: pd.DataFrame, out_csv: Path) -> None:
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df_concat.to_csv(out_csv, index=False)
