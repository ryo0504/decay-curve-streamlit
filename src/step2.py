from __future__ import annotations
from pathlib import Path
import pandas as pd
from typing import List, Tuple

def build_segment_table(
    data: pd.DataFrame,
    segments: List[Tuple[int, int]],
) -> pd.DataFrame:
    """
    segments: [(lpoint, rpoint), ...] rpoint is exclusive
    各区間の time/abs を列方向に並べる（列名をユニーク化）
    """
    df_list = []
    for i, (l, r) in enumerate(segments, start=1):
        dfn = data.iloc[l:r].copy().reset_index(drop=True)

        # 列名をユニークにする（重要）
        dfn = dfn.rename(columns={
            "time": f"time_{i:02d}",
            "abs":  f"abs_{i:02d}",
        })
        df_list.append(dfn)

    if not df_list:
        return pd.DataFrame()

    return pd.concat(df_list, axis=1)

def save_segment_csv(df_result: pd.DataFrame, out_csv: Path) -> None:
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df_result.to_csv(out_csv, index=False)


def boundary_preview(
    data: pd.DataFrame,
    segments_idx: List[Tuple[int, int]],
    k: int = 5,
) -> pd.DataFrame:
    """
    各区間の開始(i0)と終了(i1-1)の周辺±k点を縦持ちで返す。
    """
    rows = []
    n = len(data)

    for seg_i, (i0, i1) in enumerate(segments_idx, start=1):
        # start付近
        for off in range(-k, k + 1):
            j = i0 + off
            if 0 <= j < n:
                rows.append({
                    "segment": seg_i,
                    "boundary": "start",
                    "offset": off,
                    "index": j,
                    "time": data.loc[j, "time"],
                    "abs": data.loc[j, "abs"],
                })

        # end付近（区間の最後の点 = i1-1）
        end_idx = i1 - 1
        for off in range(-k, k + 1):
            j = end_idx + off
            if 0 <= j < n:
                rows.append({
                    "segment": seg_i,
                    "boundary": "end",
                    "offset": off,
                    "index": j,
                    "time": data.loc[j, "time"],
                    "abs": data.loc[j, "abs"],
                })

    return pd.DataFrame(rows)




