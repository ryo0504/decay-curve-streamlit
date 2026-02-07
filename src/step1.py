from __future__ import annotations

import math
from pathlib import Path
import pandas as pd


def convert_timeseries_files(
    input_dir: Path,
    output_dir: Path,
    pattern: str = "*.時系列",
    encoding: str = "utf-8",
) -> pd.DataFrame:
    """
    Convert proprietary '.時系列' files into CSV (time, abs).

    - Input:  input_dir / pattern (default: '*.時系列')
    - Output: output_dir / <original_stem>.csv

    Returns a summary DataFrame with:
      - source_path
      - output_csv
      - file_stem
      - wavelength_floor (optional; extracted if possible)
      - n_rows
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    paths = sorted(input_dir.glob(pattern))

    summary_rows: list[dict] = []

    for path in paths:
        file_stem = path.stem  # cross-platform (Mac/Windows)

        # Read as generic table; original code assumes tab-delimited-ish text
        # Using engine="python" helps with irregular whitespace sometimes.
        col_names = [f"c{i:02d}" for i in range(5)]
        with path.open("r", encoding=encoding, errors="ignore") as f:
            df = pd.read_table(f, names=col_names, engine="python")

        # --- (Optional) Extract wavelength from 2nd row 1st col ---
        wavelength_floor = None
        try:
            single_g_at = str(df.iloc[1, 0])
            original_wavelength = single_g_at.replace("Trend type: Single g at ", "")
            wavelength_floor = str(math.floor(float(original_wavelength)))
        except Exception:
            # keep None if format differs
            wavelength_floor = None

        # --- Find header row that contains "Date" ---
        # Original logic: find first row where any cell contains "Date"
        mask = df.astype(str).apply(lambda r: r.str.contains("Date", na=False)).any(axis=1)
        if not mask.any():
            raise ValueError(f'"Date" header row not found in file: {path}')

        date_idx = mask[mask].index[0]

        header = df.loc[date_idx].astype(str).tolist()
        output_df = df.loc[date_idx + 1 :].copy()
        output_df.columns = header
        output_df = output_df.reset_index(drop=True)

        # Convert time column to float, then rename columns to time/abs.
        # Original assumes:
        #   output_df.columns[1] -> time
        #   output_df.columns[2] -> abs
        if len(output_df.columns) < 3:
            raise ValueError(f"Unexpected column count after header parsing in file: {path}")

        time_col = output_df.columns[1]
        abs_col = output_df.columns[2]

        output_df[time_col] = output_df[time_col].map(lambda x: float(x))
        output_df = output_df.rename(columns={time_col: "time", abs_col: "abs"})
        output_df = output_df.loc[:, ["time", "abs"]]

        out_csv = output_dir / f"{file_stem}.csv"
        output_df.to_csv(out_csv, index=False)

        summary_rows.append(
            {
                "source_path": str(path),
                "output_csv": str(out_csv),
                "file_stem": file_stem,
                "wavelength_floor": wavelength_floor,
                "n_rows": int(len(output_df)),
            }
        )

    return pd.DataFrame(summary_rows)


if __name__ == "__main__":
    # Example local run (from project root):
    #   python -m src.step1
    summary = convert_timeseries_files(
        input_dir=Path("data/raw/test"),
        output_dir=Path("artifacts/step1"),
    )
    print(summary)
