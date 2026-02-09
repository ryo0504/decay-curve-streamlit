from pathlib import Path
import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from src.step5 import integrate_vis_folder, save_integral_csv

st.set_page_config(page_title="Step5: VIS integrate", layout="wide")
st.title("可視光強度の波長範囲を積分する")

# ============================================================
# 0) Utilities
# ============================================================
DATA_ROOT = Path("data")          # プロジェクトの data ルート
RAW_ROOT = DATA_ROOT / "raw"      # data/raw
ARTIFACTS_ROOT = Path("artifacts")

def list_dirs_one_level(root: Path, default: Path | None = None) -> list[Path]:
    """root直下のディレクトリ候補（default含む、重複除去）"""
    dirs = []
    if default and default.exists():
        dirs.append(default)
    if root.exists():
        dirs += [p for p in sorted(root.glob("*")) if p.is_dir()]
    seen = set()
    out = []
    for d in dirs:
        rp = d.resolve()
        if rp not in seen:
            out.append(d)
            seen.add(rp)
    return out

def build_synced_step5_out_dir(vis_dir: Path) -> Path:
    """
    vis_dir が data/raw 配下なら、その相対パスを artifacts/step5 配下へ写像する。
    例:
      data/raw/S-5M/0/vis -> artifacts/step5/S-5M/0/vis
    """
    vis_dir = vis_dir.expanduser().resolve()
    raw_root = RAW_ROOT.resolve()
    artifacts = ARTIFACTS_ROOT.resolve()

    try:
        rel = vis_dir.relative_to(raw_root)  # e.g. "S-5M/0/vis"
        return (artifacts / "step5" / rel).resolve()
    except Exception:
        return (artifacts / "step5" / vis_dir.name).resolve()

def vis_glob_from_dir(vis_dir: Path) -> str:
    return str((vis_dir / "*.txt").as_posix())

# ============================================================
# 1) Input folder selection
# ============================================================
st.subheader("入力")

colA, colB, colC = st.columns([2, 2, 3])

# sample_name 選択（data/raw 直下）
with colA:
    sample_candidates = list_dirs_one_level(RAW_ROOT)
    if not sample_candidates:
        st.error(f"{RAW_ROOT} が存在しない or 空です。")
        st.stop()

    sample_dir = st.selectbox(
        "sample_name を選択（data/raw 配下）",
        options=sample_candidates,
        format_func=lambda p: p.name,
    )

# temp 選択（data/raw/{sample} 直下のディレクトリ）
with colB:
    temp_candidates = list_dirs_one_level(sample_dir)
    if not temp_candidates:
        st.error(f"{sample_dir} 配下に温度フォルダがありません。")
        st.stop()

    temp_dir = st.selectbox(
        "temp を選択",
        options=temp_candidates,
        format_func=lambda p: p.name,
    )

# vis フォルダ（通常は vis 固定だが、手入力で柔軟化）
with colC:
    default_vis_dir = temp_dir / "vis"
    vis_dir_text = st.text_input(
        "vis フォルダ（必要ならパスを直接編集）",
        value=str(default_vis_dir),
        help="例: data/raw/S-5M/0/vis",
    )
    vis_dir = Path(vis_dir_text).expanduser()

st.write(f"対象フォルダ: {vis_dir}")

if not vis_dir.exists():
    st.error(f"vis フォルダが存在しません: {vis_dir}")
    st.stop()

# integrate_vis_folder が glob を受けるので glob を作る
file_glob = vis_glob_from_dir(vis_dir)
st.caption(f"入力ファイル: {file_glob}")

# ============================================================
# 2) Integration range
# ============================================================
st.subheader("積分範囲（波長 nm）")
col1, col2 = st.columns(2)
with col1:
    wl_min = st.number_input("λ_min (nm)", value=400.0)
with col2:
    wl_max = st.number_input("λ_max (nm)", value=550.0)

st.caption("注意: nmを波数(cm^-1)に変換して、その範囲で photon_count を trapz 積分します。")

# ============================================================
# 3) Run
# ============================================================
run = st.button("積分を実行")

if run:
    try:
        df_out = integrate_vis_folder(
            file_glob,
            wavelength_nm_min=float(wl_min),
            wavelength_nm_max=float(wl_max),
        )
        st.session_state.step5_df = df_out
        st.session_state.step5_vis_dir = str(vis_dir)
        st.success(f"完了: {len(df_out)} ファイル")
    except Exception as e:
        st.error(f"失敗: {e}")

# ============================================================
# 4) Results + Save
# ============================================================
if "step5_df" in st.session_state:
    df_out = st.session_state.step5_df

    st.subheader("結果（light_intensity）")
    st.dataframe(df_out, use_container_width=True)

    st.subheader("保存")

    # 入力フォルダに同期した artifacts/step5 配下の出力先を提案
    synced_out_dir = build_synced_step5_out_dir(Path(st.session_state.get("step5_vis_dir", str(vis_dir))))
    colO1, colO2 = st.columns([2, 3])
    with colO1:
        out_dir_text = st.text_input(
            "出力フォルダ（入力フォルダに同期）",
            value=str(synced_out_dir),
        )
        out_dir = Path(out_dir_text).expanduser()
    with colO2:
        st.caption("出力は artifacts/step5 配下に同期して作成されます（必要なら手入力で上書き可）。")

    out_dir.mkdir(parents=True, exist_ok=True)

    out_csv = out_dir / "vis_integrated.csv"
    st.write(f"保存先: {out_csv}")

    if st.button("CSVを保存"):
        save_integral_csv(df_out, out_csv)
        st.success(f"保存しました: {out_csv}")
