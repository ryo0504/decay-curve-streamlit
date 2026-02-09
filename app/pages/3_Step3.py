from pathlib import Path

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from src.step3 import (
    build_combined_dataset_from_segments,
    fit_f1_on_combined,
    save_fit_csv,
    calc_rate_constants,
    build_param_df,
)

st.set_page_config(page_title="Step3: Kinetics Fitting", layout="wide")
st.title("速度論モデルでのフィッティング")

# ============================================================
# 0) Utilities
# ============================================================
ARTIFACTS_ROOT = Path("artifacts")

def list_candidate_dirs(root: Path, default: Path) -> list[Path]:
    """UI用：root配下のディレクトリ候補を列挙（default含む）"""
    dirs = []
    if default.exists():
        dirs.append(default)
    if root.exists():
        dirs += [p for p in sorted(root.glob("*")) if p.is_dir()]
    # 重複除去（順序保持）
    seen = set()
    uniq = []
    for d in dirs:
        if d.resolve() not in seen:
            uniq.append(d)
            seen.add(d.resolve())
    return uniq

def ensure_has_csv(in_dir: Path) -> list[Path]:
    csvs = sorted(in_dir.glob("*.csv"))
    return csvs

def build_synced_out_dir(in_dir: Path) -> Path:
    """
    in_dir が artifacts/step2/raw 配下なら、その raw 配下の相対パスを
    artifacts/step3 配下へ写像する。

    例:
      artifacts/step2/raw           -> artifacts/step3
      artifacts/step2/raw/foo       -> artifacts/step3/foo
      artifacts/step2/raw/foo/bar   -> artifacts/step3/foo/bar
    """
    in_dir = in_dir.resolve()
    root = ARTIFACTS_ROOT.resolve()

    step2_raw = (root / "step2" / "raw").resolve()

    try:
        tail = in_dir.relative_to(step2_raw)   # raw配下だけ取り出す（空でもOK）
        return (root / "step3" / tail).resolve()
    except Exception:
        # 想定外の場所なら artifacts/step3/<入力フォルダ名> にまとめる
        return (root / "step3" / in_dir.name).resolve()



# ============================================================
# 1) Input folder selection (Step2 CSV)
# ============================================================
st.subheader("入力（Step2 CSV）")


STEP2_ROOT = ARTIFACTS_ROOT / "step2" / "raw"
default_in_dir = STEP2_ROOT
candidate_dirs = list_candidate_dirs(STEP2_ROOT, default_in_dir)



colA, colB = st.columns([2, 3])
with colA:
    in_dir = st.selectbox(
        "Step2 CSV のフォルダを選択",
        options=candidate_dirs,
        index=0 if default_in_dir in candidate_dirs else 0,
        format_func=lambda p: str(p),
    )

with colB:
    # 手入力でも変えられるように（フォルダピッカーが無いので実用的）
    in_dir_text = st.text_input("またはパスを直接入力", value=str(in_dir))
    in_dir = Path(in_dir_text).expanduser()

csv_files = ensure_has_csv(in_dir)
if not in_dir.exists():
    st.error(f"フォルダが存在しません: {in_dir}")
    st.stop()

if not csv_files:
    st.warning(f"{in_dir} にCSVがありません。Step2を先に実行してください。")
    st.stop()

selected = st.selectbox(
    "フィットする Step2 CSV を選択",
    csv_files,
    format_func=lambda p: p.name,
)
df_step2 = pd.read_csv(selected)


# ============================================================
# 2) Measurement conditions
# ============================================================
st.subheader("測定条件")

temp = st.number_input("Temperature (°C)", value=30)
wavelength = st.number_input("Wavelength (nm)", value=490)
filter_combination = st.selectbox(
    "Filter combination",
    ["001", "010", "011", "100", "101", "110"],
)
period_3 = st.number_input(
    "Period 3 (optional)",
    value=0,
    help="0 の場合は period_3 なしとして扱います",
)

def build_column_name(temp, wavelength, filter_combination, period_3):
    if period_3 > 0:
        return f"{temp} {period_3} {wavelength} {filter_combination}"
    else:
        return f"{temp} {wavelength} {filter_combination}"

column_name = build_column_name(
    temp=temp,
    wavelength=wavelength,
    filter_combination=filter_combination,
    period_3=period_3,
)
st.caption(f"→ 出力名: {column_name}")


# ============================================================
# 3) Kinetics model selection (UI editable)
# ============================================================
st.subheader("速度論モデル")

# いまは fit_f1_on_combined しか無いので、UIは将来拡張前提の形にしておく
MODEL_REGISTRY = {
    "F1 (default)": {
        "fit_func": fit_f1_on_combined,
        "label": "現在のデフォルトモデル",
        # ここに数式イメージなど出したければ latex を置ける（例）
        "latex": r"y = A \left( 1 - \frac{1 - e^{D x}}{1 - C e^{D x}} \right) + B"
    },
    # 将来追加するならここに：
    # "F2": {"fit_func": fit_f2_on_combined, "label": "...", "latex": r"..."},
}

model_key = st.selectbox(
    "使用するモデル",
    options=list(MODEL_REGISTRY.keys()),
    index=0,
    help="デフォルトは現状のモデル（F1）です。将来モデルを追加したらここで切替可能になります。",
)

with st.expander("モデルの説明（表示）", expanded=False):
    st.write(MODEL_REGISTRY[model_key]["label"])
    # 必要なら latex を表示
    st.latex(MODEL_REGISTRY[model_key]["latex"])


# ============================================================
# 4) Fitting parameters
# ============================================================
st.subheader("フィッティング設定")

plot_number = st.number_input(
    "Amin 計算に用いる末尾点数 (plot_number)",
    min_value=1,
    max_value=10000,
    value=20,
    step=1,
)

col1, col2, col3 = st.columns(3)
with col1:
    p0_B = st.number_input("初期値 B", value=0.0)
with col2:
    p0_C = st.number_input("初期値 C", value=0.5)
with col3:
    p0_D = st.number_input("初期値 D", value=-0.005, format="%.6f")


# ============================================================
# 5) Run fitting
# ============================================================
if st.button("統合 → フィット実行"):
    try:
        combined = build_combined_dataset_from_segments(
            df_step2,
            plot_number=int(plot_number),
        )

        fit_func = MODEL_REGISTRY[model_key]["fit_func"]
        res = fit_func(
            combined,
            p0=(float(p0_B), float(p0_C), float(p0_D)),
        )

        st.session_state.fit_res = res
        st.session_state.fit_column_name = column_name
        st.session_state.fit_model_key = model_key
        st.session_state.fit_in_dir = str(in_dir)

        st.success(
            f"フィット成功：データ点数 = {len(combined)}, "
            f"セグメント数 = {combined['seg_id'].nunique()}"
        )

    except Exception as e:
        st.error(f"フィットに失敗しました: {e}")


# ============================================================
# 6) Show results
# ============================================================
if "fit_res" in st.session_state:
    res = st.session_state.fit_res

    B, C, D = res.popt
    eB, eC, eD = res.perr
    k1, k2 = calc_rate_constants(B, C, D)

    st.subheader("最適化パラメータ・速度定数")

    df_params = build_param_df(
        popt=res.popt,
        column_name=st.session_state.fit_column_name,
    )
    st.dataframe(df_params, use_container_width=True)

    st.write(
        {
            "model": st.session_state.get("fit_model_key", "N/A"),
            "R2": res.r2,
            "t95": res.t95,
            "k1": k1,
            "k2": k2,
        }
    )

    # ========================================================
    # Plot (matplotlib)
    # ========================================================
    st.subheader("フィット結果（規格化データ・結合）")

    x = res.df_fit["norm_time"].to_numpy()
    y = res.df_fit["norm_abs"].to_numpy()
    yfit = res.df_fit["fit_curve"].to_numpy()

    fig, ax = plt.subplots(figsize=(8, 5), dpi=130)
    ax.plot(x, y, linestyle="none", marker="o", markersize=3, label="data")
    ax.plot(x, yfit, linewidth=2, label="fit")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Normalized absorbance")
    ax.legend()
    ax.grid(True, alpha=0.3)

    st.pyplot(fig, use_container_width=True)


    # ========================================================
    # Save outputs (out folder synced with input folder)
    # ========================================================
    st.subheader("保存")

    # out_dir は入力ディレクトリに同期（UIからも編集可能）
    synced_out_root = build_synced_out_dir(Path(st.session_state.get("fit_in_dir", str(in_dir))))
    colO1, colO2 = st.columns([2, 3])
    with colO1:
        out_root = st.text_input("出力フォルダ（入力フォルダに同期）", value=str(synced_out_root))
        out_root = Path(out_root).expanduser()
    with colO2:
        st.caption("例: artifacts/step3/step2 など。入力フォルダを変えると同期先の提案も変わります。")

    combined_dir = out_root / "combined_data"
    param_dir = out_root / "fit_params"
    fig_dir = out_root / "fit_fig"

    combined_dir.mkdir(parents=True, exist_ok=True)
    param_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    combined_csv = combined_dir / f"{column_name}_combined.csv"
    param_csv = param_dir / f"{column_name}_params.csv"
    fig_path = fig_dir / f"{column_name}.png"

    if st.button("解析結果を保存"):
        # 結合データ
        save_fit_csv(res.df_fit, combined_csv)

        # パラメータ
        # df_params.to_csv(param_csv, index=False)
        df_params_save = pd.DataFrame(
            {column_name: [B, C, D, k1, k2]},
            index=["opt_B", "opt_C", "opt_D", "k_1", "k_2"]
        )
        df_params_save.to_csv(param_csv, index=True, header=True, index_label="")

        # 図（matplotlib → PNG）
        fig.savefig(fig_path, bbox_inches="tight")

        st.success("保存が完了しました")
        st.write(
            {
                "combined_csv": str(combined_csv),
                "param_csv": str(param_csv),
                "figure": str(fig_path),
                "out_root": str(out_root),
            }
        )
