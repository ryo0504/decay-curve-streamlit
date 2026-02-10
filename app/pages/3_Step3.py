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
st.title("Step3: 速度論モデルでのフィッティング")

# ============================================================
# 0) Utilities
# ============================================================
ARTIFACTS_ROOT = Path("artifacts")
STEP2_ROOT = ARTIFACTS_ROOT / "step2" / "raw"
STEP3_ROOT = ARTIFACTS_ROOT / "step3"

def list_dirs_one_level(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return [p for p in sorted(root.glob("*")) if p.is_dir()]

def list_csv_files(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return sorted(root.glob("*.csv"))

def pick_path_with_fallback(label: str, default_path: Path) -> Path:
    """テキスト入力で柔軟にパス上書きできるようにする"""
    ptxt = st.text_input(label, value=str(default_path))
    return Path(ptxt).expanduser()

def build_step3_out_root(sample_name: str, temp_name: str) -> Path:
    # artifacts/step3/{sample}/{temp}
    return STEP3_ROOT / sample_name / temp_name

def default_step2_dir(sample_name: str, temp_name: str) -> Path:
    # artifacts/step2/raw/{sample}/{temp}
    return STEP2_ROOT / sample_name / temp_name


# ============================================================
# 1) Input selection (sample/temp driven)
# ============================================================
st.subheader("入力（Step2 CSV）")

colS, colT = st.columns([2, 2])

with colS:
    sample_dirs = list_dirs_one_level(STEP2_ROOT)

    # legacy: artifacts/step2/raw/*.csv しか無いケースの救済
    legacy_csvs = list_csv_files(STEP2_ROOT)

    if (not sample_dirs) and legacy_csvs:
        st.info("旧構成（artifacts/step2/raw/*.csv）を検出しました。legacy モードで動作します。")
        sample_name = "__legacy__"
        temp_name = "__legacy__"
    else:
        if not sample_dirs:
            st.error("artifacts/step2/raw 配下に sample フォルダが見つかりません。")
            st.stop()

        sample_dir = st.selectbox(
            "sample_name",
            options=sample_dirs,
            format_func=lambda p: p.name,
        )
        sample_name = sample_dir.name

with colT:
    if sample_name == "__legacy__":
        temp_name = "__legacy__"
    else:
        temp_dirs = list_dirs_one_level(STEP2_ROOT / sample_name)
        if not temp_dirs:
            st.error(f"sample={sample_name} 配下に temp フォルダが見つかりません。")
            st.stop()

        temp_dir = st.selectbox(
            "temp",
            options=temp_dirs,
            format_func=lambda p: p.name,
        )
        temp_name = temp_dir.name

# デフォルト入力ディレクトリ
if sample_name == "__legacy__":
    default_in_dir = STEP2_ROOT
else:
    default_in_dir = default_step2_dir(sample_name, temp_name)

# ディレクトリは直入力で上書き可能
in_dir = pick_path_with_fallback("Step2 CSV フォルダ（必要ならパス直入力で上書き）", default_in_dir)

st.caption(f"Input dir: {in_dir}")

if not in_dir.exists():
    st.error(f"フォルダが存在しません: {in_dir}")
    st.stop()

csv_files = list_csv_files(in_dir)
if not csv_files:
    st.warning(f"{in_dir} にCSVがありません。Step2を先に実行してください。")
    st.stop()

# まずは「フォルダ内から選択」のデフォルト
selected_csv = st.selectbox(
    "フィットする Step2 CSV を選択",
    options=csv_files,
    format_func=lambda p: p.name,
)

# ただし最終的にはパス直入力で上書きできるようにする（Step6と同じ）
selected_csv = pick_path_with_fallback("または Step2 CSV のパスを直接入力", selected_csv)

if not selected_csv.exists():
    st.error(f"CSVが見つかりません: {selected_csv}")
    st.stop()

df_step2 = pd.read_csv(selected_csv)


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

MODEL_REGISTRY = {
    "F1 (default)": {
        "fit_func": fit_f1_on_combined,
        "label": "現在のデフォルトモデル",
        "latex": r"y = A \left( 1 - \frac{1 - e^{D x}}{1 - C e^{D x}} \right) + B"
    },
}

model_key = st.selectbox(
    "使用するモデル",
    options=list(MODEL_REGISTRY.keys()),
    index=0,
)

with st.expander("モデルの説明（表示）", expanded=False):
    st.write(MODEL_REGISTRY[model_key]["label"])
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
run = st.button("統合 → フィット実行")

if run:
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

        st.session_state.step3_fit_res = res
        st.session_state.step3_column_name = column_name
        st.session_state.step3_model_key = model_key
        st.session_state.step3_in_dir = str(in_dir)
        st.session_state.step3_selected_csv = str(selected_csv)
        st.session_state.step3_sample = sample_name
        st.session_state.step3_temp = temp_name

        st.success(
            f"フィット成功：データ点数 = {len(combined)}, "
            f"セグメント数 = {combined['seg_id'].nunique()}"
        )

    except Exception as e:
        st.error(f"フィットに失敗しました: {e}")


# ============================================================
# 6) Show results + Save
# ============================================================
if "step3_fit_res" in st.session_state:
    res = st.session_state.step3_fit_res

    B, C, D = res.popt
    eB, eC, eD = res.perr
    k1, k2 = calc_rate_constants(B, C, D)

    st.subheader("最適化パラメータ・速度定数")

    df_params = build_param_df(
        popt=res.popt,
        column_name=st.session_state.step3_column_name,
    )
    st.dataframe(df_params, use_container_width=True)

    st.write(
        {
            "model": st.session_state.get("step3_model_key", "N/A"),
            "R2": res.r2,
            "t95": res.t95,
            "k1": k1,
            "k2": k2,
        }
    )

    # ---- Plot ----
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

    # ---- Save ----
    st.subheader("保存")

    sample_name = st.session_state.get("step3_sample", "unknown_sample")
    temp_name = st.session_state.get("step3_temp", "unknown_temp")

    # 出力先：入力選択に同期
    if sample_name == "__legacy__":
        default_out_root = STEP3_ROOT / "legacy"
    else:
        default_out_root = build_step3_out_root(sample_name, temp_name)

    colO1, colO2 = st.columns([2, 3])
    with colO1:
        out_root = pick_path_with_fallback("出力フォルダ（入力に同期）", default_out_root)
    with colO2:
        st.caption("基本は artifacts/step3/{sample}/{temp} 配下に出す想定（必要なら上書き可）。")

    combined_dir = out_root / "combined_data"
    param_dir = out_root / "fit_params"
    fig_dir = out_root / "fit_fig"

    combined_dir.mkdir(parents=True, exist_ok=True)
    param_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    column_name = st.session_state.get("step3_column_name", "unknown")
    combined_csv = combined_dir / f"{column_name}_combined.csv"
    param_csv = param_dir / f"{column_name}_params.csv"
    fig_path = fig_dir / f"{column_name}.png"

    if st.button("解析結果を保存"):
        # 結合データ
        save_fit_csv(res.df_fit, combined_csv)

        # パラメータ
        df_params_save = pd.DataFrame(
            {column_name: [B, C, D, k1, k2]},
            index=["opt_B", "opt_C", "opt_D", "k_1", "k_2"]
        )
        df_params_save.to_csv(param_csv, index=True, header=True, index_label="")

        # 図
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
