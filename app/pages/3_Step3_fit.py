from pathlib import Path
import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from src.step3 import (
    build_combined_dataset_from_segments,
    fit_f1_on_combined,
    save_fit_csv,
    calc_rate_constants,
    build_param_df,
)

st.title("Step3: Kinetic fitting (F1 model)")

# ============================================================
# Input CSV (from Step2)
# ============================================================
in_dir = Path("artifacts/step2")
csv_files = sorted(in_dir.glob("*.csv"))
if not csv_files:
    st.warning("artifacts/step2 にCSVがありません。Step2を先に実行してください。")
    st.stop()

selected = st.selectbox(
    "フィットする Step2 CSV を選択",
    csv_files,
    format_func=lambda p: p.name,
)
df_step2 = pd.read_csv(selected)

# ============================================================
# Measurement conditions
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
# Fitting parameters
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
# Run fitting
# ============================================================
if st.button("統合 → フィット実行"):
    try:
        combined = build_combined_dataset_from_segments(
            df_step2,
            plot_number=int(plot_number),
        )

        res = fit_f1_on_combined(
            combined,
            p0=(float(p0_B), float(p0_C), float(p0_D)),
        )

        st.session_state.fit_res = res
        st.session_state.fit_column_name = column_name

        st.success(
            f"フィット成功：データ点数 = {len(combined)}, "
            f"セグメント数 = {combined['seg_id'].nunique()}"
        )

    except Exception as e:
        st.error(f"フィットに失敗しました: {e}")

# ============================================================
# Show results
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
    st.dataframe(df_params)

    st.write(
        {
            "R2": res.r2,
            "t95": res.t95,
        }
    )

    # ========================================================
    # Plot (combined normalized data)
    # ========================================================
    st.subheader("フィット結果（規格化データ・結合）")

    x = res.df_fit["norm_time"]
    y = res.df_fit["norm_abs"]
    yfit = res.df_fit["fit_curve"]

    fig = go.Figure()
    fig.add_trace(
        go.Scattergl(
            x=x,
            y=y,
            mode="markers",
            marker=dict(size=4),
            name="data",
        )
    )
    fig.add_trace(
        go.Scattergl(
            x=x,
            y=yfit,
            mode="lines",
            name="fit",
        )
    )

    fig.update_layout(
        height=520,
        margin=dict(l=20, r=20, t=40, b=20),
        xaxis_title="Normalized time",
        yaxis_title="Normalized absorbance",
    )

    st.plotly_chart(fig, use_container_width=True)

    # ========================================================
    # Save outputs
    # ========================================================
    st.subheader("保存")

    base_dir = Path("artifacts/step3")
    combined_dir = base_dir / "combined_data"
    param_dir = base_dir / "fit_params"
    fig_dir = base_dir / "fit_fig"

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
        df_params.to_csv(param_csv)

        # 図（Plotly → PNG）
        fig.write_image(fig_path, scale=2)

        st.success("保存が完了しました")
        st.write(
            {
                "combined_csv": combined_csv,
                "param_csv": param_csv,
                "figure": fig_path,
            }
        )
