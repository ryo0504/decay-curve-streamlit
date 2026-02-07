from pathlib import Path
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np

from src.step5 import integrate_vis_folder, save_integral_csv

st.title("Step5: 可視光強度（vis *.txt）の波長範囲積分")

st.subheader("入力")
file_glob = st.text_input(
    "入力ファイルパス",
    value="data/raw/S-5M/0/vis/*.txt",
)

st.subheader("積分範囲（波長 nm）")
col1, col2 = st.columns(2)
with col1:
    wl_min = st.number_input("λ_min (nm)", value=400.0)
with col2:
    wl_max = st.number_input("λ_max (nm)", value=550.0)

st.caption("注意: nmを波数(cm^-1)に変換して、その範囲で photon_count を trapz 積分します。")

run = st.button("積分を実行")

if run:
    try:
        df_out = integrate_vis_folder(file_glob, wavelength_nm_min=float(wl_min), wavelength_nm_max=float(wl_max))
        st.session_state.step5_df = df_out
        st.success(f"完了: {len(df_out)} ファイル")
    except Exception as e:
        st.error(f"失敗: {e}")

if "step5_df" in st.session_state:
    df_out = st.session_state.step5_df

    st.subheader("結果（light_intensity）")
    st.dataframe(df_out)

    # 簡易プロット（棒グラフ）
    st.subheader("プロット")
    fig = go.Figure()
    fig.add_trace(go.Bar(x=df_out.index.astype(str), y=df_out["light_intensity"]))
    fig.update_layout(
        height=420,
        margin=dict(l=20, r=20, t=40, b=120),
        xaxis_title="file",
        yaxis_title="integrated intensity",
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("保存")
    default_out = Path("artifacts/step5") / "vis_integrated.csv"
    out_path = st.text_input("出力CSVパス", value=str(default_out))

    if st.button("CSVを保存"):
        save_integral_csv(df_out, Path(out_path))
        st.success(f"保存しました: {out_path}")
