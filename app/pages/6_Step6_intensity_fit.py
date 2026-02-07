from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from src.step6 import (
    load_step4_step5,
    merge_by_order,
    compute_k_parameters,
    save_parameters_csv,
    save_merged_csv,
)

st.title("Step6: Light intensity vs k1/k2 — 線形フィットで kp, kt を算出")

# 入力ファイル
st.subheader("入力ファイル")
fit_params_path = st.text_input(
    "Step4: fit_params_concat.csv",
    value="artifacts/step4/fit_params_concat.csv",
)

vis_intensity_path = st.text_input(
    "Step5: vis_integrated.csv",
    value="artifacts/step5/vis_integrated.csv",
)

# 設定
st.subheader("設定")
intensity_eval = st.number_input("kp算出に用いる光強度（I_eval）", value=1e8, format="%.0f")

run = st.button("Step6を実行（突合→プロット→線形フィット）")

if run:
    try:
        df_fit, df_vis = load_step4_step5(Path(fit_params_path), Path(vis_intensity_path))

        # まずは元コード踏襲：順番で合わせる
        df_merged = merge_by_order(df_fit, df_vis)

        res = compute_k_parameters(df_merged, intensity_eval=float(intensity_eval))

        st.session_state.step6_res = res
        st.success("完了")

    except Exception as e:
        st.error(f"失敗: {e}")

if "step6_res" in st.session_state:
    res = st.session_state.step6_res
    dfm = res.df_merged

    st.subheader("突合データ（確認）")
    st.dataframe(dfm[["light_intensity", "k_1", "k_2"]].head(50))

    st.subheader("フィット結果")
    st.write({
        "k1_slope": res.fit_k1.slope,
        "k1_intercept(=kt1)": res.fit_k1.intercept,
        "k1_R2": res.fit_k1.r2,
        "k2_slope": res.fit_k2.slope,
        "k2_intercept(=kt2)": res.fit_k2.intercept,
        "k2_R2": res.fit_k2.r2,
        "k_t1": res.k_t1,
        "k_t2": res.k_t2,
        "k_p1": res.k_p1,
        "k_p2": res.k_p2,
        "temperature": res.temperature,
    })

    # ---- プロット（matplotlibで保存しやすく） ----
    x = pd.to_numeric(dfm["light_intensity"], errors="coerce").to_numpy()
    y1 = pd.to_numeric(dfm["k_1"], errors="coerce").to_numpy()
    y2 = pd.to_numeric(dfm["k_2"], errors="coerce").to_numpy()

    mask1 = np.isfinite(x) & np.isfinite(y1)
    mask2 = np.isfinite(x) & np.isfinite(y2)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.set_xlabel(r"Light intensity (count cm$^{-1}$)")
    ax.set_ylabel(r"Kinetic parameters (s$^{-1}$)")
    ax.tick_params(direction="out")
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)

    ax.plot(x[mask1], y1[mask1], "o", color="red", label=r"$k_1$")
    ax.plot(x[mask2], y2[mask2], "o", color="blue", label=r"$k_2$")

    # フィット線
    xx = np.linspace(np.nanmin(x), np.nanmax(x), 200)
    ax.plot(xx, res.fit_k1.slope * xx + res.fit_k1.intercept, "--", color="red")
    ax.plot(xx, res.fit_k2.slope * xx + res.fit_k2.intercept, "--", color="blue")

    ax.annotate(r"$R^2$=" + str(np.round(res.fit_k1.r2, 3)), xy=(0.62, 0.55), xycoords="axes fraction", color="red")
    ax.annotate(r"$R^2$=" + str(np.round(res.fit_k2.r2, 3)), xy=(0.62, 0.45), xycoords="axes fraction", color="blue")

    ax.legend()
    fig.tight_layout()
    st.pyplot(fig)

    # ---- 保存 ----
    st.subheader("保存")
    base = Path("artifacts/step6")
    base.mkdir(parents=True, exist_ok=True)

    temp_label = str(int(res.temperature)) if res.temperature is not None and float(res.temperature).is_integer() else str(res.temperature or "unknown")

    out_params_csv = base / f"{temp_label}_parameters.csv"
    out_fig_png = base / f"{temp_label}_parameters.png"
    out_merged_csv = base / f"{temp_label}_merged.csv"

    if st.button("結果を保存（CSV + PNG）"):
        save_parameters_csv(res, out_params_csv)
        save_merged_csv(dfm, out_merged_csv)
        fig.savefig(out_fig_png, bbox_inches="tight", pad_inches=0)

        st.success("保存しました")
        st.write({
            "parameters_csv": str(out_params_csv),
            "merged_csv": str(out_merged_csv),
            "figure_png": str(out_fig_png),
        })
