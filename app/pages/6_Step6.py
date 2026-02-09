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

st.set_page_config(page_title="Step6", layout="wide")
st.title("Light intensity vs k1/k2 → 線形フィットで kp1, kt1, kp2, kt2 を算出")

# ============================================================
# 0) Utilities
# ============================================================
ARTIFACTS_ROOT = Path("artifacts")

def list_dirs_one_level(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return [p for p in sorted(root.glob("*")) if p.is_dir()]

def safe_rel_to(root: Path, p: Path) -> Path | None:
    try:
        return p.resolve().relative_to(root.resolve())
    except Exception:
        return None

def build_step6_out_dir(sample_name: str, temp_name: str) -> Path:
    # artifacts/step6/{sample}/{temp}
    return ARTIFACTS_ROOT / "step6" / sample_name / temp_name

def pick_file_with_fallback(label: str, default_path: Path) -> Path:
    """テキスト入力で柔軟にパス上書きできるようにする"""
    ptxt = st.text_input(label, value=str(default_path))
    return Path(ptxt).expanduser()

# ============================================================
# 1) Input selection (sample/temp driven)
# ============================================================
st.subheader("入力（Step4 および Step5）")

STEP4_ROOT = ARTIFACTS_ROOT / "step4"
STEP5_ROOT = ARTIFACTS_ROOT / "step5"

colS, colT = st.columns([2, 2])

with colS:
    sample_dirs = list_dirs_one_level(STEP4_ROOT)
    # step4が無い場合でも、step5側から選べるようにする
    if not sample_dirs:
        sample_dirs = list_dirs_one_level(STEP5_ROOT)

    if not sample_dirs:
        st.error("artifacts/step4 または artifacts/step5 配下に sample フォルダが見つかりません。")
        st.stop()

    sample_dir = st.selectbox(
        "sample_name",
        options=sample_dirs,
        format_func=lambda p: p.name,
    )
    sample_name = sample_dir.name

with colT:
    temp_dirs = list_dirs_one_level(STEP4_ROOT / sample_name)
    if not temp_dirs:
        temp_dirs = list_dirs_one_level(STEP5_ROOT / sample_name)

    if not temp_dirs:
        st.error(f"sample={sample_name} 配下に temp フォルダが見つかりません。")
        st.stop()

    temp_dir = st.selectbox(
        "temp",
        options=temp_dirs,
        format_func=lambda p: p.name,
    )
    temp_name = temp_dir.name

# デフォルトの入力ファイル
default_fit = STEP4_ROOT / sample_name / temp_name / "fit_params" / "fit_params_concat.csv"
default_vis = STEP5_ROOT / sample_name / temp_name / "vis" / "vis_integrated.csv"

# 実ファイルパス（手入力で上書きOK）
fit_params_path = pick_file_with_fallback("Step4: fit_params_concat.csv", default_fit)
vis_intensity_path = pick_file_with_fallback("Step5: vis_integrated.csv", default_vis)

st.caption(f"Step4 input: {fit_params_path}")
st.caption(f"Step5 input: {vis_intensity_path}")

if not fit_params_path.exists():
    st.warning("Step4 input が見つかりません（パスを確認してください）。")
if not vis_intensity_path.exists():
    st.warning("Step5 input が見つかりません（パスを確認してください）。")

# ============================================================
# 2) Settings
# ============================================================
st.subheader("設定")
intensity_eval = st.number_input("kp算出に用いる光強度（I_vis）", value=1e8, format="%.0f")

run = st.button("Step6を実行（プロット→線形フィット）")

if run:
    try:
        df_fit, df_vis = load_step4_step5(fit_params_path, vis_intensity_path)

        # まずは元コード踏襲：順番で合わせる
        df_merged = merge_by_order(df_fit, df_vis)

        res = compute_k_parameters(df_merged, intensity_eval=float(intensity_eval))

        st.session_state.step6_res = res
        st.session_state.step6_sample = sample_name
        st.session_state.step6_temp = temp_name
        st.session_state.step6_fit_path = str(fit_params_path)
        st.session_state.step6_vis_path = str(vis_intensity_path)

        st.success("完了")

    except Exception as e:
        st.error(f"失敗: {e}")

# ============================================================
# 3) Results + plot + save
# ============================================================
if "step6_res" in st.session_state:
    res = st.session_state.step6_res
    dfm = res.df_merged

    st.subheader("突合データ（確認）")
    st.dataframe(dfm[["light_intensity", "k_1", "k_2"]].head(50), use_container_width=True)

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

    # ---- プロット（matplotlib）----
    x = pd.to_numeric(dfm["light_intensity"], errors="coerce").to_numpy()
    y1 = pd.to_numeric(dfm["k_1"], errors="coerce").to_numpy()
    y2 = pd.to_numeric(dfm["k_2"], errors="coerce").to_numpy()

    mask1 = np.isfinite(x) & np.isfinite(y1)
    mask2 = np.isfinite(x) & np.isfinite(y2)

    fig, ax = plt.subplots(figsize=(7, 6), dpi=130)
    ax.set_xlabel(r"Light intensity (count cm$^{-1}$)")
    ax.set_ylabel(r"Kinetic parameters (s$^{-1}$)")
    ax.tick_params(direction="out")
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)

    color_k1 = "red"
    color_k2 = "blue"

    ax.plot(x[mask1], y1[mask1], "o", color=color_k1, label=r"$k_1$")
    ax.plot(x[mask2], y2[mask2], "o", color=color_k2, label=r"$k_2$")

    xx = np.linspace(np.nanmin(x), np.nanmax(x), 200)
    # フィット線
    ax.plot(
        xx,
        res.fit_k1.slope * xx + res.fit_k1.intercept,
        "--",
        color=color_k1,
    )
    ax.plot(
        xx,
        res.fit_k2.slope * xx + res.fit_k2.intercept,
        "--",
        color=color_k2,
    )

    ax.annotate(
    r"$R^2$=" + str(np.round(res.fit_k1.r2, 3)),
    xy=(0.62, 0.55),
    xycoords="axes fraction",
    color=color_k1,
    )
    ax.annotate(
        r"$R^2$=" + str(np.round(res.fit_k2.r2, 3)),
        xy=(0.62, 0.45),
        xycoords="axes fraction",
        color=color_k2,
    )

    ax.legend()
    fig.tight_layout()
    st.pyplot(fig)

    # ---- 保存 ----
    st.subheader("保存")

    sample_name = st.session_state.get("step6_sample", "unknown_sample")
    temp_name = st.session_state.get("step6_temp", "unknown_temp")

    # 入力選択に同期した out_dir
    default_out_dir = build_step6_out_dir(sample_name, temp_name)

    colO1, colO2 = st.columns([2, 3])
    with colO1:
        out_dir_text = st.text_input(
            "出力フォルダ（入力に同期）",
            value=str(default_out_dir),
        )
        out_dir = Path(out_dir_text).expanduser()
    with colO2:
        st.caption("出力は artifacts/step6/{sample}/{temp} 配下に作る想定です（必要なら上書き可）。")

    out_dir.mkdir(parents=True, exist_ok=True)

    # ファイル名：温度を入れると衝突しにくいが、tempディレクトリに入れるので固定名でもOK
    out_params_csv = out_dir / "parameters.csv"
    out_fig_png = out_dir / "parameters.png"
    out_merged_csv = out_dir / "merged.csv"

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
