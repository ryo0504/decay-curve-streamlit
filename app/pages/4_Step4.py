from pathlib import Path
import streamlit as st

from src.step4 import load_and_concat_fit_params, save_concat_csv

st.set_page_config(page_title="Step4: Concat fit_params", layout="wide")
st.title("Step4: fit_params を 1つのCSVに統合")

# ============================================================
# 0) Utilities
# ============================================================
ARTIFACTS_ROOT = Path("artifacts")
STEP3_ROOT = ARTIFACTS_ROOT / "step3"
STEP4_ROOT = ARTIFACTS_ROOT / "step4"

def list_dirs_one_level(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return [p for p in sorted(root.glob("*")) if p.is_dir()]

def pick_path_with_fallback(label: str, default_path: Path) -> Path:
    """テキスト入力で柔軟にパス上書きできるようにする"""
    ptxt = st.text_input(label, value=str(default_path))
    return Path(ptxt).expanduser()

def build_step4_out_dir(sample_name: str, temp_name: str) -> Path:
    # artifacts/step4/{sample}/{temp}/fit_params
    return STEP4_ROOT / sample_name / temp_name / "fit_params"

def default_fit_params_dir(sample_name: str, temp_name: str) -> Path:
    # artifacts/step3/{sample}/{temp}/fit_params
    return STEP3_ROOT / sample_name / temp_name / "fit_params"


# ============================================================
# 1) Input selection (sample/temp driven)
# ============================================================
st.subheader("入力（Step3 の fit_params フォルダ）")

colS, colT = st.columns([2, 2])

with colS:
    sample_dirs = list_dirs_one_level(STEP3_ROOT)

    # 旧構成: artifacts/step3/fit_params しか無い場合の救済
    legacy_fit_params = STEP3_ROOT / "fit_params"
    if (not sample_dirs) and legacy_fit_params.exists():
        st.info("旧構成（artifacts/step3/fit_params）を検出しました。legacy モードで動作します。")
        sample_name = "__legacy__"
        temp_name = "__legacy__"
    else:
        if not sample_dirs:
            st.error("artifacts/step3 配下に sample フォルダが見つかりません。")
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
        temp_dirs = list_dirs_one_level(STEP3_ROOT / sample_name)
        if not temp_dirs:
            st.error(f"sample={sample_name} 配下に temp フォルダが見つかりません。")
            st.stop()

        temp_dir = st.selectbox(
            "temp",
            options=temp_dirs,
            format_func=lambda p: p.name,
        )
        temp_name = temp_dir.name

# デフォルト入力（ディレクトリ）
if sample_name == "__legacy__":
    default_in_dir = legacy_fit_params
else:
    default_in_dir = default_fit_params_dir(sample_name, temp_name)

fit_params_dir = pick_path_with_fallback("fit_params フォルダ（必要ならパス直入力で上書き）", default_in_dir)

st.caption(f"Input dir: {fit_params_dir}")

if not fit_params_dir.exists():
    st.error(f"フォルダが存在しません: {fit_params_dir}")
    st.stop()

# ============================================================
# 2) Run concat
# ============================================================
run = st.button("fit_params を読み込んで統合")

if run:
    try:
        df_concat = load_and_concat_fit_params(fit_params_dir)
        st.session_state.step4_df_concat = df_concat
        st.session_state.step4_fit_params_dir = str(fit_params_dir)
        st.session_state.step4_sample = sample_name
        st.session_state.step4_temp = temp_name

        st.success(f"統合完了: {len(df_concat)} 条件")
        st.dataframe(df_concat, use_container_width=True)

    except Exception as e:
        st.error(f"失敗: {e}")

# ============================================================
# 3) Save
# ============================================================
if "step4_df_concat" in st.session_state:
    st.subheader("保存")

    sample_name = st.session_state.get("step4_sample", "unknown_sample")
    temp_name = st.session_state.get("step4_temp", "unknown_temp")

    # 出力先：入力（sample/temp）に同期
    if sample_name == "__legacy__":
        # legacy のときは step4/fit_params に寄せる
        default_out_dir = STEP4_ROOT / "fit_params"
    else:
        default_out_dir = build_step4_out_dir(sample_name, temp_name)

    colO1, colO2 = st.columns([2, 3])
    with colO1:
        out_dir = pick_path_with_fallback("出力フォルダ（入力に同期）", default_out_dir)
    with colO2:
        st.caption("基本は artifacts/step4 配下に出す想定（必要なら上書き可）。")

    out_dir.mkdir(parents=True, exist_ok=True)

    out_csv = out_dir / "fit_params_concat.csv"
    st.caption(f"Output: {out_csv}")

    if st.button("統合CSVを保存"):
        try:
            save_concat_csv(st.session_state.step4_df_concat, out_csv)
            st.success(f"保存しました: {out_csv}")
        except Exception as e:
            st.error(f"保存に失敗: {e}")
