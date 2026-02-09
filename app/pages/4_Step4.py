from pathlib import Path
import streamlit as st

from src.step4 import load_and_concat_fit_params, save_concat_csv

st.set_page_config(page_title="Step4: Concat fit_params", layout="wide")
st.title("fit_paramsを1つのCSVに統合")

# ============================================================
# 0) Utilities
# ============================================================
ARTIFACTS_ROOT = Path("artifacts")

def list_candidate_dirs(root: Path, default: Path) -> list[Path]:
    """UI用：root配下のディレクトリ候補を列挙（default含む、重複除去）"""
    dirs = []
    if default.exists():
        dirs.append(default)
    if root.exists():
        dirs += [p for p in sorted(root.glob("*")) if p.is_dir()]
    seen = set()
    uniq = []
    for d in dirs:
        rp = d.resolve()
        if rp not in seen:
            uniq.append(d)
            seen.add(rp)
    return uniq

def build_step4_out_dir_from_fit_params_dir(fit_params_dir: Path) -> Path:
    """
    入力が artifacts/step3/.../fit_params の場合、
    artifacts/step4/... に同期して出力先ルートを作る。

    例:
      artifacts/step3/fit_params                  -> artifacts/step4/fit_params
      artifacts/step3/step2_raw/foo/fit_params    -> artifacts/step4/step2_raw/foo/fit_params
    """
    root = ARTIFACTS_ROOT.resolve()
    fit_params_dir = fit_params_dir.resolve()
    step3_root = (root / "step3").resolve()

    try:
        rel = fit_params_dir.relative_to(step3_root)  # e.g. "step2_raw/foo/fit_params"
        return (root / "step4" / rel).resolve()
    except Exception:
        # 想定外なら step4/<入力フォルダ名> に寄せる
        return (root / "step4" / fit_params_dir.name).resolve()

# ============================================================
# 1) Input folder selection
# ============================================================
st.subheader("入力（fit_params フォルダ）")

default_fit_params_dir = ARTIFACTS_ROOT / "step3" / "fit_params"
candidate_dirs = list_candidate_dirs(ARTIFACTS_ROOT / "step3", default_fit_params_dir)

colA, colB = st.columns([2, 3])
with colA:
    fit_params_dir = st.selectbox(
        "対象フォルダを選択",
        options=candidate_dirs,
        index=0 if default_fit_params_dir in candidate_dirs else 0,
        format_func=lambda p: str(p),
    )

with colB:
    fit_params_dir_text = st.text_input("またはパスを直接入力", value=str(fit_params_dir))
    fit_params_dir = Path(fit_params_dir_text).expanduser()

st.write(f"対象フォルダ: {fit_params_dir}")

if not fit_params_dir.exists():
    st.error(f"フォルダが存在しません: {fit_params_dir}")
    st.stop()

# ============================================================
# 2) Run concat
# ============================================================
if st.button("fit_params を読み込んで統合"):
    try:
        df_concat = load_and_concat_fit_params(fit_params_dir)
        st.session_state.df_concat = df_concat
        st.session_state.fit_params_dir = str(fit_params_dir)
        st.success(f"統合完了: {len(df_concat)} 条件")
        st.dataframe(df_concat, use_container_width=True)
    except Exception as e:
        st.error(f"失敗: {e}")

# ============================================================
# 3) Save
# ============================================================
if "df_concat" in st.session_state:
    st.subheader("保存")

    # 入力フォルダに同期した step4 出力先を提案
    synced_out_dir = build_step4_out_dir_from_fit_params_dir(Path(st.session_state.get("fit_params_dir", str(fit_params_dir))))
    colO1, colO2 = st.columns([2, 3])
    with colO1:
        out_dir = st.text_input("出力フォルダ（入力フォルダに同期）", value=str(synced_out_dir))
        out_dir = Path(out_dir).expanduser()
    with colO2:
        st.caption("出力は必ず artifacts/step4 配下になる想定です（必要なら手で上書き可）。")

    out_dir.mkdir(parents=True, exist_ok=True)

    out_csv = out_dir / "fit_params_concat.csv"
    st.write(f"保存先: {out_csv}")

    if st.button("統合CSVを保存"):
        save_concat_csv(st.session_state.df_concat, out_csv)
        st.success(f"保存しました: {out_csv}")
