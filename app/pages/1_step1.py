from pathlib import Path
import streamlit as st
from src.step1 import convert_timeseries_files

st.title("時系列ファイルをCSVに変換")

DEFAULT_INPUT_DIR = Path("data/raw/S-5M/0")
BASE_OUTPUT_DIR = Path("artifacts/step1")
PATTERN = "*.時系列"

# -------------------------
# helpers
# -------------------------
def derive_output_dir(input_dir: Path, base_output: Path) -> Path:
    """
    input_dir が data/ 配下なら data/ を除いた相対パスを base_output の下に作る。
    それ以外なら外部パス扱いで base_output/_external/... に逃がす。
    """
    input_dir = Path(input_dir)
    base_output = Path(base_output)

    parts = input_dir.parts
    if len(parts) >= 1 and parts[0] == "data":
        rel = Path(*parts[1:])  # drop 'data'
        return base_output / rel

    # 例: 絶対パスや data 以外 → 安全にフォールバック
    # 末尾2階層くらいを使って識別（必要なら調整）
    tail = Path(*parts[-3:]) if len(parts) >= 3 else Path(input_dir.name)
    return base_output / "_external" / tail

# -------------------------
# UI state
# -------------------------
if "step1_input_dir" not in st.session_state:
    st.session_state.step1_input_dir = str(DEFAULT_INPUT_DIR)

if "step1_base_output_dir" not in st.session_state:
    st.session_state.step1_base_output_dir = str(BASE_OUTPUT_DIR)

# -------------------------
# UI
# -------------------------
st.subheader("入出力フォルダ")

input_dir_str = st.text_input(
    "input_dir（*.時系列が存在するパス）",
    value=st.session_state.step1_input_dir,
    help="例: data/raw/S-5M/0",
)
st.session_state.step1_input_dir = input_dir_str

base_out_str = st.text_input(
    "output_base_dir（通常は artifacts/step1 のままでOKです）",
    value=st.session_state.step1_base_output_dir,
    help='通常は "artifacts/step1" のままでOK',
)
st.session_state.step1_base_output_dir = base_out_str

input_dir = Path(st.session_state.step1_input_dir)
base_output_dir = Path(st.session_state.step1_base_output_dir)
output_dir = derive_output_dir(input_dir, base_output_dir)

st.write("変換後のcsvファイルが保存されるフォルダ")
st.code(str(output_dir))

# 入力存在チェック & 検出件数
if input_dir.exists():
    files = sorted(input_dir.glob(PATTERN))
    st.success(f"入力フォルダOK: {input_dir}（{len(files)} files）")
    if files:
        with st.expander("検出したファイル名（先頭20件）"):
            for p in files[:20]:
                st.write(p.name)
else:
    st.warning(f"入力フォルダが見つかりません: {input_dir}")

st.divider()

# 実行
if st.button("変換実行"):
    try:
        summary = convert_timeseries_files(
            input_dir=input_dir,
            output_dir=output_dir,
            pattern=PATTERN,
        )
        st.success(f"変換完了 出力先：{output_dir}")
        st.dataframe(summary)
        
    except Exception as e:
        st.error(f"変換に失敗しました: {e}")
    
