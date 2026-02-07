from pathlib import Path
import streamlit as st

from src.step4 import load_and_concat_fit_params, save_concat_csv

st.title("Step4: fit_params をまとめて 1つのCSVに統合")

fit_params_dir = Path("artifacts/step3/fit_params")
st.write(f"対象フォルダ: {fit_params_dir}")

if st.button("fit_params を読み込んで統合"):
    try:
        df_concat = load_and_concat_fit_params(fit_params_dir)
        st.session_state.df_concat = df_concat
        st.success(f"統合完了: {len(df_concat)} 条件")
        st.dataframe(df_concat)
    except Exception as e:
        st.error(f"失敗: {e}")

if "df_concat" in st.session_state:
    st.subheader("保存")
    out_csv = Path("artifacts/step4") / "fit_params_concat.csv"
    if st.button("統合CSVを保存"):
        save_concat_csv(st.session_state.df_concat, out_csv)
        st.success(f"保存しました: {out_csv}")
