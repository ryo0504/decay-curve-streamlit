from pathlib import Path
import streamlit as st
from src.step1 import convert_timeseries_files

st.title("Step1: .時系列 → CSV変換")

if st.button("変換実行"):
    summary = convert_timeseries_files(
        input_dir=Path("data/raw/S-5M/0"),
        output_dir=Path("artifacts/step1"),
        pattern="*.時系列",
    )
    st.dataframe(summary)
