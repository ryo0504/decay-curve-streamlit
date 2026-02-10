import streamlit as st
from datetime import datetime

st.set_page_config(
    page_title="Lab App",
    page_icon="",
    layout="wide",
)

st.title("Lab App")
st.caption("Bleaching curve 解析アプリ")

st.markdown(
    """
このアプリは、Bleaching curveの速度論解析の一連の流れを  
**Step1 → Step6** に分けて実行するためのツールです。

---

### ▶ 使い方
- **左サイドバー**から各 Step ページを開いてください
- 基本的な流れ：
  1. Step1：時系列データのCSV変換
  2. Step2：Bleaching curve の区間選択
  3. Step3：速度論モデルによるフィッティング
  4. Step4：複数の温度で測定したデータの統合
  5. Step5：測定光データの積分
  6. Step6：光強度依存性から **kp1, kp2, kt1, kt2** の算出

途中の Step から再実行しても問題ありません  
（入出力パスが合っていればOKです）
"""
)

# ============================================================
# Footer
# ============================================================
st.divider()
st.caption(f"© {datetime.now().year} Ryo Koibuchi. All rights reserved.")
