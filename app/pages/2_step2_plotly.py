from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from src.step2 import build_segment_table, save_segment_csv, boundary_preview

st.title("Step2 (Plotly): ズーム表示範囲内でドラッグ選択 → 区間化")

# --- 入力CSV選択（Step1成果物）---
step1_dir = Path("artifacts/step1")
csv_files = sorted(step1_dir.glob("*.csv"))
if not csv_files:
    st.warning("artifacts/step1 にCSVがありません。Step1を先に実行してください。")
    st.stop()

selected = st.selectbox("処理するCSVを選択", csv_files, format_func=lambda p: p.name)
df = pd.read_csv(selected)

t = df["time"].to_numpy()
a = df["abs"].to_numpy()
t_min, t_max = float(np.min(t)), float(np.max(t))

# --- state ---
if "segments_time" not in st.session_state:
    st.session_state.segments_time = []  # list[(t0,t1)]
if "last_saved_segments_time" not in st.session_state:
    st.session_state.last_saved_segments_time = None  # for post-save plot

# =========================
# 1) 表示範囲（ズーム）をUIで指定
# =========================
st.subheader("表示範囲（ズーム）")
view_t0, view_t1 = st.slider(
    "表示する time 範囲（この範囲内の点だけをドラッグ選択できます）",
    min_value=t_min,
    max_value=t_max,
    value=(t_min, t_max),
)

# 表示範囲に対応する index
v0 = int(np.searchsorted(t, view_t0, side="left"))
v1 = int(np.searchsorted(t, view_t1, side="right"))
t_view = t[v0:v1]
a_view = a[v0:v1]

# y表示範囲（自動）
auto_y = st.checkbox("y軸を表示範囲に自動フィット", value=True)
if auto_y:
    if len(a_view) > 0:
        y0, y1 = float(np.min(a_view)), float(np.max(a_view))
        pad = 0.05 * (y1 - y0) if y1 > y0 else 1.0
        y_range = (y0 - pad, y1 + pad)
    else:
        y_range = (float(np.min(a)), float(np.max(a)))
else:
    y_min_default, y_max_default = float(np.min(a)), float(np.max(a))
    y_range = st.slider(
        "表示する abs 範囲（手動）",
        min_value=y_min_default,
        max_value=y_max_default,
        value=(y_min_default, y_max_default),
    )

# =========================
# 2) Plotly 図（表示範囲内だけ描画）
# =========================
fig = go.Figure()
# 表示範囲だけ描画する（＝“画面に表示されている点”が選択対象になる）
fig.add_trace(
    go.Scattergl(
        x=t_view,
        y=a_view,
        mode="markers",
        marker=dict(size=3),
        name="data(view)",
        customdata=np.arange(v0, v1),  # 元データの index を持たせる
    )
)

# 追加済み区間を縦帯で表示（全体座標系のtimeなので、そのままvrectでOK）
for (s0, s1) in st.session_state.segments_time:
    fig.add_vrect(x0=s0, x1=s1, fillcolor="red", opacity=0.15, line_width=0)

fig.update_layout(
    height=520,
    dragmode="select",  # "lasso" にしたければここだけ変更
    title="表示範囲内でドラッグ選択 → 下で区間に追加",
    margin=dict(l=20, r=20, t=50, b=20),
)
fig.update_xaxes(range=[view_t0, view_t1])
fig.update_yaxes(range=list(y_range))

event = st.plotly_chart(fig, use_container_width=True, on_select="rerun")

# =========================
# 3) 選択点 → 区間化して追加
# =========================
selected_points = []
try:
    selected_points = event.selection.get("points", [])
except Exception:
    selected_points = []

if selected_points:
    xs = np.array([p["x"] for p in selected_points], dtype=float)
    seg_t0, seg_t1 = float(xs.min()), float(xs.max())

    st.info(f"選択点から推定した区間: time {seg_t0:.6g} 〜 {seg_t1:.6g}（点数: {len(xs)}）")

    c1, c2 = st.columns(2)
    with c1:
        if st.button("この選択を区間として追加"):
            if seg_t0 < seg_t1:
                st.session_state.segments_time.append((seg_t0, seg_t1))
                st.rerun()
            else:
                st.error("区間がゼロ幅です。もう少し広く選択してください。")
    with c2:
        if st.button("この選択を破棄"):
            st.rerun()
else:
    st.caption("まだ点が選択されていません（表示範囲内でドラッグしてください）。")

# --- 区間管理 ---
st.subheader("区間リスト")
st.write(st.session_state.segments_time)

c3, c4 = st.columns(2)
with c3:
    if st.button("最後の区間を削除"):
        if st.session_state.segments_time:
            st.session_state.segments_time.pop()
            st.rerun()
with c4:
    if st.button("全クリア"):
        st.session_state.segments_time = []
        st.rerun()

# =========================
# 4) 確定 → CSV出力 → 選択範囲のグラフ表示
# =========================
st.subheader("CSV出力")
if st.button("確定してCSVを作成"):
    # time区間 → index区間へ
    segments_idx = []
    for (s0, s1) in st.session_state.segments_time:
        i0 = int(np.searchsorted(t, s0, side="left"))
        i1 = int(np.searchsorted(t, s1, side="right"))
        if i0 < i1:
            segments_idx.append((i0, i1))

    df_result = build_segment_table(df, segments_idx)
    if df_result.empty:
        st.error("区間が未設定です。")
    else:
        out_csv = Path("artifacts/step2") / f"{selected.stem}_segments.csv"
        save_segment_csv(df_result, out_csv)
        st.success(f"保存しました: {out_csv}")

        # ここで「確定時に選択範囲だけのグラフ」を表示するため保存
        st.session_state.last_saved_segments_time = list(st.session_state.segments_time)

        st.dataframe(df_result.head(50))
        st.rerun()

# =========================
# 5) 確定後：選択範囲のグラフを表示
# =========================
if st.session_state.last_saved_segments_time:
    st.subheader("確定した選択範囲のグラフ（確認用）")

    # 境界の確認
    segments_idx = []
    for (s0, s1) in st.session_state.last_saved_segments_time:
        i0 = int(np.searchsorted(t, s0, side="left"))
        i1 = int(np.searchsorted(t, s1, side="right"))
        if i0 < i1:
            segments_idx.append((i0, i1))

    preview = boundary_preview(df.reset_index(drop=True), segments_idx, k=5)
    st.subheader("境界の確認（start/end ±5点）")
    def highlight_start_offset0(row):
        if (row["boundary"] == "start") and (row["offset"] == 0):
            return ["background-color: #4169e1"] * len(row)
        return [""] * len(row)

    styled = preview.style.apply(highlight_start_offset0, axis=1)
    st.dataframe(styled, use_container_width=True)

    # 選択区間だけ抽出（連結して1つの点群として表示）
    seg_points_t = []
    seg_points_a = []
    t0s = []
    t1s = []

    for (s0, s1) in st.session_state.last_saved_segments_time:
        i0 = int(np.searchsorted(t, s0, side="left"))
        i1 = int(np.searchsorted(t, s1, side="right"))
        if i0 < i1:
            seg_points_t.append(t[i0:i1])
            seg_points_a.append(a[i0:i1])
            t0s.append(s0)
            t1s.append(s1)

    if seg_points_t:
        tt = np.concatenate(seg_points_t)
        aa = np.concatenate(seg_points_a)

        fig2 = go.Figure()
        fig2.add_trace(go.Scattergl(x=tt, y=aa, mode="markers", marker=dict(size=3), name="selected"))

        for (s0, s1) in st.session_state.last_saved_segments_time:
            fig2.add_vrect(x0=s0, x1=s1, fillcolor="red", opacity=0.15, line_width=0)

        # 表示範囲は選択区間をちょい余白付きで自動ズーム
        z0, z1 = min(t0s), max(t1s)
        fig2.update_xaxes(range=[z0, z1])

        y0, y1 = float(np.min(aa)), float(np.max(aa))
        pad = 0.05 * (y1 - y0) if y1 > y0 else 1.0
        fig2.update_yaxes(range=[y0 - pad, y1 + pad])

        fig2.update_layout(height=420, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.info("選択範囲に点がありませんでした。")
