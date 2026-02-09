from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from src.step2 import build_segment_table, save_segment_csv, boundary_preview

def apply_white_theme(fig):
    fig.update_layout(
        template="plotly_white",          # ← まずテンプレを白に固定
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(color="black"),
    )
    fig.update_xaxes(
        tickfont=dict(color="black"),     # ← 目盛り文字（0,500,1000…）
        title_font=dict(color="black"),
        showgrid=True,
        gridcolor="lightgray",
        zeroline=False,
        showline=True,
        linecolor="black",
        ticks="outside",
    )
    fig.update_yaxes(
        tickfont=dict(color="black"),
        title_font=dict(color="black"),
        showgrid=True,
        gridcolor="lightgray",
        zeroline=False,
        showline=True,
        linecolor="black",
        ticks="outside",
    )


def step2_base_from_step1(step1_dir: Path, default_base: Path) -> Path:
    parts = list(step1_dir.parts)
    if "step1" in parts:
        idx = parts.index("step1")
        parts[idx] = "step2"
        return Path(*parts)
    return default_base


st.title("Bleaching curveの選択")

# -------------------------
# UI params (Step1入力 & Step2出力)
# -------------------------
# =========================
# Defaults
# =========================
DEFAULT_STEP1_DIR = Path("artifacts/step1/raw")
DEFAULT_STEP2_OUT_BASE = Path("artifacts/step2")

if "step2_step1_dir" not in st.session_state:
    st.session_state.step2_step1_dir = str(DEFAULT_STEP1_DIR)

if "step2_auto_out" not in st.session_state:
    st.session_state.step2_auto_out = True

if "step2_out_base_dir" not in st.session_state:
    # ★ DEFAULT_STEP2_OUT_BASE を基準に、step1_dir から導出
    st.session_state.step2_out_base_dir = str(
        step2_base_from_step1(
            Path(st.session_state.step2_step1_dir),
            DEFAULT_STEP2_OUT_BASE,
        )
    )
st.subheader("入出力の設定")
c0, c1, c2 = st.columns([2, 2, 1])

with c0:
    step1_dir_str = st.text_input(
        "入力CSVフォルダ（Step1成果物）",
        value=st.session_state.step2_step1_dir,
    )

with c2:
    auto_out = st.checkbox(
        "出力先を自動追従",
        value=st.session_state.step2_auto_out,
    )

# --- state更新 ---
step1_dir = Path(step1_dir_str)
st.session_state.step2_step1_dir = step1_dir_str
st.session_state.step2_auto_out = auto_out

# ★ 自動追従ONなら、DEFAULT_STEP2_OUT_BASE と同期しつつ更新
if st.session_state.step2_auto_out:
    st.session_state.step2_out_base_dir = str(
        step2_base_from_step1(step1_dir, DEFAULT_STEP2_OUT_BASE)
    )

with c1:
    out_base_dir_str = st.text_input(
        "出力フォルダ（Step2成果物）",
        value=st.session_state.step2_out_base_dir,
    )
    st.session_state.step2_out_base_dir = out_base_dir_str

step1_dir = Path(st.session_state.step2_step1_dir)
out_base_dir = Path(st.session_state.step2_out_base_dir)

st.caption("※ Docker運用ではパスはコンテナ内パスです。ホスト側のファイルを読む場合は docker-compose の volumes 設定が必要です。")

# -------------------------
# 入力CSV検出
# -------------------------
if not step1_dir.exists():
    st.error(f"入力フォルダが見つかりません: {step1_dir}")
    st.stop()

csv_files = sorted(step1_dir.rglob("*.csv"))  # 下位フォルダも含めて拾う（迷い減る）
if not csv_files:
    st.warning(f"{step1_dir} 配下にCSVがありません。Step1を先に実行してください。")
    st.stop()

st.success(f"CSV検出: {len(csv_files)} files")

with st.expander("検出したCSV（先頭30件）"):
    for p in csv_files[:30]:
        st.write(str(p))

# -------------------------
# 選択（変更されたらセグメントをリセット）
# -------------------------
selected = st.selectbox(
    "処理するCSVを選択",
    csv_files,
    format_func=lambda p: str(p.relative_to(step1_dir)) if p.is_relative_to(step1_dir) else p.name,
    key="step2_selected_csv",
)

# セグメントstate初期化（ファイルを変えたとき事故るので）
if "step2_last_selected" not in st.session_state:
    st.session_state.step2_last_selected = None

if st.session_state.step2_last_selected != str(selected):
    st.session_state.step2_last_selected = str(selected)
    st.session_state.segments_time = []
    st.session_state.last_saved_segments_time = None

# --- state ---
if "segments_time" not in st.session_state:
    st.session_state.segments_time = []  # list[(t0,t1)]
if "last_saved_segments_time" not in st.session_state:
    st.session_state.last_saved_segments_time = None  # for post-save plot

# -------------------------
# CSV読み込み
# -------------------------
df = pd.read_csv(selected)

# 列チェック（step1出力が time/abs 前提）
required_cols = {"time", "abs"}
if not required_cols.issubset(df.columns):
    st.error(f"CSVに必要な列 {required_cols} がありません。見つかった列: {list(df.columns)}")
    st.stop()

t = df["time"].to_numpy()
a = df["abs"].to_numpy()
t_min, t_max = float(np.min(t)), float(np.max(t))

st.info(f"選択中: {selected.name}（N={len(df)}）")
st.write("出力先（自動）:", str(out_base_dir / f"{Path(selected).stem}_segments.csv"))


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
        marker=dict(size=4, color="rgba(0, 0, 128, 0.85)"),  # ←濃く
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
    xaxis_title="Time (s)",
    yaxis_title="Abs",
)
apply_white_theme(fig)  # ★追加

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
        out_base_dir = Path(st.session_state.step2_out_base_dir)
        out_base_dir.mkdir(parents=True, exist_ok=True)

        out_csv = out_base_dir / f"{Path(selected).stem}_segments.csv"
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
        fig2.add_trace(go.Scattergl(x=tt, y=aa, mode="markers", marker=dict(size=4, color="rgba(0, 0, 128, 0.85)"), name="selected"))

        for (s0, s1) in st.session_state.last_saved_segments_time:
            fig2.add_vrect(x0=s0, x1=s1, fillcolor="red", opacity=0.15, line_width=0)

        # 表示範囲は選択区間をちょい余白付きで自動ズーム
        z0, z1 = min(t0s), max(t1s)
        fig2.update_xaxes(range=[z0, z1])

        y0, y1 = float(np.min(aa)), float(np.max(aa))
        pad = 0.05 * (y1 - y0) if y1 > y0 else 1.0
        fig2.update_yaxes(range=[y0 - pad, y1 + pad])

        fig2.update_layout(
            height=420, 
            margin=dict(l=20, r=20, t=40, b=20),
            xaxis_title="Time (s)",
            yaxis_title="Abs",
        )
        apply_white_theme(fig2)  # ★追加

        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.info("選択範囲に点がありませんでした。")
