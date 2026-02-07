# Streamlit App for Analyzing Decay Curve

研究室内向けの解析アプリ（Streamlit + Dockerで作成）

## Features
- Step1: 時系列データ変換
- Step2: カーブの選択
- Step3: 速度論モデルでフィッティング（k1, k2の算出）
- Step4: パラメーターの集約
- Step5: 可視光強度の積分値算出
- Step6: 速度定数（kt1, kt2, kp1, kp2）の算出

## How to Run
```bash
docker compose up --build
```

## Open
http://localhost:8501

