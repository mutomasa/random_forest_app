# Random Forest Decision Boundary (Animation)

2次元特徴量データに対するランダムフォーレストの予測境界を、木の本数を増やしながらアニメーションで可視化する Streamlit アプリです。

## 機能
- 2つの特徴量を用いた2Dデータ（moons / circles / iris(2D)）の選択
- Plotlyで予測境界（決定領域）を描画
- 木の本数（n_estimators）を増やしていく過程をアニメーション表示
- 「木の数が増えると予測がどう安定するか」を直感的に観察

## ディレクトリ構成
```
random_forest_app/
├── pyproject.toml
├── README.md
├── uv.lock
└── src/
    └── app.py
```

## セットアップ（uv）
```bash
cd /home/mutomasa/research_engineering/mutomasa/random_forest_app
uv sync   # 依存関係を同期
```
不足がある場合:
```bash
uv add streamlit plotly scikit-learn numpy pandas
```

## 起動方法
```bash
# 推奨（プロジェクト環境で実行）
uv run streamlit run src/app.py --server.port 8502

# もしくは仮想環境を手動で有効化してから
# source .venv/bin/activate
# streamlit run src/app.py --server.port 8502
```
- ブラウザが自動で開かない場合は、コンソールのURL（例: http://localhost:8502）へアクセス。
- 既に別アプリが8501使用中のため、`--server.port 8502` の利用を推奨。

## 使い方
- サイドバーで設定:
  - データセット: moons / circles / iris(2D)
  - サンプル数（moons/circles）
  - ノイズ量（moons/circles）
  - 最大ツリー数、増分
- 画面下部中央の操作UI:
  - ▶ Play / ❚❚ Pause ボタンでアニメーション制御
  - スライダーで `n_estimators` を直接指定

## 実装のポイント
- コード: `src/app.py`
- 学習: `RandomForestClassifier`（scikit-leーン）
- 可視化: Plotly Contour + Scatter
- アニメーション: Plotlyの`frames`で木の本数ごとの境界を切替
- 前処理: `StandardScaler` による標準化
- UI配置: ボタン/スライダーはグラフ下部中央に配置（視認性改善）

## トラブルシューティング
- ModuleNotFoundError（例: plotly）
  - `uv sync` を実行、または `uv add plotly`
- 別環境で起動してしまう
  - `uv run ...` を使用し、必ずカレントを `random_forest_app` にして実行
- ポート競合
  - `--server.port 8502` など別ポートを指定
- Staticフォルダの警告
  - 本アプリは静的ファイル未使用のため無視して問題ありません

## ライセンス
MIT
