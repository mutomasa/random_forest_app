import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.datasets import make_moons, make_circles, load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def make_dataset(name: str, n_samples: int = 500, noise: float = 0.2):
    if name == "moons":
        X, y = make_moons(n_samples=n_samples, noise=noise, random_state=42)
        feat_names = ["x1", "x2"]
    elif name == "circles":
        X, y = make_circles(n_samples=n_samples, noise=noise, factor=0.5, random_state=42)
        feat_names = ["x1", "x2"]
    else:  # iris (2D選択)
        iris = load_iris()
        X = iris.data[:, :2]  # sepal length/width
        y = iris.target
        feat_names = iris.feature_names[:2]
    return pd.DataFrame(X, columns=feat_names), pd.Series(y, name="target")


def fit_rf_frames(X: np.ndarray, y: np.ndarray, n_estimators_list):
    frames = []
    for n in n_estimators_list:
        clf = RandomForestClassifier(n_estimators=n, max_depth=None, random_state=42)
        clf.fit(X, y)
        frames.append((n, clf))
    return frames


def decision_surface(clf, X, y, resolution: int = 200):
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, resolution),
        np.linspace(y_min, y_max, resolution),
    )
    grid = np.c_[xx.ravel(), yy.ravel()]
    Z = clf.predict(grid).reshape(xx.shape)

    fig = go.Figure()
    fig.add_trace(
        go.Contour(
            x=np.linspace(x_min, x_max, resolution),
            y=np.linspace(y_min, y_max, resolution),
            z=Z,
            colorscale=[[0, "#8ecae6"], [0.5, "#8ecae6"], [0.5, "#ffb703"], [1, "#ffb703"]],
            showscale=False,
            opacity=0.4,
        )
    )
    # 元データ
    for cls in np.unique(y):
        pts = X[y == cls]
        fig.add_trace(
            go.Scatter(
                x=pts[:, 0],
                y=pts[:, 1],
                mode="markers",
                name=f"class {cls}",
            )
        )
    fig.update_layout(
        xaxis_title="x1",
        yaxis_title="x2",
        margin=dict(l=0, r=0, t=30, b=120),  # 下部にコントロール用スペース
        legend=dict(orientation="h", y=1.04, x=0.5, xanchor="center"),
    )
    return fig


def main():
    st.set_page_config(page_title="RandomForest Decision Boundary (Animation)", layout="wide")
    st.title("🌲 Random Forest: 予測境界のアニメーション")

    with st.sidebar:
        st.subheader("データセット/特徴量")
        ds = st.selectbox("データセット", ["moons", "circles", "iris(2D)"], index=0)
        n_samples = st.slider("サンプル数 (moons/circles)", 100, 2000, 500, 50)
        noise = st.slider("ノイズ (moons/circles)", 0.0, 0.5, 0.2, 0.05)

        st.subheader("ランダムフォーレスト")
        max_trees = st.slider("最大ツリー数", 1, 200, 100, 1)
        step = st.slider("増分", 1, 20, 5, 1)

    # データ生成/取得
    if ds in ("moons", "circles"):
        Xdf, yser = make_dataset(ds, n_samples=n_samples, noise=noise)
    else:
        Xdf, yser = make_dataset("iris")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(Xdf.values)
    y = yser.values

    # 学習しながらフレームを作成
    n_list = list(range(1, max_trees + 1, step))
    frames = fit_rf_frames(X_scaled, y, n_list)

    # アニメーション用フレーム
    anim_frames = []
    base_fig = None
    for n, clf in frames:
        fig = decision_surface(clf, X_scaled, y)
        fig.update_layout(title=f"Decision Boundary (n_estimators={n})")
        # 1フレームを画像化せず、Plotlyのフレームとして構成
        anim_frames.append(go.Frame(data=fig.data, name=str(n), layout=fig.layout))
        if base_fig is None:
            base_fig = fig

    if base_fig is None:
        st.warning("学習に失敗しました")
        return

    base_fig.update(frames=anim_frames)
    base_fig.update_layout(
        updatemenus=[
            {
                "type": "buttons",
                "showactive": True,
                "direction": "left",
                "x": 0.5,
                "y": -0.12,  # チャート下に配置
                "xanchor": "center",
                "yanchor": "top",
                "pad": {"r": 8, "t": 2},
                "bgcolor": "rgba(245,245,245,0.9)",
                "bordercolor": "#cccccc",
                "borderwidth": 1,
                "buttons": [
                    {
                        "label": "▶ Play",
                        "method": "animate",
                        "args": [None, {"fromcurrent": True, "frame": {"duration": 300, "redraw": True}, "transition": {"duration": 0}}],
                    },
                    {
                        "label": "❚❚ Pause",
                        "method": "animate",
                        "args": [[None], {"mode": "immediate", "frame": {"duration": 0, "redraw": False}, "transition": {"duration": 0}}],
                    },
                ],
            }
        ],
        sliders=[
            {
                "active": 0,
                "x": 0.5,
                "y": -0.20,  # ボタンのさらに下に配置
                "xanchor": "center",
                "len": 0.9,
                "pad": {"b": 0, "t": 0},
                "currentvalue": {"prefix": "n_estimators=", "visible": True},
                "steps": [
                    {"label": str(n), "method": "animate", "args": [[str(n)], {"frame": {"duration": 0, "redraw": True}, "transition": {"duration": 0}}]}
                    for n in n_list
                ],
            }
        ],
    )

    st.plotly_chart(base_fig, use_container_width=True)


if __name__ == "__main__":
    main()


