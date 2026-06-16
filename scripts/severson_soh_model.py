r"""
SOH 估计 · 建模与评估  (复现三篇论文的公共主线)

- 按【电芯】划分训练/测试(测试电芯完全未见，杜绝同芯循环泄漏)—— 关键规范
- 三模型对比: 岭回归(论文B) / SVR-RBF(论文C) / 随机森林(论文A)
- 指标: RMSPE(%)、MAE(%SOH)、R²
- 画几个测试电芯的 SOH 衰减曲线(真值 vs 预测)

运行: .venv\Scripts\python.exe severson_soh_model.py
"""
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error

warnings.filterwarnings("ignore")
HERE = Path(__file__).parent
df = pd.read_csv(HERE / "data" / "severson_soh.csv")

FEATURES = ["chargetime", "IR", "Tavg", "Tmax", "Tmin",
            "frac_3p3", "frac_3p2", "frac_3p1", "frac_3p0", "frac_2p8", "v_median"]
X = df[FEATURES].to_numpy()
y = df["SOH"].to_numpy()
groups = df["cell"].to_numpy()

# 按电芯划分 70/30
gss = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=0)
tr, te = next(gss.split(X, y, groups))
print("=" * 60)
print("SOH 估计 (Severson 公共数据, 复现三篇论文主线)")
print(f"样本 {len(df)} | 特征 {len(FEATURES)} | 训练电芯 {len(np.unique(groups[tr]))} / 测试电芯 {len(np.unique(groups[te]))}")
print("=" * 60)

sc = StandardScaler().fit(X[tr])
Xtr, Xte = sc.transform(X[tr]), sc.transform(X[te])
ytr, yte = y[tr], y[te]


def rmspe(t, p):
    return float(np.sqrt(np.mean(((p - t) / t) ** 2)) * 100)


models = {
    "岭回归 Ridge (论文B)": Ridge(alpha=1.0),
    "SVR-RBF (论文C)": SVR(C=10, gamma="scale", epsilon=0.005),
    "随机森林 RF (论文A)": RandomForestRegressor(n_estimators=200, n_jobs=-1, random_state=0),
}
preds = {}
print(f"\n{'模型':<22}{'RMSPE(%)':>10}{'MAE(%SOH)':>12}{'R²':>8}")
print("-" * 52)
for name, m in models.items():
    m.fit(Xtr, ytr)
    p = m.predict(Xte)
    preds[name] = p
    print(f"{name:<22}{rmspe(yte, p):>10.3f}{mean_absolute_error(yte, p) * 100:>12.3f}{r2_score(yte, p):>8.3f}")

# ---- 画测试电芯的 SOH 衰减曲线(用最好的 RF) ----
best = "随机森林 RF (论文A)"
df_te = df.iloc[te].copy()
df_te["pred"] = preds[best]
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cells = list(pd.unique(df_te["cell"]))[:6]
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    for ax, c in zip(axes.ravel(), cells):
        d = df_te[df_te.cell == c].sort_values("cycle")
        ax.plot(d.cycle, d.SOH, "k-", lw=2, label="True SOH")
        ax.plot(d.cycle, d.pred, "r.", ms=4, label="Predicted")
        ax.set_title(c, fontsize=8); ax.set_xlabel("cycle"); ax.set_ylabel("SOH")
        ax.grid(alpha=0.3)
    axes.ravel()[0].legend(fontsize=8)
    fig.suptitle(f"SOH estimation on unseen test cells ({best})")
    fig.tight_layout()
    png = HERE / "soh_degradation_curves.png"
    fig.savefig(png, dpi=300)
    print(f"\n衰减曲线已保存: {png}")
except Exception as e:
    print(f"(画图跳过: {e})")

print("\nSOH 估计管线运行成功 [OK]")
