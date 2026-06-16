r"""
补充实验 S5: 跨批次分布漂移量化(MMD)与可迁移性的关系
回应"跨批次崩塌为什么发生、是否只是 batch3 协议不同"的质疑,
并给出一个【可操作的警戒阈值】。

做法:
  - 9 个格式无关特征, 标准化(在全体上拟合)。
  - 对每一对批次 (i, j) 计算 RBF-MMD(分布距离, 中位带宽启发式)。
  - 对每个有序对 train=i -> test=j, 用 RF 拟合批次 i, 在批次 j 上测 R²(log 寿命)。
  - 给出 MMD vs 迁移 R² 的关系与阈值。
输出: 打印 + 图 supp_drift_mmd.png + 表 data/supp_drift_mmd.csv
"""
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from itertools import product
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.metrics.pairwise import rbf_kernel, euclidean_distances

warnings.filterwarnings("ignore")
HERE = Path(__file__).parent
df = pd.read_csv(HERE / "data" / "severson_features_full.csv")
FEAT = ["var_dQ", "min_dQ", "mean_dQ", "qd_slope", "qd_intercept",
        "qd2", "qd_max_minus_qd2", "chargetime_5", "temp_integral"]
df = df.dropna(subset=FEAT + ["cycle_life", "batch"]).reset_index(drop=True)
BATCH = {"2017-05-12": "b1", "2017-06-30": "b2", "2018-04-12": "b3", "2019-01-24": "b4"}
df["b"] = df["batch"].map(BATCH)
order = ["b1", "b2", "b3", "b4"]

sc = StandardScaler().fit(df[FEAT])
Xs = sc.transform(df[FEAT])
y = np.log(df["cycle_life"].to_numpy())
Z = {b: Xs[df.b.values == b] for b in order}
Y = {b: y[df.b.values == b] for b in order}

# 中位带宽启发式(在全体成对距离上)
d = euclidean_distances(Xs)
gamma = 1.0 / (2.0 * np.median(d[d > 0]) ** 2)


def mmd2(A, B):
    Kaa, Kbb, Kab = rbf_kernel(A, A, gamma), rbf_kernel(B, B, gamma), rbf_kernel(A, B, gamma)
    m, n = len(A), len(B)
    saa = (Kaa.sum() - np.trace(Kaa)) / (m * (m - 1))
    sbb = (Kbb.sum() - np.trace(Kbb)) / (n * (n - 1))
    return float(saa + sbb - 2 * Kab.mean())


# MMD 矩阵
M = pd.DataFrame(index=order, columns=order, dtype=float)
for i, j in product(order, order):
    M.loc[i, j] = 0.0 if i == j else np.sqrt(max(mmd2(Z[i], Z[j]), 0))

print("=" * 60)
print("跨批次分布距离 MMD (RBF, 中位带宽) 与迁移 R²")
print("=" * 60)
print("\nMMD 矩阵(行=train, 列=test):")
print(M.round(3).to_string())

# 迁移 R²:train 单批次 -> test 另一批次
rows = []
print(f"\n{'train->test':<14}{'MMD':>8}{'transfer R²':>14}")
for i, j in product(order, order):
    if i == j:
        continue
    m = RandomForestRegressor(300, min_samples_leaf=2, n_jobs=-1, random_state=0).fit(Z[i], Y[i])
    r2 = r2_score(Y[j], m.predict(Z[j]))
    rows.append({"train": i, "test": j, "mmd": M.loc[i, j], "r2": r2})
    print(f"{i+'->'+j:<14}{M.loc[i,j]:>8.3f}{r2:>14.3f}")

res = pd.DataFrame(rows)
res.to_csv(HERE / "data" / "supp_drift_mmd.csv", index=False)
corr = res["mmd"].corr(res["r2"], method="spearman")
# 阈值:能把 R²>=0 与 R²<0 分开的 MMD
pos = res[res.r2 >= 0]["mmd"]; neg = res[res.r2 < 0]["mmd"]
thr = (pos.max() + neg.min()) / 2 if len(pos) and len(neg) else np.nan
print(f"\nSpearman(MMD, R²) = {corr:.3f}")
print(f"R²>=0 的最大 MMD = {pos.max():.3f}; R²<0 的最小 MMD = {neg.min():.3f}")
print(f"=> 经验警戒阈值: MMD > {thr:.2f} 时, 单批次直接迁移 R² 转为负。")

try:
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    im = ax1.imshow(M.values.astype(float), cmap="viridis")
    ax1.set_xticks(range(4)); ax1.set_xticklabels(order); ax1.set_yticks(range(4)); ax1.set_yticklabels(order)
    for i in range(4):
        for j in range(4):
            ax1.text(j, i, f"{M.values[i,j]:.2f}", ha="center", va="center", color="w", fontsize=9)
    ax1.set_title("(a) Cross-batch distribution distance (MMD)"); fig.colorbar(im, ax=ax1, fraction=0.046)
    ax2.scatter(res.mmd, res.r2, s=60, edgecolor="k")
    for _, r in res.iterrows():
        ax2.annotate(f"{r.train}->{r.test}", (r.mmd, r.r2), fontsize=7, xytext=(3, 3), textcoords="offset points")
    if np.isfinite(thr):
        ax2.axvline(thr, color="r", ls="--", lw=1, label=f"threshold MMD={thr:.2f}")
    ax2.axhline(0, color="0.5", ls=":")
    ax2.set_xlabel("MMD (train vs test batch)"); ax2.set_ylabel("Transfer R² (log cycle life)")
    ax2.set_title("(b) Larger drift -> worse transfer"); ax2.legend(fontsize=8); ax2.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(HERE / "supp_drift_mmd.png", dpi=300)
    print("\n图已保存: supp_drift_mmd.png")
except Exception as e:
    print(f"(画图跳过: {e})")
