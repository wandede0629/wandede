r"""
补充实验 S1: SOH 随机森林的 permutation importance(复核 IC 主导是否稳健)

审稿人指出 impurity importance 对连续/相关特征有偏。这里用 permutation importance
(在未见测试电芯上, 多次重复)作稳健性复核, 并与 impurity importance 对照。
输出: 打印 + 图 supp_permutation_importance.png + 表 data/supp_perm_importance.csv
"""
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance

warnings.filterwarnings("ignore")
HERE = Path(__file__).parent
df = pd.read_csv(HERE / "data" / "severson_soh.csv")
FEATURES = ["chargetime", "IR", "Tavg", "Tmax", "Tmin",
            "frac_3p3", "frac_3p2", "frac_3p1", "frac_3p0", "frac_2p8", "v_median",
            "ic_peak", "v_ic_peak", "ic_fwhm", "ic_area"]
X = df[FEATURES].to_numpy(); y = df["SOH"].to_numpy(); g = df["cell"].to_numpy()

# 多种子取均值, 提高稳健性
n_seeds = 5
imp_perm = np.zeros((n_seeds, len(FEATURES)))
imp_mdi = np.zeros((n_seeds, len(FEATURES)))
for s in range(n_seeds):
    tr, te = next(GroupShuffleSplit(1, test_size=0.30, random_state=s).split(X, y, g))
    sc = StandardScaler().fit(X[tr])
    rf = RandomForestRegressor(300, min_samples_leaf=2, n_jobs=-1, random_state=0).fit(sc.transform(X[tr]), y[tr])
    imp_mdi[s] = rf.feature_importances_
    r = permutation_importance(rf, sc.transform(X[te]), y[te], n_repeats=10, random_state=0, n_jobs=-1)
    imp_perm[s] = r.importances_mean

perm_mean, perm_std = imp_perm.mean(0), imp_perm.std(0)
mdi_mean = imp_mdi.mean(0)
# 归一化 permutation importance 便于与 MDI 比例对照
perm_frac = perm_mean / perm_mean.sum()

res = pd.DataFrame({"feature": FEATURES, "MDI_impurity": mdi_mean,
                    "perm_importance": perm_mean, "perm_std": perm_std,
                    "perm_fraction": perm_frac}).sort_values("perm_importance", ascending=False)
res.to_csv(HERE / "data" / "supp_perm_importance.csv", index=False)

print("=" * 70)
print("SOH 特征重要性: permutation vs impurity (5 seeds, cell-level test)")
print("=" * 70)
print(f"{'feature':<12}{'MDI(impurity)':>14}{'perm(mean±std)':>20}{'perm_frac':>12}")
for _, r in res.iterrows():
    print(f"{r.feature:<12}{r.MDI_impurity:>14.3f}{r.perm_importance:>13.4f}±{r.perm_std:<5.4f}{r.perm_fraction:>12.3f}")

top = res.iloc[0]
print(f"\n最重要特征(permutation): {top.feature}, 占比 {top.perm_fraction:.2f}")
print(f"对照 impurity 中 ic_peak 占比: {mdi_mean[FEATURES.index('ic_peak')]:.2f}")

try:
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    r2 = res.sort_values("perm_importance")
    plt.figure(figsize=(7.4, 5.5))
    plt.barh(r2.feature, r2.perm_importance, xerr=r2.perm_std, color="#4c72b0")
    plt.xlabel("Permutation importance (drop in R², test cells)")
    plt.title("SOH permutation importance (5 seeds, mean ± s.d.)")
    plt.tight_layout(); plt.savefig(HERE / "supp_permutation_importance.png", dpi=300, bbox_inches="tight")
    print("\n图已保存: supp_permutation_importance.png")
except Exception as e:
    print(f"(画图跳过: {e})")
