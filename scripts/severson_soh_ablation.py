r"""
SOH 估计 · 特征消融实验(论文用)

逐组累加特征，看每组贡献(按电芯划分，5 个随机种子取均值±标准差):
  S1 基础HI        : chargetime, IR, Tavg, Tmax, Tmin
  S2 +曲线形状     : + frac@电压阈值, v_median
  S3 +IC增量容量   : + ic_peak, v_ic_peak, ic_fwhm, ic_area  (= 完整)

运行: .venv\Scripts\python.exe severson_soh_ablation.py
"""
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error

warnings.filterwarnings("ignore")
df = pd.read_csv(Path(__file__).parent / "data" / "severson_soh.csv")

S1 = ["chargetime", "IR", "Tavg", "Tmax", "Tmin"]
SHAPE = ["frac_3p3", "frac_3p2", "frac_3p1", "frac_3p0", "frac_2p8", "v_median"]
IC = ["ic_peak", "v_ic_peak", "ic_fwhm", "ic_area"]
SETS = {
    "S1 基础HI(5)": S1,
    "S2 +曲线形状(11)": S1 + SHAPE,
    "S3 +IC 完整(15)": S1 + SHAPE + IC,
}
y = df["SOH"].to_numpy(); g = df["cell"].to_numpy()


def rmspe(t, p):
    return np.sqrt(np.mean(((p - t) / t) ** 2)) * 100


print("=" * 60)
print("SOH 特征消融 (随机森林, 按电芯划分, 5 种子均值±std)")
print("=" * 60)
print(f"\n{'特征集':<18}{'RMSPE(%)':>14}{'MAE(%SOH)':>14}{'R²':>12}")
print("-" * 58)
for name, feats in SETS.items():
    X = df[feats].to_numpy()
    rs, ms, r2s = [], [], []
    for seed in range(5):
        tr, te = next(GroupShuffleSplit(1, test_size=0.30, random_state=seed).split(X, y, g))
        sc = StandardScaler().fit(X[tr])
        m = RandomForestRegressor(300, min_samples_leaf=2, n_jobs=-1, random_state=0)
        m.fit(sc.transform(X[tr]), y[tr])
        p = m.predict(sc.transform(X[te]))
        rs.append(rmspe(y[te], p)); ms.append(mean_absolute_error(y[te], p) * 100); r2s.append(r2_score(y[te], p))
    print(f"{name:<18}{np.mean(rs):>8.3f}±{np.std(rs):<4.2f}{np.mean(ms):>9.3f}±{np.std(ms):<4.2f}"
          f"{np.mean(r2s):>8.3f}±{np.std(r2s):<4.2f}")

print("\n结论: 加入曲线形状与 IC 特征对精度的边际贡献;IC 主要把预测力集中到强物理特征上。")
print("\n消融实验运行成功 [OK]")
