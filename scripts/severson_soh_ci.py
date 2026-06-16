r"""
SOH 估计 · 论文级版本(IC 特征 + 置信区间)  —— 复现论文 A 的核心卖点

- 特征 = 11 基础 HI + 4 个 IC(增量容量 dQ/dV)特征
- 随机森林点预测 + 两种预测区间:
    (1) 树间分位数区间(自适应宽度)
    (2) 分割 conformal 校准(保证 ~90% 覆盖率)
- 指标: RMSPE / MAE / R²，以及区间质量 PICP(覆盖率) / MPIW(平均宽度)
- 画测试电芯 SOH 衰减曲线 + 90% 置信带

运行: .venv\Scripts\python.exe severson_soh_ci.py
"""
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import GroupShuffleSplit
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error

warnings.filterwarnings("ignore")
HERE = Path(__file__).parent
df = pd.read_csv(HERE / "data" / "severson_soh.csv")

BASE = ["chargetime", "IR", "Tavg", "Tmax", "Tmin",
        "frac_3p3", "frac_3p2", "frac_3p1", "frac_3p0", "frac_2p8", "v_median"]
IC = ["ic_peak", "v_ic_peak", "ic_fwhm", "ic_area"]
FEATURES = BASE + IC

X = df[FEATURES].to_numpy(); y = df["SOH"].to_numpy(); groups = df["cell"].to_numpy()


def rmspe(t, p):
    return float(np.sqrt(np.mean(((p - t) / t) ** 2)) * 100)


def split_by_cell(X, y, g, test_size, seed):
    tr, te = next(GroupShuffleSplit(1, test_size=test_size, random_state=seed).split(X, y, g))
    return tr, te


# 70% 训练芯 / 30% 测试芯；训练芯再分出校准集做 conformal
tr, te = split_by_cell(X, y, groups, 0.30, 0)
g_tr = groups[tr]
ptr_local, cal_local = split_by_cell(X[tr], y[tr], g_tr, 0.25, 1)   # 训练内部:正式训练/校准
ptr, cal = tr[ptr_local], tr[cal_local]

print("=" * 64)
print("SOH 估计 · 论文级 (IC 特征 + 置信区间)")
print(f"样本 {len(df)} | 特征 {len(FEATURES)} (含 {len(IC)} 个 IC) | "
      f"训练芯 {len(np.unique(groups[ptr]))} / 校准芯 {len(np.unique(groups[cal]))} / 测试芯 {len(np.unique(groups[te]))}")
print("=" * 64)

rf = RandomForestRegressor(n_estimators=400, min_samples_leaf=2, n_jobs=-1, random_state=0)
rf.fit(X[ptr], y[ptr])

# ---- 点预测 ----
def tree_preds(M, Xq):
    return np.stack([t.predict(Xq) for t in M.estimators_], axis=0)   # (n_trees, n)

p_te = rf.predict(X[te])
print(f"\n点预测:  RMSPE={rmspe(y[te], p_te):.3f}%   MAE={mean_absolute_error(y[te], p_te)*100:.3f}%SOH   R²={r2_score(y[te], p_te):.3f}")

# ---- 区间(1) 树间分位数 90% ----
tp_te = tree_preds(rf, X[te])
lo_q, hi_q = np.percentile(tp_te, 5, axis=0), np.percentile(tp_te, 95, axis=0)
picp_q = float(np.mean((y[te] >= lo_q) & (y[te] <= hi_q)) * 100)
mpiw_q = float(np.mean(hi_q - lo_q) * 100)

# ---- 区间(2) 分割 conformal(保证覆盖率) ----
res_cal = np.abs(y[cal] - rf.predict(X[cal]))
n = len(res_cal)
qlevel = min(1.0, np.ceil((n + 1) * 0.90) / n)        # 有限样本校正
qhat = float(np.quantile(res_cal, qlevel))
lo_c, hi_c = p_te - qhat, p_te + qhat
picp_c = float(np.mean((y[te] >= lo_c) & (y[te] <= hi_c)) * 100)
mpiw_c = float(np.mean(hi_c - lo_c) * 100)

print("\n90% 预测区间质量(目标覆盖率 90%):")
print(f"  {'方法':<26}{'PICP覆盖率(%)':>14}{'MPIW平均宽度(%SOH)':>20}")
print(f"  {'树间分位数':<24}{picp_q:>14.1f}{mpiw_q:>20.2f}")
print(f"  {'conformal(校准)':<22}{picp_c:>14.1f}{mpiw_c:>20.2f}   (±{qhat*100:.2f}%SOH)")

# ---- 画 6 个测试电芯衰减曲线 + 置信带 ----
df_te = df.iloc[te].copy(); df_te["pred"] = p_te; df_te["lo"] = lo_c; df_te["hi"] = hi_c
try:
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cells = list(pd.unique(df_te["cell"]))[:6]
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    for ax, c in zip(axes.ravel(), cells):
        d = df_te[df_te.cell == c].sort_values("cycle")
        ax.fill_between(d.cycle, d.lo, d.hi, color="red", alpha=0.2, label="90% prediction interval")
        ax.plot(d.cycle, d.SOH, "k-", lw=2, label="True SOH")
        ax.plot(d.cycle, d.pred, "r.", ms=3, label="Predicted")
        ax.set_title(c, fontsize=8); ax.set_xlabel("cycle"); ax.set_ylabel("SOH"); ax.grid(alpha=0.3)
    axes.ravel()[0].legend(fontsize=7)
    fig.suptitle("SOH estimation: IC features + 90% split-conformal prediction interval")
    fig.tight_layout()
    png = HERE / "soh_confidence_curves.png"
    fig.savefig(png, dpi=300)
    print(f"\n置信带衰减曲线已保存: {png}")
except Exception as e:
    print(f"(画图跳过: {e})")

# ---- IC 特征重要性 ----
imp = sorted(zip(FEATURES, rf.feature_importances_), key=lambda x: -x[1])
print("\n特征重要性 Top-8:")
for name, v in imp[:8]:
    tag = " [IC]" if name in IC else ""
    print(f"  {name:<12}{v:.3f}{tag}")

print("\nSOH 论文级管线(IC+置信区间)运行成功 [OK]")
