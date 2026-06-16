r"""
补充实验 S6: 主动学习失效的根因诊断
回应"主动学习无效是池子同质 还是 不确定度估计不可靠?"

(1) RF 不确定度(树间标准差) vs 真实绝对误差 的 Spearman 相关
    -> 低相关 = 不确定度本身无信息量, 难怪 AL 无效
(2) 换高斯过程(自带较好校准的不确定度)重做 AL vs random
    -> 若 GP-AL 仍≈random, 说明池子同质是主因; 若 GP-AL 显著更优, 说明是 RF 不确定度质量问题

数据: 同质 batches 1-2 (severson_b12.csv, 94 cells, 寿命预测), 12 seeds。
"""
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import spearmanr
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
from sklearn.metrics import r2_score

warnings.filterwarnings("ignore")
HERE = Path(__file__).parent
df = pd.read_csv(HERE / "data" / "severson_b12.csv")
F = [c for c in df.columns if c != "cycle_life"]
X = df[F].to_numpy(); y = np.log(df["cycle_life"].to_numpy())
N_SEEDS, SEED0, BATCH, ROUNDS, TEST = 12, 6, 3, 10, 0.30


def rf():
    return RandomForestRegressor(300, min_samples_leaf=2, n_jobs=-1, random_state=0)


def gp():
    k = ConstantKernel(1.0) * RBF([1.0] * X.shape[1]) + WhiteKernel(0.1)
    return GaussianProcessRegressor(kernel=k, normalize_y=True, n_restarts_optimizer=1, random_state=0)


# ---------- (1) 不确定度 vs 误差相关性 ----------
print("=" * 60)
print("(1) RF 树间不确定度 vs 真实绝对误差 的相关性")
print("=" * 60)
rho_rf, rho_gp = [], []
for s in range(N_SEEDS):
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.4, random_state=s)
    sc = StandardScaler().fit(Xtr)
    m = rf().fit(sc.transform(Xtr), ytr)
    pe = np.stack([t.predict(sc.transform(Xte)) for t in m.estimators_])
    std = pe.std(0); err = np.abs(pe.mean(0) - yte)
    rho_rf.append(spearmanr(std, err).correlation)
    g = gp().fit(sc.transform(Xtr), ytr)
    mu, gstd = g.predict(sc.transform(Xte), return_std=True)
    rho_gp.append(spearmanr(gstd, np.abs(mu - yte)).correlation)
print(f"  RF  不确定度-误差 Spearman = {np.nanmean(rho_rf):.3f} ± {np.nanstd(rho_rf):.3f}")
print(f"  GP  不确定度-误差 Spearman = {np.nanmean(rho_gp):.3f} ± {np.nanstd(rho_gp):.3f}")
print("  (越接近 1 = 不确定度越能指示真实误差; 接近 0 = 无信息量)")


# ---------- (2) GP 主动学习 vs 随机 ----------
def one_run(seed, strategy, model_kind):
    rng = np.random.default_rng(seed)
    n = len(X); idx = rng.permutation(n); nt = int(n * TEST)
    te = idx[:nt]; pool = list(idx[nt:])
    lab = [pool.pop(rng.integers(len(pool))) for _ in range(SEED0)]
    sc = StandardScaler().fit(X[idx[nt:]])
    Xte, yte = sc.transform(X[te]), y[te]
    for r in range(ROUNDS):
        mk = (gp() if model_kind == "gp" else rf()).fit(sc.transform(X[lab]), y[lab])
        if not pool:
            break
        Xp = sc.transform(X[pool])
        if strategy == "active":
            if model_kind == "gp":
                _, std = mk.predict(Xp, return_std=True)
            else:
                std = np.stack([t.predict(Xp) for t in mk.estimators_]).std(0)
            sel = np.argsort(std)[::-1][:BATCH]
        else:
            sel = rng.permutation(len(pool))[:BATCH]
        for j in sorted(sel, reverse=True):
            lab.append(pool.pop(j))
    return r2_score(yte, mk.predict(Xte))


print("\n" + "=" * 60)
print("(2) 高斯过程主动学习 vs 随机 (12 seeds, 最终预算)")
print("=" * 60)
for mk in ["gp"]:
    a = np.array([one_run(s, "active", mk) for s in range(N_SEEDS)])
    r = np.array([one_run(s, "random", mk) for s in range(N_SEEDS)])
    print(f"  GP-active R² = {a.mean():.3f} ± {a.std():.3f}   GP-random R² = {r.mean():.3f} ± {r.std():.3f}   diff = {a.mean()-r.mean():+.3f}")

print("\n结论将据上述两点综合判断(见汇报)。")
