r"""
主动学习样本效率实验(规范版, 供 Figure 6)
- 同质数据 batches 1-2(severson_b12.csv, 94 cells, 寿命预测)
- 池式主动学习, RF 代理, 不确定度 = 树间标准差
- 多种子(N_SEEDS)重复, 报告 mean ± s.d.
- 与随机采样在相同标注预算下对比

运行: .venv\Scripts\python.exe active_learning_experiment.py
"""
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

warnings.filterwarnings("ignore")
HERE = Path(__file__).parent
df = pd.read_csv(HERE / "data" / "severson_b12.csv")
FEATURES = [c for c in df.columns if c != "cycle_life"]
X_all = df[FEATURES].to_numpy()
y_all = np.log(df["cycle_life"].to_numpy())

N_SEEDS = 12
SEED0 = 6           # 初始标注电芯数
BATCH = 3           # 每轮揭示
ROUNDS = 12
TEST_FRAC = 0.30


def rf():
    return RandomForestRegressor(n_estimators=300, min_samples_leaf=2, n_jobs=-1, random_state=0)


def one_run(seed, strategy):
    rng = np.random.default_rng(seed)
    n = len(X_all)
    idx = rng.permutation(n)
    n_test = int(n * TEST_FRAC)
    te = idx[:n_test]
    pool = list(idx[n_test:])
    labelled = [pool.pop(rng.integers(len(pool))) for _ in range(SEED0)]
    sc = StandardScaler().fit(X_all[idx[n_test:]])  # 在全部池特征上拟合标准化(不含测试)
    Xte = sc.transform(X_all[te]); yte = y_all[te]
    counts, r2s = [], []
    for r in range(ROUNDS):
        m = rf().fit(sc.transform(X_all[labelled]), y_all[labelled])
        r2s.append(r2_score(yte, m.predict(Xte))); counts.append(len(labelled))
        if not pool:
            break
        if strategy == "active":
            Xp = sc.transform(X_all[pool])
            std = np.stack([t.predict(Xp) for t in m.estimators_]).std(0)
            order = np.argsort(std)[::-1][:BATCH]
        else:
            order = rng.permutation(len(pool))[:BATCH]
        for j in sorted(order, reverse=True):
            labelled.append(pool.pop(j))
    return counts, r2s


def aggregate(strategy):
    curves = [one_run(s, strategy) for s in range(N_SEEDS)]
    L = min(len(c[0]) for c in curves)
    counts = curves[0][0][:L]
    mat = np.array([c[1][:L] for c in curves])
    return counts, mat.mean(0), mat.std(0)


print("=" * 60)
print(f"主动学习样本效率 (homogeneous batches 1-2, {len(df)} cells, {N_SEEDS} seeds)")
print("=" * 60)
ca, ma, sa = aggregate("active")
cr, mr, sr = aggregate("random")
print(f"\n{'#cells':>8}{'active R2':>16}{'random R2':>16}")
for i in range(len(ca)):
    print(f"{ca[i]:>8}{ma[i]:>10.3f}±{sa[i]:<4.2f}{mr[i]:>10.3f}±{sr[i]:<4.2f}")
print(f"\n最终({ca[-1]} cells): active {ma[-1]:.3f}±{sa[-1]:.2f}  vs  random {mr[-1]:.3f}±{sr[-1]:.2f}")

try:
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.figure(figsize=(7, 4.6))
    plt.plot(ca, ma, "o-", color="#1f77b4", label="Active learning (uncertainty)")
    plt.fill_between(ca, ma - sa, ma + sa, color="#1f77b4", alpha=0.18)
    plt.plot(cr, mr, "s--", color="#ff7f0e", label="Random sampling")
    plt.fill_between(cr, mr - sr, mr + sr, color="#ff7f0e", alpha=0.18)
    plt.xlabel("# labelled cells (revealed)"); plt.ylabel("Test R² (cycle-life)")
    plt.title(f"Active learning vs random ({N_SEEDS} seeds, mean ± s.d.)")
    plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(HERE / "battery_learning_curve.png", dpi=300)
    print("\nbattery_learning_curve.png 已更新")
except Exception as e:
    print(f"(画图跳过: {e})")
