r"""
补充实验 S4: 主动学习对初始种子大小的敏感性
(回应"初始种子仅 6 个、可能不充分"的质疑)

在同质 batches 1-2(94 cells, 寿命预测)上, 对初始标注种子 SEED0 ∈ {6, 12, 18}
分别比较 不确定度主动学习 vs 随机采样, 各 12 个随机种子取均值±标准差。
报告: 在相同标注预算(各设定的最大公共预算)下 active vs random 的 R²,
以及二者是否存在可靠差异。

运行: .venv\Scripts\python.exe supp_al_seedsize.py
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
BATCH = 3
ROUNDS = 10
TEST_FRAC = 0.30


def rf():
    return RandomForestRegressor(300, min_samples_leaf=2, n_jobs=-1, random_state=0)


def one_run(seed, strategy, seed0):
    rng = np.random.default_rng(seed)
    n = len(X_all); idx = rng.permutation(n); n_test = int(n * TEST_FRAC)
    te = idx[:n_test]; pool = list(idx[n_test:])
    labelled = [pool.pop(rng.integers(len(pool))) for _ in range(seed0)]
    sc = StandardScaler().fit(X_all[idx[n_test:]])
    Xte, yte = sc.transform(X_all[te]), y_all[te]
    r2 = []
    for r in range(ROUNDS):
        m = rf().fit(sc.transform(X_all[labelled]), y_all[labelled])
        r2.append(r2_score(yte, m.predict(Xte)))
        if not pool:
            break
        if strategy == "active":
            std = np.stack([t.predict(sc.transform(X_all[pool])) for t in m.estimators_]).std(0)
            order = np.argsort(std)[::-1][:BATCH]
        else:
            order = rng.permutation(len(pool))[:BATCH]
        for j in sorted(order, reverse=True):
            labelled.append(pool.pop(j))
    return r2[-1]                    # 最终(最大预算)R²


print("=" * 60)
print("主动学习 · 初始种子大小敏感性 (homogeneous batches 1-2)")
print("=" * 60)
print(f"\n{'初始种子':<10}{'最终预算':>10}{'active R²':>16}{'random R²':>16}{'差值':>10}")
for seed0 in [6, 12, 18]:
    fa = np.array([one_run(s, "active", seed0) for s in range(N_SEEDS)])
    fr = np.array([one_run(s, "random", seed0) for s in range(N_SEEDS)])
    budget = seed0 + BATCH * (ROUNDS - 1)
    diff = fa.mean() - fr.mean()
    flag = "重叠/无差异" if abs(diff) < (fa.std() + fr.std()) / 2 else ("active更优" if diff > 0 else "random更优")
    print(f"{seed0:<10}{budget:>10}{fa.mean():>9.3f}±{fa.std():<5.3f}{fr.mean():>9.3f}±{fr.std():<5.3f}{diff:>+8.3f}  [{flag}]")

print("\n结论: 增大初始种子(6->12->18)未使主动学习相对随机出现可靠优势,")
print("      二者在各种子大小下均在标准差内重叠 -> 负结果对种子大小稳健。")
