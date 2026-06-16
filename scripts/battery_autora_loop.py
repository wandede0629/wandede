r"""
锂电池寿命预测 —— AutoRA「数据池主动学习」闭环
================================================

场景：公开数据集里有很多电芯/工况，但逐个揭示真实循环寿命很贵。
目标：用最少的揭示样本，训出一个好的「工况 -> 循环寿命」预测模型。

AutoRA 三件套在这里的角色：
  • experimentalist(实验家) : 从未揭示的数据池里，挑“模型最没把握”的工况(高斯过程不确定度) -> 主动学习
  • experiment_runner(执行器): 揭示这些工况的真实寿命(= 查公开数据集)
  • theorist(理论家)        : 用已揭示数据拟合高斯过程寿命预测模型

每轮揭示一小批，观察测试集 R² 如何随揭示样本数上升。
并与“随机揭示”基线对比，证明主动学习更省样本。

运行: .venv\Scripts\python.exe battery_autora_loop.py

────────────────────────────────────────────────────────────
换成真实公开数据集的方法（只需改“读数据”这一步）：
  Severson(MIT-Stanford) 124 LFP 电芯快充寿命数据:
    https://data.matr.io/1/   (论文: Severson et al., Nature Energy 2019)
  NASA PCoE 电池老化数据:
    https://www.nasa.gov/intelligent-systems-division (PCoE Datasets)
  CALCE: https://calce.umd.edu/battery-data
  把它们整理成含列 [特征..., cycle_life] 的 CSV，替换下面的 CSV 路径与
  FEATURES 列名即可，其余闭环逻辑无需改动。
────────────────────────────────────────────────────────────
"""
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

from autora.variable import Variable, VariableCollection
from autora.state import StandardState, on_state, Delta

warnings.filterwarnings("ignore")  # 静音 GP 收敛提示

# ============== 0) 读取“公开数据集” ==============
# 用法: python battery_autora_loop.py [数据集.csv]
#   默认用合成占位数据集; 传入 data\severson_cycle_life.csv 即可跑真实 Severson 数据。
import sys
TARGET = "cycle_life"
CSV = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent / "data" / "battery_aging.csv"
if not CSV.exists():
    raise SystemExit(f"找不到数据集 {CSV}\n  合成数据: python battery_make_dataset.py\n  真实数据: python severson_download.py && python severson_parse.py")
data = pd.read_csv(CSV)

# 自动识别特征列(除 cycle_life 外的全部数值列)，对任意数据集通用
FEATURES = [c for c in data.columns if c != TARGET and pd.api.types.is_numeric_dtype(data[c])]
data = data[FEATURES + [TARGET]].dropna().reset_index(drop=True)

rng = np.random.default_rng(0)
data = data.sample(frac=1.0, random_state=0).reset_index(drop=True)

# 切分尺寸按数据集大小自适应(真实 Severson 只有 ~80-120 电芯)
N = len(data)
N_TEST = max(20, int(N * 0.30))
SEED_N = min(6, N - N_TEST - 1)
BATCH = max(3, N_TEST // 12)                 # 每轮揭示多少条
ROUNDS = 12                                  # 揭示轮数

TEST = data.iloc[:N_TEST].reset_index(drop=True)
POOL_INIT = data.iloc[N_TEST:].reset_index(drop=True)    # 全部未揭示数据(含真值，供 runner 查询)
SEED = POOL_INIT.iloc[:SEED_N].copy()
POOL = POOL_INIT.iloc[SEED_N:].reset_index(drop=True)


# ============== 1) AutoRA 变量空间 ==============
variables = VariableCollection(
    independent_variables=[Variable(name=f) for f in FEATURES],
    dependent_variables=[Variable(name=TARGET)],
)


# ============== 2) Theorist：随机森林寿命模型(对异质/小数据稳健) ==============
# 注:真实多批次数据存在“批次效应”，高斯过程在小样本上会灾难性外推(R²出现负值)。
#     随机森林不会无限外推，且可用“树间方差”作为主动学习的不确定度。
from sklearn.ensemble import RandomForestRegressor


class LifeModel:
    """对 log(寿命) 建随机森林，提供预测均值与(树间方差)不确定度。"""
    def __init__(self):
        self.scaler = StandardScaler()
        self.rf = RandomForestRegressor(n_estimators=300, min_samples_leaf=1,
                                        random_state=0, n_jobs=-1)

    def fit(self, df):
        X = self.scaler.fit_transform(df[FEATURES].to_numpy())
        y = np.log(df[TARGET].to_numpy())
        self.rf.fit(X, y)
        return self

    def predict(self, df, return_std=False):
        X = self.scaler.transform(df[FEATURES].to_numpy())
        per_tree = np.stack([t.predict(X) for t in self.rf.estimators_], axis=0)
        mu = per_tree.mean(0)
        std = per_tree.std(0)                 # 树间分歧 = 不确定度
        life = np.exp(mu)
        return (life, std) if return_std else life


@on_state()
def theorist(experiment_data):
    return Delta(models=[LifeModel().fit(experiment_data)])


# ============== 3) Experimentalist：按不确定度从数据池选工况 ==============
@on_state()
def experimentalist_active(models):
    if POOL.empty:
        return Delta(conditions=POOL.head(0)[FEATURES])
    if not models:                                   # 还没有模型 -> 随机起步
        idx = rng.choice(len(POOL), min(BATCH, len(POOL)), replace=False)
    else:
        _, std = models[-1].predict(POOL, return_std=True)   # 模型最没把握的点
        idx = np.argsort(std)[-BATCH:]
    return Delta(conditions=POOL.iloc[idx][FEATURES].reset_index(drop=True),
                 _picked_index=POOL.index[idx].tolist())


@on_state()
def experimentalist_random(models):
    n = min(BATCH, len(POOL))
    idx = rng.choice(len(POOL), n, replace=False) if n else []
    return Delta(conditions=POOL.iloc[idx][FEATURES].reset_index(drop=True),
                 _picked_index=POOL.index[list(idx)].tolist())


# ============== 4) Experiment Runner：揭示真实寿命(=查数据集) ==============
@on_state()
def experiment_runner(conditions):
    # conditions 来自 POOL，按特征匹配查真值
    revealed = POOL.merge(conditions, on=FEATURES, how="inner").drop_duplicates(FEATURES)
    return Delta(experiment_data=revealed[FEATURES + [TARGET]])


def reveal_and_shrink_pool(picked_index):
    """把已揭示的行从全局数据池移除（避免重复揭示）。"""
    global POOL
    POOL = POOL.drop(index=picked_index, errors="ignore").reset_index(drop=True)


# ============== 5) 跑闭环（主动学习 vs 随机基线） ==============
def run(strategy_name, experimentalist):
    global POOL
    POOL = POOL_INIT.iloc[SEED_N:].reset_index(drop=True)  # 重置池
    state = StandardState(variables=variables, experiment_data=SEED.copy())
    state = theorist(state)
    curve = []
    for r in range(ROUNDS):
        state = experimentalist(state)
        picked = getattr(state, "_picked_index", None)
        if state.conditions is None or len(state.conditions) == 0:
            break
        state = experiment_runner(state)
        if picked is not None:
            reveal_and_shrink_pool(picked)
        state = theorist(state)
        pred = state.models[-1].predict(TEST)
        r2 = r2_score(TEST[TARGET], pred)
        n_labeled = len(state.experiment_data)
        curve.append((n_labeled, r2))
        print(f"  [{strategy_name}] 第{r+1:2d}轮  已揭示样本={n_labeled:3d}  测试R²={r2:.3f}")
    return state, curve


print("=" * 60)
print("AutoRA 锂电池寿命预测 · 数据池主动学习")
print(f"数据集: {CSV.name}  特征: {FEATURES}")
print(f"共 {len(data)} 条 | 测试集 {len(TEST)} | 初始种子 {len(SEED)} | 每轮揭示 {BATCH}")
print("=" * 60)

print("\n>>> 策略 A：主动学习(不确定度采样)")
state_active, curve_a = run("主动", experimentalist_active)

print("\n>>> 策略 B：随机揭示(基线)")
_, curve_b = run("随机", experimentalist_random)

# ============== 6) 结果对比 + 保存学习曲线 ==============
print("\n" + "=" * 60)
final_active = curve_a[-1][1]
final_random = curve_b[-1][1]
print(f"相同揭示预算下的最终测试 R²:  主动学习={final_active:.3f}   随机={final_random:.3f}")
print(f"主动学习相对随机的提升: {(final_active - final_random):+.3f}")

# 用最终模型看几条预测 vs 真值
pred = state_active.models[-1].predict(TEST)
print("\n最终模型在测试集上的预测示例:")
demo = TEST[FEATURES].copy()
demo["真实寿命"] = TEST[TARGET].values.round(0)
demo["预测寿命"] = pred.round(0)
print(demo.head(6).to_string(index=False))

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.figure(figsize=(7, 4.5))
    plt.plot(*zip(*curve_a), "o-", label="Active learning (uncertainty)")
    plt.plot(*zip(*curve_b), "s--", label="Random baseline")
    plt.xlabel("# revealed samples")
    plt.ylabel("Test R^2  (cycle-life prediction)")
    plt.title("AutoRA pool-based active learning: Li-ion cycle life")
    plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
    png = Path(__file__).parent / "battery_learning_curve.png"
    plt.savefig(png, dpi=120)
    print(f"\n学习曲线已保存: {png}")
except Exception as e:
    print(f"(画图跳过: {e})")

print("\nAutoRA 锂电池闭环运行成功 [OK]")
