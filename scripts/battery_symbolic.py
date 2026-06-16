r"""
符号回归理论家 —— 让 AutoRA 自动“发现”锂电池寿命的可解释数学公式。

不同于随机森林/高斯过程(黑箱)，符号回归(遗传编程)直接演化出一个
人类可读的数学表达式:  log(cycle_life) = f(特征...)

工具: gplearn(遗传编程符号回归)
对照: Severson 论文的“方差模型”  log10(寿命) ≈ a + b·log10(var ΔQ)

运行: .venv\Scripts\python.exe battery_symbolic.py [数据集.csv]
"""
import sys
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from gplearn.genetic import SymbolicRegressor
from gplearn.functions import _Function
import sympy as sp

from autora.variable import Variable, VariableCollection
from autora.state import StandardState, on_state, Delta

warnings.filterwarnings("ignore")

# ============== 0) 数据 ==============
TARGET = "cycle_life"
CSV = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent / "data" / "severson_features.csv"
data = pd.read_csv(CSV)
FEATURES = [c for c in data.columns if c != TARGET and pd.api.types.is_numeric_dtype(data[c])]
data = data[FEATURES + [TARGET]].dropna().reset_index(drop=True)
print(f"数据集: {CSV.name} | {len(data)} 电芯 | 特征: {FEATURES}")

X = data[FEATURES].to_numpy()
y = np.log(data[TARGET].to_numpy())          # 在 log(寿命) 空间发现公式
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0)


# ============== 1) gplearn 程序树 -> sympy 可读公式 ==============
def program_to_sympy(program, feature_names):
    syms = sp.symbols(feature_names)
    it = iter(program)

    def build():
        node = next(it)
        if isinstance(node, _Function):
            a = [build() for _ in range(node.arity)]
            n = node.name
            return {
                "add": lambda: a[0] + a[1], "sub": lambda: a[0] - a[1],
                "mul": lambda: a[0] * a[1], "div": lambda: a[0] / a[1],
                "sqrt": lambda: sp.sqrt(sp.Abs(a[0])), "log": lambda: sp.log(sp.Abs(a[0])),
                "abs": lambda: sp.Abs(a[0]), "neg": lambda: -a[0], "inv": lambda: 1 / a[0],
            }[n]()
        if isinstance(node, int):
            return syms[node]
        return sp.Float(round(float(node), 4))

    return build()


# ============== 2) Theorist：符号回归 ==============
class SymbolicTheorist:
    def __init__(self):
        self.sr = SymbolicRegressor(
            population_size=5000, generations=40,
            function_set=("add", "sub", "mul", "div", "sqrt", "log", "neg", "inv"),
            metric="rmse", parsimony_coefficient=0.002,
            const_range=(-6.0, 6.0), p_crossover=0.7, p_subtree_mutation=0.12,
            p_hoist_mutation=0.05, p_point_mutation=0.1,
            max_samples=0.9, verbose=0, random_state=0, n_jobs=-1,
        )

    def fit(self, Xtr, ytr):
        self.sr.fit(Xtr, ytr)
        return self

    def predict(self, X):
        return self.sr.predict(X)

    def formula(self):
        try:
            expr = program_to_sympy(self.sr._program.program, FEATURES)
            return sp.simplify(expr)
        except Exception:
            return str(self.sr._program)


@on_state()
def symbolic_theorist(experiment_data):
    Xt = experiment_data[FEATURES].to_numpy()
    yt = np.log(experiment_data[TARGET].to_numpy())
    return Delta(models=[SymbolicTheorist().fit(Xt, yt)])


# ============== 3) 用 AutoRA state 跑“发现” ==============
variables = VariableCollection(
    independent_variables=[Variable(name=f) for f in FEATURES],
    dependent_variables=[Variable(name=TARGET)],
)
train_df = pd.DataFrame(Xtr, columns=FEATURES)
train_df[TARGET] = np.exp(ytr)
state = StandardState(variables=variables, experiment_data=train_df)

print("\n正在用遗传编程演化寿命公式(约 25 代)...")
state = symbolic_theorist(state)
model = state.models[-1]

# ============== 4) 结果(符号回归发现结构 + 线性校准系数) ==============
def both_r2(y_log_true, y_log_pred, tag):
    r2_log = r2_score(y_log_true, y_log_pred)
    r2_cyc = r2_score(np.exp(y_log_true), np.exp(y_log_pred))
    print(f"  {tag}:  R²(log空间)={r2_log:.3f}   R²(寿命空间)={r2_cyc:.3f}")
    return r2_log

# 原始公式
raw_tr = model.predict(Xtr); raw_te = model.predict(Xte)
# 校准:用训练集对“公式输出”做线性标定 a·formula + b，修正系数/截距
cal = LinearRegression().fit(raw_tr.reshape(-1, 1), ytr)
cal_te = cal.predict(raw_te.reshape(-1, 1))

print("\n" + "=" * 64)
print("AutoRA 符号回归发现的寿命公式(结构):")
print("   f =", model.formula())
print(f"校准后:  log(cycle_life) = {cal.coef_[0]:.3f}·f + {cal.intercept_:.3f}")
print("=" * 64)
both_r2(yte, raw_te, "符号回归(原始公式)  ")
both_r2(yte, cal_te, "符号回归(线性校准后)")

# 对照:论文“方差模型”(只用 var_dQ 单特征线性)
if "var_dQ" in FEATURES:
    j = FEATURES.index("var_dQ")
    lr = LinearRegression().fit(Xtr[:, [j]], ytr)
    both_r2(yte, lr.predict(Xte[:, [j]]), "论文方差模型(仅var_dQ) ")
    print(f"     log(寿命) = {lr.coef_[0]:.3f}·var_dQ + {lr.intercept_:.3f}")

print("\n符号回归理论家运行成功 [OK]")
