"""
生成一个【占位】锂电池老化数据集 -> data/battery_aging.csv

用物理规律(Arrhenius 温度加速 + C-rate 幂律 + DoD 幂律)模拟真实公开数据集的样子:
    实验条件 (c_rate, temperature_C, dod)  ->  循环寿命 cycle_life (到 80% SOH 的圈数)

⚠️ 这是合成数据，仅用于先把 AutoRA 闭环跑通。
   换成真实公开数据集(Severson/NASA/CALCE)的方法见 battery_autora_loop.py 顶部说明。
"""
import numpy as np
import pandas as pd
from pathlib import Path

rng = np.random.default_rng(42)

# ---- 真实(但对模型隐藏)的衰减规律 ----
R = 8.314          # 气体常数
Ea = 30000.0       # 活化能 J/mol -> 控制温度敏感度
Tref = 298.15      # 参考温度 25°C
C0 = 800.0         # 25°C / 1C / 满 DoD 下的基准寿命(圈)
p_crate = 0.85     # C-rate 越大寿命越短
q_dod = 1.5        # DoD 越深寿命越短


def true_cycle_life(c_rate, temp_C, dod):
    T_K = temp_C + 273.15
    arr = np.exp(Ea / R * (1.0 / T_K - 1.0 / Tref))   # 高温 -> 寿命短
    return C0 * arr * c_rate ** (-p_crate) * dod ** (-q_dod)


# ---- 在工况空间里采样一批“电芯/实验” ----
N = 400
c_rate = rng.uniform(0.5, 4.0, N)
temp_C = rng.uniform(15.0, 45.0, N)
dod = rng.uniform(0.4, 1.0, N)

life = true_cycle_life(c_rate, temp_C, dod)
life *= rng.lognormal(0.0, 0.08, N)        # 测量噪声(±8%)

df = pd.DataFrame({
    "c_rate": np.round(c_rate, 3),
    "temperature_C": np.round(temp_C, 2),
    "dod": np.round(dod, 3),
    "cycle_life": np.round(life, 1),
})

out = Path(__file__).parent / "data" / "battery_aging.csv"
out.parent.mkdir(exist_ok=True)
df.to_csv(out, index=False)
print(f"已生成数据集: {out}  ({len(df)} 条)")
print(df.head())
print(f"\n寿命范围: {df.cycle_life.min():.0f} ~ {df.cycle_life.max():.0f} 圈")
