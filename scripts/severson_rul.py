r"""
RUL(剩余使用寿命)预测  —— 复现论文 B 主线:SOH 特征 -> 老化模型 -> RUL

定义:EOL(寿命终点) = SOH 首次降到 0.80。
      某循环的 RUL = EOL 循环 − 当前循环(还能用多少圈)。

特征 = 15 个 HI(含 IC) + 当前 SOH + 退化速率(SOH 局部斜率)
模型 = 随机森林 + conformal 置信区间(给出 RUL 的 ±区间)
划分 = 按电芯(测试芯未见)

运行: .venv\Scripts\python.exe severson_rul.py
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
df = pd.read_csv(HERE / "data" / "severson_soh.csv").sort_values(["cell", "cycle"]).reset_index(drop=True)

NOMINAL = 1.1            # 这批 LFP 电芯标称容量(Ah)
EOL_CAP = 0.80 * NOMINAL  # 寿命终点: 容量降到标称的 80% (= 0.88 Ah, Severson 自身定义)
HI = ["chargetime", "IR", "Tavg", "Tmax", "Tmin",
      "frac_3p3", "frac_3p2", "frac_3p1", "frac_3p0", "frac_2p8", "v_median",
      "ic_peak", "v_ic_peak", "ic_fwhm", "ic_area"]


# ---- 为每个电芯计算 EOL 循环、RUL、退化速率 ----
def build_rul(df):
    out = []
    for cell, d in df.groupby("cell"):
        d = d.sort_values("cycle")
        cyc = d["cycle"].to_numpy(float)
        soh = d["SOH"].to_numpy(float)
        cap = d["cap"].to_numpy(float)                  # 绝对容量，用标称 80% 定义 EOL
        # Severson 把电芯循环到 ~0.88Ah(80%) 即停止记录，多数电芯最低容量卡在 0.880-0.881。
        # 纳入条件放宽到 0.90(已衰减到 EOL 附近)，其记录终止处视作 EOL。
        if cap.min() > 0.90 or len(d) < 8:
            continue
        if (cap <= EOL_CAP).any():                      # 真正跌破 0.88 -> 插值求交叉
            k = int(np.argmax(cap <= EOL_CAP))
            if k == 0:
                eol_cycle = cyc[0]
            else:
                x0, x1, y0, y1 = cyc[k - 1], cyc[k], cap[k - 1], cap[k]
                eol_cycle = x0 + (EOL_CAP - y0) * (x1 - x0) / (y1 - y0)
        else:                                           # 止于 EOL 附近 -> 末循环作为 EOL
            eol_cycle = cyc[-1]
        slope = np.gradient(soh, cyc)                   # 退化速率 dSOH/dcycle
        slope = pd.Series(slope).rolling(5, min_periods=1, center=True).mean().to_numpy()
        dd = d.copy()
        dd["soh_now"] = soh
        dd["soh_slope"] = slope
        dd["RUL"] = eol_cycle - cyc
        dd = dd[dd["RUL"] >= 0]                          # 只保留 EOL 之前
        out.append(dd)
    return pd.concat(out, ignore_index=True)


def slope_by_cell(frame, value_col):
    """逐电芯计算 value_col 对 cycle 的局部斜率(滚动平滑)。frame 需含 cell/cycle 列。"""
    fr = frame.reset_index(drop=True)
    s = np.zeros(len(fr))
    for _, idx in fr.groupby("cell").groups.items():
        sub = fr.loc[idx].sort_values("cycle")
        gg = np.gradient(sub[value_col].to_numpy(float), sub["cycle"].to_numpy(float))
        gg = pd.Series(gg).rolling(5, min_periods=1, center=True).mean().to_numpy()
        s[fr.index.get_indexer(sub.index)] = gg
    return s


data = build_rul(df)
FEATURES = HI + ["soh_now", "soh_slope"]
X = data[FEATURES].to_numpy(); y = data["RUL"].to_numpy(); g = data["cell"].to_numpy()

# 70/30 按芯划分；训练内再分校准集做 conformal
tr, te = next(GroupShuffleSplit(1, test_size=0.30, random_state=0).split(X, y, g))
ptr_l, cal_l = next(GroupShuffleSplit(1, test_size=0.25, random_state=1).split(X[tr], y[tr], g[tr]))
ptr, cal = tr[ptr_l], tr[cal_l]

print("=" * 64)
print("RUL 剩余寿命预测 (Severson 公共数据, 复现论文B主线)")
print(f"样本 {len(data)} | 特征 {len(FEATURES)} | "
      f"训练芯 {len(np.unique(g[ptr]))} / 校准芯 {len(np.unique(g[cal]))} / 测试芯 {len(np.unique(g[te]))}")
print(f"RUL 范围: {y.min():.0f} ~ {y.max():.0f} 圈")
print("=" * 64)

rf = RandomForestRegressor(n_estimators=400, min_samples_leaf=2, n_jobs=-1, random_state=0).fit(X[ptr], y[ptr])
p = rf.predict(X[te])                                    # oracle(实测 SOH 输入)预测,作对照
rmse = float(np.sqrt(np.mean((p - y[te]) ** 2)))
print(f"\noracle(实测SOH)点预测:  MAE={mean_absolute_error(y[te], p):.1f} 圈   R²={r2_score(y[te], p):.3f}")
res = np.abs(y[cal] - rf.predict(X[cal]))
n = len(res); qhat = float(np.quantile(res, min(1.0, np.ceil((n + 1) * 0.9) / n)))

# ---- 端到端(可部署): SOH 模型样本外预测 -> 作为 RUL 输入(主图所用)----
trd = data.iloc[ptr].copy(); cald = data.iloc[cal].copy(); ted = data.iloc[te].copy()
soh_m = RandomForestRegressor(400, min_samples_leaf=2, n_jobs=-1, random_state=0).fit(trd[HI], trd["soh_now"])
for fr in (cald, ted):
    fr["soh_pred"] = soh_m.predict(fr[HI])
    fr["slope_pred"] = slope_by_cell(fr, "soh_pred")
fE = HI + ["soh_in", "slope_in"]
trd2 = trd.rename(columns={"soh_now": "soh_in", "soh_slope": "slope_in"})
rfE = RandomForestRegressor(400, min_samples_leaf=2, n_jobs=-1, random_state=0).fit(trd2[fE], trd2["RUL"])
def feE(fr):
    x = fr[HI].copy(); x["soh_in"] = fr["soh_pred"].values; x["slope_in"] = fr["slope_pred"].values
    return x[fE]
pE = rfE.predict(feE(ted)); pE_cal = rfE.predict(feE(cald))
resE = np.abs(ted["RUL"].values - pE)
resE_cal = np.abs(cald["RUL"].values - pE_cal)
nE = len(resE_cal); qE = float(np.quantile(resE_cal, min(1.0, np.ceil((nE + 1) * 0.9) / nE)))
loE, hiE = pE - qE, pE + qE
picpE = float(np.mean((ted["RUL"].values >= loE) & (ted["RUL"].values <= hiE)) * 100)
print(f"端到端(出样预测SOH)点预测:  MAE={mean_absolute_error(ted['RUL'], pE):.1f} 圈   R²={r2_score(ted['RUL'], pE):.3f}")
print(f"端到端 conformal 90% 区间:  覆盖率 PICP={picpE:.1f}%   宽度 ±{qE:.0f} 圈")

# 按 RUL 区间看端到端误差(近 EOL vs 远 EOL)
print("\n端到端 分段 MAE(圈):")
for a, bnd in [(0, 100), (100, 300), (300, 700), (700, 9999)]:
    m = (ted["RUL"].values >= a) & (ted["RUL"].values < bnd)
    if m.sum():
        print(f"  RUL∈[{a},{bnd}):  MAE={mean_absolute_error(ted['RUL'].values[m], pE[m]):.1f}  (n={m.sum()})")

# ---- 画测试电芯 RUL 曲线: 端到端为主(预测SOH输入)+ oracle 对照 ----
doracle = data.iloc[te].copy(); doracle["pred_oracle"] = p
ted = ted.copy(); ted["pred"] = pE; ted["lo"] = np.clip(loE, 0, None); ted["hi"] = hiE
ted = ted.merge(doracle[["cell", "cycle", "pred_oracle"]], on=["cell", "cycle"], how="left")
try:
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cells = list(pd.unique(ted["cell"]))[:6]
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    for ax, c in zip(axes.ravel(), cells):
        d = ted[ted.cell == c].sort_values("cycle")
        ax.fill_between(d.cycle, d.lo, d.hi, color="steelblue", alpha=0.2, label="90% interval (end-to-end)")
        ax.plot(d.cycle, d.RUL, "k-", lw=2, label="True RUL")
        ax.plot(d.cycle, d.pred, "b.", ms=3, label="End-to-end (predicted SOH)")
        ax.plot(d.cycle, d.pred_oracle, color="0.55", ls="--", lw=1.1, label="Oracle (measured SOH)")
        ax.set_title(c, fontsize=8); ax.set_xlabel("cycle"); ax.set_ylabel("RUL (cycles)"); ax.grid(alpha=0.3)
    axes.ravel()[0].legend(fontsize=7)
    fig.suptitle("End-to-end RUL on unseen cells (out-of-sample SOH input; 90% split-conformal interval), oracle for comparison")
    fig.tight_layout()
    png = HERE / "rul_curves.png"
    fig.savefig(png, dpi=300)
    print(f"\nRUL 曲线已保存: {png}")
except Exception as e:
    print(f"(画图跳过: {e})")

imp = sorted(zip(FEATURES, rf.feature_importances_), key=lambda x: -x[1])
print("\n特征重要性 Top-6:", ", ".join(f"{k}={v:.2f}" for k, v in imp[:6]))
print("\nRUL 预测管线运行成功 [OK]")
