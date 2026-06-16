r"""
补充实验 S3: RUL 端到端对比(回应"RUL 用实测 SOH 是上限"的质疑)

三种 SOH 输入设定,按电芯划分、5 个随机种子取均值±标准差:
  (A) no-SOH      : 仅 15 个 HI 特征(无 SOH),完全不依赖容量标签
  (B) end-to-end  : HI + 【样本外预测 SOH】及其斜率(SOH 模型在训练芯上训练,预测测试芯)-> 可部署
  (C) oracle      : HI + 【实测 SOH】及其斜率(= 主文设定,上限)

报告每种设定的 RUL R²/MAE(cycles),并给出 RUL 目标值分布(中位数/范围/相对误差),
让 MAE 有上下文。

运行: .venv\Scripts\python.exe supp_rul_endtoend.py
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

HI = ["chargetime", "IR", "Tavg", "Tmax", "Tmin",
      "frac_3p3", "frac_3p2", "frac_3p1", "frac_3p0", "frac_2p8", "v_median",
      "ic_peak", "v_ic_peak", "ic_fwhm", "ic_area"]
EOL_CAP = 0.80 * 1.1


def build_rul(df):
    out = []
    for cell, d in df.groupby("cell"):
        d = d.sort_values("cycle")
        cyc = d["cycle"].to_numpy(float); cap = d["cap"].to_numpy(float); soh = d["SOH"].to_numpy(float)
        if cap.min() > 0.90 or len(d) < 8:
            continue
        if (cap <= EOL_CAP).any():
            k = int(np.argmax(cap <= EOL_CAP))
            eol = cyc[0] if k == 0 else cyc[k-1] + (EOL_CAP - cap[k-1]) * (cyc[k]-cyc[k-1]) / (cap[k]-cap[k-1])
        else:
            eol = cyc[-1]
        dd = d.copy()
        dd["RUL"] = eol - cyc
        dd = dd[dd["RUL"] >= 0]
        out.append(dd)
    return pd.concat(out, ignore_index=True)


def slope_by_cell(frame, value_col):
    s = np.zeros(len(frame))
    for cell, idx in frame.groupby("cell").groups.items():
        sub = frame.loc[idx].sort_values("cycle")
        g = np.gradient(sub[value_col].to_numpy(float), sub["cycle"].to_numpy(float))
        g = pd.Series(g).rolling(5, min_periods=1, center=True).mean().to_numpy()
        s[frame.index.get_indexer(sub.index)] = g
    return s


data = build_rul(df)
data["soh_slope_meas"] = slope_by_cell(data, "SOH")
y = data["RUL"].to_numpy(); g = data["cell"].to_numpy()

print("=" * 66)
print("RUL 端到端对比 (按电芯划分, 5 seeds, 随机森林)")
print("=" * 66)
print(f"RUL 分布: 中位数={np.median(y):.0f}  均值={y.mean():.0f}  范围=[{y.min():.0f}, {y.max():.0f}] cycles")

def rf(): return RandomForestRegressor(300, min_samples_leaf=2, n_jobs=-1, random_state=0)

res = {"A_noSOH": [], "B_endtoend": [], "C_oracle": []}
mae = {"A_noSOH": [], "B_endtoend": [], "C_oracle": []}
for seed in range(5):
    tr, te = next(GroupShuffleSplit(1, test_size=0.30, random_state=seed).split(data, y, g))
    trd, ted = data.iloc[tr].copy(), data.iloc[te].copy()

    # --- SOH 模型: 训练芯 HI->SOH, 预测测试芯(样本外) ---
    soh_model = rf().fit(trd[HI], trd["SOH"])
    ted = ted.copy(); ted["soh_pred"] = soh_model.predict(ted[HI])
    ted["soh_slope_pred"] = slope_by_cell(ted.reset_index(drop=True), "soh_pred")

    # (A) no-SOH
    m = rf().fit(trd[HI], trd["RUL"]); p = m.predict(ted[HI])
    res["A_noSOH"].append(r2_score(ted.RUL, p)); mae["A_noSOH"].append(mean_absolute_error(ted.RUL, p))
    # (C) oracle: 训练&测试都用实测 SOH
    fC = HI + ["SOH", "soh_slope_meas"]
    m = rf().fit(trd[fC], trd["RUL"]); p = m.predict(ted[fC])
    res["C_oracle"].append(r2_score(ted.RUL, p)); mae["C_oracle"].append(mean_absolute_error(ted.RUL, p))
    # (B) end-to-end: 训练用实测SOH, 测试用样本外预测SOH
    m = rf().fit(trd[HI + ["SOH", "soh_slope_meas"]].rename(columns={"SOH": "soh_in", "soh_slope_meas": "slope_in"}), trd["RUL"])
    teB = ted[HI].copy(); teB["soh_in"] = ted["soh_pred"]; teB["slope_in"] = ted["soh_slope_pred"]
    p = m.predict(teB[HI + ["soh_in", "slope_in"]])
    res["B_endtoend"].append(r2_score(ted.RUL, p)); mae["B_endtoend"].append(mean_absolute_error(ted.RUL, p))

print(f"\n{'设定':<22}{'R² (mean±std)':>20}{'MAE cycles':>16}{'相对MAE':>10}")
labels = {"A_noSOH": "A 无SOH(纯HI)", "B_endtoend": "B 端到端(预测SOH)", "C_oracle": "C oracle(实测SOH)"}
med = np.median(y)
for k in ["A_noSOH", "B_endtoend", "C_oracle"]:
    r = np.array(res[k]); mm = np.array(mae[k])
    print(f"{labels[k]:<22}{r.mean():>10.3f}±{r.std():<6.3f}{mm.mean():>10.1f}±{mm.std():<4.1f}{mm.mean()/med*100:>9.1f}%")

print("\n结论: oracle(C) 是上限; 端到端(B)用样本外预测SOH是可部署性能;")
print("      无SOH(A)显示纯曲线特征已能预测RUL的程度。三者差距量化了'实测SOH'带来的乐观偏差。")
