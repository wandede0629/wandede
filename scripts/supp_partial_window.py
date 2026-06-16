r"""
补充实验 S7: 部分电压窗口特征 -> 面向真实 BMS 的 SOH 估计
回应"所有特征依赖完整充放电曲线, 真实 EV 极少全充全放"。

从每个循环的放电 Q(V) 曲线只取一段电压【窗口】内的曲线特征(局部 IC 峰 + 局部容量比例
+ 局部中位电压)+ 全局 summary HI, 估计 SOH。两个扫描:
  (A) 宽度扫描(窗口居中于 LFP 平台 ~3.2V): 0.15/0.30/0.45/0.75/1.50 V ≈ 10/20/30/50/100%
  (B) 位置扫描(固定 20% 宽 = 0.30 V): 中心 2.4/2.8/3.1/3.2/3.3 V
每个配置: RF, 5 个随机种子, 按电芯划分, 报告 SOH R²。

结论: 多窄/在哪段窗口就能达到接近完整曲线的精度 -> 可部署性。
输出: 打印 + 图 supp_partial_window.png + 表 data/supp_partial_window.csv
"""
import warnings
import numpy as np
import pandas as pd
import h5py
from pathlib import Path
from scipy.signal import savgol_filter
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

warnings.filterwarnings("ignore")
HERE = Path(__file__).parent
DATA = HERE / "data"
BATCHES = ["2017-05-12_batchdata_updated_struct_errorcorrect.mat",
           "2017-06-30_batchdata_updated_struct_errorcorrect.mat",
           "2018-04-12_batchdata_updated_struct_errorcorrect.mat"]
N_PER_CELL = 60
WIDTH_SET = [("w10", 3.2, 0.15), ("w20", 3.2, 0.30), ("w30", 3.2, 0.45),
             ("w50", 3.2, 0.75), ("w100", 2.75, 1.50)]            # (name, center, width)
LOC_SET = [("c2p4", 2.4, 0.30), ("c2p8", 2.8, 0.30), ("c3p1", 3.1, 0.30),
           ("c3p2", 3.2, 0.30), ("c3p3", 3.3, 0.30)]
SUMMARY = ["chargetime", "IR", "Tavg", "Tmax", "Tmin"]


def window_feats(v, q, ic, lo, hi):
    msk = (v >= lo) & (v <= hi)
    if msk.sum() < 5:
        return [np.nan, np.nan, np.nan]
    vv, qq, ii = v[msk], q[msk], ic[msk]
    ic_peak = float(np.nanmax(ii))
    cap_win = float(qq.max() - qq.min())                # 窗口内释放容量
    frac = (qq - qq.min()) / (qq.max() - qq.min() + 1e-9)
    vmed = float(np.interp(0.5, frac[::-1], vv[::-1]))   # frac 随 v 递减, 反转后插值
    return [ic_peak, cap_win, vmed]


def extract():
    rows = []
    for fn in BATCHES:
        with h5py.File(DATA / fn, "r") as f:
            b = f["batch"]; n = b["cycle_life"].shape[0]
            for i in range(n):
                s = f[b["summary"][i, 0]]
                qd = np.array(s["QDischarge"]).flatten().astype(float)
                summ = {k: np.array(s[k2]).flatten().astype(float)
                        for k, k2 in zip(SUMMARY, ["chargetime", "IR", "Tavg", "Tmax", "Tmin"])}
                valid = np.where(np.isfinite(qd) & (qd > 0.5))[0]
                cyc_ds = f[b["cycles"][i, 0]]; ncyc = cyc_ds["Qdlin"].shape[0]
                valid = valid[valid < ncyc]
                if valid.size < 30:
                    continue
                q0 = np.median(qd[valid[:10]])
                vdlin = np.array(f[b["Vdlin"][i, 0]]).flatten().astype(float)
                idx = valid[np.linspace(0, valid.size - 1, N_PER_CELL).astype(int)]
                for j in idx:
                    soh = qd[j] / q0
                    if not (0.6 <= soh <= 1.05):
                        continue
                    ql = np.array(f[cyc_ds["Qdlin"][int(j), 0]]).flatten().astype(float)
                    m = np.isfinite(vdlin) & np.isfinite(ql)
                    v, q = vdlin[m], ql[m]
                    if v.size < 30:
                        continue
                    o = np.argsort(v); v, q = v[o], q[o]
                    ic = np.abs(np.gradient(q, v))
                    w = min(51, (len(ic) // 2) * 2 - 1)
                    if w >= 5:
                        ic = savgol_filter(ic, w, 3)
                    ic = np.clip(ic, 0, None)
                    rec = {"cell": f"{fn[:10]}_{i}", "SOH": soh}
                    for k in SUMMARY:
                        rec[k] = summ[k][j]
                    for name, c, wd in WIDTH_SET + LOC_SET:
                        p, cw, vm = window_feats(v, q, ic, c - wd / 2, c + wd / 2)
                        rec[name + "_icpeak"] = p; rec[name + "_cap"] = cw; rec[name + "_vmed"] = vm
                    rows.append(rec)
    return pd.DataFrame(rows)


csv = DATA / "supp_partial_window.csv"
if csv.exists():
    df = pd.read_csv(csv)
else:
    print("提取部分窗口特征中 ...")
    df = extract(); df.to_csv(csv, index=False)
print(f"样本 {len(df)}, 电芯 {df.cell.nunique()}")


def eval_cfg(name):
    feats = SUMMARY + [name + "_icpeak", name + "_cap", name + "_vmed"]
    d = df.dropna(subset=feats + ["SOH"])
    X = d[feats].to_numpy(); y = d.SOH.to_numpy(); g = d.cell.to_numpy()
    r2 = []
    for s in range(5):
        tr, te = next(GroupShuffleSplit(1, test_size=0.30, random_state=s).split(X, y, g))
        sc = StandardScaler().fit(X[tr])
        m = RandomForestRegressor(300, min_samples_leaf=2, n_jobs=-1, random_state=0).fit(sc.transform(X[tr]), y[tr])
        r2.append(r2_score(y[te], m.predict(sc.transform(X[te]))))
    return np.mean(r2), np.std(r2)


print("\n(A) 宽度扫描(窗口居中于平台 3.2V):")
wid_pct = {"w10": 10, "w20": 20, "w30": 30, "w50": 50, "w100": 100}
A = []
for name, _, _ in WIDTH_SET:
    m, sd = eval_cfg(name); A.append((wid_pct[name], m, sd))
    print(f"  {wid_pct[name]:>4}% 窗口   R² = {m:.3f} ± {sd:.3f}")

print("\n(B) 位置扫描(固定 20% 宽):")
B = []
for name, c, _ in LOC_SET:
    m, sd = eval_cfg(name); B.append((c, m, sd))
    print(f"  中心 {c:.1f} V   R² = {m:.3f} ± {sd:.3f}")

try:
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.6))
    w, mw, sw = zip(*A)
    ax1.errorbar(w, mw, yerr=sw, fmt="o-", capsize=3)
    ax1.set_xlabel("Voltage-window width (% of span)"); ax1.set_ylabel("SOH R² (cell-level CV)")
    ax1.set_title("(a) Width sweep (window centred on plateau)"); ax1.grid(alpha=0.3)
    c, mc, sc_ = zip(*B)
    ax2.errorbar(c, mc, yerr=sc_, fmt="s-", color="#d62728", capsize=3)
    ax2.axvspan(3.1, 3.35, color="green", alpha=0.12, label="LFP plateau / IC peak")
    ax2.set_xlabel("20%-window centre voltage (V)"); ax2.set_ylabel("SOH R² (cell-level CV)")
    ax2.set_title("(b) Location sweep (fixed 20% width)"); ax2.legend(fontsize=8); ax2.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(HERE / "supp_partial_window.png", dpi=300)
    print("\n图已保存: supp_partial_window.png")
except Exception as e:
    print(f"(画图跳过: {e})")
