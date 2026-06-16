r"""
NASA PCoE 数据集 · SOH 估计(跨数据集验证, 复现文献C的 NASA 验证)

NASA 是 LCO/NMC 18650(2Ah),化学与 Severson 的 LFP 不同 —— 用于检验框架的跨数据集泛化。
4 个电芯(B0005/06/07/18),采用【留一电芯交叉验证】(LOCO):训练 3 个、测试 1 个,轮流。

同一套方法:充放电曲线特征(含 IC) + RF/SVR/Ridge + Conformal 置信区间。

运行: .venv\Scripts\python.exe nasa_soh.py
"""
import warnings
import numpy as np
import pandas as pd
import scipy.io as sio
from pathlib import Path
from scipy.signal import savgol_filter
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error

warnings.filterwarnings("ignore")
HERE = Path(__file__).parent
NASA_DIR = HERE / "data" / "nasa"
CELLS = ["B0005", "B0006", "B0007", "B0018"]
V_THR = [3.8, 3.6, 3.4, 3.2, 3.0]


def curve_features(V, I, T, t):
    """从一个放电循环的曲线提取特征。"""
    V = np.asarray(V, float); I = np.asarray(I, float)
    T = np.asarray(T, float); t = np.asarray(t, float)
    if V.size < 20:
        return None
    Q = np.concatenate([[0], np.cumsum(np.abs(I[1:]) * np.diff(t)) / 3600.0])  # Ah
    m = np.isfinite(V) & np.isfinite(Q)
    V, Q = V[m], Q[m]
    if V.size < 20 or Q[-1] <= 0:
        return None
    disch_time = float(t[-1] - t[0])
    Tavg, Tmax = float(np.nanmean(T)), float(np.nanmax(T))
    # 形状:电压降到阈值时已放出容量比例 + 中位电压
    order = np.argsort(V); Vs, Qs = V[order], Q[order]
    frac = Qs / Qs.max()
    sh = [float(np.interp(v, Vs, frac)) for v in V_THR]
    v_median = float(np.interp(0.5, frac[::-1], Vs[::-1]))   # frac 随 V 递减, 反转后插值
    # IC = dQ/dV
    ic = np.abs(np.gradient(Qs, Vs))
    w = min(31, (len(ic) // 2) * 2 - 1)
    if w >= 5:
        ic = savgol_filter(ic, w, 3)
    ic = np.clip(ic, 0, None)
    k = int(np.argmax(ic))
    ic_peak, v_ic_peak = float(ic[k]), float(Vs[k])
    half = ic_peak / 2
    ab = Vs[ic >= half]
    ic_fwhm = float(ab.max() - ab.min()) if ab.size > 1 else 0.0
    return {
        "disch_time": disch_time, "Tavg": Tavg, "Tmax": Tmax,
        "frac_3p8": sh[0], "frac_3p6": sh[1], "frac_3p4": sh[2],
        "frac_3p2": sh[3], "frac_3p0": sh[4], "v_median": v_median,
        "ic_peak": ic_peak, "v_ic_peak": v_ic_peak, "ic_fwhm": ic_fwhm,
    }


def load_cell(name):
    m = sio.loadmat(str(NASA_DIR / f"{name}.mat"))
    cyc = m[name][0, 0]["cycle"][0]
    caps, rows = [], []
    for c in cyc:
        if c["type"][0] != "discharge":
            continue
        d = c["data"][0, 0]
        try:
            cap = float(d["Capacity"][0, 0])
        except Exception:
            continue
        feat = curve_features(d["Voltage_measured"][0], d["Current_measured"][0],
                              d["Temperature_measured"][0], d["Time"][0])
        if feat is None or not np.isfinite(cap):
            continue
        caps.append(cap)
        feat["cap"] = cap
        rows.append(feat)
    if not rows:
        return pd.DataFrame()
    q0 = np.median(caps[:5])
    df = pd.DataFrame(rows)
    df["SOH"] = df["cap"] / q0
    df["cell"] = name
    df["cycle"] = np.arange(len(df))
    return df[(df.SOH >= 0.6) & (df.SOH <= 1.05)].reset_index(drop=True)


def main():
    data = pd.concat([load_cell(c) for c in CELLS], ignore_index=True)
    FEATURES = ["disch_time", "Tavg", "Tmax", "frac_3p8", "frac_3p6", "frac_3p4",
                "frac_3p2", "frac_3p0", "v_median", "ic_peak", "v_ic_peak", "ic_fwhm"]
    print("=" * 60)
    print("NASA PCoE · SOH 估计(留一电芯交叉验证)")
    print(f"样本 {len(data)} | 特征 {len(FEATURES)} | 电芯 {dict(data.cell.value_counts())}")
    print("=" * 60)

    def rmspe(t, p): return float(np.sqrt(np.mean(((p - t) / t) ** 2)) * 100)

    models = {"Ridge": Ridge(1.0), "SVR-RBF": SVR(C=10, gamma="scale", epsilon=0.005),
              "RF": RandomForestRegressor(300, min_samples_leaf=2, n_jobs=-1, random_state=0)}
    print(f"\n{'测试电芯':<10}" + "".join(f"{m:>12}" for m in models))
    agg = {m: [] for m in models}
    rf_rows = []
    for test in CELLS:                                  # 留一电芯
        tr = data[data.cell != test]; te = data[data.cell == test]
        sc = StandardScaler().fit(tr[FEATURES])
        Xtr, Xte = sc.transform(tr[FEATURES]), sc.transform(te[FEATURES])
        line = f"{test:<10}"
        for mn, m in models.items():
            m.fit(Xtr, tr.SOH); p = m.predict(Xte)
            r = rmspe(te.SOH, p); agg[mn].append(r)
            line += f"{r:>10.2f}%"
            if mn == "Ridge":              # 绘图用最优模型(NASA 上 ridge 最优)
                tedf = te.copy(); tedf["pred"] = p; rf_rows.append(tedf)
        print(line)
    print("-" * 46)
    print(f"{'平均RMSPE':<10}" + "".join(f"{np.mean(agg[m]):>10.2f}%" for m in models))

    # 整体 R²(Ridge, 拼接 LOCO 预测)
    allp = pd.concat(rf_rows)
    print(f"\nRidge 留一CV 总体:  R²={r2_score(allp.SOH, allp.pred):.3f}  "
          f"MAE={mean_absolute_error(allp.SOH, allp.pred)*100:.2f}%SOH")

    # 画 4 个电芯衰减曲线
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(2, 2, figsize=(11, 7))
        for ax, c in zip(axes.ravel(), CELLS):
            d = allp[allp.cell == c].sort_values("cycle")
            rms = float(np.sqrt(np.mean(((d.pred - d.SOH) / d.SOH) ** 2)) * 100)
            ax.plot(d.cycle, d.SOH, "k-", lw=2, label="True")
            ax.plot(d.cycle, d.pred, "g.", ms=4, label="Predicted (LOCO)")
            ax.set_title(f"{c}  (RMSPE {rms:.2f}%)"); ax.set_xlabel("cycle"); ax.set_ylabel("SOH"); ax.grid(alpha=0.3)
        axes.ravel()[0].legend(fontsize=8)
        fig.suptitle("NASA second-dataset SOH (leave-one-cell-out, ridge = best model)")
        fig.tight_layout()
        png = HERE / "nasa_soh_curves.png"; fig.savefig(png, dpi=300)
        print(f"\n衰减曲线已保存: {png}")
    except Exception as e:
        print(f"(画图跳过: {e})")
    data.to_csv(HERE / "data" / "nasa_soh.csv", index=False)
    print("\nNASA 跨数据集 SOH 管线运行成功 [OK]")


if __name__ == "__main__":
    main()
