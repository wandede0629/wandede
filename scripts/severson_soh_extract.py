r"""
SOH 估计 · 逐循环特征提取  ->  data\severson_soh.csv

复现三篇 SOH 论文的公共思路:从每个循环的充/放电信号提取“健康指标(HI)”，
预测该循环的 SOH(= 当前放电容量 / 初始容量)。

每行 = 某电芯的某一循环:
  HI 特征(BMS 可测，不直接用总容量):
    chargetime  充电时间(老化->变化)
    IR          内阻(老化->升高)
    Tavg/Tmax/Tmin  温度
    放电电压曲线【形状】特征(归一化，不泄漏容量幅值):
       frac@3.3/3.2/3.1/3.0/2.8V  电压降到该值时已释放容量的比例
       v_median  释放 50% 容量时的电压(老化->极化漂移)
  标签: SOH

注: batch4(2019)未记录内阻 IR，故本管线用 batches 1-3。
运行: .venv\Scripts\python.exe severson_soh_extract.py
"""
import numpy as np
import pandas as pd
import h5py
from pathlib import Path
from scipy.signal import savgol_filter

DATA_DIR = Path(__file__).parent / "data"
BATCH_FILES = [
    "2017-05-12_batchdata_updated_struct_errorcorrect.mat",
    "2017-06-30_batchdata_updated_struct_errorcorrect.mat",
    "2018-04-12_batchdata_updated_struct_errorcorrect.mat",
]
V_THRESHOLDS = [3.3, 3.2, 3.1, 3.0, 2.8]
N_SAMPLE_PER_CELL = 120          # 每个电芯采样多少个循环(子采样以控制时长)


def shape_features(v, q):
    """放电电压曲线形状特征:不同电压阈值处累计释放容量比例 + 中位电压。"""
    m = np.isfinite(v) & np.isfinite(q)
    v, q = v[m], q[m]
    if v.size < 10 or np.nanmax(q) <= 0:
        return None
    order = np.argsort(v)                 # 电压升序
    v, q = v[order], q[order]
    qmax = q.max()
    frac = q / qmax                       # 累计释放容量比例 [0,1]
    feats = [float(np.interp(t, v, frac)) for t in V_THRESHOLDS]   # frac@阈值
    v_median = float(np.interp(0.5, frac[::-1], v[::-1]))          # 50%容量处电压(frac 随 v 递减, 反转后插值)
    return feats + [v_median]


def ic_features(v, q):
    """增量容量 IC = |dQ/dV| 曲线特征(LFP 在 ~3.2-3.3V 有特征峰)。"""
    m = np.isfinite(v) & np.isfinite(q)
    v, q = v[m], q[m]
    if v.size < 30 or np.nanmax(q) <= 0:
        return None
    order = np.argsort(v)
    v, q = v[order], q[order]
    ic = np.abs(np.gradient(q, v))                 # dQ/dV
    w = min(51, (len(ic) // 2) * 2 - 1)            # 平滑(奇数窗)
    if w >= 5:
        ic = savgol_filter(ic, w, 3)
    ic = np.clip(ic, 0, None)
    k = int(np.argmax(ic))
    ic_peak = float(ic[k])                         # 峰高(老化->下降)
    v_peak = float(v[k])                           # 峰位电压(老化->漂移)
    half = ic_peak / 2.0
    above = v[ic >= half]
    fwhm = float(above.max() - above.min()) if above.size > 1 else 0.0   # 峰宽
    win = (v >= 3.15) & (v <= 3.35)                # 平台窗口内 IC 面积
    area = float(np.trapz(ic[win], v[win])) if win.sum() > 1 else 0.0
    return [ic_peak, v_peak, fwhm, area]


def extract_cell(f, b, i, source):
    s = f[b["summary"][i, 0]]
    qd = np.array(s["QDischarge"]).flatten().astype(float)
    ir = np.array(s["IR"]).flatten().astype(float)
    ct = np.array(s["chargetime"]).flatten().astype(float)
    tav = np.array(s["Tavg"]).flatten().astype(float)
    tmx = np.array(s["Tmax"]).flatten().astype(float)
    tmn = np.array(s["Tmin"]).flatten().astype(float)
    cyc = np.array(s["cycle"]).flatten().astype(float)

    valid = np.where(np.isfinite(qd) & (qd > 0.5))[0]
    if valid.size < 30:
        return []
    q0 = np.median(qd[valid[:10]])        # 初始容量(前若干有效循环)
    if not np.isfinite(q0) or q0 <= 0:
        return []

    cyc_ds = f[b["cycles"][i, 0]]
    n_cyc = cyc_ds["Qdlin"].shape[0]
    vdlin = np.array(f[b["Vdlin"][i, 0]]).flatten().astype(float)

    idx = valid[valid < n_cyc]
    if idx.size > N_SAMPLE_PER_CELL:      # 均匀子采样
        idx = idx[np.linspace(0, idx.size - 1, N_SAMPLE_PER_CELL).astype(int)]

    rows = []
    for j in idx:
        soh = qd[j] / q0
        if not (0.6 <= soh <= 1.05):
            continue
        try:
            qdlin = np.array(f[cyc_ds["Qdlin"][int(j), 0]]).flatten().astype(float)
        except Exception:
            continue
        sf = shape_features(vdlin, qdlin)
        icf = ic_features(vdlin, qdlin)
        if sf is None or icf is None:
            continue
        rows.append({
            "cell": f"{source}_{i}", "cycle": float(cyc[j]),
            "chargetime": ct[j], "IR": ir[j],
            "Tavg": tav[j], "Tmax": tmx[j], "Tmin": tmn[j],
            "frac_3p3": sf[0], "frac_3p2": sf[1], "frac_3p1": sf[2],
            "frac_3p0": sf[3], "frac_2p8": sf[4], "v_median": sf[5],
            "ic_peak": icf[0], "v_ic_peak": icf[1], "ic_fwhm": icf[2], "ic_area": icf[3],
            "cap": float(qd[j]),            # 绝对放电容量(Ah)，供按标称容量定义 EOL
            "SOH": soh,
        })
    return rows


def main():
    all_rows = []
    for fn in BATCH_FILES:
        path = DATA_DIR / fn
        if not path.exists():
            print(f"(缺 {fn}，跳过)")
            continue
        print(f"提取 {fn} ...")
        with h5py.File(path, "r") as f:
            b = f["batch"]
            n = b["cycle_life"].shape[0]
            cnt = 0
            for i in range(n):
                try:
                    r = extract_cell(f, b, i, fn[:10])
                    all_rows.extend(r); cnt += len(r)
                except Exception as e:
                    print(f"  (跳过 cell {i}: {e})")
            print(f"  得到 {cnt} 个(循环)样本")

    df = pd.DataFrame(all_rows).dropna().reset_index(drop=True)
    out = DATA_DIR / "severson_soh.csv"
    df.to_csv(out, index=False)
    print(f"\n共 {len(df)} 个样本, 来自 {df['cell'].nunique()} 个电芯 -> {out}")
    print(f"SOH 范围: {df.SOH.min():.3f} ~ {df.SOH.max():.3f}")
    print("\n下一步: .venv\\Scripts\\python.exe severson_soh_model.py")


if __name__ == "__main__":
    main()
