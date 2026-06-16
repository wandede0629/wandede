r"""
Path A 跨化学挑战实验: LFP (Severson) -> LCO (CALCE CS2)
诚实地检验"框架普适性 = 方法论可迁移, 而非模型可零样本迁移"。

共享特征(仅用放电 V/Q 曲线, 两化学可同口径计算):
  ic_peak, v_ic_peak, ic_fwhm, ic_area, v_median, frac@5 个电压阈值。
阈值有两种取法:
  - "lfp_fixed": 写死的 LFP 阈值/IC 窗 (3.1-3.35V 平台) —— 朴素迁移
  - "native":   按该化学放电电压范围的百分位数据驱动取阈值 + 数据定位 IC 窗 —— 重提特征

四臂:
  A1 zero-shot     : 在 LFP 特征(lfp_fixed)上训模型, 直接预测 LCO(lfp_fixed) —— 预期失败
  A2 native re-fit : 在 LCO(native) 上 cell-level 多 seed 重拟合 —— 预期恢复
  A2c window-ctrl  : 在 LCO(lfp_fixed) 上 cell-level 重拟合 —— 隔离"窗口错位 vs 化学迁移"
  A3 few-shot      : LCO(native), 改变训练电芯数 N —— 给出适配预算
全程报 split-conformal 90% 覆盖率。
运行: .venv\Scripts\python.exe crosschem_experiment.py
"""
import warnings
import io
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import h5py
from scipy.signal import savgol_filter
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import r2_score

warnings.filterwarnings("ignore")
HERE = Path(__file__).parent
DATA = HERE / "data"
CALCE = DATA / "calce"
N_SEEDS = 5
FEATCOLS = ["ic_peak", "v_ic_peak", "ic_fwhm", "ic_area", "v_median",
            "frac0", "frac1", "frac2", "frac3", "frac4"]

# 写死的 LFP 假设(朴素迁移用)
LFP_THR = [3.3, 3.2, 3.1, 3.0, 2.8]
LFP_IC = (3.1, 3.35)

SEVERSON_BATCHES = ["2017-05-12_batchdata_updated_struct_errorcorrect.mat",
                    "2017-06-30_batchdata_updated_struct_errorcorrect.mat",
                    "2018-04-12_batchdata_updated_struct_errorcorrect.mat"]


def rmspe(t, p):
    return float(np.sqrt(np.mean(((p - t) / t) ** 2)) * 100)


def curve_feats(v, q, thr, ic_win):
    """v 升序, q 为放电累计容量(从 0); thr=5 个电压阈值; ic_win=(lo,hi)."""
    o = np.argsort(v)
    v, q = np.asarray(v)[o], np.asarray(q)[o]
    if v.size < 30 or q.max() <= 0:
        return None
    frac = q / q.max()                         # 已释放容量比例
    ic = np.clip(savgol_filter(np.abs(np.gradient(q, v)),
                               min(31, (v.size // 2) * 2 - 1), 3), 0, None)
    k = int(np.argmax(ic))
    ic_peak, v_ic_peak = float(ic[k]), float(v[k])
    half = ic_peak / 2
    ab = v[ic >= half]
    ic_fwhm = float(ab.max() - ab.min()) if ab.size > 1 else 0.0
    mwin = (v >= ic_win[0]) & (v <= ic_win[1])
    ic_area = float(np.trapz(ic[mwin], v[mwin])) if mwin.sum() > 1 else 0.0
    vmed = float(np.interp(0.5, frac, v))      # frac 随 v 升而增
    fr = [float(np.interp(t, v, frac)) for t in thr]
    return [ic_peak, v_ic_peak, ic_fwhm, ic_area, vmed] + fr


def pctl_thresholds(vlo, vhi):
    """数据驱动阈值: 放电电压范围的 [85,70,55,40,20] 百分位。"""
    span = vhi - vlo
    return [vlo + f * span for f in (0.85, 0.70, 0.55, 0.40, 0.20)]


# ---------- LFP 特征(Severson, lfp_fixed 口径) ----------
def build_lfp():
    cache = CALCE / "lfp_shared_feats.csv"
    if cache.exists():
        return pd.read_csv(cache)
    rows = []
    for fn in SEVERSON_BATCHES:
        with h5py.File(DATA / fn, "r") as f:
            b = f["batch"]; n = b["cycle_life"].shape[0]
            for i in range(n):
                s = f[b["summary"][i, 0]]
                qd = np.array(s["QDischarge"]).flatten().astype(float)
                valid = np.where(np.isfinite(qd) & (qd > 0.5))[0]
                cyc_ds = f[b["cycles"][i, 0]]
                valid = valid[valid < cyc_ds["Qdlin"].shape[0]]
                if valid.size < 30:
                    continue
                q0 = np.median(qd[valid[:10]])
                vd = np.array(f[b["Vdlin"][i, 0]]).flatten().astype(float)
                for j in valid[np.linspace(0, valid.size - 1, 40).astype(int)]:
                    soh = qd[j] / q0
                    if not (0.5 <= soh <= 1.05):
                        continue
                    ql = np.array(f[cyc_ds["Qdlin"][int(j), 0]]).flatten().astype(float)
                    m = np.isfinite(vd) & np.isfinite(ql)
                    fe = curve_feats(vd[m], ql[m], LFP_THR, LFP_IC)
                    if fe is None:
                        continue
                    rows.append({"cell": f"{fn[:10]}_{i}", "cycle": int(j), "SOH": soh,
                                 **dict(zip(FEATCOLS, fe))})
    df = pd.DataFrame(rows)
    df.to_csv(cache, index=False)
    return df


# ---------- LCO 特征(CALCE, 两种口径) ----------
def build_lco(mode):
    """mode='lfp_fixed' 或 'native'. 返回特征表。"""
    soh = pd.read_csv(CALCE / "calce_soh.csv")
    curves = np.load(CALCE / "calce_curves.npz")
    # 估计 LCO 放电电压范围(用于 native 阈值)
    if mode == "native":
        vmins, vmaxs, vpeaks = [], [], []
        for key in list(curves.files)[:200]:
            v = curves[key][0]
            vmins.append(v.min()); vmaxs.append(v.max())
        vlo, vhi = float(np.median(vmins)), float(np.median(vmaxs))
        thr = pctl_thresholds(vlo, vhi)
        # 数据定位 IC 窗: 健康循环 IC 峰中位 ± 0.15V
        for key in list(curves.files)[:60]:
            v, q = curves[key]
            o = np.argsort(v); v, q = v[o], q[o]
            if v.size < 30:
                continue
            ic = np.clip(savgol_filter(np.abs(np.gradient(q, v)), min(31, (v.size//2)*2-1), 3), 0, None)
            vpeaks.append(float(v[int(np.argmax(ic))]))
        vpk = float(np.median(vpeaks))
        ic_win = (vpk - 0.15, vpk + 0.15)
        print(f"  [native] LCO 电压范围 {vlo:.2f}-{vhi:.2f}V; 阈值={[round(t,2) for t in thr]}; IC 窗={tuple(round(x,2) for x in ic_win)}")
    else:
        thr, ic_win = LFP_THR, LFP_IC
    rows = []
    for _, r in soh.iterrows():
        key = f"{r.cell}|{int(r.cycle)}"
        if key not in curves.files:
            continue
        v, q = curves[key]
        fe = curve_feats(v, q, thr, ic_win)
        if fe is None:
            continue
        rows.append({"cell": r.cell, "cycle": int(r.cycle), "SOH": float(r.SOH),
                     **dict(zip(FEATCOLS, fe))})
    return pd.DataFrame(rows)


def fit_eval(tr, te, seed=0):
    sc = StandardScaler().fit(tr[FEATCOLS])
    m = RandomForestRegressor(300, min_samples_leaf=2, n_jobs=-1, random_state=seed)
    m.fit(sc.transform(tr[FEATCOLS]), tr.SOH)
    p = m.predict(sc.transform(te[FEATCOLS]))
    return r2_score(te.SOH, p), rmspe(te.SOH.values, p)


def cellcv(df, n_train=None, seeds=N_SEEDS):
    """cell-level 多 seed; 可限定训练电芯数 n_train(few-shot)。返回 (R²均±sd, RMSPE均±sd, cover均)。"""
    cells = df.cell.unique()
    r2s, rms, covs = [], [], []
    for s in range(seeds):
        rng = np.random.RandomState(s)
        perm = rng.permutation(cells)
        if n_train is None:
            ntr = max(1, int(round(0.6 * len(cells))))
        else:
            ntr = min(n_train, len(cells) - 1)
        tr_cells, te_cells = perm[:ntr], perm[ntr:]
        te = df[df.cell.isin(te_cells)]
        if len(te) < 10 or len(tr_cells) < 1:
            continue
        # 真正的 cell-level conformal: 留出 1 个训练芯做校准(模型未见该芯)
        if len(tr_cells) >= 2:
            fit_cells, cal_cells = tr_cells[:-1], tr_cells[-1:]
        else:
            fit_cells, cal_cells = tr_cells, tr_cells   # 仅 1 芯时退化(覆盖率不可靠,会标注)
        tr_fit = df[df.cell.isin(fit_cells)]; cal = df[df.cell.isin(cal_cells)]
        if len(tr_fit) < 20:
            continue
        sc = StandardScaler().fit(tr_fit[FEATCOLS])
        m = RandomForestRegressor(300, min_samples_leaf=2, n_jobs=-1, random_state=s)
        m.fit(sc.transform(tr_fit[FEATCOLS]), tr_fit.SOH)
        res = np.abs(cal.SOH.values - m.predict(sc.transform(cal[FEATCOLS])))
        q90 = np.quantile(res, 0.90) if len(res) else 0.05
        p = m.predict(sc.transform(te[FEATCOLS]))
        r2s.append(r2_score(te.SOH, p)); rms.append(rmspe(te.SOH.values, p))
        covs.append(float(np.mean(np.abs(te.SOH.values - p) <= q90)))
    f = lambda a: (float(np.mean(a)), float(np.std(a)))
    return f(r2s), f(rms), (float(np.mean(covs)) if covs else float("nan"))


def calib_sweep(df, seeds=8):
    """回应审稿: 在小样本 LCO 上, 改变 conformal 校准【电芯数】(1/2/3), 看覆盖率是否从退化恢复。
    固定 2 个测试芯; 其余 4 芯中取 n_cal 个作校准, 剩余作拟合。同时报 R² 以显示小样本下
    '增加校准芯' 与 '拟合芯减少' 的权衡。返回 [(n_cal, cover%, cover_sd, width, R²)]."""
    cells = df.cell.unique()
    rows = []
    for n_cal in (1, 2, 3):
        covs, wids, r2s = [], [], []
        for s in range(seeds):
            perm = np.random.RandomState(s).permutation(cells)
            te = df[df.cell.isin(perm[:2])]
            rest = perm[2:]
            cal = df[df.cell.isin(rest[:n_cal])]
            fit = df[df.cell.isin(rest[n_cal:])]
            if fit.cell.nunique() < 1 or len(fit) < 20:
                continue
            sc = StandardScaler().fit(fit[FEATCOLS])
            m = RandomForestRegressor(300, min_samples_leaf=2, n_jobs=-1, random_state=s).fit(sc.transform(fit[FEATCOLS]), fit.SOH)
            resid = np.abs(cal.SOH.values - m.predict(sc.transform(cal[FEATCOLS])))
            q90 = float(np.quantile(resid, 0.90))
            p = m.predict(sc.transform(te[FEATCOLS]))
            covs.append(float(np.mean(np.abs(te.SOH.values - p) <= q90)))
            wids.append(2 * q90); r2s.append(r2_score(te.SOH, p))
        if covs:
            rows.append((n_cal, float(np.mean(covs) * 100), float(np.std(covs) * 100),
                         float(np.mean(wids)), float(np.mean(r2s)), float(np.mean([fit.cell.nunique()]))))
    return rows


def main():
    print("=" * 64)
    print("Path A 跨化学: LFP (Severson) -> LCO (CALCE CS2)")
    print("=" * 64)
    lfp = build_lfp()
    print(f"LFP 特征: {len(lfp)} 样本 / {lfp.cell.nunique()} 芯")
    lco_naive = build_lco("lfp_fixed")
    lco_native = build_lco("native")
    print(f"LCO 特征: {len(lco_native)} 样本 / {lco_native.cell.nunique()} 芯")

    # A1: zero-shot LFP->LCO (lfp_fixed 口径, 同列名)
    sc = StandardScaler().fit(lfp[FEATCOLS])
    m = RandomForestRegressor(300, min_samples_leaf=2, n_jobs=-1, random_state=0)
    m.fit(sc.transform(lfp[FEATCOLS]), lfp.SOH)
    p = m.predict(sc.transform(lco_naive[FEATCOLS]))
    a1_r2, a1_rms = r2_score(lco_naive.SOH, p), rmspe(lco_naive.SOH.values, p)

    # A2 / A2c / A3
    a2 = cellcv(lco_native)
    a2c = cellcv(lco_naive)
    fewshot = {n: cellcv(lco_native, n_train=n) for n in (1, 2, 3)}
    calsweep = calib_sweep(lco_native)

    print("\n" + "-" * 64)
    print(f"{'臂':<26}{'R²':>14}{'RMSPE(%)':>14}{'cover':>9}")
    print("-" * 64)
    print(f"{'A1 zero-shot LFP->LCO':<26}{a1_r2:>14.3f}{a1_rms:>13.2f}{'—':>9}")
    print(f"{'A2c re-fit (LFP 窗)':<26}{a2c[0][0]:>9.3f}±{a2c[0][1]:.3f}{a2c[1][0]:>9.2f}±{a2c[1][1]:.2f}{a2c[2]*100:>7.0f}%")
    print(f"{'A2 re-fit (native)':<26}{a2[0][0]:>9.3f}±{a2[0][1]:.3f}{a2[1][0]:>9.2f}±{a2[1][1]:.2f}{a2[2]*100:>7.0f}%")
    print("-" * 64)
    print("A3 few-shot (native, 改变训练电芯数 N):")
    for n, (r2, rm, cv) in fewshot.items():
        print(f"   N={n} 芯训练: R²={r2[0]:.3f}±{r2[1]:.3f}  RMSPE={rm[0]:.2f}±{rm[1]:.2f}  cover={cv*100:.0f}%")
    print("-" * 64)
    print("校准芯数扫描 (2 测试芯; conformal 90% 目标; 显示小样本下覆盖率恢复 vs 拟合芯减少的权衡):")
    print(f"   {'校准芯数':<8}{'覆盖率':>14}{'区间宽':>10}{'R²':>8}{'拟合芯':>8}")
    for n_cal, cov, cov_sd, wid, r2, nfit in calsweep:
        print(f"   {n_cal:<8}{cov:>9.0f}±{cov_sd:<4.0f}{wid:>10.3f}{r2:>8.3f}{nfit:>7.0f}")
    print("-" * 64)
    print("\n诚实解读: A1 零样本失败 = 一个冻结的 LFP 模型不能跨化学;")
    print("A2 native 恢复 = 方法论(管线/协议/conformal)可迁移; A2c vs A2 = 窗口错位的代价;")
    print("A3 = 适配预算(需要多少目标化学电芯)。")

    # 存结果供作图/写作
    out = {"A1_r2": a1_r2, "A1_rmspe": a1_rms,
           "A2_r2": a2[0], "A2_rmspe": a2[1], "A2_cover": a2[2],
           "A2c_r2": a2c[0], "A2c_rmspe": a2c[1], "A2c_cover": a2c[2],
           "fewshot": {str(k): {"r2": v[0], "rmspe": v[1], "cover": v[2]} for k, v in fewshot.items()},
           "calib_sweep": [{"n_cal": n, "cover": c, "cover_sd": csd, "width": w, "r2": r2, "n_fit": nf}
                           for n, c, csd, w, r2, nf in calsweep]}
    import json
    (CALCE / "crosschem_results.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\n结果 -> {CALCE/'crosschem_results.json'}")


if __name__ == "__main__":
    main()
