r"""
图升级 v3(审稿人 4-8 项):
  S3  特征-SOH Spearman 相关性热图(总体+分批次)        -> figS3_corr_heatmap.png
  S4  RUL 误差-时域图(分箱 MAE + p90|err| vs 固定区间) -> figS4_rul_horizon.png
  F2  Fig 2 加标注层: (a)IC峰/FWHM/面积窗 (b)ΔQ (c)阈值特征定义 -> fig2_ic_curves.png
  GA  Graphical abstract                                  -> graphical_abstract.png
运行: .venv\Scripts\python.exe make_figures_v3.py
"""
import warnings
import numpy as np
import pandas as pd
import h5py
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from scipy.signal import savgol_filter
from scipy.stats import spearmanr
from sklearn.model_selection import GroupShuffleSplit
from sklearn.ensemble import RandomForestRegressor

warnings.filterwarnings("ignore")
HERE = Path(__file__).parent
DATA = HERE / "data"
plt.rcParams.update({"font.size": 9, "axes.titlesize": 9.5, "legend.fontsize": 7.5,
                     "xtick.labelsize": 8, "ytick.labelsize": 8, "savefig.dpi": 300})
PANEL = dict(fontsize=11, fontweight="bold", va="top", ha="left")
FEAT = ["chargetime", "IR", "Tavg", "Tmax", "Tmin", "frac_3p3", "frac_3p2",
        "frac_3p1", "frac_3p0", "frac_2p8", "v_median", "ic_peak", "v_ic_peak", "ic_fwhm", "ic_area"]


# ============ S3 相关性热图 ============
def s3_corr():
    df = pd.read_csv(DATA / "severson_soh.csv")
    df["batch"] = df.cell.str[:10]
    cols = ["all", "2017-05-12", "2017-06-30", "2018-04-12"]
    M = pd.DataFrame(index=FEAT, columns=cols, dtype=float)
    for f in FEAT:
        M.loc[f, "all"] = spearmanr(df[f], df.SOH).correlation
        for b in cols[1:]:
            d = df[df.batch == b]
            M.loc[f, b] = spearmanr(d[f], d.SOH).correlation
    fig, ax = plt.subplots(figsize=(6.4, 7.2))
    im = ax.imshow(M.values.astype(float), cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    for i in range(len(FEAT)):
        for j in range(len(cols)):
            v = M.values[i, j]
            ax.text(j, i, f"{v:+.2f}", ha="center", va="center", fontsize=7.8,
                    color="white" if abs(v) > 0.6 else "black")
    ax.set_xticks(range(4), ["All", "Batch 1", "Batch 2", "Batch 3"])
    ax.set_yticks(range(len(FEAT)), FEAT)
    ax.set_title("Spearman correlation of each HI with SOH")
    fig.colorbar(im, ax=ax, fraction=0.046, label="Spearman rho")
    fig.tight_layout(); fig.savefig(HERE / "figS3_corr_heatmap.png", bbox_inches="tight")
    plt.close(fig); print("S3 -> figS3_corr_heatmap.png")


# ============ S4 RUL 误差-时域 ============
def s4_rul_horizon():
    df = pd.read_csv(DATA / "severson_soh.csv").sort_values(["cell", "cycle"]).reset_index(drop=True)
    EOL_CAP = 0.88
    out = []
    for cell, d in df.groupby("cell"):
        d = d.sort_values("cycle"); cyc = d.cycle.to_numpy(float); cap = d.cap.to_numpy(float)
        if cap.min() > 0.90 or len(d) < 8:
            continue
        if (cap <= EOL_CAP).any():
            k = int(np.argmax(cap <= EOL_CAP))
            eol = cyc[0] if k == 0 else cyc[k-1] + (EOL_CAP - cap[k-1]) * (cyc[k]-cyc[k-1]) / (cap[k]-cap[k-1])
        else:
            eol = cyc[-1]
        dd = d.copy(); dd["RUL"] = eol - cyc; out.append(dd[dd.RUL >= 0])
    data = pd.concat(out, ignore_index=True)
    g = data.cell.to_numpy()
    sl = np.zeros(len(data))
    for c_, idx in data.groupby("cell").groups.items():
        sub = data.loc[idx].sort_values("cycle")
        gr = np.gradient(sub.SOH.to_numpy(float), sub.cycle.to_numpy(float))
        sl[data.index.get_indexer(sub.index)] = pd.Series(gr).rolling(5, min_periods=1, center=True).mean()
    data["soh_slope"] = sl
    F = FEAT + ["SOH", "soh_slope"]
    X = data[F].to_numpy(); y = data.RUL.to_numpy()
    tr, te = next(GroupShuffleSplit(1, test_size=0.30, random_state=0).split(X, y, g))
    pl, cl = next(GroupShuffleSplit(1, test_size=0.25, random_state=1).split(X[tr], y[tr], g[tr]))
    rf = RandomForestRegressor(400, min_samples_leaf=2, n_jobs=-1, random_state=0).fit(X[tr][pl], y[tr][pl])
    res = np.sort(np.abs(y[tr][cl] - rf.predict(X[tr][cl])))
    n = len(res); q90 = res[min(n-1, int(np.ceil((n+1)*0.9))-1)]
    p = rf.predict(X[te]); err = np.abs(p - y[te]); rul = y[te]

    bins = [(0, 100), (100, 300), (300, 700), (700, 2000)]
    labels = [f"{a}-{b}" for a, b in bins]
    mae = [err[(rul >= a) & (rul < b)].mean() for a, b in bins]
    p90 = [np.percentile(err[(rul >= a) & (rul < b)], 90) for a, b in bins]
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    xs = np.arange(len(bins))
    ax.bar(xs, mae, 0.55, color="#4c72b0", label="MAE per horizon bin")
    ax.plot(xs, p90, "o-", color="#dd8452", lw=1.8, label="90th-percentile |error|")
    ax.axhline(q90, color="r", ls="--", lw=1.4, label=f"Fixed conformal half-width ({q90:.0f} cyc)")
    ax.set_xticks(xs, labels); ax.set_xlabel("True RUL horizon (cycles)")
    ax.set_ylabel("Absolute error (cycles)")
    ax.set_title("RUL error grows with horizon; a fixed interval is conservative near EOL")
    ax.legend(); ax.grid(alpha=0.3, axis="y")
    fig.tight_layout(); fig.savefig(HERE / "figS4_rul_horizon.png", bbox_inches="tight")
    plt.close(fig); print(f"S4 -> figS4_rul_horizon.png (q90={q90:.0f})")


# ============ Fig 2 标注层 ============
def fig2_annotated():
    with h5py.File(DATA / "2017-05-12_batchdata_updated_struct_errorcorrect.mat", "r") as f:
        b = f["batch"]; i = 5
        qd = np.array(f[b["summary"][i, 0]]["QDischarge"]).flatten().astype(float)
        vd = np.array(f[b["Vdlin"][i, 0]]).flatten().astype(float)
        cyc_ds = f[b["cycles"][i, 0]]
        valid = np.where(np.isfinite(qd) & (qd > 0.5))[0]
        valid = valid[valid < cyc_ds["Qdlin"].shape[0]]
        q0 = np.median(qd[valid[:10]])
        # 曲线族: 全寿命均匀取 ~24 个循环, 按循环数渐变着色 (借鉴 Ren et al. Fig.2 的连续渐变)
        idxs = [int(valid[int(fr)]) for fr in np.linspace(0, valid.size - 1, 24)]
        curves = []
        for j in idxs:
            q = np.array(f[cyc_ds["Qdlin"][j, 0]]).flatten().astype(float)
            m = np.isfinite(vd) & np.isfinite(q)
            v, qq = vd[m], q[m]; o = np.argsort(v)
            curves.append((j, qd[j] / q0, v[o], qq[o]))

    from matplotlib import cm, colors as mcolors
    norm = mcolors.Normalize(vmin=min(c[0] for c in curves), vmax=max(c[0] for c in curves))
    sm = cm.ScalarMappable(norm=norm, cmap="viridis")
    fig, (aA, aB, aC) = plt.subplots(1, 3, figsize=(13.6, 4.3))
    colors = [sm.to_rgba(c[0]) for c in curves]
    # (a) IC 曲线族(渐变) + 标注
    ics = []
    for (j, soh, v, q), c in zip(curves, colors):
        ic = np.clip(savgol_filter(np.abs(np.gradient(q, v)), 51, 3), 0, None)
        ics.append(ic)
        aA.plot(v, ic, color=c, lw=0.9, alpha=0.9)
    fig.colorbar(sm, ax=aA, fraction=0.05, pad=0.02, label="Cycle number")
    v0, ic0 = curves[0][2], ics[0]
    k = int(np.argmax(ic0))
    aA.annotate("peak height", (v0[k], ic0[k]), xytext=(v0[k] - 0.62, ic0[k] * 0.86),
                fontsize=8, arrowprops=dict(arrowstyle="->", lw=0.9))
    aA.plot([v0[k], v0[k]], [0, ic0[k]], ls=":", color="0.4", lw=1)
    aA.text(v0[k] + 0.015, 0.25, "peak position", fontsize=7.5, rotation=90, color="0.35")
    half = ic0[k] / 2; ab = v0[ic0 >= half]
    aA.annotate("", (ab.min(), half), (ab.max(), half), arrowprops=dict(arrowstyle="<->", lw=1, color="0.3"))
    aA.text((ab.min() + ab.max()) / 2, half * 1.1, "FWHM", fontsize=8, ha="center")
    aA.axvspan(3.15, 3.35, color="green", alpha=0.10)
    aA.text(3.25, ic0.max() * 0.55, "IC area\nwindow", fontsize=7.5, ha="center", color="green")
    aA.set_xlim(2.6, 3.5); aA.set_xlabel("Voltage (V)"); aA.set_ylabel("IC |dQ/dV| (Ah/V)")
    aA.set_title("IC curve family over ageing"); aA.grid(alpha=0.3)
    aA.text(-0.13, 1.07, "(a)", transform=aA.transAxes, **PANEL)

    # (b) ΔQ(V) 曲线族(渐变)
    vref, qref = curves[0][2], curves[0][3]
    for (j, soh, v, q), c in zip(curves, colors):
        aB.plot(v, q - np.interp(v, vref, qref), color=c, lw=0.9, alpha=0.9)
    fig.colorbar(sm, ax=aB, fraction=0.05, pad=0.02, label="Cycle number")
    aB.set_xlim(2.0, 3.5); aB.set_xlabel("Voltage (V)")
    aB.set_ylabel(r"$\Delta Q(V) = Q_{cyc} - Q_{early}$  (Ah)")
    aB.set_title(r"$\Delta Q(V)$ grows with ageing"); aB.grid(alpha=0.3)
    aB.text(-0.13, 1.07, "(b)", transform=aB.transAxes, **PANEL)

    # (c) 阈值特征定义
    j, soh, v, q = curves[0]
    frac = q / q.max()
    aC.plot(v, frac, "k-", lw=1.7)
    for thr in [3.3, 3.2, 3.1, 3.0, 2.8]:
        fr = float(np.interp(thr, v, frac))
        aC.plot([thr, thr], [0, fr], ls="--", color="C0", lw=0.9)
        aC.plot([v.min(), thr], [fr, fr], ls="--", color="C0", lw=0.9)
        aC.plot(thr, fr, "o", ms=4, color="C0")
        aC.text(thr, -0.06, f"{thr}", fontsize=7, ha="center", color="C0")
    vm = float(np.interp(0.5, frac[::-1], v[::-1]))   # frac 随 v 递减, 需反转后插值
    aC.plot(vm, 0.5, "s", ms=6, color="C3")
    aC.annotate("median voltage\n(50% capacity)", (vm, 0.5), xytext=(vm - 0.75, 0.62),
                fontsize=7.5, arrowprops=dict(arrowstyle="->", lw=0.9, color="C3"), color="C3")
    aC.set_xlim(2.0, 3.5); aC.set_ylim(-0.12, 1.05)
    aC.set_xlabel("Voltage (V)"); aC.set_ylabel("Fraction of capacity released")
    aC.set_title("Curve-shape features (fractions @ thresholds)"); aC.grid(alpha=0.3)
    aC.text(-0.13, 1.07, "(c)", transform=aC.transAxes, **PANEL)

    fig.tight_layout(); fig.savefig(HERE / "fig2_ic_curves.png", bbox_inches="tight")
    plt.close(fig); print("Fig 2 (annotated, 3 panels) -> fig2_ic_curves.png")


# ============ Graphical abstract ============
def graphical_abstract():
    fig = plt.figure(figsize=(11, 4.4))
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 11); ax.set_ylim(0, 4.4); ax.axis("off")
    boxes = [
        (0.25, 2.0, 2.25, 1.9, "#dbe9f6", "Public datasets", "Severson LFP 182 cells\nNASA LCO 4 cells"),
        (2.85, 2.0, 2.3, 1.9, "#d8f0d8", "Curve features", "IC peak dominates\n(0.88 importance)"),
        (5.5, 2.0, 2.5, 1.9, "#fdf0d0", "Strict protocols", "cell-level / batch-aware\nmulti-seed / LOCO"),
        (8.35, 2.0, 2.4, 1.9, "#f3dcec", "Calibrated outputs", "SOH 1.26% RMSPE\nRUL R² 0.87 | 90% PI"),
    ]
    for x, y, w, h, c, t, s in boxes:
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.07", fc=c, ec="0.4", lw=1.2))
        ax.text(x + w / 2, y + h - 0.32, t, ha="center", fontsize=10.5, fontweight="bold")
        ax.text(x + w / 2, y + h / 2 - 0.22, s, ha="center", fontsize=8.6)
    for x0, x1 in [(2.5, 2.85), (5.15, 5.5), (8.0, 8.35)]:
        ax.add_patch(FancyArrowPatch((x0, 2.95), (x1, 2.95), arrowstyle="-|>",
                                     mutation_scale=17, lw=2.2, color="0.25"))
    ax.add_patch(FancyBboxPatch((0.25, 0.35), 10.5, 1.25, boxstyle="round,pad=0.07",
                                fc="#fdeaea", ec="#c0392b", lw=1.3))
    ax.text(5.5, 1.28, "Rigorous evaluation exposes what single splits hide", ha="center",
            fontsize=10, fontweight="bold", color="#922b21")
    ax.text(5.5, 0.78, "active learning ≈ random sampling   ·   cross-batch transfer collapses (R² < 0)   ·   "
                       "static conformal 36% coverage → adaptive conformal restores 89%",
            ha="center", fontsize=8.8, color="#6e2c23")
    fig.savefig(HERE / "graphical_abstract.png", bbox_inches="tight")
    plt.close(fig); print("GA -> graphical_abstract.png")


if __name__ == "__main__":
    s3_corr(); s4_rul_horizon(); fig2_annotated(); graphical_abstract()
    print("\n4-8 项图全部完成 [OK]")
