r"""
图升级 v2(真实数据, 统一样式):
  Fig 1  框架图重设计: 四色块分区 + 嵌真实数据缩略图 + 独立"评估协议层" + 主/诊断双色箭头
  Fig 3  Roman 式四件套: (a)SOH轨迹+区间 (b)reliability校准曲线 (c)parity (d)误差直方图
  Fig 8  迁移矩阵 R² 热图 × MMD 热图 双面板(升主文)

运行: .venv\Scripts\python.exe make_figures_v2.py
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
from sklearn.model_selection import GroupShuffleSplit
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

warnings.filterwarnings("ignore")
HERE = Path(__file__).parent
DATA = HERE / "data"

# ---------- 统一样式 ----------
plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 9.5, "axes.labelsize": 9,
    "legend.fontsize": 7.5, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "axes.linewidth": 0.8, "lines.linewidth": 1.6,
    "figure.dpi": 300, "savefig.dpi": 300,
})
PANEL = dict(fontsize=11, fontweight="bold", va="top", ha="left")


def panel_label(ax, s, dx=-0.12, dy=1.08):
    ax.text(dx, dy, s, transform=ax.transAxes, **PANEL)


# ================================================================
# Fig 1 框架图重设计
# ================================================================
def fig1():
    soh = pd.read_csv(DATA / "severson_soh.csv")
    # IC 缩略图数据(batch1 cell5 中期循环, 懒加载只读两个数据集)
    with h5py.File(DATA / "2017-05-12_batchdata_updated_struct_errorcorrect.mat", "r") as f:
        b = f["batch"]; i = 5
        vd = np.array(f[b["Vdlin"][i, 0]]).flatten().astype(float)
        cyc_ds = f[b["cycles"][i, 0]]
        q = np.array(f[cyc_ds["Qdlin"][400, 0]]).flatten().astype(float)
    m = np.isfinite(vd) & np.isfinite(q)
    v_ic, q_ic = vd[m], q[m]
    o = np.argsort(v_ic); v_ic, q_ic = v_ic[o], q_ic[o]
    ic = np.abs(np.gradient(q_ic, v_ic))
    from scipy.signal import savgol_filter
    ic = savgol_filter(ic, 51, 3)

    fig = plt.figure(figsize=(13, 6.4))
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 13); ax.set_ylim(0, 6.4); ax.axis("off")

    ZONES = [  # (x, w, color, title)
        (0.25, 2.9, "#dbe9f6", "1  Public datasets"),
        (3.45, 2.9, "#d8f0d8", "2  Curve features (15 HIs)"),
        (6.65, 2.9, "#fdf0d0", "3  Models + evaluation protocol"),
        (9.85, 2.9, "#f3dcec", "4  Calibrated outputs"),
    ]
    for x, w, c, t in ZONES:
        ax.add_patch(FancyBboxPatch((x, 0.45), w, 5.45, boxstyle="round,pad=0.06",
                                    fc=c, ec="0.45", lw=1.1, zorder=0))
        ax.text(x + w / 2, 5.62, t, ha="center", fontsize=10.5, fontweight="bold")

    def txt(x, y, s, fs=8.4, ha="left", c="0.15", w=None):
        ax.text(x, y, s, fontsize=fs, ha=ha, va="top", color=c, fontweight=w)

    # ---- Zone 1: 数据 + 容量衰减缩略图 ----
    txt(0.5, 5.25, "Severson / MATR (LFP, 1.1 Ah)\n182 parsed cells\n140 for SOH | 128 for RUL\n16,800 / 15,046 cycle samples")
    txt(0.5, 3.6, "NASA PCoE (LCO, 2 Ah)\n4 cells, 625 cycle samples")
    a1 = fig.add_axes([0.045, 0.10, 0.155, 0.27])
    for c_, d in list(soh.groupby("cell"))[2:9]:
        a1.plot(d.cycle, d.SOH, lw=0.9, alpha=0.85)
    a1.set_xlabel("cycle", fontsize=7); a1.set_ylabel("SOH", fontsize=7)
    a1.tick_params(labelsize=6); a1.set_title("capacity fade (real cells)", fontsize=7.5)

    # ---- Zone 2: 特征 + IC 缩略图 ----
    txt(3.7, 5.25, "Summary HIs (5)\ncharge time · IR · temperature", 8.2)
    txt(3.7, 4.45, "Curve shape (6)\ncapacity fractions @ V thresholds", 8.2)
    txt(3.7, 3.65, "Incremental capacity (4)\npeak · position · FWHM · area", 8.2)
    a2 = fig.add_axes([0.292, 0.10, 0.155, 0.27])
    a2.plot(v_ic, ic, color="#2c7fb8", lw=1.4)
    k = int(np.argmax(ic))
    a2.annotate("IC peak", (v_ic[k], ic[k]), xytext=(v_ic[k] - 0.55, ic[k] * 0.82),
                fontsize=7, arrowprops=dict(arrowstyle="->", lw=0.8))
    a2.set_xlim(2.4, 3.5); a2.set_xlabel("V", fontsize=7); a2.set_ylabel("|dQ/dV|", fontsize=7)
    a2.tick_params(labelsize=6); a2.set_title("IC curve (real cycle)", fontsize=7.5)

    # ---- Zone 3: 模型 + 协议层(独立色带) ----
    txt(6.9, 5.25, "Ridge  ·  SVR-RBF  ·  Random forest", 8.6, w="bold")
    txt(6.9, 4.85, "+ split-conformal intervals", 8.2)
    ax.add_patch(FancyBboxPatch((6.85, 2.55), 2.5, 1.85, boxstyle="round,pad=0.05",
                                fc="white", ec="#b8860b", lw=1.3, zorder=2))
    txt(7.0, 4.28, "Evaluation protocol", 8.6, w="bold", c="#7a5b00")
    txt(7.0, 3.94, "cell-level splits (no leakage)\nbatch-aware transfer tests\nmulti-seed mean ± s.d.\nLOCO on second dataset", 7.8)
    ax.add_patch(FancyBboxPatch((6.85, 0.62), 2.5, 1.7, boxstyle="round,pad=0.05",
                                fc="#fff7e6", ec="#c87820", lw=1.1, ls="--", zorder=2))
    txt(7.0, 2.20, "Diagnostics", 8.6, w="bold", c="#a04a00")
    txt(7.0, 1.86, "active learning vs random\nsymbolic regression law\nMMD drift quantification\nadaptive conformal (ACI)", 7.8)

    # ---- Zone 4: 输出 + 区间缩略图 ----
    txt(10.1, 5.25, "SOH:  RMSPE 1.26 ± 0.26%\nRUL:  R² 0.87 (no capacity label)", 8.4)
    txt(10.1, 4.4, "90% prediction intervals\ncoverage 90.0% (SOH)\nACI keeps coverage under drift", 8.2)
    a4 = fig.add_axes([0.785, 0.10, 0.16, 0.27])
    one = soh[soh.cell == soh.cell.unique()[7]].sort_values("cycle")
    a4.fill_between(one.cycle, one.SOH - 0.015, one.SOH + 0.015, color="red", alpha=0.18)
    a4.plot(one.cycle, one.SOH, "k-", lw=1.2)
    a4.set_xlabel("cycle", fontsize=7); a4.set_ylabel("SOH", fontsize=7)
    a4.tick_params(labelsize=6); a4.set_title("interval to scale (±1.5%)", fontsize=7.5)

    # ---- 箭头: 主流程(实线深灰) + 诊断(虚线橙) ----
    for x0, x1 in [(3.15, 3.45), (6.35, 6.65), (9.55, 9.85)]:
        ax.add_patch(FancyArrowPatch((x0, 3.4), (x1, 3.4), arrowstyle="-|>",
                                     mutation_scale=18, lw=2.2, color="0.25", zorder=3))
    ax.add_patch(FancyArrowPatch((8.1, 2.55), (8.1, 2.32), arrowstyle="-|>",
                                 mutation_scale=12, lw=1.4, color="#c87820", ls="--", zorder=3))
    fig.savefig(HERE / "fig1_framework.png", bbox_inches="tight")
    plt.close(fig); print("Fig 1 (redesigned) -> fig1_framework.png")


# ================================================================
# Fig 3 四件套: 轨迹 / reliability / parity / 误差直方图
# ================================================================
def fig3():
    df = pd.read_csv(DATA / "severson_soh.csv")
    F = ["chargetime", "IR", "Tavg", "Tmax", "Tmin", "frac_3p3", "frac_3p2",
         "frac_3p1", "frac_3p0", "frac_2p8", "v_median", "ic_peak", "v_ic_peak", "ic_fwhm", "ic_area"]
    X = df[F].to_numpy(); y = df.SOH.to_numpy(); g = df.cell.to_numpy()
    tr, te = next(GroupShuffleSplit(1, test_size=0.30, random_state=0).split(X, y, g))
    p_l, c_l = next(GroupShuffleSplit(1, test_size=0.25, random_state=1).split(X[tr], y[tr], g[tr]))
    ptr, cal = tr[p_l], tr[c_l]
    rf = RandomForestRegressor(400, min_samples_leaf=2, n_jobs=-1, random_state=0).fit(X[ptr], y[ptr])
    res_cal = np.sort(np.abs(y[cal] - rf.predict(X[cal])))
    p_te = rf.predict(X[te]); err = p_te - y[te]
    tp = np.stack([t.predict(X[te]) for t in rf.estimators_])

    fig, axes = plt.subplots(2, 2, figsize=(10.5, 8))
    (aA, aB), (aC, aD) = axes

    # (a) 轨迹 + 90% conformal 带 (4 个测试芯, 每芯独立着色, 图例嵌 per-cell RMSPE)
    n = len(res_cal); q90 = res_cal[min(n - 1, int(np.ceil((n + 1) * 0.9)) - 1)]
    dte = df.iloc[te].copy(); dte["pred"] = p_te
    cell_colors = ["C0", "C1", "C2", "C3"]
    for c_, cc in zip(list(dte.cell.unique())[:4], cell_colors):
        d = dte[dte.cell == c_].sort_values("cycle")
        rms = np.sqrt(np.mean(((d.pred - d.SOH) / d.SOH) ** 2)) * 100
        aA.fill_between(d.cycle, d.pred - q90, d.pred + q90, alpha=0.13, color=cc)
        aA.plot(d.cycle, d.SOH, "-", lw=1.2, color="k", alpha=0.75)
        aA.plot(d.cycle, d.pred, ".", ms=2.6, color=cc, label=f"cell {c_.split('_')[-1]}: RMSPE {rms:.2f}%")
    aA.plot([], [], "k-", lw=1.2, label="measured SOH")
    aA.set_xlabel("Cycle"); aA.set_ylabel("SOH")
    aA.set_title("Test-cell trajectories with 90% prediction intervals")
    aA.legend(fontsize=7, loc="lower left")
    aA.grid(alpha=0.3); panel_label(aA, "(a)")

    # (b) reliability: expected vs observed coverage
    levels = np.arange(0.50, 1.0, 0.05)
    cov_cp, cov_tq = [], []
    for lv in levels:
        k = min(n - 1, int(np.ceil((n + 1) * lv)) - 1)
        q = res_cal[k]
        cov_cp.append(np.mean(np.abs(y[te] - p_te) <= q))
        lo = np.percentile(tp, (1 - lv) / 2 * 100, axis=0)
        hi = np.percentile(tp, (1 + lv) / 2 * 100, axis=0)
        cov_tq.append(np.mean((y[te] >= lo) & (y[te] <= hi)))
    aB.plot([0.45, 1], [0.45, 1], "k--", lw=1, label="Ideal")
    aB.plot(levels, cov_cp, "o-", color="C0", label="Split-conformal")
    aB.plot(levels, cov_tq, "s--", color="C1", label="Tree-quantile")
    aB.set_xlabel("Nominal coverage"); aB.set_ylabel("Observed coverage")
    aB.set_title("Reliability of prediction intervals")
    aB.legend(); aB.grid(alpha=0.3); panel_label(aB, "(b)")

    # (c) parity
    aC.scatter(y[te], p_te, s=4, alpha=0.25, color="C0", edgecolors="none")
    lim = [min(y[te].min(), p_te.min()), max(y[te].max(), p_te.max())]
    aC.plot(lim, lim, "r--", lw=1.2)
    aC.set_xlabel("True SOH"); aC.set_ylabel("Predicted SOH")
    aC.set_title(f"Parity (R² = {r2_score(y[te], p_te):.3f}, unseen cells)")
    aC.grid(alpha=0.3); panel_label(aC, "(c)")

    # (d) 误差直方图
    aD.hist(err * 100, bins=60, color="C0", alpha=0.8, edgecolor="white", lw=0.3)
    aD.axvline(0, color="r", ls="--", lw=1.2)
    aD.set_xlabel("Prediction error (% SOH)"); aD.set_ylabel("Count")
    aD.set_title(f"Error distribution (mean {err.mean()*100:+.2f}%, s.d. {err.std()*100:.2f}%)")
    aD.grid(alpha=0.3); panel_label(aD, "(d)")

    fig.tight_layout()
    fig.savefig(HERE / "fig3_soh_panels.png", bbox_inches="tight")
    plt.close(fig); print("Fig 3 (4-panel) -> fig3_soh_panels.png")


# ================================================================
# Fig 8 迁移矩阵 R² × MMD 双热图
# ================================================================
def fig8():
    mm = pd.read_csv(DATA / "supp_drift_mmd.csv")
    feats = pd.read_csv(DATA / "severson_features_full.csv")
    F = ["var_dQ", "min_dQ", "mean_dQ", "qd_slope", "qd_intercept",
         "qd2", "qd_max_minus_qd2", "chargetime_5", "temp_integral"]
    feats = feats.dropna(subset=F + ["cycle_life", "batch"])
    BMAP = {"2017-05-12": "b1", "2017-06-30": "b2", "2018-04-12": "b3", "2019-01-24": "b4"}
    feats["b"] = feats.batch.map(BMAP)
    order = ["b1", "b2", "b3", "b4"]

    R = pd.DataFrame(index=order, columns=order, dtype=float)
    M = pd.DataFrame(0.0, index=order, columns=order)
    for _, r in mm.iterrows():
        R.loc[r.train, r.test] = r.r2; M.loc[r.train, r.test] = r.mmd
    # 对角线 = 批内 cell-level 70/30 (3 seeds 均值)
    for bb in order:
        d = feats[feats.b == bb]
        Xb = d[F].to_numpy(); yb = np.log(d.cycle_life.to_numpy())
        r2s = []
        for s in range(3):
            idx = np.random.default_rng(s).permutation(len(d)); nt = max(6, int(len(d) * 0.3))
            te_, tr_ = idx[:nt], idx[nt:]
            m_ = RandomForestRegressor(300, min_samples_leaf=2, n_jobs=-1, random_state=0).fit(Xb[tr_], yb[tr_])
            r2s.append(r2_score(yb[te_], m_.predict(Xb[te_])))
        R.loc[bb, bb] = np.mean(r2s)

    fig, (aL, aR) = plt.subplots(1, 2, figsize=(11.5, 4.8))
    im1 = aL.imshow(R.values.astype(float), cmap="RdYlGn", vmin=-1.0, vmax=1.0)
    for i in range(4):
        for j in range(4):
            v = R.values[i, j]
            aL.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=9,
                    fontweight="bold" if i == j else "normal",
                    color="black" if v > -0.3 else "white")
    aL.set_xticks(range(4), order); aL.set_yticks(range(4), order)
    aL.set_xlabel("Test batch"); aL.set_ylabel("Train batch")
    aL.set_title("Transfer R² (diagonal: within-batch CV)")
    fig.colorbar(im1, ax=aL, fraction=0.046, label="R²"); panel_label(aL, "(a)")

    im2 = aR.imshow(M.values.astype(float), cmap="viridis")
    for i in range(4):
        for j in range(4):
            aR.text(j, i, f"{M.values[i,j]:.2f}", ha="center", va="center", fontsize=9,
                    color="white" if M.values[i, j] < 0.5 else "black")
    aR.set_xticks(range(4), order); aR.set_yticks(range(4), order)
    aR.set_xlabel("Test batch"); aR.set_ylabel("Train batch")
    aR.set_title("Feature-distribution distance (MMD)")
    fig.colorbar(im2, ax=aR, fraction=0.046, label="MMD"); panel_label(aR, "(b)")

    fig.tight_layout()
    fig.savefig(HERE / "fig8_transfer_mmd.png", bbox_inches="tight")
    plt.close(fig); print("Fig 8 (transfer x MMD) -> fig8_transfer_mmd.png")


if __name__ == "__main__":
    fig1(); fig3(); fig8()
    print("\n全部 3 张升级图完成 [OK]")
