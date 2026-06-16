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

    W, H = 13.0, 8.0
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis("off")

    def txt(x, y, s, fs=8.4, ha="left", c="0.15", w=None):
        ax.text(x, y, s, fontsize=fs, ha=ha, va="top", color=c, fontweight=w)

    def fr(x, y, w, h):
        return [x / W, y / H, w / W, h / H]

    # ===== top banner: evaluation protocol governs every stage =====
    ax.add_patch(FancyBboxPatch((0.25, 7.05), 12.5, 0.80, boxstyle="round,pad=0.05",
                                fc="#e9e6f4", ec="#7a6ca5", lw=1.2, zorder=1))
    ax.text(6.5, 7.63, "Evaluation protocol", ha="center", fontsize=12, fontweight="bold")
    for lab, x in [("cell-level splits", 2.0), ("batch-aware transfer", 5.2),
                   ("multi-seed reporting", 8.35), ("LOCO validation", 11.2)]:
        ax.text(x, 7.26, lab, ha="center", fontsize=8.8)
    for xsep in (3.6, 6.78, 9.75):
        ax.plot([xsep, xsep], [7.12, 7.42], color="0.55", lw=0.8, zorder=2)

    # ===== 4 main-pipeline zones =====
    ZONES = [
        (0.25, 2.9, "#dbe9f6", "1  Public datasets"),
        (3.45, 2.9, "#d8f0d8", "2  Physically grounded curve features"),
        (6.65, 2.9, "#fdf0d0", "3  Prediction models"),
        (9.85, 2.9, "#f3dcec", "4  Calibrated outputs"),
    ]
    for x, w, c, t in ZONES:
        ax.add_patch(FancyBboxPatch((x, 1.95), w, 4.85, boxstyle="round,pad=0.06",
                                    fc=c, ec="0.45", lw=1.1, zorder=0))
        ax.text(x + w / 2, 6.62, t, ha="center", fontsize=9.8, fontweight="bold")
        ax.add_patch(FancyArrowPatch((x + w / 2, 7.03), (x + w / 2, 6.84), arrowstyle="-|>",
                                     mutation_scale=10, lw=1.0, color="#7a6ca5", ls="--", zorder=3))

    # ---- Zone 1: datasets + capacity-fade thumbnail ----
    txt(0.5, 6.20, "Severson/MATR (LFP)\nprimary benchmark", 8.2)
    txt(0.5, 5.35, "NASA PCoE (LCO)\nsecond-dataset validation", 8.2)
    txt(0.5, 4.50, "CALCE CS2 (LCO)\ntransfer stress test", 8.2)
    a1 = fig.add_axes(fr(0.62, 2.20, 2.25, 1.55))
    for c_, d in list(soh.groupby("cell"))[2:9]:
        a1.plot(d.cycle, d.SOH * 100, lw=0.9, alpha=0.85)
    a1.set_xlabel("cycle", fontsize=7); a1.set_ylabel("Capacity (%)", fontsize=7); a1.tick_params(labelsize=6)

    # ---- Zone 2: features + IC thumbnail ----
    txt(3.7, 6.18, "summary HIs", 8.4)
    txt(3.7, 5.62, "charge/discharge descriptors", 8.4)
    txt(3.7, 5.06, "Q(V) features", 8.4)
    txt(3.7, 4.50, "IC features", 8.4)
    a2 = fig.add_axes(fr(3.82, 2.20, 2.25, 1.55))
    a2.plot(v_ic, ic, color="#2c7fb8", lw=1.4)
    k = int(np.argmax(ic))
    a2.annotate("IC peak", (v_ic[k], ic[k]), xytext=(v_ic[k] - 0.6, ic[k] * 0.78),
                fontsize=6.5, arrowprops=dict(arrowstyle="->", lw=0.8))
    a2.set_xlim(2.4, 3.5); a2.set_xlabel("Voltage (V)", fontsize=7); a2.set_ylabel("|dQ/dV|", fontsize=7); a2.tick_params(labelsize=6)

    # ---- Zone 3: models + tasks ----
    for yy, s in [(6.18, "Ridge"), (5.66, "SVR"), (5.14, "Random forest")]:
        ax.text(8.1, yy, s, ha="center", va="top", fontsize=9)
    ax.plot([7.15, 9.05], [4.55, 4.55], color="#b8860b", lw=0.9, ls="--")
    ax.text(8.1, 4.18, "SOH prediction", ha="center", va="top", fontsize=8.6, style="italic")
    ax.text(8.1, 3.66, "RUL prediction", ha="center", va="top", fontsize=8.6, style="italic")

    # ---- Zone 4: outputs + SOH-with-interval thumbnail ----
    txt(10.1, 6.18, "SOH estimate", 8.4)
    txt(10.1, 5.62, "RUL estimate", 8.4)
    txt(10.1, 5.06, "split-conformal prediction interval", 7.8)
    txt(10.1, 4.50, "failure-boundary awareness", 8.2)
    a4 = fig.add_axes(fr(10.05, 2.20, 2.45, 1.55))
    one = soh[soh.cell == soh.cell.unique()[7]].sort_values("cycle")
    a4.fill_between(one.cycle, (one.SOH - 0.03) * 100, (one.SOH + 0.03) * 100, color="#5b8ff9", alpha=0.25, label="prediction interval")
    a4.plot(one.cycle, one.SOH * 100, "k-", lw=1.2, label="SOH")
    a4.axvline(one.cycle.iloc[-1], color="0.4", ls=":", lw=0.9)
    a4.annotate("RUL", (one.cycle.iloc[-1], 62), xytext=(one.cycle.iloc[-1] * 0.5, 62),
                fontsize=6.5, arrowprops=dict(arrowstyle="->", lw=0.8))
    a4.set_xlabel("cycle", fontsize=7); a4.set_ylabel("SOH (%)", fontsize=7); a4.tick_params(labelsize=6)
    a4.legend(fontsize=5.5, loc="lower left", frameon=False)

    # ---- solid arrows between zones (main pipeline) ----
    for x0, x1 in [(3.15, 3.45), (6.35, 6.65), (9.55, 9.85)]:
        ax.add_patch(FancyArrowPatch((x0, 4.4), (x1, 4.4), arrowstyle="-|>",
                                     mutation_scale=17, lw=2.2, color="0.25", zorder=3))

    # ===== bottom: diagnostic studies (separate, not deployed) =====
    ax.add_patch(FancyBboxPatch((0.25, 0.25), 12.5, 1.45, boxstyle="round,pad=0.05",
                                fc="white", ec="0.3", lw=1.2, ls="--", zorder=1))
    ax.text(6.5, 1.55, "Diagnostic studies (not part of deployed pipeline)", ha="center",
            fontsize=10.5, fontweight="bold")
    diags = [("1. Active learning", "tests label efficiency"),
             ("2. Symbolic regression", "interpretable functional\nrelationship"),
             ("3. Cross-batch drift", "MMD + transfer analysis"),
             ("4. Adaptive conformal", "recalibration under drift"),
             ("5. Cross-system transfer", "LFP-to-LCO boundary test")]
    dw = 2.34
    for j, (title_, sub) in enumerate(diags):
        dx = 0.45 + j * (dw + 0.05)
        ax.add_patch(FancyBboxPatch((dx, 0.40), dw, 0.82, boxstyle="round,pad=0.04",
                                    fc="#f7f7f7", ec="0.55", lw=0.9, zorder=2))
        ax.text(dx + 0.12, 1.10, title_, fontsize=8.0, fontweight="bold", va="top")
        ax.text(dx + 0.12, 0.82, sub, fontsize=6.6, va="top", color="0.25")

    fig.savefig(HERE / "fig1_framework.png", bbox_inches="tight")
    plt.close(fig); print("Fig 1 (two-layer redesign) -> fig1_framework.png")


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
