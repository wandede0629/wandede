r"""
生成论文剩余 3 张图:
  Fig.1  框架总览示意图        -> fig1_framework.png
  Fig.2  IC(dQ/dV)随老化演化  -> fig2_ic_curves.png
  Fig.7  符号回归 parity 图     -> fig7_sr_parity.png

运行: .venv\Scripts\python.exe make_figures.py
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
from sklearn.metrics import r2_score

warnings.filterwarnings("ignore")
HERE = Path(__file__).parent
DATA = HERE / "data"


# ============ Fig.1 框架总览 ============
def fig1_framework():
    fig, ax = plt.subplots(figsize=(12, 6.2))
    ax.set_xlim(0, 12); ax.set_ylim(0, 7); ax.axis("off")

    def box(x, y, w, h, text, color):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.08",
                                    fc=color, ec="black", lw=1.3))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=9.5, wrap=True)

    def arrow(x0, y0, x1, y1):
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>",
                                     mutation_scale=16, lw=1.4, color="0.3"))

    box(0.2, 2.6, 2.2, 1.8, "Public datasets\n\nMATR/Severson (LFP)\nNASA PCoE (LCO)", "#dbe9f6")
    box(3.0, 2.6, 2.3, 1.8, "Feature extraction\n\nCharge/discharge\ncurves + IC (dQ/dV)", "#d8f0d8")
    box(6.0, 5.0, 2.6, 1.5, "Active learning\n(sample-efficient)", "#fde9c8")
    box(6.0, 2.9, 2.6, 1.5, "Models\nRidge / SVR / RF", "#fde9c8")
    box(6.0, 0.8, 2.6, 1.5, "Symbolic regression\n(interpretable law)", "#fde9c8")
    box(9.2, 3.6, 2.5, 1.5, "Conformal\nprediction interval\n(calibrated)", "#f6d8e4")
    box(9.2, 1.4, 2.5, 1.5, "Outputs:\nSOH  &  RUL", "#e8dbf6")

    arrow(2.4, 3.5, 3.0, 3.5)
    arrow(5.3, 3.5, 6.0, 5.6); arrow(5.3, 3.5, 6.0, 3.6); arrow(5.3, 3.5, 6.0, 1.5)
    arrow(8.6, 5.6, 9.4, 5.0); arrow(8.6, 3.6, 9.2, 4.2); arrow(8.6, 1.5, 9.4, 2.2)
    arrow(10.45, 3.6, 10.45, 2.9)
    ax.set_title("Unified framework for battery SOH/RUL prognostics", fontsize=12)
    fig.tight_layout(); fig.savefig(HERE / "fig1_framework.png", dpi=300); plt.close(fig)
    print("Fig.1 done -> fig1_framework.png")


# ============ Fig.2 IC 曲线 + ΔQ(V) 随老化(双面板) ============
def _qdlin(f, cyc_ds, vdlin, j):
    q = np.array(f[cyc_ds["Qdlin"][j, 0]]).flatten().astype(float)
    m = np.isfinite(vdlin) & np.isfinite(q)
    v, qq = vdlin[m], q[m]
    order = np.argsort(v)
    return v[order], qq[order]


def fig2_ic_curves():
    path = DATA / "2017-05-12_batchdata_updated_struct_errorcorrect.mat"
    with h5py.File(path, "r") as f:
        b = f["batch"]; n = b["cycle_life"].shape[0]
        # 选一个退化充分的电芯(最低 SOH 较小、循环数足够)
        best = None
        for i in range(n):
            qd = np.array(f[b["summary"][i, 0]]["QDischarge"]).flatten().astype(float)
            v = np.where(np.isfinite(qd) & (qd > 0.5))[0]
            if v.size < 100:
                continue
            q0 = np.median(qd[v[:10]])
            span = 1 - qd[v].min() / q0
            ncyc = f[b["cycles"][i, 0]]["Qdlin"].shape[0]
            if span > 0.16 and v.max() < ncyc:
                best = (i, qd, q0, v); break
        i, qd, q0, valid = best
        cyc_ds = f[b["cycles"][i, 0]]
        vdlin = np.array(f[b["Vdlin"][i, 0]]).flatten().astype(float)
        fracs = [0.02, 0.33, 0.66, 0.95]
        idxs = [int(valid[int(fr * (valid.size - 1))]) for fr in fracs]
        colors = plt.cm.viridis(np.linspace(0, 0.85, len(idxs)))

        fig, (axA, axB) = plt.subplots(1, 2, figsize=(12, 5))
        v0, q_early = _qdlin(f, cyc_ds, vdlin, idxs[0])
        for j, c in zip(idxs, colors):
            v, qq = _qdlin(f, cyc_ds, vdlin, j)
            soh = qd[j] / q0
            ic = np.abs(np.gradient(qq, v))
            w = min(51, (len(ic) // 2) * 2 - 1)
            if w >= 5:
                ic = savgol_filter(ic, w, 3)
            axA.plot(v, ic, color=c, lw=2, label=f"cyc {j}  (SOH={soh:.2f})")
            dq = qq - np.interp(v, v0, q_early)
            axB.plot(v, dq, color=c, lw=2, label=f"cyc {j}  (SOH={soh:.2f})")
        axA.set_xlabel("Voltage (V)"); axA.set_ylabel("IC  |dQ/dV|  (Ah/V)")
        axA.set_title("(a) IC curve evolution"); axA.legend(fontsize=8); axA.grid(alpha=0.3); axA.set_xlim(2.0, 3.5)
        axB.set_xlabel("Voltage (V)"); axB.set_ylabel(r"$\Delta Q(V)=Q_{cyc}-Q_{early}$  (Ah)")
        axB.set_title(r"(b) $\Delta Q(V)$ grows with aging  → var($\Delta Q$) feature")
        axB.legend(fontsize=8); axB.grid(alpha=0.3); axB.set_xlim(2.0, 3.5)
        fig.suptitle(f"Discharge-curve degradation signatures (cell {i})")
        fig.tight_layout(); fig.savefig(HERE / "fig2_ic_curves.png", dpi=300); plt.close(fig)
    print(f"Fig.2 done -> fig2_ic_curves.png (cell {i})")


# ============ Fig.7 符号回归 parity ============
def fig7_sr_parity():
    df = pd.read_csv(DATA / "severson_features.csv")
    v = df["var_dQ"].to_numpy()
    f = -v + 1.9911 - 2.7164 / v                       # 发现的公式结构
    log_pred = 0.964 * f + 0.211                       # 线性校准
    pred = np.exp(log_pred); true = df["cycle_life"].to_numpy()
    r2 = r2_score(np.log(true), log_pred)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(true, pred, s=22, alpha=0.6, edgecolor="k", lw=0.3)
    lim = [min(true.min(), pred.min()) * 0.9, max(true.max(), pred.max()) * 1.1]
    ax.plot(lim, lim, "r--", lw=1.5, label="y = x")
    ax.set_xlabel("True cycle life"); ax.set_ylabel("Predicted (symbolic formula)")
    ax.set_title(f"Symbolic-regression parity  (log-R²={r2:.3f})")
    ax.text(0.05, 0.92, r"$\log L = 0.964(-v_{dQ}+1.99-2.72/v_{dQ})+0.211$",
            transform=ax.transAxes, fontsize=9, bbox=dict(fc="white", ec="0.6"))
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(HERE / "fig7_sr_parity.png", dpi=300); plt.close(fig)
    print(f"Fig.7 done -> fig7_sr_parity.png (log-R²={r2:.3f})")


if __name__ == "__main__":
    fig1_framework()
    fig2_ic_curves()
    fig7_sr_parity()
    print("\n全部 3 张图生成完成 [OK]")
