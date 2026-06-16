r"""
补充实验 S6: 分批次容量退化轨迹直接对比
回应"跨批次崩溃是否源于标签/协议系统性差异, 需直观展示轨迹"的质疑。
左: 各批次电芯的放电容量轨迹(细线)+批次中位轨迹(粗线);
右: 各批次寿命(cycle life)分布箱线图。
输出: supp_batch_traj.png
"""
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
HERE = Path(__file__).parent
plt.rcParams.update({"font.size": 9, "savefig.dpi": 300})

soh = pd.read_csv(HERE / "data" / "severson_soh.csv")
soh["batch"] = soh.cell.str[:10]
feats = pd.read_csv(HERE / "data" / "severson_features_full.csv").dropna(subset=["cycle_life", "batch"])
ORDER = ["2017-05-12", "2017-06-30", "2018-04-12", "2019-01-24"]
LBL = {b: f"batch {i+1}" for i, b in enumerate(ORDER)}
COL = {"2017-05-12": "#1f77b4", "2017-06-30": "#ff7f0e", "2018-04-12": "#2ca02c", "2019-01-24": "#d62728"}

fig, (aL, aR) = plt.subplots(1, 2, figsize=(12, 4.6), gridspec_kw={"width_ratios": [1.5, 1]})
# 左: 容量轨迹(逐芯细线 + 批次中位粗线; SOH csv 含 batch1-3)
for b in ORDER[:3]:
    d = soh[soh.batch == b]
    for c_, dc in list(d.groupby("cell"))[::3]:          # 抽 1/3 电芯画细线
        aL.plot(dc.cycle, dc["cap"], color=COL[b], alpha=0.12, lw=0.6)
    med = d.groupby(d.cycle // 50 * 50)["cap"].median()
    aL.plot(med.index, med.values, color=COL[b], lw=2.4, label=LBL[b])
aL.axhline(0.88, color="k", ls="--", lw=1, label="EOL (0.88 Ah)")
aL.set_xlabel("Cycle"); aL.set_ylabel("Discharge capacity (Ah)")
aL.set_title("(a) Capacity-fade trajectories by batch (batches 1-3)")
aL.legend(fontsize=8); aL.grid(alpha=0.3)
aL.text(-0.1, 1.07, "(a)", transform=aL.transAxes, fontsize=11, fontweight="bold")

# 右: 各批次寿命分布(含 batch4)
data = [np.log10(feats[feats.batch == b].cycle_life) for b in ORDER]
bp = aR.boxplot(data, tick_labels=[LBL[b] for b in ORDER], patch_artist=True)
for patch, b in zip(bp["boxes"], ORDER):
    patch.set_facecolor(COL[b]); patch.set_alpha(0.5)
aR.set_ylabel("log10 cycle life")
aR.set_title("(b) Cycle-life distribution by batch")
aR.grid(alpha=0.3, axis="y")
aR.text(-0.16, 1.07, "(b)", transform=aR.transAxes, fontsize=11, fontweight="bold")

fig.tight_layout(); fig.savefig(HERE / "supp_batch_traj.png", bbox_inches="tight")
for b in ORDER:
    d = feats[feats.batch == b].cycle_life
    print(f"{LBL[b]}: n={len(d)}  寿命中位={d.median():.0f}  IQR=[{d.quantile(.25):.0f},{d.quantile(.75):.0f}]")
print("图已保存: supp_batch_traj.png")
