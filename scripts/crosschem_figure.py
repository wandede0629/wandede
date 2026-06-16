r"""
Path A 跨化学结果图(3 面板, 论文级):
  (a) 零样本崩溃 -> 重拟合恢复(R² 柱 + 误差线)
  (b) few-shot 适配预算(R² vs 训练电芯数)
  (c) 一个留出 LCO 芯的预测 parity: 零样本(散) vs 重拟合(贴对角)
输出: crosschem_result.png + .pdf
"""
import pubstyle, vecsave  # noqa: F401  样式 + 矢量双导出
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor

from crosschem_experiment import build_lfp, build_lco, FEATCOLS, CALCE

HERE = Path(__file__).parent
res = json.loads((CALCE / "crosschem_results.json").read_text(encoding="utf-8"))

lfp = build_lfp()
lco = build_lco("native")

# panel (c) 数据: 留出 1 个 LCO 芯, 比较零样本 vs 重拟合
cells = sorted(lco.cell.unique())
test_cell = cells[-1]
te = lco[lco.cell == test_cell]
tr = lco[lco.cell != test_cell]
# 零样本: LFP 模型(lfp_fixed 口径)预测 LCO(lfp_fixed)
lco_naive = build_lco("lfp_fixed"); te_naive = lco_naive[lco_naive.cell == test_cell]
scL = StandardScaler().fit(lfp[FEATCOLS])
mL = RandomForestRegressor(300, min_samples_leaf=2, n_jobs=-1, random_state=0).fit(scL.transform(lfp[FEATCOLS]), lfp.SOH)
p_zero = mL.predict(scL.transform(te_naive[FEATCOLS]))
# 重拟合: LCO native 训练其余芯
scN = StandardScaler().fit(tr[FEATCOLS])
mN = RandomForestRegressor(300, min_samples_leaf=2, n_jobs=-1, random_state=0).fit(scN.transform(tr[FEATCOLS]), tr.SOH)
p_ref = mN.predict(scN.transform(te[FEATCOLS]))

fig, (axA, axB, axC) = plt.subplots(1, 3, figsize=(13.4, 4.2))

# (a) 柱状: 零样本 vs 重拟合
labels = ["Zero-shot\n(LFP→LCO)", "Re-fit\n(LFP window)", "Re-fit\n(native)"]
vals = [res["A1_r2"], res["A2c_r2"][0], res["A2_r2"][0]]
errs = [0, res["A2c_r2"][1], res["A2_r2"][1]]
colors = ["#d62728", "#7f7f7f", "#1f77b4"]
bars = axA.bar(labels, vals, yerr=errs, capsize=4, color=colors, alpha=0.9)
axA.axhline(0, color="k", lw=0.8)
axA.set_ylabel("SOH R² on held-out LCO cells")
axA.set_ylim(-0.05, 1.05)
axA.set_title("(a) Frozen model collapses; re-fit recovers")
for b, v in zip(bars, vals):
    axA.text(b.get_x() + b.get_width()/2, v + 0.03 if v > 0 else 0.03,
             f"{v:.2f}", ha="center", fontsize=9, fontweight="bold")
axA.annotate("", xy=(2, vals[2]+0.0), xytext=(0, vals[0]+0.0),
             arrowprops=dict(arrowstyle="->", color="0.4", lw=1.2, connectionstyle="arc3,rad=-0.3"))

# (b) few-shot 适配预算
ns = sorted(int(k) for k in res["fewshot"])
r2m = [res["fewshot"][str(n)]["r2"][0] for n in ns]
r2s = [res["fewshot"][str(n)]["r2"][1] for n in ns]
axB.errorbar(ns, r2m, yerr=r2s, fmt="o-", color="#1f77b4", capsize=4, label="LCO few-shot re-fit")
axB.axhline(res["A2_r2"][0], color="#1f77b4", ls="--", lw=1, label=f"Full re-fit ({res['A2_r2'][0]:.2f})")
axB.axhline(res["A1_r2"], color="#d62728", ls=":", lw=1.2, label=f"Zero-shot ({res['A1_r2']:.2f})")
axB.set_xlabel("# labelled LCO cells for adaptation")
axB.set_ylabel("SOH R² on held-out LCO cells")
axB.set_xticks(ns)
axB.set_ylim(-0.05, 1.05)
axB.set_title("(b) Adaptation budget")
axB.legend(fontsize=7.5, loc="lower right")

# (c) parity: 零样本 vs 重拟合(同一留出芯)
axC.scatter(te_naive.SOH, p_zero, s=10, alpha=0.35, color="#d62728", label="Zero-shot (LFP→LCO)")
axC.scatter(te.SOH, p_ref, s=10, alpha=0.45, color="#1f77b4", label="Re-fit (native)")
lim = [0.45, 1.05]
axC.plot(lim, lim, "k--", lw=1)
axC.set_xlim(lim); axC.set_ylim(0.2, 1.1)
axC.set_xlabel("True SOH"); axC.set_ylabel("Predicted SOH")
axC.set_title(f"(c) Held-out LCO cell ({test_cell})")
axC.legend(fontsize=7.5, loc="upper left")

fig.tight_layout()
fig.savefig(HERE / "crosschem_result.png", dpi=300, bbox_inches="tight")
print("图已保存: crosschem_result.png (+ .pdf)")
print(f"  panel(c) 留出芯={test_cell}: 零样本点散布, 重拟合贴对角")
