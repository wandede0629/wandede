r"""
补充实验 S8: 动态自适应共形预测(ACI)在跨批次漂移下维持覆盖
回应"static split-conformal 在分布漂移时覆盖失效"。

场景: 电芯按批次顺序流式到达(b1 训练/校准 -> 流式 b2, b3, b4)。
点模型(RF, 寿命预测)仅在 b1 上训练并固定; 随批次推进分布漂移。
对比两种 90% 区间:
  - Static split-conformal: 校准分位数 q 在 b1 上一次性确定, 固定不变。
  - ACI (Gibbs & Candes 2021): 在线根据覆盖反馈更新 α_t,
       α_{t+1} = α_t + γ(α* - err_t), q_t = 滚动残差的 (1-α_t) 分位数。

报告: 各批次的经验覆盖率(static vs ACI)、平均区间宽度, 以及滚动覆盖曲线。
输出: 打印 + 图 supp_aci.png
"""
import warnings
import numpy as np
import pandas as pd
from collections import deque
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor

warnings.filterwarnings("ignore")
HERE = Path(__file__).parent
df = pd.read_csv(HERE / "data" / "severson_features_full.csv")
FEAT = ["var_dQ", "min_dQ", "mean_dQ", "qd_slope", "qd_intercept",
        "qd2", "qd_max_minus_qd2", "chargetime_5", "temp_integral"]
df = df.dropna(subset=FEAT + ["cycle_life", "batch"]).reset_index(drop=True)
BMAP = {"2017-05-12": "b1", "2017-06-30": "b2", "2018-04-12": "b3", "2019-01-24": "b4"}
df["b"] = df["batch"].map(BMAP)

ALPHA = 0.10            # 目标 90% 覆盖
GAMMA = 0.05           # ACI 学习率
WIN = 40               # 滚动残差窗口


def run(seed):
    rng = np.random.default_rng(seed)
    b1 = df[df.b == "b1"].sample(frac=1, random_state=seed)
    n_cal = 20
    tr, cal = b1.iloc[n_cal:], b1.iloc[:n_cal]
    model = RandomForestRegressor(300, min_samples_leaf=2, n_jobs=-1, random_state=0)
    model.fit(tr[FEAT], np.log(tr.cycle_life))
    res0 = list(np.abs(np.log(cal.cycle_life.values) - model.predict(cal[FEAT])))
    q_static = np.quantile(res0, 1 - ALPHA)

    stream = pd.concat([df[df.b == bb].sample(frac=1, random_state=seed) for bb in ["b2", "b3", "b4"]])
    resid = deque(res0, maxlen=WIN)          # ACI 的滚动残差
    resid_r = deque(res0, maxlen=WIN)        # rolling 基线的滚动残差(同等信息量, 固定 alpha)
    alpha_t = ALPHA
    out = []
    for _, row in stream.iterrows():
        pred = float(model.predict(row[FEAT].to_frame().T)[0])
        ytrue = np.log(row.cycle_life)
        # static: 永不更新
        cov_s = abs(ytrue - pred) <= q_static
        # rolling recalibration: 同等在线信息, 固定 alpha=0.10 (公平基线)
        q_r = np.quantile(resid_r, 1 - ALPHA)
        cov_r = abs(ytrue - pred) <= q_r
        resid_r.append(abs(ytrue - pred))
        # ACI: 在线信息 + 自适应 alpha
        q_t = np.quantile(resid, np.clip(1 - alpha_t, 0.01, 0.999))
        cov_a = abs(ytrue - pred) <= q_t
        err_t = 0.0 if cov_a else 1.0
        alpha_t = float(np.clip(alpha_t + GAMMA * (ALPHA - err_t), 0.001, 0.5))
        resid.append(abs(ytrue - pred))
        out.append({"b": row.b, "cov_static": cov_s, "w_static": 2 * q_static,
                    "cov_roll": cov_r, "w_roll": 2 * q_r,
                    "cov_aci": cov_a, "w_aci": 2 * q_t})
    return pd.DataFrame(out)


res = pd.concat([run(s) for s in range(8)], ignore_index=True)

print("=" * 64)
print("动态共形(ACI) vs 静态共形 · 跨批次流式 (目标覆盖 90%, 8 seeds)")
print("=" * 64)
print(f"\n{'批次':<6}{'static覆盖':>11}{'rolling覆盖':>12}{'ACI覆盖':>10}{'static宽':>10}{'rolling宽':>10}{'ACI宽':>9}")
for bb in ["b2", "b3", "b4"]:
    d = res[res.b == bb]
    print(f"{bb:<6}{d.cov_static.mean()*100:>10.1f}%{d.cov_roll.mean()*100:>11.1f}%{d.cov_aci.mean()*100:>9.1f}%"
          f"{d.w_static.mean():>10.3f}{d.w_roll.mean():>10.3f}{d.w_aci.mean():>9.3f}")
print("-" * 70)
print(f"{'总体':<6}{res.cov_static.mean()*100:>10.1f}%{res.cov_roll.mean()*100:>11.1f}%{res.cov_aci.mean()*100:>9.1f}%"
      f"{res.w_static.mean():>10.3f}{res.w_roll.mean():>10.3f}{res.w_aci.mean():>9.3f}")
print("\n结论: rolling(同等在线信息,固定α)与 ACI 对比, 区分'在线信息'与'自适应α'各自的贡献。")

try:
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    # 单 seed 的滚动覆盖曲线(可视化)
    one = run(0).reset_index(drop=True)
    roll = 25
    rs = one.cov_static.rolling(roll, min_periods=5).mean()
    rr = one.cov_roll.rolling(roll, min_periods=5).mean()
    ra = one.cov_aci.rolling(roll, min_periods=5).mean()
    wa = one.w_aci.rolling(roll, min_periods=5).mean()
    wr = one.w_roll.rolling(roll, min_periods=5).mean()
    bnd = {bb: one.index[one.b == bb] for bb in ["b2", "b3", "b4"]}
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(8, 6.2), sharex=True,
                                  gridspec_kw={"height_ratios": [2.1, 1]})
    ax.plot(rs.values, label="Static split-conformal", color="#ff7f0e")
    ax.plot(rr.values, label="Rolling recalibration (fixed α)", color="#2ca02c", ls="-.")
    ax.plot(ra.values, label="Adaptive conformal (ACI)", color="#1f77b4")
    ax.axhline(0.90, color="k", ls="--", lw=1, label="Nominal 90%")
    for bb in ["b3", "b4"]:
        if len(bnd[bb]):
            for a_ in (ax, ax2):
                a_.axvline(bnd[bb][0], color="0.6", ls=":")
            ax.text(bnd[bb][0], 0.5, f" {bb} starts", rotation=90, fontsize=8, va="bottom")
    ax.set_ylabel(f"Rolling coverage (window {roll})")
    ax.set_title("Adaptive vs static conformal under cross-batch drift")
    ax.legend(fontsize=8); ax.grid(alpha=0.3); ax.set_ylim(0.4, 1.02)
    ax.text(-0.1, 1.06, "(a)", transform=ax.transAxes, fontsize=11, fontweight="bold")
    ax2.plot(wa.values, color="#1f77b4", label="ACI interval width")
    ax2.plot(wr.values, color="#2ca02c", ls="-.", label="Rolling-recalibration width")
    ax2.axhline(one.w_static.iloc[0], color="#ff7f0e", ls="--", lw=1.4, label="Static width (fixed)")
    ax2.set_xlabel("Streamed cell index (batch order)")
    ax2.set_ylabel("Interval width (log-life)")
    ax2.legend(fontsize=8); ax2.grid(alpha=0.3)
    ax2.text(-0.1, 1.10, "(b)", transform=ax2.transAxes, fontsize=11, fontweight="bold")
    fig.tight_layout(); fig.savefig(HERE / "supp_aci.png", dpi=300)
    print("图已保存: supp_aci.png (2-panel)")
except Exception as e:
    print(f"(画图跳过: {e})")
