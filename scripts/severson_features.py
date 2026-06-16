r"""
从 Severson .mat 提取【论文级早期循环特征】-> data\severson_features.csv

核心思想(Severson et al., Nature Energy 2019):
  仅用前 ~100 个循环的放电行为，就能预测电池最终寿命。
  最强特征 = ΔQ(V) = (第100循环 - 第10循环) 的放电容量-电压曲线之差，取其 log(方差)。

提取的特征:
  充电协议:   c_rate_1, soc_switch_pct, c_rate_2
  ΔQ_100-10:  var_dQ(log方差), min_dQ, mean_dQ
  容量衰减:    qd_slope, qd_intercept (循环2-100 线性拟合), qd2(循环2容量),
              qd_max_minus_qd2
  其它早期信号: chargetime_5(前5循环平均充电时间), ir_min, ir_diff(IR_100-IR_2),
              temp_integral(循环2-100 平均温度之和)
  标签:        cycle_life

用法: .venv\Scripts\python.exe severson_features.py
然后:  .venv\Scripts\python.exe battery_autora_loop.py data\severson_features.csv
"""
import numpy as np
import pandas as pd
import h5py
from pathlib import Path
from severson_parse import _mat_str, _scalar, parse_policy

DATA_DIR = Path(__file__).parent / "data"


def _arr(f, ref_or_ds):
    """读成 1D float 数组。summary 字段是直接 Dataset；Qdlin 单元是对象引用(需 f[ref])。"""
    x = ref_or_ds[()] if isinstance(ref_or_ds, h5py.Dataset) else f[ref_or_ds][()]
    return np.array(x).flatten().astype(float)


def _val_at(cyc, arr, target):
    """取最接近指定循环号的值。"""
    return float(arr[np.argmin(np.abs(cyc - target))])


def cell_features(f, b, i):
    # ---- 充电协议(仅作参考；2段/4段格式不同，不进模型特征) + 寿命 ----
    policy = _mat_str(f, b["policy_readable"][i, 0])
    feats = parse_policy(policy)                       # 2段 "C1(Q%)-C2"，batch4 为 None
    c1 = feats[0] if feats else np.nan
    soc = feats[1] if feats else np.nan
    c2 = feats[2] if feats else np.nan
    life = _scalar(f, b["cycle_life"][i, 0])
    if not np.isfinite(life) or life <= 0:
        return None

    cyc_ds = f[b["cycles"][i, 0]]
    n_cyc = cyc_ds["Qdlin"].shape[0]
    if n_cyc <= 100:
        return None

    # ---- ΔQ(V) = Qdlin(100) - Qdlin(10) ----
    q10 = _arr(f, cyc_ds["Qdlin"][9, 0])
    q100 = _arr(f, cyc_ds["Qdlin"][99, 0])
    dq = q100 - q10
    dq = dq[np.isfinite(dq)]
    if dq.size == 0:
        return None
    eps = 1e-12
    var_dq = np.log10(np.var(dq) + eps)
    min_dq = np.log10(np.abs(np.min(dq)) + eps)
    mean_dq = np.log10(np.abs(np.mean(dq)) + eps)

    # ---- summary 标量序列(逐循环) ----
    s = f[b["summary"][i, 0]]
    cyc = _arr(f, s["cycle"])
    qd = _arr(f, s["QDischarge"])
    ir = _arr(f, s["IR"])
    ct = _arr(f, s["chargetime"])
    tavg = _arr(f, s["Tavg"])

    m = (cyc >= 2) & (cyc <= 100)
    if m.sum() < 5:
        return None
    qd_slope, qd_intercept = np.polyfit(cyc[m], qd[m], 1)
    qd2 = _val_at(cyc, qd, 2)
    qd_max_minus_qd2 = float(qd[m].max()) - qd2

    m5 = (cyc >= 1) & (cyc <= 5)
    ct5 = ct[m5]
    chargetime_5 = float(ct5[ct5 > 0].mean()) if np.any(ct5 > 0) else float("nan")

    ir_pos = ir[m][ir[m] > 0]
    ir_min = float(ir_pos.min()) if ir_pos.size else float("nan")
    ir_diff = _val_at(cyc, ir, 100) - _val_at(cyc, ir, 2)
    temp_integral = float(tavg[m].sum())

    return {
        "policy": policy,
        "c_rate_1": c1, "soc_switch_pct": soc, "c_rate_2": c2,
        "var_dQ": var_dq, "min_dQ": min_dq, "mean_dQ": mean_dq,
        "qd_slope": float(qd_slope), "qd_intercept": float(qd_intercept),
        "qd2": qd2, "qd_max_minus_qd2": qd_max_minus_qd2,
        "chargetime_5": chargetime_5, "ir_min": ir_min, "ir_diff": ir_diff,
        "temp_integral": temp_integral,
        "cycle_life": round(life, 1),
    }


def main():
    mats = sorted(DATA_DIR.glob("*.mat"))
    if not mats:
        raise SystemExit("data\\ 下没有 .mat，请先 severson_download.py")
    rows = []
    for mat in mats:
        print(f"提取特征 {mat.name} ...")
        with h5py.File(mat, "r") as f:
            b = f["batch"]
            n = b["cycle_life"].shape[0]
            ok = 0
            for i in range(n):
                try:
                    r = cell_features(f, b, i)
                    if r:
                        r["batch"] = mat.name[:10]      # 批次标签(用于批次感知评估)
                        rows.append(r); ok += 1
                except Exception as e:
                    print(f"  (跳过 cell {i}: {e})")
            print(f"  {ok} 个电芯提取成功")

    df_all = pd.DataFrame(rows).reset_index(drop=True)
    df_all.to_csv(DATA_DIR / "severson_features_full.csv", index=False)

    # 模型特征 = 格式无关的早期循环特征，4 批次可统一合并
    # 注:batch4(2019 CLO 批次)未记录内阻 IR(全为0)，故 ir_min/ir_diff 不入统一特征集，
    #     以保证 4 个批次都能保留(它们在 _full.csv 中仍有记录，供仅用 batch1-3 时使用)。
    feat_cols = ["var_dQ", "min_dQ", "mean_dQ",
                 "qd_slope", "qd_intercept", "qd2", "qd_max_minus_qd2",
                 "chargetime_5", "temp_integral"]
    df = df_all[feat_cols + ["cycle_life"]].dropna().reset_index(drop=True)
    out = DATA_DIR / "severson_features.csv"
    df.to_csv(out, index=False)

    print(f"\n共 {len(df)} 个电芯, {len(feat_cols)} 个【格式无关】特征 -> {out}")
    # 看一下最强特征与寿命的相关性
    corr = np.corrcoef(df["var_dQ"], np.log10(df["cycle_life"]))[0, 1]
    print(f"log(var ΔQ) 与 log(寿命) 相关系数: {corr:.3f}  (论文核心特征)")
    print("\n下一步: .venv\\Scripts\\python.exe battery_autora_loop.py data\\severson_features.csv")


if __name__ == "__main__":
    main()
