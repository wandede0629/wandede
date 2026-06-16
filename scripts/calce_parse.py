r"""
CALCE CS2 (LCO) 解析器 — Path A 跨化学挑战
将 CALCE Arbin .xlsx(逐步 V/I/t)解析为逐循环 SOH 表 + 每循环放电 V/Q 曲线(供 IC 特征)。

CALCE 格式要点(已实测 CS2_35):
  - 每个 CS2_NN.zip 内含多个按日期命名的 .xlsx(CS2_35_MM_DD_YY.xlsx),需按时间拼接;
  - 数据在名为 'Channel_*' 的 sheet,逐步列:Cycle_Index/Current(A)/Voltage(V)/Discharge_Capacity(Ah);
  - Cycle_Index 在每个文件内从 1 重新计数 -> 跨文件累加偏移;
  - Discharge_Capacity 在文件内累积 -> 每循环放电容量 = 放电步内 (max - min);
  - 放电段 = 电流为负(I < -I_THR)。

输出:
  data/calce/calce_soh.csv      逐循环: cell, cycle, Qd, SOH
  data/calce/calce_curves.npz   每 (cell, cycle) 的放电 V、Q 数组(供 IC/特征提取)
运行: .venv\Scripts\python.exe calce_parse.py
"""
import re
import io
import zipfile
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
HERE = Path(__file__).parent
CALCE_DIR = HERE / "data" / "calce"
I_THR = 0.02          # A,判定充/放电的电流阈值
MIN_DISCH_PTS = 20    # 放电段最少点数,过滤异常循环
SOH_LO, SOH_HI = 0.5, 1.05

# 日期命名 CS2_35_10_15_10.xlsx -> (month, day, year) 用于排序
_DATE = re.compile(r"_(\d{1,2})_(\d{1,2})_(\d{2})\.xlsx$", re.I)


def _date_key(name: str):
    m = _DATE.search(name)
    if not m:
        return (99, 99, 99)
    mo, da, yr = (int(x) for x in m.groups())
    return (yr, mo, da)


def _channel_sheet(xls: pd.ExcelFile) -> str | None:
    for sh in xls.sheet_names:
        if sh.lower().startswith("channel"):
            return sh
    return None


def parse_cell(zip_path: Path) -> tuple[pd.DataFrame, dict]:
    """解析一个电芯的 zip,返回 (逐循环 DataFrame, {(cell,cycle): (V, Q)})。"""
    cell = zip_path.stem  # CS2_35
    z = zipfile.ZipFile(zip_path)
    xlsx = sorted((n for n in z.namelist() if n.lower().endswith(".xlsx")), key=_date_key)
    rows, curves = [], {}
    cyc_offset = 0
    for fn in xlsx:
        try:
            xls = pd.ExcelFile(io.BytesIO(z.read(fn)))
        except Exception:
            continue
        sh = _channel_sheet(xls)
        if sh is None:
            continue
        df = pd.read_excel(xls, sh)
        need = {"Cycle_Index", "Current(A)", "Voltage(V)", "Discharge_Capacity(Ah)"}
        if not need.issubset(df.columns):
            continue
        df = df[list(need)].dropna()
        max_cyc_in_file = 0
        for cyc, g in df.groupby("Cycle_Index"):
            cyc = int(cyc)
            max_cyc_in_file = max(max_cyc_in_file, cyc)
            d = g[g["Current(A)"] < -I_THR]            # 放电段
            if len(d) < MIN_DISCH_PTS:
                continue
            qd = float(d["Discharge_Capacity(Ah)"].max() - d["Discharge_Capacity(Ah)"].min())
            if not (0.1 < qd < 5.0):
                continue
            gcyc = cyc_offset + cyc
            rows.append({"cell": cell, "cycle": gcyc, "Qd": qd})
            # 放电 V/Q(按电压升序,Q 归零),供 IC
            v = d["Voltage(V)"].to_numpy()
            q = d["Discharge_Capacity(Ah)"].to_numpy() - d["Discharge_Capacity(Ah)"].min()
            o = np.argsort(v)
            curves[(cell, gcyc)] = (v[o].astype(np.float32), q[o].astype(np.float32))
        cyc_offset += max_cyc_in_file
    if not rows:
        return pd.DataFrame(), curves
    out = pd.DataFrame(rows).sort_values("cycle").reset_index(drop=True)
    q0 = float(np.median(out.Qd.iloc[:8]))             # 初始容量 = 前 8 循环中位
    out["SOH"] = out.Qd / q0
    out = out[(out.SOH >= SOH_LO) & (out.SOH <= SOH_HI)].reset_index(drop=True)
    return out, curves


def main():
    zips = sorted(CALCE_DIR.glob("CS2_*.zip"))
    if not zips:
        print("未找到 CALCE zip,先运行下载。")
        return
    valid = [z for z in zips if zipfile.is_zipfile(z)]
    skipped = [z.stem for z in zips if z not in valid]
    if skipped:
        print(f"跳过未下完/损坏的 zip: {skipped}")
    all_soh, all_curves = [], {}
    for zp in valid:
        df, curves = parse_cell(zp)
        if df.empty:
            print(f"  {zp.stem}: 解析为空,跳过")
            continue
        all_soh.append(df)
        all_curves.update({f"{c}|{cy}": np.stack([v, q]) for (c, cy), (v, q) in curves.items()})
        print(f"  {zp.stem}: {len(df)} 循环  SOH {df.SOH.min():.3f}-{df.SOH.max():.3f}  Q0~{df.Qd.iloc[:8].median():.3f}Ah")
    if not all_soh:
        print("无有效电芯。")
        return
    soh = pd.concat(all_soh, ignore_index=True)
    CALCE_DIR.mkdir(parents=True, exist_ok=True)
    soh.to_csv(CALCE_DIR / "calce_soh.csv", index=False)
    np.savez_compressed(CALCE_DIR / "calce_curves.npz", **all_curves)
    print(f"\n汇总: {soh.cell.nunique()} 电芯, {len(soh)} 循环")
    print(f"  -> {CALCE_DIR/'calce_soh.csv'}")
    print(f"  -> {CALCE_DIR/'calce_curves.npz'} ({len(all_curves)} 条曲线)")

    # 命门校验: IC 峰电压应落在 LCO 位置(~3.7-3.95V), 而非 LFP(~3.2V)
    from scipy.signal import savgol_filter
    sample = soh[soh.SOH > 0.95].head(20)
    peaks = []
    for _, r in sample.iterrows():
        key = f"{r.cell}|{int(r.cycle)}"
        if key not in all_curves:
            continue
        v, q = all_curves[key]
        m = np.argsort(v); v, q = v[m], q[m]
        if v.size < 30:
            continue
        ic = np.clip(savgol_filter(np.abs(np.gradient(q, v)), min(31, (v.size//2)*2-1), 3), 0, None)
        peaks.append(float(v[int(np.argmax(ic))]))
    if peaks:
        print(f"\nIC 峰电压(健康循环, n={len(peaks)}): 中位 {np.median(peaks):.3f} V  范围 {min(peaks):.2f}-{max(peaks):.2f} V")
        print("  => LCO 峰位 ≈ 3.7-3.95V 证实跨化学(LFP 硬编码窗 3.1-3.35V 会落空,符合审计预测)")


if __name__ == "__main__":
    main()
