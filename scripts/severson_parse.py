r"""
解析 data\ 下的 Severson .mat 文件 -> data\severson_cycle_life.csv

从每个电芯提取：
  充电协议 policy_readable  形如 "3.6C(80%)-3.6C"
      -> c_rate_1 (第一段倍率), soc_switch_pct (切换SOC%), c_rate_2 (第二段倍率)
  cycle_life (到 80% 容量的循环寿命)  -> 标签

输出列: [c_rate_1, soc_switch_pct, c_rate_2, cycle_life]
正好喂给 battery_autora_loop.py 做寿命预测。

用法:
  解析真实数据:  .venv\Scripts\python.exe severson_parse.py
  自测(无需数据): .venv\Scripts\python.exe severson_parse.py --selftest
"""
import re
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import h5py

DATA_DIR = Path(__file__).parent / "data"
POLICY_RE = re.compile(r"^\s*([\d.]+)C\((\d+)%\)-([\d.]+)C")


def _mat_str(f, ref):
    """把 MATLAB char 引用解成字符串。只保留可打印 ASCII，
    这样空字段(MATLAB 用大整数占位，如 barcode)会安全地解成空串而非溢出。"""
    arr = np.array(f[ref][()]).flatten()
    return "".join(chr(int(c)) for c in arr if 32 <= int(c) < 127)


def _scalar(f, ref):
    return float(np.array(f[ref][()]).flatten()[0])


def parse_policy(policy):
    """'3.6C(80%)-3.6C' -> (3.6, 80.0, 3.6); 不匹配返回 None。"""
    m = POLICY_RE.match(policy)
    if not m:
        return None
    return float(m.group(1)), float(m.group(2)), float(m.group(3))


def parse_file(path):
    """解析单个 .mat -> list[dict]。"""
    rows = []
    with h5py.File(path, "r") as f:
        batch = f["batch"]
        n = batch["cycle_life"].shape[0]
        has_barcode = "barcode" in batch
        for i in range(n):
            try:
                policy = _mat_str(f, batch["policy_readable"][i, 0])
                feats = parse_policy(policy)
                if feats is None:
                    continue
                life = _scalar(f, batch["cycle_life"][i, 0])
                if not np.isfinite(life) or life <= 0:
                    continue
                barcode = _mat_str(f, batch["barcode"][i, 0]) if has_barcode else ""
                if not barcode:                       # batch1 的 barcode 为空 -> 用唯一回退 id
                    barcode = f"{path.stem}_{i}"
                rows.append({
                    "barcode": barcode,
                    "policy": policy,
                    "c_rate_1": feats[0],
                    "soc_switch_pct": feats[1],
                    "c_rate_2": feats[2],
                    "cycle_life": round(life, 1),
                })
            except Exception as e:
                print(f"  (跳过 cell {i}: {e})")
    return rows


def main():
    mats = sorted(DATA_DIR.glob("*.mat"))
    if not mats:
        raise SystemExit("data\\ 下没有 .mat 文件。请先运行 severson_download.py")
    all_rows = []
    for m in mats:
        print(f"解析 {m.name} ...")
        rows = parse_file(m)
        print(f"  提取 {len(rows)} 个有效电芯")
        all_rows.extend(rows)

    if not all_rows:
        raise SystemExit("没有解析出任何有效电芯，请把上面的报错发给我。")
    df = pd.DataFrame(all_rows)
    # barcode 在本数据中多为空(MATLAB 占位)，按 barcode 去重仅在其有效时生效；
    # 回退 id 唯一，因此等效于保留全部有效电芯。
    df = df.sort_values("cycle_life").drop_duplicates("barcode", keep="last")
    df = df.reset_index(drop=True)

    out_full = DATA_DIR / "severson_full.csv"
    out_model = DATA_DIR / "severson_cycle_life.csv"
    df.to_csv(out_full, index=False)
    df[["c_rate_1", "soc_switch_pct", "c_rate_2", "cycle_life"]].to_csv(out_model, index=False)

    print(f"\n共 {len(df)} 个电芯")
    print(f"寿命范围: {df.cycle_life.min():.0f} ~ {df.cycle_life.max():.0f} 圈")
    print(f"特征+标签 -> {out_model}")
    print(f"含 barcode/policy 完整表 -> {out_full}")
    print("\n下一步: .venv\\Scripts\\python.exe battery_autora_loop.py data\\severson_cycle_life.csv")


# ---------------- 自测:构造 mock HDF5 验证解析逻辑(无需真实数据) ----------------
def selftest():
    import tempfile
    policies = ["3.6C(80%)-3.6C", "5.4C(60%)-3C", "4C(13%)-6C", "bad-policy"]
    lives = [1000.0, 500.0, 800.0, 700.0]
    barcodes = ["ELA0001", "ELA0002", "ELA0003", "ELA0004"]

    tmp = Path(tempfile.gettempdir()) / "severson_mock.mat"
    with h5py.File(tmp, "w") as f:
        b = f.create_group("batch")
        store = f.create_group("store")
        cl_refs = np.empty((len(policies), 1), dtype=h5py.ref_dtype)
        pol_refs = np.empty((len(policies), 1), dtype=h5py.ref_dtype)
        bc_refs = np.empty((len(policies), 1), dtype=h5py.ref_dtype)
        for i, (pol, life, bc) in enumerate(zip(policies, lives, barcodes)):
            d_cl = store.create_dataset(f"cl{i}", data=np.array([[life]]))
            cl_refs[i, 0] = d_cl.ref
            d_pol = store.create_dataset(
                f"pol{i}", data=np.array([ord(c) for c in pol], dtype=np.uint16).reshape(-1, 1))
            pol_refs[i, 0] = d_pol.ref
            d_bc = store.create_dataset(
                f"bc{i}", data=np.array([ord(c) for c in bc], dtype=np.uint16).reshape(-1, 1))
            bc_refs[i, 0] = d_bc.ref
        b.create_dataset("cycle_life", data=cl_refs)
        b.create_dataset("policy_readable", data=pol_refs)
        b.create_dataset("barcode", data=bc_refs)

    rows = parse_file(tmp)
    assert len(rows) == 3, f"应解出 3 个有效电芯(1 个 bad-policy 被跳过), 实际 {len(rows)}"
    r0 = next(r for r in rows if r["barcode"] == "ELA0001")
    assert (r0["c_rate_1"], r0["soc_switch_pct"], r0["c_rate_2"], r0["cycle_life"]) == (3.6, 80.0, 3.6, 1000.0)
    r1 = next(r for r in rows if r["barcode"] == "ELA0002")
    assert (r1["c_rate_1"], r1["soc_switch_pct"], r1["c_rate_2"]) == (5.4, 60.0, 3.0)
    assert parse_policy("bad-policy") is None
    print("自测通过 ✅  解析逻辑(HDF5引用解引用 + char解码 + 协议正则)全部正确")
    print("  样例:", {k: r0[k] for k in ("policy", "c_rate_1", "soc_switch_pct", "c_rate_2", "cycle_life")})


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        selftest()
    else:
        main()
