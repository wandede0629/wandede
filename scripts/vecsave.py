"""导入即生效: 任何 savefig 到 .png 时, 自动在同目录同名保存 .pdf 矢量版(投稿用)。"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.figure as _mf
from pathlib import Path

_orig = _mf.Figure.savefig


def _dual(self, fname, *a, **k):
    _orig(self, fname, *a, **k)
    try:
        p = Path(str(fname))
        if p.suffix.lower() == ".png":
            k2 = {kk: v for kk, v in k.items() if kk != "dpi"}
            _orig(self, p.with_suffix(".pdf"), *a, **k2)
    except Exception as e:
        print(f"(vecsave: pdf 导出失败 {fname}: {e})")


_mf.Figure.savefig = _dual
