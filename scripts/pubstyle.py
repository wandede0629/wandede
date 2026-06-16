"""导入即生效: 把全局 matplotlib 切到顶刊房屋风格(Arial / 去上右轴线 / 无框图例 /
极简网格 / 外向刻度 / 加粗轴线)。在出图脚本前 import 即可, 不改各脚本绘图逻辑。

依据 scientific-figure-making 技能的 design-theory: 极简、高对比、出版导向。
仅做结构性升级(字体/spine/legend/grid/tick/linewidth), 不改各图既有配色。
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# 语义调色板(供需要时显式取用; 不强制覆盖既有 hex)
PALETTE = {
    "blue_main": "#0F4D92", "blue_secondary": "#3775BA",
    "green": "#2E8B57", "red_strong": "#B64342",
    "neutral": "#767676", "highlight": "#E1A100",
}

plt.rcParams.update({
    # 字体: Arial 优先
    "font.family": ["Arial", "DejaVu Sans", "sans-serif"],
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 9.5,
    "figure.titlesize": 13,
    # 轴线: 去上右, 加粗
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 1.3,
    "axes.edgecolor": "#222222",
    "axes.labelcolor": "#222222",
    "axes.titlecolor": "#222222",
    "axes.titleweight": "medium",
    # 图例: 无框
    "legend.frameon": False,
    "legend.handlelength": 1.6,
    "legend.columnspacing": 1.2,
    # 网格: 极简(留极淡水平参考, 不喧宾夺主)
    "axes.grid": False,
    "grid.color": "#B8B8B8",
    "grid.linewidth": 0.6,
    "grid.alpha": 0.35,
    # 刻度: 外向, 与轴线协调
    "xtick.direction": "out",
    "ytick.direction": "out",
    "xtick.major.width": 1.1,
    "ytick.major.width": 1.1,
    "xtick.color": "#222222",
    "ytick.color": "#222222",
    "xtick.major.size": 4.0,
    "ytick.major.size": 4.0,
    # 线条与导出
    "lines.linewidth": 1.8,
    "lines.markersize": 5.0,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "svg.fonttype": "none",        # 矢量保留可编辑文字
    "pdf.fonttype": 42,            # TrueType 嵌入, 期刊友好
    "ps.fonttype": 42,
})
