# -*- coding: utf-8 -*-
"""
Journal-style 3D scatter plot: Train vs. Test
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import font_manager as fm
from pathlib import Path

# ---------- 字体：Times New Roman ----------
# assert fm.findfont("Times New Roman", fallback_to_default=False), \
#        "系统中未找到 Times New Roman，请先安装或改用 LaTeX 方案"

# mpl.rcParams.update({
#     "font.family"      : "Times New Roman",
#     "mathtext.fontset" : "custom",
#     "mathtext.rm"      : "Times New Roman",
#     "mathtext.it"      : "Times New Roman:italic",
#     "mathtext.bf"      : "Times New Roman:bold",
# })

mpl.rcParams.update({
    "font.family"      : "Arial",          # 普通文字
    "mathtext.fontset" : "custom",
    "mathtext.rm"      : "Arial",          # 正体
    "mathtext.it"      : "Arial:italic",   # 斜体
    "mathtext.bf"      : "Arial:bold",     # 粗体
    "text.usetex"      : False,            # 用 mathtext 而非 LaTeX
    "axes.unicode_minus": False,           # 负号不变方块
    # 可选：给非数学文本提供 Unicode 回退（例如带圈数字）
    "font.sans-serif"  : ["Arial", "DejaVu Sans", "Noto Sans", "Segoe UI"],
})

# ---------- 期刊格式参数 ----------
journal = dict(
    figSize_cm    = (12, 8),
    fontSize      = 14,
    axisLineWidth = 1.0,
    markerSize    = 7,
    dpi           = 1200,
)

cm2inch = lambda cm: float(cm) / 2.54
mpl.rcParams.update({
    "figure.figsize" : tuple(cm2inch(x) for x in journal["figSize_cm"]),
    "figure.dpi"     : journal["dpi"],
    "font.size"      : journal["fontSize"],
    "axes.linewidth" : journal["axisLineWidth"],
    "xtick.direction": "in", "ytick.direction": "in",
    "legend.frameon" : False,
    "text.usetex"    : False,
})

# ---------- 函数：两类数据 ----------
def plot_3d_two_sets(train_pts, test_pts, *,
                     xlim=None, ylim=None, zlim=None,
                     elev=20, azim=-60,
                     xlabel=r"$t_{\mathrm{B}}\,(\mathrm{mm})$",
                     ylabel = r"$d_{\mathrm{p}}\,(\mathrm{mm})$",
                     zlabel = r"$V_{\mathrm{p}}\,(\mathrm{km/s})$",
                     title=None,
                     save_path=None):

    """
    期刊格式 3D 散点：训练集 / 测试集

    Parameters
    ----------
    train_pts, test_pts : (N_i, 3) array-like
        每类样本的三维坐标。
    """
    train_pts = np.asarray(train_pts, dtype=float)
    test_pts  = np.asarray(test_pts,  dtype=float)

    for arr in (train_pts, test_pts):
        if arr.ndim != 2 or arr.shape[1] != 3:
            raise ValueError("输入必须是形如 (N, 3) 的二维数组")

        # 颜色 & 形状：Train = 蓝色圆点，Test = 绿色方块
    color_map = {"Train": "red", "Test": "blue"}
    marker_map = {"Train": "o", "Test": "o"}  # ←★ 新增行

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    scatter_handles = []
    for pts, label, legend_text in ((train_pts, "Train", "Non-Perforation"), (test_pts, "Test", "Perforation")):
        h = ax.scatter(
            pts[:, 0], pts[:, 1], pts[:, 2],
            c=color_map[label],
            marker=marker_map[label],  # ←★ 按标签取形状
            s=journal["markerSize"] ** 2,
            edgecolors="none",
            linewidths=0,
            depthshade=True,
            label=legend_text
        )
        scatter_handles.append(h)

    # ===== 坐标范围 =====
    if xlim is not None:
        ax.set_xlim(*xlim)      # xlim 应是 (min, max)
    if ylim is not None:
        ax.set_ylim(*ylim)
    if zlim is not None:
        ax.set_zlim(*zlim)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel(zlabel)
    if title:
        ax.set_title(title, pad=8)
    ax.view_init(elev=elev, azim=azim)

    # ax.legend(handles=scatter_handles, loc="best", ncol=1)
    # 2) 用 fig.legend 放到图外上方，并排两列
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels,
               loc='upper center', bbox_to_anchor=(0.5, 1.02),
               ncol=2, frameon=False, handlelength=1.2, handletextpad=0.6)

    # 3) 适度增大页边（constrained_layout 会考虑到 fig.legend）    可选微调：
    fig.set_constrained_layout_pads(w_pad=0.1, h_pad=0.1, pad=0.6)
    fig.tight_layout(pad=0.4)

    if save_path is not None:
        save_path = Path(save_path).with_suffix(Path(save_path).suffix or ".tif")
        fig.savefig(save_path, dpi=journal["dpi"], bbox_inches="tight", pad_inches=0.5)
        print(f"Figure saved to {save_path.resolve()}")
    else:
        plt.show()

    return fig, ax


# ---------- 示例 ----------
if __name__ == "__main__":
    # 训练集
    train_NP = np.array([
        [1, 3, 1.89], [1, 3, 3.81], [1, 3, 4.53],
        [1, 4, 3.81], [1, 4, 4.53],
        [1, 5, 3.81], [1, 5, 4.53], [1, 5, 5.15],
        [2, 3, 3.81], [2, 3, 4.53], [2, 3, 6.41],
        [2, 4, 1.89], [2, 4, 3.81], [2, 4, 4.53], [2, 4, 6.41],
        [2, 5, 4.53], [2, 5, 2.92], [2, 5, 5.15], [2, 5, 5.70],
        [2, 6, 3.81], [2, 6, 4.53],
        [1, 2.9, 4.02], [1, 2.9, 5.00],
        [1, 3.97, 4.00],
        [1.5, 2.9, 2.64], [1.5, 2.9, 3.27],
        [2, 2.9, 3.22], [2, 2.9, 4.93],
        [1, 5, 5.70], [2, 3, 1.89], [1.5, 2.9, 4.95], [2, 5, 3.81]
    ], dtype=float)
    train_P = np.array([
        [1, 4, 1.89], [1, 5, 1.89], [1, 5, 2.92], [1, 6, 1.89],
        [1, 6, 3.81], [1, 6, 4.53], [2, 5, 1.89], [2, 6, 1.89]
    ], dtype=float)  # 建议指定 dtype，省得混入字符串
    # test = np.array([
    #     [1, 6, 3.81], [1, 6, 4.53], [2, 5, 1.89], [2, 6, 1.89]
    # ], dtype=float)     # 建议指定 dtype，省得混入字符串

    plot_3d_two_sets(
        train_NP, train_P,
        xlim=(0.5, 2.5),
        ylim=(2.5, 6.5),
        zlim=(1.5, 6.5),
        save_path="sim_conditions_PNP_arial.tif"
    )

