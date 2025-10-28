import os, glob
import pandas as pd
from pathlib import Path
import os
import torch
import numpy as np
import argparse
import matplotlib as mpl
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold
import torch.nn.functional as F
import pandas as pd
import numpy as np
from io import StringIO
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import matplotlib.colors as mcolors
from matplotlib.colors import ListedColormap, BoundaryNorm
from scipy.spatial import ConvexHull
from typing import Optional, Tuple
from scipy.ndimage import gaussian_filter
from skimage.transform import rescale
from skimage.measure import label, regionprops
import matplotlib.cm as cm

# ==== Journal style ===========================================================
journal = {
    "fig_size": (9/2.54, 7/2.54),   # cm → inch
    "font_name": "Arial",
    "font_size": 12,
    "line_width": 1.5,
    "axis_line_width": 1.0,
    "dpi": 1200
}
SAVE_EXT = "tiff"  # 可改为: "pdf" / "png" / "svg"

def apply_journal_style(j):
    mpl.rcParams.update({
        # 尺寸/分辨率
        "figure.figsize": j["fig_size"],
        "savefig.dpi": j["dpi"],
        # 字体与字号
        "font.family": j["font_name"],
        "font.size": j["font_size"],
        "axes.labelsize": j["font_size"],
        "axes.titlesize": j["font_size"],
        "xtick.labelsize": j["font_size"],
        "ytick.labelsize": j["font_size"],
        "legend.fontsize": j["font_size"],
        # 线宽与轴样式
        "lines.linewidth": j["line_width"],
        "axes.linewidth": j["axis_line_width"],
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.major.size": 3,
        "ytick.major.size": 3,
        "xtick.minor.size": 1.5,
        "ytick.minor.size": 1.5,
        "xtick.major.width": j["axis_line_width"],
        "ytick.major.width": j["axis_line_width"],
        "xtick.minor.width": j["axis_line_width"],
        "ytick.minor.width": j["axis_line_width"],
        "legend.frameon": False,
        "axes.grid": False,
        "pdf.fonttype": 42,   # 避免 Type3
        "ps.fonttype": 42,
        # 数学字体用 STIX（变量/希腊字母有斜体字形）
        "mathtext.fontset": "stix",
        # 不强制 regular，以便变量可斜体：若想全局变量斜体可启用下一行
        # "mathtext.default": "it",
    })

def savefig_journal(fig, path, j, ext="tiff"):
    ext = ext.lower()
    kwargs = dict(dpi=j["dpi"], bbox_inches="tight")
    if ext in ("tif", "tiff"):
        kwargs.update(format="tiff", pil_kwargs={"compression": "tiff_lzw"})
    fig.savefig(path, **kwargs)

def imsave_journal(path, arr, cmap=None, j=journal, ext="tiff"):
    """对纯图像矩阵的保存也使用期刊参数（如需要 TIFF+LZW）。"""
    ext = ext.lower()
    if ext in ("tif", "tiff"):
        plt.imsave(path, arr, cmap=cmap, format="tiff",
                   dpi=j["dpi"], pil_kwargs={"compression": "tiff_lzw"})
    else:
        plt.imsave(path, arr, cmap=cmap, format=ext, dpi=j["dpi"])

apply_journal_style(journal)
# ============================================================================

def plot_radial_comparison(stats_t, stats_p, out_dir, prefix='', pixel2mm=100/151,
                           jconf=journal, save_ext=SAVE_EXT):
    """
    比较 target 与 pred 的 radial 统计并保存曲线图
    metrics 包含：
      - count, count_density,
      - mean_depth, depth_density,
      - mean_diam, diam_density,
      - mean_volume, volume_density,
      - mean_area, area_density
    """
    import os
    import matplotlib.pyplot as plt
    from sklearn.metrics import r2_score  # 新增
    os.makedirs(out_dir, exist_ok=True)

    # 待绘制的指标及标签
    metrics = [
        ('count', 'Pit Count'),
        ('count_density', 'Pit Count Density'),
        ('mean_depth', 'Mean Depth'),
        ('depth_density', 'Depth Density'),
        ('mean_diam', 'Mean Diameter'),
        ('diam_density', 'Diameter Density'),
        ('mean_volume', 'Mean Volume'),
        ('volume_density', 'Volume Density'),
        ('mean_area', r'$\mathit{\bar{A}_{\mathit{D}}}$ (mm²)'),
        ('area_density', r'$\mathit{\Gamma}_{\mathit{D}}$'),
    ]

    for key, ylabel in metrics:
        if key not in stats_t:
            continue  # 跳过不存在的字段

        # # 计算 R²（回归拟合程度）
        # try:
        #     r2 = r2_score(stats_t[key], stats_p[key])
        # except Exception:
        #     r2 = float('nan')  # 如果数据不匹配或出错

        fig, ax = plt.subplots(figsize=jconf["fig_size"])
        ax.plot(stats_t['r_mid'] * pixel2mm, stats_t[key], 'o-', linewidth=jconf["line_width"], label='target')
        ax.plot(stats_p['r_mid'] * pixel2mm, stats_p[key], 'x-', linewidth=jconf["line_width"], label='prediction')

        ax.set_xlabel('Radius (mm)')
        ax.set_ylabel(ylabel)
        # ax.set_title(f'{prefix} {ylabel} vs Radius')  # 期刊常不需要标题，如需可打开
        ax.minorticks_on()
        ax.tick_params(top=True, right=True)
        ax.legend(loc="best", frameon=False)

        fname = os.path.join(out_dir, f"{prefix}_radial_{key}_center.{save_ext}")
        fig.tight_layout()
        savefig_journal(fig, fname, jconf, ext=save_ext)
        plt.close(fig)

ROOT = r"D:\SIMULATION\p01_DebrisCloudDamageDataBase\params\thickness_binary_128"  # 你的 param_dir
PIXEL2MM = 100/151

patterns = [
    os.path.join(ROOT, "fold_4", "compute-area-0.35_eval_13"),  # 你的eval子目录模式
    os.path.join(ROOT, "fold_1", "epoch500_compute-area-0.45_eval_10"),  # 你的eval子目录模式
    os.path.join(ROOT, "fold_1", "epoch500_compute-area-0.6_eval_9"),  # 你的eval子目录模式
    os.path.join(ROOT, "fold_3", "compute-area-0.45_eval_0"),  # 你的eval子目录模式
]

for pat in patterns:
    for subdir in glob.glob(pat, recursive=True):
        t_csv = Path(subdir) / "target_stats.csv"
        p_csv = Path(subdir) / "pred_stats.csv"
        if not (t_csv.exists() and p_csv.exists()):
            continue
        df_t = pd.read_csv(t_csv)
        df_p = pd.read_csv(p_csv)

        plot_radial_comparison(
            df_t, df_p,
            out_dir=subdir,
            prefix="redraw",
            pixel2mm=PIXEL2MM,
            jconf=journal,
            save_ext=SAVE_EXT,
        )
        print(f"[OK] redrawn: {subdir}")
