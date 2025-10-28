import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.colors import BoundaryNorm

# ========= Journal style =========
journal = {
    "fig_size": (16/2.54, 16/2.54),  # cm -> inch
    "font_name": "Arial",
    "font_size": 18,
    "line_width": 1.2,
    "axis_line_width": 1.0,
    "dpi": 1200,
}
SAVE_EXT = "tiff"  # 可改 "pdf"/"png"/"svg"/"tiff"

def apply_journal_style(j):
    mpl.rcParams.update({
        "figure.figsize": j["fig_size"],
        "savefig.dpi": j["dpi"],
        "font.family": j["font_name"],
        "font.size": j["font_size"],
        "axes.labelsize": j["font_size"],
        "axes.titlesize": j["font_size"],
        "xtick.labelsize": j["font_size"],
        "ytick.labelsize": j["font_size"],
        "legend.fontsize": j["font_size"],
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
        "axes.grid": False,
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "mathtext.fontset": "stix",
    })

def savefig_journal(fig, path, j, ext="tiff"):
    ext = ext.lower()
    kwargs = dict(dpi=j["dpi"], bbox_inches="tight")
    if ext in ("tif", "tiff"):
        kwargs.update(format="tiff", pil_kwargs={"compression": "tiff_lzw"})
    fig.savefig(path, **kwargs)

apply_journal_style(journal)
# =================================

def make_percentile_discrete_norm(data_for_bins, pct_edges, cmap_name='jet'):
    """
    自定义百分位边界 → 离散 colormap + BoundaryNorm
    pct_edges: 升序百分位端点（0~100），如 [0,1,11,25,50,75,90,100]
    """
    pct_edges = np.asarray(pct_edges, dtype=float)
    assert np.all(np.diff(pct_edges) >= 0), "pct_edges 必须非降序"
    assert pct_edges[0] >= 0 and pct_edges[-1] <= 100 and pct_edges.size >= 2

    data_flat = np.asarray(data_for_bins).ravel()
    # 用 nanpercentile 更稳健
    boundaries = np.nanpercentile(data_flat, pct_edges)

    # 防重复边界（常数或极端情况下）
    eps = np.finfo(float).eps
    for i in range(1, len(boundaries)):
        if not np.isfinite(boundaries[i]) or boundaries[i] <= boundaries[i-1]:
            boundaries[i] = boundaries[i-1] + eps

    n_bins = len(boundaries) - 1
    cmap = mpl.cm.get_cmap(cmap_name, n_bins)   # 离散 jet
    norm = BoundaryNorm(boundaries, ncolors=n_bins, clip=True)
    return cmap, norm, boundaries

# --------- 数据读取与处理 ----------
path = r'D:\SIMULATION\p01_DebrisCloudDamageDataBase\originalData\eval\original_elementValue_npyFiles\VMStress_Data\Bumper1_dp397_vp400_elementMax_VMStress.npy'

data = np.load(path)                         # (181, 3750000)
data_3d = data.reshape((-1, 15, 500, 500))  # (?, 15, 500, 500)
data_3d = data_3d[-1, :, :, :]              # 取最后一组 (15, 500, 500)

# 最大投影（或改为你需要的其它投影/切片）
max_img = np.nanmax(data_3d, axis=0)

# --------- 自定义百分位边界（例：前1%，后按 10%,14%,25%,25%,15%,10%） ----------
PCT_EDGES = [0, 99, 99.1, 99.2, 99.3, 99.4, 99.5, 99.6, 99.7, 99.8, 99.9, 99.95, 100]
cmap, norm, boundaries = make_percentile_discrete_norm(data_for_bins=data_3d,
                                                       pct_edges=PCT_EDGES,
                                                       cmap_name='coolwarm')

# --------- 绘图 ----------
fig, ax = plt.subplots(figsize=journal["fig_size"])
im = ax.imshow(max_img, cmap=cmap, norm=norm, origin='lower', interpolation='nearest')

# 色条：用“数值边界”作为刻度
import matplotlib.ticker as mticker

# —— 色条放在图的下方（水平）——
cbar = fig.colorbar(
    im, ax=ax,
    boundaries=boundaries,          # 你的离散边界
    ticks=boundaries,               # 用“数值边界”做刻度
    orientation='horizontal',       # 关键：水平
    fraction=0.045,                  # 色条相对宽度
    pad=0.08                        # 与主图的间距，可按需调
)

# 数值格式（注意现在在 x 轴上）

import matplotlib.ticker as mticker

fmt_sci1 = mticker.FuncFormatter(lambda x, pos: f'{x:.1e}')

# 水平色条（在图下方）
cbar.ax.xaxis.set_major_formatter(fmt_sci1)

# 垂直色条（在图右侧）则改为：
# cbar.ax.yaxis.set_major_formatter(fmt_sci1)

# cbar.ax.xaxis.set_major_formatter(mticker.FormatStrFormatter('%.3g'))
keep = boundaries[::4]  # 每隔一个
cbar.set_ticks(keep)

cbar.set_label('Von Mises Stress (Mbar)', labelpad=6)


ax.minorticks_on()
ax.tick_params(top=True, right=True)
fig.tight_layout()

# --------- 保存 ----------
out_dir = os.path.join(os.path.dirname(path), "plots")
os.makedirs(out_dir, exist_ok=True)
fname = os.path.splitext(os.path.basename(path))[0] + "_max_projection_coolwarm_discrete_pct"
out_path = os.path.join(out_dir, f"{fname}.{SAVE_EXT}")
savefig_journal(fig, out_path, journal, ext=SAVE_EXT)
plt.close(fig)
print("[Saved]", out_path)
