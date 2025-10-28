import os
import numpy as np
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# ===== Journal style =====
journal = {
    "fig_size": (10/2.54, 8/2.54),   # cm -> inch
    "font_name": "Arial",   # 可改 "Arial"
    "font_size": 15,
    "line_width": 1.2,
    "axis_line_width": 1.0,
    "dpi": 1200
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
    # seaborn 主题尽量“干净”
    sns.set_theme(context="paper", style="white", font_scale=1)

def savefig_journal(fig, path, j, ext="tiff"):
    ext = ext.lower()
    kwargs = dict(dpi=j["dpi"], bbox_inches="tight")
    if ext in ("tif", "tiff"):
        kwargs.update(format="tiff", pil_kwargs={"compression": "tiff_lzw"})
    fig.savefig(path, **kwargs)

apply_journal_style(journal)
# =========================

# 截取 colormap
def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=256):
    return mcolors.LinearSegmentedColormap.from_list(
        f'trunc({cmap.name},{minval:.2f},{maxval:.2f})',
        cmap(np.linspace(minval, maxval, n))
    )

# 示例数据
norm_cm = np.array([[0.98, 0.02],
                    [0.41, 0.59]])

fig, ax = plt.subplots(figsize=journal["fig_size"])
cmap = truncate_colormap(plt.get_cmap("Blues"), 0.30, 0.95)

# 画热力图
sns.heatmap(
    norm_cm,
    annot=np.round(norm_cm, 2),
    fmt=".2f",
    cmap=cmap,
    vmin=0.0, vmax=1.0,
    xticklabels=['Pred NP', 'Pred P'],
    yticklabels=['True NP', 'True P'],
    square=True,
    linewidths=0.0,
    cbar_kws={"shrink": 0.9, "pad": 0.02},
    annot_kws={"size": journal["font_size"]}
)

# 坐标轴刻度文字大小
ax.tick_params(axis='both', which='major',
               labelsize=journal["font_size"],
               length=3, width=journal["axis_line_width"],
               direction='in')

# 轴与标签（期刊风格：无标题、刻度朝内）
# ax.set_xlabel('Predicted')
# ax.set_ylabel('True')
# ax.minorticks_on()
# ax.tick_params(top=True, right=True)

fig.tight_layout()
out_path = "top_folds_confusion_matrix_0910." + SAVE_EXT
savefig_journal(fig, out_path, journal, ext=SAVE_EXT)
plt.close(fig)
