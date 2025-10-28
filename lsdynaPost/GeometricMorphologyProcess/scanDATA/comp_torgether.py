"""
multi_plot_pits.py
读取已计算好的径向统计 (radial_stats.csv) → 在同一张画布上对比绘图（含期刊输出规范）
"""
import os, glob
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

# ==== 0. 期刊输出规范 ====
journal = {
    "fig_size": (10/2.54, 9/2.54),   # cm → inch
    "font_name": "Arial Unicode MS",
    "font_size": 12,
    "line_width": 1.5,
    "axis_line_width": 1.0,
    "dpi": 1200
}
SAVE_EXT = "tiff"  # 可改 "png" / "pdf" / "svg" 等

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
        # 矢量字体（避免 Type 3）
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        # "mathtext.fontset": "stix",  # 提供希腊/拉丁的数学斜体字形
        "mathtext.default": "regular",
    })

def savefig_journal(fig, path, j, ext="tiff"):
    ext = ext.lower()
    kwargs = dict(dpi=j["dpi"], bbox_inches="tight")
    if ext in ("tif", "tiff"):
        kwargs.update(format="tiff", pil_kwargs={"compression": "tiff_lzw"})
    fig.savefig(path, **kwargs)

apply_journal_style(journal)

# ==== 1. 配置 ====
# PREFIXES = ['dwh2', 'dwh3', 'dwh4', 'dwh5', 'dwh6', 'dwh7']  # 要比较的样件
PREFIXES = ['dwh2', 'dwh4', 'dwh7']  # 要比较的样件
ANALYSIS_DIR = r'D:\PHDstudent\博士论文\损伤代理模型\01 小论文\实验数据\板扫描模型\out\analysis'
PIXEL2MM = 160 / 1000  # 分辨率 mm/px

# ==== 2. 读取已计算好的径向统计 CSV ====
radial_stats = {}  # {prefix: DataFrame}

for pre in PREFIXES:
    csv_path = os.path.join(ANALYSIS_DIR, f"{pre}_pits_state_radial.csv")
    if not os.path.isfile(csv_path):
        print(f"[WARN] 找不到 {csv_path}，已跳过。")
        continue

    df_stats = pd.read_csv(csv_path)
    if not {'r_mid', 'count', 'count_density'}.issubset(df_stats.columns):
        raise KeyError(f"{csv_path} 缺少必要列")

    radial_stats[pre] = df_stats

if not radial_stats:
    raise RuntimeError("没有任何径向统计数据可绘图，请检查 CSV 路径和文件名。")

# ==== 3. 通用绘图函数 ====
# def plot_multi(stats_dict, x_key, metrics, x_label, title_suffix, out_prefix,
#                x_transform=lambda x: x, out_dir=ANALYSIS_DIR, jconf=journal, save_ext=SAVE_EXT):
#     """
#     stats_dict : {prefix: DataFrame}
#     x_key      : DataFrame 中作为 x 轴的列名
#     metrics    : [(col, y_label), ...]
#     x_transform: 对 x 轴做单位换算，如 radius*pixel2mm
#     """
#     os.makedirs(out_dir, exist_ok=True)
#
#     for col, y_label in metrics:
#         fig, ax = plt.subplots(figsize=jconf["fig_size"])
#         for pre, st in stats_dict.items():
#             x = x_transform(st[x_key].values)
#             ax.plot(x, st[col].values, '-', linewidth=jconf["line_width"], label=pre)
#
#         ax.set_xlabel(x_label)
#         ax.set_ylabel(y_label)
#         # ax.set_title(f"{y_label} vs {title_suffix}")
#         ax.minorticks_on()
#         ax.grid(False)  # 期刊通常关闭背景网格；如需网格改 True
#         ax.legend(loc="best", frameon=False)
#
#         fname = os.path.join(out_dir, f"findcenter_complete_{out_prefix}_{col}_bin5mm.{save_ext}")
#         fig.tight_layout()
#         savefig_journal(fig, fname, jconf, ext=save_ext)
#         plt.close(fig)
#         print(f"[INFO] 已保存: {fname}")
def plot_multi(
    stats_dict, x_key, metrics, x_label, title_suffix, out_prefix,
    x_transform=lambda x: x, out_dir=ANALYSIS_DIR,
    legend=None,                # dict / list / callable / None
    hide_prefixes=None,         # 可选：不想出现在图例里的前缀集合
    legend_kw=None,             # 传给 ax.legend 的参数 dict（如 loc、ncol 等）
    jconf=journal, save_ext=SAVE_EXT
):
    """
    legend:
      - dict:   { 'dwh2': '10 J', 'dwh3': '20 J', ... }
      - list:   ['10 J','20 J', ...]（顺序需与 stats_dict 的插入顺序一致）
      - func:   lambda pre: f'Case {pre[-1]}'
      - None:   使用默认的 pre
    hide_prefixes:
      - set/list，如 {'dwh5','dwh7'}，这些曲线仍绘制但不进图例
    """
    os.makedirs(out_dir, exist_ok=True)
    # 默认 legend 样式（期刊友好）
    default_legend_kw = dict(loc="best", frameon=False, ncol=1, handlelength=2.0, columnspacing=1.0, borderaxespad=0.5)

    for col, y_label in metrics:
        fig, ax = plt.subplots(figsize=jconf["fig_size"])
        lines = []
        labels = []

        # 确保顺序稳定：radial_stats 是按 PREFIXES 构造的，dict 保持插入顺序
        for idx, (pre, st) in enumerate(stats_dict.items()):
            x = x_transform(st[x_key].values)

            # 生成自定义图例文本
            if hide_prefixes and pre in set(hide_prefixes):
                lbl = "_nolegend_"  # matplotlib 约定：以下划线开头的 label 不进图例
            else:
                if isinstance(legend, dict):
                    lbl = legend.get(pre, pre)
                elif isinstance(legend, list):
                    # 按顺序取；长度需匹配
                    if idx >= len(legend):
                        raise ValueError("legend 列表长度不足以匹配所有曲线。")
                    lbl = legend[idx]
                elif callable(legend):
                    lbl = legend(pre)
                else:
                    lbl = pre  # 默认

            line, = ax.plot(x, st[col].values, '-', linewidth=jconf["line_width"], label=lbl)
            lines.append(line)
            labels.append(lbl)

        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        # ax.set_title(f"{y_label} vs {title_suffix}")
        ax.minorticks_on()
        ax.grid(False)

        # 仅当存在非隐藏标签时再画图例
        show_labels = [lb for lb in labels if not (isinstance(lb, str) and lb.startswith("_"))]
        if len(show_labels) > 0:
            # 过滤掉隐藏句柄
            show_handles = [ln for ln, lb in zip(lines, labels) if not (isinstance(lb, str) and lb.startswith("_"))]
            kw = default_legend_kw.copy()
            if legend_kw: kw.update(legend_kw)
            ax.legend(show_handles, show_labels, **kw)

        fname = os.path.join(out_dir, f"findcenter_complete_{out_prefix}_{col}_bin5mm.{save_ext}")
        fig.tight_layout()
        savefig_journal(fig, fname, jconf, ext=save_ext)
        plt.close(fig)
        print(f"[INFO] 已保存: {fname}")


# ==== 4. 绘制径向对比 ====
radial_metrics = [
    ('count', 'Pit Count'),
    ('count_density', 'Pit Count Density'),
    ('mean_depth', 'Mean Depth (mm)'),
    ('depth_density', 'Depth Density'),
    ('mean_diam', 'Mean Diameter (mm)'),
    ('diam_density', 'Diameter Density'),
    ('mean_volume', 'Mean Volume Proxy'),
    ('volume_density', 'Volume Density'),
    ('mean_area', r'$ACI_{pit}$,  $\mathit{\bar{A}_{\mathit{D}}}$ (mm²)'),
    ('area_density', r'$CDI_{pit}$,  $\mathit{\Gamma}_{\mathit{D}}$ (%)'),
]

LEGEND_MAP = {
    'dwh2': '① 4.01 / 3.97 / 1.0',
    'dwh3': '15 J, 30°',
    'dwh4': '② 3.70 / 3.18 / 1.0',
    'dwh5': '25 J, 45°',
    'dwh6': '30 J, 45°',
    'dwh7': '③ 2.70 / 3.18 / 2.0',
}

plot_multi(
    radial_stats,
    x_key='r_mid',
    metrics=radial_metrics,
    x_label='Radius (mm)',
    title_suffix='Radius',
    out_prefix='multi_radial',
    x_transform=lambda r: r * PIXEL2MM,  # px → mm
    legend=LEGEND_MAP,
    # legend_kw=dict(loc='upper right', ncol=2)

)
