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

from dataset_pt import PreprocessedPTDataset
from model.model_cnn_151 import SimpleCNN
from train_SM import edge_loss



# ==== Journal style ===========================================================
journal = {
    "fig_size": (9/2.54, 7/2.54),   # cm → inch
    "font_name": "Times New Roman",
    "font_size": 10,
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

def compute_radius_99(df, img_shape, center=None, coverage=0.99, verbose=True):
    """
    计算覆盖99%损伤面积所需半径
    df: 包含列 ['y_px','x_px','area_px2'] 的 DataFrame
    img_shape: (H,W)
    """
    H, W = img_shape[-2:]
    if center is None:
        center = (H/2.0, W/2.0)
    cy, cx = center
    r = np.hypot(df['x_px'] - cx, df['y_px'] - cy)
    df_sorted = df.assign(r=r).sort_values('r')
    cum_area = df_sorted['area_px2'].cumsum()
    total_area = cum_area.iloc[-1]
    idx_99 = np.searchsorted(cum_area, coverage * total_area)
    r_99 = df_sorted.iloc[idx_99]['r']
    if verbose:
        print(f"[INFO] 99% area within radius <= {r_99:.2f}px")
    return r_99


def radial_bin_stats(df, img_shape, bin_width_px=8, center=None):
    """
    按同心圆环分箱统计并按环面积归一化:
      - r_mid           : 环中心平均半径
      - count           : 坑总数
      - count_density   : 坑密度 (#/px^2)
      - mean_depth      : 平均坑深
      - depth_density   : 坑深累积密度 (sum_depth/环面积)
      - mean_diam       : 平均坑直径
      - diam_density    : 直径累积密度 (sum_diam/环面积)
      - mean_volume     : 平均体积代理
      - volume_density  : 体积代理密度 (sum_volume/环面积)
      - mean_area       : 平均坑面积
      - area_density    : 坑总面积密度 (sum_area/环面积)
    df: 包含 ['y_px','x_px','depth_est','radius_px','volume_px3','area_px2']
    """
    import numpy as np
    import pandas as pd

    H, W = img_shape[-2:]
    if center is None:
        center = (H / 2.0, W / 2.0)
    cx, cy = center

    r = np.hypot(df['x_px'] - cx, df['y_px'] - cy)
    # print(r)
    bin_idx = (r // bin_width_px).astype(int)
    # print(bin_idx)
    df2 = df.assign(r=r, bin=bin_idx)

    # grp = df2.groupby('bin')
    # stats = grp.agg(
    #     r_mid      = ('r',        'mean'),
    #     count      = ('r',        'size'),
    #     sum_depth  = ('depth_est','sum'),
    #     mean_depth = ('depth_est','mean'),
    #     sum_diam   = ('radius_px', lambda x: (2 * x).sum()),
    #     mean_diam  = ('radius_px', lambda x: (2 * x).mean()),
    #     sum_volume = ('volume_px3','sum'),
    #     mean_volume= ('volume_px3','mean'),
    #     sum_area   = ('area_px2',  'sum'),
    #     mean_area  = ('area_px2',  'mean'),
    # ).reset_index()
    # ========== 新增：确保所有 bin 都有记录 ==========
    max_bin = int(np.hypot(H, W)/2 // bin_width_px) + 1
    all_bins = pd.Index(np.arange(0, max_bin), name='bin')

    grp = df2.groupby('bin')
    stats = grp.agg(
        r_mid=('r', 'mean'),
        count=('r', 'size'),
        sum_depth=('depth_est', 'sum'),
        mean_depth=('depth_est', 'mean'),
        sum_diam=('radius_px', lambda x: (2 * x).sum()),
        mean_diam=('radius_px', lambda x: (2 * x).mean()),
        sum_volume=('volume_px3', 'sum'),
        mean_volume=('volume_px3', 'mean'),
        sum_area=('area_px2', 'sum'),
        mean_area=('area_px2', 'mean'),
    )

    # 用所有 bin 的 index 填补缺失 bin
    stats = stats.reindex(all_bins).reset_index()
    # print(stats['bin'])

    # 对于空 bin，r_mid 用中心值近似填补
    stats['r_mid'] = stats['r_mid'].fillna((stats['bin'] + 0.5) * bin_width_px)
    # print(stats['r_mid'])

    # 缺失的统计数据填 0（或 np.nan，根据你是否想要可视化为0还是缺失）
    stats[['count', 'sum_depth', 'sum_diam', 'sum_volume', 'sum_area']] = \
        stats[['count', 'sum_depth', 'sum_diam', 'sum_volume', 'sum_area']].fillna(0)

    k = stats['bin'].values
    r_min = k * bin_width_px
    r_max = (k + 1) * bin_width_px
    ring_area = np.pi * (r_max ** 2 - r_min ** 2)
    stats['ring_area_px2'] = ring_area

    stats['count_density']  = stats['count']      / stats['ring_area_px2']
    stats['depth_density']  = stats['sum_depth']  / stats['ring_area_px2']
    stats['diam_density']   = stats['sum_diam']   / stats['ring_area_px2']
    stats['volume_density'] = stats['sum_volume'] / stats['ring_area_px2']
    stats['area_density']   = stats['sum_area']   / stats['ring_area_px2']

    return stats

def extract_pits(img, bg_factor, gf_sigma, pixel2mm=100/151):
    """
    提取图像中的坑信息，返回包含过滤后斑点信息的 DataFrame。
    """
    # 如果输入为 torch.Tensor，先转到 CPU numpy
    if isinstance(img, torch.Tensor):
        img = img.detach().cpu().numpy().squeeze()

    Z = img
    print(np.max(Z), np.min(Z))
    # print(Z.shape)

    # ======= 1. 背景扣除 =======
    background = gaussian_filter(Z, sigma=gf_sigma)  # sigma越小，平滑效果越强
    Z_res = background - Z  # 小坑为负
    # Z_res = -img
    print(np.max(Z_res), np.min(Z_res))
    background_threshold = (np.max(Z_res)-np.min(Z_res))*bg_factor + np.min(Z_res)

    # ======= 3. 使用阈值对图像进行二值化处理 =======
    Z_binary = Z_res > background_threshold  # 将大于背景的区域设为 1，小于背景的区域设为 0

    # ======= 4. 统计 Z_binary 中斑点的个数和直径 =======
    labeled_image = label(Z_binary)
    regions = regionprops(labeled_image, intensity_image=Z_res)

    spots_info = []
    all_areas = []  # 用来存储所有斑点的面积

    for region in regions:
        # print(f"Region bbox: {region.bbox}")  # 输出 bbox，检查其返回的内容

        area = region.area
        all_areas.append(area)

        diameter = 2 * np.sqrt(area / np.pi)
        minr, minc, maxr, maxc = region.bbox
        region_max_value = np.max(Z_res[minr:maxr, minc:maxc])

        # 计算半径和体积代理
        radius_px = pixel2mm * diameter / 2  # 假设坑是圆形的
        depth_est = pixel2mm * region_max_value  # 可以使用最大值作为深度估计，或者根据实际情况调整
        volume_px3 = pixel2mm**2 * area * depth_est / 3  # 用面积和深度估计计算体积代理

        spots_info.append({
            'label': region.label,
            'area': area,
            'diameter': diameter,
            'max_value_in_region': region_max_value,
            'x_px': region.centroid[1],
            'y_px': region.centroid[0],
            'area_px2': area,
            'depth_est': depth_est,  # 添加 depth_est
            'radius_px': radius_px,  # 添加 radius_px
            'volume_px3': volume_px3  # 添加 volume_px3
        })

    # 如果没有任何斑点，直接返回空
    if not all_areas:
        print("[WARN] No pits detected in this image—返回空结果。")
        cols = ['label', 'area', 'diameter', 'max_value_in_region', 'x_px', 'y_px', 'area_px2']
        return pd.DataFrame(columns=cols)

    # 计算面积的 99% 分位数
    area_threshold = np.percentile(all_areas, 0)
    filtered_spots_info = [spot for spot in spots_info if spot['area'] >= area_threshold]

    # 将过滤后的斑点信息转换为 DataFrame
    filtered_spots_df = pd.DataFrame(filtered_spots_info)

    # print("过滤后的斑点统计结果：")
    # print(filtered_spots_df)

    # r_99 = compute_radius_99(filtered_spots_df, Z.shape)
    # print(f"99% 的损伤面积覆盖半径为: {r_99*0.2*2:.2f} mm")

    # # ======= 6. 可视化原始图像与二值化结果 =======
    # fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    # ax[0].imshow(-Z, cmap='coolwarm');
    # ax[0].set_title('Residual Depth')
    # ax[1].imshow(Z_binary, cmap='gray');
    # ax[1].set_title('Binary Image (Thresholding)')
    #
    # # 在二值化图像上标出斑点（只标出过滤后的斑点）
    # for region in regions:
    #     # print(f"Region bbox: {region.bbox}")  # 输出 bbox，检查其返回的内容
    #     area = region.area
    #     if area >= area_threshold:
    #         minr, minc, maxr, maxc = region.bbox
    #         ax[1].add_patch(
    #             plt.Rectangle((minc, minr), maxc - minc, maxr - minr, edgecolor='red', facecolor='none', linewidth=0.5))
    #
    # for a in ax: a.axis('off')
    # plt.tight_layout();
    # plt.show()

    return filtered_spots_df, Z_binary

# def plot_radial_cum_comparison(stats_t, stats_p, out_dir, prefix='', pixel2mm=1.51):
#     """
#     比较 target 与 pred 的 radial 统计并保存曲线图
#     metrics 包含：
#       - count, count_density,
#       - mean_depth, depth_density,
#       - mean_diam, diam_density,
#       - mean_volume, volume_density
#     """
#     os.makedirs(out_dir, exist_ok=True)
#     # 待绘制的指标及标签
#     metrics = [
#         ('count', 'Pit Count'),
#         ('count_density', 'Pit Count Density'),
#         ('mean_depth', 'Mean Depth'),
#         ('depth_density', 'Depth Density'),
#         ('mean_diam', 'Mean Diameter'),
#         ('diam_density', 'Diameter Density'),
#         ('mean_volume', 'Mean Volume'),
#         ('volume_density', 'Volume Density'),
#     ]
#
#     for key, ylabel in metrics:
#         plt.figure()
#         # 横轴：半径转换为物理单位
#         radius = stats_t['r_max'] * pixel2mm
#         plt.plot(radius, stats_t[key], 'o-', linewidth=5, label='target')
#         plt.plot(radius, stats_p[key], 's--', label='pred')
#         plt.xlabel('Radius (mm)')
#         plt.ylabel(ylabel)
#         plt.title(f'{prefix} {ylabel} vs Radius')
#         plt.legend()
#         plt.grid(True)
#         fname = os.path.join(out_dir, f"{prefix}_radial_{key}_cum_center.png")
#         plt.tight_layout()
#         plt.savefig(fname, dpi=300)
#         plt.close()

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
        ('mean_area', r'Average Damage Area $\mathit{\bar{A}}$'),
        ('area_density', r'Damage Area Density $\mathit{\Gamma}_{\mathit{D}}$'),
    ]

    for key, ylabel in metrics:
        if key not in stats_t:
            continue  # 跳过不存在的字段

        # 计算 R²（回归拟合程度）
        try:
            r2 = r2_score(stats_t[key], stats_p[key])
        except Exception:
            r2 = float('nan')  # 如果数据不匹配或出错

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


def plot_distribution_curves(df_t, df_p, out_dir, prefix='', bins=20,
                             jconf=journal, save_ext=SAVE_EXT):
    """
    绘制并保存所有坑原始分布统计曲线:
      - 直径分布: 坑直径 vs 数量
      - 深度分布: 坑深度 vs 数量
      - 体积分布: 坑体积 vs 数量
    """
    os.makedirs(out_dir, exist_ok=True)
    dist_relations = [
        (2*df_t['radius_px'], 2*df_p['radius_px'], 'diameter', 'Pit Diameter (px)'),
        (df_t['depth_est'], df_p['depth_est'], 'depth', 'Pit Depth'),
        (df_t['volume_px3'], df_p['volume_px3'], 'volume', 'Pit Volume Proxy')
    ]
    for data_t, data_p, key, xlabel in dist_relations:
        # 定义统一bins
        min_val = min(data_t.min(), data_p.min())
        max_val = max(data_t.max(), data_p.max())
        bin_edges = np.linspace(min_val, max_val, bins+1)
        cnt_t, _ = np.histogram(data_t, bins=bin_edges)
        cnt_p, _ = np.histogram(data_p, bins=bin_edges)
        centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        fig, ax = plt.subplots(figsize=jconf["fig_size"])
        ax.plot(centers, cnt_t, 'o-', label='target', linewidth=jconf["line_width"])
        ax.plot(centers, cnt_p, 's--', label='pred', linewidth=jconf["line_width"])
        ax.set_xlabel(xlabel)
        ax.set_ylabel('Pit count')
        # ax.set_title(f'{prefix} Pit count distribution by {xlabel}')
        ax.minorticks_on()
        ax.tick_params(top=True, right=True)
        ax.legend(loc="best", frameon=False)

        fname = os.path.join(out_dir, f"{prefix}_dist_{key}.{save_ext}")
        fig.tight_layout()
        savefig_journal(fig, fname, jconf, ext=save_ext)
        plt.close(fig)

def save_pit_results(df, out_dir, prefix=''):
    """
    将pit点DataFrame保存为CSV
    """
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{prefix}_pits.csv")
    df.to_csv(path, index=False)
    print(f"[Saved] {path}")

def evaluate_model(model, dataloader, device, param_dir, opt, k,
                   global_stats=None, max_visualizations=12):
    out_root = args.param_dir

    model.eval()
    val_losses, ssim_scores, psnr_scores, mse_scores = [], [], [], []
    vis_counter = 0

    # 构造半截 colormap（例子：取 coolwarm 的后半部分暖色区）
    full_cmap = cm.get_cmap('coolwarm', 256)  # 原始 coolwarm，有 256 个颜色
    half_cmap = full_cmap(np.linspace(0, 0.5, 128))  # 取 50%~100% 的颜色
    half_cmap = cm.colors.ListedColormap(half_cmap)  # 转成新的 cmap 对象

    with torch.no_grad():
        for batch_idx, (signals, targets) in enumerate(dataloader):
            signals, targets = signals.to(device), targets.to(device)
            preds = model(signals)
            loss_v = F.mse_loss(preds, targets)
            val_losses.append(loss_v.item())
            ssim_scores.append(1 - F.l1_loss(preds, targets).item())
            psnr_scores.append(10 * torch.log10(1 / loss_v).item())
            mse_scores.append(F.mse_loss(preds, targets).item())

            max_per_batch = 10  # 每个 batch 最多保存几个样本

            # 可视化每个样本，限制最大图像数量
            for i in range(min(signals.shape[0], max_per_batch)):
                if vis_counter >= max_visualizations:
                    break

                pred_img = preds[i].cpu().numpy()
                target_img = targets[i].cpu().numpy()

                if global_stats is not None:
                    pred_img = pred_img * (global_stats['target_max'] - global_stats['target_min']) + global_stats['target_min']
                    target_img = target_img * (global_stats['target_max'] - global_stats['target_min']) + global_stats['target_min']

                num_channels = target_img.shape[0]
                for c in range(num_channels):
                    save_dir = os.path.join(param_dir, f"epoch500_compute-area-0.45_eval_{vis_counter}") #0.25
                    os.makedirs(save_dir, exist_ok=True)  # 如果不存在则创建

                    # —— 统一 raster 输出：TIFF+LZW，指定 dpi
                    raw_t = os.path.join(param_dir, f"eval_{vis_counter}_target_C{c}.{SAVE_EXT}")
                    imsave_journal(raw_t, target_img[c], cmap='coolwarm', ext=SAVE_EXT)
                    raw_p = os.path.join(param_dir, f"eval_{vis_counter}_pred_C{c}.{SAVE_EXT}")
                    imsave_journal(raw_p, pred_img[c], cmap='coolwarm', ext=SAVE_EXT)

                    # 期刊配色版本
                    save_t = os.path.join(save_dir, f"target_C{c}_g.{SAVE_EXT}")
                    imsave_journal(save_t, target_img[c], cmap=half_cmap, ext=SAVE_EXT)
                    save_p = os.path.join(save_dir, f"pred_C{c}_g.{SAVE_EXT}")
                    imsave_journal(save_p, pred_img[c], cmap=half_cmap, ext=SAVE_EXT)

                    np.save(os.path.join(save_dir, f"target_C{c}.npy"), target_img[c])
                    np.save(os.path.join(save_dir, f"pred_C{c}.npy"), pred_img[c])

                    # df_p = extract_pits(-pred_img[c], bg_factor=0.47, gf_sigma=5)

                    pi, ti = preds[c], targets[c]
                    # 提取pit数据
                    # df_t = extract_pits(-ti, bg_factor=0.47, gf_sigma=5)
                    # df_p = extract_pits(-pi, bg_factor=0.47, gf_sigma=5)
                    df_t, _ = extract_pits(-ti, bg_factor=0.5, gf_sigma=5)
                    df_p, p_binary = extract_pits(-pi, bg_factor=0.45, gf_sigma=5)
                    # 也按 journal 保存二值图
                    save_pb = os.path.join(save_dir, f"pred_binary_C{c}_g.{SAVE_EXT}")
                    imsave_journal(save_pb, p_binary, cmap=half_cmap, ext=SAVE_EXT)

                    # 保存点结果
                    subdir = save_dir
                    save_pit_results(df_t, subdir, 'target')
                    save_pit_results(df_p, subdir, 'pred')

                    # —— 先计算一下 r_99（假设你已经在这一通道里算出了 df_t、df_p）
                    # 这里以 target 为例，pred 同理
                    r_99_t = compute_radius_99(df_t, target_img[c].shape)

                    # —— 同样的逻辑用于 pred
                    r_99_p = compute_radius_99(df_p, pred_img[c].shape)

                    # 分箱统计
                    stats_t = radial_bin_stats(df_t, ti.shape, bin_width_px=8)
                    stats_p = radial_bin_stats(df_p, pi.shape, bin_width_px=8)
                    # 保存统计表
                    stats_t.to_csv(os.path.join(subdir, 'target_stats.csv'), index=False)
                    stats_p.to_csv(os.path.join(subdir, 'pred_stats.csv'), index=False)

                    # 曲线图按期刊规范保存
                    plot_radial_comparison(stats_t, stats_p, subdir, prefix=f"batch{batch_idx}_C{c}",
                                           jconf=journal, save_ext=SAVE_EXT)
                    # plot_distribution_curves(df_t, df_p, subdir, prefix=f"batch{batch_idx}_C{c}",
                    #                          jconf=journal, save_ext=SAVE_EXT)

                vis_counter += 1

            if vis_counter >= max_visualizations:
                break

    return {
        'val_loss': np.mean(val_losses),
        'ssim': np.mean(ssim_scores),
        'psnr': np.mean(psnr_scores),
        'mse': np.mean(mse_scores)
    }

def main(pt_dir, param_dir, opt, batch_size, num_workers, device, k_folds):
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    dataset = PreprocessedPTDataset(pt_dir)
    print(f"Loaded dataset with {len(dataset)} samples.")

    kf = KFold(n_splits=k_folds, shuffle=False)
    all_metrics = []

    for fold, (_, val_idx) in enumerate(kf.split(dataset)):
        print(f"\n=== Evaluating Fold {fold + 1} ===")
        # print(f"Validation indices for fold {fold + 1}: {val_idx}")
        fold_dir = os.path.join(param_dir, f"fold_{fold + 1}")
        # model_path = os.path.join(fold_dir, f"{opt}_best_model_fold{fold + 1}.pth")
        model_path = os.path.join(fold_dir, f"{opt}_PTModel_fold{fold + 1}_epoch500.pth")


        if not os.path.exists(model_path):
            print(f"[WARNING] Model file not found: {model_path}")
            continue

        val_subset = Subset(dataset, val_idx)
        val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

        model = SimpleCNN(opt=opt, drop_path_rate=0.2).to(device)
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)

        metrics = evaluate_model(model, val_loader, device, fold_dir, opt, fold, max_visualizations=100)
        all_metrics.append(metrics)

        print(f"Fold {fold + 1} Metrics:")
        for k, v in metrics.items():
            print(f"  {k}: {v:.6f}")

    print("\n=== Summary ===")
    for k in all_metrics[0].keys():
        values = [m[k] for m in all_metrics]
        print(f"{k}: mean={np.mean(values):.6f}, std={np.std(values):.6f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pt_dir', type=str,
            default=r'D:\SIMULATION\p01_DebrisCloudDamageDataBase\DataBase\PreprocessedPT\thinckness_binary_128')
    parser.add_argument('--opt', type=str, default='EPS', help='Target type: EPS, Ek, damageM')
    parser.add_argument('--param_dir', type=str,
            default=r'D:\SIMULATION\p01_DebrisCloudDamageDataBase\params\thickness_binary_128')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--k_folds', type=int, default=8)
    args = parser.parse_args()

    main(**vars(args))
