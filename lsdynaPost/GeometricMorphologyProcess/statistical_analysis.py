import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from skimage import filters, measure, morphology, segmentation, feature
from skimage.transform import rescale
from skimage.feature import blob_log
from scipy import ndimage as ndi
import pandas as pd
import os


# ===== 新增工具函数 =========================================
def radial_bin_stats(df, img_shape, bin_width_px=1, center=None):
    """
    以圆心为中心做同心圆环，将坑按半径分箱并统计:
      - 斑点数量 (count)
      - 平均坑深 (mean_depth)
      - 平均直径 (mean_diam)

    Parameters
    ----------
    df : DataFrame
        必须含列 'y_px', 'x_px', 'depth_est', 'radius_px'
    img_shape : tuple (H, W)
    bin_width_px : int
        每个圆环宽度（像素）
    center : (cy, cx) or None
        圆心(像素坐标)。None 时取几何中心

    Returns
    -------
    stats : DataFrame
        列: r_mid, count, mean_depth, mean_diam
            r_mid 为该圆环中心半径
    """
    H, W = img_shape
    if center is None:
        center = (H / 2.0, W / 2.0)
    cy, cx = center

    # 求每个坑距圆心的半径
    r = np.hypot(df['x_px'] - cx, df['y_px'] - cy)

    # 分箱
    bin_idx = (r // bin_width_px).astype(int)
    df = df.assign(r=r, bin=bin_idx)

    grp = df.groupby('bin')
    stats = grp.agg(
        r_mid=('r', lambda x: x.mean()),  # 该环中心半径
        count=('r', 'size'),
        mean_depth=('depth_est', 'mean'),
        mean_diam=('radius_px', lambda x: (2 * x).mean())
    ).reset_index(drop=True)

    return stats


def plot_radial_curves(stats):
    """
    将 radial_bin_stats 输出的 DataFrame 画成三条曲线:
      斑点数量 / 平均深度 / 平均直径  随半径变化
    每个指标单独一张图，避免子图配色限制
    """
    # 曲线 1: 斑点数量
    plt.figure()
    plt.plot(stats['r_mid'], stats['count'], marker='o')
    plt.xlabel('Radius (px)')
    plt.ylabel('Pit count')
    plt.title('Pit count vs radius')
    plt.grid(True)

    # 曲线 2: 平均坑深
    plt.figure()
    plt.plot(stats['r_mid'], stats['mean_depth'], marker='o')
    plt.xlabel('Radius (px)')
    plt.ylabel('Mean depth')
    plt.title('Mean pit depth vs radius')
    plt.grid(True)

    # 曲线 3: 平均直径
    plt.figure()
    plt.plot(stats['r_mid'], stats['mean_diam'], marker='o')
    plt.xlabel('Radius (px)')
    plt.ylabel('Mean diameter (px)')
    plt.title('Mean pit diameter vs radius')
    plt.grid(True)

    plt.show()


def compute_radius_99(df, img_shape, verbose=True):
    """
    根据坑信息 DataFrame 计算：若以图像中心为圆心，
    圆半径达到多少像素即可覆盖 99% 的损伤面积 (area_px2)。

    Parameters
    ----------
    df : DataFrame
        必须包含列 'y_px', 'x_px', 'area_px2'
    img_shape : tuple
        (H, W)，用于确定圆心
    Returns
    -------
    r_99 : float
        覆盖 99% area 的半径（像素）
    """
    H, W = img_shape
    cy, cx = H / 2.0, W / 2.0

    # 距离圆心的半径
    r = np.hypot(df['x_px'] - cx, df['y_px'] - cy)

    # 按半径排序并累计面积
    df_sorted = df.assign(r=r).sort_values('r')
    cum_area = df_sorted['area_px2'].cumsum()
    total_area = cum_area.iloc[-1]

    # 找到 >= 99% 总面积的位置
    idx_99 = np.searchsorted(cum_area, 0.99 * total_area)
    r_99 = df_sorted.iloc[idx_99]['r']

    if verbose:
        print(f"[INFO] 99% 的坑面积位于半径 ≤ {r_99:.2f} 像素内")
    return r_99

def radius_for_damage(mask, center=None, coverage=0.99, pixel_size=0.20):
    """
    计算以 center 为圆心，覆盖 mask 中 coverage 比例像素所需最小半径.
    mask: 2-D bool / 0-1 数组，True 代表损伤像素
    center: (y0, x0)，默认取图像几何中心
    coverage: 希望覆盖比例（0–1）
    pixel_size: 单像素对应的物理尺寸。=1 时返回像素半径
    """
    H, W = mask.shape
    if center is None:
        center = (H / 2.0, W / 2.0)

    y, x = np.indices(mask.shape)
    r = np.hypot(x - center[1], y - center[0])

    # 仅统计损伤像素
    r_vals = r[mask]
    if r_vals.size == 0:
        return 0.0      # 没有损伤

    r_sorted = np.sort(r_vals)
    cutoff_idx = int(np.ceil(coverage * r_sorted.size)) - 1
    cutoff_idx = np.clip(cutoff_idx, 0, r_sorted.size - 1)
    radius_px = r_sorted[cutoff_idx]        # 像素半径
    return radius_px * pixel_size

def visualize_one_topbottom(np_file, up_factor=4):
    # ======= 0. 读数据 =======
    data = np.load(np_file, allow_pickle=True).item()
    Z = (data['bottom_coords'][:,2]-data['top_coords'][:,2]).reshape((500,500))

    # ======= 1. 背景扣除 =======
    background = gaussian_filter(Z, sigma=100)
    Z_res = Z - background          # 小坑为负
    # Z_res = -Z_res                  # 翻转成 “坑深 > 0”

    # ======= 2. 上采样 (亚像素插值) =======
    Z_hr = rescale(Z_res, up_factor, order=3, mode='reflect',
                   anti_aliasing=True, preserve_range=True)
    print(Z_hr.max())

    # ======= 3. LoG 圆斑检测 =======
    # σ 定义在放大后的像素坐标里
    min_sigma  = 3 * up_factor / 1.414
    max_sigma  = 30 * up_factor / 1.414
    blobs = blob_log(Z_hr, min_sigma=min_sigma, max_sigma=max_sigma,
                     num_sigma=30, threshold=0.005*Z_hr.max(), overlap=0.5)

    # blob_log 返回 (y, x, σ)
    # 把坐标映射回原图 (除以 up_factor)
    blobs[:, :2] /= up_factor
    blobs[:, 2]  /= up_factor

    # ======= 4. 可视化斑点 =======
    fig, ax = plt.subplots(1, 2, figsize=(10,5))
    ax[0].imshow(Z_res, cmap='coolwarm'); ax[0].set_title('Residual depth')
    ax[1].imshow(Z_res, cmap='gray');    ax[1].set_title('LoG blobs')
    for y, x, r in blobs:
        c = plt.Circle((x, y), r, color='red', linewidth=1, fill=False)
        ax[1].add_patch(c)
    for a in ax: a.axis('off')
    plt.tight_layout(); plt.show()

    # ---- 5. 生成 DataFrame 统计 (修正版) --------------------------
    rows = np.clip(np.rint(blobs[:, 0]).astype(int), 0, Z_res.shape[0] - 1)
    cols = np.clip(np.rint(blobs[:, 1]).astype(int), 0, Z_res.shape[1] - 1)

    pit_info = {
        'y_px': blobs[:, 0],
        'x_px': blobs[:, 1],
        'radius_px': blobs[:, 2],
        'depth_est': Z_res[rows, cols],  # 直接一次性索引
    }

    df = pd.DataFrame(pit_info)
    df['area_px2'] = np.pi * df['radius_px']**2
    df['volume_proxy'] = df['area_px2'] * df['depth_est']   # 简单体积指标
    print(df.head(), '\n识别小坑数量:', len(df))

    # ---- 5.1 计算 99% 面积对应半径 -------------------------------
    r_99 = compute_radius_99(df, Z.shape)  # Z 是原始 2D 深度图
    print(r_99*0.2*2)

    # ---- 6.2 半径分箱统计并绘图 -------------------------------
    print(Z.shape)
    stats = radial_bin_stats(df, Z.shape, bin_width_px=10)
    print(stats.head())          # 如需查看表格
    plot_radial_curves(stats)



def batch_visualize_topbottom(np_file_dir, np_file_suffix):
    """
    批量可视化 np_file_dir 目录下所有以指定后缀结尾的 topbottom_R*.npy 文件
    """
    if not os.path.isdir(np_file_dir):
        print(f"[Error] {np_file_dir} is not a valid directory.")
        return

    files = sorted([
        f for f in os.listdir(np_file_dir)
        if f.endswith(np_file_suffix)
    ])

    if len(files) == 0:
        print(f"[Warning] No matching '*{np_file_suffix}' files found.")
        return

    print(f"[Info] Found {len(files)} files for visualization.\n")

    for fname in files:
        full_path = os.path.join(np_file_dir, fname)
        print(full_path)
        visualize_one_topbottom(full_path)


if __name__ == "__main__":
    # ========== 示例路径，请按需修改 ==========
    np_file_dir = r"D:\SIMULATION\p01_DebrisCloudDamageDataBase\originalData\eval\original_elementValue_npyFiles\damageM_Data\NP"
    np_file_suffix = "_damageM.npy"

    batch_visualize_topbottom(np_file_dir, np_file_suffix)