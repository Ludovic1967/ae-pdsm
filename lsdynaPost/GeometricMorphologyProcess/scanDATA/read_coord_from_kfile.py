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
import os, torch
from scipy.ndimage import gaussian_filter
from skimage.transform import rescale
from skimage.measure import label, regionprops
from matplotlib import cm

def find_mbr_angle(xy: np.ndarray) -> float:
    """
    给定 (N,2) 点云，返回最小面积外接矩形的旋转角度 θ_star (弧度) ，
    使得旋转 -θ_star 后矩形边平行 X/Y 轴。
    """
    hull = ConvexHull(xy)
    pts = xy[hull.vertices]

    # 凸包每条边方向
    n = len(pts)
    best_area, best_angle = np.inf, 0.0
    for i in range(n):
        p, q = pts[i], pts[(i+1) % n]
        edge = q - p
        angle = np.arctan2(edge[1], edge[0])            # (-π, π]
        angle = (angle + np.pi/2) % (np.pi/2)           # 归一化到 [0, π/2)

        # 旋转 -angle，把该边对齐 X 轴
        c, s = np.cos(-angle), np.sin(-angle)
        R = np.array([[c, -s],
                      [s,  c]])
        rot = xy @ R.T
        w = rot[:,0].max() - rot[:,0].min()
        h = rot[:,1].max() - rot[:,1].min()
        area = w * h
        if area < best_area:
            best_area, best_angle = area, angle
            best_w, best_h = w, h

    # 可选: 总想让“长边” → +X
    if best_h > best_w:        # 垂直方向更长 ⇒ 再转 90°
        best_angle += np.pi/2
    return best_angle

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

# def compute_thickness(ids: np.ndarray,
#                       coords: np.ndarray,
#                       radius: float = 75.0) -> np.ndarray:
#     """
#     Compute local thickness along the global normal for a plate mesh,
#     but only for nodes within `radius` mm of the centroid.
#
#     Parameters
#     ----------
#     ids : np.ndarray, shape (N,)
#         Node identifiers.
#     coords : np.ndarray, shape (N, 3)
#         XYZ coordinates of nodes.
#     radius : float
#         Distance threshold (in same units as coords) from centroid;
#         only nodes within this radius are processed.
#
#     Returns
#     -------
#     thickness_all : np.ndarray, shape (N,)
#         Thickness at each node (NaN outside the region or for bottom-surface nodes).
#     """
#     # 1. 计算重心
#     centroid = coords.mean(axis=0)
#
#     # 2. 计算每个节点到重心的径向距离，并筛选出内部节点
#     radial_dist = np.linalg.norm(coords - centroid, axis=1)
#     within_mask = radial_dist <= radius
#
#     # 3. PCA 得到全局法向
#     X = coords - centroid
#     cov = X.T @ X
#     eigvals, eigvecs = np.linalg.eigh(cov)
#     normal = eigvecs[:, np.argmin(eigvals)]
#     normal /= np.linalg.norm(normal)
#
#     # 4. 构造旋转矩阵，将法向对齐到 Z 轴
#     target = np.array([0.0, 0.0, 1.0])
#     v = np.cross(normal, target)
#     s = np.linalg.norm(v)
#     c = np.dot(normal, target)
#     if s < 1e-8:
#         R = np.eye(3)
#     else:
#         vx = np.array([[0, -v[2], v[1]],
#                        [v[2], 0, -v[0]],
#                        [-v[1], v[0], 0]])
#         R = np.eye(3) + vx + vx @ vx * ((1 - c) / (s**2))
#
#     # 5. 旋转到对齐坐标系
#     aligned = (R @ X.T).T
#     z = aligned[:, 2]
#
#     # 6. 在内部节点上计算中位面，把内部的上半面视为 top
#     median_z = np.median(z[within_mask])
#     global_top = z > median_z
#     top_mask = global_top & within_mask
#     bottom_mask = (~global_top) & within_mask
#
#     # 7. 最近邻匹配（仅用内部 bottom 节点）
#     top_pts = aligned[top_mask]
#     bottom_pts = aligned[bottom_mask]
#     tree = cKDTree(bottom_pts[:, :2])
#     _, idx = tree.query(top_pts[:, :2])
#
#     # 8. 计算厚度
#     thickness = top_pts[:, 2] - bottom_pts[idx, 2]
#
#     # 9. 映射回所有节点
#     thickness_all = np.full(ids.shape, np.nan, dtype=np.float32)
#     thickness_all[top_mask] = thickness
#
#     return thickness_all

# def visualize_thickness_grid(ids: np.ndarray, coords: np.ndarray, thickness: np.ndarray, grid_size: int = 300):
#     """
#     Project thickness values along the global normal into a 2D grid and visualize.
#
#     Parameters
#     ----------
#     ids : np.ndarray, shape (N,)
#         Node identifiers (not used in visualization).
#     coords : np.ndarray, shape (N, 3)
#         Original XYZ coordinates of nodes.
#     thickness : np.ndarray, shape (N,)
#         Thickness values per node (NaN for bottom-surface nodes).
#     grid_size : int
#         Number of grid points along each axis for interpolation.
#     """
#     # Recompute global alignment
#     centroid = coords.mean(axis=0)
#     X = coords - centroid
#     cov = X.T @ X
#     eigvals, eigvecs = np.linalg.eigh(cov)
#     normal = eigvecs[:, np.argmin(eigvals)]
#     normal /= np.linalg.norm(normal)
#     # Build rotation matrix (Rodrigues)
#     target = np.array([0.0, 0.0, 1.0])
#     v = np.cross(normal, target)
#     s = np.linalg.norm(v)
#     c = np.dot(normal, target)
#     if s < 1e-8:
#         R = np.eye(3)
#     else:
#         vx = np.array([[0, -v[2], v[1]],
#                        [v[2], 0, -v[0]],
#                        [-v[1], v[0], 0]])
#         R = np.eye(3) + vx + vx @ vx * ((1 - c) / (s**2))
#     aligned = (R @ X.T).T
#
#     # Select top-surface points
#     mask_top = ~np.isnan(thickness)
#     pts2d = aligned[mask_top, :2]
#     thick_vals = thickness[mask_top]
#
#     # Define grid
#     xi = np.linspace(pts2d[:,0].min(), pts2d[:,0].max(), grid_size)
#     yi = np.linspace(pts2d[:,1].min(), pts2d[:,1].max(), grid_size)
#     XI, YI = np.meshgrid(xi, yi)
#
#     # Interpolate thickness onto grid
#     TI = griddata(pts2d, thick_vals, (XI, YI), method='linear')
#
#     # Visualization
#     plt.figure()
#     plt.imshow(TI, extent=(xi.min(), xi.max(), yi.min(), yi.max()),
#                origin='lower', aspect='auto')
#     plt.xlabel('Aligned X')
#     plt.ylabel('Aligned Y')
#     plt.title('Plate Thickness Projection')
#     plt.colorbar(label='Thickness')
#     plt.show()
#
#     return TI

def load_k_file(path, chunksize=200_000):
    ids_list, coords_list = [], []
    with open(path, 'r') as f:
        eof = False
        while not eof:
            buf = []
            for _ in range(chunksize):
                line = f.readline()
                if not line:      # 读到文件尾
                    eof = True
                    break
                s = line.lstrip()
                if not s or s[0] in ('*', '$'):
                    continue       # 跳过注释
                buf.append(line)
            if buf:
                chunk_txt = ''.join(buf)
                df = pd.read_csv(
                    StringIO(chunk_txt),
                    delim_whitespace=True,
                    header=None,
                    names=['id','x','y','z'],
                    dtype={'id':np.int32,'x':np.float32,'y':np.float32,'z':np.float32},
                    engine='c'
                )
                ids_list.append(df['id'].values)
                coords_list.append(df[['x','y','z']].values)

    ids    = np.concatenate(ids_list,   axis=0)
    coords = np.vstack(coords_list)     # shape=(N,3)
    return ids, coords

def radial_bin_stats(df, img_shape, bin_width_px=1, center=None):
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
    bin_idx = (r // bin_width_px).astype(int)
    df2 = df.assign(r=r, bin=bin_idx)

    grp = df2.groupby('bin')
    stats = grp.agg(
        r_mid      = ('r',        'mean'),
        count      = ('r',        'size'),
        sum_depth  = ('depth_est','sum'),
        mean_depth = ('depth_est','mean'),
        sum_diam   = ('radius_px', lambda x: (2 * x).sum()),
        mean_diam  = ('radius_px', lambda x: (2 * x).mean()),
        sum_volume = ('volume_px3','sum'),
        mean_volume= ('volume_px3','mean'),
        sum_area   = ('area_px2',  'sum'),
        mean_area  = ('area_px2',  'mean'),
    ).reset_index()

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


def radial_cumulative_stats(df, img_shape, bin_width_px=1, center=None):
    """
    使用同心圆进行累积统计并按圆面积归一化:
      - r_max           : 当前圆的最大半径
      - count           : 坑总数（从中心到当前圆）
      - count_density   : 坑密度 (#/px^2)
      - mean_depth      : 平均坑深（所有坑的平均值）
      - depth_density   : 坑深累积密度 (sum_depth/圆面积)
      - mean_diam       : 平均坑直径
      - diam_density    : 直径累积密度 (sum_diam/圆面积)
      - mean_volume     : 平均体积代理
      - volume_density  : 体积代理密度 (sum_volume/圆面积)
    df: 包含 ['y_px','x_px','depth_est','radius_px','volume_px3']
    """
    import numpy as np
    import pandas as pd

    # 图像中心
    H, W = img_shape[-2:]
    if center is None:
        center = (H / 2.0, W / 2.0)
    cx, cy = center
    print(cx, cy)

    # 计算每个坑到中心的距离
    r = np.hypot(df['x_px'] - cx, df['y_px'] - cy)
    df = df.assign(r=r)

    # 最大距离决定分箱数量
    r_max = r.max()
    max_bin = int(np.ceil(r_max / bin_width_px))

    # 初始化累积列表
    cumulative_stats = []

    for k in range(1, max_bin + 1):
        r_thresh = k * bin_width_px
        mask = df['r'] <= r_thresh
        sub_df = df[mask]

        if len(sub_df) == 0:
            continue

        area = np.pi * r_thresh**2
        count = len(sub_df)
        sum_depth = sub_df['depth_est'].sum()
        sum_diam = (2 * sub_df['radius_px']).sum()
        sum_volume = sub_df['volume_px3'].sum()

        cumulative_stats.append({
            'bin': k,
            'r_max': r_thresh,
            'count': count,
            'count_density': count / area,
            'mean_depth': sub_df['depth_est'].mean(),
            'depth_density': sum_depth / area,
            'mean_diam': (2 * sub_df['radius_px']).mean(),
            'diam_density': sum_diam / area,
            'mean_volume': sub_df['volume_px3'].mean(),
            'volume_density': sum_volume / area,
            'circle_area_px2': area
        })

    return pd.DataFrame(cumulative_stats)


def angular_bin_stats(df, img_shape, bin_width_rad=np.pi/18, center=None):
    """
    按角度分段统计（弧度制）:
      - angle_mid      : 分组中点角度 (rad)
      - count          : 坑总数
      - count_density  : 坑密度 (#/rad)
      - mean_depth     : 平均坑深
      - depth_density  : 坑深累积密度 (sum_depth/弧度宽度)
      - mean_diam      : 平均坑直径
      - diam_density   : 直径累积密度 (sum_diam/弧度宽度)
      - mean_volume    : 平均体积代理
      - volume_density : 体积代理密度 (sum_volume/弧度宽度)
    df: 包含 ['y_px','x_px','depth_est','radius_px','volume_px3']
    img_shape: (H, W) 或 (C, H, W)
    """
    H, W = img_shape[-2:]
    if center is None:
        center = (H/2.0, W/2.0)
    cx, cy = center

    # 1. 计算角度（-π 到 π），再映射到 [0, 2π)
    dx = df['x_px'] - cx
    dy = df['y_px'] - cy
    angles = np.arctan2(dy, dx)
    angles = (angles + 2*np.pi) % (2*np.pi)

    # 2. 分箱
    bin_idx = (angles // bin_width_rad).astype(int)
    df2 = df.assign(angle=angles, bin=bin_idx)

    # 3. 聚合统计
    grp = df2.groupby('bin')
    stats = grp.agg(
        angle_mid   = ('angle',    'mean'),
        count       = ('angle',    'size'),
        sum_depth   = ('depth_est','sum'),
        mean_depth  = ('depth_est','mean'),
        sum_diam    = ('radius_px', lambda x: (2*x).sum()),
        mean_diam   = ('radius_px', lambda x: (2*x).mean()),
        sum_volume  = ('volume_px3','sum'),
        mean_volume = ('volume_px3','mean'),
    ).reset_index()

    # 4. 归一化为密度 (# per radian)
    stats['angle_width_rad']  = bin_width_rad
    stats['count_density']    = stats['count']      / stats['angle_width_rad']
    stats['depth_density']    = stats['sum_depth']  / stats['angle_width_rad']
    stats['diam_density']     = stats['sum_diam']   / stats['angle_width_rad']
    stats['volume_density']   = stats['sum_volume'] / stats['angle_width_rad']

    return stats


def plot_radial_comparison(stats_t, out_dir, prefix='', pixel2mm=0.2):
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
        ('mean_area', 'Mean Area'),
        ('area_density', 'Area Density'),
    ]

    for key, ylabel in metrics:
        if key not in stats_t:
            continue  # 跳过不存在的字段
        plt.figure()
        radius = stats_t['r_mid'] * pixel2mm  # 半径转换为毫米
        plt.plot(radius, stats_t[key], 'o-', linewidth=2, label='target')
        plt.xlabel('Radius (mm)')
        plt.ylabel(ylabel)
        plt.title(f'{prefix} {ylabel} vs Radius')
        plt.legend()
        plt.grid(True)
        fname = os.path.join(out_dir, f"{prefix}_radial_{key}_center.png")
        plt.tight_layout()
        plt.savefig(fname, dpi=300)
        plt.close()


def plot_radial_cum_comparison(stats_t, out_dir, prefix='', pixel2mm=0.2):
    """
    比较 target 与 pred 的 radial 统计并保存曲线图
    metrics 包含：
      - count, count_density,
      - mean_depth, depth_density,
      - mean_diam, diam_density,
      - mean_volume, volume_density
    """
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
    ]

    for key, ylabel in metrics:
        plt.figure()
        # 横轴：半径转换为物理单位
        radius = stats_t['r_max'] * pixel2mm
        plt.plot(radius, stats_t[key], 'o-', linewidth=5, label='target')
        # plt.plot(radius, stats_p[key], 's--', label='pred')
        plt.xlabel('Radius (mm)')
        plt.ylabel(ylabel)
        plt.title(f'{prefix} {ylabel} vs Radius')
        plt.legend()
        plt.grid(True)
        fname = os.path.join(out_dir, f"{prefix}_radial_{key}_cum_center.png")
        plt.tight_layout()
        plt.savefig(fname, dpi=300)
        plt.close()

def plot_angular_comparison(stats_t, out_dir, prefix=''):
    """
    比较 target 与 pred 的 angular 统计并保存曲线图
    metrics 包含：
      - count, count_density,
      - mean_depth, depth_density,
      - mean_diam, diam_density,
      - mean_volume, volume_density
    x 轴：angle_mid (rad)
    """
    os.makedirs(out_dir, exist_ok=True)
    metrics = [
        ('count', 'Pit Count'),
        ('count_density', 'Pit Count Density'),
        ('mean_depth', 'Mean Depth'),
        ('depth_density', 'Depth Density'),
        ('mean_diam', 'Mean Diameter'),
        ('diam_density', 'Diameter Density'),
        ('mean_volume', 'Mean Volume'),
        ('volume_density', 'Volume Density'),
    ]
    for key, ylabel in metrics:
        plt.figure()
        angles = stats_t['angle_mid']
        plt.plot(angles, stats_t[key], 'o-', label='target')
        # plt.plot(angles, stats_p[key], 's--', label='pred')
        plt.xlabel('Angle (rad)')
        plt.ylabel(ylabel)
        plt.title(f'{prefix} {ylabel} vs Angle')
        plt.legend()
        plt.grid(True)
        fname = os.path.join(out_dir, f"{prefix}_angular_{key}.png")
        plt.tight_layout()
        plt.savefig(fname, dpi=300)
        plt.close()

def plot_value_count_distribution(df, out_dir, prefix='', bins=20, pixel_size=1.0):
    """
    绘制并保存坑的直径、深度和体积值 vs 数量分布曲线：
      - 横轴：将数值范围等分为 bins 段，取每段中点
      - 纵轴：该区间内的坑数量
    参数：
      df       : 包含字段 ['radius_px','depth_est','volume_px3'] 的 DataFrame
      out_dir  : 保存图像的目录
      prefix   : 文件名前缀
      bins     : 分段数，默认为 20
      pixel_size : 像素到物理单位的换算 (如 mm/px)，默认 1.0 可保留像素单位
    """
    os.makedirs(out_dir, exist_ok=True)

    # 定义要绘制的三种分布：(系列数据, 键名, 横轴标签)
    dist_specs = [
        (2 * df['radius_px'],  'diameter', 'Pit Diameter'),
        (    df['depth_est'],  'depth',    'Pit Depth'),
        (    df['volume_px3'],  'volume',   'Pit Volume Proxy'),
    ]

    for data, key, xlabel in dist_specs:
        # 1. 计算分段边界
        min_val, max_val = data.min(), data.max()
        if min_val == max_val:
            # 全部相同，跳过绘图
            print(f"[WARN] All values identical for {key}, skipping.")
            continue

        edges = np.linspace(min_val, max_val, bins + 1)
        counts, _ = np.histogram(data, bins=edges)

        # 2. 计算每个区间的中点
        centers = (edges[:-1] + edges[1:]) / 2

        # 3. 绘图
        plt.figure()
        plt.plot(centers, counts, 'o-', label=f'{key} distribution')
        plt.xlabel(f'{xlabel}')
        plt.ylabel('Pit count')
        plt.title(f'{prefix} {xlabel} Distribution')
        plt.grid(True)
        plt.tight_layout()

        # 4. 保存
        fname = os.path.join(out_dir, f"{prefix}_dist_{key}.png")
        plt.savefig(fname, dpi=300)
        plt.close()
        print(f"[Saved] {fname}")

def extract_pits(img, bg_factor, gf_sigma, pixel2mm):
    """
    提取图像中的坑信息，返回包含过滤后斑点信息的 DataFrame。
    """
    # 如果输入为 torch.Tensor，先转到 CPU numpy
    if isinstance(img, torch.Tensor):
        img = img.detach().cpu().numpy().squeeze()

    Z = img
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
        depth_est = region_max_value  # 可以使用最大值作为深度估计，或者根据实际情况调整
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

    print("过滤后的斑点统计结果：")
    print(filtered_spots_df)

    r_99 = compute_radius_99(filtered_spots_df, Z.shape)
    print(f"99% 的损伤面积覆盖半径为: {r_99*0.2*2:.2f} mm")

    # ======= 6. 可视化原始图像与二值化结果 =======
    # 3. 获取 coolwarm colormap 的一半（这里取上半部分）
    full_cmap = cm.get_cmap('coolwarm', 256)  # 原 colormap
    half_cmap = full_cmap(np.linspace(0, 0.5, 128))  # 取后半部分（偏暖色）
    half_cmap = cm.colors.ListedColormap(half_cmap)  # 转成新的 cmap



    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(-Z, cmap='coolwarm');
    ax[0].set_title('Residual Depth')
    ax[1].imshow(Z_binary, cmap=half_cmap);
    ax[1].set_title('Binary Image (Thresholding)')

    # scatter_yx = [[466, 475]]
    # # 添加散点
    # if scatter_yx:
    #     scatter_yx = np.array(scatter_yx)
    #     ax[1].scatter(scatter_yx[:, 0], scatter_yx[:, 1], s=5, c='lime', marker='o', label='Spot Centroids')
    #     ax[1].legend(loc='upper right', fontsize=6)

    # # 在二值化图像上标出斑点（只标出过滤后的斑点）
    # for region in regions:
    #     # print(f"Region bbox: {region.bbox}")  # 输出 bbox，检查其返回的内容
    #     area = region.area
    #     if area >= area_threshold:
    #         minr, minc, maxr, maxc = region.bbox
    #         ax[0].add_patch(
    #             plt.Rectangle((minc, minr), maxc - minc, maxr - minr, edgecolor='red', facecolor='none', linewidth=0.5))

    # for a in ax: a.axis('off')
    plt.tight_layout();
    plt.show()

    return filtered_spots_df
def compute_contour_depth(ids: np.ndarray,
                          coords: np.ndarray,
                          radius: float = 180.0,
                          return_transform: bool = False):
    """
    扩展版:
    1. 先利用 PCA 估算面的法向 normal，并将其旋转到全局 Z 轴。
    2. 在旋转后坐标系下，用外接长方体(bounding-box)确定模型中心，
       再将模型平移使其中心位于原点。
    3. 以 Z 轴为旋转轴，做一次 2D PCA，使板的两条主边分别与全局 X/Y 轴对齐。
    4. 返回深度(depth_all)，可选整体 4×4 变换矩阵 (centroid→对齐→平移→面内旋转)。

    参数
    ----
    ids      : 节点编号数组，与 coords 一一对应
    coords   : (N,3) 节点坐标
    radius   : 只在半径 radius 内计算深度，超出部分置 NaN
    return_transform : 若为 True，额外返回整体齐次变换矩阵 T (4×4)

    返回
    ----
    depth_all           : (N,) 节点深度 (aligned Z)
    可选 T (4×4)        : 将原始 coords 变换到最终坐标系的矩阵
    """
    # ---------- 基本准备 ----------
    centroid = coords.mean(axis=0)                            # 质心
    X0 = coords - centroid                                    # 去质心坐标
    radial_dist = np.linalg.norm(X0, axis=1)
    within_mask = radial_dist <= radius

    # ---------- 1) 法向 → Z ----------
    cov3 = X0.T @ X0
    eigvals3, eigvecs3 = np.linalg.eigh(cov3)
    normal = eigvecs3[:, np.argmin(eigvals3)]                 # 法向量
    normal /= np.linalg.norm(normal)

    z_axis = np.array([0., 0., 1.])
    v = np.cross(normal, z_axis)
    s = np.linalg.norm(v)
    c = float(np.dot(normal, z_axis))
    if s < 1e-8:                                              # 已对齐
        R1 = np.eye(3)
    else:                                                     # Rodrigues
        vx = np.array([[    0, -v[2],  v[1]],
                       [ v[2],     0, -v[0]],
                       [-v[1],  v[0],     0]])
        R1 = np.eye(3) + vx + vx @ vx * ((1 - c) / s**2)

    aligned = (R1 @ X0.T).T                                   # 坐标已使 normal//Z

    # ---------- 2) 平移中心至原点 ----------
    # 只需关注 XY 平面外框
    xy_min = aligned[:, :2].min(axis=0)
    xy_max = aligned[:, :2].max(axis=0)
    center_xy = (xy_min + xy_max) * 0.5
    T_shift = np.eye(4)
    T_shift[:3, 3] = np.array([-center_xy[0], -center_xy[1], 0.])  # 仅 XY 平移

    aligned_center = aligned.copy()
    aligned_center[:, 0] -= center_xy[0]
    aligned_center[:, 1] -= center_xy[1]

    # ---------- 3) 面内旋转，使主边 ∥ X/Y ----------
    angle = find_mbr_angle(aligned[:, :2])  # ← 新函数

    cos_a, sin_a = np.cos(-angle), np.sin(-angle)
    Rz = np.array([[cos_a, -sin_a, 0.],
                   [sin_a,  cos_a, 0.],
                   [   0.,     0., 1.]])

    aligned_final = (Rz @ aligned_center.T).T

    # ---------- 深度 ----------
    z_final = aligned_final[:, 2]
    depth_all = np.full(ids.shape, np.nan, dtype=np.float32)
    depth_all[within_mask] = z_final[within_mask]

    # ---------- 返回 ----------
    if return_transform:
        # 将所有变换拼成齐次矩阵:  T = (平移质心) → R1 → shift → Rz
        T_centroid = np.eye(4)
        T_centroid[:3, 3] = -centroid
        T_R1 = np.eye(4)
        T_R1[:3, :3] = R1
        T_Rz = np.eye(4)
        T_Rz[:3, :3] = Rz

        T = T_Rz @ T_shift @ T_R1 @ T_centroid
        return depth_all, T
    return depth_all

BBox = Tuple[Tuple[float, float], Tuple[float, float]]
def visualize_depth_grid(coords_aligned: np.ndarray,
                         depth         : np.ndarray,
                         grid_size     : int = 500,
                         *,
                         bbox: Optional[BBox] = None,
                         save_npy: Optional[str] = None,
                         save_txt: Optional[str] = None):
    """
    根据已经对齐的坐标+深度绘制 2D 网格图，并截取到 bbox。
    -------------------------------------------------------------------
    参数
    ----
    coords_aligned : (N,3)  已完成对齐的坐标
    depth          : (N,)   深度；NaN 表示无效
    grid_size      : int    网格分辨率 (默认 1000)
    bbox           : ((xmin,xmax),(ymin,ymax))
                     若为 None ⇒ 自动用点云最小外接矩形
    save_npy       : 路径；若给出则 np.save
    save_txt       : 路径；若给出则 np.savetxt
    -------------------------------------------------------------------
    返回
    ----
    ZI             : (grid_size, grid_size)  截取后二维深度阵列
    """
    # ---------- 1) 有效点 ----------
    print(depth.shape)
    mask   = ~np.isnan(depth)
    pts2d  = coords_aligned[mask, :2]
    vals   = depth[mask]

    print("有效深度点数:", np.sum(~np.isnan(depth)))
    print("bbox =", bbox)  # 如果你传了的话
    print(vals.shape)
    print("pts2d.shape =", pts2d.shape)  # 留意是不是 0
    print("pts2d x范围:", pts2d[:, 0].min(), pts2d[:, 0].max())
    print("pts2d y范围:", pts2d[:, 1].min(), pts2d[:, 1].max())

    # ---------- 2) 确定边界 bbox ----------
    if bbox is None:
        xmin, xmax = pts2d[:, 0].min(), pts2d[:, 0].max()
        ymin, ymax = pts2d[:, 1].min(), pts2d[:, 1].max()
    else:
        (xmin, xmax), (ymin, ymax) = bbox
        # 可选：把 bbox 之外的点去掉，提高插值效率
        keep = ((pts2d[:, 0] >= xmin) & (pts2d[:, 0] <= xmax) &
                (pts2d[:, 1] >= ymin) & (pts2d[:, 1] <= ymax))
        pts2d, vals = pts2d[keep], vals[keep]

    # ---------- 3) 网格 ----------
    xi = np.linspace(xmin, xmax, grid_size)
    yi = np.linspace(ymin, ymax, grid_size)
    XI, YI = np.meshgrid(xi, yi)
    ZI = griddata(pts2d, vals, (XI, YI), method='linear')
    n_valid = np.count_nonzero(~np.isnan(ZI))
    print("网格有效点数 =", n_valid)

    print("ZI 统计：",
          "min =", np.nanmin(ZI),
          "max =", np.nanmax(ZI))

    # 若想把 NaN 区域填最近邻，避免花斑
    if np.isnan(ZI).any():
        ZI_near = griddata(pts2d, vals, (XI, YI), method='nearest')
        ZI = np.where(np.isnan(ZI), ZI_near, ZI)

    print(ZI.shape)

    # ---------- 4) 可视化 ----------
    base_cmap = plt.get_cmap('Blues')
    colors = base_cmap(np.linspace(0, 1, 30))
    discrete_cmap = ListedColormap(colors, name='coolwarm_discrete')

    # plt.figure()
    # plt.imshow(-ZI, extent=(xmin, xmax, ymin, ymax),
    #            origin='lower', aspect='auto', cmap='coolwarm')
    # plt.xlabel('Aligned X')
    # plt.ylabel('Aligned Y')
    # plt.title('Contour Depth Projection')
    # plt.colorbar(label='Depth')
    # plt.show()

    # ---------- 5) 保存 ----------
    if save_npy is not None:
        os.makedirs(os.path.dirname(save_npy), exist_ok=True)
        np.save(save_npy, ZI)

    if save_txt is not None:
        os.makedirs(os.path.dirname(save_txt), exist_ok=True)
        np.savetxt(save_txt, ZI, fmt='%.6f')

    return ZI

###########################################################################
gridSize = 1000
H = 160  # 物理尺寸
save_dir = r'D:\PHDstudent\博士论文\损伤代理模型\01 小论文\实验数据\板扫描模型\out\analysis'

# # ############################---===【dwh3】===---############################
# PREFIX = 'dwh3'
# ids, coords = load_k_file(r'D:\PHDstudent\博士论文\损伤代理模型\01 小论文\实验数据\板扫描模型\dwh-3-大板\dwh-3-front.k')
#
# depth, T = compute_contour_depth(ids, coords, return_transform=True)
# coords_h = np.c_[coords, np.ones(len(coords))]      # (N,4) 齐次
# coords_aligned = (T @ coords_h.T).T[:, :3]          # 全量变换 → (N,3)
#
# # 可视化时直接用 coords_aligned
# depth = visualize_depth_grid(coords_aligned, depth, bbox=((-80, 80), (-80, 80)), grid_size=gridSize,
#     save_npy=r'D:\PHDstudent\博士论文\损伤代理模型\01 小论文\实验数据\板扫描模型\out\dwh3_depth_160x160.npy')         # 函数签名略改
# df = extract_pits(depth, bg_factor=0.41, gf_sigma=5, pixel2mm=H/gridSize)  # dwh3
#
# df_path = os.path.join(save_dir, f'{PREFIX}_pits.csv')   # 也可以改成 .pkl
# df.to_csv(df_path, index=False, encoding='utf-8-sig')
# print(f'[INFO] 已将坑信息保存到 {df_path}')
#
# plot_value_count_distribution(df, prefix=PREFIX, bins=80, out_dir=save_dir)
# state_radial = radial_bin_stats(df, depth.shape, bin_width_px=31, center=[566, 451])
# state_path = os.path.join(save_dir, f'{PREFIX}_pits_state_radial.csv')   # 也可以改成 .pkl
# state_radial.to_csv(state_path, index=False, encoding='utf-8-sig')
# plot_radial_comparison(state_radial, pixel2mm=H/gridSize, prefix=PREFIX, out_dir=save_dir)
#
# state_angular = angular_bin_stats(df, depth.shape, bin_width_rad=np.pi/18)
# plot_angular_comparison(state_angular, prefix=PREFIX, out_dir=save_dir)

# ############################---===【dwh2】===---############################
# PREFIX = 'dwh2'
# ids, coords = load_k_file(r'D:\PHDstudent\博士论文\损伤代理模型\01 小论文\实验数据\板扫描模型\dwh-2-小板\dwh-2-front.k') #-
# depth, T = compute_contour_depth(ids, coords, return_transform=True)
# coords_h = np.c_[coords, np.ones(len(coords))]      # (N,4) 齐次
# coords_aligned = (T @ coords_h.T).T[:, :3]          # 全量变换 → (N,3)
#
# # 可视化时直接用 coords_aligned
# depth = -visualize_depth_grid(coords_aligned, depth, bbox=((-80, 80), (-80, 80)), grid_size=gridSize,
#     save_npy=r'D:\PHDstudent\博士论文\损伤代理模型\01 小论文\实验数据\板扫描模型\out\dwh2_depth_160x160.npy')         # 函数签名略改
#
# df = extract_pits(depth, bg_factor=0.535, gf_sigma=5, pixel2mm=H/gridSize)  # dwh2
#
# df_path = os.path.join(save_dir, f'{PREFIX}_pits.csv')   # 也可以改成 .pkl
# df.to_csv(df_path, index=False, encoding='utf-8-sig')
# print(f'[INFO] 已将坑信息保存到 {df_path}')
#
# plot_value_count_distribution(df, prefix=PREFIX, bins=80, out_dir=save_dir)
#
# state_radial = radial_bin_stats(df, depth.shape, bin_width_px=31, center=[430, 450])
# state_path = os.path.join(save_dir, f'{PREFIX}_pits_state_radial.csv')   # 也可以改成 .pkl
# state_radial.to_csv(state_path, index=False, encoding='utf-8-sig')
# plot_radial_comparison(state_radial, pixel2mm=H/gridSize, prefix=PREFIX, out_dir=save_dir)
#
# # state_radial_cum = radial_cumulative_stats(df, depth.shape, bin_width_px=31, center=[440, 440])
# # plot_radial_cum_comparison(state_radial_cum, pixel2mm=H/gridSize, prefix=PREFIX, out_dir=save_dir)
# # print(depth.shape)
#
# state_angular = angular_bin_stats(df, depth.shape, bin_width_rad=np.pi/18)
# plot_angular_comparison(state_angular, prefix=PREFIX, out_dir=save_dir)

# ############################---===【dwh5】===---############################
# PREFIX = 'dwh5'
# ids, coords = load_k_file(r'D:\PHDstudent\博士论文\损伤代理模型\01 小论文\实验数据\板扫描模型\dwh-5\dwh-5-front.k')  #-
# depth, T = compute_contour_depth(ids, coords, return_transform=True)
# coords_h = np.c_[coords, np.ones(len(coords))]      # (N,4) 齐次
# coords_aligned = (T @ coords_h.T).T[:, :3]          # 全量变换 → (N,3)
#
# # 可视化时直接用 coords_aligned
# depth = -visualize_depth_grid(coords_aligned, depth, bbox=((-80, 80), (-80, 80)), grid_size=gridSize,
#     save_npy=r'D:\PHDstudent\博士论文\损伤代理模型\01 小论文\实验数据\板扫描模型\out\dwh5_depth_160x160.npy')         # 函数签名略改
#
# df = extract_pits(depth, bg_factor=0.40, gf_sigma=5, pixel2mm=H/gridSize)
# df_path = os.path.join(save_dir, f'{PREFIX}_pits.csv')   # 也可以改成 .pkl
# df.to_csv(df_path, index=False, encoding='utf-8-sig')
# print(f'[INFO] 已将坑信息保存到 {df_path}')

# ############################---===【dwh4】===---############################
# PREFIX = 'dwh4'
#
# ids, coords = load_k_file(r'D:\PHDstudent\博士论文\损伤代理模型\01 小论文\实验数据\板扫描模型\dwh-4\dwh-4-front.k')
# depth, T = compute_contour_depth(ids, coords, return_transform=True)
# coords_h = np.c_[coords, np.ones(len(coords))]      # (N,4) 齐次
# coords_aligned = (T @ coords_h.T).T[:, :3]          # 全量变换 → (N,3)
#
# # 可视化时直接用 coords_aligned
# depth = visualize_depth_grid(coords_aligned, depth, bbox=((-80, 80), (-80, 80)), grid_size=gridSize,
#     save_npy=r'D:\PHDstudent\博士论文\损伤代理模型\01 小论文\实验数据\板扫描模型\out\dwh4_depth_160x160.npy')         # 函数签名略改
#
# df = extract_pits(depth, bg_factor=0.37, gf_sigma=5, pixel2mm=H/gridSize)
#
# df_path = os.path.join(save_dir, f'{PREFIX}_pits.csv')   # 也可以改成 .pkl
# df.to_csv(df_path, index=False, encoding='utf-8-sig')
# print(f'[INFO] 已将坑信息保存到 {df_path}')
#
# plot_value_count_distribution(df, prefix=PREFIX, bins=80, out_dir=save_dir)
# state_radial = radial_bin_stats(df, depth.shape, bin_width_px=31, center=[550, 401])
# state_path = os.path.join(save_dir, f'{PREFIX}_pits_state_radial.csv')   # 也可以改成 .pkl
# state_radial.to_csv(state_path, index=False, encoding='utf-8-sig')
# plot_radial_comparison(state_radial, pixel2mm=H/gridSize, prefix=PREFIX, out_dir=save_dir)
#
# state_angular = angular_bin_stats(df, depth.shape, bin_width_rad=np.pi/18)
# plot_angular_comparison(state_angular, prefix=PREFIX, out_dir=save_dir)

# ############################---===【dwh6】===---############################
# PREFIX = 'dwh6'
# ids, coords = load_k_file(r'D:\PHDstudent\博士论文\损伤代理模型\01 小论文\实验数据\板扫描模型\dwh-6\dwh-6-front.k')
# depth, T = compute_contour_depth(ids, coords, return_transform=True)
# coords_h = np.c_[coords, np.ones(len(coords))]      # (N,4) 齐次
# coords_aligned = (T @ coords_h.T).T[:, :3]          # 全量变换 → (N,3)
#
# # 可视化时直接用 coords_aligned
# depth = visualize_depth_grid(coords_aligned, depth, bbox=((-80, 80), (-80, 80)), grid_size=gridSize,
#     save_npy=r'D:\PHDstudent\博士论文\损伤代理模型\01 小论文\实验数据\板扫描模型\out\dwh6_depth_160x160.npy')         # 函数签名略改
#
# df = extract_pits(depth, bg_factor=0.375, gf_sigma=5, pixel2mm=H/gridSize)
#
# df_path = os.path.join(save_dir, f'{PREFIX}_pits.csv')   # 也可以改成 .pkl
# df.to_csv(df_path, index=False, encoding='utf-8-sig')
# print(f'[INFO] 已将坑信息保存到 {df_path}')
#
# plot_value_count_distribution(df, prefix=PREFIX, bins=80, out_dir=save_dir)
# state_radial = radial_bin_stats(df, depth.shape, bin_width_px=31, center=[315, 477])
# state_path = os.path.join(save_dir, f'{PREFIX}_pits_state_radial.csv')   # 也可以改成 .pkl
# state_radial.to_csv(state_path, index=False, encoding='utf-8-sig')
# plot_radial_comparison(state_radial, pixel2mm=H/gridSize, prefix=PREFIX, out_dir=save_dir)
#
# state_angular = angular_bin_stats(df, depth.shape, bin_width_rad=np.pi/18)
# plot_angular_comparison(state_angular, prefix=PREFIX, out_dir=save_dir)

###########################---===【dwh7】===---############################
PREFIX = 'dwh7'
ids, coords = load_k_file(r'D:\PHDstudent\博士论文\损伤代理模型\01 小论文\实验数据\板扫描模型\dwh-7\dwh-7-front.k')
depth, T = compute_contour_depth(ids, coords, return_transform=True)
coords_h = np.c_[coords, np.ones(len(coords))]      # (N,4) 齐次
coords_aligned = (T @ coords_h.T).T[:, :3]          # 全量变换 → (N,3)

# 可视化时直接用 coords_aligned
depth = visualize_depth_grid(coords_aligned, depth, bbox=((-80, 80), (-80, 80)), grid_size=gridSize,
    save_npy=r'D:\PHDstudent\博士论文\损伤代理模型\01 小论文\实验数据\板扫描模型\out\dwh7_depth_160x160.npy')         # 函数签名略改

df = extract_pits(depth, bg_factor=0.545, gf_sigma=5, pixel2mm=H/gridSize)

df_path = os.path.join(save_dir, f'{PREFIX}_pits.csv')   # 也可以改成 .pkl
df.to_csv(df_path, index=False, encoding='utf-8-sig')
print(f'[INFO] 已将坑信息保存到 {df_path}')

plot_value_count_distribution(df, prefix=PREFIX, bins=80, out_dir=save_dir)
state_radial = radial_bin_stats(df, depth.shape, bin_width_px=31, center=[466, 475])
state_path = os.path.join(save_dir, f'{PREFIX}_pits_state_radial.csv')   # 也可以改成 .pkl
state_radial.to_csv(state_path, index=False, encoding='utf-8-sig')
plot_radial_comparison(state_radial, pixel2mm=H/gridSize, prefix=PREFIX, out_dir=save_dir)

state_angular = angular_bin_stats(df, depth.shape, bin_width_rad=np.pi/18)
plot_angular_comparison(state_angular, prefix=PREFIX, out_dir=save_dir)

# ############################---===【dwh8】===---############################
# PREFIX = 'dwh8'
# ids, coords = load_k_file(r'D:\PHDstudent\博士论文\损伤代理模型\01 小论文\实验数据\板扫描模型\dwh-8\dwh-8-front.k')
# depth, T = compute_contour_depth(ids, coords, return_transform=True)
# coords_h = np.c_[coords, np.ones(len(coords))]      # (N,4) 齐次
# coords_aligned = (T @ coords_h.T).T[:, :3]          # 全量变换 → (N,3)
#
# # 可视化时直接用 coords_aligned
# depth = visualize_depth_grid(coords_aligned, depth, bbox=((-80, 80), (-80, 80)), grid_size=gridSize,
#     save_npy=r'D:\PHDstudent\博士论文\损伤代理模型\01 小论文\实验数据\板扫描模型\out\dwh8_depth_160x160.npy')         # 函数签名略改
#
# df = extract_pits(depth, bg_factor=0.40, gf_sigma=5, pixel2mm=H/gridSize)
#


# plot_value_count_distribution(df, prefix=PREFIX, bins=80, out_dir=save_dir)
#
# state_radial = radial_bin_stats(df, depth.shape, bin_width_px=31)
# plot_radial_comparison(state_radial, pixel2mm=H/gridSize, prefix=PREFIX, out_dir=save_dir)
# print(depth.shape)
#
# state_angular = angular_bin_stats(df, depth.shape, bin_width_rad=np.pi/18)
# plot_angular_comparison(state_angular, prefix=PREFIX, out_dir=save_dir)