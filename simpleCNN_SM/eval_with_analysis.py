import os
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split
import kornia
from kornia.losses import ssim_loss

from dataset_pt import PreprocessedPTDataset
from train_SM import initialize_weights, set_seed, edge_loss
from model.model_cnn import SimpleCNN
from utils.logger import Logger
import os
import numpy as np
import torch
import torch.nn.functional as F
from skimage.feature import blob_log
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from skimage.transform import rescale
from skimage.measure import label, regionprops
import pandas as pd
import os

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


def radial_bin_stats(df, img_shape, bin_width_px=1, center=None):
    """
    按同心圆环分箱统计:
      - count: 坑个数
      - mean_depth: 平均坑深
      - mean_diam: 平均坑直径
      - sum_volume: 总体积代理 (area*depth)
    df: 包含列 ['y_px','x_px','depth_est','radius_px','volume_px3']
    """
    H, W = img_shape[-2:]
    if center is None:
        center = (H/2.0, W/2.0)
    cy, cx = center
    r = np.hypot(df['x_px'] - cx, df['y_px'] - cy)
    bin_idx = (r // bin_width_px).astype(int)
    df2 = df.assign(r=r, bin=bin_idx)
    grp = df2.groupby('bin')
    stats = grp.agg(
        r_mid=('r','mean'),
        count=('r','size'),
        mean_depth=('depth_est','mean'),
        mean_diam=('radius_px', lambda x: (2*x).mean()),
        sum_volume=('volume_px3','sum')
    ).reset_index(drop=True)
    return stats

def extract_pits(img):
    """
    提取图像中的坑信息，返回包含过滤后斑点信息的 DataFrame。
    """
    # 如果输入为 torch.Tensor，先转到 CPU numpy
    if isinstance(img, torch.Tensor):
        img = img.detach().cpu().numpy().squeeze()

    Z = img
    # print(Z.shape)

    # ======= 1. 背景扣除 =======
    background = gaussian_filter(Z, sigma=1000)
    Z_res = background - Z  # 小坑为负
    # print()
    background_threshold = (np.max(Z_res)-np.min(Z_res))*0.2 + np.min(Z_res)

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
        radius_px = diameter / 2  # 假设坑是圆形的
        depth_est = region_max_value  # 可以使用最大值作为深度估计，或者根据实际情况调整
        volume_px3 = area * depth_est / 3  # 用面积和深度估计计算体积代理

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

    # 计算面积的 99% 分位数
    area_threshold = np.percentile(all_areas, 50)
    filtered_spots_info = [spot for spot in spots_info if spot['area'] > area_threshold]

    # 将过滤后的斑点信息转换为 DataFrame
    filtered_spots_df = pd.DataFrame(filtered_spots_info)

    print("过滤后的斑点统计结果：")
    print(filtered_spots_df)

    r_99 = compute_radius_99(filtered_spots_df, Z.shape)
    print(f"99% 的损伤面积覆盖半径为: {r_99*0.2*2:.2f} mm")

    # ======= 6. 可视化原始图像与二值化结果 =======
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(Z_res, cmap='coolwarm');
    ax[0].set_title('Residual Depth')
    ax[1].imshow(Z_binary, cmap='gray');
    ax[1].set_title('Binary Image (Thresholding)')

    for a in ax: a.axis('off')
    plt.tight_layout();
    plt.show()

    return filtered_spots_df

def plot_radial_comparison(stats_t, stats_p, out_dir, prefix=''):
    """
    比较target与pred的radial统计并保存图像
    """
    os.makedirs(out_dir, exist_ok=True)
    metrics = [('count','Pit count'), ('mean_depth','Mean depth'), ('mean_diam','Mean diameter')]
    for key, ylabel in metrics:
        plt.figure()
        plt.plot(stats_t['r_mid']*0.2, stats_t[key], 'o-', label='target')
        plt.plot(stats_p['r_mid']*0.2, stats_p[key], 's--', label='pred')
        plt.xlabel('Radius (mm)')
        plt.ylabel(ylabel)
        plt.title(f'{prefix} {ylabel} vs radius')
        plt.legend()
        plt.grid(True)
        fname = os.path.join(out_dir, f"{prefix}_{key}.png")
        plt.tight_layout()
        plt.savefig(fname, dpi=300)
        plt.close()
def plot_radial_comparison(stats_t, stats_p, out_dir, prefix='', pixel2mm=0.2):
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
        plt.plot(stats_t['r_mid'] * pixel2mm, stats_t[key], 'o-', linewidth=2, label='target')
        plt.plot(stats_p['r_mid'] * pixel2mm, stats_p[key], 'o-', linewidth=2, label='target')
        plt.xlabel('Radius (mm)')
        plt.ylabel(ylabel)
        plt.title(f'{prefix} {ylabel} vs Radius')
        plt.legend()
        plt.grid(True)
        fname = os.path.join(out_dir, f"{prefix}_radial_{key}_center.png")
        plt.tight_layout()
        plt.savefig(fname, dpi=300)
        plt.close()

def plot_distribution_curves(df_t, df_p, out_dir, prefix='', bins=20):
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
        plt.figure()
        plt.plot(centers, cnt_t, 'o-', label='target')
        plt.plot(centers, cnt_p, 's--', label='pred')
        plt.xlabel(xlabel)
        plt.ylabel('Pit count')
        plt.title(f'{prefix} Pit count distribution by {xlabel}')
        plt.legend()
        plt.grid(True)
        fname = os.path.join(out_dir, f"{prefix}_dist_{key}.png")
        plt.tight_layout()
        plt.savefig(fname, dpi=150)
        plt.close()

def save_pit_results(df, out_dir, prefix=''):
    """
    将pit点DataFrame保存为CSV
    """
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{prefix}_pits.csv")
    df.to_csv(path, index=False)
    print(f"[Saved] {path}")

def eval_model(pt_dir, opt, model_path, param_dir, batch_size=64, num_workers=4, device='cuda:0', seed=20250410):
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    print(f"Evaluating model: {model_path}")

    out_root = args.param_dir

    # 加载验证集（用相同的随机种子划分）
    dataset = PreprocessedPTDataset(pt_dir)
    set_seed(seed)
    train_size = int(0.8 * len(dataset))
    valid_size = len(dataset) - train_size
    _, valid_dataset = random_split(dataset, [train_size, valid_size])
    valid_dl = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False,
                          num_workers=num_workers, pin_memory=True)

    # 加载 global_stats（反归一化用）
    global_stats_file = f'params/global_stats_{opt}.npy'
    if os.path.exists(global_stats_file):
        global_stats = np.load(global_stats_file, allow_pickle=True).item()
    else:
        global_stats = None

    # 模型初始化与加载
    model = SimpleCNN(opt).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device)['model_state_dict'])
    model.eval()

    # 评估指标
    ssim_scores, psnr_scores, mse_scores = [], [], []

    with torch.no_grad():
        for batch_idx, (signals, targets) in enumerate(valid_dl):
            signals, targets = signals.to(device), targets.to(device)
            preds = model(signals)

            # 评估
            loss = F.mse_loss(preds, targets)
            ssim_scores.append(ssim_loss(preds, targets, window_size=11).item())
            psnr_scores.append(10 * torch.log10(1 / loss).item())
            mse_scores.append(F.mse_loss(preds, targets).item())

            # 可视化前3张
            if batch_idx < 300:
                # 获取完整 target / prediction 并反归一化
                pred_img = preds[0].cpu().numpy()  # [C, H, W]
                target_img = targets[0].cpu().numpy()  # [C, H, W]

                if global_stats is not None:
                    pred_img = pred_img * (global_stats['target_max'] - global_stats['target_min']) + global_stats[
                        'target_min']
                    target_img = target_img * (global_stats['target_max'] - global_stats['target_min']) + global_stats[
                        'target_min']

                # target_img, pred_img: [C, H, W] 的 numpy 数组
                num_channels = target_img.shape[0]
                for c in range(num_channels):
                    # 保存 target
                    save_t = os.path.join(param_dir, f"eval_{batch_idx}_target_C{c}.png")
                    plt.imsave(save_t, target_img[c], cmap='coolwarm', format='png')

                    # 保存 pred
                    save_p = os.path.join(param_dir, f"eval_{batch_idx}_pred_C{c}.png")
                    plt.imsave(save_p, pred_img[c], cmap='coolwarm', format='png')

                    pi, ti = preds[c], targets[c]
                    # 提取pit数据
                    df_t = extract_pits(ti)
                    df_p = extract_pits(pi)
                    # 保存点结果
                    subdir = os.path.join(out_root, f"batch{batch_idx}_C{c}")
                    save_pit_results(df_t, subdir, 'target')
                    save_pit_results(df_p, subdir, 'pred')

                    # —— 先计算一下 r_99（假设你已经在这一通道里算出了 df_t、df_p）
                    # 这里以 target 为例，pred 同理
                    r_99_t = compute_radius_99(df_t, target_img[c].shape)

                    # —— 同样的逻辑用于 pred
                    r_99_p = compute_radius_99(df_p, pred_img[c].shape)

                    # 分箱统计
                    stats_t = radial_bin_stats(df_t, ti.shape, bin_width_px=10)
                    stats_p = radial_bin_stats(df_p, pi.shape, bin_width_px=10)
                    # 保存统计表
                    stats_t.to_csv(os.path.join(subdir, 'target_stats.csv'), index=False)
                    stats_p.to_csv(os.path.join(subdir, 'pred_stats.csv'), index=False)
                    # 绘制半径曲线并保存
                    plot_radial_comparison(stats_t, stats_p, subdir, prefix=f"batch{batch_idx}_C{c}")
                    # 绘制原始分布统计曲线并保存
                    plot_distribution_curves(df_t, df_p, subdir, prefix=f"batch{batch_idx}_C{c}")
                # plt.tight_layout()
                # plt.savefig(os.path.join(param_dir, f"eval_sample_{batch_idx}.png"))
                # plt.close()

    # 打印平均指标
    print(f"[Evaluation Result]")
    print(f"Avg SSIM: {np.mean(ssim_scores):.4f}")
    print(f"Avg PSNR: {np.mean(psnr_scores):.2f}")
    print(f"Avg MSE : {np.mean(mse_scores):.6f}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str,
                        default=r'D:\SIMULATION\p01_DebrisCloudDamageDataBase\params\thickness\0613_cnn18\EPS_PTModel_epoch900.pth')
    # 420 440 460 530 620 640 650 700 900 960 1010 1020
    parser.add_argument('--pt_dir', type=str,
                        default=r'D:\SIMULATION\p01_DebrisCloudDamageDataBase\DataBase\PreprocessedPT\thickness')
    parser.add_argument('--opt', type=str, default='EPS', help='Target type: EPS, Ek, damageM')
    parser.add_argument('--param_dir', type=str, default='params/thickness/cnn18/valid_900_250705/')

    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--seed', type=int, default=20250410)
    # parser.add_argument('--param_dir', type=str, required=True)
    # parser.add_argument('--bin_width', type=int, default=10)
    # parser.add_argument('--analyze_n', type=int, default=100)

    args = parser.parse_args()


    os.makedirs(args.param_dir, exist_ok=True)
    eval_model(**vars(args))
