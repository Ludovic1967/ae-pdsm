import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from skimage.transform import rescale
from skimage.measure import label, regionprops
import pandas as pd
import os


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


def visualize_one_topbottom(np_file, up_factor=4, background_threshold=0):
    # ======= 0. 读数据 =======
    data = np.load(np_file, allow_pickle=True).item()
    Z = (data['bottom_coords'][:, 2] - data['top_coords'][:, 2]).reshape((500, 500))
    # Z = (data['top_coords'][:, 2]).reshape((500, 500))

    # ======= 1. 背景扣除 =======
    background = gaussian_filter(Z, sigma=1000)
    Z_res = Z - background  # 小坑为负
    print(np.max(Z_res), np.min(Z_res))

    print((np.max(Z_res)-background_threshold)/(np.max(Z_res)-np.min(Z_res)))

    # ======= 2. 上采样 (亚像素插值) =======
    # Z_hr = rescale(Z_res, up_factor, order=3, mode='reflect',
    #                anti_aliasing=True, preserve_range=True)

    # ======= 3. 使用阈值对图像进行二值化处理 =======
    Z_binary = Z_res > background_threshold  # 将大于背景的区域设为 1，小于背景的区域设为 0

    # ======= 4. 统计 Z_binary 中斑点的个数和直径 =======
    # 进行连通区域标记
    labeled_image = label(Z_binary)

    # 获取斑点的属性
    regions = regionprops(labeled_image, intensity_image=Z_res)

    # 存储斑点信息
    spots_info = []
    all_areas = []  # 用来存储所有斑点的面积

    for region in regions:
        # 记录斑点的面积
        area = region.area
        all_areas.append(area)

        # 斑点的直径（通过面积估算直径，假设斑点为圆形）
        diameter = 2 * np.sqrt(area / np.pi)

        # 获取对应区域在原始 Z 图像中的最大值
        minr, minc, maxr, maxc = region.bbox
        region_max_value = np.max(Z_res[minr:maxr, minc:maxc])

        # 记录斑点的属性
        spots_info.append({
            'label': region.label,
            'area': area,
            'diameter': diameter,
            'max_value_in_region': region_max_value,
            'x_px': region.centroid[1],  # x 坐标
            'y_px': region.centroid[0],  # y 坐标
            'area_px2': area  # 将 area 作为 area_px2 传递
        })

    # 计算面积的 99% 分位数
    area_threshold = np.percentile(all_areas, 50)

    # 过滤掉面积小于 99% 分位数的斑点
    filtered_spots_info = [spot for spot in spots_info if spot['area'] > area_threshold]

    # 将过滤后的斑点信息转换为 DataFrame
    filtered_spots_df = pd.DataFrame(filtered_spots_info)

    # 打印统计结果
    print("过滤后的斑点统计结果：")
    print(filtered_spots_df)

    # ======= 5. 计算 99% 面积对应的半径 =======
    # r_99 = compute_radius_99(filtered_spots_df, Z.shape)
    # print(f"99% 的损伤面积覆盖半径为: {r_99*0.2*2:.2f} mm")

    # ======= 6. 可视化原始图像与二值化结果 =======
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(Z_res, cmap='coolwarm');
    ax[0].set_title('Residual Depth')
    ax[1].imshow(Z_binary, cmap='gray');
    ax[1].set_title('Binary Image (Thresholding)')

    # 在二值化图像上标出斑点（只标出过滤后的斑点）
    for region in regions:
        # print(f"Region bbox: {region.bbox}")  # 输出 bbox，检查其返回的内容
        area = region.area
        if area >= area_threshold:
            minr, minc, maxr, maxc = region.bbox
            ax[1].add_patch(
                plt.Rectangle((minc, minr), maxc - minc, maxr - minr, edgecolor='red', facecolor='none', linewidth=0.5))

    # # 画出包含99%面积的圆
    # cy, cx = Z.shape[0] / 2, Z.shape[1] / 2  # 圆心是图像中心
    # circle = plt.Circle((cx, cy), r_99, color='blue', fill=False, linewidth=0.5)
    # ax[1].add_patch(circle)

    for a in ax: a.axis('off')
    plt.tight_layout();
    plt.show()

    file_name = os.path.basename(np_file).replace('.npy', '')

    fig.savefig(fr"D:\PHDstudent\博士论文\损伤代理模型\01 小论文\二值化处理\{file_name}.tif", dpi=300, bbox_inches="tight", pad_inches=0.5)


if __name__ == "__main__":
    # ========== 示例路径，请按需修改 ==========
    np_file_dir = r"D:\SIMULATION\p01_DebrisCloudDamageDataBase\originalData\original_elementValue_npyFiles\damageM_Data\NP"
    np_file_suffix = "_damageM.npy"

    files = sorted([
        f for f in os.listdir(np_file_dir)
        if f.endswith(np_file_suffix)
    ])

    if len(files) == 0:
        print(f"[Warning] No matching '*{np_file_suffix}' files found.")

    for fname in files:
        full_path = os.path.join(np_file_dir, fname)
        print(full_path)
        visualize_one_topbottom(full_path, background_threshold=6e-4)
