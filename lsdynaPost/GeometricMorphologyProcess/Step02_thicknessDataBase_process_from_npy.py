import os
import re
import numpy as np
from scipy.interpolate import griddata
from lsdynaPost.VirtualRearWall.feather_rotate_utils import feather_rotate_cosine_with_blur

def get_outer_ring_mean(array, border=2):
    if array.ndim != 2:
        raise ValueError("只支持二维数组")
    h, w = array.shape
    mask = np.zeros_like(array, dtype=bool)
    mask[:border, :] = True
    mask[-border:, :] = True
    mask[:, :border] = True
    mask[:, -border:] = True
    ring_values = array[mask]
    ring_values = ring_values[~np.isnan(ring_values)]
    return np.mean(ring_values) if ring_values.size > 0 else 0.0

def coords_to_grid_image(coords, values, x_range=5.0, y_range=5.0, step=0.02):
    grid_x, grid_y = np.meshgrid(
        np.linspace(-x_range, x_range, int(2 * x_range / step)),
        np.linspace(-y_range, y_range, int(2 * y_range / step))
    )
    grid_z = griddata(coords[:, :2], values, (grid_x, grid_y), method='cubic')
    cval = get_outer_ring_mean(grid_z, border=2)
    grid_z = np.nan_to_num(grid_z, nan=cval)
    return grid_z

def generate_rotated_image_stack(file_path, output_dir, random_angle_dir,
                                 x_range=5.0, y_range=5.0, step=0.02, N=64):
    basename = os.path.basename(file_path)
    try:
        # 去掉扩展名，再按下划线分段
        name_no_ext = os.path.splitext(basename)[0]
        parts = name_no_ext.split('_')

        bumper_num = dp_num = vp_num = None
        # 逐段查找 Bumper、dp、vp 前缀的数字（含小数）
        for part in parts:
            if part.startswith('Bumper'):
                m = re.search(r'Bumper(\d+(?:\.\d+)?)', part)
                if m: bumper_num = m.group(1)
            elif part.startswith('dp'):
                m = re.search(r'dp(\d+(?:\.\d+)?)', part)
                if m: dp_num = m.group(1)
            elif part.startswith('vp'):
                m = re.search(r'vp(\d+(?:\.\d+)?)', part)
                if m: vp_num = m.group(1)

        if None in (bumper_num, dp_num, vp_num):
            raise ValueError(f"无法从文件名中解析数字: {basename}")

    except Exception as e:
        print(f"[Error] 文件名解析失败: {file_path}, error: {e}")
        return

    # 加载数据
    try:
        data = np.load(file_path, allow_pickle=True).item()
        top_coords = data['top_coords']
        bottom_coords = data['bottom_coords']
    except Exception as e:
        print(f"[Error] 加载或字段缺失: {file_path}, error: {e}")
        return

    top_image    = coords_to_grid_image(top_coords, top_coords[:, 2], x_range, y_range, step)
    bottom_image = coords_to_grid_image(bottom_coords, bottom_coords[:, 2], x_range, y_range, step)
    h, w = top_image.shape

    # 读取旋转角度
    angle_file = os.path.join(
        random_angle_dir,
        f"Bumper{bumper_num}mm_dp{dp_num}mm_random_rotate.txt"
    )
    if not os.path.isfile(angle_file):
        print(f"[Error] 角度文件不存在: {angle_file}")
        return
    with open(angle_file, 'r') as f:
        angles = [float(line.strip()) for line in f if line.strip()]
    if len(angles) < N:
        print(f"[Warning] 角度不足 N={N}: 仅有 {len(angles)} 个")
        return
    angles = angles[:N]

    cval_top    = get_outer_ring_mean(top_image)
    cval_bottom = get_outer_ring_mean(bottom_image)

    # 构建旋转图像堆栈
    image_stack = np.zeros((N, 1, h, w), dtype=np.float32)
    for i, angle in enumerate(angles):
        img_t = feather_rotate_cosine_with_blur(top_image, angle, cval_top)
        img_b = feather_rotate_cosine_with_blur(bottom_image, angle, cval_bottom)
        image_stack[i, 0] = img_b - img_t  #板相对厚度变化
        # image_stack[i, 0] = img_t  # 板几何形貌

    # 保存
    os.makedirs(output_dir, exist_ok=True)
    save_name = f"Bumper{bumper_num}_dp{dp_num}_vp{vp_num}_thickness_R{N}.npy"
    save_path = os.path.join(output_dir, save_name)
    np.save(save_path, image_stack)
    print(f"[Done] 保存至: {save_path}, {image_stack.shape}")


def batch_process_top_bottom_dir(
    input_dir,
    output_dir,
    random_angle_dir,
    x_range=5.0,
    y_range=5.0,
    step=0.02,
    N=64
):
    for fname in os.listdir(input_dir):
        if not fname.endswith(".npy"):
            continue
        fpath = os.path.join(input_dir, fname)
        generate_rotated_image_stack(
            file_path=fpath,
            output_dir=output_dir,
            random_angle_dir=random_angle_dir,
            x_range=x_range,
            y_range=y_range,
            step=step,
            N=N
        )

if __name__ == "__main__":
    input_dir = r"D:\SIMULATION\p01_DebrisCloudDamageDataBase\originalData\original_elementValue_npyFiles\damageM_Data\NP"
    output_dir = r"D:\SIMULATION\p01_DebrisCloudDamageDataBase\DataBase\DataBase_thickness500_128"
    random_angle_dir = r"D:\SIMULATION\p01_DebrisCloudDamageDataBase\originalData\data_augmentation\512"
    batch_process_top_bottom_dir(
        input_dir=input_dir,
        output_dir=output_dir,
        random_angle_dir=random_angle_dir,
        x_range=5.0,
        y_range=5.0,
        step=0.02,
        N=128
    )
