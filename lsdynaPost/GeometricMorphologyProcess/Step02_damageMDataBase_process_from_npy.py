import os
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

def generate_rotated_image_stack(file_path, output_dir, random_angle_dir, x_range=5.0, y_range=5.0, step=0.02, N=64):
    basename = os.path.basename(file_path)
    try:
        # name_parts = basename.split('_')
        # bumper_num = ''.join(filter(str.isdigit, name_parts[0]))
        # dp_num = ''.join(filter(str.isdigit, name_parts[1]))
        # vp_num = ''.join(filter(str.isdigit, name_parts[2]))
        file_name_parts = basename.split('_')
        bumper_num = file_name_parts[0]
        dp_num = file_name_parts[1]
        vp_num = file_name_parts[2]
    except Exception as e:
        print(f"[Error] 文件名解析失败: {file_path}, error: {e}")
        return

    try:
        data = np.load(file_path, allow_pickle=True).item()
        top_coords = data['top_coords']
        bottom_coords = data['bottom_coords']
    except Exception as e:
        print(f"[Error] 加载或字段缺失: {file_path}, error: {e}")
        return

    top_image = coords_to_grid_image(top_coords, top_coords[:, 2], x_range, y_range, step)
    bottom_image = coords_to_grid_image(bottom_coords, bottom_coords[:, 2], x_range, y_range, step)

    h, w = top_image.shape

    # 获取旋转角度
    angle_file = os.path.join(random_angle_dir, f"{bumper_num}mm_{dp_num}mm_random_rotate.txt")
    if not os.path.isfile(angle_file):
        print(f"[Error] 角度文件不存在: {angle_file}")
        return

    with open(angle_file, 'r') as f:
        angles = [float(line.strip()) for line in f if line.strip()]
    if len(angles) < N:
        print(f"[Warning] 角度不足 N={N}: 仅有 {len(angles)} 个")
        return
    angles = angles[:N]

    cval_top = get_outer_ring_mean(top_image)
    cval_bottom = get_outer_ring_mean(bottom_image)

    # 旋转增强
    image_stack = np.zeros((N, 2, h, w), dtype=np.float32)
    for i, angle in enumerate(angles):
        image_stack[i, 0] = feather_rotate_cosine_with_blur(top_image, angle, cval_top)
        image_stack[i, 1] = feather_rotate_cosine_with_blur(bottom_image, angle, cval_bottom)

    # 保存
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    save_name = f"Bumper{bumper_num}_dp{dp_num}_vp{vp_num}_damageM_R{N}.npy"
    save_path = os.path.join(output_dir, save_name)
    np.save(save_path, image_stack)
    print(f"[Done] 保存至: {save_path}")

def batch_process_top_bottom_dir(
    input_dir,
    output_dir,
    random_angle_dir,
    x_range=5.0,
    y_range=5.0,
    step=0.06,
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
    output_dir = r"D:\SIMULATION\p01_DebrisCloudDamageDataBase\DataBase\DataBase_DamageMorphology_64"
    random_angle_dir = r"D:\SIMULATION\p01_DebrisCloudDamageDataBase\originalData\data_augmentation\512"
    batch_process_top_bottom_dir(
        input_dir=input_dir,
        output_dir=output_dir,
        random_angle_dir=random_angle_dir,
        x_range=5.0,
        y_range=5.0,
        step=0.02,
        N=64
    )
