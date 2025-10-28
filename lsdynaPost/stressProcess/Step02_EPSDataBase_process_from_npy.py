import os
import numpy as np
from lsdynaPost.VirtualRearWall.feather_rotate_utils import get_outer_ring_mean, feather_rotate_cosine_with_blur

def generate_FandB_images_with_rotation(
    vm_npy_file,
    x_range,
    y_range,
    output_dir,
    random_angle_dir,
    data_label,  # "EPStrain" 或 "VMStress"
    N=64
):
    """
    1. 加载 elementMax_{data_label}.npy 的最后一时刻数据。
    2. 重塑为 (切片数, H, W)，沿 z 轴求和，得到 2D 图 eps。
    3. 计算 eps 外圈均值 cval，用于旋转时填充。
    4. 从随机角度文件读取前 N 个角度，对 eps 旋转并堆叠成 (N,1,H,W)。
    5. 保存为 <原名>_FandB_last{data_label}_R{N}.npy。
    """
    if not os.path.isfile(vm_npy_file):
        print(f"[Error] 文件不存在: {vm_npy_file}")
        return

    basename = os.path.basename(vm_npy_file)
    parts = basename.split('_')
    try:
        bumper_num = ''.join(filter(str.isdigit, parts[0]))
        dp_num     = ''.join(filter(str.isdigit, parts[1]))
        vp_num     = ''.join(filter(str.isdigit, parts[2]))
    except Exception:
        print(f"[Error] 无法解析文件名: {basename}")
        return

    data_array = np.load(vm_npy_file)[-1, :]
    data_3d = data_array.reshape((-1, 500, 500))
    # print(np.max(data_3d),np.min(data_3d))
    eps = np.sum(data_3d, axis=0)/data_3d.shape[0]
    # print(np.max(eps),np.min(eps))

    cval = get_outer_ring_mean(eps, border=10)

    angle_file = os.path.join(
        random_angle_dir,
        f"Bumper{bumper_num}mm_dp{dp_num}mm_random_rotate.txt"
    )
    if not os.path.isfile(angle_file):
        print(f"[Error] 随机角度文件不存在: {angle_file}")
        return

    with open(angle_file) as f:
        angles = [float(line) for line in f if line.strip()]
    if len(angles) < N:
        print(f"[Warning] 角度数 ({len(angles)}) 少于 N={N}")
        return
    angles = angles[:N]

    height = int((2 * x_range) / 0.02)
    width  = int((2 * y_range) / 0.02)
    rotated = np.zeros((N, 1, height, width), dtype=np.float32)

    for i, ang in enumerate(angles):
        rotated[i, 0] = feather_rotate_cosine_with_blur(eps, ang, cval)

    os.makedirs(output_dir, exist_ok=True)
    save_name = f"Bumper{bumper_num}_dp{dp_num}_vp{vp_num}_FandB_last{data_label}_R{N}.npy"
    save_path = os.path.join(output_dir, save_name)
    np.save(save_path, rotated)
    print(f"[Info] 已保存旋转图像到: {save_path}")


def batch_generate_FandB_images_with_rotation(
    vm_data_dir,
    x_range,
    y_range,
    output_dir,
    random_angle_dir,
    data_label,  # "EPStrain" 或 "VMStress"
    N=64
):
    """
    批量处理 vm_data_dir 下所有 elementMax_{data_label}.npy 文件，
    如果输出已存在则跳过处理，避免重复。
    """
    suffix = f"_elementMax_{data_label}.npy"
    # 预先读取已有的输出文件列表，避免重复处理
    existing_outputs = set(os.listdir(output_dir)) if os.path.isdir(output_dir) else set()

    for fname in os.listdir(vm_data_dir):
        if not fname.endswith(suffix):
            continue
        # 解析 bumper/dp/vp
        parts = fname.split('_')
        try:
            bumper_num = ''.join(filter(str.isdigit, parts[0]))
            dp_num     = ''.join(filter(str.isdigit, parts[1]))
            vp_num     = ''.join(filter(str.isdigit, parts[2]))
        except Exception:
            print(f"[Error] 无法解析文件名: {fname}")
            continue

        # 构造目标输出文件名
        save_name = f"Bumper{bumper_num}_dp{dp_num}_vp{vp_num}_FandB_last{data_label}_R{N}.npy"
        if save_name in existing_outputs:
            print(f"[Skip] 已存在: {save_name}")
            continue

        path = os.path.join(vm_data_dir, fname)
        print(f"[Batch] 处理 {fname} -> {save_name}")
        generate_FandB_images_with_rotation(
            vm_npy_file=path,
            x_range=x_range,
            y_range=y_range,
            output_dir=output_dir,
            random_angle_dir=random_angle_dir,
            data_label=data_label,
            N=N
        )


if __name__ == "__main__":
    ep_strain_dir = r"D:\SIMULATION\p01_DebrisCloudDamageDataBase\originalData\original_elementValue_npyFiles\EPStrain_Data"
    output_dir_strain = r"D:\SIMULATION\p01_DebrisCloudDamageDataBase\DataBase\DataBase_EffectivePlasticStrain"
    random_angle_dir_strain = r"D:\SIMULATION\p01_DebrisCloudDamageDataBase\originalData\data_augmentation\512"
    x_range = y_range = 5.0
    N = 512

    # EPStrain
    batch_generate_FandB_images_with_rotation(
        vm_data_dir=ep_strain_dir,
        x_range=x_range,
        y_range=y_range,
        output_dir=output_dir_strain,
        random_angle_dir=random_angle_dir_strain,
        data_label="EPStrain",
        N=N
    )

    # VMStress (若需要，可类似调用)
    # vm_stress_dir = r"...\VMStress_Data"
    # output_dir_stress = r"...\DataBase_VMStress"
    # random_angle_dir_stress = r"...\data_augmentation\512"
    # batch_generate_FandB_images_with_rotation(
    #     vm_data_dir=vm_stress_dir,
    #     x_range=x_range,
    #     y_range=y_range,
    #     output_dir=output_dir_stress,
    #     random_angle_dir=random_angle_dir_stress,
    #     data_label="VMStress",
    #     N=N
    # )
