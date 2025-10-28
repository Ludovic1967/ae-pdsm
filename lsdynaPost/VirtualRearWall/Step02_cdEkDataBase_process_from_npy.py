import os
import re
import numpy as np

# 从 feather_rotate_utils.py 导入函数
from feather_rotate_utils import get_outer_ring_mean, feather_rotate_cosine_with_blur

def rotate_phi_n_times(
    phi_npy_file,
    random_angle_dir,
    output_dir,
    num_rotations,
    border=2,
    fade_width=15,
    gaussian_sigma=2.0,
    valid_threshold=1e-20
):
    """
    读取单个 φ(x,y) 文件 (形如 Bumper2_dp4_vp189_dcEk.npy)，
    根据与该文件对应的随机旋转角度文件 random_rotate_Bumper{bumper_num}mm_dp{dp_num}mm.txt，
    进行 N 次羽化旋转并堆叠输出。

    参数：
    ----------
    phi_npy_file : str
        原始 φ(x,y) 文件路径，例如: Bumper2_dp4_vp189_dcEk.npy
    random_angle_dir : str
        随机旋转角度文件所在的目录
        文件名格式: random_rotate_Bumper{bumper_num}mm_dp{dp_num}mm.txt
    output_dir : str
        旋转后的结果保存目录
    num_rotations : int
        旋转的次数
    border : int
        计算外圈均值时，图像外圈的厚度
    fade_width : int
        feather_rotate_cosine_with_blur 中余弦过渡的宽度
    gaussian_sigma : float
        feather_rotate_cosine_with_blur 中高斯模糊的 sigma
    valid_threshold : float
        feather_rotate_cosine_with_blur 中判定像素有效性的阈值
    """

    # ========== 1) 解析 BumperX, dpY, vpZ ==========
    basename = os.path.basename(phi_npy_file)
    pattern = r'Bumper(\d+)_dp(\d+)_vp(\d+)_dcEk\.npy'
    match = re.match(pattern, basename)
    if not match:
        print(f"[Error] 文件名不符合规则，跳过: {basename}")
        return

    bumper_num = match.group(1)  # 例如 '2'
    dp_num     = match.group(2)  # 例如 '4'
    vp_num     = match.group(3)  # 例如 '189'

    # ========== 2) 读取原始场数据 ==========
    if not os.path.isfile(phi_npy_file):
        print(f"[Error] 找不到文件: {phi_npy_file}")
        return

    Phi = np.load(phi_npy_file)  # shape: (H, W)
    if Phi.ndim != 2:
        print(f"[Warning] 期望 Phi 为 2D，但读到 shape={Phi.shape}")
    height, width = Phi.shape
    print(f"[Info] 加载 φ(x,y): {basename}, shape={Phi.shape}")

    # ========== 3) 根据 bumper_num & dp_num 拼装随机角度文件名，读取旋转角度 ==========
    random_angle_filename = f"Bumper{bumper_num}mm_dp{dp_num}mm_random_rotate.txt"
    random_angle_file = os.path.join(random_angle_dir, random_angle_filename)
    if not os.path.isfile(random_angle_file):
        print(f"[Error] Random angle file not found: {random_angle_file}")
        return

    with open(random_angle_file, 'r') as f:
        lines = f.readlines()
    angles = [float(line.strip()) for line in lines]

    if len(angles) < num_rotations:
        print(f"[Warning] The angle file has only {len(angles)} angles, less than {num_rotations}!")
        return
    angles = angles[:num_rotations]

    # ========== 4) 计算外圈均值 cval_outer，并对 Phi 进行 N 次羽化旋转 ==========
    cval_outer = get_outer_ring_mean(Phi, border=border)

    rotated_phis = np.zeros((num_rotations, height, width), dtype=np.float32)
    for i, angle in enumerate(angles):
        rotated_image = feather_rotate_cosine_with_blur(
            image=Phi,
            angle=angle,
            cval=cval_outer,
            fade_width=fade_width,
            gaussian_sigma=gaussian_sigma,
            valid_threshold=valid_threshold
        )
        rotated_phis[i] = rotated_image.astype(np.float32)

    # ========== 5) 保存结果 ==========
    #   例如命名: Bumper2_dp4_vp189_dcEk_RN.npy
    output_filename = f'Bumper{bumper_num}_dp{dp_num}_vp{vp_num}_dcEk_R{num_rotations}.npy'
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, output_filename)
    np.save(save_path, rotated_phis)
    print(f"[Info] 已保存旋转结果到: {save_path}")


def batch_rotate_phi_n_times(
    phi_dir,
    random_angle_dir,
    output_dir,
    num_rotations,
    border=2,
    fade_width=15,
    gaussian_sigma=2.0,
    valid_threshold=1e-20
):
    """
    批量遍历 phi_dir 下的所有 BumperX_dpY_vpZ_dcEk.npy 文件，
    读取与之对应的 random_rotate_Bumper{bumper_num}mm_dp{dp_num}mm.txt 文件，
    进行 N 次羽化旋转，并将结果保存到 output_dir。

    参数：
    ----------
    phi_dir : str
        存放原始 xxx_dcEk.npy 文件的目录
    random_angle_dir : str
        随机旋转角度文件所在的目录，
        文件名格式例如: random_rotate_Bumper2mm_dp4mm.txt
    output_dir : str
        结果保存目录
    num_rotations : int
        旋转的次数
    border : int
        用于计算外圈均值的图像外圈厚度
    fade_width : int
        余弦羽化的过渡宽度
    gaussian_sigma : float
        高斯模糊标准差
    valid_threshold : float
        有效像素判定阈值
    """

    if not os.path.isdir(phi_dir):
        print(f"[Error] Phi 目录不存在: {phi_dir}")
        return

    file_list = os.listdir(phi_dir)
    for filename in file_list:
        if filename.endswith('_dcEk.npy'):
            phi_npy_file = os.path.join(phi_dir, filename)
            rotate_phi_n_times(
                phi_npy_file=phi_npy_file,
                random_angle_dir=random_angle_dir,
                output_dir=output_dir,
                num_rotations=num_rotations,
                border=border,
                fade_width=fade_width,
                gaussian_sigma=gaussian_sigma,
                valid_threshold=valid_threshold
            )


if __name__ == "__main__":
    """
    使用示例:
        1) 修改下面路径为实际路径。
        2) 运行脚本，即可对所有 xxx_dcEk.npy 文件逐一读取对应的随机角度文件，
           并进行 N 次羽化旋转。
    """

    # ========== 用户根据实际情况设置 ==========
    phi_dir = r"D:\SIMULATION\p01_DebrisCloudDamageDataBase\originalData\original_DebrisCloudEk_npyFiles"  # 存放 xxx_dcEk.npy 的目录
    random_angle_dir = r"D:\SIMULATION\p01_DebrisCloudDamageDataBase\originalData\data_augmentation\512"  # 存放随机角度文件的目录
    output_dir = r"D:\SIMULATION\p01_DebrisCloudDamageDataBase\DataBase\DataBase_DebrisCloudEk"  # 旋转后结果的保存目录

    # 其他参数可根据需求自行调整
    border = 2            # 用于计算外圈均值
    fade_width = 15       # 余弦羽化的过渡宽度
    gaussian_sigma = 2.0  # 高斯模糊标准差
    valid_threshold = 1e-20
    num_rotations = 512   # 修改为需要的旋转次数

    batch_rotate_phi_n_times(
        phi_dir=phi_dir,
        random_angle_dir=random_angle_dir,
        output_dir=output_dir,
        num_rotations=num_rotations,
        border=border,
        fade_width=fade_width,
        gaussian_sigma=gaussian_sigma,
        valid_threshold=valid_threshold
    )
