import os
import numpy as np
import matplotlib.pyplot as plt


def visualize_one_FandB_R64(np_file):
    """
    加载并可视化单个 BumperX_dpY_vpZ_FandB_maxVMStress_R64.npy 文件。
    data.shape 应为 [64, 2, height, width]:
        - data[i, 0, :, :] 表示第 i 个旋转得到的前层图像
        - data[i, 1, :, :] 表示第 i 个旋转得到的后层图像
    """
    if not os.path.isfile(np_file):
        print(f"[Error] File not found: {np_file}")
        return

    data = np.load(np_file)
    print(f"[Info] Loaded data from: {np_file}")
    print(f"[Info] Data shape: {data.shape}")  # 期望 (64, 2, H, W)

    # 简单形状检查
    if len(data.shape) != 4 or data.shape[1] != 2:
        print(f"[Warning] Data shape not as expected [64, 2, height, width].")

    # 读取相关维度
    num_rotations = data.shape[0]  # 通常是 64
    height = data.shape[2]
    width = data.shape[3]

    # 如果 num_rotations != 64，可以根据实际情况决定如何可视化
    # 这里默认以 64 为主，如果不足则会出现空余子图；如果超过可能会截断。
    if num_rotations < 64:
        print(f"[Warning] The file has only {num_rotations} rotations, less than 64.")

    # ============== 可视化 front channel ==============
    fig_front, axes_front = plt.subplots(8, 8, figsize=(12, 12))
    fig_front.suptitle(f"Front Channel - {os.path.basename(np_file)}")

    for i in range(min(num_rotations, 64)):
        row = i // 8
        col = i % 8
        ax = axes_front[row][col]
        ax.imshow(data[i, 0, :, :], cmap='jet', origin='lower')
        ax.set_title(f"Rot #{i}")
        ax.axis('off')

    # # ============== 可视化 back channel ==============
    # fig_back, axes_back = plt.subplots(8, 8, figsize=(12, 12))
    # fig_back.suptitle(f"Back Channel - {os.path.basename(np_file)}")
    #
    # for i in range(min(num_rotations, 64)):
    #     row = i // 8
    #     col = i % 8
    #     ax = axes_back[row][col]
    #     ax.imshow(data[i, 1, :, :], cmap='jet', origin='lower')
    #     ax.set_title(f"Rot #{i}")
    #     ax.axis('off')

    plt.show()


def batch_visualize_FandB_R64(np_file_dir, np_file_suffix):
    """
    批量遍历 np_file_dir 下所有匹配 '*FandB_maxVMStress_R64.npy' 的文件，
    依次调用可视化函数进行查看。
    """
    if not os.path.isdir(np_file_dir):
        print(f"[Error] {np_file_dir} is not a valid directory.")
        return

    # 收集所有符合要求的文件
    file_list = [
        f for f in os.listdir(np_file_dir)
        if f.endswith(np_file_suffix)
    ]
    file_list.sort()  # 如需按名称排序

    if len(file_list) == 0:
        print(f"[Warning] No '*{np_file_suffix}' files found in {np_file_dir}")
        return

    print(f"[Info] Found {len(file_list)} files to visualize.\n")

    # 遍历并可视化
    for filename in file_list:
        np_file = os.path.join(np_file_dir, filename)
        visualize_one_FandB_R64(np_file)
        # 每次可视化都会弹出两个窗口(前层、后层)。在大量文件下，
        # 可以考虑自动关闭 plt.show() 或将所有子图绘制到同一窗口。


if __name__ == "__main__":
    """
    示例主函数：批量检验某个目录下的所有 FandB_maxVMStress_R64 文件。
    """
    # ========== 根据需要修改以下路径 ==========
    np_file_dir = r"D:\SIMULATION\p01_DebrisCloudDamageDataBase\DataBase\DataBase_EffectivePlasticStrain"  # 存放 BumperX_dpY_vpZ_FandB_maxVMStress_R64.npy 文件的目录
    np_file_suffix = 'FandB_maxVMStress_R64.npy'
    # np_file_dir = r"D:\SIMULATION\p01_DebrisCloudDamageDataBase\DataBase\DataBase_EffectivePlasticStrain"  # 存放 BumperX_dpY_vpZ_FandB_maxVMStress_R64.npy 文件的目录
    # np_file_suffix = 'FandB_lastEPStrain_R512.npy'

    batch_visualize_FandB_R64(np_file_dir, np_file_suffix)
