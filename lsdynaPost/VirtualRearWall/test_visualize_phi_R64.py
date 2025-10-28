import os
import numpy as np
import matplotlib.pyplot as plt

def visualize_one_phi_R64(np_file):
    """
    加载并可视化单个 BumperX_dpY_vpZ_dcEk_R64.npy 文件。
    其数据形状应为 [64, height, width]:
      - data[i, :, :] 表示第 i 个旋转得到的图像。
    """
    if not os.path.isfile(np_file):
        print(f"[Error] File not found: {np_file}")
        return

    data = np.load(np_file)  # shape: (64, H, W)
    print(f"[Info] Loaded data from: {np_file}")
    print(f"[Info] Data shape: {data.shape}")  # 期望 (64, H, W)

    # 简单形状检查
    if len(data.shape) != 3:
        print(f"[Warning] Data shape not as expected [64, H, W].")

    num_rotations = data.shape[0]  # 通常是 64
    height = data.shape[1]
    width  = data.shape[2]

    # 如果 num_rotations != 64，可以根据实际情况决定如何可视化。
    if num_rotations < 64:
        print(f"[Warning] The file has only {num_rotations} rotations, less than 64.")

    # ============== 可视化 64 张旋转图像 ==============
    fig, axes = plt.subplots(8, 8, figsize=(12, 12))
    fig.suptitle(f"Rotations - {os.path.basename(np_file)}")

    for i in range(min(num_rotations, 64)):
        row = i // 8
        col = i % 8
        ax = axes[row][col]
        ax.imshow(data[i, :, :], cmap='jet', origin='lower')
        ax.set_title(f"Rot #{i}")
        ax.axis('off')

    plt.tight_layout()
    plt.show()


def batch_visualize_phi_R64(np_file_dir):
    """
    批量遍历 np_file_dir 下所有匹配 '*_dcEk_R64.npy' 的文件，
    依次调用可视化函数进行查看。
    """
    if not os.path.isdir(np_file_dir):
        print(f"[Error] {np_file_dir} is not a valid directory.")
        return

    # 收集所有符合要求的文件
    file_list = [
        f for f in os.listdir(np_file_dir)
        if f.endswith("_dcEk_R64.npy")
    ]
    file_list.sort()  # 如需按名称排序

    if len(file_list) == 0:
        print(f"[Warning] No '*_dcEk_R64.npy' files found in {np_file_dir}")
        return

    print(f"[Info] Found {len(file_list)} files to visualize.\n")

    # 遍历并可视化
    for filename in file_list:
        np_file = os.path.join(np_file_dir, filename)
        visualize_one_phi_R64(np_file)
        # 每次可视化都会弹出一个窗口(8x8 子图)。若文件很多，可考虑更改逻辑或自动保存图像。


if __name__ == "__main__":
    """
    示例主函数：批量检验某个目录下的所有 *_dcEk_R64.npy 文件。
    """
    # ========== 根据需要修改以下路径 ==========
    np_file_dir = r"D:\SIMULATION\p01_DebrisCloudDamageDataBase\DataBase\DataBase_DebrisCloudEk"  # 存放 BumperX_dpY_vpZ_dcEk_R64.npy 文件的目录

    batch_visualize_phi_R64(np_file_dir)
