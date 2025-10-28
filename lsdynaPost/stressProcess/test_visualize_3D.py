import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # 导入3D绘图工具
from matplotlib.colors import LightSource


def visualize_one_FandB_R64(np_file):
    """
    加载并可视化单个 BumperX_dpY_vpZ_FandB_maxVMStress_R64.npy 文件。
    data.shape 应为 [64, 2, height, width]：
        - data[i, 0, :, :] 表示第 i 个旋转得到的前层图像
        - data[i, 1, :, :] 表示第 i 个旋转得到的后层图像
    每次随机选择一组数据进行3D可视化，显示效果模拟三维激光轮廓仪扫描：
        - 使用光照阴影效果（LightSource）生成立体感
        - 底部添加轮廓线
        - z轴范围为数据最大值的35倍
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

    num_rotations = data.shape[0]  # 通常为 64
    height = data.shape[2]
    width = data.shape[3]

    if num_rotations < 64:
        print(f"[Warning] The file has only {num_rotations} rotations, less than 64.")

    # 随机选择一组旋转数据
    rotation_idx = np.random.randint(0, min(num_rotations, 64))
    print(f"[Info] Randomly selected rotation: {rotation_idx}")

    # 构建坐标网格，用于3D表面图绘制
    x = np.arange(width)
    y = np.arange(height)
    X, Y = np.meshgrid(x, y)

    # 创建图形窗口，左右两个子图分别展示前层和后层数据的3D图
    fig = plt.figure(figsize=(14, 6))

    # 设置一个光源，模拟激光扫描的光照效果
    ls = LightSource(azdeg=315, altdeg=45)

    # 前层数据3D表面图
    ax1 = fig.add_subplot(121, projection='3d')
    front_data = data[rotation_idx, 0, :, :]
    # 生成阴影色彩
    rgb_front = ls.shade(front_data, cmap=plt.cm.Blues, vert_exag=1, blend_mode='soft')
    surf1 = ax1.plot_surface(X, Y, front_data, rstride=1, cstride=1,
                             facecolors=rgb_front, linewidth=0, antialiased=False)
    # 添加底部轮廓线
    ax1.contour(X, Y, front_data, zdir='z', offset=0, cmap=plt.cm.Greys)
    ax1.set_title(f"Front Channel - Rotation #{rotation_idx}")
    ax1.set_xlabel("X")
    ax1.set_ylabel("Y")
    ax1.set_zlabel("Value (Height)")
    # 设置 z 轴显示范围为最大值的35倍
    max_z_front = np.max(front_data)
    ax1.set_zlim(0, 10 * max_z_front)
    # 调整视角：仰视角和方位角，可根据需要微调
    ax1.view_init(elev=45, azim=-120)
    fig.colorbar(surf1, ax=ax1, shrink=0.5, aspect=10)

    # 后层数据3D表面图
    ax2 = fig.add_subplot(122, projection='3d')
    back_data = data[rotation_idx, 1, :, :]
    rgb_back = ls.shade(back_data, cmap=plt.cm.Blues, vert_exag=1, blend_mode='soft')
    surf2 = ax2.plot_surface(X, Y, back_data, rstride=1, cstride=1,
                             facecolors=rgb_back, linewidth=0, antialiased=False)
    ax2.contour(X, Y, back_data, zdir='z', offset=0, cmap=plt.cm.Greys)
    ax2.set_title(f"Back Channel - Rotation #{rotation_idx}")
    ax2.set_xlabel("X")
    ax2.set_ylabel("Y")
    ax2.set_zlabel("Value (Height)")
    max_z_back = np.max(back_data)
    ax2.set_zlim(0, 10 * max_z_back)
    ax2.view_init(elev=45, azim=-120)
    fig.colorbar(surf2, ax=ax2, shrink=0.5, aspect=10)

    plt.tight_layout()
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
    file_list.sort()  # 按名称排序

    if len(file_list) == 0:
        print(f"[Warning] No '*{np_file_suffix}' files found in {np_file_dir}")
        return

    print(f"[Info] Found {len(file_list)} files to visualize.\n")

    # 遍历并可视化
    for filename in file_list:
        np_file = os.path.join(np_file_dir, filename)
        visualize_one_FandB_R64(np_file)
        # 每次可视化都会弹出一个窗口（包含前层、后层的3D图）。
        # 根据需要，可以考虑自动关闭 plt.show() 或将所有子图绘制到同一窗口。


if __name__ == "__main__":
    """
    示例主函数：批量检验某个目录下的所有 FandB_maxVMStress_R64 文件。
    """
    # ========== 根据需要修改以下路径 ==========
    # np_file_dir = r"D:\SIMULATION\p01_DebrisCloudDamageDataBase\DataBase\DataBase_VonMisesStress"  # 存放文件的目录
    # np_file_suffix = 'FandB_maxVMStress_R64.npy'
    np_file_dir = r"D:\SIMULATION\p01_DebrisCloudDamageDataBase\DataBase\DataBase_EffectivePlasticStrain"  # 另一个文件目录示例
    np_file_suffix = 'FandB_maxEPStrain_R64.npy'

    batch_visualize_FandB_R64(np_file_dir, np_file_suffix)
