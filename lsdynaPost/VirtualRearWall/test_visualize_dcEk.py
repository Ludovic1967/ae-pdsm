import os
import numpy as np
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import re

def extract_parameters(filename):
    """
    从文件名中提取 Bumper, dp, vp 参数。

    示例文件名: Bumper2_dp4_vp189_dcEk.npy
    返回: {'bumper': '2', 'dp': '4', 'vp': '189'}
    """
    pattern = r'Bumper(\d+)_dp(\d+)_vp(\d+)_dcEk\.npy'
    match = re.match(pattern, filename)
    if match:
        return {
            'bumper': match.group(1),
            'dp': match.group(2),
            'vp': match.group(3)
        }
    else:
        return None

def visualize_phi(phi, output_image_path, title='Field φ(x,y)'):
    """
    可视化 Phi(x,y) 并保存图像。

    参数:
        phi (np.ndarray): Phi 数据，二维数组。
        output_image_path (str): 保存图像的路径。
        title (str): 图像标题。
    """
    plt.figure(figsize=(10, 9))
    # 使用对数色彩映射以更好地显示数据范围较大的情况
    # norm = mcolors.LogNorm(vmin=phi[phi > 0].min(), vmax=phi.max()) if np.any(phi > 0) else None
    # 使用 pcolormesh 进行绘图，假设 X 和 Y 范围与生成 Phi 时相同
    x_min, x_max = -5, 5
    y_min, y_max = -5, 5
    num_points = phi.shape[0]
    x_values = np.linspace(x_min, x_max, num_points)
    y_values = np.linspace(y_min, y_max, num_points)
    X, Y = np.meshgrid(x_values, y_values)

    # 绘制 Phi
    # mesh = plt.pcolormesh(X, Y, phi, cmap='jet', vmin=0, vmax=0.005, shading='auto')
    mesh = plt.pcolormesh(X, Y, phi, cmap='jet', shading='auto') #【不限制vmin vmax】
    plt.colorbar(mesh, label='φ(x,y)')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(title)
    plt.tight_layout()

    # 保存图像
    plt.savefig(output_image_path, dpi=300)
    plt.close()
    print(f"已保存图像到 {output_image_path}")

def main():
    # 设置包含 Phi 文件的目录（与之前的输出目录一致）
    phi_directory = r'D:\SIMULATION\p01_DebrisCloudDamageDataBase\originalData\original_DebrisCloudEk_npyFiles'  # 请根据实际情况修改

    # 设置可视化输出的目录
    visualization_output_dir = os.path.join(phi_directory, 'Visualizations_parameter_calibration')
    os.makedirs(visualization_output_dir, exist_ok=True)

    # 遍历目录中的所有 .npy 文件
    for filename in os.listdir(phi_directory):
        if filename.endswith('_dcEk.npy'):
            params = extract_parameters(filename)
            if params is None:
                print(f"跳过不符合命名规则的文件: {filename}")
                continue

            bumper = params['bumper']
            dp = params['dp']
            vp = params['vp']

            # 构建 Phi 文件的完整路径
            phi_filepath = os.path.join(phi_directory, filename)

            try:
                # 加载 Phi 数据
                Phi = np.load(phi_filepath)
                print(f"加载 Phi 文件: {phi_filepath}, 形状: {Phi.shape}")

                # 构建图像保存路径，例如 Bumper2_dp4_vp189_phi.stressProcess
                image_filename = f'Bumper{bumper}_dp{dp}_vp{vp}_phi.png'
                image_filepath = os.path.join(visualization_output_dir, image_filename)

                # 定义图像标题
                title = f'Field φ(x,y) - Bumper{bumper} dp{dp} vp{vp}'

                # 可视化并保存图像
                visualize_phi(Phi, image_filepath, title=title)

            except Exception as e:
                print(f"无法处理文件 {filename}: {e}")

    print("所有 Phi 文件的可视化已完成。")

if __name__ == "__main__":
    main()
