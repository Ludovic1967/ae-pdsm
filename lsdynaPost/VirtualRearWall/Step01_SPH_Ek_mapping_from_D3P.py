import os
from lsreader import D3plotReader, DataType as dt
from scipy.ndimage import gaussian_filter
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from skimage.transform import rotate
import time
import random

# 需要处理的主路径
# main_path = r'\\DESKTOP-SVFNCL2\database\MeshSizeCompare\sph2+RP'
main_path = r'\\Desktop-svfncl2\backup\MeshSizeCompare\sph2+RP\evaluation' # 【eval】

# 输出结果保存的固定路径
# output_path = r'D:\SIMULATION\p01_DebrisCloudDamageDataBase\originalData\original_DebrisCloudEk_npyFiles'  # 请根据需要修改
output_path = r'D:\SIMULATION\p01_DebrisCloudDamageDataBase\originalData\eval\original_DebrisCloudEk_npyFiles'  # 请根据需要修改

# 确保输出路径存在
os.makedirs(output_path, exist_ok=True)

# 1) 首先检索 output_path 中已存在的文件（仅示例检索 .npy 文件）
existing_outputs = set(os.listdir(output_path))  # 将已存在的文件统一收集到集合中，方便后续判断

# 遍历主路径下所有三级文件夹
for bumper_folder in os.listdir(main_path):
    bumper_path = os.path.join(main_path, bumper_folder)
    if not os.path.isdir(bumper_path):
        continue  # 跳过非文件夹

    for dp_folder in os.listdir(bumper_path):
        dp_path = os.path.join(bumper_path, dp_folder)
        if not os.path.isdir(dp_path):
            continue  # 跳过非文件夹

        for vp_folder in os.listdir(dp_path):
            vp_path = os.path.join(dp_path, vp_folder)
            if not os.path.isdir(vp_path):
                continue  # 跳过非文件夹

            # 检查该文件夹是否包含 'd3plot182' 文件
            d3plot182_path = os.path.join(vp_path, 'd3plot182')
            if not os.path.isfile(d3plot182_path):
                continue  # 不满足条件，跳过

            # 在正式读取 d3plot 前，根据文件夹名称构造输出文件名
            bumper_num = ''.join(filter(str.isdigit, bumper_folder))
            dp_num = ''.join(filter(str.isdigit, dp_folder))
            vp_num = ''.join(filter(str.isdigit, vp_folder))
            output_filename = f'Bumper{bumper_num}_dp{dp_num}_vp{vp_num}_dcEk.npy'
            output_filepath = os.path.join(output_path, output_filename)

            # 2) 判断该输出文件是否已存在
            if output_filename in existing_outputs:
                print(f"【已存在】跳过处理: {output_filename}")
                continue  # 无需处理后续逻辑

            print(f"Processing folder: {vp_path}")

            # 读取 'd3plot' 文件
            d3plot = os.path.join(vp_path, 'd3plot')
            if not os.path.isfile(d3plot):
                print(f"  d3plot 文件不存在，跳过 {vp_path}")
                continue  # 如果 'd3plot' 文件不存在，跳过

            try:
                dr = D3plotReader(d3plot)
            except Exception as e:
                print(f"  无法读取 d3plot 文件: {e}")
                continue  # 读取失败，跳过

            num_states = dr.get_data(dt.D3P_NUM_STATES)  # 获取时态数量
            num_parts = dr.get_data(dt.D3P_NUM_PARTS)    # 获取part数量
            print(f"  num_states: {num_states}, num_parts: {num_parts}")

            # 读取所需数据
            order_state = round(num_states * 10 / 182)  # 碎片云膨胀稳定的时刻
            order_state = 10  # 碎片云膨胀稳定的时刻 【目前暂时正确输出d3plot都是182个】
            try:
                node_SPH_mass = dr.get_data(dt.D3P_SPH_MASS, ist=order_state, ask_for_numpy_array=True)
                node_initial_coor1 = dr.get_data(dt.D3P_NODE_COORDINATES, ist=0, ipart=0, ask_for_numpy_array=True)
                node_initial_coor2 = dr.get_data(dt.D3P_NODE_COORDINATES, ist=0, ipart=1, ask_for_numpy_array=True)
                node_initial_coor = np.concatenate((node_initial_coor1, node_initial_coor2))

                node_end_coor1 = dr.get_data(dt.D3P_NODE_COORDINATES, ist=order_state, ipart=0, ask_for_numpy_array=True)
                node_end_coor2 = dr.get_data(dt.D3P_NODE_COORDINATES, ist=order_state, ipart=1, ask_for_numpy_array=True)
                node_end_coor = np.concatenate((node_end_coor1, node_end_coor2))

                node_end_vel1 = dr.get_data(dt.D3P_NODE_VELOCITIES, ist=order_state, ipart=0, ask_for_numpy_array=True)
                node_end_vel2 = dr.get_data(dt.D3P_NODE_VELOCITIES, ist=order_state, ipart=1, ask_for_numpy_array=True)
                node_end_vel = np.concatenate((node_end_vel1, node_end_vel2))

                node_SPH_neighbors = dr.get_data(dt.D3P_SPH_NUMBER_OF_PARTICLE_NEIGHBORS,
                                                 ist=order_state, ask_for_numpy_array=True)
            except Exception as e:
                print(f"  数据读取失败: {e}")
                continue  # 数据读取失败，跳过

            # 假设虚拟后壁位置Z = 10.0（请根据实际情况修改此值）
            Z = 10.0

            # (1) 计算Ek_end、t_res和node_VirutalRearWall_coor
            node_end_vel2 = node_end_vel**2
            Ek_end = 0.5 * node_SPH_mass * np.sum(node_end_vel2, axis=1)

            valid_z_vel_mask = node_end_vel[:, 2] != 0.0
            t_res = np.full(node_end_vel.shape[0], np.nan)
            t_res[valid_z_vel_mask] = (Z - node_end_coor[valid_z_vel_mask, 2]) / node_end_vel[valid_z_vel_mask, 2]

            node_VirutalRearWall_coor = np.zeros_like(node_end_coor)
            node_VirutalRearWall_coor[:, 0] = node_end_coor[:, 0] + node_end_vel[:, 0] * t_res
            node_VirutalRearWall_coor[:, 1] = node_end_coor[:, 1] + node_end_vel[:, 1] * t_res
            node_VirutalRearWall_coor[:, 2] = node_end_coor[:, 2] + node_end_vel[:, 2] * t_res

            # (2) 筛选条件：
            condition_1 = (node_end_coor[:, 2] - node_initial_coor[:, 2]) > 0
            condition_2 = (node_VirutalRearWall_coor[:, 0] > -5) & (node_VirutalRearWall_coor[:, 0] < 5)
            condition_3 = (node_VirutalRearWall_coor[:, 1] > -5) & (node_VirutalRearWall_coor[:, 1] < 5)
            condition_4 = (t_res > 0) & (t_res < 5000)
            condition_5 = node_SPH_neighbors != 0

            node_forword_mask = condition_1 & condition_2 & condition_3 & condition_4 & condition_5
            # node_forword_mask = condition_2 & condition_3 & condition_4
            print(f"  Number of forward nodes: {np.sum(node_forword_mask)}")

            points = node_VirutalRearWall_coor[node_forword_mask, 0:2]
            ek_values = Ek_end[node_forword_mask]

            if points.size == 0:
                print("  无符合条件的点，跳过此文件夹")
                continue

            # 定义场函数参数
            sigma = 0.06
            n_eta = 6.93

            x_min, x_max = -5, 5
            y_min, y_max = -5, 5
            grid_resolution = 0.02
            num_points = round((x_max - x_min) / grid_resolution)
            print(f'  Grid resolution: {num_points}x{num_points}')

            x_values = np.linspace(x_min, x_max, num_points)
            y_values = np.linspace(y_min, y_max, num_points)
            X, Y = np.meshgrid(x_values, y_values)

            # 使用单精度浮点数以减少内存使用
            X = X.astype(np.float32)
            Y = Y.astype(np.float32)
            points = points.astype(np.float32)
            ek_values = ek_values.astype(np.float32)

            Phi = np.zeros((num_points, num_points), dtype=np.float32)
            chunk_size = 800

            for start in range(0, len(points), chunk_size):
                end = min(start + chunk_size, len(points))
                dx = X[:, :, None] - points[start:end, 0]
                dy = Y[:, :, None] - points[start:end, 1]
                dist2 = dx**2 + dy**2

                factor = (n_eta * ek_values[start:end]) / (2 * np.pi * sigma**2)
                f_values = factor * np.exp(-dist2 / (2 * sigma**2))

                Phi += np.sum(f_values, axis=2)
                del dx, dy, dist2, f_values

            # 可视化 φ(x,y)（可选）
            # plt.figure(figsize=(10,9))
            # plt.pcolormesh(X, Y, Phi, cmap='jet', shading='auto')  # coolwarm
            # plt.colorbar(label='φ(x,y)')
            # plt.xlabel('x')
            # plt.ylabel('y')
            # plt.title('Field φ(x,y)')
            # plt.tight_layout()
            # plt.show()

            # 保存Phi到固定路径下
            try:
                np.save(output_filepath, Phi)
                print(f"  Phi 已保存到 {output_filepath}")
            except Exception as e:
                print(f"  保存 Phi 失败: {e}")

print("所有文件夹处理完毕。")
