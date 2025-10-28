import csv
import gc
from lsreader import D3plotReader, DataType as dt
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brute
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import RegularGridInterpolator
import os

def extract_points_ek_values(dr, z_virtual_wall=10.0):
    try:
        num_states = dr.get_data(dt.D3P_NUM_STATES)
        order_state = 25  # 默认膨胀稳定帧

        node_SPH_mass = dr.get_data(dt.D3P_SPH_MASS, ist=order_state, ask_for_numpy_array=True)
        node_initial_coor = np.concatenate([
            dr.get_data(dt.D3P_NODE_COORDINATES, ist=0, ipart=0, ask_for_numpy_array=True),
            dr.get_data(dt.D3P_NODE_COORDINATES, ist=0, ipart=1, ask_for_numpy_array=True)
        ])
        node_end_coor = np.concatenate([
            dr.get_data(dt.D3P_NODE_COORDINATES, ist=order_state, ipart=0, ask_for_numpy_array=True),
            dr.get_data(dt.D3P_NODE_COORDINATES, ist=order_state, ipart=1, ask_for_numpy_array=True)
        ])
        node_end_vel = np.concatenate([
            dr.get_data(dt.D3P_NODE_VELOCITIES, ist=order_state, ipart=0, ask_for_numpy_array=True),
            dr.get_data(dt.D3P_NODE_VELOCITIES, ist=order_state, ipart=1, ask_for_numpy_array=True)
        ])
        node_SPH_neighbors = dr.get_data(dt.D3P_SPH_NUMBER_OF_PARTICLE_NEIGHBORS,
                                         ist=order_state, ask_for_numpy_array=True)
    except Exception as e:
        print(f"  [错误] 数据提取失败: {e}")
        return None, None

    Ek_end = 0.5 * node_SPH_mass * np.sum(node_end_vel**2, axis=1)

    valid_mask = node_end_vel[:, 2] != 0
    t_res = np.full_like(Ek_end, np.nan)
    t_res[valid_mask] = (z_virtual_wall - node_end_coor[valid_mask, 2]) / node_end_vel[valid_mask, 2]

    node_rearwall_coor = node_end_coor + node_end_vel * t_res[:, np.newaxis]

    # 括号修复后的条件筛选
    mask = (
        ((node_end_coor[:, 2] - node_initial_coor[:, 2]) > 0) &
        ((node_rearwall_coor[:, 0] > -5) & (node_rearwall_coor[:, 0] < 5)) &
        ((node_rearwall_coor[:, 1] > -5) & (node_rearwall_coor[:, 1] < 5)) &
        ((t_res > 0) & (t_res < 1000)) &
        (node_SPH_neighbors != 0)
    )

    points = node_rearwall_coor[mask, :2]
    ek_values = Ek_end[mask]

    if len(points) == 0:
        print("  [跳过] 没有符合条件的粒子点")
        return None, None

    return points.astype(np.float32), ek_values.astype(np.float32)


def gaussian_surface(x, y, n_eta, sigma, points, ek_values, chunk_size=200):
    """
    内存优化版本：使用高斯核密度求解二维能量场 Φ(x, y)
    :param x: 所有 element 的 x 坐标 (N,)
    :param y: 所有 element 的 y 坐标 (N,)
    :param n_eta: 核权重参数
    :param sigma: 高斯宽度
    :param points: 投影粒子点坐标 (M, 2)
    :param ek_values: 粒子动能 (M,)
    :param chunk_size: 分块处理数量
    :return: z 值数组 (N,)
    """
    z = np.zeros_like(x, dtype=np.float32) + 10
    factor = (n_eta * ek_values) / (2 * np.pi * sigma**2)

    for start in range(0, len(points), chunk_size):
        end = min(start + chunk_size, len(points))
        dx = x[:, None] - points[start:end, 0]  # shape: (N, chunk)
        dy = y[:, None] - points[start:end, 1]
        dist2 = dx**2 + dy**2
        z += np.sum(factor[start:end] * np.exp(-dist2 / (2 * sigma**2)), axis=1)

    return z




# points = node_VirutalRearWall_coor[node_forword_mask, 0:2]
# ek_values = Ek_end[node_forword_mask]
# 这俩一一对应
def objective(params, coords, D, points, ek_values):
    n_eta, sigma = params
    n_eta = int(round(n_eta))
    x, y, z_real = coords[:, 0], coords[:, 1], coords[:, 2]
    z_surface = gaussian_surface(x, y, n_eta, sigma, points, ek_values)
    mask = z_real < z_surface
    return -D[mask].sum()

def process_single_case(element, D, points, ek_values, case_id, output_dir=None):
    element_coords = element
    D_values = D
    if D_values.ndim == 2:
        D_values = D_values[:, 0]

    param_ranges = (
        slice(0.1, 0.4, 0.5),       # n_eta  slice(start, stop, step)
        slice(0.5, 5, 10),   # sigma
    )
    # param_ranges = (
    #     0.4,  # n_eta  slice(start, stop, step)
    #     1.0,  # sigma
    # )
    result = brute(objective, param_ranges, args=(element_coords, D_values, points, ek_values), finish=None)
    optimal_n_eta = result[0]
    optimal_sigma = result[1]
    max_D_sum = -objective((optimal_n_eta, optimal_sigma), element_coords, D_values, points, ek_values)

    # 构造最终 surface 用于可视化
    x, y, z_real = element_coords[:, 0], element_coords[:, 1], element_coords[:, 2]
    z_surface = gaussian_surface(x, y, optimal_n_eta, optimal_sigma, points, ek_values)

    mask = z_real < z_surface

    # ========= 可视化优化 =========
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    # 曲面可视化部分（基于稀疏网格）
    grid_sample = 100  # 控制平滑度
    x_lin = np.linspace(np.min(x), np.max(x), grid_sample)
    y_lin = np.linspace(np.min(y), np.max(y), grid_sample)
    X, Y = np.meshgrid(x_lin, y_lin)
    Z = gaussian_surface(X.ravel(), Y.ravel(), optimal_n_eta, optimal_sigma)
    Z = Z.reshape(X.shape)

    ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.5, edgecolor='none')

    # 绘制采样后的 element 点
    max_plot = 3000
    indices = np.random.choice(len(x), size=min(len(x), max_plot), replace=False)

    x_s, y_s, z_s = x[indices], y[indices], z_real[indices]
    mask_s = z_s < gaussian_surface(x_s, y_s, optimal_n_eta, optimal_sigma)

    # ax.scatter(x_s[~mask_s], y_s[~mask_s], z_s[~mask_s], c='gray', alpha=0.2, s=2, label='Above')
    # ax.scatter(x_s[mask_s], y_s[mask_s], z_s[mask_s], c='red', alpha=0.4, s=4, label='Below')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f'n_eta={optimal_n_eta}, sigma={optimal_sigma:.2f}, Max D Sum={max_D_sum:.2f}')
    ax.legend()
    plt.tight_layout()

    if output_dir:
        # os.makedirs(output_dir, exist_ok=True)
        # base_name = os.path.basename(D_file).replace('.npy', '')
        # fig_path = os.path.join(output_dir, f'{base_name}_3D_surface.png')
        # plt.savefig(fig_path, dpi=300)
        # print(f"[Saved] {fig_path}")
        plt.close()
    else:
        plt.show()

    return {
        'case_id': case_id,
        'optimal_n_eta': optimal_n_eta,
        'optimal_sigma': optimal_sigma,
        'max_D_sum': max_D_sum,
    }

# def process_single_case(element, D, points, ek_values, case_id, output_dir=None):
#     element_coords = element
#     D_values = D
#     if D_values.ndim == 2:
#         D_values = D_values[:, 0]
#
#     param_ranges = (
#         slice(0, 200, 1),       # n_eta: 范围自己根据数据调，这里给示例
#         slice(1e-4, 1, 0.01),  # sigma: 高斯核宽度
#     )
#     result = brute(objective, param_ranges, args=(element_coords, D_values, points, ek_values), finish=None)
#     optimal_n_eta = result[0]
#     optimal_sigma = result[1]
#     max_D_sum = -objective((optimal_n_eta, optimal_sigma), element_coords, D_values, points, ek_values)
#
#     x, y, z_real = element_coords[:, 0], element_coords[:, 1], element_coords[:, 2]
#     # 设置网格范围
#     x_min, x_max = -5, 5
#     y_min, y_max = -5, 5
#     grid_resolution = 0.02  # 网格精度
#
#     x_values = np.arange(x_min, x_max, grid_resolution)
#     y_values = np.arange(y_min, y_max, grid_resolution)
#     X, Y = np.meshgrid(x_values, y_values)
#
#     z_surface = gaussian_surface(X, Y, optimal_n_eta, optimal_sigma, points, ek_values)
#
#     # 创建插值器
#     interpolator = RegularGridInterpolator((y_values, x_values), z_surface, bounds_error=False, fill_value=0.0)
#
#     # 提取每个 element 的 x, y
#     x = element_coords[:, 0]
#     y = element_coords[:, 1]
#     z_real = element_coords[:, 2]
#
#     # 组合坐标 (注意顺序: y, x)
#     query_points = np.stack([y, x], axis=-1)
#     z_interp = interpolator(query_points)
#
#     # 进行比较
#     mask = z_real < z_interp
#
#     fig = plt.figure(figsize=(12, 8))
#     ax = fig.add_subplot(111, projection='3d')
#     ax.scatter(x[~mask], y[~mask], z_real[~mask], c='gray', alpha=0.3, label='Above surface')
#     ax.scatter(x[mask], y[mask], z_real[mask], c='red', alpha=0.6, label='Below surface')
#     ax.plot_surface(X, Y, z_surface, cmap='viridis', alpha=0.4)  # 更漂亮的能量场面
#
#     ax.set_xlabel('X')
#     ax.set_ylabel('Y')
#     ax.set_zlabel('Z')
#     title = f'n_eta={optimal_n_eta}, sigma={optimal_sigma:.2f}, Max D Sum={max_D_sum:.2f}'
#     ax.set_title(title)
#     ax.legend()
#     plt.tight_layout()
#     plt.show()
#
#     # if output_dir:
#     #     os.makedirs(output_dir, exist_ok=True)
#     #     base_name = os.path.basename(D_file).replace('.npy', '')
#     #     fig_path = os.path.join(output_dir, f'{base_name}_3D_surface.png')
#     #     plt.savefig(fig_path, dpi=300)
#     #     print(f"[Saved] {fig_path}")
#     #     plt.close()
#     # else:
#
#     return {
#         'case_id': case_id,
#         'optimal_n_eta': optimal_n_eta,
#         'optimal_sigma': optimal_sigma,
#         'max_D_sum': max_D_sum,
#     }

# def batch_process(element_dir, D_dir, output_dir):
#     results = []
#     D_files = [f for f in os.listdir(D_dir) if f.endswith('.npy')]
#     for D_file in D_files:
#         element_file = os.path.join(element_dir, D_file.replace('elementLast_D', 'element_coords'))
#         D_file_path = os.path.join(D_dir, D_file)
#         if not os.path.exists(element_file):
#             print(f"[Skip] Missing element file for {D_file}")
#             continue
#         print(f"[Processing] {D_file}")
#         result = process_single_case(element_file, D_file_path, output_dir)
#         results.append(result)
#     return results

# 定义主路径
main_path = r'\\DESKTOP-SVFNCL2\database\MeshSizeCompare\sph2+RP'
# main_path = r'D:\SIMULATION\p01_DebrisCloudDamageDataBase' # 【回家测试用】

# 定义保存结果的路径
# D_output_path = r'D:\SIMULATION\p01_DebrisCloudDamageDataBase\originalData\original_elementValue_npyFiles\D_Data'
#
# # # 如果保存路径不存在，则创建
# if not os.path.exists(D_output_path):
#     os.makedirs(D_output_path)
# # if not os.path.exists(EPstrain_output_path):
# #     os.makedirs(EPstrain_output_path)
#
#
# existing_D_files = set(os.listdir(D_output_path))
results = []
total_case_count = 0
processed_count = 0

# 遍历主路径下的所有 Bumper 文件夹
for bumper_folder in os.listdir(main_path):
    bumper_path = os.path.join(main_path, bumper_folder)
    if not os.path.isdir(bumper_path):
        continue  # 跳过非文件夹

    # 遍历 Bumper 文件夹下的所有 dp 文件夹
    for dp_folder in os.listdir(bumper_path):
        dp_path = os.path.join(bumper_path, dp_folder)
        if not os.path.isdir(dp_path):
            continue  # 跳过非文件夹

        # 遍历 dp 文件夹下的所有 vp 文件夹
        for vp_folder in os.listdir(dp_path):
            vp_path = os.path.join(dp_path, vp_folder)

            total_case_count += 1  # 统计总case数量
            if not os.path.isdir(vp_path):
                continue  # 跳过非文件夹

            # 检查当前 vp 文件夹中是否存在 d3plot182 文件
            d3plot182_file = os.path.join(vp_path, 'd3plot182')
            if not os.path.isfile(d3plot182_file):
                continue  # 不存在 d3plot182 文件，跳过

            # 提取 Bumper、dp、vp 的编号，用于后续文件命名
            try:
                bumper_num = ''.join(filter(str.isdigit, bumper_folder))
                dp_num = ''.join(filter(str.isdigit, dp_folder))
                vp_num = ''.join(filter(str.isdigit, vp_folder))
            except Exception as e:
                print(f"Error extracting numbers from folder names: {e}")
                continue  # 跳过当前文件夹

            try:
                vp_num_int = int(vp_num)
                dp_num_int = int(dp_num)
                if vp_num_int < 300 or dp_num_int > 5:
                    print(f"[跳过] 不满足处理条件: vp_num={vp_num_int} < 300 或 dp_num={dp_num_int} > 5")
                    continue
            except ValueError as e:
                print(f"[跳过] 无法解析数字: {e}")
                continue

            # 构造输出文件名
            # VMstress_output_filename = f"Bumper{bumper_num}_dp{dp_num}_vp{vp_num}_elementMax_VMStress.npy"
            # EPstrain_output_filename = f"Bumper{bumper_num}_dp{dp_num}_vp{vp_num}_elementMax_EPStrain.npy"
            # D_output_filename = f"Bumper{bumper_num}_dp{dp_num}_vp{vp_num}_elementLast_D.npy"

            # 如果输出文件已经存在，则跳过处理
            # if (D_output_filename in existing_D_files):
            #         # and (damageM_output_filename in existing_damageM_files)):
            #     print(f"Skip already processed:  {D_output_filename}")
            #     continue

            print(f"Processing folder: {vp_path}")

            # 读取 d3plot 文件（与 d3plot182 放在同一路径，名字为 d3plot）
            case_id = f"Bumper{bumper_num}_dp{dp_num}_vp{vp_num}"

            try:
                d3plot_file = os.path.join(vp_path, 'd3plot')
                dr = D3plotReader(d3plot_file)
            except Exception as e:
                print(f"读取 d3plot 失败: {e}")
                continue

            try:
                num_states = dr.get_data(dt.D3P_NUM_STATES)
                element_D = dr.get_data(dt.D3P_SOLID_HISTORY_VAR, ist=num_states - 1, ipt=2, ihv=5,
                                        ask_for_numpy_array=True)
                element = dr.get_data(dt.D3P_SOLID_ELEMENT_CENTROID, ist=num_states - 1, ipart=2,
                                      ask_for_numpy_array=True)
                print(f"[✓] 成功提取 element 和 D 值")
            except Exception as e:
                print(f"[×] 提取 element/D 失败: {e}")
                continue

            points, ek_values = extract_points_ek_values(dr)

            if points is None or ek_values is None:
                print(f"[×] 无有效粒子点，跳过 {case_id}")
                continue  # 如果提取失败，跳过
            print(f"[✓] 有效粒子点数：{len(points)}，准备进行参数拟合")

            result = process_single_case(element, element_D, points, ek_values, case_id)
            print(f"[✓] 拟合完成：n_eta={result['optimal_n_eta']}, sigma={result['optimal_sigma']:.3f}")
            results.append(result)
            processed_count += 1
            print(f"[进度] 当前处理完成：{processed_count}/{total_case_count} case\n")

            # print(element.shape)
            #
            # # 打印结果信息
            # print(f"element_D.shape: {element_D.shape}")
            # print(f"最小值: {np.min(element_D)}, 最大值: {np.max(element_D)}")
            #
            # D_output_file = os.path.join(D_output_path, D_output_filename)
            #
            # # 保存结果
            # np.save(D_output_file, element_D)
            # print(f"Saved D_data to {D_output_file}\n")

            # 释放内存
            del element_D
            del dr
            gc.collect()

output_csv = os.path.join(os.getcwd(), 'parameter_fitting_results.csv')

with open(output_csv, mode='w', newline='') as file:
    writer = csv.DictWriter(file, fieldnames=['case_id', 'optimal_n_eta', 'optimal_sigma', 'max_D_sum'])
    writer.writeheader()
    writer.writerows(results)

print(f"\n[完成] 所有参数拟合结果已保存至：{output_csv}")

