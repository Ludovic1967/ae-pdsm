import numpy as np
import matplotlib.pyplot as plt
import gc
from scipy.optimize import differential_evolution
from scipy.ndimage import gaussian_filter
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import csv
from lsreader import D3plotReader, DataType as dt

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
# =======================
# 高斯叠加改为网格卷积版本
# =======================
def generate_energy_field(points, ek_values, grid_size=100, xlim=(-5, 5), ylim=(-5, 5)):
    x = np.linspace(xlim[0], xlim[1], grid_size)
    y = np.linspace(ylim[0], ylim[1], grid_size)
    X, Y = np.meshgrid(x, y)

    field = np.zeros_like(X, dtype=np.float32)
    x_idx = np.clip(((points[:, 0] - xlim[0]) / (xlim[1] - xlim[0]) * (grid_size - 1)).astype(int), 0, grid_size - 1)
    y_idx = np.clip(((points[:, 1] - ylim[0]) / (ylim[1] - ylim[0]) * (grid_size - 1)).astype(int), 0, grid_size - 1)

    for xi, yi, ek in zip(x_idx, y_idx, ek_values):
        field[yi, xi] += ek

    return X, Y, field


# =======================
# surface评估函数
# =======================
def eval_surface(xyz, D_values, X, Y, base_field, n_eta, sigma):
    blurred_field = gaussian_filter(base_field * n_eta, sigma=sigma)
    interp_z = interpolate_field(X, Y, blurred_field, xyz[:, :2])
    mask = xyz[:, 2] < interp_z
    return -D_values[mask].sum()


# =======================
# 简单双线性插值器
# =======================
def interpolate_field(X, Y, Z, query_points):
    x = X[0, :]
    y = Y[:, 0]
    xi = np.clip(np.searchsorted(x, query_points[:, 0]) - 1, 0, len(x) - 2)
    yi = np.clip(np.searchsorted(y, query_points[:, 1]) - 1, 0, len(y) - 2)

    x1, x2 = x[xi], x[xi + 1]
    y1, y2 = y[yi], y[yi + 1]

    Q11 = Z[yi, xi]
    Q12 = Z[yi + 1, xi]
    Q21 = Z[yi, xi + 1]
    Q22 = Z[yi + 1, xi + 1]

    xq, yq = query_points[:, 0], query_points[:, 1]
    denom = (x2 - x1) * (y2 - y1)
    denom = np.where(denom == 0, 1e-6, denom)

    interp = (Q11 * (x2 - xq) * (y2 - yq) +
              Q21 * (xq - x1) * (y2 - yq) +
              Q12 * (x2 - xq) * (yq - y1) +
              Q22 * (xq - x1) * (yq - y1)) / denom
    return interp


# =======================
# 单case处理函数
# =======================
def process_single_case_fast(element_coords, D_values, points, ek_values, case_id, output_dir=None, draw_fig=False):
    if D_values.ndim == 2:
        D_values = D_values[:, 0]

    max_points = 8000
    if len(points) > max_points:
        indices = np.random.choice(len(points), max_points, replace=False)
        points = points[indices]
        ek_values = ek_values[indices]

    X, Y, base_field = generate_energy_field(points, ek_values)

    def objective(params):
        n_eta, sigma = params
        return eval_surface(element_coords, D_values, X, Y, base_field, n_eta, sigma)

    bounds = [(0.1, 1.0), (0.5, 5.0)]
    result = differential_evolution(objective, bounds, polish=True)
    optimal_n_eta, optimal_sigma = result.x
    max_D_sum = -result.fun

    if draw_fig:
        blurred_field = gaussian_filter(base_field * optimal_n_eta, sigma=optimal_sigma)
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(X, Y, blurred_field, cmap='viridis', alpha=0.8, edgecolor='none')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(f'{case_id}\nn_eta={optimal_n_eta:.3f}, sigma={optimal_sigma:.3f}, MaxD={max_D_sum:.2f}')
        plt.tight_layout()

        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            fig_path = os.path.join(output_dir, f'{case_id}_surface.png')
            plt.savefig(fig_path, dpi=300)
            print(f"[Saved] {fig_path}")
            plt.close()
        else:
            plt.show()

    return {
        'case_id': case_id,
        'optimal_n_eta': optimal_n_eta,
        'optimal_sigma': optimal_sigma,
        'max_D_sum': max_D_sum,
    }


# =======================
# 批量处理函数 (多线程 + 进度条)
# =======================
def batch_process_fast(case_list, output_dir=None, max_workers=4, draw_fig=False):
    results = []

    def worker(case):
        element_coords, D_values, points, ek_values, case_id = case
        try:
            return process_single_case_fast(element_coords, D_values, points, ek_values, case_id, output_dir, draw_fig)
        except Exception as e:
            print(f"[Error] {case_id}: {e}")
            return None

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(worker, case): case for case in case_list}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing cases"):
            result = future.result()
            if result:
                results.append(result)

    return results

# =======================
# 示例如何调用（批量）
# =======================
def batch_process_from_d3plot(main_path, output_csv_path, output_dir=None, max_workers=4, draw_fig=False):
    cases = []
    total_case_count = 0

    for bumper_folder in os.listdir(main_path):
        bumper_path = os.path.join(main_path, bumper_folder)
        if not os.path.isdir(bumper_path):
            continue

        for dp_folder in os.listdir(bumper_path):
            dp_path = os.path.join(bumper_path, dp_folder)
            if not os.path.isdir(dp_path):
                continue

            for vp_folder in os.listdir(dp_path):
                vp_path = os.path.join(dp_path, vp_folder)
                if not os.path.isdir(vp_path):
                    continue

                d3plot182_file = os.path.join(vp_path, 'd3plot182')
                if not os.path.isfile(d3plot182_file):
                    continue

                try:
                    bumper_num = ''.join(filter(str.isdigit, bumper_folder))
                    dp_num = ''.join(filter(str.isdigit, dp_folder))
                    vp_num = ''.join(filter(str.isdigit, vp_folder))
                    vp_num_int = int(vp_num)
                    dp_num_int = int(dp_num)
                    if vp_num_int < 300 or dp_num_int > 5:
                        continue
                except ValueError:
                    continue

                try:
                    d3plot_file = os.path.join(vp_path, 'd3plot')
                    dr = D3plotReader(d3plot_file)
                except Exception as e:
                    print(f"读取d3plot失败: {e}")
                    continue

                try:
                    num_states = dr.get_data(dt.D3P_NUM_STATES)
                    element_D = dr.get_data(dt.D3P_SOLID_HISTORY_VAR, ist=num_states - 1, ipt=2, ihv=5, ask_for_numpy_array=True)
                    element = dr.get_data(dt.D3P_SOLID_ELEMENT_CENTROID, ist=num_states - 1, ipart=2, ask_for_numpy_array=True)
                except Exception as e:
                    print(f"提取element/D失败: {e}")
                    continue

                points, ek_values = extract_points_ek_values(dr)
                if points is None or ek_values is None:
                    continue

                case_id = f"Bumper{bumper_num}_dp{dp_num}_vp{vp_num}"
                cases.append((element, element_D, points, ek_values, case_id))
                total_case_count += 1

                del dr
                gc.collect()

    print(f"共收集到 {total_case_count} 个可处理case，开始批量拟合...")

    summary = batch_process_fast(cases, output_dir=output_dir, max_workers=max_workers, draw_fig=draw_fig)

    with open(output_csv_path, mode='w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=['case_id', 'optimal_n_eta', 'optimal_sigma', 'max_D_sum'])
        writer.writeheader()
        writer.writerows(summary)

    print(f"[完成] 所有参数拟合结果已保存至: {output_csv_path}")

# =======================
# 示例调用（自动提取d3plot批量处理）
# =======================
main_path = r'\\DESKTOP-SVFNCL2\database\MeshSizeCompare\sph2+RP'
output_csv = 'fit_results.csv'
batch_process_from_d3plot(main_path, output_csv, output_dir='results', max_workers=8, draw_fig=True)
