import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from scipy.ndimage import gaussian_filter
import gc
from lsreader import D3plotReader, DataType as dt

# 假设你的 extract_points_ek_values 已定义
from fit_n import extract_points_ek_values, generate_energy_field, interpolate_field

# =======================
# 验证函数
# =======================
def validate_fit(element_coords, D_values, points, ek_values, X, Y, base_field, optimal_n_eta, optimal_sigma, case_id, output_dir):
    blurred_field = gaussian_filter(base_field * optimal_n_eta, sigma=optimal_sigma)

    interp_z = interpolate_field(X, Y, blurred_field, element_coords[:, :2])
    mask = element_coords[:, 2] < interp_z

    coverage_ratio = np.sum(mask) / len(mask)
    covered_D_sum = D_values[mask].sum()
    total_D_sum = D_values.sum()

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, blurred_field, cmap='viridis', alpha=0.5, edgecolor='none')

    ax.scatter(element_coords[mask, 0], element_coords[mask, 1], element_coords[mask, 2], color='red', s=2, label='Below Surface')
    ax.scatter(element_coords[~mask, 0], element_coords[~mask, 1], element_coords[~mask, 2], color='blue', s=2, label='Above Surface')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f'{case_id}\nCoverage: {coverage_ratio:.1%}, D Covered: {covered_D_sum/total_D_sum:.1%}')
    ax.legend()
    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    fig_path = os.path.join(output_dir, f'{case_id}_validate.png')
    plt.savefig(fig_path, dpi=300)
    plt.close()
    print(f"[Saved] {fig_path}")

    return {
        'case_id': case_id,
        'coverage_ratio': coverage_ratio,
        'covered_D_ratio': covered_D_sum / total_D_sum
    }

# =======================
# 主脚本入口
# =======================
def validate_from_d3plot(
    csv_path,
    main_path,
    output_dir='validation_results'
):
    os.makedirs(output_dir, exist_ok=True)
    summary = []

    df = pd.read_csv(csv_path)
    print(f"共加载 {len(df)} 个case，开始验证...")

    for idx, row in df.iterrows():
        case_id = row['case_id']
        optimal_n_eta = row['optimal_n_eta']
        optimal_sigma = row['optimal_sigma']

        try:
            # 解析case_id
            bumper_num = case_id.split('_')[0].replace('Bumper', '')
            dp_num = case_id.split('_')[1].replace('dp', '')
            vp_num = case_id.split('_')[2].replace('vp', '')

            vp_path = os.path.join(main_path, f'Bumper{bumper_num}', f'dp{dp_num}', f'vp{vp_num}')
            d3plot_path = os.path.join(vp_path, 'd3plot')

            if not os.path.isfile(d3plot_path):
                print(f"[跳过] 找不到d3plot文件: {d3plot_path}")
                continue

            # 读取d3plot数据
            dr = D3plotReader(d3plot_path)

            num_states = dr.get_data(dt.D3P_NUM_STATES)
            element_D = dr.get_data(dt.D3P_SOLID_HISTORY_VAR, ist=num_states - 1, ipt=2, ihv=5, ask_for_numpy_array=True)
            element = dr.get_data(dt.D3P_SOLID_ELEMENT_CENTROID, ist=num_states - 1, ipart=2, ask_for_numpy_array=True)

            points, ek_values = extract_points_ek_values(dr)
            if points is None or ek_values is None:
                print(f"[跳过] {case_id} 没有有效粒子点")
                continue

            # 生成基础场
            X, Y, base_field = generate_energy_field(points, ek_values)

            result = validate_fit(
                element, element_D, points, ek_values,
                X, Y, base_field, optimal_n_eta, optimal_sigma,
                case_id, output_dir
            )
            summary.append(result)

            del dr, element, element_D, points, ek_values, X, Y, base_field
            gc.collect()

        except Exception as e:
            print(f"[Error] {case_id} 验证失败: {e}")
            continue

    # 保存总体统计
    summary_df = pd.DataFrame(summary)
    summary_path = os.path.join(output_dir, 'validation_summary.csv')
    summary_df.to_csv(summary_path, index=False)
    print(f"[完成] 验证统计已保存至：{summary_path}")

# =======================
# 使用示例
# =======================
if __name__ == '__main__':
    csv_path = r'fit_results.csv'
    main_path = r'\\DESKTOP-SVFNCL2\database\MeshSizeCompare\sph2+RP'
    output_dir = r'\validation_results'

    validate_from_d3plot(csv_path, main_path, output_dir)
