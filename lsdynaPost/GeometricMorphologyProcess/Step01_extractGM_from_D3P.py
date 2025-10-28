import os
import numpy as np
from lsreader import D3plotReader, DataType as dt
from messagRead import extract_deleted_nodes
import open3d as o3d
import matplotlib.pyplot as plt

def process_vp_folder(vp_path):
    """
    对单个 vp 文件夹进行处理，提取有限元模型中外层的上、下表面节点
    返回值：top_coords, bottom_coords；若出错则返回 None
    """
    d3plot_file = os.path.join(vp_path, 'd3plot')
    if not os.path.isfile(d3plot_file):
        print(f"d3plot 文件不存在: {vp_path}")
        return None

    try:
        dr = D3plotReader(d3plot_file)
    except Exception as e:
        print(f"读取 d3plot 文件失败: {vp_path}, 错误信息: {e}")
        return None

    # -----------------------------
    # Step 1: 提取被删除的节点编号，并减去固定偏移量
    # -----------------------------
    num_states = dr.get_data(dt.D3P_NUM_STATES)
    print(f"状态总数: {num_states}")
    temp = dr.get_data(dt.D3P_SPH_MASS, ist=0, ask_for_numpy_array=True)

    # 提取被删除的节点编号（返回的是字符串列表）
    _, deleted_element_ids_str = extract_deleted_nodes(vp_path)
    # 假设需要减去的固定偏移量为 OFFSET（根据实际情况修改）
    OFFSET = len(temp) + 1
    deleted_element_ids = [int(id_str) - OFFSET for id_str in deleted_element_ids_str]
    print(f"deleted_element_ids总数: {len(deleted_element_ids)}")

    # -----------------------------
    # Step 2: 读取初始状态节点坐标，并过滤被删除的节点
    # -----------------------------
    elementCoord_Initial = dr.get_data(dt.D3P_SOLID_ELEMENT_CENTROID, ist=0, ipart=2, ask_for_numpy_array=True)
    nElement = elementCoord_Initial.shape[0]
    print(f"初始状态element总数: {nElement}")

    mask = np.ones(nElement, dtype=bool)
    for element_id in deleted_element_ids:
        if 0 <= element_id < nElement:
            mask[element_id] = False
        else:
            print(f"警告：element_id {element_id} 超出范围")
    elementCoord_Initial_filtered = elementCoord_Initial[mask, :]
    print(f"过滤后element数量: {elementCoord_Initial_filtered.shape[0]}")

    # -----------------------------
    # Step 3: 根据 x 轴坐标分切片，记录每个切片的原索引信息
    # -----------------------------
    x_coords = np.round(elementCoord_Initial_filtered[:, 0], decimals=6)
    unique_x = np.unique(x_coords)
    unique_x_sorted = np.sort(unique_x)[::-1]

    slices = {}
    for ux in unique_x_sorted:
        indices = np.where(np.isclose(x_coords, ux, atol=1e-6))[0]
        slices[ux] = indices

    # -----------------------------
    # Step 4: 读取结束状态节点坐标数据，并同样过滤
    # -----------------------------
    elementCoord_end = dr.get_data(dt.D3P_SOLID_ELEMENT_CENTROID, ist=num_states - 1, ipart=2, ask_for_numpy_array=True)
    elementCoord_end = elementCoord_end[mask, :]

    # -----------------------------
    # Step 5: 在每个切片内按 y 坐标分组，提取结束状态下 z 坐标最大和最小的节点（上、下表面轮廓）
    # -----------------------------
    top_contour_indices = []
    bottom_contour_indices = []
    for ux, indices in slices.items():
        y_coords = np.round(elementCoord_Initial_filtered[indices, 1], decimals=6)
        unique_y = np.unique(y_coords)
        for uy in unique_y:
            group_idx = indices[np.where(np.isclose(y_coords, uy, atol=1e-6))[0]]
            z_vals = elementCoord_end[group_idx, 2]
            max_z_local_idx = np.argmax(z_vals)
            min_z_local_idx = np.argmin(z_vals)
            top_contour_indices.append(group_idx[max_z_local_idx])
            bottom_contour_indices.append(group_idx[min_z_local_idx])
    print(f"提取的上表面轮廓element数量: {len(top_contour_indices)}")
    print(f"提取的下表面轮廓element数量: {len(bottom_contour_indices)}")

    top_coords = elementCoord_end[top_contour_indices, :]
    bottom_coords = elementCoord_end[bottom_contour_indices, :]

    # 如需要可添加可视化代码，此处为批量处理时可注释掉
    # pcd_top = o3d.geometry.PointCloud()
    # pcd_top.points = o3d.utility.Vector3dVector(top_coords[:, :3])
    # pcd_bottom = o3d.geometry.PointCloud()
    # pcd_bottom.points = o3d.utility.Vector3dVector(bottom_coords[:, :3])
    # o3d.visualization.draw_geometries([pcd_top])
    # o3d.visualization.draw_geometries([pcd_bottom])

    return top_coords, bottom_coords

def main():
    # # 主路径（需要处理的有限元数据存放路径）
    # main_path = r'\\DESKTOP-SVFNCL2\database\MeshSizeCompare\sph2+RP'
    # # 输出结果保存的固定路径
    # output_path = r'D:\SIMULATION\p01_DebrisCloudDamageDataBase\originalData\original_elementValue_npyFiles\damageM_Data'

    # 主路径（需要处理的有限元数据存放路径）
    main_path = r'\\Desktop-svfncl2\backup\MeshSizeCompare\sph2+RP\evaluation'
    # 输出结果保存的固定路径
    output_path = r'D:\SIMULATION\p01_DebrisCloudDamageDataBase\originalData\eval\original_elementValue_npyFiles\damageM_Data'
    os.makedirs(output_path, exist_ok=True)

    # 在输出路径下创建3个子文件夹
    folder1 = os.path.join(output_path, 'NP')  # 两个表面 z 坐标范围均小于 t
    folder2 = os.path.join(output_path, 'folder2')  # 一大一小于 t
    folder3 = os.path.join(output_path, 'P')  # 两个表面 z 坐标范围均大于 t
    for folder in [folder1, folder2, folder3]:
        os.makedirs(folder, exist_ok=True)

    # 遍历主路径下所有三级文件夹
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

                # 检查该文件夹是否包含 'd3plot182' 文件（依据参考代码要求）
                d3plot182_path = os.path.join(vp_path, 'd3plot182')
                if not os.path.isfile(d3plot182_path):
                    continue

                # 构造输出文件名（根据文件夹名称中的数字）
                bumper_num = ''.join(filter(str.isdigit, bumper_folder))
                dp_num = ''.join(filter(str.isdigit, dp_folder))
                vp_num = ''.join(filter(str.isdigit, vp_folder))
                output_filename = f'Bumper{bumper_num}_dp{dp_num}_vp{vp_num}_damageM.npy'

                # 如果该输出文件已存在于任一结果文件夹中，则跳过
                if (output_filename in os.listdir(folder1) or
                    output_filename in os.listdir(folder2) or
                    output_filename in os.listdir(folder3)):
                    print(f"【已存在】跳过处理: {output_filename}")
                    continue

                print(f"Processing folder: {vp_path}")
                res = process_vp_folder(vp_path)
                if res is None:
                    continue
                top_coords, bottom_coords = res

                # 根据题目要求计算 t 与两个表面的 z 坐标范围
                t = -1.15 * (np.mean(bottom_coords[:, 2]) - np.mean(top_coords[:, 2]))
                range_top = np.max(top_coords[:, 2]) - np.min(top_coords[:, 2])
                range_bottom = np.max(bottom_coords[:, 2]) - np.min(bottom_coords[:, 2])
                print(f"t = {t:.4f}, range_top = {range_top:.4f}, range_bottom = {range_bottom:.4f}")

                # 根据比较结果选择保存文件夹
                # if range_top < t and range_bottom < t:
                #     save_folder = folder1
                if range_top > t and range_bottom > t:
                    save_folder = folder3
                else:
                    save_folder = folder1

                save_path = os.path.join(save_folder, output_filename)
                # 保存结果，结果为字典，包含上表面、下表面坐标以及指标信息
                result = {
                    'top_coords': top_coords,
                    'bottom_coords': bottom_coords
                }
                np.save(save_path, result)
                print(f" [SAVE info] 结果已保存至: {save_path}")

if __name__ == '__main__':
    main()
