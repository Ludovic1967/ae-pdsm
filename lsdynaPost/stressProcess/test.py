import os
import numpy as np
from lsreader import D3plotReader, DataType as dt

# 从我们新建的 feather_rotate_utils.py 文件中导入封装的函数
from lsdynaPost.VirtualRearWall.feather_rotate_utils import get_outer_ring_mean, feather_rotate_cosine_with_blur

def generate_FandB_images_with_rotation(
    vm_stress_npy_file,
    element_initial_coor,
    element_end_coor,
    z_front,
    z_back,
    x_range,
    y_range,
    output_dir,
    random_angle_dir
):
    """
    1. 加载 VMstress npy 文件。
    2. 根据前/后层索引 (z_front, z_back)，从末态坐标中找元素位置，映射到 image_front / image_back。
    3. 从随机角度 txt 文件中读取 64 个旋转角度，对前后层图像进行旋转并堆叠成 [64, 2, height, width]。
       在此过程中，空白区域用 '最外圈平均值' 填充。
    4. 结果以 BumperX_dpY_vpZ_FandB_maxVMStress_R64.npy 命名保存。
    """

    basename = os.path.basename(vm_stress_npy_file)
    # 假设文件名格式如：Bumper2_dp4_vp189_elementMax_VMStress.npy
    try:
        name_parts = basename.split('_')
        bumper_part = name_parts[0]   # Bumper2
        dp_part     = name_parts[1]   # dp4
        vp_part     = name_parts[2]   # vp189

        bumper_num = ''.join(filter(str.isdigit, bumper_part))  # '2'
        dp_num     = ''.join(filter(str.isdigit, dp_part))      # '4'
        vp_num     = ''.join(filter(str.isdigit, vp_part))      # '189'
    except Exception as e:
        print(f"[Error] Fail to parse file name: {basename}, error: {e}")
        return

    # -------------- 1. 加载 VMstress npy 文件 --------------
    if not os.path.isfile(vm_stress_npy_file):
        print(f"[Error] VMstress file not found: {vm_stress_npy_file}")
        return
    element_Max_VonMisesStress = np.load(vm_stress_npy_file)  # shape: [num_elements]

    # -------------- 2. 获取前/后层索引 --------------
    if element_initial_coor.shape[0] != element_Max_VonMisesStress.shape[0]:
        print("[Error] The length of coordinates and stress data does not match!")
        return

    mask_front = np.where(element_initial_coor[:, 2] == z_front)[0]
    mask_back  = np.where(element_initial_coor[:, 2] == z_back)[0]

    front_end_coord = element_initial_coor[mask_front]  # shape: [N_front, 3]
    back_end_coord  = element_initial_coor[mask_back]   # shape: [N_back,  3]

    front_stress = element_Max_VonMisesStress[mask_front]  # shape: [N_front]
    back_stress  = element_Max_VonMisesStress[mask_back]   # shape: [N_back]

    # -------------- 3. 映射到 2D image_front / image_back --------------
    step = 0.02
    height = int((2 * x_range) / step)  # [-x_range, x_range]
    width  = int((2 * y_range) / step)  # [-y_range, y_range]

    image_front = np.zeros((height, width), dtype=np.float32)
    image_back  = np.zeros((height, width), dtype=np.float32)

    def coord_to_index(x, y):
        row = (x + x_range) / (2 * x_range) * (height - 1)
        col = (y + y_range) / (2 * y_range) * (width - 1)
        row = int(np.clip(row, 0, height - 1))
        col = int(np.clip(col, 0, width - 1))
        return row, col

    for i in range(front_end_coord.shape[0]):
        x_i, y_i = front_end_coord[i, 0], front_end_coord[i, 1]
        row, col = coord_to_index(x_i, y_i)
        image_front[row, col] = front_stress[i]

    for i in range(back_end_coord.shape[0]):
        x_i, y_i = back_end_coord[i, 0], back_end_coord[i, 1]
        row, col = coord_to_index(x_i, y_i)
        image_back[row, col] = back_stress[i]

    # (1) 计算前图像外圈均值 cval_front
    cval_front = get_outer_ring_mean(image_front, border=2)
    # (2) 计算后图像外圈均值 cval_back
    cval_back  = get_outer_ring_mean(image_back, border=2)
    # -------------- 4. 从 txt 文件中读取 64 个旋转角度，并对 image_front/back 进行旋转 --------------
    random_angle_filename = f"Bumper{bumper_num}mm_dp{dp_num}mm_random_rotate.txt"
    random_angle_file = os.path.join(random_angle_dir, random_angle_filename)
    if not os.path.isfile(random_angle_file):
        print(f"[Error] Random angle file not found: {random_angle_file}")
        return

    with open(random_angle_file, 'r') as f:
        lines = f.readlines()
    angles = [float(line.strip()) for line in lines]
    if len(angles) < 64:
        print(f"[Warning] The angle file has only {len(angles)} angles, less than 64!")
        return
    angles = angles[:64]

    # -- 计算前后图像外圈平均值，用于 rotate 时的 cval 填充 --
    rotated_images = np.zeros((64, 2, height, width), dtype=np.float32)
    for i, angle in enumerate(angles):
        # ★ 用 feather_rotate 替换掉直接的 rotate 调用
        final_front = feather_rotate_cosine_with_blur(image_front, angle, cval_front)
        final_back  = feather_rotate_cosine_with_blur(image_back,  angle, cval_back)

        rotated_images[i, 0] = final_front
        rotated_images[i, 1] = final_back

    # -------------- 5. 保存结果 --------------
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    save_basename = f"Bumper{bumper_num}_dp{dp_num}_vp{vp_num}_FandB_maxEPStrain_R64.npy"
    save_path = os.path.join(output_dir, save_basename)

    np.save(save_path, rotated_images)
    print(f"[Info] Rotated 64 images saved to: {save_path}")


def batch_generate_FandB_images_with_rotation(
    vm_stress_dir,
    main_path,
    x_range,
    y_range,
    output_dir,
    random_angle_dir
):
    """
    在给定目录下批量遍历 *elementMax_VMStress.npy 文件。
    对每个文件:
      1) 解析 BumperX, dpY, vpZ
      2) 找到对应 d3plot 路径: main_path/BumperXmm/dpYmm/vpZ/d3plot
      3) 初始化 dr, 获取初始/末态坐标
      4) 调用 generate_FandB_images_with_rotation(...) 生成并保存结果
    """

    # ========== 遍历 ep_strain_dir 下所有文件 ==========
    for file_name in os.listdir(vm_stress_dir):
        if not file_name.endswith("_elementMax_EPStrain.npy"):
            continue  # 跳过非目标文件

        vm_stress_npy_file = os.path.join(vm_stress_dir, file_name)
        print(f"[Batch] Processing: {vm_stress_npy_file}")

        # ========== 1) 根据文件名解析 BumperX, dpY, vpZ ==========
        try:
            name_parts = file_name.split('_')
            bumper_part = name_parts[0]  # Bumper2
            dp_part     = name_parts[1]  # dp4
            vp_part     = name_parts[2]  # vp189

            bumper_num = ''.join(filter(str.isdigit, bumper_part))  # '2'
            dp_num     = ''.join(filter(str.isdigit, dp_part))      # '4'
            vp_num     = ''.join(filter(str.isdigit, vp_part))      # '189'
        except Exception as e:
            print(f"[Error] Parse file name failed: {file_name}, error: {e}")
            continue

        # ========== 2) 组合 d3plot 路径并检查 ==========
        bumper_folder_name = f"Bumper{bumper_num}mm"
        dp_folder_name     = f"dp{dp_num}mm"
        vp_folder_name     = f"vp{vp_num}"

        d3plot_dir = os.path.join(main_path, bumper_folder_name, dp_folder_name, vp_folder_name)
        d3plot_file = os.path.join(d3plot_dir, "d3plot")

        if not os.path.isfile(d3plot_file):
            print(f"[Error] d3plot not found in: {d3plot_file}")
            continue

        print(f"  => Found d3plot: {d3plot_file}")

        # ========== 3) 初始化 dr, 获取初始坐标 & 末态坐标 ==========
        dr = D3plotReader(d3plot_file)
        num_states = dr.get_data(dt.D3P_NUM_STATES)
        element_initial_coor = dr.get_data(dt.D3P_SOLID_ELEMENT_CENTROID, ist=0, ask_for_numpy_array=True)
        element_end_coor     = dr.get_data(dt.D3P_SOLID_ELEMENT_CENTROID, ist=num_states - 1, ask_for_numpy_array=True)

        # 这里仅示例使用固定的 z_front / z_back 取值
        # 您可根据自己的需求对 z_front / z_back 的获取方式进行修改
        z_front = element_initial_coor[1, 2]         # 1st layer
        z_back  = element_initial_coor[3740000, 2]   # last layer

        # ========== 4) 调用函数处理并保存 ==========
        generate_FandB_images_with_rotation(
            vm_stress_npy_file=vm_stress_npy_file,
            element_initial_coor=element_initial_coor,
            element_end_coor=element_end_coor,
            z_front=z_front,
            z_back=z_back,
            x_range=x_range,
            y_range=y_range,
            output_dir=output_dir,
            random_angle_dir=random_angle_dir
        )


if __name__ == "__main__":
    """
    运行示例：
      1) 设置批量遍历所需的参数
      2) 调用 batch_generate_FandB_images_with_rotation()
    """

    # ========== 用户配置部分 ==========
    ep_strain_dir   = r"D:\SIMULATION\p01_DebrisCloudDamageDataBase\originalData\original_elementMax_npyFiles\EPStrain_Data"
    main_path       = r"\\DESKTOP-SVFNCL2\database\MeshSizeCompare\sph2+RP"
    # main_path = r'D:\SIMULATION\p01_DebrisCloudDamageDataBase' # 【回家测试用】
    output_dir      = r"D:\SIMULATION\p01_DebrisCloudDamageDataBase\DataBase\DataBase_EffectivePlasticStrain"
    random_angle_dir= r"D:\SIMULATION\p01_DebrisCloudDamageDataBase\originalData\data_augmentation"

    x_range = 5.0
    y_range = 5.0

    batch_generate_FandB_images_with_rotation(
        vm_stress_dir=ep_strain_dir,
        main_path=main_path,
        x_range=x_range,
        y_range=y_range,
        output_dir=output_dir,
        random_angle_dir=random_angle_dir
    )
