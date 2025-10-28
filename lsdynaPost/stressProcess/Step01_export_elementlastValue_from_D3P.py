import os
import numpy as np
import gc
from lsreader import D3plotReader, DataType as dt

# 定义主路径
main_path = r'\\DESKTOP-SVFNCL2\database\MeshSizeCompare\sph2+RP'
# main_path = r'D:\SIMULATION\p01_DebrisCloudDamageDataBase' # 【回家测试用】

# 定义保存结果的路径
# VMstress_output_path = r'D:\SIMULATION\p01_DebrisCloudDamageDataBase\originalData\original_elementValue_npyFiles\VMStress_Data'  # 请根据需要修改
# EPstrain_output_path = r'D:\SIMULATION\p01_DebrisCloudDamageDataBase\originalData\original_elementValue_npyFiles\EPStrain_Data'      # 请根据需要修改
D_output_path = r'D:\SIMULATION\p01_DebrisCloudDamageDataBase\originalData\original_elementValue_npyFiles\D_Data'

# # 如果保存路径不存在，则创建
if not os.path.exists(D_output_path):
    os.makedirs(D_output_path)
# if not os.path.exists(EPstrain_output_path):
#     os.makedirs(EPstrain_output_path)

# 预先读取已有的输出文件列表，避免重复处理
# existing_VMstress_files = set(os.listdir(VMstress_output_path))
# existing_EPstrain_files = set(os.listdir(EPstrain_output_path))
existing_D_files = set(os.listdir(D_output_path))

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

            # 构造输出文件名
            # VMstress_output_filename = f"Bumper{bumper_num}_dp{dp_num}_vp{vp_num}_elementMax_VMStress.npy"
            # EPstrain_output_filename = f"Bumper{bumper_num}_dp{dp_num}_vp{vp_num}_elementMax_EPStrain.npy"
            D_output_filename = f"Bumper{bumper_num}_dp{dp_num}_vp{vp_num}_elementLast_D.npy"

            # 如果输出文件已经存在，则跳过处理
            if (D_output_filename in existing_D_files):
                    # and (damageM_output_filename in existing_damageM_files)):
                print(f"Skip already processed:  {D_output_filename}")
                continue

            print(f"Processing folder: {vp_path}")

            # 读取 d3plot 文件（与 d3plot182 放在同一路径，名字为 d3plot）
            d3plot_file = os.path.join(vp_path, 'd3plot')
            dr = D3plotReader(d3plot_file)

            # 获取时态数量
            num_states = dr.get_data(dt.D3P_NUM_STATES)
            print(f"Number of states: {num_states}")

            # 初始化列表以存储所有时态的 Von Mises 应力和有效塑性应变
            # element_VonMisesStress_all = []
            # element_EffectivePlasticStrain_all = []

            # # 遍历所有时态，提取数据
            # for ist in range(num_states - 1):
            #     element_VonMisesStress = dr.get_data(dt.D3P_SOLID_VON_MISES_STRESS, ist=ist, ask_for_numpy_array=True)
            #     element_VonMisesStress_all.append(element_VonMisesStress)
            #
            #     element_eps = dr.get_data(dt.D3P_SOLID_EFFECTIVE_PLASTIC_STRAIN, ist=ist, ask_for_numpy_array=True)
            #     element_EffectivePlasticStrain_all.append(element_eps)
            #
            #     if ist % 20 == 0:
            #         print(f"Processing state: {ist}")
            #
            #     # 打印第一次的部分数据以检查
            #     if ist == 0:
            #         print(f"element_VonMisesStress at ist={ist}:")
            #         print(element_VonMisesStress[:50])  # 打印前50个值
            #         print(f"element_VonMisesStress.shape: {element_VonMisesStress.shape}")

            # 将所有时态的数据堆叠成数组
            # element_VonMisesStress_all = np.array(element_VonMisesStress_all)
            # element_EffectivePlasticStrain_all = np.array(element_EffectivePlasticStrain_all)
            element_D = dr.get_data(dt.D3P_SOLID_HISTORY_VAR, ist=num_states-1, ipt=2, ihv=5, ask_for_numpy_array=True)

            # 计算每个元素的最大Von Mises应力和最大有效塑性应变（修改）
            # element_Max_VonMisesStress = np.max(element_VonMisesStress_all, axis=0)
            # element_Max_eps = np.max(element_EffectivePlasticStrain_all, axis=0)

            # 先保存完整数据，后续再处理
            # element_Max_VonMisesStress = element_VonMisesStress_all
            # element_Max_eps = element_EffectivePlasticStrain_all

            # 打印结果信息
            print(f"element_D.shape: {element_D.shape}")
            print(f"最小值: {np.min(element_D)}, 最大值: {np.max(element_D)}")
            # print("element_Max_VonMisesStress 前50个值:")
            # print(element_Max_VonMisesStress[:50])  # 打印前50个最大值
            # print("element_Max_eps 前50个值:")
            # print(element_Max_eps[:50])  # 打印前50个最大值

            # 构造保存文件路径
            # VMstress_output_file = os.path.join(VMstress_output_path, VMstress_output_filename)
            # EPstrain_output_file = os.path.join(EPstrain_output_path, EPstrain_output_filename)
            D_output_file = os.path.join(D_output_path, D_output_filename)

            # 保存结果
            np.save(D_output_file, element_D)
            print(f"Saved D_data to {D_output_file}\n")

            # np.save(EPstrain_output_file, element_Max_eps)
            # print(f"Saved Max EP Strain to {EPstrain_output_file}\n")

            # 释放内存
            del element_D
            del dr
            gc.collect()
