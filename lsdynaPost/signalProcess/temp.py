import os
import numpy as np
from scipy.ndimage import zoom  # 新增：用于放缩

# 定义N，表示生成的组合次数（可以根据需要调整）
N = 128  # 你可以修改这个值为任何整数

# 定义路径
# main_path = r'\\DESKTOP-SVFNCL2\database\MeshSizeCompare\sph2+RP'  # 训练、验证数据集
main_path = r'\\Desktop-svfncl2\backup\MeshSizeCompare\sph2+RP\evaluation'  # 测试数据集
spectrum_data_path = r'D:\SIMULATION\p01_DebrisCloudDamageDataBase\originalData\eval\original_nodoutData_CWTspectrum_npyFiles\acc'  # 根据实际情况修改
random_combinations_path = r'D:\SIMULATION\p01_DebrisCloudDamageDataBase\originalData\data_augmentation\512'  # 根据实际情况修改
output_path = r'D:\SIMULATION\p01_DebrisCloudDamageDataBase\DataBase\DataBase_Signal_CWT_128'  # 结果保存路径

os.makedirs(output_path, exist_ok=True)

# 定义目标空间尺寸（示例：256×256，可根据需要修改，ResNet输入接口大小）
target_h, target_w = 224, 224

# 遍历Spectrum数据文件夹中的所有 .npy 文件
for root, _, files in os.walk(spectrum_data_path):
    for file in files:
        if file.endswith('_az_CWTspectrum.npy'):
            print(f"正在处理文件: {file}")
            file_path = os.path.join(root, file)

            # 加载Spectrum数据 (原始尺寸: [n, W, H])
            try:
                spectrum_data = np.load(file_path)
                print(f"加载 Spectrum 数据，原始形状: {spectrum_data.shape}")
            except Exception as e:
                print(f"加载 Spectrum 数据时出错: {file_path}\n错误信息: {e}")
                continue

            # 放缩 Spectrum 数据至指定的 [n, target_h, target_w] 大小
            original_shape = spectrum_data.shape  # [n, W, H]
            zoom_factors = (1, target_h / original_shape[1], target_w / original_shape[2])
            spectrum_data = zoom(spectrum_data, zoom_factors)
            print(f"放缩后的 Spectrum 数据形状: {spectrum_data.shape}")

            # 获取原始路径信息
            file_name_parts = file.split('_')
            bumper_folder = file_name_parts[0]
            dp_folder = file_name_parts[1]
            vp_folder = file_name_parts[2]

            # 通过文件名构造路径
            node_files_path = os.path.join(main_path, f"{bumper_folder}mm", f"{dp_folder}mm")

            # 读取节点编号文件并排序
            try:
                # 读取节点编号文件 node_ids_output_A.txt
                with open(os.path.join(node_files_path, 'node_ids_output_A.txt'), 'r') as f:
                    node_ids_A = []
                    for line in f.readlines():
                        # 处理每一行，去除逗号并排除0
                        node_ids_A.extend([int(x) for x in line.strip().split(',') if int(x) != 0])

                # 读取节点编号文件 node_ids_output_B.txt
                with open(os.path.join(node_files_path, 'node_ids_output_B.txt'), 'r') as f:
                    node_ids_B = []
                    for line in f.readlines():
                        # 处理每一行，去除逗号并排除0
                        node_ids_B.extend([int(x) for x in line.strip().split(',') if int(x) != 0])

                # 合并并升序排列节点编号
                node_ids = np.sort(np.concatenate((node_ids_A, node_ids_B)))
            except Exception as e:
                print(f"读取节点编号文件时出错: {e}")
                continue

            # 读取随机组合文件 (例如 Bumper1mm_dp3mm_random_Gaugecombinations.txt)
            random_combinations_file = f"{bumper_folder}mm_{dp_folder}mm_random_Gaugecombinations.txt"
            try:
                with open(os.path.join(random_combinations_path, random_combinations_file), 'r') as f:
                    random_combinations = []
                    for line in f.readlines():
                        # 处理每一行，去除逗号并转换为整数，排除0
                        combination = [int(x) for x in line.strip().split(',') if int(x) != 0]
                        # 确保每个组合包含4个节点编号
                        if len(combination) == 4:
                            random_combinations.append(combination)
                        else:
                            print(f"随机组合不符合预期，跳过组合: {combination}")

                print(f"随机组合文件 {random_combinations_file} 读取成功，包含 {len(random_combinations)} 个组合")
            except Exception as e:
                print(f"读取随机组合文件时出错: {random_combinations_file}\n错误信息: {e}")
                continue

            # 确保每个组合包含 4 个节点
            if len(random_combinations[0]) != 4:
                print(f"随机组合文件格式不正确: {random_combinations_file}")
                continue

            # 处理每个组合，提取对应的 Spectrum 数据
            all_combined_spectrograms = []

            for combo in random_combinations[:N]:  # 只选择前N个组合
                # 使用 np.where 查找每个节点在 node_ids 中的索引
                indices = [np.where(node_ids == node)[0][0] for node in combo]  # 获取每个节点的索引
                selected_spectrum = spectrum_data[indices, :, :]  # 提取对应的频谱图

                if selected_spectrum.shape != (4, target_h, target_w):
                    print(f"频谱数据形状不符合预期, 跳过组合: {combo}")
                    continue

                # 将选中的 4 个通道组合成一个新的 4 通道数据
                all_combined_spectrograms.append(selected_spectrum)

            # 将N个组合保存为形状 (N, 4, target_h, target_w) 的数组
            all_combined_spectrograms = np.array(all_combined_spectrograms)

            # 保存为新的 Spectrum 文件
            output_filename = f"{bumper_folder}_{dp_folder}_{vp_folder}_az_Spectrum_randC{N}.npy"
            output_filepath = os.path.join(output_path, output_filename)

            try:
                np.save(output_filepath, all_combined_spectrograms)
                print(f"保存随机组合频谱图: {output_filepath}")
            except Exception as e:
                print(f"保存随机组合频谱图时出错: {output_filepath}\n错误信息: {e}")
