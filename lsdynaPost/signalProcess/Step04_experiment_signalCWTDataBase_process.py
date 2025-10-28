import os
import numpy as np
from scipy.ndimage import zoom
from itertools import product

# 定义路径
spectrum_data_path = r'D:\SIMULATION\p01_DebrisCloudDamageDataBase\Experiment_Val\originalData\original_nodoutData_CWTspectrum_npyFiles'
output_path = r'D:\SIMULATION\p01_DebrisCloudDamageDataBase\Experiment_Val\DataBase\DataBase_Signal_CWT\900us'
os.makedirs(output_path, exist_ok=True)

# 定义目标空间尺寸
target_h, target_w = 224, 224

# 预先定义 4 个组的索引
groups = [
    list(range(0, 3)),   # 通道 0,1,2
    list(range(3, 6)),   # 通道 3,4,5
    list(range(6, 9)),   # 通道 6,7,8
    list(range(9, 12)),  # 通道 9,10,11
]

# 生成所有可能的组合 (共 3*3*3*3 = 81 种)
all_combinations = list(product(*groups))
print(f"一共 {len(all_combinations)} 种通道组合。")

# 遍历所有 .npy 文件
for root, _, files in os.walk(spectrum_data_path):
    for file in files:
        if not file.endswith('_cwt_900us.npy'):
            continue

        print(f"正在处理：{file}")
        spectrum_data = np.load(os.path.join(root, file))   # [12, W, H]

        # 检查通道数
        if spectrum_data.shape[0] != 12:
            print(f"警告：{file} 的第一维不是 12，跳过")
            continue

        # 缩放到 [12, target_h, target_w]
        ori_h, ori_w = spectrum_data.shape[1], spectrum_data.shape[2]
        zoom_f = (1, target_h / ori_h, target_w / ori_w)
        spectrum_data = zoom(spectrum_data, zoom_f, order=1)

        # 遍历所有组合，提取并收集
        combined_list = []
        for combo in all_combinations:
            # combo 是一个长度为4的元组，如 (0,4,7,10)
            sel = spectrum_data[list(combo), :, :]   # [4, H, W]
            combined_list.append(sel)

        # 转成数组，形状 [81, 4, H, W]
        combined_arr = np.stack(combined_list, axis=0)

        # 构造输出文件名
        name_base = os.path.splitext(file)[0]
        out_name = f"{name_base}_allC{len(all_combinations)}.npy"
        out_path = os.path.join(output_path, out_name)

        # 保存
        np.save(out_path, combined_arr)
        print(f"已保存：{out_name}, 形状 {combined_arr.shape}")
