import os
import numpy as np
import scipy.io as sio
import pywt
from scipy.ndimage import zoom

# 输入nodout数据文件夹路径
nodout_data_path = r'D:\SIMULATION\p01_DebrisCloudDamageDataBase\Experiment_Val\originalData\original_signal_matFiles'  # 修改为实际路径

# 输出CWT频谱图保存路径
cwt_save_path = r'D:\SIMULATION\p01_DebrisCloudDamageDataBase\Experiment_Val\originalData\original_nodoutData_CWTspectrum_npyFiles'  # 修改为实际路径
os.makedirs(cwt_save_path, exist_ok=True)


# 小波变换函数
def morlet_wavelet_transform(signal, scales, sampling_rate, wavelet='cmor2.5-1.5'):
    coefficients, frequencies = pywt.cwt(signal, scales, wavelet, sampling_period=1 / sampling_rate)
    return np.abs(coefficients).astype(np.float32), frequencies


# 小波变换参数
sampling_rate = 60e6  # 采样率
scales = np.arange(50, 750)  # 小波尺度范围
resized_shape = (224, 224)  # 缩放后的频谱图尺寸

# 需要提取的列索引（注意Python从0开始计数）
selected_columns = list(range(1, 8)) + list(range(9, 14))  # 第2-8列和10-14列

# 遍历所有.mat文件
for file in os.listdir(nodout_data_path):
    if file.endswith('.mat'):
        file_path = os.path.join(nodout_data_path, file)
        try:
            mat_data = sio.loadmat(file_path)
            print(f"文件 {file} 中的所有键: {mat_data.keys()}")

            # 寻找正确的数据字段
            # 假设有某个字段，比如 'signal_data'、'vel_xyz' 或直接是数组
            # possible_keys = ['signal_data', 'vel_xyz', 'data']
            signal_data = mat_data['Dwh']
            # for key in possible_keys:
            #     if key in mat_data:
            #         signal_data =
            #         break

            if signal_data is None:
                print(f"文件 {file} 中找不到预期的数据字段，跳过。")
                continue

            print(f"读取到信号数据，形状为: {signal_data.shape}")

            # 提取需要的列
            numSample = 10800*5
            trigPosition = int(120000*0.3)
            velocity_data = signal_data[trigPosition-round(numSample*0.222):trigPosition+round(numSample*0.778), selected_columns]  # 只取指定列
            print(f"提取的 velocity_data 形状为: {velocity_data.shape}")

        except Exception as e:
            print(f"加载.mat文件时出错: {file_path}\n错误信息: {e}")
            continue

        # 对每一列（即每个传感器数据）进行小波变换
        all_spectrograms = []

        for i in range(velocity_data.shape[1]):  # 遍历选中的列
            signal = velocity_data[:, i]
            wavelet_coefficients, _ = morlet_wavelet_transform(signal, scales, sampling_rate)

            # 缩放到指定尺寸
            zoom_factors = (resized_shape[0] / wavelet_coefficients.shape[0],
                            resized_shape[1] / wavelet_coefficients.shape[1])
            resized_coeffs = wavelet_coefficients

            all_spectrograms.append(resized_coeffs)

        all_spectrograms = np.stack(all_spectrograms, axis=0)  # 形成形状 [nSensor, H, W]

        # 保存频谱图
        save_name = os.path.splitext(file)[0] + '_cwt_900us.npy'
        output_filepath = os.path.join(cwt_save_path, save_name)
        try:
            np.save(output_filepath, all_spectrograms)
            print(f"保存频谱图: {output_filepath}")
        except Exception as e:
            print(f"保存频谱图时出错: {output_filepath}\n错误信息: {e}")
