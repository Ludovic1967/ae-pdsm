import os
import numpy as np
import scipy.io as sio
import pywt
import re
import time, threading
from scipy.ndimage import zoom
from concurrent.futures import ProcessPoolExecutor

# 输入nodout数据文件夹路径
nodout_data_path = r'D:\SIMULATION\p01_DebrisCloudDamageDataBase\originalData\eval\original_nodoutData_matFiles'  # 修改为实际路径
# nodout_data_path = r'D:\SIMULATION\p01_DebrisCloudDamageDataBase\originalData\original_nodoutData_matFiles'  # 修改为实际路径

# 输出CWT频谱图保存路径
cwt_save_path = r'D:\SIMULATION\p01_DebrisCloudDamageDataBase\originalData\eval\original_nodoutData_CWTspectrum_npyFiles\acc'  # 修改为实际路径
# cwt_save_path = r'D:\SIMULATION\p01_DebrisCloudDamageDataBase\originalData\original_nodoutData_CWTspectrum_npyFiles_1e8\acc'  # 修改为实际路径
os.makedirs(cwt_save_path, exist_ok=True)

# 创建线程锁
lock = threading.Lock()

# Morlet小波变换函数
def morlet_wavelet_transform(signal, scales, sampling_rate, wavelet='cmor2.5-1.5'):
    coefficients, frequencies = pywt.cwt(signal, scales, wavelet, sampling_period=1/sampling_rate)
    return np.abs(coefficients).astype(np.float32), frequencies

# 遍历nodout数据文件夹中的所有 .mat 文件
for root, _, files in os.walk(nodout_data_path):
    for file in files:
        if file.endswith('_nodout.mat'):
            print(f"正在处理文件: {file}")
            file_path = os.path.join(root, file)

            # 从路径中提取 Bumper / dp / vp 参数（包含小数）
            match = re.search(r'Bumper([\d.]+)mm[\\/]+dp([\d.]+)mm[\\/]+vp(\d+)', file_path)
            if match:
                bumper_value = match.group(1)
                dp_value = match.group(2)
                vp_value = match.group(3)
                output_filename = f"B{bumper_value}_dp{dp_value}_vp{vp_value}_az_CWTspectrum.npy"
            else:
                # 若匹配失败，退回默认生成方式
                base_name = os.path.splitext(file)[0]
                output_filename = f"{base_name}_az_CWTspectrum.npy"
                print(f"⚠️ 警告：未能识别路径中的物理参数，使用默认文件名：{output_filename}")

            output_filepath = os.path.join(cwt_save_path, output_filename)

            if os.path.exists(output_filepath):
                print(f"Skip already processed: {output_filename}")
                continue

            # 加载mat文件
            try:
                mat_data = sio.loadmat(file_path)
                print(f"文件 {file} 中的所有键: {mat_data.keys()}")
                if 'signal_data' not in mat_data:
                    print(f"文件 {file} 中没有找到 'signal_data' 键，请检查文件结构。")
                    continue
                signal_data = mat_data['signal_data']
                print(f"signal_data 的类型: {type(signal_data)}")
            except Exception as e:
                print(f"加载.mat文件时出错: {file_path}\n错误信息: {e}")
                continue

            # 检查 signal_data 是否为结构化数组
            if isinstance(signal_data, np.ndarray) and signal_data.dtype.names:
                print(f"signal_data 的字段名: {signal_data.dtype.names}")
            else:
                print("signal_data 不是一个结构化数组，跳过此文件。")
                continue

            # 获取信号数据 (signal_data.vel_xyz)
            try:
                velocity_data = signal_data['acc_xyz'][0][0][:, 2, :]  # z方向速度
                print(f"acc_data 的形状: {velocity_data.shape}")
            except KeyError:
                print(f"文件 {file} 中没有 'acc_xyz' 字段，跳过")
                continue

            # 小波变换参数
            sampling_rate = 1e8  # 1e8
            scales = np.arange(50, 750)
            resized_shape = (500, 500)

            all_spectrograms = []

            for i in range(128):
                signal = velocity_data[:, i]
                wavelet_coefficients, _ = morlet_wavelet_transform(signal, scales, sampling_rate)
                resized_coeffs = wavelet_coefficients  # 可选缩放：zoom(resized_shape / wavelet_coefficients.shape)
                all_spectrograms.append(resized_coeffs)

            all_spectrograms = np.stack(all_spectrograms, axis=0)

            # 保存频谱图
            try:
                np.save(output_filepath, all_spectrograms)
                print(f"保存频谱图: {output_filepath}")
            except Exception as e:
                print(f"保存频谱图时出错: {output_filepath}\n错误信息: {e}")

# # 处理单个文件的函数
# def process_file(file_path, cwt_save_path):
#     print(f"正在处理文件: {file_path}")
#
#     # 从路径中提取 Bumper / dp / vp 参数（包含小数）
#     match = re.search(r'Bumper([\d.]+)[\\/]+dp([\d.]+)[\\/]+vp(\d+)', file_path)
#     if match:
#         bumper_value = match.group(1)
#         dp_value = match.group(2)
#         vp_value = match.group(3)
#         output_filename = f"B{bumper_value}_dp{dp_value}_vp{vp_value}_az_CWTspectrum.npy"
#     else:
#         # 若匹配失败，退回默认生成方式
#         base_name = os.path.splitext(os.path.basename(file_path))[0]  # 获取不带扩展名的文件名
#         output_filename = f"{base_name}_az_CWTspectrum.npy"
#         print(f"⚠️ 警告：未能识别路径中的物理参数，使用默认文件名：{output_filename}")
#
#     output_filepath = os.path.join(cwt_save_path, output_filename)
#
#     # 使用线程锁来确保文件检查和保存操作不会发生竞争
#     with lock:
#         # 确保跳过已存在文件
#         if os.path.exists(output_filepath):
#             print(f"Skip already processed: {output_filename}")
#             return  # 如果文件已经处理过，跳过
#
#     # 加载.mat文件
#     try:
#         mat_data = sio.loadmat(file_path)
#         print(f"文件 {file_path} 中的所有键: {mat_data.keys()}")
#         if 'signal_data' not in mat_data:
#             print(f"文件 {file_path} 中没有找到 'signal_data' 键，请检查文件结构。")
#             return
#         signal_data = mat_data['signal_data']
#         print(f"signal_data 的类型: {type(signal_data)}")
#     except Exception as e:
#         print(f"加载.mat文件时出错: {file_path}\n错误信息: {e}")
#         return
#
#     # 检查 signal_data 是否为结构化数组
#     if isinstance(signal_data, np.ndarray) and signal_data.dtype.names:
#         print(f"signal_data 的字段名: {signal_data.dtype.names}")
#     else:
#         print("signal_data 不是一个结构化数组，跳过此文件。")
#         return
#
#     # 获取信号数据 (signal_data.vel_xyz)
#     try:
#         velocity_data = signal_data['acc_xyz'][0][0][:, 2, :]  # z方向速度
#         print(f"acc_data 的形状: {velocity_data.shape}")
#     except KeyError:
#         print(f"文件 {file_path} 中没有 'acc_xyz' 字段，跳过")
#         return
#
#     # 小波变换参数
#     sampling_rate = 1e8  # 1e8
#     scales = np.arange(50, 750)
#     resized_shape = (500, 500)
#
#     all_spectrograms = []
#
#     for i in range(128):
#         signal = velocity_data[:, i]
#         wavelet_coefficients, _ = morlet_wavelet_transform(signal, scales, sampling_rate)
#         resized_coeffs = wavelet_coefficients  # 可选缩放：zoom(resized_shape / wavelet_coefficients.shape)
#         all_spectrograms.append(resized_coeffs)
#
#     all_spectrograms = np.stack(all_spectrograms, axis=0)
#
#     # 保存频谱图
#     try:
#         np.save(output_filepath, all_spectrograms)
#         print(f"保存频谱图: {output_filepath}")
#     except Exception as e:
#         print(f"保存频谱图时出错: {output_filepath}\n错误信息: {e}")
#
#     # 控制每个进程的CPU使用率，稍作休息
#     time.sleep(0.1)  # 休眠0.1秒，防止CPU过载
#
#
# # 主程序：遍历文件并进行并行处理
# def main():
#     max_workers = 8  # 控制最大并行进程数，避免占用过高的CPU资源
#
#     with ProcessPoolExecutor(max_workers=max_workers) as executor:
#         futures = []
#         for root, _, files in os.walk(nodout_data_path):
#             for file in files:
#                 if file.endswith('_nodout.mat'):
#                     file_path = os.path.join(root, file)
#                     futures.append(executor.submit(process_file, file_path, cwt_save_path))
#
#         # 等待所有任务完成
#         for future in futures:
#             future.result()
#
# # 调用主程序
# if __name__ == "__main__":
#     main()
