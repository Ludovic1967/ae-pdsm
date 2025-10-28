import os
import numpy as np
import matplotlib.pyplot as plt

# 输入CWT频谱图数据文件夹路径
cwt_data_path = r'D:\SIMULATION\p01_DebrisCloudDamageDataBase\Experiment_Val\originalData\original_nodoutData_CWTspectrum_npyFiles'  # 修改为实际路径

# 可视化函数
def visualize_cwt_spectrogram(cwt_data, signal_idx):
    """
    可视化指定信号的CWT频谱图。

    参数:
    cwt_data -- CWT频谱图数据，形状为 [nSensor, 500, 500]
    signal_idx -- 信号索引，表示选择第几个信号
    """
    plt.figure(figsize=(8, 6))
    plt.imshow(cwt_data[signal_idx, :, :], aspect='auto', cmap='jet', origin='lower')
    plt.colorbar(label='Magnitude')
    plt.title(f'CWT Spectrum of Sensor {signal_idx + 1}')
    plt.xlabel('Time')
    plt.ylabel('Scale')
    plt.tight_layout()
    plt.show()

# 遍历CWT频谱图数据文件夹中的所有 .npy 文件
for root, _, files in os.walk(cwt_data_path):
    for file in files:
        if file.endswith('_cwt_180us.npy'):
            print(f"正在加载文件: {file}")
            file_path = os.path.join(root, file)

            try:
                # 加载CWT频谱图数据
                cwt_data = np.load(file_path)
                print(f"成功加载 {file}, 形状: {cwt_data.shape}")

                # 检查数据是否为 [11, 500, 500]
                # if cwt_data.shape[1:] != (500, 500):
                #     print(f"警告: {file} 的单个频谱图形状不是 (500, 500)，跳过该文件")
                #     continue

                n_sensors = cwt_data.shape[0]

                # 可视化每个信号的CWT频谱图
                for signal_idx in range(n_sensors):
                    visualize_cwt_spectrogram(cwt_data, signal_idx)

            except Exception as e:
                print(f"加载或可视化文件 {file} 时出错: {e}")
