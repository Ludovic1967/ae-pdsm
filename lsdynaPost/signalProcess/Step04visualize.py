import os
import numpy as np
import matplotlib.pyplot as plt

# 输入路径：保存随机组合频谱图的目录
input_path = r'D:\SIMULATION\p01_DebrisCloudDamageDataBase\DataBase\DataBase_Signal_CWT'  # 根据实际情况修改

# 遍历目录中的所有 .npy 文件
for root, _, files in os.walk(input_path):
    for file in files:
        if file.endswith('_vz_Spectrum_randC64.npy'):
            # 读取已保存的随机组合频谱图数据
            file_path = os.path.join(root, file)
            try:
                spectrogram_data = np.load(file_path)
                print(f"加载文件 {file}，数据形状: {spectrogram_data.shape}")
            except Exception as e:
                print(f"加载频谱图时出错: {file_path}\n错误信息: {e}")
                continue

            # 可视化每个组合的频谱图
            # 假设 spectrogram_data 的形状是 (64, 4, 500, 500)
            num_combinations = spectrogram_data.shape[0]
            num_channels = spectrogram_data.shape[1]

            # 创建一个新的图形
            for comb_idx in range(num_combinations):
                fig, axes = plt.subplots(1, num_channels, figsize=(15, 5))
                for ch_idx in range(num_channels):
                    ax = axes[ch_idx]
                    ax.imshow(spectrogram_data[comb_idx, ch_idx, :, :], aspect='auto', cmap='jet')
                    ax.set_title(f"通道 {ch_idx + 1} - 组合 {comb_idx + 1}")
                    ax.axis('off')

                # 保存每个组合的可视化图像
                output_image_filename = f"{file.split('.')[0]}_comb{comb_idx + 1}.png"
                output_image_filepath = os.path.join(root, output_image_filename)
                plt.tight_layout()
                plt.savefig(output_image_filepath)
                plt.close(fig)  # 关闭图像，避免内存泄漏

            print(f"保存了 {num_combinations} 个组合的频谱图到文件夹 {root}")
