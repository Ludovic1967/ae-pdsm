# import numpy as np
# import os
# import matplotlib.pyplot as plt
# from scipy.ndimage import gaussian_filter
#
# # 设置输出路径
# enhanced_dir = r'D:\SIMULATION\p01_DebrisCloudDamageDataBase\DataBase\DataBase_thickness_128'  # ← 修改为你的增强结果保存路径
#
# # 获取所有增强后的文件（匹配 *_enhanced.npy）
# enhanced_files = sorted([f for f in os.listdir(enhanced_dir) if f.endswith('.npy')])
#
# # 可选：最多显示前 N 个
# max_display = 40
#
# for i, fname in enumerate(enhanced_files):
#     if i >= max_display:
#         break
#
#     file_path = os.path.join(enhanced_dir, fname)
#     data = np.load(file_path)  # shape: (N, 1, 151, 151)
#
#     first_img = (data[0, 0])*1  # 取第一张图像
#     # background = gaussian_filter(first_img, sigma=1000)
#     # Z_res = (first_img - background) * 1  # 小坑为负
#     # first_img = Z_res
#
#     # 图像归一化到 0~1（min-max）
#     img_minmax = (first_img - np.min(first_img)) / (np.max(first_img) - np.min(first_img) + 1e-8)
#
#     # 图像标准化：zero-mean, unit-std
#     img_std = (first_img - np.mean(first_img)) / (np.std(first_img) + 1e-8)
#     # 为了显示，再归一化回 0~1
#     # img_std_disp = (img_std - np.min(img_std)) / (np.max(img_std) - np.min(img_std) + 1e-8)
#     img_std_disp = img_std
#
#     # 可视化：原图、归一化、标准化
#     fig, ax = plt.subplots(2, 2, figsize=(8, 6))
#
#     ax[0, 0].imshow(first_img, cmap='coolwarm')
#     ax[0, 0].set_title(f'{fname}\nOriginal Enhanced')
#     ax[0, 0].axis('off')
#
#     alpha = 0.25
#     a=np.exp(first_img-10)
#     ax[0, 1].imshow(a, cmap='coolwarm')
#     # ax[0, 1].imshow(first_img - background , cmap='hot')
#     ax[0, 1].set_title(f'{np.max(a)-np.min(a)},{(np.max(a)-np.min(a))/np.min(a)}')
#     ax[0, 1].axis('off')  # 空白或可加别的东西
#
#     a=np.exp(np.exp(a))
#     ax[1, 0].imshow(a, cmap='coolwarm')
#     ax[1, 0].set_title(f'{np.max(a)-np.min(a)},{(np.max(a)-np.min(a))/np.min(a)}')
#     ax[1, 0].axis('off')
#
#     a=np.log1p(np.log1p(np.log1p(np.log1p(first_img))))
#     ax[1, 1].imshow(a, cmap='coolwarm')
#     ax[1, 1].set_title(f'{np.max(a)-np.min(a)},{(np.max(a)-np.min(a))/np.min(a)}')
#     ax[1, 1].axis('off')
#
#     plt.tight_layout()
#     plt.show()

import numpy as np
import os
import matplotlib.pyplot as plt

# 设置输入路径
enhanced_dir = r'D:\SIMULATION\p01_DebrisCloudDamageDataBase\DataBase\DataBase_thickness500_binary_128'  # ← 修改为你的路径
output_dir = os.path.join(enhanced_dir, 'visualizations')  # 输出图像保存路径
os.makedirs(output_dir, exist_ok=True)

# 获取所有.npy文件
enhanced_files = sorted([f for f in os.listdir(enhanced_dir) if f.endswith('.npy')])

# 可选：最多显示前 N 个
max_display = 40

for i, fname in enumerate(enhanced_files):
    if i >= max_display:
        break

    file_path = os.path.join(enhanced_dir, fname)
    data = np.load(file_path)  # shape: (N, 1, H, W)

    # 获取第一张图像
    first_img = data[0, 0]

    # 图像归一化到 0~1（min-max）
    img_normalized = first_img
                      # (first_img - np.min(first_img)) / (np.max(first_img) - np.min(first_img) + 1e-8))

    # 创建图像
    plt.figure(figsize=(4, 4))
    plt.imshow(img_normalized, cmap='coolwarm')
    plt.axis('off')
    plt.title(fname)

    # 保存为 PNG 文件（同名）
    save_path = os.path.join(output_dir, os.path.splitext(fname)[0] + '.png')
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.close()
