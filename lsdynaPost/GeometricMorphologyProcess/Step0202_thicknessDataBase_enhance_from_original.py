import numpy as np
import os
import cv2
import json
from tqdm import tqdm
from scipy.ndimage import gaussian_filter
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from skimage.transform import rescale
from skimage.measure import label, regionprops
import pandas as pd
import os


def enhance_small_spots_via_dog(image, sigma_small=0.5, sigma_large=2.0, alpha=2.0):
    image = image.astype(np.float32)
    blur_small = cv2.GaussianBlur(image, (3, 3), sigmaX=sigma_small)
    blur_large = cv2.GaussianBlur(image, (9, 9), sigmaX=sigma_large)
    dog = blur_small - blur_large
    dog_norm = cv2.normalize(dog, None, 0, 255, cv2.NORM_MINMAX)
    dog_enhanced = cv2.convertScaleAbs(dog_norm, alpha=alpha, beta=0)
    enhanced = image + dog_enhanced
    enhanced_clipped = np.clip(enhanced, 0, 255).astype(np.float32)
    return enhanced_clipped, dog_enhanced


# === 设置参数 ===
sigma_small = 0.1
sigma_large = 5.0
alpha = 2.60

# 输入输出路径
input_dir = r'D:\SIMULATION\p01_DebrisCloudDamageDataBase\DataBase\DataBase_thickness500_128'  # 修改为你的 .npy 文件所在路径
output_dir = r'D:\SIMULATION\p01_DebrisCloudDamageDataBase\DataBase\DataBase_thickness500_binary_128'  # 修改为输出保存路径
os.makedirs(output_dir, exist_ok=True)

# 遍历处理
for fname in tqdm(os.listdir(input_dir)):
    if fname.endswith('.npy'):
        input_path = os.path.join(input_dir, fname)
        data = np.load(input_path)  # shape: (N, 1, 151, 151)

        N = data.shape[0]
        enhanced_all = []
        norm_params = {}
        background_threshold = 5e-3  # 6e-4

        for i in range(N):
            img = data[i, 0]

            # # 归一化
            # min_val = float(np.min(img))
            # max_val = float(np.max(img))
            # norm_img = (img - min_val) / (max_val - min_val + 1e-8)
            # norm_img = np.log1p((img+1)*10000)

            # background = gaussian_filter(img, sigma=1000)
            # Z_res = (img - background) * 1  # 小坑为负
            # Z_res = img
            Z = img

            # ======= 1. 背景扣除 =======
            background = gaussian_filter(Z, sigma=10)
            Z_res = Z - background  # 小坑为负

            # ======= 2. 使用阈值对图像进行二值化处理 =======
            Z_binary = Z_res > background_threshold  # 将大于背景的区域设为 1，小于背景的区域设为 0

            # enhanced, _ = enhance_small_spots_via_dog(
            #     Z_res, sigma_small=sigma_small, sigma_large=sigma_large, alpha=alpha
            # )
            enhanced = Z_binary

            enhanced_all.append(enhanced[np.newaxis, ...])  # shape: (1, 151, 151)

            # 只保存第一个图的归一化参数，其他是旋转版本
            # if i == 0:
            #     norm_params['min_val'] = min_val
            #     norm_params['max_val'] = max_val

        enhanced_array = np.stack(enhanced_all, axis=0)  # shape: (N, 1, 151, 151)

        # 保存增强图像
        np.save(os.path.join(output_dir, fname.replace('.npy', '_binary.npy')), enhanced_array)

        # 保存参数（只保存一次）
        # param_dict = {
        #     'sigma_small': sigma_small,
        #     'sigma_large': sigma_large,
        #     'alpha': alpha,
        #     'min_val': norm_params['min_val'],
        #     'max_val': norm_params['max_val']
        # }
        # json_path = os.path.join(output_dir, fname.replace('.npy', '_params.json'))
        # with open(json_path, 'w') as f:
        #     json.dump(param_dict, f, indent=4)

print("处理完成！")
