import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os, torch
from scipy.ndimage import gaussian_filter
from skimage.measure import label, regionprops
from scipy.ndimage import zoom  # 用于插值
from matplotlib import cm

bg_factor = 0.35
gf_sigma = 5
fold_name = 'fold_2'
base_name = 'sample_00180'
save_jpg_dir = r'D:\SIMULATION\p01_DebrisCloudDamageDataBase\params\thickness_binary_128'
save_dir = r'D:\SIMULATION\p01_DebrisCloudDamageDataBase\params\thickness_binary_128'
out_name = f"{fold_name}_{base_name}_pred.npy"
out_path = os.path.join(save_dir, out_name)
# 假设 save_dir 和 c 已经定义好
# save_npy = os.path.join(save_dir, f"pred_C{c}.npy")

# 读取数据
data = np.load(out_path).squeeze()   # 默认会加载成保存时的 numpy 数组类型
print(data.shape)  # 查看形状
print(data.dtype)  # 查看数据类型

# 3. 获取 coolwarm colormap 的一半（这里取上半部分）
full_cmap = cm.get_cmap('coolwarm', 256)         # 原 colormap
half_cmap = full_cmap(np.linspace(0, 0.5, 128))  # 取后半部分（偏暖色）
half_cmap = cm.colors.ListedColormap(half_cmap)  # 转成新的 cmap

data_resized = data
# 4. 归一化数据到 [0,1]，映射到颜色
norm_data = (data_resized - np.min(data_resized)) / (np.max(data_resized) - np.min(data_resized))
colored_img = half_cmap(norm_data)               # shape: (1000,1000,4)

# 5. 去掉 alpha 通道（只保留 RGB）
rgb_img = (colored_img[:, :, :3] * 255).astype(np.uint8)

# 6. 保存为 jpeg
from PIL import Image
save_jpeg = os.path.join(save_jpg_dir, f"{fold_name}_{base_name}_pred.jpeg")
Image.fromarray(rgb_img).save(save_jpeg, quality=95)

print("JPEG 保存路径:", save_jpeg)


