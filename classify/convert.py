import os
import numpy as np
import torch

def convert_npy_to_pt(npy_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    files = [f for f in os.listdir(npy_dir) if f.endswith('.npy')]

    for fname in files:
        fpath = os.path.join(npy_dir, fname)
        arr = np.load(fpath)
        tensor = torch.from_numpy(arr)
        torch.save(tensor, os.path.join(output_dir, fname.replace('.npy', '.pt')))
        print(f"Converted: {fname} ➜ .pt")

# 示例调用（你替换成自己的路径）
convert_npy_to_pt(
    npy_dir=r'D:\SIMULATION\p01_DebrisCloudDamageDataBase\DataBase\DataBase_Signal_CWT_512',
    output_dir=r'D:\SIMULATION\p01_DebrisCloudDamageDataBase\DataBase\DataBase_Signal_CWT_pt'
)
