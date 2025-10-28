import os
import numpy as np
import torch
from torch.utils.data import Dataset

class NumpySignalDataset(Dataset):
    def __init__(self, data_dir, label_np_dir, label_p_dir, transform=None):
        """
        data_dir: 原始数据 .npy 文件目录
        label_np_dir: NP 类别标签所在目录
        label_p_dir: P 类别标签所在目录
        transform: 可选的数据增强操作
        """
        self.samples = []
        self.transform = transform
        self.label_map = {'NP': 0, 'P': 1}

        # 遍历数据目录
        for fname in os.listdir(data_dir):
            if not fname.endswith('.pt') or '_Spectrum_' not in fname:
                continue

            data_path = os.path.join(data_dir, fname)

            # 构造对应的标签文件前缀
            base_name = fname.split('_vz_')[0]  # 如 Bumper1_dp3_vp189

            label_file_np = os.path.join(label_np_dir, base_name + '_damageM.npy')
            label_file_p  = os.path.join(label_p_dir, base_name + '_damageM.npy')

            if os.path.isfile(label_file_np):
                label = self.label_map['NP']
            elif os.path.isfile(label_file_p):
                label = self.label_map['P']
            else:
                continue  # 没有匹配到标签，跳过

            # 读取数据
            data = torch.load(data_path)  # 已是 tensor，不需要再转换

            for i in range(data.shape[0]):
                self.samples.append((data[i], label))

        if len(self.samples) == 0:
            print("⚠️ Warning: Dataset is empty! Please check file matching.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x, y = self.samples[idx]
        x = x.clone().detach().float()  # shape: [4, W, H]
        y = torch.tensor(y, dtype=torch.long)
        if self.transform:
            x = self.transform(x)
        return x, y
