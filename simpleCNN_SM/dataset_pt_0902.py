import os
import torch
from torch.utils.data import Dataset

class PreprocessedPTDataset(Dataset):
    def __init__(self, pt_dir, return_index=True):
        """
        pt_dir: 存放 .pt 文件的目录
        return_index: 是否返回索引（默认 True）
                      如果 True: __getitem__ 返回 (signal, target, idx)
                      如果 False: __getitem__ 返回 (signal, target)
        """
        self.pt_dir = pt_dir
        self.files = sorted(f for f in os.listdir(pt_dir) if f.endswith('.pt'))
        self.return_index = return_index

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = os.path.join(self.pt_dir, self.files[idx])
        signal, target = torch.load(path)

        if self.return_index:
            # 返回 (signal, target, idx) 或者 (signal, target, 文件名) 都可以
            # 这里选择文件名更直观
            file_id = os.path.splitext(self.files[idx])[0]  # 去掉 .pt 后缀
            return signal, target, file_id
        else:
            return signal, target
