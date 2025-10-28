# dataset_pt.py
import os
import torch
from torch.utils.data import Dataset

class PreprocessedPTDataset(Dataset):
    def __init__(self, pt_dir):
        self.pt_dir = pt_dir
        self.files = sorted(f for f in os.listdir(pt_dir) if f.endswith('.pt'))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = os.path.join(self.pt_dir, self.files[idx])
        signal, target = torch.load(path)
        return signal, target
