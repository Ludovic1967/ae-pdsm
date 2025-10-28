from torchinfo import summary
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
import argparse
import matplotlib.pyplot as plt
from torch.nn.utils import clip_grad_norm_
from torch.cuda import amp
from train_SM import SimpleCNN, ssim


device = 'cuda:0'
opt = 'STRESS'
predictor = SimpleCNN(opt).to(device)

param_dir = 'params/'
model_path = os.path.join(param_dir, f"{opt}_Direc_predictor_epoch_1400.pth")
if os.path.exists(model_path):
    print(f"Loading pre-trained model from {model_path}")
    predictor.load_state_dict(torch.load(model_path, map_location=device))
# 假设 predictor 是你训练的模型，且输入尺寸是 (batch_size, channels, height, width)
# summary(predictor, input_size=(4, 500, 500))  # 例如输入是 (4, 500, 500) 大小的图像
# 假设 predictor 是你训练的模型
print(predictor)
