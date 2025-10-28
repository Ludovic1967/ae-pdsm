import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(dim),
            nn.ReLU(),
            nn.Conv2d(dim, dim, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(dim)
        )

    def forward(self, x):
        return x + self.block(x)


class SimpleCNN(nn.Module):
    def __init__(self, opt, embedding_dim=256, size=50, dim=64):
        super().__init__()

        # 设置输出通道数为2
        self.input_dim = 2  # 根据需求设置为2

        self.embedding_dim = embedding_dim
        self.size = size  # 潜在空间的空间尺寸，例如50

        # 编码器：将输入从 [7, 18000, 1] 压缩到嵌入空间 [embedding_dim, size, size]
        self.encoder = nn.Sequential(
            nn.Conv2d(7, 32, kernel_size=4, stride=4, padding=0),  # 输出: [32, 4500, 1]
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=4, padding=0),  # 输出: [64, 1125, 1]
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=4, padding=0),  # 输出: [128, 281, 1]
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=4, padding=0),  # 输出: [256, 70, 1]
            nn.ReLU(),
            nn.Conv2d(256, embedding_dim, kernel_size=2, stride=2, padding=0),  # 输出: [embedding_dim, 35, 1]
            nn.ReLU(),
        )

        # 全连接层：将编码器的输出映射到潜在向量
        self.fc = nn.Sequential(
            nn.Linear(embedding_dim * 35 * 1, embedding_dim * size * size),
            nn.ReLU(),
        )

        # 解码器：将潜在向量上采样到 [2, 800, 800]
        self.decoder = nn.Sequential(
            nn.Conv2d(embedding_dim, dim, kernel_size=3, stride=1, padding=1),  # 输出: [dim, 50, 50]
            ResBlock(dim),
            ResBlock(dim),
            ResBlock(dim),
            ResBlock(dim),
            nn.ReLU(True),
            nn.ConvTranspose2d(dim, dim, kernel_size=4, stride=2, padding=1),  # 输出: [dim, 100, 100]
            nn.ReLU(True),
            nn.ConvTranspose2d(dim, dim, kernel_size=4, stride=2, padding=1),  # 输出: [dim, 200, 200]
            nn.ReLU(True),
            nn.ConvTranspose2d(dim, dim, kernel_size=4, stride=2, padding=1),  # 输出: [dim, 400, 400]
            nn.ReLU(True),
            nn.ConvTranspose2d(dim, self.input_dim, kernel_size=4, stride=2, padding=1),  # 输出: [2, 800, 800]
            nn.Tanh()
        )

    def forward(self, x):
        # x: [nbatch, 7, 18000, 1]
        encoded = self.encoder(x)  # [nbatch, embedding_dim, 35, 1]
        encoded = encoded.view(encoded.size(0), -1)  # [nbatch, embedding_dim * 35 * 1]
        latent = self.fc(encoded)  # [nbatch, embedding_dim * size * size] 即 [nbatch, 64 * 50 * 50]
        latent = latent.view(encoded.size(0), self.embedding_dim, self.size, self.size)  # [nbatch, 64, 50, 50]
        pre = self.decoder(latent)  # [nbatch, 2, 800, 800]
        return pre
