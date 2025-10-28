import torch
import torch.nn as nn
import torch.nn.functional as F
from model.resnet_cbam_reg import resnet34_cbam, resnet18_cbam  # 导入你的 CBAM-ResNet 模型

#  双通道damageM预测效果不好；两个通道输出容易一样难以区分（fc的作用比am大？）

# ===== 新增: DropPath（stochastic depth）=======
try:
    from timm.models.layers import drop_path   # timm>=0.9
except ImportError:
    drop_path = None

class ResBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(dim, dim, kernel_size=(3, 1), stride=(1, 1), padding=(1, 0)),
            nn.BatchNorm2d(dim),
            nn.ReLU(),
            nn.Conv2d(dim, dim, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)),
            nn.BatchNorm2d(dim)
        )

    def forward(self, x):
        return x + self.block(x)

class SimpleCNN(nn.Module):
    def __init__(self, opt, embedding_dim=128, size=50, dim=512, drop_path_rate: float = 0.2):
        super().__init__()

        if opt == 'damageM':
            self.output_dim = 2
        elif opt in ['EPS', 'Ek', 'damageM0', 'damageM1']:
            self.output_dim = 1
        else:
            raise ValueError(f"Unsupported opt: {opt}")

        self.embedding_dim = embedding_dim
        self.size = size

        # 使用 CBAM + ResNet 作为 encoder
        self.encoder = resnet18_cbam(pretrained=False)  # 用CBAM的ResNet34替换原来的encoder
        # self.encoder.avgpool = nn.Identity()  # ← 关键：取消全局池化
        self.encoder.fc = nn.Identity()  # 去掉原有的全连接层（即分类头）

        # timm 写法：把全网随机深度注入 ResNet
        if drop_path is not None:
            for m in self.encoder.modules():
                if isinstance(m, nn.Sequential):
                    for b in m:
                        if hasattr(b, 'drop_path'):
                            b.drop_path.drop_prob = drop_path_rate

        # 解码器部分
        self.decoder = nn.Sequential(
            ResBlock(dim),
            nn.Dropout2d(0.3),
            ResBlock(dim),
            nn.ConvTranspose2d(dim, 128, kernel_size=4, stride=2, padding=1),  # -> [64, 28, 28]
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # -> [64, 28, 28]
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),  # -> [32, 56, 56]
            nn.ReLU(),
            nn.Dropout2d(0.2),  # 添加 Dropout
            nn.ConvTranspose2d(32, self.output_dim, kernel_size=4, stride=2, padding=1),  # -> [16, 112, 112]
            # nn.ReLU(),
            # nn.ConvTranspose2d(16, 8, kernel_size=4, stride=2, padding=1),  # -> [8, 224, 224]
            # nn.ReLU(),
            # nn.GroupNorm(num_groups=2, num_channels=8),
            # nn.ConvTranspose2d(8, self.output_dim, kernel_size=4, stride=2, padding=1),  # -> [C, 448, 448]
            nn.Sigmoid()
        )

    def forward(self, x, return_features=False):
        # print(f"Input shape before encoder: {x.shape}")  # 打印输入形状
        # # 如果x有多余的维度，可以通过squeeze去掉
        # x = x.squeeze(1)  # 去除额外的维度
        # print(f"Input shape before encoder: {x.shape}")  # 打印输入形状
        encoded = self.encoder(x)  # 输入到CBAM的ResNet
        # print(f"Encoded shape: {encoded.shape}")  # 打印编码器的输出形状
        decoded = self.decoder(encoded)  # 通过解码器生成输出
        # print(f"Decoded shape: {decoded.shape}")  # 打印编码器的输出形状
        output = F.interpolate(decoded, size=(151, 151), mode='bilinear', align_corners=False)
        if return_features:
            return output, encoded
        else:
            return output

