import torch
import torch.nn as nn
import torch.nn.functional as F
from model.resnet_cbam_reg import resnet18_cbam  # 导入你的 CBAM-ResNet 模型

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
        # x: [B, dim, H, W] -> out: [B, dim, H, W]
        return x + self.block(x)

class SimpleCNN(nn.Module):
    def __init__(self, opt, embedding_dim=64, size=15, dim=64, drop_path_rate: float = 0.2):
        super().__init__()

        # 根据 opt 决定输出通道 C
        if opt == 'damageM':
            self.output_dim = 2
        elif opt in ['EPS', 'Ek']:
            self.output_dim = 1
        else:
            raise ValueError(f"Unsupported opt: {opt}")

        self.embedding_dim = embedding_dim  # latent 通道数
        self.size = size                    # latent 空间尺寸

        # ------------------ Encoder (ResNet18+CBAM) ------------------
        # 输入 x: [B, 4, 224, 224]
        self.encoder = resnet18_cbam(pretrained=False)
        # 下面这一行其实不会被用到，因为 resnet_cbam 的 forward 不调用 self.fc
        self.encoder.fc = nn.Identity()

        # timm 写法：把全网随机深度注入 ResNet
        if drop_path is not None:
            for m in self.encoder.modules():
                if isinstance(m, nn.Sequential):
                    for b in m:
                        if hasattr(b, 'drop_path'):
                            b.drop_path.drop_prob = drop_path_rate

        # ResNet18-CBAM 各层输出（输入4通道→7×7特征图）：
        # conv1: [B, 4, 224,224] → stride2 → [B,64,112,112]
        # maxpool:→ [B,64, 56, 56]
        # layer1:→ [B,64, 56, 56]
        # layer2: stride2 → [B,128, 28, 28]
        # layer3: stride2 → [B,256, 14, 14]
        # layer4: stride2 → [B,512,  7,  7]
        # 返回 encoded: [B,512,7,7]

        # ——【遗漏修正】需要一个全连接把 flattened encoder 输出映射为 latent 向量
        # encoded.view 后长度：512*7*7 = 25088
        self.fc = nn.Linear(512 * 7 * 7, embedding_dim * size * size)

        # ------------------ Decoder ------------------
        # latent 初始形状：[B, embedding_dim=32,   size=18,  size=18]
        self.decoder = nn.Sequential(
            ResBlock(dim),  # [B,64,25,25] → [B,64,25,25]
            ResBlock(dim),  # [B,64,25,25] → [B,64,25,25]

            # 上采样 25→36
            nn.ConvTranspose2d(dim, 32, kernel_size=4, stride=2, padding=1),  # → [B,32, 50, 50]
            nn.ReLU(),
            nn.Dropout2d(0.2),                                                  # [B,32, 50, 50]

            # 上采样 50→72
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),     # → [B,16,100,100]
            nn.ReLU(),

            # 上采样 50→144
            nn.ConvTranspose2d(16, 8, kernel_size=4, stride=2, padding=1),  # → [B,16,100,100]
            nn.ReLU(),
            nn.GroupNorm(num_groups=2, num_channels=8),                         # [B, 8,200,200]

            # 上采样 100→288
            nn.ConvTranspose2d(8, 4, kernel_size=4, stride=2, padding=1),      # → [B, 8,200,200]
            nn.ReLU(),


            # 上采样 200→576
            nn.ConvTranspose2d(4, self.output_dim, kernel_size=4, stride=2, padding=1),  # → [B, C,400,400]
            nn.Sigmoid()                                                            # [B, C,400,400]
        )

    def forward(self, x):
        # x: [B,4,224,224]
        encoded = self.encoder(x)                      # → [B,512,  7,  7]
        encoded = encoded.view(x.size(0), -1)          # → [B,512*7*7=25088]
        latent = self.fc(encoded)                      # → [B,64*25*25=40000]
        latent = latent.view(x.size(0),                  # → [B,64,25,25]
                             self.embedding_dim,
                             self.size,
                             self.size)

        pre = self.decoder(latent)                      # → [B,C,400,400]
        pre = F.interpolate(pre,                         # → [B,C,500,500]
                            size=(500, 500),
                            mode='bilinear',
                            align_corners=False)
        return pre
