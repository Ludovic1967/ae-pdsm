# train_sm.py
import random
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
from torchvision.models import vgg16
from torchvision import transforms

from CombinedDataset import CombinedDataset

torch.cuda.empty_cache()
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

# SSIM 相关函数 (可放在代码顶部)
import math
#
# def gaussian_window(window_size, sigma):
#     gauss = torch.Tensor([
#         math.exp(-(x - window_size//2)**2 / float(2*sigma**2))
#         for x in range(window_size)
#     ])
#     return gauss / gauss.sum()
#
# def create_window(window_size, channel=1):
#     _1D_window = gaussian_window(window_size, sigma=1.5).unsqueeze(1)
#     _2D_window = _1D_window.mm(_1D_window.t()).float()
#     window = _2D_window.unsqueeze(0).unsqueeze(0)
#     window = window.expand(channel, 1, window_size, window_size).contiguous()
#     return window
#
# def ssim(img1, img2, window_size=21, size_average=True):
#     (_, channel, _, _) = img1.size()
#     window = create_window(window_size, channel=channel).to(img1.device)
#
#     mu1 = F.conv2d(img1, window, padding=window_size//2, groups=channel)
#     mu2 = F.conv2d(img2, window, padding=window_size//2, groups=channel)
#
#     mu1_sq   = mu1 * mu1
#     mu2_sq   = mu2 * mu2
#     mu1_mu2  = mu1 * mu2
#
#     sigma1_sq = F.conv2d(img1*img1, window, padding=window_size//2, groups=channel) - mu1_sq
#     sigma2_sq = F.conv2d(img2*img2, window, padding=window_size//2, groups=channel) - mu2_sq
#     sigma12   = F.conv2d(img1*img2, window, padding=window_size//2, groups=channel) - mu1_mu2
#
#     C1 = 0.01**2
#     C2 = 0.03**2
#
#     ssim_map = ((2.0 * mu1_mu2 + C1) * (2.0 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
#     return ssim_map.mean() if size_average else ssim_map
#
# def ssim_loss(img1, img2, window_size=21, size_average=True):
#     return 1.0 - ssim(img1, img2, window_size, size_average)

# 先把 3×3 Laplacian kernel写成常量（CPU→GPU只发生一次）
_LAPLACE_KERNEL = torch.tensor([[[[0, 1, 0],
                                  [1,-4, 1],
                                  [0, 1, 0]]]], dtype=torch.float32)

def edge_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    pred, target: [B, C, H, W]  —  任意通道数 C
    可在 autocast(fp16/TF32) 下直接使用，支持反传
    """
    C, device, dtype = pred.shape[1], pred.device, pred.dtype

    # ① 将 1×1×3×3 kernel 复制到 C 组，并放到同一 device / dtype
    weight = _LAPLACE_KERNEL.to(device=device, dtype=dtype).expand(C, 1, 3, 3)

    # ② 一次 depth-wise 卷积完成所有通道边缘提取
    pred_edge   = F.conv2d(pred,   weight, padding=1, groups=C)
    target_edge = F.conv2d(target, weight, padding=1, groups=C)

    # ③ 直接 L1 损失；不再做通道循环
    return F.l1_loss(pred_edge, target_edge)

class VGGPerceptualLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        vgg = vgg16(pretrained=True).features[:16]  # 使用 VGG16 的前几层
        for param in vgg.parameters():
            param.requires_grad = False
        self.vgg = vgg.eval()

        self.transform = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                              std=[0.229, 0.224, 0.225])

    def forward(self, pred, target):
        # 保证是单通道输入：如果是 2 通道，取其中一个通道（例如通道 0）
        if pred.size(1) == 2:
            pred = pred[:, 0:1, :, :]
            target = target[:, 0:1, :, :]

        pred_rgb = pred.repeat(1, 3, 1, 1)
        target_rgb = target.repeat(1, 3, 1, 1)
        pred_norm = self.transform(pred_rgb)
        target_norm = self.transform(target_rgb)
        feat_pred = self.vgg(pred_norm)
        feat_target = self.vgg(target_norm)
        return F.mse_loss(feat_pred, feat_target)


# ResBlock 类和 SimpleCNN 类保持不变，只修改了输出激活为 nn.Sigmoid()
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
    def __init__(self, opt, embedding_dim=128, size=50, dim=128):
        super().__init__()

        if opt == 'Ek':
            self.output_dim = 1
        elif opt in ['EPS', 'damageM']:
            self.output_dim = 2
        else:
            raise ValueError(f"Unsupported opt: {opt}")

        self.embedding_dim = embedding_dim
        self.size = size

        self.encoder = nn.Sequential(
            nn.Conv2d(4, 16, kernel_size=4, stride=2, padding=1),  # [4, 224, 224] -> [16, 112, 112]
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=1),  # [32, 112, 112] -> [32, 56, 56]
            nn.ReLU(),
            nn.BatchNorm2d(32),  # 添加批归一化
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),  # [64, 56, 56] -> [64, 28, 28]
            nn.ReLU(),
            nn.BatchNorm2d(64),  # 添加批归一化
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),  # [128, 28, 28] -> [128, 14, 14]
            nn.ReLU(),
            # nn.BatchNorm2d(128),  # 添加批归一化
        )

        # self.fc = nn.Sequential(
        #     nn.Linear(embedding_dim * 14 * 14, embedding_dim * size * size),
        #     nn.ReLU(),
        # )

        # self.decoder = nn.Sequential(
        #     nn.Conv2d(embedding_dim, dim, kernel_size=3, stride=1, padding=1),  # [64, 20, 20] 30
        #     ResBlock(dim),
        #     ResBlock(dim),
        #     ResBlock(dim),
        #     ResBlock(dim),
        #     nn.ReLU(True),
        #     nn.ConvTranspose2d(dim, dim, kernel_size=4, stride=2, padding=1),  # -> [dim, 40, 40]
        #     nn.ReLU(True),
        #     nn.ConvTranspose2d(dim, dim, kernel_size=4, stride=2, padding=1),  # -> [dim, 80, 80]
        #     nn.ReLU(True),
        #     nn.ConvTranspose2d(dim, dim, kernel_size=4, stride=2, padding=1),  # -> [dim, 160, 160]
        #     nn.ReLU(True),
        #     nn.ConvTranspose2d(dim, dim, kernel_size=4, stride=2, padding=1),  # -> [dim, 320, 320]
        #     nn.ReLU(True),
        #     nn.Conv2d(dim, self.output_dim, kernel_size=3, stride=1, padding=1),  # -> [1, 320, 320] 480
        #     nn.Sigmoid()
        # )
        self.decoder = nn.Sequential(
            ResBlock(dim),
            ResBlock(dim),
            nn.ConvTranspose2d(dim, 64, kernel_size=4, stride=2, padding=1),  # -> [64, 28, 28]
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),  # -> [32, 56, 56]
            nn.ReLU(),
            nn.Dropout2d(0.2),  # 添加 Dropout
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),  # -> [16, 112, 112]
            nn.ReLU(),
            nn.ConvTranspose2d(16, 8, kernel_size=4, stride=2, padding=1),  # -> [8, 224, 224]
            nn.ReLU(),
            # nn.Dropout2d(0.2),  # 添加 Dropout
            nn.GroupNorm(num_groups=2, num_channels=8),
            nn.ConvTranspose2d(8, self.output_dim, kernel_size=4, stride=2, padding=1),  # -> [C, 448, 448]
            # nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)  # [B, 64, 14, 14]
        decoded = self.decoder(encoded)  # [B, C, 224, 224]
        return F.interpolate(decoded, size=(500, 500), mode='bilinear', align_corners=False)


# 权重初始化函数
def initialize_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        nn.init.constant_(m.bias, 0)

# 设置随机种子
def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 如果使用多GPU
    torch.backends.cudnn.deterministic = True  # 使得卷积操作可复现
    torch.backends.cudnn.benchmark = False  # 禁止cudnn自动优化

# 定义训练函数
def train_simCNN(signals_dir, images_dir, Ek_dir, eps_dir, opt, param_dir,
                 batch_size, num_workers, device, num_epoch, eval_step, save_every, learning_rate=1e-4):

    device = torch.device(device if torch.cuda.is_available() else 'cpu')

    print(f'learning_rate: {learning_rate}')

    # Load the dataset and split it into training/validation sets
    global_stats_file = 'params/global_stats.npy'
    full_dataset = CombinedDataset(signals_dir, images_dir, Ek_dir, eps_dir, opt=opt, global_stats_file=global_stats_file)
    print(len(full_dataset))
    # print(full_dataset[0].shape)

    seed = 42  # 假设你选择的随机种子是42
    seed = 20250217  # 假设你选择的随机种子是42
    # seed = 2025021702  # 假设你选择的随机种子是42
    set_seed(seed)

    # 然后进行数据集划分
    train_size = int(0.8 * len(full_dataset))
    valid_size = len(full_dataset) - train_size
    train_dataset, valid_dataset = random_split(full_dataset, [train_size, valid_size])

    # 保存随机种子
    np.save(os.path.join(param_dir, 'random_seed.npy'), seed)

    # 后续可以通过相同的种子来复现数据划分
    saved_seed = np.load(os.path.join(param_dir, 'random_seed.npy')).item()
    set_seed(saved_seed)
    train_dataset, valid_dataset = random_split(full_dataset, [train_size, valid_size])
    print(len(train_dataset), len(valid_dataset))

    train_dl = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    valid_dl = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    print(len(train_dl), len(valid_dl))
    print(train_dl)

    # # 获取第一个批次
    # signals, images = next(iter(train_dl))
    #
    # # 打印信号和图像的内容
    # print("First batch - signals:")
    # print(signals)
    # print("First batch - images:")
    # print(images)

    # Initialize model with output_dim=5
    predictor = SimpleCNN(opt).to(device)
    predictor.apply(initialize_weights)

    # Load existing model if available
    model_path = os.path.join(param_dir, f"{opt}_Direc_predictor_epoch_500.pth")
    if os.path.exists(model_path):
        print(f"Loading pre-trained model from {model_path}")
        predictor.load_state_dict(torch.load(model_path, map_location=device))

    optimizer = torch.optim.Adam(predictor.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epoch, eta_min=1e-8)

    scaler = amp.GradScaler()

    epoch = 0
    while epoch < num_epoch:
        epoch += 1
        tmp_loss_rec = []
        predictor.train()

        for idx, (signals, images) in enumerate(train_dl):
            signals = signals.to(device)
            images = images.to(device)
            # print(signals.shape, images.shape)

            optimizer.zero_grad()
            with amp.autocast():
                pre_images = predictor(signals)
                loss = F.mse_loss(pre_images, images)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            tmp_loss_rec.append(loss.item())

            # if idx == 0:  # 打印第一个batch的内容
            #     print("First batch - Predicted images (pre_images):")
            #     # print(pre_images[0])  # 打印第一个样本的预测结果
            #     print(f"Min value: {pre_images.min()}, Max value: {pre_images.max()}")  # 检查值的范围
            #     # print(f"Ground truth (images): {images[0]}")  # 打印第一个样本的标签（图像）

        scheduler.step()

        if epoch % eval_step == 0 or epoch == num_epoch or epoch == 1:
            predictor.eval()
            with torch.no_grad():
                train_loss_mean = np.mean(tmp_loss_rec)
                tmp_valid_loss_rec = []
                ssim_scores = []
                psnr_scores = []
                mse_scores = []

                for idx, (signals, images) in enumerate(valid_dl):
                    signals = signals.to(device)
                    images = images.to(device)
                    pre_images = predictor(signals)
                    loss_v = ssim_loss(pre_images, images)

                    # Calculate metrics
                    ssim_score = ssim(pre_images, images).item()
                    psnr_score = 10 * torch.log10(1 / loss_v).item()
                    mse_score = F.mse_loss(pre_images, images).item()

                    tmp_valid_loss_rec.append(loss_v.item())
                    ssim_scores.append(ssim_score)
                    psnr_scores.append(psnr_score)
                    mse_scores.append(mse_score)

                valid_loss_mean = np.mean(tmp_valid_loss_rec) if len(tmp_valid_loss_rec) > 0 else 0.0
                avg_ssim = np.mean(ssim_scores)
                avg_psnr = np.mean(psnr_scores)
                avg_mse = np.mean(mse_scores)

                print(f"Epoch {epoch}, Train Loss: {train_loss_mean:.5f}, "
                      f"Valid Loss: {valid_loss_mean:.5f}, SSIM: {avg_ssim:.5f}, "
                      f"PSNR: {avg_psnr:.5f}, MSE: {avg_mse:.5f}")

                # Save the model and prediction results
                if epoch % save_every == 0:
                    print(f'learning_rate: {learning_rate}')
                    save_path = os.path.join(param_dir, f"{opt}_Direc_predictor_epoch_{epoch}.pth")
                    torch.save(predictor.state_dict(), save_path)
                    print(f"Model saved to {save_path}")

                    fig, axs = plt.subplots(1, 2)
                    axs[0].imshow(images[0].cpu().numpy()[0], cmap='jet')
                    axs[0].set_title("Ground Truth")
                    axs[1].imshow(pre_images[0].cpu().numpy()[0], cmap='jet')
                    axs[1].set_title("Prediction")
                    plt.savefig(os.path.join(param_dir, f"epoch_{epoch}_prediction.png"))
                    plt.close()

    print(f"Training completed for {opt}.")


# 主函数
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train SimpleCNN Model')
    parser.add_argument('--opt', type=str, default='Ek', help='Option parameter (EPS, Ek, damageM)')
    parser.add_argument('--signals_dir', type=str,
                        default=r'D:\SIMULATION\p01_DebrisCloudDamageDataBase\DataBase\DataBase_Signal_CWT_512',
                        help='Path to the signals .npy files directory')
    parser.add_argument('--images_dir', type=str,
                        default=r'D:\SIMULATION\p01_DebrisCloudDamageDataBase\DataBase\DataBase_VonMisesStress',
                        help='Path to the images .npy files directory')
    parser.add_argument('--Ek_dir', type=str,
                        default=r'D:\SIMULATION\p01_DebrisCloudDamageDataBase\DataBase\DataBase_DebrisCloudEk',
                        help='Path to the images .npy files directory')
    parser.add_argument('--eps_dir', type=str,
                        default=r'D:\SIMULATION\p01_DebrisCloudDamageDataBase\DataBase\DataBase_EffectivePlasticStrain',
                        help='Path to the images .npy files directory')
    parser.add_argument('--param_dir', type=str, default='params/', help='Directory to save model parameters')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--num_workers', type=int, default=8, help='Number of workers for data loading')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device to use for training')
    parser.add_argument('--num_epoch', type=int, default=2000, help='Number of epochs')
    parser.add_argument('--eval_step', type=int, default=2, help='Evaluate every N epochs')
    parser.add_argument('--save_every', type=int, default=100, help='Save model every N epochs')
    parser.add_argument('--learning_rate', type=float, default=5e-5, help='Learning rate')

    args = parser.parse_args()

    os.makedirs(args.param_dir, exist_ok=True)

    # 开始训练
    train_simCNN(
        signals_dir=args.signals_dir,
        images_dir=args.images_dir,
        Ek_dir=args.Ek_dir,
        eps_dir=args.eps_dir,
        opt=args.opt,
        param_dir=args.param_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=args.device,
        num_epoch=args.num_epoch,
        eval_step=args.eval_step,
        save_every=args.save_every,
        learning_rate=args.learning_rate
    )