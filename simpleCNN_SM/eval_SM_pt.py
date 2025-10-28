import os
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split
import kornia
from kornia.losses import ssim_loss

from dataset_pt import PreprocessedPTDataset
from train_SM import initialize_weights, set_seed, edge_loss
from model.model_cnn import SimpleCNN
from utils.logger import Logger

def eval_model(pt_dir, opt, model_path, param_dir, batch_size=64, num_workers=4, device='cuda:0', seed=20250410):
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    print(f"Evaluating model: {model_path}")

    # 加载验证集（用相同的随机种子划分）
    dataset = PreprocessedPTDataset(pt_dir)
    set_seed(seed)
    train_size = int(0.0 * len(dataset))
    valid_size = len(dataset) - train_size
    _, valid_dataset = random_split(dataset, [train_size, valid_size])
    valid_dl = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False,
                          num_workers=num_workers, pin_memory=True)

    # 加载 global_stats（反归一化用）
    global_stats_file = f'params/global_stats_{opt}.npy'
    if os.path.exists(global_stats_file):
        global_stats = np.load(global_stats_file, allow_pickle=True).item()
    else:
        global_stats = None

    # 模型初始化与加载
    model = SimpleCNN(opt).to(device)
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model.eval()

    # 评估指标
    ssim_scores, psnr_scores, mse_scores = [], [], []

    with torch.no_grad():
        for batch_idx, (signals, targets) in enumerate(valid_dl):
            signals, targets = signals.to(device), targets.to(device)
            preds = model(signals)

            # 评估
            loss = F.mse_loss(preds, targets)
            ssim_scores.append(ssim_loss(preds, targets, window_size=11).item())
            psnr_scores.append(10 * torch.log10(1 / loss).item())
            mse_scores.append(F.mse_loss(preds, targets).item())

            # 可视化前3张
            if batch_idx < 3000:
                # 获取完整 target / prediction 并反归一化
                pred_img = preds[0].cpu().numpy()  # [C, H, W]
                target_img = targets[0].cpu().numpy()  # [C, H, W]

                # if global_stats is not None:
                #     pred_img = pred_img * (global_stats['target_max'] - global_stats['target_min']) + global_stats[
                #         'target_min']
                #     target_img = target_img * (global_stats['target_max'] - global_stats['target_min']) + global_stats[
                #         'target_min']

                # 可视化所有通道
                num_channels = target_img.shape[0]
                fig, axs = plt.subplots(2, num_channels, figsize=(4 * num_channels, 6), squeeze=False)

                for c in range(num_channels):
                    axs[0, c].imshow(target_img[c], cmap='coolwarm')
                    axs[0, c].set_title(f'Target C{c}')
                    axs[1, c].imshow(pred_img[c], cmap='coolwarm')
                    axs[1, c].set_title(f'Pred C{c}')

                plt.tight_layout()
                plt.savefig(os.path.join(param_dir, f"eval_sample_{batch_idx}.png"))

                # # target_img, pred_img: [C, H, W] 的 numpy 数组
                # num_channels = target_img.shape[0]
                # for c in range(num_channels):
                #     # 保存 target
                #     save_t = os.path.join(param_dir, f"eval_{batch_idx}_target_C{c}.png")
                #     plt.imsave(save_t, -target_img[c], cmap='coolwarm', format='png')
                #
                #     # 保存 pred
                #     save_p = os.path.join(param_dir, f"eval_{batch_idx}_pred_C{c}.png")
                #     plt.imsave(save_p, -pred_img[c], cmap='coolwarm', format='png')
                # # plt.close()

                # plt.tight_layout()
                # plt.savefig(os.path.join(param_dir, f"eval_sample_{batch_idx}.png"))
                # plt.close()

    # 打印平均指标
    print(f"[Evaluation Result]")
    print(f"Avg SSIM: {np.mean(ssim_scores):.4f}")
    print(f"Avg PSNR: {np.mean(psnr_scores):.2f}")
    print(f"Avg MSE : {np.mean(mse_scores):.6f}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str,
                        default=r'D:\SIMULATION\p01_DebrisCloudDamageDataBase\params\thickness_binary_128\Ek_PTModel_epoch250.pth')
    # 420 440 460 530 620 640 650 700 900 960 1010 1020
    parser.add_argument('--pt_dir', type=str,
                        default=r'D:\SIMULATION\p01_DebrisCloudDamageDataBase\DataBase\PreprocessedPT\thinckness_binary_128')
    parser.add_argument('--opt', type=str, default='Ek', help='Target type: EPS, Ek, damageM')
    parser.add_argument('--param_dir', type=str, default='params/thinckness_binary_128/')

    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--seed', type=int, default=20250410)
    args = parser.parse_args()

    os.makedirs(args.param_dir, exist_ok=True)
    eval_model(**vars(args))
