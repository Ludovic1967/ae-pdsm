import torch
import numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
from train_SM import SimpleCNN, ssim, ssim_loss, set_seed
from CombinedDataset import CombinedDataset
import argparse
import os


# 初始化模型
def load_model(model_path, device='cuda:0'):
    model = SimpleCNN(opt='STRESS', embedding_dim=64, size=20, dim=64).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model


# 加载数据集并进行划分
def load_data(signals_dir, images_dir, param_dir, batch_size, num_workers, device):
    full_dataset = CombinedDataset(signals_dir, images_dir)

    train_size = int(0.8 * len(full_dataset))
    valid_size = len(full_dataset) - train_size
    train_dataset, valid_dataset = random_split(full_dataset, [train_size, valid_size])

    train_dl = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    valid_dl = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return valid_dl


# 可视化预测结果
def visualize_prediction(images, pre_images, save_path=None):
    images_np = images.cpu().numpy()
    pre_images_np = pre_images.cpu().numpy()
    print(images_np.shape, pre_images_np.shape)

    plt.figure(figsize=(10, 10))

    # Ground Truth 和 Prediction 图像
    plt.subplot(2, 5, 1)
    plt.imshow(images_np[0, 0], cmap='jet')
    plt.title("Ground Truth Image 1")

    plt.subplot(2, 5, 6)
    plt.imshow(pre_images_np[0, 0], cmap='jet')
    plt.title("Predicted Image 1")

    plt.subplot(2, 5, 2)
    plt.imshow(images_np[0, 1], cmap='jet')
    plt.title("Ground Truth Image 2")

    plt.subplot(2, 5, 7)
    plt.imshow(pre_images_np[0, 1], cmap='jet')
    plt.title("Predicted Image 2")

    plt.subplot(2, 5, 3)
    plt.imshow(images_np[0, 2], cmap='jet')
    plt.title("Ground Truth Image 3")

    plt.subplot(2, 5, 8)
    plt.imshow(pre_images_np[0, 2], cmap='jet')
    plt.title("Predicted Image 3")

    plt.subplot(2, 5, 4)
    plt.imshow(images_np[0, 3], cmap='jet')
    plt.title("Ground Truth Image 4")

    plt.subplot(2, 5, 9)
    plt.imshow(pre_images_np[0, 3], cmap='jet')
    plt.title("Predicted Image 4")

    plt.subplot(2, 5, 5)
    plt.imshow(images_np[0, 4], cmap='jet')
    plt.title("Ground Truth Image 5")

    plt.subplot(2, 5, 10)
    plt.imshow(pre_images_np[0, 4], cmap='jet')
    plt.title("Predicted Image 5")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()


# 计算并打印 SSIM, PSNR, MSE 等指标
def calculate_metrics(pre_images, images):
    ssim_score = ssim(pre_images, images).item()
    psnr_score = 10 * torch.log10(1 / F.mse_loss(pre_images, images)).item()  # PSNR
    mse_score = F.mse_loss(pre_images, images).item()  # MSE

    print(f"SSIM: {ssim_score:.5f}, PSNR: {psnr_score:.5f}, MSE: {mse_score:.5f}")
    return ssim_score, psnr_score, mse_score


# 主函数
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate SimpleCNN Model')
    parser.add_argument('--model_path', type=str,
                        default='params/STRESS_Direc_predictor_epoch_500_t1.pth', help='Path to the trained model')
    parser.add_argument('--signals_dir', type=str,
                        default=r'D:\SIMULATION\p01_DebrisCloudDamageDataBase\DataBase\eval\DataBase_Signal_CWT',
                        help='Path to the signals .npy files directory')
    parser.add_argument('--images_dir', type=str,
                        default=r'D:\SIMULATION\p01_DebrisCloudDamageDataBase\DataBase\eval\DataBase_VonMisesStress',
                        help='Path to the images .npy files directory')
    parser.add_argument('--Ek_dir', type=str,
                        default=r'D:\SIMULATION\p01_DebrisCloudDamageDataBase\DataBase\eval\DataBase_DebrisCloudEk',
                        help='Path to the images .npy files directory')
    parser.add_argument('--eps_dir', type=str,
                        default=r'D:\SIMULATION\p01_DebrisCloudDamageDataBase\DataBase\eval\DataBase_EffectivePlasticStrain',
                        help='Path to the images .npy files directory')
    parser.add_argument('--param_dir', type=str, default='params', help='Directory to save model parameters')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size')
    parser.add_argument('--num_workers', type=int, default=8, help='Number of workers for data loading')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device to use for evaluation')

    args = parser.parse_args()

    # 设置设备
    device = args.device if torch.cuda.is_available() else 'cpu'

    # 加载模型和数据
    model = load_model(args.model_path, device)
    global_stats_file = 'params/global_stats.npy'
    full_dataset = CombinedDataset(args.signals_dir, args.images_dir, args.Ek_dir, args.eps_dir, global_stats_file=global_stats_file)
    train_size = int(0.8 * len(full_dataset))
    valid_size = len(full_dataset) - train_size
    # 后续可以通过相同的种子来复现数据划分
    # saved_seed = np.load(os.path.join(args.param_dir, 'random_seed.npy')).item()
    # set_seed(saved_seed)
    train_dataset, valid_dataset = random_split(full_dataset, [train_size, valid_size])
    # print(len(train_dataset), len(valid_dataset))

    train_dl = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    valid_dl = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    # valid_dl = train_dl
    # 初始化评估指标
    total_ssim = 0.0
    total_psnr = 0.0
    total_mse = 0.0
    num_samples = len(valid_dl.dataset)

    # 进行批量预测和评估
    with torch.no_grad():
        for idx, (signals, images) in enumerate(valid_dl):
            signals = signals.to(device)
            images = images.to(device)

            # 模型预测
            pre_images = model(signals)

            # **检查模型输出前几个** pre_images 的内容（检查是否全为0）
            if idx == 0:  # 打印第一个batch的内容
                print(images.shape, pre_images.shape)
                print("First batch - Predicted images (pre_images):")
                # print(pre_images[0])  # 打印第一个样本的预测结果
                print(f"Min value: {pre_images.min()}, Max value: {pre_images.max()}")  # 检查值的范围
                # print(f"Ground truth (images): {images[0]}")  # 打印第一个样本的标签（图像）

            # 计算评估指标
            ssim_score, psnr_score, mse_score = calculate_metrics(pre_images, images)

            total_ssim += ssim_score
            total_psnr += psnr_score
            total_mse += mse_score

            # 可视化部分结果
            if idx % 1 == 0:  # 每10个样本保存一次预测结果
                save_path = f"prediction_epoch_{idx}.png"
                visualize_prediction(images, pre_images, save_path)

    # 输出整体评估结果
    avg_ssim = total_ssim / num_samples
    avg_psnr = total_psnr / num_samples
    avg_mse = total_mse / num_samples

    print(f"Average SSIM: {avg_ssim:.5f}, Average PSNR: {avg_psnr:.5f}, Average MSE: {avg_mse:.5f}")
