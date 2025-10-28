# train_SM_pt.py
import os
import torch
import numpy as np
import argparse
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch.cuda import amp
from torch.utils.data import DataLoader, random_split
import kornia
from kornia.losses import ssim_loss
from torchvision import transforms as T
import random, math
from sklearn.model_selection import KFold


from dataset_pt import PreprocessedPTDataset
from train_SM import initialize_weights, set_seed, edge_loss
# from model.model_fccnn import SimpleCNN
from model.model_cnn import SimpleCNN
from utils.logger import Logger


# ---------- 数据增广 ----------
def build_transforms():
    return T.Compose([
        T.Resize(256),
        T.RandomCrop(224),
        T.RandomHorizontalFlip(),
        T.RandAugment(num_ops=2, magnitude=7),
        T.ToTensor(),               # 假设数据已做 0-1 归一化
    ])

# ---------- MixUp / CutMix on-the-fly ----------
def mixup_cutmix(x, y, alpha=1.0):
    lam = np.random.beta(alpha, alpha)
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    mixed_y = lam * y + (1 - lam) * y[index, :]
    return mixed_x, mixed_y

def train_from_pt(pt_dir, opt, param_dir, batch_size, num_workers, device, num_epoch,
                  eval_step, save_every, learning_rate, early_stop_patience=20,
                  resume=None, n_splits=5):

    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    print(f"learning_rate: {learning_rate}")

    dataset = PreprocessedPTDataset(pt_dir)
    print(f"Loaded preprocessed .pt dataset with {len(dataset)} samples.")

    seed = 20250410
    set_seed(seed)
    train_size = int(0.15 * len(dataset))
    valid_size = len(dataset) - train_size
    train_dataset, valid_dataset = random_split(dataset, [train_size, valid_size])
    print(f"Train indices for fold : {train_dataset}")
    print(f"Validation indices for fold : {valid_dataset}")

    train_dl = DataLoader(train_dataset, batch_size=batch_size, shuffle=False,
                          num_workers=num_workers, pin_memory=True)
    valid_dl = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False,
                          num_workers=num_workers, pin_memory=True)

    model = SimpleCNN(opt=opt, drop_path_rate=0.2).to(device)

    # 冻结前两段 5 epoch
    for n, p in model.encoder.named_parameters():
        if n.startswith(('conv1', 'bn1', 'layer1', 'layer2')):
            p.requires_grad = False

    # ---------- Optimizer / Scheduler ----------
    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=3e-3,
        steps_per_epoch=len(train_dl), epochs=num_epoch,
        pct_start=0.3, final_div_factor=1e3)
    scaler = amp.GradScaler()

    # ---------- SWA ----------
    swa_model = torch.optim.swa_utils.AveragedModel(model)
    swa_start = int(num_epoch * 0.7)  # 70% 处开始
    swa_scheduler = torch.optim.swa_utils.SWALR(optimizer, swa_lr=1e-4)

    if resume is not None:
        ckpt = torch.load(resume, map_location=device)
        # # 如果你之前只保存了 state_dict：
        # model.load_state_dict(ckpt)
        # print(f"=> loaded model weights from '{resume}'")
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        start_epoch = ckpt['epoch'] + 1
        best_val = ckpt.get('best_val', np.inf)
        print(f"=> Resumed from epoch {ckpt['epoch']}, best_val={best_val}")
        log_path = os.path.join(param_dir, f"{opt}_training_log_retrain_from_epoch{ckpt['epoch']}.txt")
    else:
        model.apply(initialize_weights)
        log_path = os.path.join(param_dir, f"{opt}_training_log.txt")



    # perceptual_loss_fn = VGGPerceptualLoss().to(device)


    logger = Logger(log_path)
    train_loss_curve, val_loss_curve, ssim_curve, psnr_curve, mse_curve = [], [], [], [], []

    best_val = +np.inf
    patience_counter = 0
    best_model_path = os.path.join(param_dir, f"{opt}_best_model.pth")

    for epoch in range(1, num_epoch + 1):
        model.train()
        losses = []

        # ———— 解除冻结 ————
        if epoch == 6:  # 第 6 个 epoch 起全量训练
            for p in model.parameters():
                p.requires_grad = True

        OFFSET = 300

        ssim_weight = min(0.01 + (epoch-OFFSET) * 0.001, 0.05)
        edge_weight = min(0.01 + (epoch-OFFSET) * 0.001, 0.05)
        # perceptual_weight = 0.0 if epoch < 10 else min((epoch - 9) * 0.001, 0.01)

        for signals, targets in train_dl:
            signals, targets = signals.to(device), targets.to(device)  #[:,-1:,:,:]

            # ☆ MixUp / CutMix  ^
            if random.random() < 0.5:
                signals, targets = mixup_cutmix(signals, targets, alpha=1.0)

            optimizer.zero_grad()
            with amp.autocast():
                preds = model(signals)
                # print(f'signals shape: {signals.shape}')
                # print(f'targets shape: {targets.shape}')
                # print(f'preds shape: {preds.shape}')

                # huber = F.smooth_l1_loss(preds, targets)
                huber = F.mse_loss(preds, targets)
                ssim_l = ssim_loss(preds, targets, window_size=11)
                # perceptual = perceptual_loss_fn(preds[0:1], targets[0:1])
                edge_l = edge_loss(preds, targets)
                # 不归一化，或者只使用 soft log 处理
                # perceptual = torch.log(1 + perceptual) / 10

                use_combo_loss = (epoch >= OFFSET and np.mean(ssim_curve[-50:]) >= 0.80)

                if use_combo_loss:

                    # loss = huber + ssim_weight * ssim_l + perceptual_weight * perceptual + edge_weight * edge_l
                    loss = huber + ssim_weight * ssim_l + edge_weight * edge_l
                else:
                    # loss = huber + edge_weight * edge_l
                    loss = huber

                # 防止 loss 爆炸或为无效值
                if not torch.isfinite(loss) or loss.item() > 10:
                    print(f"[WARNING] Invalid loss detected: {loss.item()}, skipping step.")
                    continue

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # 加入梯度裁剪
            scaler.step(optimizer)
            scaler.update()

            losses.append(loss.item())

        # torch.cuda.empty_cache()

        # —— LR Scheduler / SWA ——
        if epoch > swa_start:
            swa_model.update_parameters(model)
            swa_scheduler.step()
        else:
            scheduler.step()

        # —— BN 更新（训练完） ——
        if epoch == num_epoch:
            torch.optim.swa_utils.update_bn(train_dl, swa_model, device=device)
            torch.save(swa_model.module.state_dict(),
                       os.path.join(param_dir, f"{opt}_swa_final.pth"))

        if epoch % eval_step == 0 or epoch == 1:
            model.eval()
            val_losses, ssim_scores, psnr_scores, mse_scores = [], [], [], []
            with torch.no_grad():
                for signals, targets in valid_dl:
                    signals, targets = signals.to(device), targets.to(device)  #[:,-1:,:,:]
                    preds = model(signals)
                    # print('!!!!!!')

                    loss_v = F.mse_loss(preds, targets)
                    val_losses.append(loss_v.item())
                    ssim_scores.append(1-ssim_loss(preds, targets, window_size=11).item())
                    psnr_scores.append(10 * torch.log10(1 / loss_v).item())
                    mse_scores.append(F.mse_loss(preds, targets).item())
                # 在验证时加这段，帮你排查模型输出范围
                print(f"[DEBUG] Pred: min={preds.min().item():.4f}, max={preds.max().item():.4f}")
                print(f"[DEBUG] Target: min={targets.min().item():.4f}, max={targets.max().item():.4f}")
                # print(f"Channel 0 std: {preds[:, 0].std().item():.5f}, Channel 1 std: {preds[:, 1].std().item():.5f}")
                # print(f"Inter-channel diff: {(preds[:, 0] - preds[:, 1]).abs().mean().item():.6f}")

            avg_train = np.mean(losses)
            avg_val = np.mean(val_losses)
            avg_ssim = np.mean(ssim_scores)

            train_loss_curve.append(avg_train)
            val_loss_curve.append(avg_val)
            ssim_curve.append(avg_ssim)
            psnr_curve.append(np.mean(psnr_scores))
            mse_curve.append(np.mean(mse_scores))

            logger.append(f"[Epoch {epoch}] Train Loss: {avg_train:.8f}, "
                          f"Val Loss: {avg_val:.8f}, "
                          f"Val SSIM: {avg_ssim:.8f}, "
                          f"PSNR: {np.mean(psnr_scores):.8f}, "
                          f"MSE: {np.mean(mse_scores):.8e}, "
                          f"Loss Breakdown: huber={huber.item():.8f}, ssim_l={ssim_l.item():.8f}, edge_l={edge_l.item():.8f}")

            # Early Stopping
            if avg_val < best_val:
                best_val = avg_val
                patience_counter = 0
                torch.save(model.state_dict(), best_model_path)
                logger.append(f"New best model saved at epoch {epoch} with SSIM={avg_ssim:.4f}")
            else:
                patience_counter += 1
                logger.append(f"No improvement. Patience counter: {patience_counter}/{early_stop_patience}")
            #
            # if patience_counter >= early_stop_patience:
            #     logger.append(f"Early stopping triggered at epoch {epoch}")
            #     break

            if epoch % save_every == 0:
                # torch.save(model.state_dict(), os.path.join(param_dir, f"{opt}_PTModel_epoch{epoch}.pth"))
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'best_val': best_val,
                }, os.path.join(param_dir, f"{opt}_PTModel_epoch{epoch}.pth"))

                # 可视化：反归一化 target 和 prediction（假设 0-1 归一化）
                fig, axs = plt.subplots(3, targets.shape[1], figsize=(4 * targets.shape[1], 8), squeeze=False)
                for c in range(targets.shape[1]):
                    axs[0, c].imshow(targets[0, c].detach().cpu().numpy(), cmap='jet')
                    axs[0, c].set_title(f"Target C{c}")
                    axs[0, c].axis('off')
                    axs[1, c].imshow(preds[0, c].detach().cpu().numpy(), cmap='jet')
                    axs[1, c].set_title(f"Pred C{c}")
                    axs[1, c].axis('off')
                    residual = np.abs(preds[0, c].cpu().numpy() - targets[0, c].cpu().numpy())
                    axs[2, c].imshow(residual, cmap='hot')  # 残差图
                    axs[2, c].set_title(f"Residual C{c}")

                plt.tight_layout()
                plt.savefig(os.path.join(param_dir, f"{opt}_vis_epoch_{epoch}.png"))
                plt.close()

    np.save(os.path.join(param_dir, f"{opt}_train_loss.npy"), train_loss_curve)
    np.save(os.path.join(param_dir, f"{opt}_val_loss.npy"), val_loss_curve)
    np.save(os.path.join(param_dir, f"{opt}_ssim.npy"), ssim_curve)
    np.save(os.path.join(param_dir, f"{opt}_psnr.npy"), psnr_curve)
    np.save(os.path.join(param_dir, f"{opt}_mse.npy"), mse_curve)

    plt.figure()
    plt.plot(train_loss_curve, label='Train Loss')
    plt.plot(val_loss_curve, label='Val Loss')
    plt.xlabel("Epochs (eval step)")
    plt.ylabel("Loss")
    plt.legend()
    plt.title(f"Loss Curve for {opt}")
    plt.savefig(os.path.join(param_dir, f"{opt}_loss_curve.png"))
    plt.close()

    logger.append("Training finished.")
    logger.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pt_dir', type=str,
            default=r'D:\SIMULATION\p01_DebrisCloudDamageDataBase\DataBase\PreprocessedPT\thinckness_binary_128')
    parser.add_argument('--opt', type=str, default='Ek', help='Target type: EPS, Ek, damageM')
    parser.add_argument('--param_dir', type=str,
            default=r'D:\SIMULATION\p01_DebrisCloudDamageDataBase\params\thickness_binary_128')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--num_epoch', type=int, default=2000)
    parser.add_argument('--eval_step', type=int, default=1)
    parser.add_argument('--save_every', type=int, default=10)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--early_stop_patience', type=int, default=50)
    # parser.add_argument('--resume', type=str,
    #         default=r'D:\SIMULATION\p01_DebrisCloudDamageDataBase\params\thinckness_binary_128\Ek_PTModel_epoch370.pth',
    #         help='path to .pth checkpoint to resume training from')
    args = parser.parse_args()

    os.makedirs(args.param_dir, exist_ok=True)
    train_from_pt(**vars(args))
