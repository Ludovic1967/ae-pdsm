from torch.utils.data import DataLoader, Subset

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
from model.model_cnn_151 import SimpleCNN
from utils.logger import Logger

def build_transforms():
    return T.Compose([
        T.Resize(256),
        T.RandomCrop(224),
        T.RandomHorizontalFlip(),
        T.RandAugment(num_ops=2, magnitude=7),
        T.ToTensor(),
    ])

def mixup_cutmix(x, y, alpha=1.0):
    lam = np.random.beta(alpha, alpha)
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    mixed_y = lam * y + (1 - lam) * y[index, :]
    return mixed_x, mixed_y

def train_one_fold(train_dl, valid_dl, fold_idx, fold_dir, opt, device, num_epoch, eval_step, save_every, learning_rate, early_stop_patience):
    model = SimpleCNN(opt=opt, drop_path_rate=0.2).to(device)

    for n, p in model.encoder.named_parameters():
        if n.startswith(('conv1', 'bn1', 'layer1', 'layer2')):
            p.requires_grad = False

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=3e-3,
                                                    steps_per_epoch=len(train_dl), epochs=num_epoch,
                                                    pct_start=0.3, final_div_factor=1e3)
    scaler = amp.GradScaler()
    swa_model = torch.optim.swa_utils.AveragedModel(model)
    swa_start = int(num_epoch * 0.7)
    swa_scheduler = torch.optim.swa_utils.SWALR(optimizer, swa_lr=1e-4)

    logger = Logger(os.path.join(fold_dir, f"fold_{fold_idx}_log.txt"))
    train_loss_curve, val_loss_curve, ssim_curve, psnr_curve, mse_curve = [], [], [], [], []
    best_val = +np.inf
    patience_counter = 0
    best_model_path = os.path.join(fold_dir, f"{opt}_best_model_fold{fold_idx}.pth")

    for epoch in range(1, num_epoch + 1):
        model.train()
        losses = []
        if epoch == 6:
            for p in model.parameters():
                p.requires_grad = True

        OFFSET = 200
        ssim_weight = min(0.01 + (epoch - OFFSET) * 0.001, 0.05)
        edge_weight = min(0.01 + (epoch - OFFSET) * 0.001, 0.05)

        for signals, targets in train_dl:
            signals, targets = signals.to(device), targets.to(device)
            if epoch > 10 and random.random() < 0.1:
                signals, targets = mixup_cutmix(signals, targets, alpha=1.0)

            optimizer.zero_grad()
            with amp.autocast():
                preds = model(signals)
                huber = F.mse_loss(preds, targets)
                ssim_l = ssim_loss(preds, targets, window_size=11)
                edge_l = edge_loss(preds, targets)
                use_combo_loss = (epoch >= OFFSET and np.mean(ssim_curve[-50:]) >= 0.75) if ssim_curve else False
                if use_combo_loss:
                    loss = huber + ssim_weight * ssim_l + edge_weight * edge_l
                else:
                    loss = huber
                if not torch.isfinite(loss) or loss.item() > 10:
                    continue
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            losses.append(loss.item())

        if epoch > swa_start:
            swa_model.update_parameters(model)
            swa_scheduler.step()
        else:
            scheduler.step()

        if epoch == num_epoch:
            torch.optim.swa_utils.update_bn(train_dl, swa_model, device=device)
            torch.save(swa_model.module.state_dict(), os.path.join(fold_dir, f"{opt}_swa_final_fold{fold_idx}.pth"))

        if epoch % eval_step == 0 or epoch == 1:
            model.eval()
            val_losses, ssim_scores, psnr_scores, mse_scores = [], [], [], []
            with torch.no_grad():
                for signals, targets in valid_dl:
                    signals, targets = signals.to(device), targets.to(device)
                    preds = model(signals)
                    loss_v = F.mse_loss(preds, targets)
                    val_losses.append(loss_v.item())
                    ssim_scores.append(1 - ssim_loss(preds, targets, window_size=11).item())
                    psnr_scores.append(10 * torch.log10(1 / loss_v).item())
                    mse_scores.append(F.mse_loss(preds, targets).item())

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
            if avg_val < best_val:
                best_val = avg_val
                patience_counter = 0
                torch.save(model.state_dict(), best_model_path)
                logger.append(f"New best model saved at epoch {epoch}")
            else:
                patience_counter += 1
                logger.append(f"No improvement. Patience: {patience_counter}/{early_stop_patience}")

            if epoch % save_every == 0:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'best_val': best_val,
                }, os.path.join(fold_dir, f"{opt}_PTModel_fold{fold_idx}_epoch{epoch}.pth"))

                fig, axs = plt.subplots(3, targets.shape[1], figsize=(4 * targets.shape[1], 8), squeeze=False)
                for c in range(targets.shape[1]):
                    axs[0, c].imshow(targets[0, c].detach().cpu().numpy(), cmap='coolwarm')
                    axs[0, c].set_title(f"Target C{c}")
                    axs[0, c].axis('off')
                    axs[1, c].imshow(preds[0, c].detach().cpu().numpy(), cmap='coolwarm')
                    axs[1, c].set_title(f"Pred C{c}")
                    axs[1, c].axis('off')
                    residual = np.abs(preds[0, c].cpu().numpy() - targets[0, c].cpu().numpy())
                    axs[2, c].imshow(residual, cmap='hot')
                    axs[2, c].set_title(f"Residual C{c}")

                plt.tight_layout()
                plt.savefig(os.path.join(fold_dir, f"{opt}_vis_fold{fold_idx}_epoch_{epoch}.png"))
                plt.close()

            # if patience_counter >= early_stop_patience:
            #     logger.append(f"Early stopping at epoch {epoch}")
            #     break

    logger.append("Training finished.")
    logger.close()

def train_from_pt(pt_dir, opt, param_dir, batch_size, num_workers, device, num_epoch,
                  eval_step, save_every, learning_rate, early_stop_patience=20, k_folds=5, start_fold=1):

    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    dataset = PreprocessedPTDataset(pt_dir)
    print(f"Loaded dataset with {len(dataset)} samples.")
    set_seed(20250410)
    kf = KFold(n_splits=k_folds, shuffle=False)

    for fold, (train_idx, val_idx) in enumerate(kf.split(dataset)):
        if fold + 1 < start_fold:
            print(f"[INFO] Skipping Fold {fold + 1} (before start_fold={start_fold})")
            continue

        print(f"\n===== Fold {fold + 1}/{k_folds} =====")
        print(f"Train indices for fold {fold + 1}: {train_idx}")
        print(f"Validation indices for fold {fold + 1}: {val_idx}")

        fold_dir = os.path.join(param_dir, f"fold_{fold + 1}")
        os.makedirs(fold_dir, exist_ok=True)

        train_subset = Subset(dataset, train_idx)
        val_subset = Subset(dataset, val_idx)

        train_dl = DataLoader(train_subset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        val_dl = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

        train_one_fold(train_dl, val_dl, fold + 1, fold_dir, opt, device, num_epoch, eval_step, save_every, learning_rate, early_stop_patience)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pt_dir', type=str,
            default=r'D:\SIMULATION\p01_DebrisCloudDamageDataBase\DataBase\PreprocessedPT\thinckness_binary_128')
    parser.add_argument('--opt', type=str, default='EPS', help='Target type: EPS, Ek, damageM')
    parser.add_argument('--param_dir', type=str,
            default=r'D:\SIMULATION\p01_DebrisCloudDamageDataBase\params\thickness_binary_128')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--num_epoch', type=int, default=2000)
    parser.add_argument('--eval_step', type=int, default=1)
    parser.add_argument('--save_every', type=int, default=20)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--early_stop_patience', type=int, default=50)
    parser.add_argument('--k_folds', type=int, default=8)
    # parser.add_argument('--start_fold', type=int, default=4, help='Start training from this fold number')
    args = parser.parse_args()

    os.makedirs(args.param_dir, exist_ok=True)
    train_from_pt(**vars(args))