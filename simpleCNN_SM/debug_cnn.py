# train_SM_pt.py
import os
import torch
import numpy as np
import argparse
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch.cuda import amp
from torch.utils.data import DataLoader, random_split
from torch.profiler import profile, ProfilerActivity, schedule
from time import perf_counter
from torchmetrics.image import StructuralSimilarityIndexMeasure as SSIM
import kornia
from kornia.losses import ssim_loss

from dataset_pt import PreprocessedPTDataset
from train_SM import initialize_weights, set_seed, VGGPerceptualLoss, edge_loss
# from model.model_fccnn import SimpleCNN
from model.model_cnn import SimpleCNN
from utils.logger import Logger


def train_from_pt(pt_dir, opt, param_dir, batch_size, num_workers, device, num_epoch,
                  eval_step, save_every, learning_rate, early_stop_patience=20, resume=None):
    device = torch.device(device if torch.cuda.is_available() else 'cpu')

    # SSIM(
    #     data_range=1.0,  # 你的张量已经 Sigmoid→0-1
    #     kernel_size=(21, 21),  # 与原来保持一致
    #     sigma=(1.5, 1.5),
    #     reduction='elementwise_mean'
    # ).to(device)

    print(f"learning_rate: {learning_rate}")

    dataset = PreprocessedPTDataset(pt_dir)
    print(f"Loaded preprocessed .pt dataset with {len(dataset)} samples.")

    seed = 20250410
    set_seed(seed)
    train_size = int(0.8 * len(dataset))
    valid_size = len(dataset) - train_size
    train_dataset, valid_dataset = random_split(dataset, [train_size, valid_size])

    train_dl = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                          num_workers=num_workers, pin_memory=True,
                          persistent_workers=True, prefetch_factor=4)
    valid_dl = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False,
                          num_workers=num_workers, pin_memory=True,
                          persistent_workers=True, prefetch_factor=4)

    model = SimpleCNN(opt=opt).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epoch, eta_min=1e-8)
    scaler = amp.GradScaler()

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
        fetch_t, compute_t = 0.0, 0.0
        tic_epoch = perf_counter()
        losses = []

        OFFSET = 300

        ssim_weight = min(0.0001 + (epoch-OFFSET) * 0.000001, 0.0005)
        edge_weight = min(0.001 + (epoch-OFFSET) * 0.00001, 0.005)
        # perceptual_weight = 0.0 if epoch < 10 else min((epoch - 9) * 0.001, 0.01)

        for signals, targets in train_dl:

            tic_fetch = perf_counter()  # ← 开始计 fetch
            signals, targets = signals.to(device), targets.to(device)
            # print(signals.dtype, targets.dtype)  # 应分别看到 torch.float16 / torch.float32 vs torch.float64

            torch.cuda.synchronize()
            fetch_t += perf_counter() - tic_fetch  # ← fetch 用时

            tic_compute = perf_counter()  # ← 开始计 compute
            with amp.autocast():
                preds = model(signals)
                # print(preds.dtype)
                loss = F.mse_loss(preds, targets)
                ssim_l = ssim_loss(preds, targets, window_size=11)
                # perceptual = perceptual_loss_fn(preds[0:1], targets[0:1])
                edge_l = edge_loss(preds, targets)
            scaler.scale(loss).backward();
            scaler.step(optimizer);
            scaler.update()
            torch.cuda.synchronize()
            compute_t += perf_counter() - tic_compute  # ← compute 用时

        # torch.cuda.synchronize()
        print(
            f"[Epoch {epoch}] fetch {fetch_t:.1f}s, compute {compute_t:.1f}s, total {perf_counter() - tic_epoch:.1f}s")

        val_fetch, val_comp = 0.0, 0.0

        for signals, targets in valid_dl:
            ## ① 计数据加载 + H→D 搬运时间
            t0 = perf_counter()
            signals = signals.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            torch.cuda.synchronize()  # 把异步 copy 同步
            t1 = perf_counter()
            val_fetch += t1 - t0

            ## ② 计前向 (必要时含损失) + CUDNN 计算时间
            with torch.cuda.amp.autocast():
                preds = model(signals)
                # 如果你在验证时还要算 loss，就放这里
                loss = F.mse_loss(preds, targets)
                ssim_l = ssim_loss(preds, targets, window_size=11, reduction='mean')
                # perceptual = perceptual_loss_fn(preds[0:1], targets[0:1])
                edge_l = edge_loss(preds, targets)

            torch.cuda.synchronize()  # 等 GPU 做完
            t2 = perf_counter()
            val_comp += t2 - t1

        # --------- 打印 ----------
        print(f"val fetch {val_fetch:.1f}s, val compute {val_comp:.1f}s, "
              f"total {val_fetch + val_comp:.1f}s (loop={len(valid_dl)})")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pt_dir', type=str,
            default=r'D:\SIMULATION\p01_DebrisCloudDamageDataBase\DataBase\PreprocessedPT\EPS')
    parser.add_argument('--opt', type=str, default='EPS',
                        help='Target type: EPS, Ek, damageM')
    parser.add_argument('--param_dir', type=str,
                        default=r'D:\SIMULATION\p01_DebrisCloudDamageDataBase\params\EPS')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--num_epoch', type=int, default=1000)
    parser.add_argument('--eval_step', type=int, default=1)
    parser.add_argument('--save_every', type=int, default=10)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--early_stop_patience', type=int, default=50)
    # parser.add_argument('--resume', type=str,
    #                     default='D:\PROGRAM\DebrisCloudSM2024\VQ-SM-main\simpleCNN_SM\params\damageM\damageM_PTModel_epoch620.pth',
    #                     help='path to .pth checkpoint to resume training from')
    args = parser.parse_args()

    os.makedirs(args.param_dir, exist_ok=True)

    train_from_pt(**vars(args))
