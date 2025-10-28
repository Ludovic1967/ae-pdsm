#!/usr/bin/env python
# validate.py
# ------------------------------------------------------------
# 用已训练好的 best_model.ckpt 在验证集上推理并计算指标
# ------------------------------------------------------------
import argparse, os, random, sys, time, math
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, average_precision_score,
                             classification_report, confusion_matrix,
                             roc_curve, precision_recall_curve, auc)
import matplotlib.pyplot as plt
import seaborn as sns
from torchvision import models

# -----------------------------------------------------------------
# 请按实际工程结构修改下面两行的 import 路径
from Dataset_signal import NumpySignalDataset
from model import resnet_cbam_reg                    # 你的 resnet_cbam 实现
# -----------------------------------------------------------------


def set_random_seed(seed=42):
    """确保可复现"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# =============== 1. 一些工具函数 ================= #
def load_weights(model, ckpt_path, device="cpu"):
    """自动处理 DataParallel 前缀，加载 state_dict"""
    ckpt = torch.load(ckpt_path, map_location=device)
    state_dict = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    try:
        model.load_state_dict(state_dict)
    except RuntimeError:  # 可能是 DataParallel 的'module.'前缀问题
        from collections import OrderedDict
        new_state = OrderedDict()
        for k, v in state_dict.items():
            new_state[k.replace("module.", "")] = v
        model.load_state_dict(new_state)
    return model


@torch.no_grad()
def evaluate_on_loader(model, loader, device):
    """返回 numpy 数组: labels, scores(取类1的 softmax prob)"""
    model.eval()
    all_labels, all_scores = [], []
    for inputs, labels in loader:
        inputs = inputs.to(device)
        labels = labels.squeeze().to(device)
        logits = model(inputs)
        probs = torch.softmax(logits, dim=1)[:, 1]          # class 1 概率
        all_scores.extend(probs.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    return np.asarray(all_labels), np.asarray(all_scores)


def plot_confusion_matrix(y_true, y_pred, save_path, normalize=True):
    cm = confusion_matrix(y_true, y_pred)
    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1, keepdims=True)
    plt.figure(figsize=(4, 4))
    sns.heatmap(cm, annot=True,
                fmt=".2f" if normalize else "d",
                cmap="Blues",
                xticklabels=["NP", "P"], yticklabels=["NP", "P"])
    plt.xlabel("Predicted"); plt.ylabel("True")
    plt.title("Confusion Matrix" + (" (Norm)" if normalize else ""))
    plt.tight_layout(); plt.savefig(save_path, dpi=300); plt.close()


def plot_roc_pr_curves(y_true, y_scores, save_path):
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))

    # ROC
    ax[0].plot(fpr, tpr, lw=1.5, label=f"AUC={auc(fpr, tpr):.4f}")
    ax[0].plot([0, 1], [0, 1], "--", lw=1)
    ax[0].set_xlabel("False Positive Rate"); ax[0].set_ylabel("True Positive Rate")
    ax[0].set_title("ROC Curve"); ax[0].legend()

    # PR
    ax[1].plot(recall, precision, lw=1.5, label=f"AUC={auc(recall, precision):.4f}")
    ax[1].set_xlabel("Recall"); ax[1].set_ylabel("Precision")
    ax[1].set_title("PR Curve"); ax[1].legend()

    plt.tight_layout(); plt.savefig(save_path, dpi=300); plt.close()


# =============== 2. 主函数 ================= #
def main(args):
    set_random_seed(args.seed)

    # -------- 数据集 & DataLoader -------- #
    full_dataset = NumpySignalDataset(
        data_dir=args.data_dir,
        label_np_dir=args.label_np,
        label_p_dir=args.label_p
    )

    # 只需要验证集：按与训练时一致的切分方式取前 val_split 部分
    indices = list(range(len(full_dataset)))
    val_size = int(args.val_split * len(full_dataset))
    indices.sort()             # 保证与训练时的 random.shuffle 顺序独立
    val_indices = indices[:val_size]
    val_dataset = Subset(full_dataset, val_indices)

    val_loader = DataLoader(val_dataset,
                            batch_size=args.batch_size,
                            shuffle=False, num_workers=4)

    # -------- 模型 -------- #
    device = torch.device("cuda:0" if (torch.cuda.is_available() and args.use_cuda) else "cpu")

    # model = resnet_cbam.resnet18_cbam(pretrained=False)

    model = models.resnet18(pretrained=False)
    model.conv1 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)  # 4-channel input

    model.fc = nn.Linear(model.fc.in_features, 2)
    model = load_weights(model, args.ckpt_path, device)
    model = model.to(device)
    model.eval()

    # -------- 推理 & 计算指标 -------- #
    y_true, y_scores = evaluate_on_loader(model, val_loader, device)
    y_pred = (y_scores >= 0.5).astype(int)

    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec  = recall_score(y_true, y_pred, zero_division=0)
    f1   = f1_score(y_true, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_true, y_scores)
    pr_auc  = average_precision_score(y_true, y_scores)

    print("=== Validation Set Results ===")
    print(f"Accuracy   = {acc:.4f}")
    print(f"Precision  = {prec:.4f}")
    print(f"Recall     = {rec:.4f}")
    print(f"F1 Score   = {f1:.4f}")
    print(f"ROC-AUC    = {roc_auc:.4f}")
    print(f"PR-AUC     = {pr_auc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=["Class0 (NP)", "Class1 (P)"]))

    # -------- 图表输出 -------- #
    os.makedirs(args.out_dir, exist_ok=True)
    plot_confusion_matrix(y_true, y_pred,
                          os.path.join(args.out_dir, "confusion_matrix.png"),
                          normalize=True)
    plot_roc_pr_curves(y_true, y_scores,
                       os.path.join(args.out_dir, "roc_pr_curves.png"))
    print(f"\n图表已保存到 {args.out_dir}")


# =============== 3. CLI 参数 ================= #
if __name__ == "__main__":
    parser = argparse.ArgumentParser("Validate trained model on validation set")
    parser.add_argument('--data_dir', type=str,
                        default=r'D:\SIMULATION\p01_DebrisCloudDamageDataBase\DataBase\eval\DataBase_Signal_CWT_pt',
                        help='Path to directory containing all .npy data')
    parser.add_argument('--label_np', type=str,
                        default=r'D:\SIMULATION\p01_DebrisCloudDamageDataBase\originalData\eval\original_elementValue_npyFiles\damageM_Data\NP',
                        help='Path to NP label folder')
    parser.add_argument('--label_p', type=str,
                        default=r'D:\SIMULATION\p01_DebrisCloudDamageDataBase\originalData\eval\original_elementValue_npyFiles\damageM_Data\P',
                        help='Path to P label folder')
    parser.add_argument("--ckpt_path", type=str,
                        # default="./checkpoint/resnet18_cbam/best_model.ckpt",
                        default="./checkpoint/resnet18_baseline/best_model.ckpt",
                        help="训练阶段保存的 best_model.ckpt 路径")
    parser.add_argument("--val_split", type=float, default=1.0,
                        help="验证集比例，应与训练阶段保持一致")
    parser.add_argument("--batch_size", type=int, default=256,
                        help="验证时 batch size")
    parser.add_argument("--out_dir",   type=str, default="./val_results",
                        help="混淆矩阵、ROC/PR 曲线等图表输出目录")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use_cuda", action="store_true",
                        help="显式指定使用 GPU (若可用)")
    args = parser.parse_args()
    main(args)
