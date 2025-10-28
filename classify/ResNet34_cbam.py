import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, Subset
from torchvision import models
from trainer.trainer import Trainer
from utils.logger import Logger
from torchnet.meter import ClassErrorMeter
from tensorboardX import SummaryWriter
import torch.backends.cudnn as cudnn
import numpy as np
import random
from model import resnet_cbam_reg
import matplotlib.pyplot as plt
import seaborn as sns
# 一些 sklearn 评估函数
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, classification_report,
    confusion_matrix, roc_curve, auc, precision_recall_curve
)

from Dataset_signal import NumpySignalDataset

def set_random_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def evaluate_on_loader(model, data_loader, device):
    model.eval()
    all_labels = []
    all_scores = []
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            logits = model(inputs)
            probs  = torch.softmax(logits, dim=1)
            pos_probs = probs[:, 1]
            all_labels.append(labels.cpu().numpy())
            all_scores.append(pos_probs.cpu().numpy())
    all_labels = np.concatenate(all_labels)
    all_scores = np.concatenate(all_scores)
    return all_labels, all_scores

def plot_learning_curves(history, save_path):
    """
    history: dict 包含 'train_loss','val_loss','train_acc','val_acc' 四个键，各对应一个列表
    """
    train_losses = history['train_loss']
    val_losses   = history['val_loss']
    train_accs   = history['train_acc']
    val_accs     = history['val_acc']
    epochs = list(range(1, len(train_losses) + 1))

    plt.figure(figsize=(10,4))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, 'b-', label='Train Loss')
    plt.plot(epochs, val_losses,   'r--', label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training & Validation Loss')
    plt.legend()
    plt.grid(alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accs, 'b-', label='Train Acc')
    plt.plot(epochs, val_accs,   'r--', label='Val Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training & Validation Accuracy')
    plt.legend()
    plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_confusion_matrix(y_true, y_pred, save_path, normalize=True):
    labels = ['Class 0','Class 1']
    cm_counts = confusion_matrix(y_true, y_pred)
    if normalize:
        cm_norm = confusion_matrix(y_true, y_pred, normalize='true')
    else:
        cm_norm = None

    plt.figure(figsize=(8,4))
    plt.subplot(1, 2, 1)
    sns.heatmap(cm_counts, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix (Counts)')

    if normalize:
        plt.subplot(1, 2, 2)
        sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Greens',
                    xticklabels=labels, yticklabels=labels)
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.title('Confusion Matrix (Normalized)')

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_roc_pr_curves(y_true, y_scores, save_path):
    # ROC
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    # PR
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    pr_auc  = average_precision_score(y_true, y_scores)

    plt.figure(figsize=(12,5))
    plt.subplot(1, 2, 1)
    plt.plot(fpr, tpr, color='darkorange', lw=2,
             label=f'ROC (AUC={roc_auc:.3f})')
    plt.plot([0,1], [0,1], color='navy', lw=1, linestyle='--', label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc='lower right')
    plt.grid(alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.plot(recall, precision, color='purple', lw=2,
             label=f'PR (AP={pr_auc:.3f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc='lower left')
    plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def main(args):

    set_random_seed(args.seed)

    log_dir = './logs'
    os.makedirs(log_dir, exist_ok=True)
    logger = Logger(os.path.join(log_dir, args.model + '.log'))
    logger.append(vars(args))

    writer = SummaryWriter() if args.display else None

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    gpus = args.gpu.split(',')
    is_use_cuda = torch.cuda.is_available()
    print(is_use_cuda)
    cudnn.benchmark = True

    # Load and split dataset
    full_dataset = NumpySignalDataset(
        data_dir=args.data_dir,
        label_np_dir=args.label_np,
        label_p_dir=args.label_p
    )

    indices = list(range(len(full_dataset)))
    val_size = int(args.val_split * len(full_dataset))
    random.shuffle(indices)
    val_indices = indices[:val_size]
    train_indices = indices[val_size:]

    train_dataset = Subset(full_dataset, train_indices)
    val_dataset = Subset(full_dataset, val_indices)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size * len(gpus), shuffle=True, num_workers=4, pin_memory=False)
    val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False, num_workers=4)

    # 模型构建
    model = resnet_cbam.resnet18_cbam(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, 2)

    if is_use_cuda and len(gpus) == 1:
        model = model.cuda()
    elif is_use_cuda:
        model = nn.DataParallel(model.cuda())

    criterion = [nn.CrossEntropyLoss()]
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
    lr_schedule = lr_scheduler.MultiStepLR(optimizer, milestones=[30, 60], gamma=0.1)

    metric = [ClassErrorMeter([1], True)]

    trainer = Trainer(
        model, args.model, criterion, optimizer, lr_schedule, log_batchs=1,
        is_use_cuda=is_use_cuda, train_data_loader=train_loader,
        valid_data_loader=val_loader, metric=metric, start_epoch=0,
        num_epochs=100, is_debug=args.debug, logger=logger, writer=writer
    )

    trainer.fit()
    logger.append('Training complete.')

    # ———— 4. 绘制学习曲线（如果 Trainer.history 可用） ————
    # 假设 trainer.history 是一个 dict，包含 'train_loss','val_loss','train_acc','val_acc'
    try:
        history = trainer.history
        plot_learning_curves(history, os.path.join(log_dir, 'learning_curves.png'))
        logger.append('Saved learning curves.')
    except Exception as e:
        logger.append(f"Could not plot learning curves: {e}")

    # ———— 5. 在验证集上推理并计算指标 ——
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    best_model = trainer.model  # 确保 model 是训练完后的最优权重
    val_labels, val_scores = evaluate_on_loader(best_model, val_loader, device)

    # —— 5.1 计算指标 ——
    val_pred_labels = (val_scores >= 0.5).astype(int)
    acc = accuracy_score(val_labels, val_pred_labels)
    prec = precision_score(val_labels, val_pred_labels, zero_division=0)
    rec = recall_score(val_labels, val_pred_labels, zero_division=0)
    f1 = f1_score(val_labels, val_pred_labels, zero_division=0)
    roc_auc = roc_auc_score(val_labels, val_scores)
    pr_auc = average_precision_score(val_labels, val_scores)

    print("=== Validation Set Results ===")
    print(f"Accuracy   = {acc:.4f}")
    print(f"Precision  = {prec:.4f}")
    print(f"Recall     = {rec:.4f}")
    print(f"F1 Score   = {f1:.4f}")
    print(f"ROC-AUC    = {roc_auc:.4f}")
    print(f"PR-AUC     = {pr_auc:.4f}")
    print("Classification Report:")
    print(classification_report(val_labels, val_pred_labels, target_names=['Class0', 'Class1']))

    # 同时把这些统计结果写入日志
    logger.append(f"Validation Accuracy: {acc:.4f}")
    logger.append(f"Validation Precision: {prec:.4f}")
    logger.append(f"Validation Recall: {rec:.4f}")
    logger.append(f"Validation F1: {f1:.4f}")
    logger.append(f"Validation ROC-AUC: {roc_auc:.4f}")
    logger.append(f"Validation PR-AUC: {pr_auc:.4f}")

    # —— 5.2 绘制混淆矩阵 ——
    plot_confusion_matrix(val_labels, val_pred_labels,
                          os.path.join(log_dir, 'confusion_matrix.png'), normalize=True)
    logger.append('Saved confusion matrix.')

    # —— 5.3 绘制 ROC & PR 曲线 ——
    plot_roc_pr_curves(val_labels, val_scores,
                       os.path.join(log_dir, 'roc_pr_curves.png'))
    logger.append('Saved ROC and PR curves.')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Vanilla ResNet34 Training Script (no CBAM)')
    parser.add_argument('--data_dir', type=str,
                        default=r'D:\SIMULATION\p01_DebrisCloudDamageDataBase\DataBase\DataBase_Signal_CWT_pt',
                        help='Path to directory containing all .npy data')
    parser.add_argument('--label_np', type=str,
                        default=r'D:\SIMULATION\p01_DebrisCloudDamageDataBase\originalData\original_elementValue_npyFiles\damageM_Data\NP',
                        help='Path to NP label folder')
    parser.add_argument('--label_p', type=str,
                        default=r'D:\SIMULATION\p01_DebrisCloudDamageDataBase\originalData\original_elementValue_npyFiles\damageM_Data\P',
                        help='Path to P label folder')
    parser.add_argument('--gpu', default='cuda:0', type=str, help='GPU ID(s) to use')
    parser.add_argument('--model', default='resnet18_cbam', type=str, help='Model name used for logging')
    parser.add_argument('--batch_size', default=8, type=int, help='Batch size per GPU')
    parser.add_argument('--display', action='store_true', help='Use Tensorboard for visualization')
    parser.add_argument('--debug', action='store_true', help='Debug mode')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--val_split', type=float, default=0.2, help='Validation set ratio (default 0.2)')
    args = parser.parse_args()
    main(args)
