import os
import argparse
from PIL import Image                 # 抢先加载 Pillow
import torchvision                    # 再安全地加载 torchvision
from torchvision import models
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold  # 新增 KFold 用于 K折交叉验证
from tensorboardX import SummaryWriter
import torch.backends.cudnn as cudnn
import numpy as np
import random
from model import resnet_cbam_cls
from Dataset_signal import NumpySignalDataset
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, classification_report
)
from sklearn.metrics import precision_recall_curve
import warnings
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

def find_threshold_min_precision(y_true, y_prob, min_precision=0.80):
    """
    在验证集上找出 precision ≥ min_precision 时 recall 最大的阈值。若达不到要求，
    就回退到使 F1 最大的阈值，并给出警告。
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    thresholds = np.append(thresholds, 1.0)   # 保证长度与 precision/recall 对齐

    mask = precision >= min_precision
    if mask.any():
        # 在满足 precision 的阈值里选 recall 最大的
        best_idx = np.argmax(recall[mask])
        return thresholds[mask][best_idx]
    else:
        warnings.warn(
            f"Required precision {min_precision} not reached; "
            "fallback to threshold that maximises F1.")
        f1 = 2 * precision * recall / (precision + recall + 1e-15)
        best_idx = np.argmax(f1)
        return thresholds[best_idx]

def set_random_seed(seed=42):
    """确保可复现"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# 评估函数
# def evaluate_model(model, val_loader, device):
#     model.eval()
#     all_labels = []
#     all_preds = []
#     with torch.no_grad():
#         for inputs, labels in val_loader:
#             inputs = inputs.to(device)
#             labels = labels.to(device)
#             outputs = model(inputs)
#             _, predicted = torch.max(outputs.data, 1)
#             all_labels.extend(labels.cpu().numpy())
#             all_preds.extend(predicted.cpu().numpy())
#
#             # print(f"Predicted labels: {all_preds[:10]}")
#             # print(f"True labels: {all_labels[:10]}")
#
#     # 计算各项指标
#     acc = accuracy_score(all_labels, all_preds)
#     prec = precision_score(all_labels, all_preds)
#
#     # 修改: 使用 zero_division=1 来避免无真阳性时的警告
#     rec = recall_score(all_labels, all_preds, zero_division=1)
#     f1 = f1_score(all_labels, all_preds)
#
#     # 计算 ROC-AUC 和 PR-AUC
#     # 修改: 判断是否有两个类别，避免计算 ROC-AUC 时只有一个类别的错误
#     try:
#         if len(np.unique(all_labels)) > 1:
#             roc_auc = roc_auc_score(all_labels, all_preds)
#         else:
#             roc_auc = None  # 如果没有负类样本，跳过 ROC-AUC 计算
#     except ValueError:
#         roc_auc = None
#
#     # rec = recall_score(all_labels, all_preds)
#     # f1 = f1_score(all_labels, all_preds)
#     # roc_auc = roc_auc_score(all_labels, all_preds)
#     pr_auc = average_precision_score(all_labels, all_preds)
#
#     return acc, prec, rec, f1, roc_auc, pr_auc
def evaluate_model(model, val_loader, device, min_precision=0.90):
    model.eval()
    all_labels = []
    all_probs  = []          # ★ 收集正类概率
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs  = inputs.to(device)
            labels  = labels.to(device)
            outputs = model(inputs)

            # outputs.shape == (batch, 2)；取 softmax 后索引 1 的概率
            probs = torch.softmax(outputs, dim=1)[:, 1]

            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    all_labels = np.asarray(all_labels)
    all_probs  = np.asarray(all_probs)

    # ---------- ① 在验证集上找阈值 ----------
    thr = find_threshold_min_precision(all_labels, all_probs,
                                       min_precision=min_precision)

    # ---------- ② 用该阈值得到离散预测 ----------
    all_preds = (all_probs >= thr).astype(int)

    # ---------- ③ 计算指标 ----------
    acc  = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds, zero_division=1)
    rec  = recall_score(all_labels, all_preds, zero_division=1)
    f1   = f1_score(all_labels, all_preds, zero_division=1)

    try:
        roc_auc = roc_auc_score(all_labels, all_probs)  # 用概率算 AUC 更稳
    except ValueError:
        roc_auc = None

    pr_auc = average_precision_score(all_labels, all_probs)

    # ② 该折的 2×2 混淆矩阵
    cm = confusion_matrix(all_labels, all_preds, labels=[0, 1])  # [[TN,FP],[FN,TP]]

    return acc, prec, rec, f1, roc_auc, pr_auc, thr, cm   # ★ 返回阈值方便调试


# 主评估函数
def evaluate_all_models(args):
    set_random_seed(args.seed)

    # 加载数据集
    full_dataset = NumpySignalDataset(
        data_dir=args.data_dir,
        label_np_dir=args.label_np,
        label_p_dir=args.label_p
    )

    # KFold划分数据集
    kf = KFold(n_splits=args.k_folds, shuffle=False)
    fold_results = []
    conf_list = []  # ⬅️ 新增：保存每折的混淆矩阵
    fold_metrics = []  # [(acc, cm), …]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    for fold, (train_idx, val_idx) in enumerate(kf.split(full_dataset)):
        print(f"Evaluating Fold {fold + 1}/{args.k_folds}")
        print(f"Train indices for fold {fold + 1}: {train_idx}")
        print(f"Validation indices for fold {fold + 1}: {val_idx}")

        # 创建数据加载器
        val_subset = Subset(full_dataset, val_idx)
        val_loader = DataLoader(val_subset, batch_size=args.batch_size, shuffle=False, num_workers=4)

        # 加载当前折叠的模型
        # # cbam
        # model = resnet_cbam_cls.resnet18_cbam(pretrained=False)
        # model.fc = torch.nn.Linear(model.fc.in_features, 2)  # 确保模型输出层是二分类
        # model.load_state_dict(torch.load(f'model_fold_{fold + 1}.pth'))
        # model = model.to(device)  # 将模型移动到设备

        # baseline
        model = models.resnet18(pretrained=False)
        model.conv1 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)  # 4-channel input
        model.fc = nn.Linear(model.fc.in_features, 2)  # 2-class output
        model.load_state_dict(torch.load(f'model_fold_{fold + 1}.pth'))
        model.to(device)

        # 评估模型
        # acc, prec, rec, f1, roc_auc, pr_auc = evaluate_model(model, val_loader, device)
        acc, prec, rec, f1, roc_auc, pr_auc, thr, cm = evaluate_model(
            model, val_loader, device, min_precision=args.min_prec)
        # conf_list.append(cm.astype(np.float32))  # 保存为 float 便于后续平均
        fold_metrics.append((acc, cm))

        print(f"Best threshold (precision≥{args.min_prec}) = {thr:.3f}")

        # 打印并记录结果
        print(f"Fold {fold + 1} Results:")
        print(f"Accuracy   = {acc:.4f}")
        print(f"Precision  = {prec:.4f}")
        print(f"Recall     = {rec:.4f}")
        print(f"F1 Score   = {f1:.4f}")

        if roc_auc is not None:
            print(f"ROC-AUC    = {roc_auc:.4f}")
        else:
            print(f"ROC-AUC    = Not Available")

        # print(f"ROC-AUC    = {roc_auc:.4f}")
        print(f"PR-AUC     = {pr_auc:.4f}")

        fold_results.append((acc, prec, rec, f1, roc_auc, pr_auc))

    # 处理并计算所有折叠的平均结果，忽略 None 值
    # 使用 nanmean 和 nanstd 来忽略 None
    fold_results_no_none = [
        (acc, prec, rec, f1, roc_auc if roc_auc is not None else np.nan, pr_auc)
        for acc, prec, rec, f1, roc_auc, pr_auc in fold_results
    ]

    fold_results_no_none = np.array(fold_results_no_none)

    # -------- ① 选出 acc 最大的前 5 组 --------
    top_k = 5
    k = min(top_k, len(fold_results_no_none))  # 防止不足 5 行
    idx_top = np.argsort(fold_results_no_none[:, 0])[::-1][:k]  # 按 acc 降序
    top_fold_results = fold_results_no_none[idx_top]

    # -------- ② 计算均值与方差（或标准差）--------
    avg_results = np.nanmean(top_fold_results, axis=0)
    std_results = np.nanstd(top_fold_results, axis=0)  # 方差

    # 计算所有折叠的平均结果
    # avg_results, std_results = np.nanmean(fold_results_no_none, axis=0), np.nanstd(fold_results_no_none, axis=0)
    print("\n=== Average Results ===")
    print(f"Average Accuracy   = {avg_results[0]:.4f}")
    print(f"Average Precision  = {avg_results[1]:.4f}")
    print(f"Average Recall     = {avg_results[2]:.4f}")
    print(f"Average F1 Score   = {avg_results[3]:.4f}")
    print(f"Average ROC-AUC    = {avg_results[4]:.4f}" if avg_results[4] is not None else "Average ROC-AUC    = Not Available")
    print(f"Average PR-AUC     = {avg_results[5]:.4f}")

    print(f"Std Accuracy   = {std_results[0]:.4f}")
    print(f"Std Precision  = {std_results[1]:.4f}")
    print(f"Std Recall     = {std_results[2]:.4f}")
    print(f"Std F1 Score   = {std_results[3]:.4f}")
    print(f"Std ROC-AUC    = {std_results[4]:.4f}")
    print(f"Std PR-AUC     = {std_results[5]:.4f}")

    # 混淆矩阵
    top_k = 5
    # ① 按 accuracy 降序排
    fold_metrics_sorted = sorted(fold_metrics, key=lambda x: x[0], reverse=True)

    # ② 取前 k 个（若折数不足自动取全）
    top_metrics = fold_metrics_sorted[:top_k]

    # ③ 把这 k 个折的 cm 累加或求平均
    #   这里用 “平均” => 每个元素 = top-k 折计数平均
    cm_stack = np.stack([m[1] for m in top_metrics], axis=0)
    avg_cm = np.mean(cm_stack, axis=0)  # (2,2)

    # ④ 行归一化便于观看百分比
    norm_cm = avg_cm / avg_cm.sum(axis=1, keepdims=True)

    # ⑤ 绘图
    fig, ax = plt.subplots(figsize=(4, 4))
    sns.heatmap(norm_cm,
                annot=np.round(norm_cm, 2),
                fmt=".2f",
                cmap="Blues",
                xticklabels=['Pred NP', 'Pred P'],
                yticklabels=['True NP', 'True P'],
                cbar=False,
                ax=ax)

    ax.set_title(f"Avg Confusion Matrix of Top {len(top_metrics)} Folds")
    plt.tight_layout()
    plt.savefig("top_folds_confusion_matrix.png")  # 或 plt.show()
    print("\n已保存混淆矩阵至 top_folds_confusion_matrix.png")


# 调用主评估函数
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate K-Fold Models')
    parser.add_argument('--data_dir', type=str,
                        default=r'D:\SIMULATION\p01_DebrisCloudDamageDataBase\DataBase\DataBase_Signal_CWT_pt', help='Path to training data')
    parser.add_argument('--label_np', type=str,
                        default=r'D:\SIMULATION\p01_DebrisCloudDamageDataBase\originalData\original_elementValue_npyFiles\damageM_Data\NP',
                        help='Path to NP label folder')
    parser.add_argument('--label_p', type=str,
                        default=r'D:\SIMULATION\p01_DebrisCloudDamageDataBase\originalData\original_elementValue_npyFiles\damageM_Data\P',
                        help='Path to P label folder')
    parser.add_argument('--gpu', default='cuda:0', type=str, help='GPU ID(s) to use')
    parser.add_argument('--batch_size', default=8, type=int, help='Batch size per GPU')
    parser.add_argument('--k_folds', type=int, default=8, help='Number of folds for cross-validation')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--min_prec', type=float, default=0.80,
                        help='Minimum precision to keep when searching threshold')

    args = parser.parse_args()

    evaluate_all_models(args)
