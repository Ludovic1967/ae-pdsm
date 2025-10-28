import os
import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Subset, DataLoader
from sklearn.model_selection import KFold
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib as mpl
import matplotlib.pyplot as plt
import argparse
from sklearn.cluster import KMeans

from model.model_cnn_151 import SimpleCNN
from dataset_pt import PreprocessedPTDataset

# ---------------- Journal style (统一期刊输出) ----------------
journal = {
    "fig_size": (12/2.54, 11/2.54),  # cm -> inch
    "font_name": "Arial",  # 可改 "Arial"
    "font_size": 15,
    "line_width": 1.2,
    "axis_line_width": 1.0,
    "dpi": 1200
}
SAVE_EXT = "tiff"  # 可改 "pdf"/"png"/"svg"/"tiff"

def apply_journal_style(j):
    mpl.rcParams.update({
        # 尺寸/分辨率
        "figure.figsize": j["fig_size"],
        "savefig.dpi": j["dpi"],
        # 字体与字号
        "font.family": j["font_name"],
        "font.size": j["font_size"],
        "axes.labelsize": j["font_size"],
        "axes.titlesize": j["font_size"],
        "xtick.labelsize": j["font_size"],
        "ytick.labelsize": j["font_size"],
        "legend.fontsize": j["font_size"],
        # 线宽与坐标轴
        "lines.linewidth": j["line_width"],
        "axes.linewidth": j["axis_line_width"],
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.major.size": 3,
        "ytick.major.size": 3,
        "xtick.minor.size": 1.5,
        "ytick.minor.size": 1.5,
        "xtick.major.width": j["axis_line_width"],
        "ytick.major.width": j["axis_line_width"],
        "xtick.minor.width": j["axis_line_width"],
        "ytick.minor.width": j["axis_line_width"],
        "legend.frameon": False,
        "axes.grid": False,
        # 嵌入字体/数学字体
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "mathtext.fontset": "stix",
    })

def savefig_journal(fig, path, j, ext="tiff"):
    ext = ext.lower()
    kwargs = dict(dpi=j["dpi"], bbox_inches="tight")
    if ext in ("tif", "tiff"):
        kwargs.update(format="tiff", pil_kwargs={"compression": "tiff_lzw"})
    fig.savefig(path, **kwargs)

apply_journal_style(journal)
# ---------------------------------------------------------

def extract_features(model, dataloader, device):
    model.eval()
    all_features, all_outputs = [], []
    with torch.no_grad():
        for x, _ in dataloader:  # 不需要标签
            x = x.to(device)
            encoded = model.encoder(x)  # [B, C, H, W]
            pooled = F.adaptive_avg_pool2d(encoded, (1, 1)).squeeze(-1).squeeze(-1)  # [B, C]
            all_features.append(pooled.cpu().numpy())

            pred = model(x)
            output_mean = pred.view(pred.size(0), -1).mean(dim=1).cpu().numpy()
            all_outputs.append(output_mean)

    return np.concatenate(all_features), np.concatenate(all_outputs)

# ---------------- 可视化（train/val 合并） ----------------
def visualize_train_val_together(features_train, preds_train,
                                 features_val, preds_val,
                                 method='pca', fold_idx=0,
                                 save_dir='./analysis', n_components=2,
                                 jconf=journal, save_ext=SAVE_EXT):
    os.makedirs(save_dir, exist_ok=True)

    features_all = np.concatenate([features_train, features_val], axis=0)
    set_labels = np.array([0] * len(features_train) + [1] * len(features_val))

    # 降维
    if method == 'pca':
        reducer = PCA(n_components=n_components)
    elif method == 'tsne':
        reducer = TSNE(n_components=n_components, random_state=42, init='pca')
    else:
        raise ValueError("Method must be 'pca' or 'tsne'.")
    reduced = reducer.fit_transform(features_all)

    # 绘图（期刊风格）
    if n_components == 2:
        fig, ax = plt.subplots(figsize=jconf["fig_size"])
        sc = ax.scatter(reduced[:, 0], reduced[:, 1],
                        c=set_labels, cmap='coolwarm', s=10, alpha=0.8)
        ax.set_xlabel('Component 1')
        ax.set_ylabel('Component 2')
        ax.minorticks_on()
        ax.tick_params(top=True, right=True)
        # 图例（自定义标签）
        handles, _ = sc.legend_elements()
        ax.legend(handles, ['Train', 'Val'], loc='best', frameon=False)
    elif n_components == 3:
        fig = plt.figure(figsize=jconf["fig_size"])
        ax = fig.add_subplot(111, projection='3d')
        sc = ax.scatter(reduced[:, 0], reduced[:, 1], reduced[:, 2],
                        c=set_labels, cmap='coolwarm', s=10, alpha=0.8)
        ax.set_xlabel('Component 1')
        ax.set_ylabel('Component 2')
        ax.set_zlabel('Component 3')
        # 3D 也尽量简洁
        for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
            axis._axinfo["grid"]['linewidth'] = 0.3
        handles, _ = sc.legend_elements()
        ax.legend(handles, ['Train', 'Val'], loc='best', frameon=False)
    else:
        print(f"[WARNING] Cannot plot n_components={n_components}; skipping.")
        return

    fig.tight_layout()
    save_path = os.path.join(save_dir, f"{method}_fold{fold_idx}_trainval_dim{n_components}.{save_ext}")
    savefig_journal(fig, save_path, jconf, ext=save_ext)
    plt.close(fig)

# ---------------- 单集可视化（train 或 val） ----------------
def visualize(features, preds, method='pca', fold_idx=0,
              save_dir='./analysis', n_components=2,
              jconf=journal, save_ext=SAVE_EXT):
    os.makedirs(save_dir, exist_ok=True)

    mean_preds = preds
    # 依据均值分箱（示例 4 桶）
    bins = np.linspace(mean_preds.min(), mean_preds.max(), num=4)
    pseudo_labels = np.digitize(mean_preds, bins) - 1

    # 降维
    if method == 'pca':
        reducer = PCA(n_components=n_components)
    elif method == 'tsne':
        reducer = TSNE(n_components=n_components, random_state=42, init='pca')
    else:
        raise ValueError("Method must be 'pca' or 'tsne'.")
    reduced = reducer.fit_transform(features)

    # 绘图（期刊风格）
    if n_components == 2:
        fig, ax = plt.subplots(figsize=jconf["fig_size"])
        sc = ax.scatter(reduced[:, 0], reduced[:, 1],
                        c=pseudo_labels, cmap='tab10', s=10, alpha=0.8)
        ax.set_xlabel('Component 1')
        ax.set_ylabel('Component 2')
        ax.minorticks_on()
        ax.tick_params(top=True, right=True)
        # 如需色条可启用：
        # cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
        # cbar.set_label('Pseudo Class')
    elif n_components == 3:
        fig = plt.figure(figsize=jconf["fig_size"])
        ax = fig.add_subplot(111, projection='3d')
        sc = ax.scatter(reduced[:, 0], reduced[:, 1], reduced[:, 2],
                        c=pseudo_labels, cmap='tab10', s=10, alpha=0.8)
        ax.set_xlabel('Component 1')
        ax.set_ylabel('Component 2')
        ax.set_zlabel('Component 3')
        for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
            axis._axinfo["grid"]['linewidth'] = 0.3
    else:
        print(f"[WARNING] Cannot plot n_components={n_components}; skipping.")
        return

    fig.tight_layout()
    save_path = os.path.join(save_dir, f"{method}_fold{fold_idx}_dim{n_components}.{save_ext}")
    savefig_journal(fig, save_path, jconf, ext=save_ext)
    plt.close(fig)

# ---------------- 主流程 ----------------
def main(pt_dir, param_dir, opt, batch_size, num_workers, device, k_folds, method='tsne'):
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    dataset = PreprocessedPTDataset(pt_dir)
    print(f"Loaded dataset with {len(dataset)} samples.")

    kf = KFold(n_splits=k_folds, shuffle=False)

    for fold, (train_idx, val_idx) in enumerate(kf.split(dataset)):
        print(f"\n--- Feature Visualization: Fold {fold + 1} ---")

        fold_dir = os.path.join(param_dir, f"fold_{fold + 1}")
        model_path = os.path.join(fold_dir, f"{opt}_PTModel_fold{fold + 1}_epoch500.pth")
        if not os.path.exists(model_path):
            print(f"[WARNING] Model not found: {model_path}")
            continue

        train_subset = Subset(dataset, train_idx)
        val_subset   = Subset(dataset, val_idx)
        print(f"Train samples: {len(train_subset)}, Val samples: {len(val_subset)}")

        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        val_loader   = DataLoader(val_subset,   batch_size=batch_size, shuffle=False, num_workers=num_workers)

        model = SimpleCNN(opt=opt, drop_path_rate=0.2).to(device)
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint)

        saveDir = './feature_distribution_analysis/test_analysis3_tiff'
        os.makedirs(saveDir, exist_ok=True)

        # 提取 & 可视化训练集
        features_train, pseudo_train = extract_features(model, train_loader, device)
        visualize(features_train, pseudo_train, method='pca',  fold_idx=f"{fold}_train", save_dir=saveDir,
                  n_components=2, jconf=journal, save_ext=SAVE_EXT)
        visualize(features_train, pseudo_train, method='pca',  fold_idx=f"{fold}_train", save_dir=saveDir,
                  n_components=3, jconf=journal, save_ext=SAVE_EXT)
        visualize(features_train, pseudo_train, method='tsne', fold_idx=f"{fold}_train", save_dir=saveDir,
                  n_components=2, jconf=journal, save_ext=SAVE_EXT)
        visualize(features_train, pseudo_train, method='tsne', fold_idx=f"{fold}_train", save_dir=saveDir,
                  n_components=3, jconf=journal, save_ext=SAVE_EXT)

        # 提取 & 可视化验证集
        features_val, pseudo_val = extract_features(model, val_loader, device)
        visualize(features_val, pseudo_val, method='pca',  fold_idx=f"{fold}_val", save_dir=saveDir,
                  n_components=2, jconf=journal, save_ext=SAVE_EXT)
        visualize(features_val, pseudo_val, method='pca',  fold_idx=f"{fold}_val", save_dir=saveDir,
                  n_components=3, jconf=journal, save_ext=SAVE_EXT)
        visualize(features_val, pseudo_val, method='tsne', fold_idx=f"{fold}_val", save_dir=saveDir,
                  n_components=2, jconf=journal, save_ext=SAVE_EXT)
        visualize(features_val, pseudo_val, method='tsne', fold_idx=f"{fold}_val", save_dir=saveDir,
                  n_components=3, jconf=journal, save_ext=SAVE_EXT)

        # 合并可视化
        visualize_train_val_together(features_train, pseudo_train, features_val, pseudo_val,
                                     method='pca',  fold_idx=fold, save_dir=saveDir, n_components=2,
                                     jconf=journal, save_ext=SAVE_EXT)
        visualize_train_val_together(features_train, pseudo_train, features_val, pseudo_val,
                                     method='pca',  fold_idx=fold, save_dir=saveDir, n_components=3,
                                     jconf=journal, save_ext=SAVE_EXT)
        visualize_train_val_together(features_train, pseudo_train, features_val, pseudo_val,
                                     method='tsne', fold_idx=fold, save_dir=saveDir, n_components=2,
                                     jconf=journal, save_ext=SAVE_EXT)
        visualize_train_val_together(features_train, pseudo_train, features_val, pseudo_val,
                                     method='tsne', fold_idx=fold, save_dir=saveDir, n_components=3,
                                     jconf=journal, save_ext=SAVE_EXT)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pt_dir', type=str,
                        default=r'D:\SIMULATION\p01_DebrisCloudDamageDataBase\DataBase\PreprocessedPT\thinckness_binary_128')
    parser.add_argument('--opt', type=str, default='EPS')
    parser.add_argument('--param_dir', type=str,
                        default=r'D:\SIMULATION\p01_DebrisCloudDamageDataBase\params\thickness_binary_128')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--k_folds', type=int, default=8)
    parser.add_argument('--method', type=str, choices=['tsne', 'pca'], default='tsne')
    args = parser.parse_args()
    main(**vars(args))
