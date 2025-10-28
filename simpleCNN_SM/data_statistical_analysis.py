import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from scipy.stats import skew, kurtosis, shapiro, zscore
from sklearn.decomposition import PCA
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

from dataset_pt import PreprocessedPTDataset

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
import imageio

def pca_and_cluster_save(inputs_all, outputs_all, dataset_indices, save_dir, k_neighbors=5, n_clusters=3):
    """
    inputs_all: Tensor [N, 4, 224, 224]
    outputs_all: Tensor [N, 1, 151, 151]
    dataset_indices: 样本索引
    """
    os.makedirs(save_dir, exist_ok=True)

    # ===== 1. 每个样本计算4通道均值特征 =====
    features = inputs_all.view(inputs_all.size(0), 4, -1).mean(dim=2).cpu().numpy()

    # ===== 2. PCA到2维 =====
    pca = PCA(n_components=2)
    features_2d = pca.fit_transform(features)
    print(f"[PCA] explained variance: {pca.explained_variance_ratio_}")

    # ===== 3. kNN 平均距离计算（仅作参考分析） =====
    nbrs = NearestNeighbors(n_neighbors=k_neighbors+1).fit(features_2d)
    distances, _ = nbrs.kneighbors(features_2d)
    mean_dist = distances[:, 1:].mean(axis=1)
    print(f"[kNN Density] mean distance: min={mean_dist.min():.4f}, max={mean_dist.max():.4f}")

    # ===== 4. KMeans 聚类成 3 类 =====
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(features_2d)
    for cl in range(n_clusters):
        print(f"[Cluster {cl}] samples: {(labels==cl).sum()}")

    # ===== 5. 保存样本 =====
    def save_sample(idx, input_tensor, output_tensor, folder):
        os.makedirs(folder, exist_ok=True)
        sample_id = dataset_indices[idx]
        # 保存4个通道
        for ch in range(4):
            ch_data = input_tensor[ch].cpu().numpy()
            ch_img = (ch_data * 255).clip(0, 255).astype(np.uint8)
            imageio.imwrite(os.path.join(folder, f"sample_{sample_id}_ch{ch+1}.png"), ch_img)
        # 保存标签
        label_data = output_tensor[0].cpu().numpy()
        label_img = (label_data * 255).clip(0, 255).astype(np.uint8)
        imageio.imwrite(os.path.join(folder, f"sample_{sample_id}_label.png"), label_img)

    for i in range(len(inputs_all)):
        folder = os.path.join(save_dir, f"cluster_{labels[i]}")
        # save_sample(i, inputs_all[i], outputs_all[i], folder)

    # ===== 6. 绘制PCA聚类结果 =====
    plt.figure(figsize=(6, 6))
    colors = ['red', 'blue', 'green']
    for cl in range(n_clusters):
        mask = labels == cl
        plt.scatter(features_2d[mask, 0], features_2d[mask, 1], c=colors[cl], label=f"Cluster {cl}", alpha=0.6)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("PCA Clustering (3 classes)")
    plt.legend()
    plt.tight_layout()
    # plt.savefig(os.path.join(save_dir, "pca_clusters.png"))
    plt.savefig(
        os.path.join(save_dir, "pca_clusters.tiff"),
        dpi=600,  # 分辨率 300 DPI，出版和打印常用
        format='tiff',  # 明确指定 TIFF 格式
        bbox_inches='tight',  # 去掉多余白边
        pil_kwargs={"compression": "tiff_lzw"}  # 使用 LZW 压缩，减小文件体积
    )
    plt.close()


# from sklearn.decomposition import PCA
# from sklearn.neighbors import NearestNeighbors
# import imageio
#
# def pca_and_save_samples(inputs_all, outputs_all, dataset_indices, save_dir,
#                          k_neighbors=5, split_quantile=0.5):
#     """
#     inputs_all: Tensor [N, 4, 224, 224]
#     outputs_all: Tensor [N, 1, 151, 151]
#     dataset_indices: list, 样本在原始dataset中的索引（方便保存时命名）
#     """
#     os.makedirs(save_dir, exist_ok=True)
#     dense_dir = os.path.join(save_dir, "dense_samples")
#     sparse_dir = os.path.join(save_dir, "sparse_samples")
#     os.makedirs(dense_dir, exist_ok=True)
#     os.makedirs(sparse_dir, exist_ok=True)
#
#     # ===== 1. 每个样本计算4通道均值特征 =====
#     features = inputs_all.view(inputs_all.size(0), 4, -1).mean(dim=2).cpu().numpy()
#
#     # ===== 2. PCA到2维 =====
#     pca = PCA(n_components=2)
#     features_2d = pca.fit_transform(features)
#     print(f"[PCA] explained variance: {pca.explained_variance_ratio_}")
#
#     # ===== 3. kNN距离作为密度指标 =====
#     nbrs = NearestNeighbors(n_neighbors=k_neighbors+1).fit(features_2d)
#     distances, _ = nbrs.kneighbors(features_2d)
#     mean_dist = distances[:, 1:].mean(axis=1)  # 去掉自身距离(第0列)
#
#     # ===== 4. 阈值划分密集/稀疏 =====
#     threshold = np.quantile(mean_dist, split_quantile)
#     dense_mask = mean_dist <= threshold
#     sparse_mask = ~dense_mask
#
#     print(f"[Density Split] Dense: {dense_mask.sum()} samples, Sparse: {sparse_mask.sum()} samples")
#
#     # ===== 5. 保存样本 =====
#     def save_sample(idx, input_tensor, output_tensor, folder):
#         sample_id = dataset_indices[idx]
#         # 保存4个通道
#         for ch in range(4):
#             ch_data = input_tensor[ch].cpu().numpy()
#             ch_img = (ch_data * 255).clip(0, 255).astype(np.uint8)
#             imageio.imwrite(os.path.join(folder, f"sample_{sample_id}_ch{ch+1}.png"), ch_img)
#         # 保存标签
#         label_data = output_tensor[0].cpu().numpy()
#         label_img = (label_data * 255).clip(0, 255).astype(np.uint8)
#         imageio.imwrite(os.path.join(folder, f"sample_{sample_id}_label.png"), label_img)
#
#     for i in range(len(inputs_all)):
#         if dense_mask[i]:
#             save_sample(i, inputs_all[i], outputs_all[i], dense_dir)
#         else:
#             save_sample(i, inputs_all[i], outputs_all[i], sparse_dir)
#
#     # ===== 6. 绘制PCA结果散点图并保存 =====
#     plt.figure(figsize=(6, 6))
#     plt.scatter(features_2d[dense_mask, 0], features_2d[dense_mask, 1], c='blue', label='Dense', alpha=0.6)
#     plt.scatter(features_2d[sparse_mask, 0], features_2d[sparse_mask, 1], c='red', label='Sparse', alpha=0.6)
#     plt.xlabel("PC1")
#     plt.ylabel("PC2")
#     plt.title("PCA Sample Distribution")
#     plt.legend()
#     plt.tight_layout()
#     plt.savefig(os.path.join(save_dir, f"pca_scatter_dense_sparse_{split_quantile}.png"))
#     plt.close()

def compute_statistics(tensor, name=""):
    flat = tensor.flatten().cpu().numpy()
    stats = {
        "mean": np.mean(flat),
        "std": np.std(flat),
        "min": np.min(flat),
        "max": np.max(flat),
        "skew": skew(flat),
        "kurtosis": kurtosis(flat)
    }
    print(f"\n[{name}] Statistics:")
    for k, v in stats.items():
        print(f"{k}: {v:.4f}")
    return flat, stats

def detect_outliers(data, threshold=3.0):
    z_scores = zscore(data)
    return np.sum(np.abs(z_scores) > threshold)

def plot_distribution(data, title, save_path):
    plt.figure(figsize=(10, 4))
    sns.histplot(data, bins=50, kde=True)
    plt.title(title)
    plt.tight_layout()
    # plt.savefig(save_path + "_hist_kde.png")
    plt.savefig(
        save_path + "_hist_kde.tiff",
        dpi=600,  # 高分辨率
        format='tiff',  # 明确指定格式
        bbox_inches='tight',  # 去掉多余空白
        pil_kwargs={"compression": "tiff_lzw"}  # 无损压缩
    )

    plt.close()

    plt.figure(figsize=(5, 5))
    sns.boxplot(y=data)
    plt.title(title + " - Boxplot")
    plt.tight_layout()
    # plt.savefig(save_path + "_box.png")
    plt.savefig(
        save_path + "_box.tiff",
        dpi=300,  # 高分辨率
        format='tiff',  # 明确指定格式
        bbox_inches='tight',  # 去掉多余空白
        pil_kwargs={"compression": "tiff_lzw"}  # 无损压缩
    )

    plt.close()

def analyze_input_output_distributions(dataloader, save_dir="analysis_output", max_batches=10):
    os.makedirs(save_dir, exist_ok=True)
    inputs_list = []
    outputs_list = []

    print(f"\nLoading up to {max_batches} batches for statistical analysis...")

    for i, (inputs, targets) in enumerate(tqdm(dataloader, desc="Loading data")):
        if i >= max_batches:
            break
        inputs_list.append(inputs)
        outputs_list.append(targets)

    inputs_all = torch.cat(inputs_list, dim=0)    # [N, 4, 224, 224]
    outputs_all = torch.cat(outputs_list, dim=0)  # [N, 1, 151, 151]

    N = inputs_all.shape[0]

    ### 1~3. 输入通道统计 + 可视化 + 异常值检测 ###
    all_channels = []
    stats_all = []

    for c in range(4):
        channel_data = inputs_all[:, c, :, :].reshape(-1)
        flat, stats = compute_statistics(channel_data, f"Input Channel {c+1}")
        outliers = detect_outliers(flat)
        # print(f"  -> Outliers (|Z| > 3): {outliers} ({100*outliers/len(flat):.2f}%)")
        # stats["outliers (>3σ)"] = outliers
        # plot_distribution(flat, f"Input Channel {c+1}", os.path.join(save_dir, f"input_channel_{c+1}"))
        all_channels.append(flat)
        stats_all.append(stats)
    #
    # # Save all channel stats
    # df_stats = pd.DataFrame(stats_all, index=[f"Channel_{i+1}" for i in range(4)])
    # df_stats.to_csv(os.path.join(save_dir, "input_channel_statistics.csv"))

    ### 4. 通道间 Pearson 相关性 ###
    df_corr = pd.DataFrame(np.array(all_channels).T, columns=[f"C{i+1}" for i in range(4)])
    corr_matrix = df_corr.corr()
    plt.figure(figsize=(6, 5))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
    # plt.title("Input Channel Pearson Correlation")
    plt.tight_layout()
    # plt.savefig(os.path.join(save_dir, "input_channel_correlation.png"))
    plt.savefig(
        os.path.join(save_dir, "input_channel_correlation.tiff"),
        dpi=600,  # 分辨率 300 DPI，出版和打印常用
        format='tiff',  # 明确指定 TIFF 格式
        bbox_inches='tight',  # 去掉多余白边
        pil_kwargs={"compression": "tiff_lzw"}  # 使用 LZW 压缩，减小文件体积
    )
    plt.close()

    ### 5. 输出标签分析 ###
    output_flat = outputs_all.reshape(-1).cpu().numpy()
    output_unique, output_counts = np.unique(output_flat, return_counts=True)
    print(f"\n[Output] Unique values: {dict(zip(output_unique.astype(int), output_counts))}")
    if set(output_unique) <= {0, 1}:
        print("→ 输出为二值图，类别不平衡比例如下：")
        ratio = output_counts / np.sum(output_counts)
        for val, r in zip(output_unique.astype(int), ratio):
            print(f"  Value {val}: {r:.2%}")
    else:
        flat, out_stats = compute_statistics(outputs_all.reshape(-1), "Output")
        plot_distribution(flat, "Output", os.path.join(save_dir, "output"))

        # 正态性检验
        sample = flat[:min(5000, len(flat))]
        stat, p_val = shapiro(sample)
        print(f"\n[Output Distribution] Shapiro-Wilk test:")
        print(f"Statistic={stat:.4f}, p-value={p_val:.4f}")
        if p_val < 0.05:
            print("→ 输出不服从正态分布（拒绝原假设）")
        else:
            print("→ 输出可能近似服从正态分布")

    ### 6. 样本间一致性分析（PCA + t-SNE） ###
    print("\n[PCA] Computing sample-level feature map means...")
    feature_means = inputs_all.mean(dim=(2, 3))  # shape [N, 4]
    feature_np = feature_means.cpu().numpy()

    # --- 二维 PCA ---
    pca_2d = PCA(n_components=2)
    pca_result_2d = pca_2d.fit_transform(feature_np)

    plt.figure(figsize=(6, 6))
    plt.scatter(pca_result_2d[:, 0], pca_result_2d[:, 1], s=15, alpha=0.7)
    # plt.title("PCA (2D) of Input Samples (4D mean features)")
    plt.xlabel(f"PC1 ({pca_2d.explained_variance_ratio_[0] * 100:.1f}%)")
    plt.ylabel(f"PC2 ({pca_2d.explained_variance_ratio_[1] * 100:.1f}%)")
    plt.tight_layout()
    # plt.savefig(os.path.join(save_dir, "pca_sample_distribution_2d.png"))
    plt.savefig(
        os.path.join(save_dir, "pca_sample_distribution_2d.tiff"),
        dpi=600,  # 分辨率 300 DPI，出版和打印常用
        format='tiff',  # 明确指定 TIFF 格式
        bbox_inches='tight',  # 去掉多余白边
        pil_kwargs={"compression": "tiff_lzw"}  # 使用 LZW 压缩，减小文件体积
    )
    plt.close()
    print(f"→ 2D PCA explained variance: {pca_2d.explained_variance_ratio_}")

    # --- 三维 PCA ---
    pca_3d = PCA(n_components=3)
    pca_result_3d = pca_3d.fit_transform(feature_np)

    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(pca_result_3d[:, 0], pca_result_3d[:, 1], pca_result_3d[:, 2],
               c='b', s=15, alpha=0.7)
    # ax.set_title("PCA (3D) of Input Samples")
    ax.set_xlabel(f"PC1 ({pca_3d.explained_variance_ratio_[0] * 100:.1f}%)")
    ax.set_ylabel(f"PC2 ({pca_3d.explained_variance_ratio_[1] * 100:.1f}%)")
    ax.set_zlabel(f"PC3 ({pca_3d.explained_variance_ratio_[2] * 100:.1f}%)")
    plt.tight_layout()
    # plt.savefig(os.path.join(save_dir, "pca_sample_distribution_3d.png"))
    plt.savefig(
        os.path.join(save_dir, "pca_sample_distribution_3d.tiff"),
        dpi=600,  # 分辨率 300 DPI，出版和打印常用
        format='tiff',  # 明确指定 TIFF 格式
        # bbox_inches='tight',  # 去掉多余白边
        pil_kwargs={"compression": "tiff_lzw"}  # 使用 LZW 压缩，减小文件体积
    )
    plt.close()
    print(f"→ 3D PCA explained variance: {pca_3d.explained_variance_ratio_}")

    # --- 二维 t-SNE ---
    print("\n[t-SNE 2D] Running t-SNE dimensionality reduction...")
    tsne_2d = TSNE(n_components=2, perplexity=30, learning_rate=200, n_iter=1000, random_state=42)
    tsne_result_2d = tsne_2d.fit_transform(feature_np)

    plt.figure(figsize=(6, 6))
    plt.scatter(tsne_result_2d[:, 0], tsne_result_2d[:, 1], s=15, alpha=0.7)
    # plt.title("t-SNE (2D) of Input Samples")
    plt.xlabel("t-SNE Dim 1")
    plt.ylabel("t-SNE Dim 2")
    plt.tight_layout()
    # plt.savefig(os.path.join(save_dir, "tsne_sample_distribution_2d.png"))
    plt.savefig(
        os.path.join(save_dir, "tsne_sample_distribution_2d.tiff"),
        dpi=600,  # 分辨率 300 DPI，出版和打印常用
        format='tiff',  # 明确指定 TIFF 格式
        bbox_inches='tight',  # 去掉多余白边
        pil_kwargs={"compression": "tiff_lzw"}  # 使用 LZW 压缩，减小文件体积
    )
    plt.close()

    # --- 三维 t-SNE ---
    print("\n[t-SNE 3D] Running t-SNE dimensionality reduction...")
    tsne_3d = TSNE(n_components=3, perplexity=30, learning_rate=200, n_iter=1000, random_state=42)
    tsne_result_3d = tsne_3d.fit_transform(feature_np)

    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(tsne_result_3d[:, 0], tsne_result_3d[:, 1], tsne_result_3d[:, 2],
               c='g', s=15, alpha=0.7)
    # ax.set_title("t-SNE (3D) of Input Samples")
    ax.set_xlabel("t-SNE Dim 1")
    ax.set_ylabel("t-SNE Dim 2")
    ax.set_zlabel("t-SNE Dim 3")
    plt.tight_layout()
    # plt.savefig(os.path.join(save_dir, "tsne_sample_distribution_3d.png"))
    plt.savefig(
        os.path.join(save_dir, "tsne_sample_distribution_3d.tiff"),
        dpi=600,  # 分辨率 300 DPI，出版和打印常用
        format='tiff',  # 明确指定 TIFF 格式
        # bbox_inches='tight',  # 去掉多余白边
        pil_kwargs={"compression": "tiff_lzw"}  # 使用 LZW 压缩，减小文件体积
    )

    plt.close()

    dataset_indices = list(range(inputs_all.size(0)))
    pca_and_cluster_save(inputs_all, outputs_all, dataset_indices, save_dir, k_neighbors=5, n_clusters=2)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--pt_dir', type=str,
                        default="D:/SIMULATION/p01_DebrisCloudDamageDataBase/DataBase/PreprocessedPT/thinckness_binary_128")
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--max_batches', type=int, default=128)
    parser.add_argument('--save_dir', type=str, default='analysis_output_tiff')
    args = parser.parse_args()

    dataset = PreprocessedPTDataset(args.pt_dir)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    analyze_input_output_distributions(dataloader, save_dir=args.save_dir, max_batches=args.max_batches)
