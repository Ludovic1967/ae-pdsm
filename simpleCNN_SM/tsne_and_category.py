import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import imageio
import matplotlib.cm as cm
from torch.utils.data import DataLoader
from dataset_pt_0902 import PreprocessedPTDataset
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.axes_grid1 import make_axes_locatable

journal = {
    "fig_size": (16/2.54, 16/2.54),   # 转换为英寸（matplotlib 使用英寸）
    "font_name": "Times New Roman",
    "font_size": 10,
    "line_width": 1.2,
    "axis_line_width": 1.0,
    "dpi": 1200
}
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN
from sklearn.manifold import TSNE
import os

def auto_select_eps(features, min_samples=5, plot=True, save_path=None):
    """
    自动选择 DBSCAN 的 eps（使用 k-distance 曲线 + 拐点检测）

    Args:
        features: ndarray, shape (n_samples, n_features)
            输入特征
        min_samples: int
            DBSCAN 的 min_samples 参数
        plot: bool
            是否绘制 k-distance 曲线
        save_path: str or None
            保存路径，如果为 None 且 plot=True 则直接 plt.show()
    Returns:
        eps: float
            推荐的 eps 值
    """
    neighbors = NearestNeighbors(n_neighbors=min_samples)
    neighbors_fit = neighbors.fit(features)
    distances, indices = neighbors_fit.kneighbors(features)

    # 第 k 个最近邻的距离
    k_distances = np.sort(distances[:, -1])

    # ---- 拐点检测（最大曲率法）----
    # x = [0,1,2,...], y = k_distances
    x = np.arange(len(k_distances))
    y = k_distances

    # 归一化
    x_norm = (x - x.min()) / (x.max() - x.min())
    y_norm = (y - y.min()) / (y.max() - y.min())

    # 拟合直线 y=x
    line = np.linspace(0, 1, len(x))
    # 计算每个点到直线的垂直距离
    distances_to_line = np.abs(y_norm - line)

    # 最大距离点作为拐点
    knee_index = np.argmax(distances_to_line)
    eps = k_distances[knee_index]

    # ---- 可视化 ----
    if plot:
        plt.figure(figsize=(6, 4))
        plt.plot(k_distances, label="k-distance")
        plt.axvline(knee_index, color="r", linestyle="--", label="knee point")
        plt.axhline(eps, color="g", linestyle="--", label=f"eps={eps:.3f}")
        plt.xlabel("Points sorted by distance")
        plt.ylabel(f"{min_samples}-th NN Distance")
        plt.title("Automatic eps selection (knee detection)")
        plt.legend()
        if save_path:
            plt.savefig(save_path, dpi=300)
            plt.close()
        else:
            plt.show()

    return eps


def tsne_and_dbscan_save(features, labels, save_dir, eps=None, min_samples=5, auto_eps=True):
    os.makedirs(save_dir, exist_ok=True)

    # 自动选择 eps
    if auto_eps and eps is None:
        eps = auto_select_eps(features, min_samples, plot=True,
                              save_path=os.path.join(save_dir, "k_distance.png"))
        print(f"[INFO] 自动选择 eps = {eps:.3f}")

    # t-SNE 降维
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    tsne_results = tsne.fit_transform(features)

    # DBSCAN 聚类
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    cluster_labels = dbscan.fit_predict(tsne_results)

    # 聚类可视化
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(tsne_results[:, 0], tsne_results[:, 1],
                          c=cluster_labels, cmap="tab10", s=10)
    plt.colorbar(scatter, label="Cluster ID")
    plt.title(f"t-SNE + DBSCAN (eps={eps:.3f}, min_samples={min_samples})")
    plt.savefig(os.path.join(save_dir, "tsne_dbscan_clusters.png"), dpi=300)
    plt.close()

    # 真实标签可视化
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(tsne_results[:, 0], tsne_results[:, 1],
                          c=labels, cmap="tab10", s=10)
    plt.colorbar(scatter, label="True Label")
    plt.title("t-SNE + Ground Truth")
    plt.savefig(os.path.join(save_dir, "tsne_ground_truth.png"), dpi=300)
    plt.close()

    return cluster_labels




def pca_and_cluster_save(inputs_all, outputs_all, dataset_indices, save_dir, k_neighbors=50, n_clusters=3):
    """
    inputs_all: Tensor [N, 4, 224, 224]
    outputs_all: Tensor [N, 1, 151, 151]
    dataset_indices: 样本索引
    """
    os.makedirs(save_dir, exist_ok=True)

    # global_min = inputs_all.min().item()
    # global_max = inputs_all.max().item()

    # ===== 1. 每个样本计算4通道均值特征 =====
    features = inputs_all.view(inputs_all.size(0), 4, -1).mean(dim=2).cpu().numpy()

    # ===== 2. t-SNE 到 2维 =====
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
    features_2d = tsne.fit_transform(features)
    print("[t-SNE] finished dimensionality reduction")

    # ===== 3. kNN 平均距离计算（仅作参考分析） =====
    nbrs = NearestNeighbors(n_neighbors=k_neighbors+1).fit(features_2d)
    distances, _ = nbrs.kneighbors(features_2d)
    mean_dist = distances[:, 1:].mean(axis=1)
    print(f"[kNN Density] mean distance: min={mean_dist.min():.4f}, max={mean_dist.max():.4f}")

    # ===== 4. KMeans 聚类成 n_clusters 类 =====
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(features_2d)
    for cl in range(n_clusters):
        print(f"[Cluster {cl}] samples: {(labels==cl).sum()}")

    # # 自动选择 eps + 聚类
    # labels = tsne_and_dbscan_save(features, labels, save_dir="./results",
    #                                       min_samples=10, auto_eps=True)

    # ===== 5. 保存样本 =====
    def save_sample(idx, input_tensor, output_tensor, folder, add_colorbar=True):

        os.makedirs(folder, exist_ok=True)
        sample_id = dataset_indices[idx]

        # 1. 计算该样本 4 通道统一的最小/最大值
        sample_data = input_tensor.cpu().numpy()  # shape (4, H, W)
        vmin = sample_data.min()
        vmax = sample_data.max()

        jet_cmap = cm.get_cmap('jet')

        for ch in range(4):
            ch_data = sample_data[ch]
            # 2. 统一归一化
            norm_data = (ch_data - vmin) / (vmax - vmin + 1e-8)
            norm_data = np.clip(norm_data, 0, 1)
            colored_img = jet_cmap(norm_data)
            rgb_img = (colored_img[:, :, :3] * 255).astype(np.uint8)

            # --- 不带 colorbar 的原图（PIL） ---
            Image.fromarray(rgb_img).save(
                os.path.join(folder, f"sample_{sample_id}_ch{ch + 1}.jpeg"),
                quality=95
            )

            # --- 带 colorbar 的版本（matplotlib） ---
            if ch==0 and add_colorbar:
                fig, ax = plt.subplots(figsize=(3, 3), dpi=300)
                im = ax.imshow(ch_data, cmap=jet_cmap, vmin=vmin, vmax=vmax)
                ax.set_xticks([])
                ax.set_yticks([])

                # 颜色棒放在上方
                divider = make_axes_locatable(ax)
                cax = divider.append_axes("top", size="5%", pad=0.05)
                cbar = plt.colorbar(im, cax=cax, orientation='horizontal')
                cbar.ax.xaxis.set_ticks_position('top')
                cbar.ax.tick_params(labelsize=12)

                plt.savefig(
                    os.path.join(folder,
                                 f"sample_{sample_id}_ch{ch + 1}_with_bar.png"),
                    dpi=300, bbox_inches='tight'
                )
                plt.close(fig)

        # ---------- 标签图（保持不变） ----------
        full_cmap = cm.get_cmap('coolwarm', 256)
        half_cmap = full_cmap(np.linspace(0.0, 0.5, 128))
        half_cmap = cm.colors.ListedColormap(half_cmap)

        label_data = output_tensor[0].cpu().numpy()
        norm_label = (label_data - label_data.min()) / (
                label_data.max() - label_data.min() + 1e-8)
        colored_label = half_cmap(norm_label)
        rgb_label = (colored_label[:, :, :3] * 255).astype(np.uint8)
        Image.fromarray(rgb_label).save(
            os.path.join(folder, f"sample_{sample_id}_label.jpeg"),
            quality=95
        )

    for i in range(len(inputs_all)):
        folder = os.path.join(save_dir, f"cluster_{labels[i]}")
        # 如果需要保存样本就取消注释
        save_sample(i, inputs_all[i], outputs_all[i], folder)

    # ===== 6. 绘制t-SNE聚类结果 =====
    plt.figure(figsize=journal["fig_size"], dpi=journal["dpi"])
    ax = plt.gca()
    ax.set_facecolor("white")

    # 设置字体、坐标轴属性
    plt.rcParams.update({
        "font.family": journal["font_name"],
        "font.size": journal["font_size"],
        "axes.linewidth": journal["axis_line_width"],
    })

    # 用 colormap 生成足够的颜色
    cmap = plt.cm.get_cmap("tab20", len(set(labels)))
    for cl in set(labels):
        mask = labels == cl
        plt.scatter(features_2d[mask, 0], features_2d[mask, 1],
                    color=cmap(cl), label=f"Cluster {cl}", alpha=0.6)

    # 每隔1000个点标注编号
    for i in range(0, len(features_2d), 256):
        plt.text(features_2d[i, 0], features_2d[i, 1], str(dataset_indices[i]),
                 fontsize=6, color='black')

    # 坐标轴标签（支持 LaTeX）
    plt.xlabel(r"tSNE 1", fontsize=journal["font_size"], fontweight="bold")
    plt.ylabel(r"tSNE 2", fontsize=journal["font_size"], fontweight="bold")

    # 图例
    plt.legend(frameon=False, loc="best", fontsize=journal["font_size"], ncol=1)
    # plt.title(f"t-SNE Clustering ({n_clusters} classes)")
        # plt.tight_layout()
    # 保存高分辨率 tiff
    plt.savefig(
        os.path.join(save_dir, "tsne_clusters.tiff"),
        dpi=journal["dpi"],
        format="tiff",
        bbox_inches="tight",
        pil_kwargs={"compression": "tiff_lzw"}
    )
    plt.close()


if __name__ == "__main__":
    import argparse
    import torch

    parser = argparse.ArgumentParser()
    parser.add_argument('--pt_dir', type=str,
                        default="D:/SIMULATION/p01_DebrisCloudDamageDataBase/DataBase/PreprocessedPT/thinckness_binary_128")
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--max_batches', type=int, default=128)
    parser.add_argument('--save_dir', type=str, default='analysis_output_tiff_0903')
    args = parser.parse_args()

    # 加载数据
    dataset = PreprocessedPTDataset(args.pt_dir)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    all_inputs, all_outputs, all_indices = [], [], []
    for batch_idx, (inputs, outputs, indices) in enumerate(dataloader):
        all_inputs.append(inputs)
        all_outputs.append(outputs)
        all_indices.extend(indices)

        if batch_idx + 1 >= args.max_batches:
            break

    # 拼接
    inputs_all = torch.cat(all_inputs, dim=0)
    outputs_all = torch.cat(all_outputs, dim=0)
    # dataset_indices = [int(i) for i in all_indices]
    dataset_indices = all_indices  # 保留字符串，不转 int

    # 调用主函数
    pca_and_cluster_save(inputs_all, outputs_all, dataset_indices, save_dir=args.save_dir, n_clusters=6)
