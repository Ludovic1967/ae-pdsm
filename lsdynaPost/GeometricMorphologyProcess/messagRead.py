import os
import re
import numpy as np
from sklearn.cluster import DBSCAN


def cluster_points(points, eps=0.5, min_samples=3):
    """
    对给定的点进行 DBSCAN 聚类，并计算每个聚类的质心。

    参数:
        points (np.ndarray): 形状为 [n_points, 3] 的点坐标数组
        eps (float): DBSCAN 的 eps 参数
        min_samples (int): DBSCAN 的 min_samples 参数

    返回:
        labels (np.ndarray): 每个点所属聚类的标签，噪声点标记为 -1
        clusters (dict): 每个聚类标签对应的点集合字典（忽略噪声）
        centroids (dict): 每个聚类的质心坐标字典
    """
    db = DBSCAN(eps=eps, min_samples=min_samples)
    labels = db.fit_predict(points)

    clusters = {}
    for label in np.unique(labels):
        if label == -1:
            continue  # 忽略噪声点
        clusters[label] = points[labels == label]

    centroids = {label: np.mean(cluster, axis=0) for label, cluster in clusters.items()}
    return labels, clusters, centroids
def extract_deleted_nodes(vp_path):
    """
    从指定路径下所有以 "messag" 开头的文件中提取被删除的节点编号。

    参数：
        vp_path (str): 包含 messag 文件的文件夹路径

    返回：
        list: 被删除节点编号的列表（字符串形式）
    """
    # 用于匹配格式 "node  number 59888 deleted at time  1.9088E+01"
    pattern_node = re.compile(r'node\s+number\s+(\d+)\s+deleted')
    pattern_element = re.compile(r'solid\s+element\s+(\d+)\s+failed')
    deleted_node_ids = []
    deleted_element_ids = []

    # 遍历 vp_path 文件夹中所有以 "messag" 开头的文件
    for filename in os.listdir(vp_path):
        if filename.startswith("messag") and os.path.isfile(os.path.join(vp_path, filename)):
            file_path = os.path.join(vp_path, filename)
            print(f"Processing file: {file_path}")
            with open(file_path, 'r') as file:
                for line in file:
                    match_node = pattern_node.search(line)
                    if match_node:
                        node_id = match_node.group(1)
                        deleted_node_ids.append(node_id)
                        # print("提取到节点编号：", node_id)
                    match_element = pattern_element.search(line)
                    if match_element:
                        element_id = match_element.group(1)
                        deleted_element_ids.append(element_id)
                        # print("提取到节点编号：", node_id)
    return deleted_node_ids, deleted_element_ids


def fit_surface(coords):
    """
    根据输入坐标数据拟合二次多项式表面函数 z = f(x,y)

    参数：
        coords (np.ndarray): 形状为 [N,3] 的数组，每行依次为 x, y, z

    返回：
        coeffs (np.ndarray): 拟合得到的系数数组 [a0, a1, a2, a3, a4, a5]
                              对应关系 z = a0 + a1*x + a2*y + a3*x**2 + a4*x*y + a5*y**2
    """
    x = coords[:, 0]
    y = coords[:, 1]
    z = coords[:, 2]
    # 构造设计矩阵，采用二次多项式形式
    X = np.column_stack((np.ones_like(x), x, y, x ** 2, x * y, y ** 2))
    # 最小二乘求解
    coeffs, residuals, rank, s = np.linalg.lstsq(X, z, rcond=None)
    return coeffs


def surface_function(coeffs, x, y):
    """
    根据拟合系数计算表面 z 值

    参数：
        coeffs (np.ndarray): 系数数组 [a0, a1, a2, a3, a4, a5]
        x (float or np.ndarray): x 坐标值
        y (float or np.ndarray): y 坐标值

    返回：
        z 值
    """
    return coeffs[0] + coeffs[1] * x + coeffs[2] * y + coeffs[3] * x ** 2 + coeffs[4] * x * y + coeffs[5] * y ** 2


# 如果直接运行该模块，则可以进行简单测试
if __name__ == "__main__":
    # 修改为实际的 vp 文件夹路径
    test_vp_path = r"\\DESKTOP-SVFNCL2\database\MeshSizeCompare\sph2+RP\Bumper1mm\dp4mm\vp453"
    node_ids, element_ids = extract_deleted_nodes(test_vp_path)
    print("总共提取到 %d 个被删除的node编号" % len(node_ids))
    print("总共提取到 %d 个被删除的element编号" % len(element_ids))
