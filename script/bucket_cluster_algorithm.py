"""
This file is used to implement the cluster algorithm like kmeans, adaptive kmeans, graph partition, etc.
"""

import hnswlib
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans


def kmeans(
    bucket_centroids: np.ndarray, num_nodes: int, num_overlap: int
) -> np.ndarray:
    """kmeans算法，为每个数据点分配多个质心
    Args:
        bucket_centroids: 桶的质心数组，shape=(num_buckets, dim)
        num_nodes: 节点数量
        num_overlap: 每个桶分配的最大节点数量
    Returns:
        bucket_node_assignments: 桶到节点的分配数组，shape=(num_buckets, num_overlap)
    """
    num_buckets = bucket_centroids.shape[0]
    bucket_node_assignments = np.zeros((num_buckets, num_overlap), dtype=np.int32)
    bucket_kmeans = KMeans(n_clusters=num_nodes)
    node_centroids = bucket_kmeans.fit(bucket_centroids).cluster_centers_

    # For each bucket, find N closest node centroids
    for i in range(num_buckets):
        # Calculate distances to all node centroids
        distances = np.linalg.norm(bucket_centroids[i] - node_centroids, axis=1)
        # Get indices of N closest nodes
        closest_nodes = np.argsort(distances)[:num_overlap]
        bucket_node_assignments[i] = closest_nodes
    return bucket_node_assignments


def adaptive_kmeans(
    bucket_centroids: np.ndarray, num_nodes: int, num_max_assignments: int = 2
) -> np.ndarray:
    """自适应的kmeans算法，为每个数据点分配多个质心
    Args:
        bucket_centroids: 桶的质心数组，shape=(num_buckets, dim)
        num_nodes: 节点数量
        num_max_assignments: 每个桶分配的最大节点数量
    Returns:
        bucket_node_assignments: 桶到节点的分配数组，shape=(num_buckets, num_overlap)
    """
    # 1. 首先用kmeans得到初始质心
    num_buckets = bucket_centroids.shape[0]
    bucket_kmeans = KMeans(n_clusters=num_nodes)
    node_centroids = bucket_kmeans.fit(bucket_centroids).cluster_centers_

    # 2. 计算质心的密度和平均距离
    centroid_stats = compute_centroid_stats(bucket_centroids, node_centroids)

    # 3. 为每个桶分配节点，初始化为-1表示未分配
    bucket_node_assignments = np.full(
        (num_buckets, num_max_assignments), -1, dtype=np.int32
    )

    for i in range(num_buckets):
        assignments = assign_point(
            bucket_centroids[i],
            node_centroids,
            centroid_stats,
        )
        # 如果没有分配到任何节点，至少分配到最近的节点
        if len(assignments) == 0:
            distances = cdist([bucket_centroids[i]], node_centroids)[0]
            assignments = [np.argmin(distances)]

        # print("number of assignments:", len(assignments))
        bucket_node_assignments[i, : len(assignments)] = assignments

    return bucket_node_assignments


def get_graph_neighbors(index: hnswlib.Index, k=10):
    """
    获取图的邻居
    Args:
        index: hnswlib的index
        k: 每个节点的邻居数量
    Returns:
        neighbors: 每个节点的邻居列表 [num_nodes, k]
    """
    ids = index.get_ids_list()
    vectors = index.get_items(ids)
    neighbors, _ = index.knn_query(vectors, k, num_threads=16)
    return neighbors


def graph_partition(
    bucket_centroids: np.ndarray,
    num_nodes: int,
    threshold: float = 0.6,
    num_max_assignments: int = 2,
    k_neighbors: int = 10,
    num_threads: int = 16,
) -> np.ndarray:
    num_buckets = bucket_centroids.shape[0]
    bucket_node_assignments = np.full(
        (num_buckets, num_max_assignments), -1, dtype=np.int32
    )
    bucket_kmeans = KMeans(n_clusters=num_nodes)
    bucket_node_labels = bucket_kmeans.fit_predict(bucket_centroids)
    meta_hnsw = hnswlib.Index(space="l2", dim=bucket_centroids.shape[1])
    meta_hnsw.init_index(
        max_elements=bucket_centroids.shape[0], ef_construction=200, M=16
    )
    meta_hnsw.add_items(bucket_centroids, num_threads=num_threads)
    # Query bucket_centroids directly instead of round-tripping through
    # get_items() (which materialized a redundant full copy of the centroids).
    neighbors, _ = meta_hnsw.knn_query(
        bucket_centroids, k=k_neighbors, num_threads=num_threads
    )

    # Vectorized assignment: for every bucket at once, count which node each of
    # its k neighbors belongs to, then take the most frequent node != the
    # bucket's own node whose frequency >= threshold. This replaces a Python
    # loop with a per-bucket dict+sort (O(num_buckets) Python iterations).
    neighbor_nodes = bucket_node_labels[neighbors]  # (num_buckets, k)
    counts = np.zeros((num_buckets, num_nodes), dtype=np.int32)
    for j in range(num_nodes):
        counts[:, j] = (neighbor_nodes == j).sum(axis=1)
    rows = np.arange(num_buckets)
    counts[rows, bucket_node_labels] = 0  # exclude each bucket's own node

    order = np.argsort(-counts, axis=1)  # node ids, count descending
    cap_extra = num_max_assignments - 1
    cand_nodes = order[:, :cap_extra]
    cand_counts = np.take_along_axis(counts, cand_nodes, axis=1)
    valid = (cand_counts / k_neighbors) >= threshold

    bucket_node_assignments[:, 0] = bucket_node_labels
    for c in range(cap_extra):
        bucket_node_assignments[:, c + 1] = np.where(
            valid[:, c], cand_nodes[:, c], -1
        )
    return bucket_node_assignments


def assign_point(
    point: np.ndarray,
    centroids: np.ndarray,
    centroid_stats: dict,
    omega: float = 1.2,
    gamma: float = 1.1,
    capacity_threshold: float = 0.1,
    num_max_assignments: int = 2,
) -> list:
    """为单个点分配多个质心
    Args:
        point: 待分配的点，shape=(dim,)
        centroids: 质心数组，shape=(num_centroids, dim)
        centroid_stats: 质心统计信息
        gamma: 高危时的距离阈值系数
        capacity_threshold: 容量阈值
        num_max_assignments: 最大分配数量
    Returns:
        assigned_indices: 分配的质心索引列表
    """
    # 计算点到所有质心的距离
    distances = cdist([point], centroids)[0]
    sorted_indices = np.argsort(distances)

    assigned_indices = []

    for idx in sorted_indices:
        if len(assigned_indices) >= num_max_assignments:
            break

        dist = distances[idx]
        density = centroid_stats["densities"][idx]

        # 1. 检查是否高危（密度超过阈值）
        is_high_risk = density > capacity_threshold
        # print("density", density)

        avg_dist_c = centroid_stats["avg_distances"][idx]
        min_dist_c = centroid_stats["min_distances"][idx]

        # 2. 根据不同情况决定是否分配
        if is_high_risk:
            if dist < gamma * min_dist_c:  # 极端情况，仅当距离非常近时才分配
                assigned_indices.append(idx)
                # 更新质心统计信息
                centroid_stats["points_per_centroid"][idx] += 1
                centroid_stats["avg_distances"][idx] = (
                    centroid_stats["avg_distances"][idx]
                    * (centroid_stats["points_per_centroid"][idx] - 1)
                    + dist
                ) / centroid_stats["points_per_centroid"][idx]
                centroid_stats["min_distances"][idx] = min(
                    dist, centroid_stats["min_distances"][idx]
                )
        else:
            if len(assigned_indices) == 0 or dist < omega * avg_dist_c:
                assigned_indices.append(idx)
                # 更新质心统计信息
                centroid_stats["points_per_centroid"][idx] += 1
                centroid_stats["avg_distances"][idx] = (
                    centroid_stats["avg_distances"][idx]
                    * (centroid_stats["points_per_centroid"][idx] - 1)
                    + dist
                ) / centroid_stats["points_per_centroid"][idx]
                centroid_stats["min_distances"][idx] = min(
                    dist, centroid_stats["min_distances"][idx]
                )
    return assigned_indices


def compute_centroid_stats(points: np.ndarray, centroids: np.ndarray) -> dict:
    """计算质心的统计信息
    Args:
        points: 数据点数组，shape=(num_points, dim)
        centroids: 质心数组，shape=(num_centroids, dim)
    Returns:
        stats: 包含质心统计信息的字典
    """
    num_points = len(points)
    num_centroids = len(centroids)

    # 计算所有点到所有质心的距离矩阵
    distances = cdist(points, centroids)

    # 计算每个质心的统计信息
    stats = {
        "min_distances": np.min(distances, axis=0),  # 每个质心到最近点的距离
        "avg_distances": np.mean(distances, axis=0),  # 每个质心到所有点的平均距离
        "assignments": np.argmin(distances, axis=1),  # 每个点分配到的质心
        "densities": np.zeros(num_centroids),  # 每个质心的密度
        "points_per_centroid": np.zeros(
            num_centroids, dtype=np.int32
        ),  # 每个质心分配的点数，初始为0
    }

    # 计算每个质心的密度（分配到该质心的点数）
    unique, counts = np.unique(stats["assignments"], return_counts=True)
    stats["densities"][unique] = counts / num_points

    return stats
