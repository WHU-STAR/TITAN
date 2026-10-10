"""Partition bucket centroids onto workers with k-means or graph-aware overlap."""

import hnswlib
import numpy as np
from sklearn.cluster import KMeans


def kmeans(
    bucket_centroids: np.ndarray, num_nodes: int, num_overlap: int
) -> np.ndarray:
    """Assign each bucket to its nearest partition centroids.

    Returns an array of shape (num_buckets, num_overlap).
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


def get_graph_neighbors(index: hnswlib.Index, k=10):
    """Return the k graph neighbors of every indexed vector."""
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

