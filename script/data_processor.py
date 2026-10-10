import os
import sys
import yaml
import json
from typing import Literal
from node import Node, Bucket
import gc
import numpy as np
from tqdm import tqdm
from data_loader import get_dataset
from utils import Timer, generate_path_script
from pyfzf.pyfzf import FzfPrompt
# torch is only used by commented-out GPU kmeans code; kahip is only used by the
# "pyramid" partition method. Guard both so the HNSW-shard ("similarity") path
# can run without them installed.
try:
    import torch
except ImportError:
    torch = None
try:
    import kahip
except ImportError:
    kahip = None
import random
import struct
import hnswlib
import networkx as nx
from bucket_cluster_algorithm import adaptive_kmeans, graph_partition, kmeans
from hierarchical_kmeans import hierarchical_KMeans
from data_writer import (
    save_node_data,
    save_node_data_streamed,
    save_meta_index_data,
)

EF = 200  # ef_construction
M = 32
THREAD_NUM = -1
NUM_OVERLAP = 1  # overlap num of nodes, default set to 2, equals when using similarity-based partition, overlapping item to 2 nodes.

# DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
DEVICE = "cpu"

def np_find_nearest_point_idx_and_furthest_distance(
    data: np.ndarray,
    target: np.ndarray,
    indices: np.ndarray,
    space: str = "l2",
):
    """
    given data and target, find the index of the nearest point in data, and return the absolute index in original data using indices.
    """
    if len(indices) == 0:
        raise ValueError("indices cannot be empty.")
    if space == "l2":
        distances = np.sum((data - target) ** 2, axis=1)
        distances = np.sqrt(distances)
    elif space == "cosine":
        # TODO: figure out
        data_norm = np.linalg.norm(data, axis=1)
        target_norm = np.linalg.norm(target)
        distances = 1 - np.dot(data, target) / (data_norm * target_norm)
    closest_idx_in_subset = np.argmin(distances)
    closest_idx = indices[closest_idx_in_subset]
    furthest_distance = np.max(distances)
    return closest_idx, furthest_distance

def pre_partition_buckets(
    schema_name: str,
    dataset_name: str,
    method_name: str,
    data: np.ndarray,
    bucket_labels: np.ndarray,
    bucket_centroids: np.ndarray,
    num_nodes: int,
    bucket_cluster_method: Literal[
        "kmeans", "adaptive_kmeans", "graph_partition"
    ] = "kmeans",
    graph_type: Literal["hnsw", "bf", "both"] = "hnsw",
    M: int = M,
    ef_construction: int = EF,
    num_threads: int = THREAD_NUM,
    num_overlap: int = NUM_OVERLAP,
):
    num_buckets = bucket_centroids.shape[0]
    # t_bucket_labels = torch.from_numpy(bucket_labels).to(DEVICE)
    labels = np.arange(len(data))
    # t_labels = torch.arange(len(data)).to(DEVICE)

    # Initialize a matrix to store node assignments for each bucket
    # Shape: (num_buckets, num_overlap)
    bucket_node_assignments = np.zeros((num_buckets, num_overlap), dtype=np.int32)

    if bucket_cluster_method == "kmeans":
        bucket_node_assignments = kmeans(bucket_centroids, num_nodes, num_overlap)
    elif bucket_cluster_method == "adaptive_kmeans":
        bucket_node_assignments = adaptive_kmeans(bucket_centroids, num_nodes)
    elif bucket_cluster_method == "graph_partition":
        bucket_node_assignments = graph_partition(bucket_centroids, num_nodes, num_max_assignments=num_overlap)

    # Save the bucket node assignments and centroids
    if bucket_cluster_method == "adaptive_kmeans":
        dirpath = (
            f"../data/{schema_name}_worker_data_hnsw_{method_name}_adaptive_kmeans"
        )
    elif bucket_cluster_method == "graph_partition":
        dirpath = (
            f"../data/{schema_name}_worker_data_hnsw_{method_name}_graph_partition"
        )
    elif num_overlap > 1:
        dirpath = f"../data/{schema_name}_worker_data_hnsw_{method_name}_overlap_{num_overlap}"
    else:
        dirpath = f"../data/{schema_name}_worker_data_hnsw_{method_name}"

    if not os.path.exists(dirpath):
        os.makedirs(dirpath)

    timer = Timer()

    # # Create a mapping from bucket to its data and indices
    # timer.tick()
    # bucket_data_map = {}
    # for bucket_idx in tqdm(range(num_buckets), desc="Preparing bucket data"):
    #     t_current_bucket_data = t_data[t_bucket_labels == bucket_idx]
    #     t_current_bucket_indices = t_labels[t_bucket_labels == bucket_idx]
    #     bucket_data_map[bucket_idx] = (
    #         t_current_bucket_data.cpu().numpy(),
    #         t_current_bucket_indices.cpu().numpy(),
    #     )
    # timer.tuck("Original method")

    # 尝试优化
    timer.tick()

    # 获取排序索引
    sort_indices = np.argsort(bucket_labels)
    sorted_bucket_labels = bucket_labels[sort_indices]
    # sorted_data = data[sort_indices]
    # sorted_labels = labels[sort_indices]

    # 找到每个 bucket 的起始位置
    unique_buckets, bucket_starts = np.unique(sorted_bucket_labels, return_index=True)
    bucket_ends = np.append(bucket_starts[1:], len(sorted_bucket_labels))

    # 构建 bucket_data_map
    bucket_data_map = {}
    for i, bucket_idx in enumerate(tqdm(unique_buckets, desc="Preparing bucket data")):
        start = bucket_starts[i]
        end = bucket_ends[i]
        bucket_data_map[bucket_idx] = sort_indices[start:end]
            # sorted_data[start:end],
            # sort_indices[start:end],
            # data[sort_indices[start:end]],
            
    

    timer.tuck("Optimized method - sorting and bucketing")

    # timer.tick()
    # df = pd.DataFrame({
    #     'bucket': t_bucket_labels.cpu().numpy(),
    #     'idx': t_labels.cpu().numpy()
    # })
    #
    # bucket_data_map = {}
    # for bucket_idx, group in df.groupby('bucket'):
    #     indices = group['idx'].values
    #     bucket_data_map[bucket_idx] = (data[indices], indices)
    # timer.tuck("Optimized method - pandas grouping")

    # from IPython import embed; embed()

    # 为每个节点创建数据。Streamed: each node's buckets are gathered in chunks
    # straight from the base memmap; one gather feeds both hnswlib (incremental
    # add_items) and the streamed worker-bin writer. Replaces the old per-bucket
    # loop (one scattered gather + Bucket object per bucket, then a giant
    # np.concatenate of ~100k arrays) which was the dominant build bottleneck.
    NODE_CHUNK_ROWS = 500_000  # 500k x dim x 4B per gather chunk (256MB at 128d)
    for i in range(num_nodes):
        # Find all buckets that should be assigned to this node (vectorized;
        # the old per-bucket membership loop was num_buckets Python iterations
        # per node).
        node_bucket_indices = np.flatnonzero(
            (bucket_node_assignments == i).any(axis=1)
        )
        node_bucket_indices = np.array(
            [b for b in node_bucket_indices if b in bucket_data_map],
            dtype=np.int64,
        )
        if node_bucket_indices.size == 0:  # Skip if no buckets assigned
            continue

        bucket_indices_list = [bucket_data_map[b] for b in node_bucket_indices]
        total_rows = sum(len(idx) for idx in bucket_indices_list)
        print(
            f"Node {i}: {node_bucket_indices.size} buckets, {total_rows} vectors (streamed)"
        )

        hnswIndex = None
        if graph_type in ("hnsw", "both"):
            hnswIndex: hnswlib.Index = hnswlib.Index(space="l2", dim=data.shape[1])
            hnswIndex.init_index(
                max_elements=total_rows, ef_construction=ef_construction, M=M
            )
        bfIndex = None
        if graph_type in ("bf", "both"):
            bfIndex: hnswlib.Index = hnswlib.BFIndex(space="l2", dim=data.shape[1])
            bfIndex.init_index(max_elements=total_rows)

        def _feed_hnsw(chunk, ids, _h=hnswIndex, _b=bfIndex):
            if _h is not None:
                _h.add_items(chunk, ids=ids, num_threads=num_threads)
            if _b is not None:
                _b.add_items(chunk, ids=ids)

        save_node_data_streamed(
            f"{dirpath}/worker-{i}.bin",
            data.shape[1],
            node_bucket_indices.astype(np.uint32),
            bucket_indices_list,
            data,
            on_bucket_data=_feed_hnsw,
            chunk_rows=NODE_CHUNK_ROWS,
        )

        if hnswIndex is not None:
            hnswIndex.save_index(f"{dirpath}/worker-{i}_hnsw.bin")
            del hnswIndex
        if bfIndex is not None:
            bfIndex.save_index(f"{dirpath}/worker-{i}_bf.bin")
            del bfIndex
        collected = gc.collect()
        print(f"Garbage collector: collected {collected} objects.")

    # Create and save meta index
    nearest_point_labels = np.zeros((bucket_centroids.shape[0]), dtype=np.int32)
    bucket_radiuses = np.zeros((bucket_centroids.shape[0]), dtype=np.float32)

    # --------------------------------- WITH_SORTED ---------------------------------
    for _b in tqdm(range(int(bucket_labels.max() + 1)), desc="Finding nearest points for each bucket centroid"):
        if _b not in bucket_data_map:
            print(f"Warning: Bucket {_b} not found in bucket_data_map")
            nearest_point_labels[_b] = -1  # or some sentinel value
            continue
        current_bucket_centroid = bucket_centroids[_b]
        current_bucket_indices = bucket_data_map[_b]  # 绝对索引
        bucket_data = data[current_bucket_indices]
        if len(bucket_data) == 0:
            print("Warning: No data points for bucket", _b)
            continue
        nearest_point_index, furthest_distance = (
            np_find_nearest_point_idx_and_furthest_distance(
                bucket_data,
                current_bucket_centroid,
                indices=current_bucket_indices,
                space="l2",
            )
        )
        nearest_point_labels[_b] = nearest_point_index
        bucket_radiuses[_b] = furthest_distance
    # nearest_indices = torch.from_numpy(nearest_point_labels).to(DEVICE)
    # furthest_distances = torch.from_numpy(bucket_radiuses).to(DEVICE)

    # from IPython import embed; embed()
    # # 批量计算所有bucket的最近点和最远距离
    # t_centroids = torch.from_numpy(bucket_centroids).float().to(DEVICE)
    # nearest_indices, furthest_distances = find_nearest_point_idx_and_furthest_distance_optimized(
    #     t_data,
    #     t_centroids,
    #     t_bucket_labels,
    #     space="l2"
    # )

    # # 将结果填充到对应的数组中
    # # unique_buckets = np.unique(bucket_labels)
    # num_skipped = 0
    # for i, bucket_id in enumerate(unique_buckets):
    #     if nearest_indices[i] == -1:
    #         num_skipped += 1
    #         continue
    # if num_skipped > 0:
    #     print(f"{num_skipped} buckets were skipped due to no data points.")

    metaHNSW = hnswlib.Index(space="l2", dim=bucket_centroids.shape[1])
    metaHNSW.init_index(
        max_elements=bucket_centroids.shape[0], ef_construction=ef_construction, M=M
    )
    metaHNSW.add_items(bucket_centroids, num_threads=num_threads)
    metaHNSW.save_index(f"{dirpath}/meta_hnsw.bin")

    bucket_sizes = np.zeros((bucket_centroids.shape[0],), dtype=np.uint32)
    for bucket_idx, indices in bucket_data_map.items():
        bucket_sizes[bucket_idx] = len(indices)

    # Save meta index data - now we save the first assigned node for each bucket
    save_meta_index_data(
        bucket_centroids,
        bucket_node_assignments,
        nearest_point_labels,
        bucket_radiuses,
        f"{dirpath}/meta_index_data.bin",
        bucket_sizes=bucket_sizes,
        total_num_vectors=len(data),
    )

    # --------------------------------- VALIDATION ---------------------------------
    # 验证每个 node 的所有 bucket 的 nearest point 是否在该 node 的 absolute_label_indices 中
    print("Randomly validating nearest points for each node...")

    num_checks_per_node = min(10, num_buckets)  # You can adjust this number
    total_validation_rounds = 0
    failed_validations = 0
    for i in range(num_nodes):
        # Find all buckets assigned to this node
        node_bucket_indices = []

        node_bucket_indices = np.where((bucket_node_assignments == i).any(axis=1))[0].tolist()

        if not node_bucket_indices:
            continue
        # Randomly sample buckets for this node
        sampled_bucket_indices = random.sample(node_bucket_indices, min(num_checks_per_node, len(node_bucket_indices)))
        # Load this node's absolute_label_indices
        node_bin_path = f"{dirpath}/worker-{i}.bin"
        with open(node_bin_path, "rb") as f:
            # Skip header (5 * 4 bytes)
            f.seek(5 * 4)
            # Read bucket_indices count
            bucket_indices_count = len(node_bucket_indices)
            # Read absolute_label_indices_count
            f.seek(0)
            header = struct.unpack("<IIIII", f.read(5 * 4))
            absolute_indices_count = header[3]
            # Skip bucket_indices
            f.seek(5 * 4 + header[2] * 4)
            # Read absolute_label_indices
            absolute_label_indices = np.frombuffer(
                f.read(absolute_indices_count * 4), dtype=np.int32
            )
            
        # Check each sampled bucket's nearest point
        for bucket_idx in sampled_bucket_indices:
            total_validation_rounds += 1
            nearest_point = nearest_point_labels[bucket_idx]
            if nearest_point not in absolute_label_indices:
                print(
                    f"Validation failed: Node {i}, bucket {bucket_idx}, nearest point {nearest_point} not in node's absolute_label_indices"
                )
            # Additionally, check if it is truly the nearest point
            bucket_indices = bucket_data_map[bucket_idx]
            bucket_data = data[bucket_indices]
            centroid = bucket_centroids[bucket_idx]
            if len(bucket_data) == 0:
                continue
            dists = np.sum((bucket_data - centroid) ** 2, axis=1)
            min_idx = np.argmin(dists)
            actual_nearest = bucket_indices[min_idx]

            if actual_nearest != nearest_point:
                print(
                    f"Nearest point mismatch: Node {i}, bucket {bucket_idx}, expected {actual_nearest}, got {nearest_point}"
                )
                failed_validations += 1
    print("Random validation finished.")
    print(f"Result: {total_validation_rounds - failed_validations}/{total_validation_rounds} checks passed.")
    # Save the full node assignment information for overlapping
    np.save(f"{dirpath}/bucket_node_assignments.npy", bucket_node_assignments)

    generate_path_script(dirpath, dataset_name)


def hash_partition_buckets(
    dataset_name: str,
    method_name: str,
    data: np.ndarray,
    num_nodes: int,
    graph_type: Literal["hnsw", "bf", "both"] = "hnsw",
    M: int = M,
    ef_construction: int = EF,
    num_threads: int = THREAD_NUM,
):
    """
    Hash partitioning method:
    1. Hash the data using a hash function based on data indices
    2. Distribute the data to the nodes based on the hash value
    3. Build local index for each node
    4. Save meta data containing node assignments and labels
    """
    print(f"Starting Hash partitioning, num_nodes={num_nodes}")

    # Step 1: Hash the data using data indices
    data_indices = np.arange(len(data))
    hash_values = data_indices % num_nodes

    print(
        f"Hash partitioning completed, distributing {len(data)} points to {num_nodes} nodes"
    )

    # Step 2: Distribute the data to nodes based on hash values (optimized)
    def split_data_by_hash(data, hash_values, num_nodes):
        """Split data based on hash values using efficient numpy operations"""
        # Use numpy's advanced indexing for much faster data splitting
        data_by_node = []
        labels_by_node = []

        for node_id in range(num_nodes):
            # Find all indices that belong to this node
            mask = hash_values == node_id
            if np.any(mask):
                # Use boolean indexing for efficient data extraction
                node_data = data[mask]
                node_labels = np.where(mask)[0]  # Get the original global indices
            else:
                # Create empty arrays with correct shape
                node_data = np.array([]).reshape(0, data.shape[1])
                node_labels = np.array([])

            data_by_node.append(node_data)
            labels_by_node.append(node_labels)

        return data_by_node, labels_by_node

    dataList, labelsList = split_data_by_hash(data, hash_values, num_nodes)
    print("Data distribution by hash completed")

    # Step 3: Create output directory
    dirpath = f"../data/{dataset_name}_worker_data_hnsw_{method_name}"
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)

    # Step 4: Build local index for each node with multiple buckets (IVF structure)
    for i in range(num_nodes):
        if len(dataList[i]) == 0:
            print(f"Warning: Node {i} has no data assigned")
            # Create empty node
            node = Node.from_buckets(
                [],
                np.array([]),
                np.array([]),
            )
        else:
            # Optimized bucket creation: create all buckets at once
            # Each bucket represents one IVF list/cluster (one vector per bucket)
            num_buckets = len(dataList[i])

            # Pre-allocate arrays for better performance
            # Use global bucket IDs instead of local ones (0, 1, 2, ...)
            # Each vector's global ID is its absolute label index
            bucket_indices = labelsList[i].astype(
                np.int32
            )  # Use global vector IDs as bucket IDs
            centroids = dataList[i].copy()  # Each vector is its own centroid

            # Create optimized node structure
            data = dataList[i]
            absolute_label_indices = labelsList[i]
            data_size_per_bucket = np.ones(
                num_buckets, dtype=np.int32
            )  # Each bucket has 1 vector

            # Create node directly without individual bucket objects
            node = Node(
                data=data,
                data_size_per_bucket=data_size_per_bucket,
                bucket_indices=bucket_indices,
                absolute_label_indices=absolute_label_indices,
                centroids=centroids,
            )

        # Build local index (same as naive method)
        if graph_type in ("hnsw", "both"):
            hnswIndex: hnswlib.Index = hnswlib.Index(space="l2", dim=node.data.shape[1])
            hnswIndex.init_index(
                max_elements=len(node.data), ef_construction=ef_construction, M=M, num_threads=num_threads
            )
            if len(node.data) > 0:
                hnswIndex.add_items(node.data, ids=node.absolute_label_indices)
            hnswIndex.save_index(f"{dirpath}/worker-{i}_hnsw.bin")

        # if graph_type in ("bf", "both"):
        #     bfIndex: hnswlib.Index = hnswlib.BFIndex(space="l2", dim=node.data.shape[1])
        #     bfIndex.init_index(max_elements=len(node.data))
        #     # No need to add items for BFIndex, because we just use ivf index
        #     if len(node.data) > 0:
        #         bfIndex.add_items(node.data, ids=node.absolute_label_indices)
        #     bfIndex.save_index(f"{dirpath}/worker-{i}_bf.bin")

        save_node_data(node, f"{dirpath}/worker-{i}.bin")
        print(
            f"Node {i} processed: {len(node.data)} vectors, {node.num_buckets} buckets (1 vector per bucket)"
        )

    # Step 5: Create and save meta data containing node assignments and labels

    # Create bucket centroids: for HAKES, each vector is its own "bucket"
    # So bucket_centroids[i] = data[i] (each vector is a bucket centroid)
    bucket_centroids = data.copy()

    # Create bucket_node_labels: for HAKES, we need to map vector IDs to nodes (optimized)
    # Since HAKES returns vector IDs, we need to create a mapping from vector ID to node
    # We'll create a large array where bucket_node_labels[vector_id] = node_id
    max_vector_id = len(data) - 1
    bucket_node_labels = np.zeros((max_vector_id + 1, 1), dtype=np.uint32)

    # Map each vector ID to its assigned node based on hash using vectorized operations
    vector_ids = np.arange(len(data))
    node_ids = vector_ids % num_nodes
    bucket_node_labels[vector_ids, 0] = node_ids

    # Create nearest_point_labels: for HAKES, each vector is its own bucket
    # So nearest_point_labels[i] = i (each vector is the nearest point to itself)
    nearest_point_labels = np.arange(len(data), dtype=np.int32)

    # Create bucket_radiuses: set to 0 for HAKES (each vector is its own bucket)
    bucket_radiuses = np.zeros(len(data), dtype=np.float32)

    bucket_sizes = np.ones(len(data), dtype=np.uint32)

    # Save meta index data (compatible with existing format)
    save_meta_index_data(
        bucket_centroids,
        bucket_node_labels,
        nearest_point_labels,
        bucket_radiuses,
        f"{dirpath}/meta_index_data.bin",
        bucket_sizes=bucket_sizes,
        total_num_vectors=len(data),
    )

    print(f"\nHash partitioning completed")
    print(f"Created {num_nodes} nodes with hash-based data distribution")
    print(f"HAKES-compatible mapping: {len(data)} vectors mapped to {num_nodes} nodes")
    print(
        f"bucketNodeLabels array size: {len(bucket_node_labels)} (matches vector count)"
    )
    print(f"Node assignments saved to {dirpath}/node_assignments.npy")
    print(f"Meta index data saved to {dirpath}/meta_index_data.bin")

    # Generate path script (skip for test datasets)
    generate_path_script(dirpath, dataset_name)


def sample_unique_indices(total: int, want: int) -> np.ndarray:
    """Memory-lean unique index sampler.

    np.random.choice(total, size=want, replace=False) internally permutes the
    full N, which costs multi-GB at N=1B. Instead draw with replacement,
    deduplicate and top up - O(want) memory. Returns a sorted unique array.
    """
    got = np.empty(0, dtype=np.int64)
    while len(got) < want:
        draw = np.random.choice(total, size=(want - len(got)) * 2, replace=True)
        got = np.unique(np.concatenate([got, draw]))
    pick = np.random.choice(len(got), size=want, replace=False)
    return np.sort(got[pick])


def pyramid_partition_buckets(
    dataset_name: str,
    method_name: str,
    data: np.ndarray,
    imblance_factor: float,
    sample_num: int,
    num_nodes: int,
    graph_type: Literal["hnsw", "bf", "both"] = "hnsw",
    M: int = M,
    ef_construction: int = EF,
    num_threads: int = THREAD_NUM,
):
    """
    pyramid partitioning method:
    1. Randomly sample points from dataset and build meta HNSW graph
    2. Apply kahip graph partitioning algorithm to divide sampled graph into partitions
    3. Assign all data points to partitions based on nearest neighbor in sampled graph
    4. Save meta index information and routing table for search
    """
    print(f"Starting pyramid partitioning, sample_num={sample_num}, num_nodes={num_nodes}")

    # Step 1: Sample points and build meta HNSW graph
    np.random.seed(42)
    total_points = len(data)
    if sample_num > total_points:
        sample_num = total_points
        print(f"Warning: sample_num adjusted to {total_points}")

    sampled_indices = sample_unique_indices(total_points, sample_num)
    sampled_points = data[sampled_indices]
    print(f"Sampled {sample_num} points for meta graph")

    # Build meta HNSW index
    meta_hnsw = hnswlib.Index(space="l2", dim=data.shape[1])
    meta_hnsw.init_index(max_elements=sample_num, ef_construction=ef_construction, M=M)
    sample_ids = np.arange(sample_num, dtype=np.uint32)
    meta_hnsw.add_items(sampled_points, ids=sample_ids, num_threads=num_threads)
    print("Meta HNSW graph built Successfully")

    # Step 2: Graph partitioning using kahip algorithm
    print("Extracting graph structure from HNSW...")

    k_neighbors = min(10, sample_num - 1)  # Number of neighbors to consider

    neighbors, distances = meta_hnsw.knn_query(sampled_points, k=k_neighbors, num_threads=num_threads)
    
    # Feed the kNN graph straight into kahip. The intermediate networkx.Graph
    # plus the per-node adjacency list-of-lists (never read afterwards) cost
    # several GB at sample_num=1M, so build the undirected edge set with
    # vectorized numpy instead: normalize every directed kNN edge to (min,max),
    # drop self-loops, deduplicate, and order by first occurrence - which
    # exactly reproduces the old networkx insertion order (an edge enters
    # G.edges() at its first sighting from either endpoint).
    src = np.repeat(np.arange(sample_num, dtype=np.int64), k_neighbors)
    dst = neighbors.reshape(-1).astype(np.int64)
    keep = src != dst  # Avoid self-loops
    src, dst = src[keep], dst[keep]
    key = np.minimum(src, dst) * sample_num + np.maximum(src, dst)
    uniq_keys, first_pos = np.unique(key, return_index=True)
    uniq_keys = uniq_keys[np.argsort(first_pos, kind="stable")]
    del src, dst, keep, key, first_pos
    u_edges = (uniq_keys // sample_num).tolist()
    v_edges = (uniq_keys % sample_num).tolist()
    del uniq_keys
    print(f"Graph constructed: {sample_num} nodes, {len(u_edges)} edges")

    # Run KaHIP/KaFFPa partitioning
    kahip_graph = kahip.kahip_graph()
    kahip_graph.set_num_nodes(sample_num)
    for u, v in zip(u_edges, v_edges):
        kahip_graph.add_undirected_edge(u, v, 1)
    # weight only nodes that have at least one edge, mirroring the old loop
    touched = np.zeros(sample_num, dtype=bool)
    touched[np.asarray(u_edges, dtype=np.int64)] = True
    touched[np.asarray(v_edges, dtype=np.int64)] = True
    for i in np.nonzero(touched)[0].tolist():
        kahip_graph.set_weight(i, 1)
    del touched
    vwgt, xadj, adjcwgt, adjncy = kahip_graph.get_csr_arrays()
    # edgecut: the number of edges that are cut by the partition
    # blocks: the partition labels

    edgecut,blocks = kahip.kaffpa(vwgt, xadj, adjcwgt, adjncy, num_nodes, imblance_factor, 0, 0, 0)

    # Convert to numpy array to ensure compatibility
    sample_partition_labels = np.array(blocks, dtype=np.int32)
    print(f"Graph partitioned into {num_nodes} partitions")
    
    # Step 3: Assign all data points based on nearest neighbor
    print("Assigning all data points to partitions...")
    point_partition_labels = np.zeros(total_points, dtype=np.int32)
    
    # For sampled points, use their partition directly
    point_partition_labels[sampled_indices] = sample_partition_labels
    
    # For non-sampled points, find nearest sampled neighbor and use its partition.
    # Streamed: walk the base in contiguous row batches (sequential memmap
    # reads) instead of gathering the whole non-sampled dataset into RAM.
    # Sampled points are their own nearest neighbour, but keep their exact
    # partition label explicitly (cheap np.where on the batch).
    ASSIGN_BATCH = 100_000  # rows; 100k x 128 x 4B = ~51MB per batch
    is_sampled = np.zeros(total_points, dtype=bool)  # 1 byte/point, freed below
    is_sampled[sampled_indices] = True

    for start in tqdm(range(0, total_points, ASSIGN_BATCH), desc="Assigning partitions"):
        end = min(start + ASSIGN_BATCH, total_points)
        batch = np.ascontiguousarray(data[start:end])  # sequential gather, batch-sized
        nearest_samples, _ = meta_hnsw.knn_query(batch, k=1, num_threads=num_threads)
        nearest_samples = nearest_samples.flatten()
        labels = sample_partition_labels[nearest_samples]
        batch_is_sampled = is_sampled[start:end]
        if batch_is_sampled.any():
            labels = np.where(batch_is_sampled, point_partition_labels[start:end], labels)
        point_partition_labels[start:end] = labels
        del batch, nearest_samples, labels

    del is_sampled
    print("All points assigned to partitions")

    # Create output directory
    dirpath = f"../data/{dataset_name}_worker_data_hnsw_{method_name}"
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)

    # Step 4: Build local indices for each node, streamed. Each node's rows
    # are gathered in chunks straight from the base memmap; one gather feeds
    # both hnswlib (incremental add_items) and the streamed worker-bin writer,
    # so peak RSS is ~hnsw index + one chunk instead of ~4x shard size.
    print("Building local indices...")
    NODE_CHUNK_ROWS = 500_000  # 500k x 128 x 4B = 256MB per gather chunk
    for node_id in range(num_nodes):
        node_rows = np.where(point_partition_labels == node_id)[0]
        if len(node_rows) == 0:
            print(f"Warning: Node {node_id} has no data points")
            continue
        print(f"Node {node_id}: {len(node_rows)} points")

        hnswIndex = None
        if graph_type in ("hnsw", "both"):
            hnswIndex = hnswlib.Index(space="l2", dim=data.shape[1])
            hnswIndex.init_index(
                max_elements=len(node_rows), ef_construction=ef_construction, M=M
            )
        bfIndex = None
        if graph_type in ("bf", "both"):
            bfIndex = hnswlib.BFIndex(space="l2", dim=data.shape[1])
            bfIndex.init_index(max_elements=len(node_rows))

        def _feed_hnsw(chunk, ids, _h=hnswIndex, _b=bfIndex):
            if _h is not None:
                _h.add_items(chunk, ids=ids, num_threads=num_threads)
            if _b is not None:
                _b.add_items(chunk, ids=ids)

        save_node_data_streamed(
            f"{dirpath}/worker-{node_id}.bin",
            data.shape[1],
            np.array([node_id]),
            [node_rows],
            data,
            on_bucket_data=_feed_hnsw,
            chunk_rows=NODE_CHUNK_ROWS,
        )

        if hnswIndex is not None:
            hnswIndex.save_index(f"{dirpath}/worker-{node_id}_hnsw.bin")
            del hnswIndex
        if bfIndex is not None:
            bfIndex.save_index(f"{dirpath}/worker-{node_id}_bf.bin")
            del bfIndex
        del node_rows
        gc.collect()

    # Step 5: Save meta index
    meta_hnsw.save_index(f"{dirpath}/meta_hnsw.bin")

    # Prepare metadata for compatibility
    bucket_centroids = sampled_points
    bucket_node_labels = sample_partition_labels.reshape(-1, 1).astype(np.uint32)
    nearest_point_labels = sampled_indices.astype(np.int32)
    bucket_radiuses = np.zeros(sample_num, dtype=np.float32)

    bucket_sizes = np.ones(sample_num, dtype=np.uint32)

    save_meta_index_data(
        bucket_centroids,
        bucket_node_labels,
        nearest_point_labels,
        bucket_radiuses,
        f"{dirpath}/meta_index_data.bin",
        bucket_sizes=bucket_sizes,
        total_num_vectors=len(data),
    )

    print(f"\nPyramid partitioning completed:")
    print(f"  - Sampled points: {sample_num}")
    print(f"  - Partitions: {num_nodes}")
    # print partition number info
    for i in range(num_nodes):
        count = np.sum(point_partition_labels == i)
        print(f"  - Node {i}: {count} points")
    print(f"bucketNodeLabels array length: {len(bucket_node_labels)}")

    # Generate path script
    generate_path_script(dirpath, dataset_name)


def naive_partition_buckets(
    schema_name: str,
    dataset_name: str,
    method_name: str,
    data: np.ndarray,
    num_nodes: int,
    graph_type: Literal["hnsw", "bf", "both"] = "hnsw",
    M: int = M,
    ef_construction: int = EF,
    num_threads: int = THREAD_NUM,
):
    # 读取数据，并平均分配到每个节点
    def split_data(data, num_nodes):
        return np.array_split(data, num_nodes)

    labels = np.arange(len(data))
    dataList = split_data(data, num_nodes)
    print("Data split completed")
    labelsList = split_data(labels, num_nodes)
    dirpath = f"../data/{schema_name}_worker_data_hnsw_{method_name}"
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)

    for i in range(num_nodes):
        node = Node.from_buckets(
            [Bucket.from_data(dataList[i], labelsList[i])],
            np.array([i]),
            np.array([i]),
        )
        if graph_type in ("hnsw", "both"):
            hnswIndex: hnswlib.Index = hnswlib.Index(space="l2", dim=node.data.shape[1])
            hnswIndex.init_index(
                max_elements=len(node.data), ef_construction=ef_construction, M=M
            )
            hnswIndex.add_items(
                node.data, ids=node.absolute_label_indices, num_threads=num_threads
            )
            hnswIndex.save_index(f"{dirpath}/worker-{i}_hnsw.bin")
        if graph_type in ("bf", "both"):
            bfIndex: hnswlib.Index = hnswlib.BFIndex(space="l2", dim=node.data.shape[1])
            bfIndex.init_index(max_elements=len(node.data))
            bfIndex.add_items(node.data, ids=node.absolute_label_indices)
            bfIndex.save_index(f"{dirpath}/worker-{i}_bf.bin")

        save_node_data(node, f"{dirpath}/worker-{i}.bin")
    generate_path_script(dirpath, dataset_name)


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    os.chdir(script_dir)

    param_config_path = os.path.join(project_root, "config", "parameters.yaml")
    partition_config_path = os.path.join(project_root, "config", "partitionConfig.yaml")

    with open(param_config_path, "r") as f:
        param_config = yaml.safe_load(f)
    with open(partition_config_path, "r") as f:
        partition_config = yaml.safe_load(f)

    param_config_keys = list(param_config.keys())
    partition_config_keys = list(partition_config.keys())

    # DEPRECATED: adaptive_kmeans
    bucket_cluster_method_keys = [
        "kmeans",
        "adaptive_kmeans",
        "graph_partition",
    ]
    fzf = FzfPrompt()

    # Check for command line arguments
    if len(sys.argv) == 4:
        chosen_dataset = sys.argv[1]
        chosen_bucket_cluster_method = sys.argv[2]
        chosen_method = sys.argv[3]
        print(f"Chosen dataset: {chosen_dataset}")
        print(f"Chosen bucket cluster method: {chosen_bucket_cluster_method}")
        print(f"Chosen method: {chosen_method}")
    elif len(sys.argv) == 2:
        chosen_dataset = sys.argv[1]
        chosen_method = "similarity_16_hnsw"
        chosen_bucket_cluster_method = "kmeans"
        if chosen_dataset not in param_config:
            print(f"Unknown dataset key: {chosen_dataset}")
            print(f"Available keys: {', '.join(param_config_keys)}")
            exit(1)
        if chosen_method not in partition_config:
            print(f"Default partition method not found: {chosen_method}")
            exit(1)
        print(f"Chosen dataset: {chosen_dataset}")
        print(f"Chosen method (default): {chosen_method}")
        print(f"Chosen bucket cluster method (default): {chosen_bucket_cluster_method}")
    else:
        print("Choose dataset:")
        chosen_dataset = fzf.prompt(param_config_keys)
        if len(chosen_dataset) == 0:
            print("No dataset chosen. Exit now.")
            exit(0)
        chosen_dataset = chosen_dataset[0]
        print(f"Chosen dataset: {chosen_dataset}")
        
        # Choose partition method
        print("Choose partition method:")
        chosen_method = fzf.prompt(partition_config_keys)
        if len(chosen_method) == 0:
            print("No partition method chosen. Exit now.")
            exit(0)
        chosen_method = chosen_method[0]
        print(f"Chosen method: {chosen_method}")

        # 如果是similarity开头，则需要选择 bucket cluster method
        if chosen_method.startswith("similarity"):
            print("Choose bucket cluster method")
            chosen_bucket_cluster_method = fzf.prompt(bucket_cluster_method_keys)
            if len(chosen_bucket_cluster_method) == 0:
                print("No bucket cluster method chosen. Exit now.")
                exit(1)
            chosen_bucket_cluster_method = chosen_bucket_cluster_method[0]
            print(f"Chosen bucket cluster method: {chosen_bucket_cluster_method}")
        else:
            chosen_bucket_cluster_method = None

    if (
        chosen_dataset is None
        or chosen_method is None
    ):
        print("No dataset or method chosen.")
        exit(1)

    dataset_config = param_config[chosen_dataset]
    dataset_name = dataset_config["dataset"]
    bucket_num = dataset_config["bucket_num"]
    partition_params = partition_config[chosen_method]
    graph_type = partition_params["graph_type"]

    if not isinstance(bucket_num, list):
        bucket_num = [bucket_num]
    base, query, ground_truth = get_dataset(dataset_name)
    if partition_params["method"] == "similarity":
        bucket_labels, centroids = hierarchical_KMeans(
            base, bucket_num, verbose=True, dataset_name=dataset_name
        )
        pre_partition_buckets(
            chosen_dataset,
            dataset_name,
            chosen_method,
            base,
            bucket_labels,
            centroids,
            partition_params["partition_num"],
            chosen_bucket_cluster_method,
            graph_type=graph_type,
            M=partition_params["M"],
            ef_construction=partition_params["ef_construction"],
            num_threads=partition_params.get("thread_num", THREAD_NUM),
            num_overlap=partition_params.get("overlap_num", NUM_OVERLAP),
        )

    elif partition_params["method"] == "naive":
        naive_partition_buckets(
            chosen_dataset,
            dataset_name,
            chosen_method,
            base,
            partition_params["partition_num"],
            graph_type=graph_type,
            M=partition_params["M"],
            ef_construction=partition_params["ef_construction"],
            num_threads=partition_params["thread_num"],
        )

    elif partition_params["method"] == "pyramid":
        pyramid_partition_buckets(
            dataset_name,
            chosen_method,
            base,
            imblance_factor=partition_params["imblance_factor"],
            sample_num=partition_params["sample_num"],
            num_nodes=partition_params["partition_num"],
            graph_type=graph_type,
            M=partition_params["M"],
            ef_construction=partition_params["ef_construction"],
            num_threads=partition_params["thread_num"],
        )

    elif partition_params["method"] == "hash" or partition_params["method"] == "hakes":
        hash_partition_buckets(
            dataset_name,
            chosen_method,
            base,
            num_nodes=partition_params["partition_num"],
            graph_type=graph_type,
            M=partition_params["M"],
            ef_construction=partition_params["ef_construction"],
            num_threads=partition_params["thread_num"],
        )
    else:
        print(f"Invalid partition method: {partition_params['method']}")
        exit(1)
